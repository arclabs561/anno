# SPDX-License-Identifier: MIT OR Apache-2.0
"""Score prior-member lookup on identity-disjoint House roll calls."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from cross_doc_partition import Observation, evaluate, partition, run_native


HERE = Path(__file__).parent
PRIOR = "2003-01-07/house/roll001"
QUERY = "2005-01-04/house/roll001"


def checked_bytes(directory, name, hashes):
    raw = (directory / name).read_bytes()
    if sha256(raw).hexdigest() != hashes[name]:
        raise ValueError(f"unexpected input checksum: {name}")
    return raw


def excluded_people(directory, hashes):
    # Include everyone ever observed, even where a later release removed an ID.
    roster = checked_bytes(directory, "roster.jsonl", hashes)
    committees = checked_bytes(directory, "committee.jsonl", hashes)
    ids = {row["id"] for line in roster.splitlines() for row in json.loads(line)["upsert"]}
    ids.update(row[3] for line in committees.splitlines() for row in json.loads(line)["upsert"] if row[3])
    return ids


def read_roll(raw, excluded):
    rows = []
    seen = set()
    votes = Counter()
    omissions = Counter()
    for entry in ET.fromstring(raw).findall("vote-data/recorded-vote"):
        member = entry.find("legislator")
        if member is None:
            raise ValueError("recorded vote has no legislator element")
        person, printed = member.get("name-id"), member.text
        votes[entry.findtext("vote")] += 1
        if person:
            if person in seen:
                raise ValueError("member ID appears twice on one roll")
            seen.add(person)
        if not person or not printed or not printed.strip():
            omissions["missing_id_or_name"] += 1
        elif person in excluded:
            omissions["earlier_pilot"] += 1
        else:
            rows.append(Observation(printed, person, "rep"))
    if not rows or len(rows) + sum(omissions.values()) != sum(votes.values()):
        raise ValueError("empty or incomplete roll")
    return rows, {"votes": dict(votes), "omitted": dict(omissions), "eligible": len(rows)}


def score_lookup(docs, groups):
    # Each returned prior occurrence is available to the reader. Its person ID
    # is only used after deciding whether the answer is unique.
    prior_groups = defaultdict(list)
    for i, member in enumerate(docs[PRIOR]):
        prior_groups[groups[PRIOR, i]].append(member)
    prior_ids = {member.person for member in docs[PRIOR]}
    outcomes = ("queries", "correct", "wrong", "no_prior_in_component", "multiple_priors_in_component")
    counts = {key: dict.fromkeys(outcomes, 0) for key in ("prior_target_present", "prior_target_absent")}
    wrong_same_spelling = 0
    for i, query in enumerate(docs[QUERY]):
        stratum = counts["prior_target_present" if query.person in prior_ids else "prior_target_absent"]
        candidates = prior_groups[groups[QUERY, i]]
        if not candidates:
            outcome = "no_prior_in_component"
        elif len(candidates) > 1:
            outcome = "multiple_priors_in_component"
        elif candidates[0].person == query.person:
            outcome = "correct"
        else:
            outcome = "wrong"
            wrong_same_spelling += candidates[0].name == query.name
        stratum[outcome] += 1
        stratum["queries"] += 1
    for stratum in counts.values():
        if sum(stratum[o] for o in outcomes[1:]) != stratum["queries"]:
            raise ValueError("query outcomes do not account for every lookup")
    pairs = evaluate(docs, groups)
    pair_counts = {key: pairs[key] for key in ("cross_document_pairs", "tp", "fp", "fn", "tn")}
    if sum(pair_counts[key] for key in ("tp", "fp", "fn", "tn")) != len(docs[PRIOR]) * len(docs[QUERY]):
        raise ValueError("pair counts do not account for the two rolls")
    return {"queries": counts, "pairs": pair_counts, "wrong_unique_with_identical_printing": wrong_same_spelling}


def measure(directory, binary, hashes):
    excluded = excluded_people(directory, hashes)
    docs, selection = {}, {}
    for doc, year in ((PRIOR, 2003), (QUERY, 2005)):
        raw = checked_bytes(directory, f"roll001-{year}.xml", hashes)
        docs[doc], selection[doc] = read_roll(raw, excluded)
    selected = {m.person for members in docs.values() for m in members}
    if not selected.isdisjoint(excluded):
        raise ValueError("earlier pilot identity leaked into the held-out panel")
    literal = {(doc, i): member.name for doc, rows in docs.items() for i, member in enumerate(rows)}
    native = run_native(binary, docs)
    prior_names = {m.person: m.name for m in docs[PRIOR]}
    changed = [(prior_names[m.person], m.name) for m in docs[QUERY]
               if m.person in prior_names and m.name != prior_names[m.person]]
    state_suffix = re.compile(r" \([A-Z]{2}\)$")
    return {
        "selection": selection, "earlier_pilot_ids": len(excluded), "eligible_ids": len(selected),
        "identity_overlap_with_earlier_pilot": len(selected & excluded),
        "shared_ids_with_changed_printing": len(changed),
        "changed_forms_equal_after_removing_state_suffix": sum(
            state_suffix.sub("", a) == state_suffix.sub("", b) for a, b in changed),
        "same_partition_as_literal": partition(native) == partition(literal),
        "literal": score_lookup(docs, literal), "native_default": score_lookup(docs, native),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="local roll XML and pilot JSONL files")
    parser.add_argument("native_binary", type=Path, help="cross_doc_track_partition executable")
    parser.add_argument("--check", action="store_true", help="compare with the frozen aggregate measurement")
    args = parser.parse_args()
    record = json.loads((HERE / "transfer-rollcalls.json").read_text())
    results = measure(args.directory, args.native_binary.resolve(), record["input_sha256"])
    print(json.dumps(results, sort_keys=True, indent=2))
    if args.check and results != record["observed"]:
        raise SystemExit("results differ from the frozen measurement")
