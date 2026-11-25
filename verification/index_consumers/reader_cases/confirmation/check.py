# SPDX-License-Identifier: MIT OR Apache-2.0
"""Replay frozen GeoNames alias queries against the shared spelling candidate policy."""
import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import json
from pathlib import Path
import sys
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "candidates"))
from policy import CandidateIndex


def digest(text):
    return sha256(text.encode("utf-8")).hexdigest()


def project(archive):
    with ZipFile(archive) as zipped:
        data = zipped.read("CH.txt")
    rows = {}
    eligible = []
    counts = Counter()
    for line in data.decode("utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) != 19:
            raise ValueError("unexpected number of GeoNames fields")
        counts["source_rows"] += 1
        geographic_id, primary = fields[0], fields[1].strip()
        if fields[6] != "P" or fields[8] != "CH":
            continue
        counts["ch_populated_places"] += 1
        if not primary:
            counts["blank_primary_excluded"] += 1
            continue
        aliases = {x.strip() for x in fields[3].split(",")} - {"", primary}
        if not aliases:
            counts["no_distinct_alternate_excluded"] += 1
            continue
        if geographic_id in rows:
            raise ValueError("duplicate eligible GeoNames ID")
        counts["eligible"] += 1
        rows[geographic_id] = primary, aliases
        eligible.append(geographic_id)
    chosen = sorted(eligible, key=lambda gid: (digest("anno-place-2026-01|" + gid), int(gid)))[:250]
    projection = []
    for i, gid in enumerate(chosen):
        primary, aliases = rows[gid]
        alias = min(aliases, key=lambda a: (digest(f"anno-alias-2026-01|{gid}|{a}"), a))
        projection.append({"geonames_id": gid, "catalog_name": primary if i < 200 else "", "query": alias})
    return projection, dict(counts), sha256(data).hexdigest()


def gallery(rows):
    primary = {}
    answer_key = {}
    for row in rows:
        if row["catalog_name"]:
            key = f"donor-{len(primary)}"
            primary[key] = row["catalog_name"]
            answer_key[key] = row["geonames_id"]
    return primary, answer_key


def evaluate(rows):
    primary, source = gallery(rows)
    if len(set(source.values())) != len(source) or len({r["geonames_id"] for r in rows}) != len(rows):
        raise ValueError("repeated entity")
    included = set(source.values())
    index = CandidateIndex({key: (name, key) for key, name in primary.items()})
    frozen_spelling = defaultdict(set)
    literal = defaultdict(set)
    for key, name in primary.items():
        frozen_spelling[name.casefold()].add(key)
        literal[name].add(key)

    out = {}
    for variant in ("literal", "native", "fold", "expanded"):
        # Fixed denominators, including zero outcomes, make each arm comparable.
        names = ("queries", "covered", "correct_in_set", "only_correct", "wrong_only",
                 "mixed", "abstained", "wrong_candidates", "returned_candidates",
                 "indexed_queries", "indexed_covered", "absent_queries", "absent_covered",
                 "ambiguous_suppressed", "added_correct", "added_wrong")
        summary = dict.fromkeys(names, 0)
        wrong_surfaces = set()
        for row in rows:
            gold, query = row["geonames_id"], row["query"]
            current = (literal if variant == "literal" else frozen_spelling).get(
                query if variant == "literal" else query.casefold(), set())
            native_group = next(iter(current)) if len(current) == 1 else None
            suggestion = index.suggest(query, native_group, "native" if variant == "literal" else variant)
            answers = current | suggestion.hits
            correct = any(source[k] == gold for k in answers)
            wrong = sum(source[k] != gold for k in answers)
            summary["queries"] += 1
            summary["covered"] += bool(answers)
            summary["abstained"] += not answers
            summary["correct_in_set"] += correct
            summary["only_correct"] += correct and not wrong
            summary["wrong_only"] += bool(answers) and not correct
            summary["mixed"] += correct and bool(wrong)
            if wrong:
                wrong_surfaces.add(query)
            summary["wrong_candidates"] += wrong
            summary["returned_candidates"] += len(answers)
            stratum = "indexed" if gold in included else "absent"
            summary[stratum + "_queries"] += 1
            summary[stratum + "_covered"] += bool(answers)
            summary["ambiguous_suppressed"] += suggestion.decision == "ambiguous"
            extra = suggestion.hits - current
            summary["added_correct"] += sum(source[k] == gold for k in extra)
            summary["added_wrong"] += sum(source[k] != gold for k in extra)
        summary["distinct_query_surfaces"] = len({r["query"] for r in rows})
        summary["distinct_wrong_surfaces"] = len(wrong_surfaces)
        out[variant] = summary
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="also reselect the frozen projection from the source CH.zip")
    args = parser.parse_args()
    metadata = json.loads((HERE / "receipt.json").read_bytes())
    for name, sha in metadata["files"].items():
        if sha256((HERE / name).read_bytes()).hexdigest() != sha:
            raise ValueError(f"changed frozen file: {name}")
    with (HERE / "places.tsv").open(encoding="utf-8", newline="") as tab:
        rows = list(csv.DictReader(tab, delimiter="\t"))
    if args.source:
        rebuilt, counts, data_hash = project(args.source)
        if rebuilt != rows or counts != metadata["selection"] or data_hash != metadata["source_txt_sha256"]:
            raise ValueError("archive changed or selected records differ")
    result = evaluate(rows)
    if result != metadata["observed"]:
        raise ValueError("candidate output changed")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
