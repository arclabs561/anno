# SPDX-License-Identifier: MIT OR Apache-2.0
"""Compare native track partitions with withheld Bioguide IDs on real dated directories."""
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
import json
from math import fsum
from pathlib import Path
import subprocess
from statistics import median
from time import perf_counter


CUTS = ("2021-07-01", "2023-07-01", "2025-07-01")
COMMITTEES = {"HSHM": "rep", "SSFI": "sen"}


@dataclass(frozen=True)
class Observation:
    name: str
    person: str
    chamber: str


def load(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def documents(roster, committees):
    docs = {}
    omitted = 0
    for cut in CUTS:
        members, assignments = {}, {}
        for release in roster:
            if release["published_at"][:10] >= cut:
                break
            for person in release["remove"]:
                del members[person]
            for row in release["upsert"]:
                members[row["id"]] = row
        for chamber in ("rep", "sen"):
            in_chamber = [r for r in members.values() if r["chamber"] == chamber]
            omitted += sum(not r["official_full"] for r in in_chamber)
            docs[f"{cut}/roster/{chamber}"] = [
                Observation(r["official_full"], r["id"], chamber)
                for r in in_chamber if r["official_full"]
            ]
        for release in committees:
            if release["published_at"][:10] >= cut:
                break
            for key in release["remove"]:
                del assignments[tuple(key)]
            for committee, name, ordinal, person, _ in release["upsert"]:
                assignments[committee, name, ordinal] = person
        for committee, chamber in COMMITTEES.items():
            docs[f"{cut}/committee/{committee}"] = [
                Observation(name, person, chamber)
                for (source, name, _ordinal), person in assignments.items()
                if source == committee and person
            ]
    return {k: sorted(rows, key=lambda r: r.name) for k, rows in docs.items()}, omitted


def run_native(binary, docs, threshold=None, split=False, cut_order=None):
    # The subprocess receives name strings and document IDs. Gold person IDs
    # are used only to reconstruct source editions above, and for scoring below.
    assignments = {}
    for pool in (("rep", "sen") if split else ("all",)):
        projected = {
            doc: [r.name for r in rows]
            for doc, rows in docs.items()
            if rows and (pool == "all" or rows[0].chamber == pool)
        }
        args = [str(binary)] if threshold is None else [str(binary), str(threshold)]
        batches = [projected] if cut_order is None else [
            {doc: names for doc, names in projected.items() if doc.split("/")[0] == cut}
            for cut in cut_order
        ]
        completed = subprocess.run(args, input=json.dumps(batches), capture_output=True,
                                   check=True, text=True)
        groups = json.loads(completed.stdout)
        if set(groups) != set(projected):
            raise ValueError("resolver output has different document keys")
        for doc, ids in groups.items():
            if len(ids) != len(projected[doc]):
                raise ValueError(f"{doc}: resolver output has different occurrence count")
            for i, identity in enumerate(ids):
                assignments[doc, i] = pool, identity
    return assignments


def cross_pairs(group):
    return (pair for pair in combinations(group, 2) if pair[0][0] != pair[1][0])


def partition(predictions):
    grouped = defaultdict(list)
    for anchor, group in predictions.items():
        grouped[group].append(anchor)
    return {frozenset(anchors) for anchors in grouped.values()}


def evaluate(docs, predictions):
    source = {(doc, i): r for doc, rows in docs.items() for i, r in enumerate(rows)}
    gold_groups, output_groups = defaultdict(list), defaultdict(list)
    for anchor, row in source.items():
        gold_groups[row.person].append(anchor)
        output_groups[predictions[anchor]].append(anchor)

    errors = {"false_merges": Counter(), "false_splits": Counter()}
    good = 0
    for group in output_groups.values():
        for i, j in cross_pairs(group):
            a, b = source[i], source[j]
            if a.person == b.person:
                good += 1
            else:
                key = tuple(sorted(((a.person, a.name), (b.person, b.name))))
                errors["false_merges"][key] += 1
    for group in gold_groups.values():
        for i, j in cross_pairs(group):
            if predictions[i] != predictions[j]:
                a, b = source[i], source[j]
                names = tuple(sorted((a.name, b.name)))
                errors["false_splits"][(a.person, *names)] += 1

    total = sum(a * b for a, b in combinations([len(rows) for rows in docs.values()], 2))
    fp, fn = (sum(tally.values()) for tally in errors.values())
    counts = {"documents": len(docs), "tracks": len(source), "gold_people": len(gold_groups),
              "output_groups": len(output_groups), "cross_document_pairs": total, "tp": good,
              "fp": fp, "fn": fn, "tn": total - good - fp - fn}
    return {**counts, **{name: tally.most_common(5) for name, tally in errors.items()}}


def reader_metrics(docs, predictions, latest_cut):
    """Score all-track B³ and the earlier occurrences returned per latest-cut query."""
    gold_sizes, group_sizes, overlaps = Counter(), Counter(), Counter()
    prior_gold, prior_groups, prior_overlaps = Counter(), Counter(), Counter()
    prior_dates = defaultdict(set)
    queries = []
    for doc, rows in docs.items():
        date = doc.split("/", 1)[0]
        for i, row in enumerate(rows):
            group = predictions[doc, i]
            person = row.person
            gold_sizes[person] += 1
            group_sizes[group] += 1
            overlaps[person, group] += 1
            if date < latest_cut:
                prior_gold[person] += 1
                prior_groups[group] += 1
                prior_overlaps[person, group] += 1
                prior_dates[group].add(date)
            elif date == latest_cut:
                queries.append((person, group))

    # Each of the n occurrences in a cell has n / group size precision and
    # n / gold size recall. Self and same-document occurrences are included.
    total = sum(gold_sizes.values())
    precision = recall = f1 = 0.0
    if total:
        precision = fsum(n * n / group_sizes[g] for (_, g), n in overlaps.items()) / total
        recall = fsum(n * n / gold_sizes[p] for (p, _), n in overlaps.items()) / total
        f1 = 2 * precision * recall / (precision + recall)

    lookup = {"queries": len(queries),
              "queries_with_history": sum(prior_gold[p] > 0 for p, _ in queries),
              "minimum_dates": {}}
    for minimum in (1, 2):
        counts = dict.fromkeys(("shown_pages", "pages_with_wrong_hits", "correct_hits",
                                "wrong_hits", "missed_hits"), 0)
        for person, group in queries:
            correct = 0
            if len(prior_dates[group]) >= minimum:
                correct = prior_overlaps[person, group]
                wrong = prior_groups[group] - correct
                counts["shown_pages"] += 1
                counts["pages_with_wrong_hits"] += wrong > 0
                counts["correct_hits"] += correct
                counts["wrong_hits"] += wrong
            counts["missed_hits"] += prior_gold[person] - correct
        lookup["minimum_dates"][str(minimum)] = counts
    return {"b3": {"precision": precision, "recall": recall, "f1": f1}, "prior_lookup": lookup}


def resolve_chamber_conflict(docs, predictions):
    """Retrospective reader rule: only split groups with conflicting contemporaneous roster roles."""
    observed = defaultdict(set)
    for (doc, i), group in predictions.items():
        cut, source, _section = doc.split("/")
        if source == "roster":
            observed[group, cut].add(docs[doc][i].chamber)
    conflicts = {group for (group, _cut), chambers in observed.items() if len(chambers) > 1}
    return {
        (doc, i): (group, docs[doc][i].chamber if group in conflicts else None)
        for (doc, i), group in predictions.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roster", type=Path)
    parser.add_argument("committees", type=Path)
    parser.add_argument("native_binary", type=Path)
    parser.add_argument("--timing-runs", type=int, default=1,
                        help="repeat default publication in fresh processes and report median time")
    args = parser.parse_args()
    if args.timing_runs <= 0:
        parser.error("--timing-runs must be positive")
    docs, omitted = documents(load(args.roster), load(args.committees))
    print("Roster occurrences omitted for absent official_full:", omitted)
    durations = []
    for _ in range(args.timing_runs):
        start = perf_counter()
        default = run_native(args.native_binary, docs)
        durations.append(perf_counter() - start)
    print("native", json.dumps(evaluate(docs, default), ensure_ascii=False))
    print("native-reader", json.dumps(reader_metrics(docs, default, CUTS[-1])))
    gated = resolve_chamber_conflict(docs, default)
    print("roster-conflict", json.dumps(evaluate(docs, gated), ensure_ascii=False))
    for label, threshold, split in [("threshold-one", 1, False), ("relaxed", 0.5, False),
                                    ("chamber", None, True)]:
        result = run_native(args.native_binary, docs, threshold, split)
        print(label, json.dumps(evaluate(docs, result), ensure_ascii=False))
        if label == "relaxed":
            print("relaxed-reader", json.dumps(reader_metrics(docs, result, CUTS[-1])))
    for order in (CUTS, CUTS[::-1]):
        staged = run_native(args.native_binary, docs, cut_order=order)
        equivalent = partition(default) == partition(staged)
        changed = sum(staged[occ] != group for occ, group in default.items())
        print(f"Three batches starting {order[0]}: memberships equal to bulk={equivalent}, "
              f"final local ID differs on {changed}/{len(staged)} occurrences")
    print(f"Default native publication: {args.timing_runs} runs, median {median(durations):.3f} s")


if __name__ == "__main__":
    main()
