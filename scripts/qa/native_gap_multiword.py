"""Measure one frozen two-token proposal rule on the three released GAP splits."""

import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path

import audit_gap_decisions as gap
import gap_receipts
from coref_gap import REVISION
from native_gap_endpoint_stages import checked_partition
from native_gap_loss import trace_documents
from native_gap_neutral_type import ROOT, evaluate, file_hash, read_sources

BASE = "01370e52ba97d5a094c3a3a0e9d3b038ef7063f9"
PATCH = ROOT / "scripts/qa/fixtures/native-gap-multiword.patch"
STAGES = ("proposed", "retained", "returned")


def transition():
    return dict(stock=0, candidate=0, gained=0, lost=0)


def count_sets(counts, stock, candidate):
    counts["stock"] += len(stock)
    counts["candidate"] += len(candidate)
    counts["gained"] += len(candidate - stock)
    counts["lost"] += len(stock - candidate)


def spans_by_stage(trace, groups):
    return {
        "proposed": set(map(tuple, trace["proposed"])),
        "retained": set(map(tuple, trace["retained"])),
        "returned": {span for group in groups for span in group},
    }


def shared_links(groups, common):
    return {
        pair for group in groups for pair in combinations(sorted(common.intersection(group)), 2)
    }


def unlinked_stage(spans, pronoun, name):
    if name not in spans["proposed"] or pronoun not in spans["proposed"]:
        if pronoun in spans["proposed"]:
            return "name_not_proposed"
        return "pronoun_not_proposed" if name in spans["proposed"] else "neither_proposed"
    for key, label in (("retained", "overlap_dedup"), ("returned", "isolated_endpoint")):
        if pronoun not in spans[key] or name not in spans[key]:
            return label
    return "separate_clusters"


def compare(rows, stock, candidate, frozen):
    correct, wrong = transition(), transition()
    coverage = {stage: transition() for stage in STAGES}
    unmarked = {stage: transition() for stage in STAGES}
    common_links = transition()
    partial = {"stock_pairs": 0, "exact_proposals_gained": 0, "correct_links_gained": 0}
    newly_wrong, formerly_correct = Counter(), Counter()
    wrong_cases = Counter()
    partitions = [[], []]
    changed = 0
    for row, old, new in zip(rows, stock, candidate, strict=True):
        if old["id"] != row["ID"] or new["id"] != row["ID"]:
            raise ValueError("worker rows differ from the source order")
        groups = [checked_partition(t) for t in (old, new)]
        for arm, partition in zip(partitions, groups):
            arm.append(partition)
        changed += groups[0] != groups[1]
        before, after = [spans_by_stage(t, g) for t, g in zip((old, new), groups)]
        if before["proposed"] - after["proposed"]:
            raise ValueError("candidate removed a stock proposal")
        targets = gap.spans(row)
        pronoun = targets[0]
        names = targets[1:]
        sides = {0, 1}
        positives = {i for i, positive in enumerate(gap.gold(row)) if positive}
        linked = [
            {i for i in sides if any(pronoun in g and names[i] in g for g in partition)}
            for partition in groups
        ]
        count_sets(correct, linked[0] & positives, linked[1] & positives)
        count_sets(wrong, linked[0] - positives, linked[1] - positives)
        for i in linked[1] - linked[0] - positives:
            newly_wrong[unlinked_stage(before, pronoun, names[i])] += 1
            if not positives:
                wrong_cases["neither_name_is_gold"] += 1
            elif linked[1] & positives:
                wrong_cases["also_linked_the_gold_name"] += 1
            else:
                wrong_cases["only_linked_the_distractor"] += 1
        for i in (linked[0] - linked[1]) & positives:
            formerly_correct[unlinked_stage(after, pronoun, names[i])] += 1
        for stage in STAGES:
            covered = [
                {i for i in positives if pronoun in spans and names[i] in spans}
                for spans in (before[stage], after[stage])
            ]
            count_sets(coverage[stage], *covered)
            count_sets(
                unmarked[stage], before[stage].difference(targets), after[stage].difference(targets)
            )
        common = (before["returned"] & after["returned"]).difference(targets)
        count_sets(common_links, *(shared_links(g, common) for g in groups))
        for i in positives:
            name = names[i]
            if name in before["proposed"] or len(row["Text"][slice(*name)].split()) < 2:
                continue
            pieces = [(a, b) for a, b in before["proposed"] if a < name[1] and name[0] < b]
            if pieces and all(name[0] <= a and b <= name[1] for a, b in pieces):
                partial["stock_pairs"] += 1
                partial["exact_proposals_gained"] += name in after["proposed"]
                partial["correct_links_gained"] += i in linked[1]

    hashes = [gap_receipts.digest(arm) for arm in partitions]
    if hashes[0] != frozen["partitions_sha256"]:
        raise ValueError("full stock partitions differ from their earlier receipt")
    if partial["stock_pairs"] != frozen["name_boundary_observations"]["multiword_only_subspans"]:
        raise ValueError("targeted stock partial-name cohort differs from its earlier receipt")
    return {
        "excerpts": len(rows),
        "gold_pairs": frozen["gold_pairs"],
        "changed_partitions": changed,
        "partition_sha256": dict(zip(("stock", "candidate"), hashes)),
        "gold_pairs_with_both_exact_spans": coverage,
        "correct_links": correct,
        "wrong_links": wrong,
        "newly_wrong_links_by_stock_stage": dict(newly_wrong),
        "newly_wrong_links_by_candidate_outcome": dict(wrong_cases),
        "lost_correct_links_by_candidate_stage": dict(formerly_correct),
        "unmarked_spans": unmarked,
        "co_clustered_pairs_among_common_unmarked_spans": common_links,
        "targeted_partial_names": partial,
        "passes_gate": correct["candidate"] > correct["stock"] and wrong["gained"] == 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "stock", "candidate", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--public-candidate", type=Path, help="verify the public coref_gap_panel example")
    args = parser.parse_args()
    rows, hashes = read_sources(args.data)
    binaries = (args.stock, args.candidate)
    sample = "we met Léonie Márquez today"
    start = sample.index("Léonie")
    span = [start, start + len("Léonie Márquez")]
    control = [
        span in trace_documents([("control", sample)], test_binary=binary)[0]["proposed"]
        for binary in binaries
    ]
    if control != [False, True]:
        raise ValueError("stock and candidate do not reproduce the project-authored control")
    frozen = json.loads((ROOT / "docs/eval/native-gap-endpoint-stages.json").read_text())
    panels = {}
    for split in ("development", "test", "validation"):
        docs = [(r["ID"], r["Text"]) for r in rows[split]]
        both = [trace_documents(docs, test_binary=binary) for binary in binaries]
        panels[split] = compare(rows[split], *both, frozen["panels"][split])
        if args.public_candidate:
            _, groups = evaluate(rows[split], "predicted", args.public_candidate)
            if gap_receipts.digest(groups) != panels[split]["partition_sha256"]["candidate"]:
                raise ValueError("candidate trace differs from the public worker")
    receipt = {
        "stock_revision": BASE,
        "candidate_patch_sha256": file_hash(PATCH),
        "source_revision": REVISION,
        "source_sha256": hashes,
        "panels": panels,
    }
    args.output.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
