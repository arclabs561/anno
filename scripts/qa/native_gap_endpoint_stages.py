"""Attribute native raw-text GAP losses and verify the entire emitted partition."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import audit_gap_decisions as gap
import gap_receipts
from coref_gap import REVISION
from native_gap_loss import stage_details, trace_documents
from native_gap_neutral_type import ROOT, evaluate, read_sources


def checked_partition(trace):
    """Reconstruct linked groups from every tracing decision, including unmarked mentions."""
    spans = list(map(tuple, trace["retained"]))
    expected = []
    # Earlier starts win; for identical starts, the widest proposal wins.
    for span in sorted(map(tuple, trace["proposed"]), key=lambda s: (s[0], -s[1])):
        if not expected or span[0] >= expected[-1][1]:
            expected.append(span)
    if spans != expected:
        raise ValueError("retained spans differ from overlap deduplication")
    roots = []
    members = defaultdict(list)
    for i, (span, decision) in enumerate(zip(spans, trace["decisions"], strict=True)):
        prior = range(i - 1, max(-1, i - 1 - decision["max_antecedents"]), -1)
        eligible = [j for j in prior if span[0] - spans[j][1] <= trace["max_distance"]]
        candidates = decision["candidates"]
        if [j for j, _, _ in candidates] != eligible:
            raise ValueError("scored antecedents differ from the distance and count limits")
        # The native loop visits nearest first and replaces a winner only on a strict gain.
        passing = [(j, score) for j, score, threshold in candidates if score > threshold]
        winner, _ = max(passing, key=lambda p: p[1], default=(None, 0))
        parent = decision["selected"]
        if parent != winner:
            raise ValueError("selected antecedent differs from the recorded scores")
        root = i if parent is None else roots[parent]
        roots.append(root)
        members[root].append(span)
    reconstructed = sorted(sorted(group) for group in members.values() if len(group) > 1)
    emitted = sorted(sorted(map(tuple, group)) for group in trace["clusters"])
    if reconstructed != emitted:
        raise ValueError("selected edges do not reproduce the full emitted partition")
    return emitted


def summarize(rows, traces, frozen, stock=None):
    docs = [(r["ID"], r["Text"], gap.spans(r), gap.gold(r)) for r in rows]
    ids_hash = gap.sha("".join(r["ID"] + "\n" for r in rows).encode())
    if ids_hash != frozen["id_list_sha256"]:
        raise ValueError("source order differs from the frozen public run")
    partitions = [checked_partition(t) for t in traces]
    partition_hash = gap_receipts.digest(partitions)
    if partition_hash != frozen["predicted"]["stock_partitions_sha256"]:
        raise ValueError("full partitions differ from the frozen public worker")
    if stock is not None and partitions != evaluate(rows, "predicted", stock)[1]:
        raise ValueError("full partitions differ from the selected public worker")
    details = stage_details(docs, traces)
    counts = frozen["predicted"]["stock"]
    if (
        details["gold_pairs"] != counts["gold_links"]
        or details["stages"].get("linked", 0) != counts["tp"]
        or len(details["false_links"]) != counts["fp"]
    ):
        raise ValueError("stage outcomes disagree with the frozen decision counts")
    path_lengths = Counter(len(r["selected_path_chars"]) - 1 for r in details["false_links"])
    aggregates = {k: v for k, v in details.items() if k not in ("false_links", "rows")}
    return aggregates | {
        "distinct_articles": len({r["URL"] for r in rows}),
        "id_list_sha256": ids_hash,
        "partitions_sha256": partition_hash,
        "false_link_path_edges": dict(sorted(path_lengths.items())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="directory containing the three pinned GAP TSVs")
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--stock", type=Path, help="also compare partitions with a fresh coref_gap_panel run"
    )
    args = parser.parse_args()
    inputs, hashes = read_sources(args.data)
    frozen = json.loads((ROOT / "docs/eval/native-gap-neutral-mention.json").read_text())["panels"]
    published = json.loads((ROOT / "docs/eval/coref-panels.json").read_text())["gap"]
    panels = {}
    for split, rows in inputs.items():
        traces = trace_documents([(r["ID"], r["Text"]) for r in rows])
        panels[split] = summarize(rows, traces, frozen[split], args.stock)
        if split == "test":
            by_id = {t["id"]: t for t in traces}
            cohorts = gap.load_rows(args.data / "gap-test.tsv", published)
            for key, cohort in cohorts.items():
                subset = [by_id[r["ID"]] for r in cohort]
                panels[key] = summarize(cohort, subset, frozen["test_subsets"][key])
    receipt = {"source_revision": REVISION, "source_sha256": hashes, "panels": panels}
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
