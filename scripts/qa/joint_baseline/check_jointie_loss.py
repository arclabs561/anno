"""Audit the expanded CrossRE receipts against source annotations and frozen eight-row receipts."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from jointie_coverage import gold_graph


def check(source, expanded, previous=None):
    rows = expanded["documents"]
    if len(rows) != len(source) or not rows:
        raise ValueError("source and receipt counts differ")
    if expanded["enrolled_documents"] != len(rows) or expanded[
        "scored_documents"
    ] != len(rows):
        raise ValueError("enrollment or scored count differs from the document rows")
    totals = Counter()
    for document, row in zip(source, rows, strict=True):
        text, gold_mentions, gold = gold_graph(document)
        if hashlib.sha256(text.encode()).hexdigest() != row["source_sha256"]:
            raise ValueError("source text digest differs")
        if row["document"] != document["doc_key"]:
            raise ValueError("document order changed")
        selected = {(tuple(h), p, tuple(t)) for h, p, t in row["decoded_edge_keys"]}
        predicted_mentions = set(map(tuple, row["decoded_mention_keys"]))
        if any(
            h not in predicted_mentions or t not in predicted_mentions
            for h, _, t in selected
        ):
            raise ValueError("decoded edge references an absent mention")
        counts = row["counts"]
        if (
            len(gold),
            len(selected),
            len(gold & selected),
            len(gold_mentions & predicted_mentions),
            len(gold_mentions),
            len(predicted_mentions),
        ) != (
            counts["gold_edges"],
            counts["decoded_edges"],
            counts["gold_edges_decoded"],
            counts["gold_mentions_decoded"],
            counts["gold_mentions"],
            counts["decoded_mentions"],
        ):
            raise ValueError(
                f"{row['document']}: decoded edge or mention counts differ"
            )
        keys = set()
        losses = Counter()
        for item in row["edge_trace"]:
            h, t = tuple(item["head"]), tuple(item["tail"])
            key = (h, item["predicate"], t)
            if key in keys:
                raise ValueError("duplicate annotated edge in trace")
            keys.add(key)
            if item["co_windows"] != [
                i
                for i, w in enumerate(row["windows"])
                if w["start"] <= min(h[1], t[1]) and max(h[2], t[2]) <= w["end"]
            ]:
                raise ValueError("co-window coverage disagrees with source spans")
            loss = item["first_loss"]
            losses[loss] += 1
            if (loss == "matched") != (key in selected):
                raise ValueError("matched edge absent from the written graph")
            if (
                loss == "edge_cap"
                and item["rank_after_gate"] <= item["per_predicate_cap"]
            ):
                raise ValueError("claimed cap loss has a retained rank")
            if (
                loss == "optimizer_did_not_select"
                and "optimizer_edge_utility" not in item
            ):
                raise ValueError("optimizer attribution lacks solver input")
        if keys != gold or losses != {
            k.removeprefix("first_loss_"): n
            for k, n in counts.items()
            if k.startswith("first_loss_")
        }:
            raise ValueError(
                f"{row['document']}: gold coverage or loss categories differ"
            )
        totals.update(counts)
    if dict(totals) != expanded["totals"]:
        raise ValueError("aggregate totals disagree with per-document counts")
    if previous:
        for old, current in zip(
            previous["documents"], rows[: len(previous["documents"])], strict=True
        ):
            if (old["document"], old["windows"], old["source_sha256"]) != (
                current["document"],
                current["windows"],
                current["source_sha256"],
            ) or any(n != current["counts"][k] for k, n in old["counts"].items()):
                raise ValueError(f"historical trace differs on {old['document']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("crossre", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--previous", type=Path)
    args = parser.parse_args()
    expanded = json.loads(args.receipt.read_bytes())
    raw_source = args.crossre.read_bytes()
    if hashlib.sha256(raw_source).hexdigest() != expanded["dataset_sha256"]:
        raise ValueError("source dataset digest differs")
    source = [
        json.loads(s) for s in raw_source.splitlines()[: len(expanded["documents"])]
    ]
    check(
        source,
        expanded,
        json.loads(args.previous.read_bytes()) if args.previous else None,
    )
    print(
        f"Verified {len(source)} source documents and {expanded['totals']['gold_edges']} labeled edges."
    )


if __name__ == "__main__":
    main()
