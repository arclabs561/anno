# SPDX-License-Identifier: MIT OR Apache-2.0
"""Verify the historical replay and compare two paired argument-gate arms."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from aggregate_receipt import project
from jointie_coverage import TEST_SHA, gold_graph


def compare(source, historical, stock, lower):
    published = Path(__file__).parent / "receipts/jointie_ai64_w384.json"
    historical_reasons = Counter()
    for row in historical["documents"]:
        for edge in row["edge_trace"]:
            stages = edge.pop("sparse_role_stages")
            if edge["first_loss"] == "not_in_sparse_edge_output":
                if len(stages) != 1:
                    raise ValueError("expected one located loss for each missing historical edge")
                historical_reasons[stages[0]] += 1
    if project(historical, "jointie") != json.loads(published.read_bytes()):
        raise ValueError("instrumentation changed the pinned historical result")
    if hashlib.sha256(source).hexdigest() != TEST_SHA:
        raise ValueError("expected pinned official AI test split")
    if stock.get("argument_threshold") is not None or lower.get("argument_threshold") != 0.10:
        raise ValueError("unexpected intervention")
    for arm in (stock, lower):
        project(arm, "jointie")
        if arm.get("first_index") != 64 or len(arm["documents"]) != 32:
            raise ValueError("expected the disjoint 32-document slice")
        for key in ("checkpoint_files", "window_words", "overlap_words", "vocabulary_sha256", "torch", "gliner2"):
            if arm[key] != historical[key]:
                raise ValueError(f"mismatched replay setting {key}")
        if "training_sha256" in arm:
            raise ValueError("expected the same broad vocabulary in both arms")

    seen = {row["source_sha256"] for row in historical["documents"]}
    gains, losses = Counter(), Counter()
    for line, old, new in zip(source.splitlines()[64:96], stock["documents"], lower["documents"], strict=True):
        doc = json.loads(line)
        text, _, gold = gold_graph(doc)
        sha = hashlib.sha256(text.encode()).hexdigest()
        if sha in seen:
            raise ValueError("overlap within or between cohorts")
        seen.add(sha)
        output = []
        for row in (old, new):
            if row["source_sha256"] != sha or row["document"] != doc["doc_key"]:
                raise ValueError("sample identity differs from declared source order")
            predictions = {(tuple(h), r, tuple(t)) for h, r, t in row["decoded_edge_keys"]}
            if (len(gold), len(predictions), len(predictions & gold)) != tuple(
                row["counts"][k] for k in ("gold_edges", "decoded_edges", "gold_edges_decoded")
            ):
                raise ValueError("recomputed scores disagree with the model receipt")
            output.append(predictions)
        for delta, metrics in ((output[1] - output[0], gains), (output[0] - output[1], losses)):
            metrics.update(gold=len(delta & gold), unmatched=len(delta - gold))
    keys = ("gold_edges", "actual_scored_edges", "gold_edges_scored", "decoded_edges", "gold_edges_decoded")
    return {"historical_sparse_losses": dict(sorted(historical_reasons.items())),
            "fresh_stock": {k: stock["totals"][k] for k in keys},
            "fresh_lower": {k: lower["totals"][k] for k in keys},
            "added": dict(gains), "removed": dict(losses)}


def compare_ranking(stock, ranked):
    totals = {"cap": Counter(), "output": Counter()}
    for row in ranked["documents"]:
        trial = row.pop("ranking_trial")
        for stage in totals:
            scores = trial[stage]
            if (scores["after"] - scores["before"] != scores["added_gold"] + scores["added_unmatched"]
                    - scores["removed_gold"] - scores["removed_unmatched"]
                    or scores["gold_after"] - scores["gold_before"] != scores["added_gold"] - scores["removed_gold"]):
                raise ValueError("paired rating deltas do not reconcile")
            totals[stage].update(scores)
        gold_capped = sum(e.get("rank_after_gate") is not None and e["rank_after_gate"] <= e["per_predicate_cap"]
                          for e in row["edge_trace"])
        if gold_capped != trial["cap"]["gold_before"] or trial["output"]["gold_before"] != row["counts"]["gold_edges_decoded"]:
            raise ValueError("stock fixed-cap and output scores disagree with the original trace")
    if ranked != stock:
        raise ValueError("stock scores or documents changed while replaying alternative ranking")
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "historical", "stock", "lower"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--ranked", type=Path, help="Inspect an additional paired fixed-cap ranking trial")
    args = parser.parse_args()
    historical, stock, lower = (json.loads(p.read_bytes()) for p in (args.historical, args.stock, args.lower))
    report = compare(args.source.read_bytes(), historical, stock, lower)
    if args.ranked:
        report["fixed_cap_ranking"] = compare_ranking(stock, json.loads(args.ranked.read_bytes()))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
