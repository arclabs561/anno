"""Replay the frozen English possessive-name candidate on all three raw-text GAP splits."""
import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile

import audit_gap_decisions as gap
from native_gap_neutral_type import ROOT, read_sources
from native_gap_loss import stage_details
from native_gap_possessives import run_trace, score


def compare(rows, before, after, frozen):
    stock, stock_masks = before
    trial, trial_masks = after
    for key in ("tp", "fp", "gold_links"):
        if stock["counts"][key] != frozen[key]:
            raise ValueError(f"stock {key} differs from the published decision count")
    changes = dict.fromkeys((
        "gold_proposals_gained", "gold_proposals_lost", "gained_proposals_linked",
        "correct_links_gained", "correct_links_lost", "wrong_links_gained", "wrong_links_lost",
    ), 0)
    for row, old, new, stock_offered, trial_offered in zip(
        rows, stock["per_id"], trial["per_id"], stock_masks, trial_masks, strict=True
    ):
        for i, gold in enumerate(gap.gold(row)):
            proposed = stock_offered[0] == stock_offered[i + 1] == "1"
            now_proposed = trial_offered[0] == trial_offered[i + 1] == "1"
            linked, now_linked = old[i + 3] == "1", new[i + 3] == "1"
            if gold and now_proposed != proposed:
                changes["gold_proposals_gained" if now_proposed else "gold_proposals_lost"] += 1
                if now_proposed and now_linked:
                    changes["gained_proposals_linked"] += 1
            if linked != now_linked:
                kind = "correct" if gold else "wrong"
                changes[f"{kind}_links_{'gained' if now_linked else 'lost'}"] += 1
    return {
        "documents": len(rows),
        "gold_pairs": stock["counts"]["gold_links"],
        "stock": {"correct": stock["counts"]["tp"], "wrong": stock["counts"]["fp"]},
        "candidate": {"correct": trial["counts"]["tp"], "wrong": trial["counts"]["fp"]},
        "within_stock_wrong_link_budget": trial["counts"]["fp"] <= stock["counts"]["fp"],
        "paired_changes": changes,
    }


def run(worker, docs, directory):
    traces = run_trace(worker, docs, directory)
    return score(docs, traces), stage_details(docs, traces)["rows"]


def follow_possessive_pairs(before, after):
    stages, ranks = Counter(), Counter()
    for prev, curr in zip(before, after, strict=True):
        if (prev["id"], prev["candidate"]) != (curr["id"], curr["candidate"]):
            raise ValueError("different labeled pair in the two traces")
        if prev.get("name_boundary_observation") == "including_possessive_clitic":
            stages[curr["first_loss_or_link"]] += 1
            if "direct_pair" in curr:
                ranks[curr["direct_pair"]] += 1
    return {"candidate_outcome": dict(stages), "candidate_direct_pair_when_retained": dict(ranks)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("stock", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    inputs, hashes = read_sources(args.data)
    frozen = json.loads((ROOT / "docs/eval/native-gap-neutral-mention.json").read_text())["panels"]
    panels = {}
    with tempfile.TemporaryDirectory() as directory:
        for split, rows in inputs.items():
            docs = [(r["ID"], r["Text"], gap.spans(r), list(gap.gold(r))) for r in rows]
            before, old_stages = run(args.stock, docs, Path(directory))
            after, new_stages = run(args.candidate, docs, Path(directory))
            panels[split] = compare(rows, before, after, frozen[split]["predicted"]["stock"])
            panels[split]["stock_possessive_pairs"] = follow_possessive_pairs(old_stages, new_stages)
    receipt = {"source_sha256": hashes, "panels": panels}
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
