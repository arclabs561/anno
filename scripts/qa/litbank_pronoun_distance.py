# SPDX-License-Identifier: MIT OR Apache-2.0
"""Replay stock scoring on the pinned supplied LitBank panel at two fixed scalar gaps."""
import argparse
from collections import Counter
import json
from pathlib import Path

from coref_gap import check_file
from native_gap_margin import diagnostics, file_hash, paired, read_panel, run, summaries

RECEIPTS = Path(__file__).resolve().parents[2] / "docs" / "eval"


def boundary_probe(gap):
    # Multibyte padding makes a byte-based distance limit fail these controls.
    text = "Ada" + "é" * gap + "he saw Zed"
    return {"ID": f"authored-gap-{gap}", "Text": text, "A": "Ada", "B": "Zed", "Pronoun": "he",
            "A-offset": "0", "B-offset": str(text.index("Zed")), "Pronoun-offset": str(3 + gap),
            "A-coref": "TRUE", "B-coref": "FALSE"}


def arm(lookups, positive):
    stages = diagnostics(lookups)
    lost = stages["gold_stage"]
    eligible = positive - lost.get("gold_outside_distance", 0) - lost.get("gold_follows_pronoun", 0)
    return {"scored_pronoun_name_pairs": sum(r.eligible for r in lookups), "gold_eligible": eligible, "gold_selected_directly": lost.get("gold_selected_directly", 0),
            "returned": summaries(lookups, [r.issued for r in lookups]), "diagnostics": stages}


def score(panel, worker):
    selection = json.loads((RECEIPTS / "litbank-pronoun-transfer.json").read_bytes())
    previous = json.loads((RECEIPTS / "litbank-pronoun-transfer-score.json").read_bytes())
    check_file(panel, selection["panel_tsv_sha256"])
    rows = read_panel(panel)
    gaps = [100, 101, 500, 501]
    cases = [boundary_probe(gap) for gap in gaps] + rows
    arms = []
    for cap in (100, 500):
        lookups = run(cases, worker, book_distance=cap == 500)
        if [row.eligible for row in lookups[:len(gaps)]] != [int(g <= cap) for g in gaps]:
            raise ValueError(f"native scored visits disagree with the {cap}-scalar boundary")
        arms.append(lookups[len(gaps):])
    stock, wide = arms
    positive = sum(r.gold is not None for r in stock)
    small, large = (arm(records, positive) for records in arms)
    if small["returned"] != previous["stock"] or small["diagnostics"] != previous["stock_diagnostics"]:
        raise ValueError("default worker differs from the previous stock evaluation")
    before = [r.issued for r in stock]
    after = [r.issued for r in wide]
    work_ids = [row["URL"] for row in rows]
    changed_works = {w for w, b, a in zip(work_ids, before, after, strict=True) if b != a}
    newly_eligible = Counter(
        later.gold_stage for earlier, later in zip(stock, wide, strict=True)
        if earlier.gold_stage == "gold_outside_distance" and later.gold_stage != "gold_outside_distance")
    if sum(newly_eligible.values()) != large["gold_eligible"] - small["gold_eligible"]:
        raise ValueError("newly eligible gold antecedents differ from eligibility change")
    return {"panel_sha256": file_hash(panel), "stock_binary_sha256": file_hash(worker),
            "rows": len(rows), "works": len(set(work_ids)), "gold_links": positive,
            "max_distance_100": small, "max_distance_500": large, "newly_eligible_gold": dict(newly_eligible),
            "paired": paired(stock, before, after, work_ids), "works_with_changed_target": len(changed_works)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", type=Path)
    parser.add_argument("stock_worker", type=Path)
    parser.add_argument("result", type=Path)
    parser.add_argument("--check", action="store_true", help="require all build-independent fields to match the receipt")
    args = parser.parse_args()
    result = score(args.panel, args.stock_worker)
    if args.check:
        expected = json.loads((RECEIPTS / "litbank-pronoun-distance.json").read_bytes())
        if {**result, "stock_binary_sha256": expected["stock_binary_sha256"]} != expected:
            raise ValueError("distance scores differ from the published receipt")
    args.result.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
