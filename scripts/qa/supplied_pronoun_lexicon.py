"""Replay a fixed case-insensitive pronoun vocabulary on supplied native mentions."""
import argparse
from collections import Counter
import json
from pathlib import Path

from audit_gap_decisions import gold
from coref_gap import check_file
from native_gap_margin import read_panel
from native_gap_neutral_type import ROOT, evaluate, read_sources


def decisions(rows, observations):
    totals = Counter(correct=0, wrong=0, abstained=0)
    for row, bits in zip(rows, observations, strict=True):
        choices = bits[3:]
        if choices.count("1") == 1:
            totals["correct" if gold(row)[choices.index("1")] else "wrong"] += 1
        else:
            totals["abstained"] += 1
    return dict(totals)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("gap_dir", "litbank_panel", "stock", "candidate"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    rows, sources = read_sources(args.gap_dir)
    frozen = json.loads((ROOT / "docs/eval/litbank-pronoun-transfer.json").read_text())
    sources["litbank"] = frozen["panel_tsv_sha256"]
    check_file(args.litbank_panel, sources["litbank"])
    rows["litbank"] = read_panel(args.litbank_panel)
    results = {}
    for split, panel in rows.items():
        before, old_groups = evaluate(panel, "supplied", args.stock)
        after, new_groups = evaluate(panel, "supplied", args.candidate)
        results[split] = {
            "documents": len(panel),
            "capitalized_target": sum(r["Pronoun"] != r["Pronoun"].lower() for r in panel),
            "stock": decisions(panel, before),
            "candidate": decisions(panel, after),
            "changed_partitions": sum(a != b for a, b in zip(old_groups, new_groups, strict=True)),
            "changed_with_lowercase_target": sum(
                a != b and r["Pronoun"].islower()
                for r, a, b in zip(panel, old_groups, new_groups, strict=True)
            ),
        }
    print(json.dumps({"source_sha256": sources, "panels": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
