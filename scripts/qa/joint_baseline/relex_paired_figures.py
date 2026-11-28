# SPDX-License-Identifier: MIT OR Apache-2.0
"""Rebuild native paired CrossRE chart inputs from aggregate measurements without running a model."""

import argparse
import json
from pathlib import Path

from relex_endpoint_budget import endpoint_chart
from relex_full_test import BASELINE, paired_interval, write_json


def row(panel, label, role):
    rules = panel["additional_edges"]
    baseline = rules["score"]
    candidate = rules["both_margins"]
    if baseline["added"] != candidate["added"]:
        raise ValueError("Confidence and margins have different review budgets")

    def arm(observed):
        return {"correct": observed["correct_added"], "unmatched": observed["unmatched_added"]}

    return {
        "label": label, "role": role, "observations": panel["stock"]["sentences"],
        "baseline": arm(baseline), "candidate": arm(candidate),
        "gained": candidate["correct_gained_vs_score_only"],
        "lost": candidate["correct_lost_vs_score_only"],
    }


def charts(balanced, full):
    def chart(title, rows):
        return {"schema": "anno.paired/v1", "title": title, "unit": "sentences",
                "baseline": "Confidence", "candidate": "Both margins", "rows": rows}

    p = full["panels"]
    followup = p["additional_test"]
    fresh = row(followup, "Additional distinct test texts", "Held out from the original study")
    fresh["interval"] = paired_interval(domain["paired_review_slots"] for domain in followup["domains"].values())
    return {
        "crossre-predicate-paired.json": chart("Extra CrossRE matches in the original ranking study", [
            row(balanced["panels"]["dev"], "Development", "Rule selection"),
            row(balanced["panels"]["test"], "Original balanced test", "Initial test panel"),
        ]),
        "crossre-full-test-paired.json": chart("Extra CrossRE matches at equal review budgets", [
            fresh, row(p["balanced_test"], "Original balanced panel", "Previously examined"),
            row(p["outside_content_holdout"], "Outside the text holdout", "Separate release occurrences"),
            row(p["full_official_test"], "Complete official test release", "Total of the preceding three rows"),
        ]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts", type=Path, default=BASELINE.parent)
    parser.add_argument("--output", type=Path, default=BASELINE.parent)
    parser.add_argument("--include-endpoint", action="store_true",
                        help="Also rebuild the endpoint-confidence chart from crossre-endpoint-budget.json")
    args = parser.parse_args()
    original = json.loads((args.receipts / BASELINE.name).read_text())
    full = json.loads((args.receipts / "crossre-full-test.json").read_text())
    figures = charts(original, full)
    if args.include_endpoint:
        endpoint = json.loads((args.receipts / "crossre-endpoint-budget.json").read_text())
        figures["crossre-endpoint-paired.json"] = endpoint_chart(endpoint)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, figure in figures.items():
        write_json(args.output / name, figure)
