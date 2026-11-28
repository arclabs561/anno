#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Compare Fastino to the training-majority class on ViRA test and validation turns."""

import argparse
import json
from pathlib import Path

EVAL = Path(__file__).resolve().parents[2] / "docs/eval"


def paired(receipt):
    if receipt["schema"] != "anno.vira-speech-act/v1":
        raise ValueError("Expected the ViRA speech act evidence receipt")
    label = receipt["splits"]["train"]["training_majority"]["label"]
    rows = []
    for split, name in (("test", "Held-out test"), ("validation", "Validation")):
        score = receipt["splits"][split]
        majority = score["training_majority"]
        if majority["label"] != label:
            raise ValueError(
                f"Baseline class differs from training majority on {split}"
            )
        # A constant baseline is right precisely on the gold turns of that class.
        support = score["by_label"][label]
        if majority["correct"] != support["gold"]:
            raise ValueError(f"Majority baseline differs from gold support on {split}")
        retained = support["correct"]
        n, correct, abstained = score["examples"], score["correct"], score["abstained"]
        unmatched = n - correct - abstained
        gained = correct - retained
        lost = majority["correct"] - retained
        if min(gained, lost, unmatched) < 0:
            raise ValueError(f"Inconsistent outcome counts on {split}")
        rows.append(
            {
                "label": name,
                "role": f"{abstained} Fastino abstentions",
                "observations": n,
                "baseline": {
                    "correct": majority["correct"],
                    "unmatched": n - majority["correct"],
                },
                "candidate": {"correct": correct, "unmatched": unmatched},
                "gained": gained,
                "lost": lost,
            }
        )
    return {
        "schema": "anno.paired/v1",
        "title": "Which ViRA turns does Fastino gain or lose?",
        "unit": "dialogue turns",
        "baseline": f"Always predict {label.replace('_', ' ')}",
        "candidate": "Fastino classifier",
        "rows": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=EVAL / "vira-speech-act.json")
    parser.add_argument(
        "--output", type=Path, default=EVAL / "vira-majority-paired.json"
    )
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(paired(json.loads(args.input.read_bytes())), indent=2) + "\n"
    )
