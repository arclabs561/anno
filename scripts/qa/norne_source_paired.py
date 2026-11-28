"""Build the portable NorNE test comparison from the six-split native replay."""

import argparse
import json
from pathlib import Path

EVAL = Path(__file__).resolve().parents[2] / "docs/eval"
STANDARDS = {"nob": "Bokmål", "nno": "Nynorsk"}


def paired(receipt):
    if (
        receipt["schema"] != "anno.norne-source-replay/v1"
        or receipt["dataset"] != "NorNE"
        or receipt["backend"] != "heuristic"
    ):
        raise ValueError("Expected the NorNE heuristic replay")
    rows = []
    for code, standard in STANDARDS.items():
        panel = receipt["splits"][f"{code}-test"]
        old, new, pairs = panel["before"], panel["after"], panel["paired_gold"]
        gold = panel["gold"]
        if (
            not 0 <= panel["source_gold_exact_before"] <= gold
            or panel["source_gold_exact_after"] != gold
            or old["gold"] != gold
            or new["gold"] != gold
            or min(pairs["retained"], pairs["gained"], pairs["lost"]) < 0
            or pairs["retained"] + pairs["gained"] + pairs["lost"] > gold
            or old["correct"] != pairs["retained"] + pairs["lost"]
            or new["correct"] != pairs["retained"] + pairs["gained"]
            or old["predicted"] < old["correct"]
            or new["predicted"] < new["correct"]
        ):
            raise ValueError(f"Paired matches disagree with the {standard} test counts")
        shifted = gold - panel["source_gold_exact_before"]
        rows.append(
            {
                "label": f"{standard} test",
                "role": f"{gold:,} gold mentions · {shifted:,} with corrected offsets",
                "observations": panel["cases"],
                "baseline": {
                    "correct": old["correct"],
                    "unmatched": old["predicted"] - old["correct"],
                },
                "candidate": {
                    "correct": new["correct"],
                    "unmatched": new["predicted"] - new["correct"],
                },
                "gained": pairs["gained"],
                "lost": pairs["lost"],
            }
        )
    return {
        "schema": "anno.paired/v1",
        "title": "How does source spacing change NorNE heuristic matches?",
        "unit": "publisher sentences",
        "baseline": "Space-joined tokens",
        "candidate": "Original Norwegian text",
        "rows": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=EVAL / "norne-source-replay.json")
    parser.add_argument("--output", type=Path, default=EVAL / "norne-source-paired.json")
    args = parser.parse_args()
    report = paired(json.loads(args.input.read_bytes()))
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
