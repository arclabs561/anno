"""Replay the native ranker on pinned GAP inputs and compare complete mention partitions."""

import argparse
import json
from pathlib import Path

import native_gap_neutral_type as study


def check(data, binary):
    rows_by_split, source_hashes = study.read_sources(data)
    saved = json.loads((study.ROOT / "docs/eval/native-gap-neutral-mention.json").read_text())
    if source_hashes != saved["source_sha256"]:
        raise ValueError("GAP inputs differ from the held-out study")
    for split, rows in rows_by_split.items():
        panel = saved["panels"][split]
        ids = "".join(row["ID"] + "\n" for row in rows)
        if study.gap.sha(ids.encode()) != panel["id_list_sha256"]:
            raise ValueError(f"{split}: panel identity or order changed")
        for policy in ("supplied", "predicted"):
            bits, partitions = study.evaluate(rows, policy, binary)
            counts = study.gap.decision_counts(rows, bits)
            digest = study.gap_receipts.digest(partitions)
            target = panel[policy]
            if counts != target["stock"] or digest != target["stock_partitions_sha256"]:
                raise ValueError(
                    f"{split}/{policy}: native output differs from the recorded stock system; "
                    f"counts={counts}, partition digest={digest}"
                )
            print(f"{split}/{policy}: {len(rows)} documents, "
                  f"{counts['tp']} correct and {counts['fp']} incorrect links")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="directory with the three publisher GAP TSV files")
    parser.add_argument("binary", type=Path, help="built coref_gap_panel Rust example")
    args = parser.parse_args()
    check(args.data, args.binary)
