#!/usr/bin/env python3
"""Fetch and verify a revision-pinned Otter mmBERT inference snapshot."""

import argparse
import json
from pathlib import Path

from _prepare_snapshot import prepare


def main():
    pins = (
        Path(__file__).resolve().parents[1]
        / "crates/anno/src/backends/otter/models.json"
    )
    models = json.loads(pins.read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=sorted(models))
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args()
    spec = models[args.variant]
    other_configs = [
        m["files"]["config.json"] for key, m in models.items() if key != args.variant
    ]
    prepare(
        args.model_dir,
        spec["repo"],
        spec["revision"],
        spec["files"],
        other_configs=other_configs,
    )
    print(f"Verified {args.variant} in {args.model_dir.resolve()}")


if __name__ == "__main__":
    main()
