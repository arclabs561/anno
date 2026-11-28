#!/usr/bin/env python3
"""Fetch a revision-pinned Otter mmBERT inference snapshot."""
import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    pins = Path(__file__).resolve().parents[1] / "crates/anno/src/backends/otter/models.json"
    models = json.loads(pins.read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=sorted(models))
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args()
    spec = models[args.variant]
    snapshot_download(
        repo_id=spec["repo"], revision=spec["revision"],
        local_dir=args.model_dir, allow_patterns=list(spec["files"]),
    )
    print(args.model_dir.resolve())


if __name__ == "__main__":
    main()
