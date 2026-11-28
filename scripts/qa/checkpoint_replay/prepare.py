"""Prepare the exact checkpoint bytes consumed by the offline CI replays."""
import argparse
import hashlib
import json
from pathlib import Path

from run import HERE, verify_assets
from stacked import stacked_assets


def assets(group):
    models = json.loads((HERE / "models.json").read_text())
    if group == "stacked":
        return dict(stacked_assets())
    # Changes to expected spans or test cases do not invalidate a weights cache.
    return {name: {key: spec[key] for key in ("repo", "revision", "files")}
            for name, spec in models.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group", choices=["native", "stacked"])
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--key", action="store_true", help="Print the cache key for these bytes")
    action.add_argument("--models", type=Path, help="Verify checkpoints in this directory")
    parser.add_argument("--fetch", action="store_true", help="Download missing or incorrect files")
    args = parser.parse_args()
    specs = assets(args.group)
    if args.key:
        manifest = json.dumps(specs, sort_keys=True, separators=(",", ":")).encode()
        print(f"anno-weights-v1-{args.group}-{hashlib.sha256(manifest).hexdigest()}")
        return
    for name, spec in specs.items():
        print(f"Verifying {spec['repo']}@{spec['revision']}", flush=True)
        verify_assets(args.models / name, spec, args.fetch)


if __name__ == "__main__":
    main()
