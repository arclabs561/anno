#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface_hub>=0.25"]
# ///
"""Prepare the pinned Kniv v7.1 English model and local Python runtime sources."""
import argparse
import runpy
from pathlib import Path

REPO = "dragonscale-ai/kniv-v7-large-en"
REVISION = "44e6d50774a22ada4fd825e1c48663afb5fd7dc0"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="destination for KNIV_MODEL_DIR")
    directory = parser.parse_args().directory
    worker = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                                "crates/anno/src/backends/kniv_worker.py"))
    files = {**worker["FILES"], **{f"kniv_v7/{name}": digest for name, digest in worker["SOURCES"].items()}}
    missing = []
    for name, digest in files.items():
        try:
            worker["verify"](directory / name, digest)
        except (FileNotFoundError, ValueError):
            missing.append(name)
    if missing:
        from huggingface_hub import snapshot_download
        snapshot_download(REPO, revision=REVISION, local_dir=directory, allow_patterns=missing,
                          force_download=True)
        for name in missing:
            worker["verify"](directory / name, files[name])
    print(f"Verified {len(files)} files in {directory.resolve()}")


if __name__ == "__main__":
    main()
