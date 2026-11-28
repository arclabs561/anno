#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface_hub>=0.25"]
# ///
"""Prepare the pinned Kniv v7.1 English model and local Python runtime sources."""

import argparse
import runpy
from pathlib import Path

from _prepare_snapshot import prepare

REPO = "dragonscale-ai/kniv-v7-large-en"
REVISION = "44e6d50774a22ada4fd825e1c48663afb5fd7dc0"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="destination for KNIV_MODEL_DIR")
    parser.add_argument(
        "--ner-only",
        action="store_true",
        help="Prepare the runtime and graphs used for fixed-label NER",
    )
    args = parser.parse_args()
    directory = args.directory
    worker = runpy.run_path(
        str(
            Path(__file__).resolve().parents[1]
            / "crates/anno/src/backends/kniv_worker.py"
        )
    )
    files = {
        **worker["FILES"],
        **{f"kniv_v7/{name}": digest for name, digest in worker["SOURCES"].items()},
    }
    if args.ner_only:
        # The upstream window method also decodes coreference, dependency and
        # semantic-role heads while returning the fixed BIO predictions.
        ner_graphs = {
            f"onnx/{head}-int8.onnx"
            for head in ("plain", "dep_rel", "srl_roles", "coref_spans")
        }
        files = {
            name: digest
            for name, digest in files.items()
            if not name.endswith(".onnx") or name in ner_graphs
        }
    prepare(
        directory,
        REPO,
        REVISION,
        {name: {"sha256": digest} for name, digest in files.items()},
    )
    print(f"Verified {len(files)} files in {directory.resolve()}")


if __name__ == "__main__":
    main()
