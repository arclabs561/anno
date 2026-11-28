"""Prepare the eight GLiNER2-PII FP32 ONNX graphs for GLINER2_PII_MODEL_DIR.

Install the converter dependencies in docs/models/gliner2-pii.md before running.
The original Fastino snapshot and the upstream export program are both pinned.
"""

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL = "fastino/gliner2-privacy-filter-PII-multi"
REVISION = "1cb4166094dc58fa8d836429f060d6c95f62b495"
EXPORT_URL = (
    "https://raw.githubusercontent.com/dariofinardi/gliner2-rs/"
    "7626d237d8c58f0fcfadd094e8360e7d5ae9c8ee/"
    "onnx_conversion_scripts/export_gliner2_onnx_fragments_v2.py"
)
EXPORT_SHA256 = "d7f94372f6fbe21c214241cd55a3c627e0cbc477a40c6cff9279147f8be44258"
SOURCE_FILES = {
    "config.json": "164f17362bcf9d114067d3465e7374bfdd79ce6b605acb745de5a49dabb9595c",
    "encoder_config/config.json": "f27dd63cc43a248d2566f0b6ad7a115db353676ce0561dcbca45bac766464c1a",
    "model.safetensors": "0280f6f39f6012da50b6640bad438d9b7e763a1b0102094115d1b710c4dd79b6",
    "tokenizer.json": "f6df10ec83bea993035b2dd7c39345a3d4fcf23421c2adb6cb4ffc1e6d1bc4b5",
    "tokenizer_config.json": "233beed1f1095cccfc7907cde31a8d90a0c6aa4fdfaf6493f8e55fd162e81ae6",
}
GRAPHS = (
    "encoder",
    "token_gather",
    "span_rep",
    "schema_gather",
    "count_pred_argmax",
    "count_lstm_fixed",
    "scorer",
    "classifier",
)


def check_digest(path, expected):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f"Unexpected SHA-256 for {path}: {digest.hexdigest()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", required=True, type=Path, help="New model directory"
    )
    parser.add_argument(
        "--source-dir", type=Path, help="Local copy of the pinned original checkpoint"
    )
    parser.add_argument(
        "--converter", type=Path, help="Local copy of the pinned upstream export script"
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    source = args.source_dir or Path(
        snapshot_download(MODEL, revision=REVISION, allow_patterns=list(SOURCE_FILES))
    )
    for name, expected in SOURCE_FILES.items():
        check_digest(source / name, expected)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as work:
        work = Path(work)
        converter = work / "export.py"
        if args.converter:
            shutil.copyfile(args.converter, converter)
        else:
            urllib.request.urlretrieve(EXPORT_URL, converter)
        check_digest(converter, EXPORT_SHA256)
        raw_graphs = work / "graphs"
        subprocess.run(
            [
                sys.executable,
                str(converter),
                "--model_path",
                str(source.resolve()),
                "--out_dir",
                str(raw_graphs),
            ],
            check=True,
        )
        snapshot = work / "snapshot"
        graphs = snapshot / "fp32_v2"
        graphs.mkdir(parents=True)
        import onnx

        for name in GRAPHS:
            graph = raw_graphs / f"{name}_fp32.onnx"
            onnx.checker.check_model(str(graph))
            graph.rename(graphs / graph.name)
        check_digest(raw_graphs / "tokenizer.json", SOURCE_FILES["tokenizer.json"])
        (raw_graphs / "tokenizer.json").rename(graphs / "tokenizer.json")
        snapshot.rename(args.output)
    print(f"Prepared GLINER2_PII_MODEL_DIR={args.output.resolve()}")


if __name__ == "__main__":
    main()
