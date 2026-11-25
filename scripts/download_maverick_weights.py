#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub>=0.20"]
# ///
"""Prepare local Maverick MES safetensors from pinned CC BY-NC-SA 4.0 checkpoints.

Run `uv run scripts/download_maverick_weights.py --model ontonotes` from an Anno
source checkout with Cargo installed. Use `--list` to see the available models.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "crates/anno/src/backends/coref"
sys.path.insert(0, str(WORKER))
from maverick_worker import sha256_file, verify_encoder

PINS = json.loads((WORKER / "maverick_models.json").read_text())
MODELS = PINS["MODELS"]


def get_cache_dir() -> Path:
    if custom := os.environ.get("ANNO_CACHE_DIR"):
        return Path(custom)
    if platform.system() == "Darwin":
        return Path.home() / "Library/Caches/anno"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "anno"


def download_model(model_name: str, output_dir: Path, cache_dir: Path | None = None) -> Path:
    model = MODELS[model_name]
    repo, revision = model["repo"], model["revision"]
    directory = output_dir / "maverick"
    directory.mkdir(parents=True, exist_ok=True)
    hub = {"repo_id": repo, "revision": revision, "cache_dir": cache_dir or directory / ".cache"}
    checkpoint = Path(hf_hub_download(filename="weights.ckpt", **hub))
    if sha256_file(checkpoint) != model["source_sha256"]:
        raise ValueError(f"Checkpoint digest mismatch for {repo}@{revision}")

    verify_encoder(PINS, cache_dir=cache_dir)
    output = directory / f"maverick-{model_name}.safetensors"
    with TemporaryDirectory(dir=directory) as scratch:
        prepared = Path(scratch) / output.name
        # Candle reads only state_dict tensors, without importing Python objects
        # or executing the training configuration embedded in the archive.
        subprocess.run(
            ["cargo", "run", "--quiet", "--locked", "-p", "anno", "--no-default-features",
             "--features", "candle", "--example", "maverick_export", "--", checkpoint, prepared],
            cwd=ROOT, check=True,
        )
        if sha256_file(prepared) != model["prepared_sha256"]:
            raise ValueError(f"Prepared checkpoint digest mismatch for {repo}@{revision}")
        config = Path(scratch) / "config.yaml"
        shutil.copyfile(hf_hub_download(filename="config.yaml", **hub), config)
        config.replace(directory / f"maverick-{model_name}-config.yaml")
        prepared.replace(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--model", choices=MODELS)
    choice.add_argument("--all", action="store_true")
    choice.add_argument("--list", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=get_cache_dir() / "models")
    args = parser.parse_args()
    if args.list:
        for name, model in MODELS.items():
            print(f"{name}: https://huggingface.co/{model['repo']}/tree/{model['revision']}")
        return
    for name in MODELS if args.all else (args.model,):
        print(download_model(name, args.output_dir))


if __name__ == "__main__":
    main()
