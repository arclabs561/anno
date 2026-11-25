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
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
from tempfile import TemporaryDirectory

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
PINS = json.loads((ROOT / "scripts/qa/maverick/models.json").read_text())
MODELS = {
    "ontonotes": (PINS["MODEL"], PINS["MODEL_REVISION"], PINS["MODEL_SHA256"]),
    "litbank": ("sapienzanlp/maverick-mes-litbank", "2cd789680357a6c1cddd36c95e022c4fef7e867b",
                "d867ab70063e4eda11d1abfe5773a2073603887e232c9c26263837400eb8b8b7"),
    "preco": ("sapienzanlp/maverick-mes-preco", "7528f57c2491941bb88f5f431519596d54151c28",
              "9e0f14b076dd9824cb4c91ccc77b97fea58a3caa8244e8e5d09f656793b3115e"),
}


def get_cache_dir() -> Path:
    if custom := os.environ.get("ANNO_CACHE_DIR"):
        return Path(custom)
    if platform.system() == "Darwin":
        return Path.home() / "Library/Caches/anno"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "anno"


def download_model(model_name: str, output_dir: Path, cache_dir: Path | None = None) -> Path:
    repo, revision, expected_sha = MODELS[model_name]
    directory = output_dir / "maverick"
    directory.mkdir(parents=True, exist_ok=True)
    hub = {"repo_id": repo, "revision": revision, "cache_dir": cache_dir or directory / ".cache"}
    checkpoint = Path(hf_hub_download(filename="weights.ckpt", **hub))
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected_sha:
        raise ValueError(f"Checkpoint digest mismatch for {repo}@{revision}")

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
        for name, (repo, revision, _) in MODELS.items():
            print(f"{name}: https://huggingface.co/{repo}/tree/{revision}")
        return
    for name in MODELS if args.all else (args.model,):
        print(download_model(name, args.output_dir))


if __name__ == "__main__":
    main()
