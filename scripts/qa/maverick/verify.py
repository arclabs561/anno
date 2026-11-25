"""Replay the pinned Maverick CPU model and worker lifecycle tests (Python 3.12)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from huggingface_hub import hf_hub_download


PINS = json.loads(Path(__file__).with_name("models.json").read_text())


def fetch_verified(cache, repo, revision, name, expected):
    path = Path(hf_hub_download(repo, name, revision=revision, cache_dir=cache))
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"{repo}@{revision}/{name}: expected {expected}, found {actual}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="working directory for the dedicated QA cache")
    root = parser.parse_args().directory.resolve() / "maverick-qa"
    hub = root / "hub"
    checkpoint = fetch_verified(hub, PINS["MODEL"], PINS["MODEL_REVISION"], "weights.ckpt",
                                PINS["MODEL_SHA256"])
    for name, digest in PINS["ENCODER_FILES"].items():
        fetch_verified(hub, PINS["ENCODER"], PINS["ENCODER_REVISION"], name, digest)
    # The upstream checkpoint initializes its encoder/tokenizer by model ID at
    # main. Resolve that ID to exactly the verified snapshot in our private cache.
    ref = hub / "models--microsoft--deberta-v3-large" / "refs" / "main"
    ref.parent.mkdir(exist_ok=True, parents=True)
    ref.write_text(PINS["ENCODER_REVISION"], encoding="ascii")
    env = {
        **os.environ,
        "MAVERICK_PYTHON": sys.executable,
        "MAVERICK_MODEL_PATH": str(checkpoint),
        "ANNO_NO_DOWNLOADS": "1",
        "HF_HOME": str(root),
        "HF_HUB_CACHE": str(hub),
        "TRANSFORMERS_CACHE": str(hub),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    subprocess.run(
        ["cargo", "test", "--locked", "-p", "anno", "--no-default-features", "--lib",
         "backends::coref::maverick::tests", "--", "--include-ignored", "--test-threads=1"],
        env=env, cwd=Path(__file__).resolve().parents[3], check=True,
    )


if __name__ == "__main__":
    main()
