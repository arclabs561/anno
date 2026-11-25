"""Replay the pinned Maverick CPU model and worker lifecycle tests (Python 3.12)."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys

from huggingface_hub import hf_hub_download


MODEL = "sapienzanlp/maverick-mes-ontonotes"
MODEL_REVISION = "087a90f0380ec1f20ed19732b657f53fd3542c81"
ENCODER = "microsoft/deberta-v3-large"
ENCODER_REVISION = "64a8c8eab3e352a784c658aef62be1662607476f"
ENCODER_FILES = {
    "config.json": "ddec8b81d079d218ce9e54fc0af5d1d5937d6d53b5d42e70c1f251a1cebc830d",
    "pytorch_model.bin": "dd5b5d93e2db101aaf281df0ea1216c07ad73620ff59c5b42dccac4bf2eef5b5",
    "spm.model": "c679fbf93643d19aab7ee10c0b99e460bdbc02fedf34b92b05af343b4af586fd",
    "tokenizer_config.json": "3f3978e0c036f2c2588cac34a6047cbb0af0b0dc1814254e291028529805496d",
}


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
    checkpoint = fetch_verified(hub, MODEL, MODEL_REVISION, "weights.ckpt",
                                "b42053ce171ea904d10d92de5458dce2e8189ef1050e1705e7fced50d2479c1f")
    for name, digest in ENCODER_FILES.items():
        fetch_verified(hub, ENCODER, ENCODER_REVISION, name, digest)
    # The upstream checkpoint initializes its encoder/tokenizer by model ID at
    # main. Resolve that ID to exactly the verified snapshot in our private cache.
    ref = hub / "models--microsoft--deberta-v3-large" / "refs" / "main"
    ref.parent.mkdir(exist_ok=True, parents=True)
    ref.write_text(ENCODER_REVISION, encoding="ascii")
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
