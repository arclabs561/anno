"""Replay the pinned Maverick CPU model and worker lifecycle tests (Python 3.12)."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from download_maverick_weights import PINS, download_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="working directory for the dedicated QA cache")
    root = parser.parse_args().directory.resolve() / "maverick-qa"
    hub = root / "hub"
    checkpoint = download_model("ontonotes", root, cache_dir=hub)
    with checkpoint.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != PINS["MODELS"]["ontonotes"]["prepared_sha256"]:
            raise ValueError("prepared tensor digest mismatch")
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
        "TORCH_FORCE_WEIGHTS_ONLY_LOAD": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    subprocess.run([sys.executable, Path(__file__).with_name("test_real_tokenizer.py")], env=env, check=True)
    subprocess.run(
        ["cargo", "test", "--locked", "-p", "anno", "--no-default-features", "--lib",
         "backends::coref::maverick::tests", "--", "--include-ignored", "--test-threads=1"],
        env=env, cwd=Path(__file__).resolve().parents[3], check=True,
    )


if __name__ == "__main__":
    main()
