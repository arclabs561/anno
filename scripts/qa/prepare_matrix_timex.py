"""Stage an independently authorized local copy of the historical TIMEX split."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory


DIGEST = "58cd74b68da63ff8acbb70e8acd0302f4edea6151a2f73787aa57799a57e93d3"
SIZE = 257382
NAME = "TimexRecognitionSentenceOriginal.cache"


def prepare(directory, source):
    with source.open("rb") as stream:
        data = stream.read(SIZE + 1)
    if len(data) != SIZE or sha256(data).hexdigest() != DIGEST:
        raise ValueError("Unexpected TIMEX test split size or SHA-256")

    directory.mkdir(parents=True, exist_ok=True)
    corpus = directory / NAME
    manifest = directory / "manifest.json"
    with TemporaryDirectory(dir=directory) as staging:
        staged = Path(staging)
        if manifest.exists():
            state = json.loads(manifest.read_bytes())
            if NAME in state["entries"]:
                # A remote-source record from an older cache would misdescribe this local import.
                del state["entries"][NAME]
                (staged / "manifest.json").write_text(json.dumps(state))

        if not corpus.is_file() or corpus.stat().st_size != SIZE or corpus.read_bytes() != data:
            (staged / NAME).write_bytes(data)
            (staged / NAME).replace(corpus)
        if (staged / "manifest.json").is_file():
            (staged / "manifest.json").replace(manifest)

    print(f"Verified local {corpus.name}: {SIZE} bytes, sha256 {DIGEST}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="local JSONL covered by your source permissions")
    options = parser.parse_args()
    prepare(Path(os.environ["ANNO_CACHE_DIR"]) / "datasets", options.source)
