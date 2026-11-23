"""Prepare the revision-pinned public TIMEX test split for a trusted matrix runner."""

from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory


SOURCE = ("https://huggingface.co/datasets/mdg-nlp/timex-recognition-sentence-original/"
          "resolve/904e35a7d2d0ee97c5594735621f94b04f8b7cef/test.jsonl")
DIGEST = "58cd74b68da63ff8acbb70e8acd0302f4edea6151a2f73787aa57799a57e93d3"
SIZE = 257382
NAME = "TimexRecognitionSentenceOriginal.cache"


def verified(path):
    return (path.is_file() and path.stat().st_size == SIZE
            and sha256(path.read_bytes()).hexdigest() == DIGEST)


def prepare(directory):
    directory.mkdir(parents=True, exist_ok=True)
    corpus = directory / NAME
    with TemporaryDirectory(dir=directory) as staging:
        if not verified(corpus):
            download = Path(staging) / NAME
            subprocess.run(["curl", "--fail", "--silent", "--show-error", "--location",
                            "--proto", "=https", "--proto-redir", "=https", "--retry", "2",
                            "--max-time", "30", "--max-filesize", str(SIZE), "-o", str(download),
                            SOURCE], check=True)
            if not verified(download):
                raise ValueError("Unexpected TIMEX test split size or SHA-256")
            download.replace(corpus)

        manifest = directory / "manifest.json"
        if manifest.exists():
            state = json.loads(manifest.read_text())
            entry = state["entries"].get(NAME)
            if entry is not None and (entry.get("source_url") != SOURCE or entry.get("sha256") != DIGEST
                                      or entry.get("resolved_url") not in (None, SOURCE)
                                      or entry.get("sentence_count") != 390):
                # A restored entry for the old Viewer split invalidates the new file.
                # The native loader accepts a raw cache without a manifest entry.
                del state["entries"][NAME]
                updated = Path(staging) / "manifest.json"
                updated.write_text(json.dumps(state))
                updated.replace(manifest)

    print(f"Verified {corpus.name}: {SIZE} bytes, sha256 {DIGEST}")


if __name__ == "__main__":
    prepare(Path(os.environ["ANNO_CACHE_DIR"]) / "datasets")
