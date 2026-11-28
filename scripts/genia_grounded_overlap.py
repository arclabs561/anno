"""Replay all pinned GENIA annotation layers through `anno viz overlap`."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

from plot_genia_nesting import (
    LEDGER,
    SPLITS,
    character_spans,
    release_files,
    sentences,
    source_identity,
)
from survey_repo_files import file_contents

REFERENCE = LEDGER.with_name("genia-native-exports.json")
TITLE = "GENIA gold span overlap across train, dev and test"


def grounded_sentence(doc_id, tokens, token_spans):
    text, spans = character_spans(tokens, token_spans)
    signals = [
        {
            "id": i,
            "location": {"Text": {"start": start, "end": end}},
            "surface": text[start:end],
            "label": label,
            "confidence": 1.0,
            "modality": "Symbolic",
            "negated": False,
        }
        for i, (start, end, label) in enumerate(spans)
    ]
    return {
        "id": doc_id,
        "text": text,
        "signals": signals,
        "tracks": {},
        "identities": {},
    }


def verify_totals(report, reference):
    splits = reference["splits"].values()
    expected = {
        "schema": "anno.overlap/v1",
        "documents": sum(row["sentences_verified"] for row in splits),
        "documents_with_overlap": sum(
            row["sentences_with_hidden_mentions"] for row in splits
        ),
        "signals": sum(row["jsonl_rows"] for row in splits),
    }
    for name, total in expected.items():
        if report[name] != total:
            raise ValueError(f"GENIA {name} disagrees with the verified native exports")
    if sum(report["coverage"].values()) != report["characters"]:
        raise ValueError("GENIA coverage must partition the reconstructed characters")
    for field in ("signals", "signals_with_overlap"):
        if sum(row[field] for row in report["labels"].values()) != report[field]:
            raise ValueError(f"GENIA label totals disagree with {field}")


def replay(anno, cache):
    verified = json.loads(REFERENCE.read_text(encoding="utf-8"))
    selected = release_files()
    characters = 0
    labels = Counter()
    cache.mkdir(parents=True, exist_ok=True)
    # Keep raw sentences and occurrences local and bound memory to one split.
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stream:
        for split in SPLITS:
            row = selected[split]
            previous = verified["splits"][split]
            data = file_contents(row, cache)
            if (
                hashlib.sha256(data).hexdigest() != row["sha256"]
                or source_identity(row) != previous["source"]
            ):
                raise ValueError(
                    f"GENIA {split} differs from the verified publisher file"
                )
            documents = signals = empty = 0
            for number, (tokens, _, spans) in enumerate(sentences(data), 1):
                doc = grounded_sentence(f"genia-{split}-{number}", tokens, spans)
                stream.write(
                    json.dumps(doc, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                documents += 1
                signals += len(spans)
                empty += not spans
                characters += len(doc["text"])
                labels.update(label for _, _, label in spans)
            if (documents, signals, empty) != (
                previous["sentences_verified"],
                previous["jsonl_rows"],
                previous["sentences_without_gold_mentions"],
            ):
                raise ValueError(
                    f"GENIA {split} denominators disagree with the native exports"
                )
        stream.seek(0)
        aggregate = subprocess.check_output(
            [anno, "viz", "overlap", "--title", TITLE, "--format", "json"], stdin=stream
        )
    report = json.loads(aggregate)
    verify_totals(report, verified)
    if (
        report["characters"] != characters
        or {name: row["signals"] for name, row in report["labels"].items()} != labels
    ):
        raise ValueError("Native overlap lost characters or changed the label counts")
    return aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output", type=Path, help="Write the verified native aggregate here."
    )
    parser.add_argument(
        "--anno", default="anno", help="Anno executable with the overlap view."
    )
    parser.add_argument(
        "--cache-dir", type=Path, help="Reuse the publisher file cache."
    )
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as temp:
        aggregate = replay(args.anno, args.cache_dir or Path(temp))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(aggregate)


if __name__ == "__main__":
    main()
