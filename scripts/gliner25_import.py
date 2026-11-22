# /// script
# requires-python = ">=3.10"
# dependencies = ["gliner2[local]==2.2.0"]
# ///
"""Extract one UTF-8 document as JSONL for `anno dataset import --format jsonl`.

    uv run scripts/gliner25_import.py letter.txt person organization > mentions.jsonl
    anno dataset import --input mentions.jsonl --format jsonl --output annotations.jsonl

The first run downloads Fastino's pinned GLiNER2.5 Multi checkpoint. Extraction
runs locally through the upstream Python boundary decoder. Each input file is a
separate source, with half-open Unicode character offsets into its original text.
"""

import argparse
import json
from pathlib import Path

from gliner2 import AutoExtractor
from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path)
    parser.add_argument("labels", nargs="+")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")

    # Preserve CRLF so offsets also index the input on disk.
    text = args.source.read_bytes().decode("utf-8")
    if not text.strip():
        return
    checkpoint = snapshot_download(
        "fastino/gliner2.5-multi-v1",
        revision="7b005274e82da37dcf1f5fd9d6a2ef413c6a004c",
        allow_patterns=["*.json", "model.safetensors"],
    )
    model = AutoExtractor.from_pretrained(checkpoint)
    result = model.extract_entities_long(
        text,
        args.labels,
        threshold=args.threshold,
        include_spans=True,
        include_confidence=True,
    )
    rows = []
    for label, mentions in result["entities"].items():
        for mention in mentions:
            start, end = mention["start"], mention["end"]
            if not (0 <= start < end <= len(text)) or text[start:end] != mention["text"]:
                raise ValueError(f"GLiNER2.5 returned an inconsistent {label} span [{start}, {end})")
            rows.append({
                "text": mention["text"],
                "type": label,
                "start": start,
                "end": end,
                "confidence": mention["confidence"],
                "source": str(args.source),
            })
    for row in sorted(rows, key=lambda row: (row["start"], row["end"], row["type"])):
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
