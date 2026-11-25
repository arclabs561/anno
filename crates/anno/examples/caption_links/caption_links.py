# SPDX-License-Identifier: MIT OR Apache-2.0
"""Attach FCoref's word-aligned cluster suggestions to caller-supplied caption cues."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


def transcript(cues):
    if not isinstance(cues, list) or not cues:
        raise ValueError("expected a nonempty array of caption cues")
    seen = set()
    positions = []
    offset = 0
    for cue in cues:
        cue_id = cue["id"]
        if not isinstance(cue_id, str) or not cue_id.strip() or cue_id in seen:
            raise ValueError("cue IDs must be unique nonempty strings")
        seen.add(cue_id)
        if not isinstance(cue["text"], str):
            raise ValueError(f"{cue_id}: expected text")
        start, end = cue["start_ms"], cue["end_ms"]
        if type(start) is not int or type(end) is not int or not 0 <= start < end:
            raise ValueError(f"{cue_id}: expected a positive integer-millisecond interval")
        stop = offset + len(cue["text"])
        positions.append((offset, stop, cue))
        offset = stop + 1  # one newline between cues
    return "\n".join(cue["text"] for cue in cues), positions


def map_clusters(text, positions, receipt):
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    result, = receipt["results"]
    if result["id"] != "captions" or result["source_sha256"] != digest:
        raise ValueError("model receipt does not match the supplied transcript")
    suggestions = []
    for cluster in result["clusters"]:
        members = []
        for start, end in cluster:
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
                raise ValueError("invalid model character range")
            covered = []
            for left, right, cue in positions:
                a, b = max(start, left), min(end, right)
                if a < b:
                    covered.append({
                        "id": cue["id"], "start_ms": cue["start_ms"], "end_ms": cue["end_ms"],
                        "chars": [a - left, b - left],
                    })
            members.append({"quote": text[start:end], "chars": [start, end], "cues": covered})
        suggestions.append(members)
    return {"text_sha256": digest, "suggested_clusters": suggestions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path, help="built fcoref_word_spans example")
    parser.add_argument("model", type=Path, help="local FCoref export")
    parser.add_argument("captions", type=Path, help="UTF-8 JSON array of caption cues")
    args = parser.parse_args()
    try:
        # Read bytes to preserve any embedded line endings in cue texts.
        cues = json.loads(args.captions.read_bytes().decode("utf-8"))
        text, positions = transcript(cues)
        with tempfile.TemporaryDirectory() as directory:
            panel = Path(directory) / "transcript.json"
            panel.write_text(json.dumps([{"id": "captions", "text": text}]), encoding="utf-8")
            raw = subprocess.check_output(
                [args.binary.resolve(), args.model.resolve(), panel],
                env={**os.environ, "ANNO_NO_DOWNLOADS": "1"},
            )
        out = map_clusters(text, positions, json.loads(raw))
    except (OSError, ValueError, TypeError, KeyError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"caption linking failed: {exc}\n")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
