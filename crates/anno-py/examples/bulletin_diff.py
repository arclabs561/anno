# SPDX-License-Identifier: MIT OR Apache-2.0
"""Find date mentions to review between two UTF-8 editions of a bulletin."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import anno_py


def read_dates(path, extractor):
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    dates = defaultdict(list)
    for entity in extractor.extract(text):
        if entity.label.upper() != "DATE":
            continue
        parts = [text[start:end] for start, end in entity.segments]
        line_start = text.rfind("\n", 0, entity.start) + 1
        line_end = text.find("\n", entity.end)
        dates[tuple(parts)].append({
            "line": text.count("\n", 0, entity.start) + 1,
            "source_line": text[line_start:line_end if line_end >= 0 else len(text)],
            "segments": entity.segments,
            "detector": entity.source,
        })
    source = {"file": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
    return source, dates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    args = parser.parse_args()
    extractor = anno_py.Extractor(backend="offline")
    before_source, before = read_dates(args.before, extractor)
    after_source, after = read_dates(args.after, extractor)
    changes = []
    for parts in sorted(before.keys() | after.keys()):
        if len(before[parts]) != len(after[parts]):
            changes.append({
                "date_pieces": parts, "before": before[parts], "after": after[parts]
            })
    print(json.dumps({
        "sources": [before_source, after_source],
        "date_count_changes": changes,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
