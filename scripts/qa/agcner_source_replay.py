"""Compare native AgCNER input, every labelled character span and heuristic scores against the publisher release."""

import argparse
import collections
import hashlib
import io
import json
import os
import re
import subprocess
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / "scripts/qa/direct_pack_sources.json"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode()


def blocks(raw):
    """Read the publisher's one-scalar, delimiter, BIO-tag rows, retaining literal whitespace."""
    current = []
    for number, line in enumerate(io.BytesIO(raw), 1):
        line = line.removesuffix(b"\n").removesuffix(b"\r").decode()
        if not line:
            if current:
                yield current
                current = []
            continue
        match = re.fullmatch(r"(.)[ \t](O|[BI]-[^\s]+)", line, re.DOTALL)
        if not match:
            raise ValueError(f"Invalid publisher BIO row on line {number}")
        current.append(match.groups())
    if current:
        yield current


def source_case(tokens, separator):
    text = separator.join(c for c, _ in tokens)
    spans = []
    offset = 0
    for char, tag in tokens:
        if tag.startswith("B-"):
            spans.append((tag[2:], offset, offset + len(char)))
        elif tag.startswith("I-"):
            if (
                not spans
                or spans[-1][0] != tag[2:]
                or spans[-1][2] + len(separator) != offset
            ):
                raise ValueError(f"Orphan publisher I tag {tag}")
            label, start, _ = spans[-1]
            spans[-1] = (label, start, offset + len(char))
        offset += len(char) + len(separator)
    return text, spans


def gold_keys(spans):
    return {
        str(i): sha(encoded([label, [{"start": start, "end": end}]]))
        for i, (label, start, end) in enumerate(spans)
    }


def get_source(spec, directory, fetch):
    path = directory / spec["sha256"]
    alternative = directory / (spec["files"][0]["split"] + ".bio")
    if path.is_file():
        raw = path.read_bytes()
    elif alternative.is_file():
        raw = alternative.read_bytes()
    elif fetch:
        with urllib.request.urlopen(spec["url"], timeout=120) as response:
            raw = response.read()
        if sha(raw) == spec["sha256"]:
            directory.mkdir(exist_ok=True, parents=True)
            path.write_bytes(raw)
    else:
        raise ValueError(
            f"Supply {alternative.name} or {spec['sha256']} in the source directory or use --fetch"
        )
    if sha(raw) != spec["sha256"]:
        raise ValueError(f"Source SHA-256 differs for {spec['url']}")
    return raw


def run_native(binary, work, cache_bytes):
    cache = work / "datasets"
    cache.mkdir(exist_ok=True, parents=True)
    (cache / "AgCNER.cache").write_bytes(cache_bytes)
    reports = work / "reports.jsonl"
    env = {
        **os.environ,
        "ANNO_CACHE_DIR": str(work.resolve()),
        "ANNO_NO_DOWNLOADS": "1",
        "ANNO_S3_CACHE": "0",
        "NO_COLOR": "1",
    }
    args = [
        binary,
        "dataset",
        "eval",
        "--dataset",
        "AgCNER",
        "--task",
        "ner",
        "--model",
        "heuristic",
        "--cached-only",
        "--reports",
        reports,
    ]
    result = subprocess.run(args, env=env, text=True, capture_output=True, check=True)
    if not re.search(r"\bSentences:\s+[1-9]", result.stdout):
        raise ValueError("CLI scored no sentences")
    return reports


def check_reports(raw, reports, legacy):
    counts = collections.Counter(
        {
            "cases": 0,
            "source_text_exact": 0,
            "gold_spans": 0,
            "gold_spans_exact_source_label_and_offsets": 0,
            "all_case_gold_exact": 0,
            "correct": 0,
            "predicted": 0,
            "source_characters": 0,
            "whitespace_tokens": 0,
            "non_bmp_tokens": 0,
            "artificial_separators": 0,
        }
    )
    with reports.open() as stream:
        for index, (tokens, line) in enumerate(
            zip(blocks(raw), stream, strict=True), 1
        ):
            report = json.loads(line)
            pair = report["pairing"]
            source, spans = source_case(tokens, "")
            keys = gold_keys(spans)
            if (
                pair["document_id"] != f"AgCNER:{index}"
                or report["gold_count"] != len(spans)
                or report["model"] != "heuristic"
            ):
                raise ValueError(
                    f"Native case ID, backend or gold count differs on row {index}"
                )
            if legacy:
                expected_text, expected_spans = source_case(tokens, " ")
                expected_keys = gold_keys(expected_spans)
                counts["artificial_separators"] += len(tokens) - 1
            else:
                expected_text, expected_keys = source, keys
            if (
                pair["source_sha256"] != sha(expected_text.encode())
                or pair["gold_keys"] != expected_keys
            ):
                raise ValueError(
                    f"Native input or labelled span fingerprint differs on row {index}"
                )
            counts.update(
                {
                    "cases": 1,
                    "source_text_exact": int(
                        pair["source_sha256"] == sha(source.encode())
                    ),
                    "gold_spans": len(spans),
                    "gold_spans_exact_source_label_and_offsets": sum(
                        pair["gold_keys"].get(k) == v for k, v in keys.items()
                    ),
                    "all_case_gold_exact": int(keys == pair["gold_keys"]),
                    "correct": report["correct"],
                    "predicted": report["predicted_count"],
                    "source_characters": len(source),
                    "whitespace_tokens": sum(c.isspace() for c, _ in tokens),
                    "non_bmp_tokens": sum(ord(c) > 0xFFFF for c, _ in tokens),
                }
            )
    return dict(sorted(counts.items()))


def study(args):
    recipe = json.loads(RECIPE.read_text())["datasets"]["AgCNER"]
    splits = {}
    for spec in recipe["sources"]:
        split = spec["files"][0]["split"]
        raw = get_source(spec, args.source_dir, args.fetch)
        legacy_cache = io.BytesIO()
        for tokens in blocks(raw):
            legacy_cache.write(
                encoded(
                    {
                        "tokens": [c for c, _ in tokens],
                        "ner_tags": [tag for _, tag in tokens],
                    }
                )
                + b"\n"
            )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = {}
            for name, binary, payload in [
                ("before_jsonl", args.before, legacy_cache.getvalue()),
                ("after_original_bio", args.after, raw),
                ("after_legacy_jsonl", args.after, legacy_cache.getvalue()),
            ]:
                result = run_native(binary, root, payload)
                rows[name] = check_reports(raw, result, name == "before_jsonl")
        if rows["after_original_bio"] != rows["after_legacy_jsonl"]:
            raise ValueError(f"Original and legacy inputs differ on {split}")
        splits[split] = {
            "source_url": spec["url"],
            "source_sha256": spec["sha256"],
            "source_bytes": len(raw),
            **rows,
        }
    return {
        "schema": "anno.agcner-source-coordinates/v1",
        "collection": recipe["source"],
        "scope": "All three Figshare v1 splits. The independent reader hashes original strings and each labelled half-open Unicode scalar range, then checks every native per-case report in release order.",
        "protocol": "Each original is evaluated through the repaired BIO reader. Both executables also evaluate equivalent legacy JSONL rows. Counts are exact entity-span and label matches from the heuristic model.",
        "command": "anno dataset eval --dataset AgCNER --task ner --model heuristic --cached-only --reports <file>",
        "splits": dict(sorted(splits.items())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--before",
        required=True,
        type=Path,
        help="Eval-enabled Anno binary from before the character-coordinate fix",
    )
    parser.add_argument(
        "--after",
        required=True,
        type=Path,
        help="Eval-enabled Anno binary containing the fix",
    )
    parser.add_argument(
        "--source-dir",
        required=True,
        type=Path,
        help="Directory containing train.bio, dev.bio and test.bio or SHA-256-named blobs",
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Fetch missing public publisher files and verify their pinned SHA-256",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = study(args)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
