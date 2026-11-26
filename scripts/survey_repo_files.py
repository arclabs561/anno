"""Measure pinned publisher files listed in docs/eval/dataset-repo-files.csv.

Run python scripts/survey_repo_files.py --output dataset-repo-files.csv.
The checked-in CSV doubles as the replay manifest. Only aggregate measurements
are written. Use --cache-dir for explicitly retained local raw-file caching.

Rows are token records for columnar formats and sentence objects for JSON.
CoNLL-U counts syntactic words (integer IDs); POS and dependency files have
no NER denominator. GENIA counts the union of its four BIO layers; HIPE counts
its literal coarse NER column; ZELDA counts entity-linked token coverage.
CrossRE counts the union of the inclusive gold entity spans in each sentence.
"""

import argparse
import csv
import hashlib
import json
import re
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "docs/eval/dataset-repo-files.csv"
POLICY = REPO / "crates/anno-eval/src/eval/shared_cache_exclusions.tsv"
BIOES_TAG = re.compile(r"(?:[BIES]-.+|O)")
BIO_TAG = re.compile(r"(?:[BI]-.+|O)")
UPOS = set(
    "ADJ ADP ADV AUX CCONJ DET INTJ NOUN NUM PART PRON PROPN PUNCT SCONJ SYM VERB X".split()
)


def file_contents(row, cache_dir):
    url = row["url"]
    allowed = (
        "https://raw.githubusercontent.com/",
        "https://media.githubusercontent.com/media/",
    )
    if (
        not url.startswith(allowed)
        or f"/{row['repository']}/{row['revision']}/" not in url
    ):
        raise ValueError("Source does not match pinned publisher file")
    cached = cache_dir / hashlib.sha256(url.encode()).hexdigest()
    if cached.is_file():
        return cached.read_bytes()
    for attempt in range(3):
        try:
            with urlopen(url, timeout=60) as response:
                data = response.read()
            break
        except (HTTPError, URLError, TimeoutError) as exc:
            if (
                attempt == 2
                or isinstance(exc, HTTPError)
                and exc.code < 500
                and exc.code != 429
            ):
                raise
            time.sleep(1 + attempt)
    cached.write_bytes(data)
    return data


def crossre_counts(lines):
    records = tokens = tagged = 0
    for line in lines:
        if not line.strip():
            continue
        record = json.loads(line)
        words, spans = record["sentence"], record["ner"]
        if not isinstance(words, list) or not isinstance(spans, list):
            raise ValueError("Expected CrossRE sentence tokens and gold entity spans")
        covered = set()
        for start, end, label in spans:
            if not label or not 0 <= start <= end < len(words):
                raise ValueError("Invalid CrossRE entity label or inclusive token range")
            covered.update(range(start, end + 1))
        records += 1
        tokens += len(words)
        tagged += len(covered)
    if not records or not tokens:
        raise ValueError("Empty CrossRE split")
    return records, tokens, tagged


def measure(data, row):
    metric = row["metric"]
    if metric == "HistNERo_tags":
        records = json.loads(data)
        words = tagged = 0
        for item in records:
            tags = item["ner_tags"]
            if len(tags) != len(item["tokens"]) or any(
                type(tag) != int or tag < 0 for tag in tags
            ):
                raise ValueError("Invalid class-ID sequence")
            words += len(tags)
            tagged += sum(tag != 0 for tag in tags)
        return len(records), words, tagged
    if metric == "CrossRE_entity_spans":
        return crossre_counts(data.splitlines())

    words = tagged = 0
    for line_number, line in enumerate(data.decode("utf-8-sig").splitlines(), start=1):
        if not line.strip() or line.startswith("-DOCSTART-"):
            continue
        if metric in {"CoNLL-U_words", "NorNE_BIO"}:
            if line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) != 10:
                raise ValueError(f"Expected 10 CoNLL-U columns at line {line_number}")
            if not fields[0].isdigit():  # Multiword surface tokens and empty nodes
                continue
            words += 1
            if metric == "CoNLL-U_words":
                continue
            tags = [
                field[5:]
                for field in fields[-1].split("|")
                if field.startswith("name=")
            ]
        elif metric == "HIPE_coarse_literal":
            if line.startswith(("# ", "TOKEN\t")):
                continue
            fields = line.split("\t")
            if len(fields) != 10:
                raise ValueError(f"Expected 10 HIPE columns at line {line_number}")
            words += 1
            tags = fields[1:2]
        else:
            if metric == "BIO_link" and line.startswith("# "):
                continue
            if row["id"] in {"Twitter2015MNER", "TwoMNER"} and line.startswith(
                "IMGID:"
            ):
                continue
            if metric == "CoNLL04_entity_BIO" and line.startswith("#doc "):
                continue
            fields = line.split("\t") if "\t" in line else line.split()
            if len(fields) < 2:
                raise ValueError(f"Missing tag at line {line_number}")
            words += 1
            if metric == "POS_tokens":
                if fields[-1] not in UPOS:
                    raise ValueError(f"Invalid UPOS tag at line {line_number}")
                continue
            if metric == "BIO_any_layer" and len(fields) != 5:
                raise ValueError(f"Expected four GENIA tags at line {line_number}")
            tags = fields[1:] if metric == "BIO_any_layer" else fields[-1:]
            if metric == "CoNLL04_entity_BIO":
                if len(fields) != 5:
                    raise ValueError(
                        f"Expected five joint extraction columns at line {line_number}"
                    )
                tags = fields[2:3]
        tag_pattern = BIOES_TAG if metric == "BIOES" else BIO_TAG
        if not tags or not all(tag_pattern.fullmatch(tag) for tag in tags):
            raise ValueError(f"Unsupported or missing BIO label at line {line_number}")
        tagged += any(tag != "O" for tag in tags)
    if not words:
        raise ValueError("Empty token sequence")
    return words, words, "" if metric in {"POS_tokens", "CoNLL-U_words"} else tagged


def survey(row, cache_dir, local_only):
    if not row["url"]:
        return row
    expected_digest = row["sha256"]
    row.update(rows="", tokens="", tagged="", reason="", sha256="")
    try:
        if row["id"] in local_only:
            raise ValueError("Registry requires authorized local source")
        data = file_contents(row, cache_dir)
        digest = hashlib.sha256(data).hexdigest()
        if expected_digest and expected_digest != digest:
            raise ValueError("File checksum differs from manifest")
        row["sha256"] = digest
        row["rows"], row["tokens"], row["tagged"] = measure(data, row)
        row["status"] = "ok"
    except (OSError, ValueError, KeyError) as exc:
        row.update(status="error", reason=str(exc))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    with args.manifest.open(newline="") as source:
        reader = csv.DictReader(source)
        fields = reader.fieldnames
        rows = list(reader)
    local_only = {
        line.split("\t")[0].removesuffix(".cache")
        for line in POLICY.read_text().splitlines()
        if line.split("\t")[1] == "local"
    }
    with tempfile.TemporaryDirectory() as temp, ThreadPoolExecutor(
        max_workers=8
    ) as workers:
        cache_dir = args.cache_dir or Path(temp)
        cache_dir.mkdir(parents=True, exist_ok=True)
        results = list(
            workers.map(lambda row: survey(row, cache_dir, local_only), rows)
        )
    with args.output.open("w", newline="") as dest:
        writer = csv.DictWriter(dest, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(results)
    print(dict(Counter(row["status"] for row in results)))


if __name__ == "__main__":
    main()
