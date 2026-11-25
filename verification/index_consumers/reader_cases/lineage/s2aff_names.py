# SPDX-License-Identifier: MIT OR Apache-2.0
"""Apply the frozen retrieval rule to S2AFF's original affiliation strings without training."""

import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request
import zipfile

from evaluate import Record
from ngram_retrieval import Query, compiled_index, decision, lookup

SOURCE = "https://raw.githubusercontent.com/allenai/S2AFF/ac8d6b58f42b253821c26fff8b013913317c5c72/data/gold_affiliation_annotations.csv"
ARCHIVE = "v1.1-2022-06-16-ror-data.zip"
ROR_URL = f"https://zenodo.org/api/records/6657125/files/{ARCHIVE}/content"


def get(cache, name, url, expected, fetch):
    path = cache / name
    if not path.is_file() and fetch:
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as tmp:
            downloaded = Path(tmp) / name
            urllib.request.urlretrieve(url, downloaded)
            with downloaded.open("rb") as f:
                if hashlib.file_digest(f, "sha256").hexdigest() != expected:
                    raise ValueError(f"published artifact hash differs for {name}")
            downloaded.replace(path)
    with path.open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != expected:
            raise ValueError(f"wrong {name}")
    return path


def reader(args, binary):
    archive = get(
        args.cache,
        ARCHIVE,
        ROR_URL,
        "d71fd8d20d753fed8f7b5e08aa2215ad4cda2b2378c7f2a9f977ad17605504a6",
        args.fetch,
    )
    labels = get(
        args.cache,
        "s2aff-gold.csv",
        SOURCE,
        "2385f28c6c1275f0114cf15c33d164672af7a2a6a88d0c03178e04236c3de8c3",
        args.fetch,
    )
    with (
        zipfile.ZipFile(archive) as compressed,
        compressed.open(ARCHIVE.replace(".zip", ".json")) as f,
    ):
        records = json.load(f)
    before = {}
    for row in records:
        aliases = {
            row["name"],
            *row["aliases"],
            *row["acronyms"],
            *(v["label"] for v in row["labels"]),
        }
        before[row["id"].removeprefix("https://ror.org/")] = Record(
            row["name"], tuple(sorted(aliases)), row["status"], ()
        )
    queries = []
    gold = {}
    with labels.open(encoding="utf-8", newline="") as f:
        for row_number, row in enumerate(csv.DictReader(f), start=1):
            if row["split"] != "test":
                continue
            ids = {
                s.removeprefix("https://ror.org/")
                for s in ast.literal_eval(row["labels"])
                if s.startswith("https://ror.org/")
            }
            gold[str(row_number)] = ids
            queries.append(Query("S2AFF", str(row_number), row["original_affiliation"], None))
    counts = Counter()
    with args.ledger.open("w", encoding="utf-8", newline="\n") as out:
        for q, display, aliases, _, rank, block in lookup(binary, before, queries):
            ids = gold[q.source]
            available = {i for i in ids if i in before and before[i].status == "active"}
            cohort = "available" if available else ("unavailable" if ids else "no_ror_label")
            counts[cohort + ".n"] += 1
            counts[cohort + ".target_blocked"] += int(bool(available.intersection(block)))
            counts[cohort + ".target_top5"] += int(
                bool(available.intersection(i for i, _, _ in rank))
            )
            winner, reason = decision(rank)
            accepted = {
                "display": display[0] if len(display) == 1 else None,
                "all_names": aliases[0] if len(aliases) == 1 else None,
                "trigram": winner,
            }
            for arm, pred in accepted.items():
                status = "abstain" if pred is None else ("correct" if pred in ids else "other")
                counts[f"{cohort}.{arm}.{status}"] += 1
            out.write(
                json.dumps(
                    {
                        "source_row": int(q.source),
                        "target_ids": sorted(ids),
                        "available_ids": sorted(available),
                        "answers": accepted,
                        "top5": rank,
                        "reason": reason,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    with args.ledger.open("rb") as f:
        return {"counts": counts, "ledger_sha256": hashlib.file_digest(f, "sha256").hexdigest()}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cache", type=Path)
    p.add_argument("--fetch", action="store_true")
    p.add_argument("--binary", type=Path, help="compiled ngrams.rs")
    p.add_argument("--ledger", type=Path, required=True)
    p.add_argument("--summary", type=Path, required=True)
    args = p.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        binary = compiled_index(tmp, args.binary)
        metrics = reader(args, binary)
    args.summary.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
