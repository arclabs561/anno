"""Replay representative complete native splits from restored cache archives, with downloads disabled."""

import argparse
import collections
import hashlib
import json
import os
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

# Use entity scoring only where the cache contains named-entity gold.
STUDIES = {
    "AgCNER": ("test", "ner"),
    "GAP": ("test", "coref"),
    "HIPE2022": ("hipe2020-de-test", "ner"),
    "HistNERo": ("dev", "ner"),
    "MasakhaNER": ("yor-test", "ner"),
    "NorNE": ("nno-test", "ner"),
    "SECFilingsNER": ("test", "ner"),
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def compare(anno, archive, item, task, work):
    cache = work / "datasets"
    cache.mkdir(exist_ok=True)
    data = archive.read(item["path"])
    source = archive.read(item["original_path"])
    if sha(data) != item["cache_sha256"] or sha(source) != item["sha256"]:
        raise ValueError(f"Wrong native or original bytes for {item['id']}")
    filename = item["id"] + ".cache"
    manifest = archive.read(str(Path(item["path"]).parent / "manifest.json"))
    if json.loads(manifest)["entries"][filename]["sha256"] != sha(data):
        raise ValueError(f"Wrong native manifest for {filename}")
    (cache / filename).write_bytes(data)
    (cache / "manifest.json").write_bytes(manifest)
    args = [
        "dataset",
        "eval",
        "--dataset",
        item["id"],
        "--task",
        task,
        "--model",
        "heuristic",
        "--cached-only",
    ]
    if task == "ner":
        args.extend(["--reports", "reports.jsonl"])
    if item["id"] == "HistNERo":
        # This historical replay measures the literal source-label baseline.
        help_text = subprocess.check_output([anno, "dataset", "eval", "--help"], text=True)
        if "--ner-scoring" in help_text:
            args.extend(["--ner-scoring", "source"])
    env = {
        **os.environ,
        "ANNO_CACHE_DIR": str(work),
        "ANNO_S3_CACHE": "0",
        "ANNO_NO_DOWNLOADS": "1",
        "NO_COLOR": "1",
    }
    proc = subprocess.run(
        [anno, *args], env=env, cwd=work, text=True, capture_output=True, check=True
    )
    output = proc.stdout + proc.stderr
    result = {
        "dataset": item["id"],
        "variant": item["variant"],
        "task": task,
        "publisher_url": item["url"],
        "publisher_sha256": item["sha256"],
        "cache_path": item["path"],
        "cache_sha256": item["cache_sha256"],
        "argv": ["anno", *args],
        "loaded": item["counts"],
    }
    if task == "ner":
        with (work / "reports.jsonl").open() as report:
            totals = collections.Counter()
            ids = set()
            for line in report:
                row = json.loads(line)
                key = row["pairing"]["document_id"]
                if key in ids:
                    raise ValueError(f"Repeated paired ID {key}")
                ids.add(key)
                totals.update(
                    {k: row[k] for k in ("correct", "gold_count", "predicted_count")}
                )
        totals["observations"] = len(ids)
        if (
            totals["observations"] != item["counts"]["sentences"]
            or totals["gold_count"] != item["counts"]["native_labels"]
        ):
            raise ValueError(
                f"Report totals differ from the restored count for {item['id']}"
            )
        result["evaluation"] = dict(sorted(totals.items()))
        if item["id"] == "HistNERo":
            labels = {}
            for name, gold, pred, correct in re.findall(
                r"^\s*(PERS|PER)\s+F1=.*?\[gold=(\d+) pred=(\d+) correct=(\d+)\]",
                output,
                re.MULTILINE,
            ):
                labels[name] = {
                    "gold": int(gold),
                    "predicted": int(pred),
                    "correct": int(correct),
                }
            if labels.keys() != {"PERS", "PER"}:
                raise ValueError("Missing HistNERo gold/predicted label breakdown")
            source_rows = json.loads(source)
            native_rows = json.loads(data)["rows"]
            if [r["row"] for r in native_rows] != source_rows:
                raise ValueError("HistNERo cache lost an original source row or field")
            count = re.search(
                r"warning: (\d+) validation warnings in gold annotations", output
            )
            result["source_label_comparison"] = labels
            result["zero_width"] = {
                "empty_tokens": sum(
                    not token for row in source_rows for token in row["tokens"]
                ),
                "empty_non_o_tokens": sum(
                    not token and label != 0
                    for row in source_rows
                    for token, label in zip(row["tokens"], row["ner_tags"], strict=True)
                ),
                "span_validation_warnings": int(count[1]) if count else 0,
                "interpretation": "Two B-PROD empty tokens in validation become zero-length gold spans. The scorer warns and includes them in the gold denominator.",
            }
    else:
        docs = re.search(r"Documents:\s*(\d+)", output)
        score = re.search(r"CoNLL F1:\s*([0-9.]+)", output)
        if not docs or not score or int(docs[1]) != item["counts"]["sentences"]:
            raise ValueError("GAP coreference did not score every original case")
        result["evaluation"] = {
            "documents": int(docs[1]),
            "conll_f1_cli_rounded": float(score[1]),
            "protocol": "Native full-pipeline heuristic coreference on the supplied texts, with exact character spans.",
        }
    return result


def run(args):
    archives = {}
    results = []
    hipe = {}
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        for artifact in sorted(args.archives.glob("anno-native-cache-*.zip")):
            with zipfile.ZipFile(artifact) as archive:
                catalog = json.loads(archive.read("sources.json"))
                archives[artifact.name] = sha(artifact.read_bytes())
                for item in catalog["inputs"]:
                    if item["id"] == "HIPE2022":
                        rows = archive.read(item["original_path"]).decode().splitlines()
                        hipe[item["variant"]] = sum(
                            line.startswith("#")
                            and not line.startswith("# ")
                            and "\t" in line
                            for line in rows
                        )
                    if item["id"] in STUDIES:
                        variant, task = STUDIES[item["id"]]
                        if item["variant"] == variant:
                            results.append(
                                compare(args.anno.resolve(), archive, item, task, work)
                            )
    if {d["dataset"] for d in results} != STUDIES.keys():
        raise ValueError("One or more required archives or profiles are missing")
    return {
        "schema": "anno.offline-native-replay/v1",
        "anno_version": subprocess.check_output(
            [args.anno, "--version"], text=True
        ).strip(),
        "environment": {
            "ANNO_CACHE_DIR": "temporary restored cache",
            "ANNO_S3_CACHE": "0",
            "ANNO_NO_DOWNLOADS": "1",
        },
        "archive_sha256": archives,
        "runs": sorted(results, key=lambda r: r["dataset"]),
        "hipe_literal_hash_audit": {
            "sources": len(hipe),
            "affected_sources": sum(bool(n) for n in hipe.values()),
            "hash_prefixed_word_rows": sum(hipe.values()),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", type=Path)
    parser.add_argument("--anno", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(args)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        f"Verified {len(result['runs'])} complete native evaluations. Saved {args.output}"
    )


if __name__ == "__main__":
    main()
