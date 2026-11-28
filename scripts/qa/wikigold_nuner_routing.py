#!/usr/bin/env python3
"""Replay NuNER routing on the WikiGold release whose bytes CI matched to S3."""

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path

import nuner_fallback_routing as routing
import replay_wikigold


ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "docs" / "eval"
PROJECTION = {label: label for label in sorted(replay_wikigold.LABELS)}
# BertNEROnnx maps its CoNLL MISC class to EntityType::custom("misc", ...).
PREDICTION_ALIASES = {"misc": "MISC"}
STUDY = EVAL / "nuner-fallback-routing.json"
# DatasetLoader::to_test_cases() in the authenticated cold-S3 WikiGold receipt.
S3_NATIVE_CASES_SHA256 = (
    "25ebf59f7db0e89410130ae9e99f1293f24548742e62185c605699eb109d916d"
)


def parse_release_with_documents(text):
    """Keep sentence occurrences in order, treating document markers as metadata."""
    cases = []
    document_ids = []
    documents = markers = tokens = tagged = 0
    new_document = True
    for block in text.strip().split("\n\n"):
        if not block.strip():
            continue
        if block.strip() == "-DOCSTART- O":
            markers += 1
            new_document = True
            continue
        documents += int(new_document)
        new_document = False
        sentence, gold = replay_wikigold.annotated_sentence(block)
        cases.append((sentence, gold))
        document_ids.append(documents)
        rows = block.splitlines()
        tokens += len(rows)
        tagged += sum(row.split()[-1] != "O" for row in rows)
    summary = {
        "documents_with_sentences": documents,
        "document_markers_excluded": markers,
        "sentence_occurrences": len(cases),
        "distinct_sentence_texts": len({sentence for sentence, _ in cases}),
        "tokens": tokens,
        "tagged_tokens": tagged,
        "gold_spans": sum(len(gold) for _, gold in cases),
        "gold_empty_sentences": sum(not gold for _, gold in cases),
    }

    return cases, summary, document_ids


def parse_release(text):
    cases, summary, _ = parse_release_with_documents(text)
    return cases, summary


def load_release_with_documents(path):
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != replay_wikigold.WIKIGOLD:
        raise ValueError("WikiGold input differs from the pinned corpus")
    with (EVAL / "s3-publisher-file-matches.csv").open(
        encoding="utf-8", newline=""
    ) as f:
        matches = [
            row
            for row in csv.DictReader(f)
            if row["cache_id"] == row["publisher_id"] == "WikiGold"
            and row["sha256"] == digest
            and row["ledger"] == "dataset-direct-files.csv"
            and row["split_or_subset"] == "complete linked release"
        ]
    cases, summary, document_ids = parse_release_with_documents(raw.decode("utf-8"))
    expected = (len(raw), len(cases), summary["tokens"], summary["tagged_tokens"])
    if not matches or not all(
        tuple(int(r[k]) for k in ("payload_bytes", "records", "tokens", "tagged"))
        == expected
        for r in matches
    ):
        raise ValueError("WikiGold release counts disagree with the preserved S3 match")
    # Serialize the same (text, Vec<GoldEntity>) as the native cache probe.
    native_cases = [
        [
            text,
            [
                {
                    "text": text[a:b],
                    "entity_type": label,
                    "original_label": label,
                    "start": a,
                    "end": b,
                }
                for a, b, label in gold
            ],
        ]
        for text, gold in cases
    ]
    gold_digest = routing.text_sha(
        json.dumps(native_cases, ensure_ascii=False, separators=(",", ":"))
    )
    if gold_digest != S3_NATIVE_CASES_SHA256:
        raise ValueError(
            "Parsed WikiGold cases disagree with the native cold-S3 gold spans"
        )
    summary.update(
        sha256=digest,
        bytes=len(raw),
        publisher=matches[0]["source"],
        native_gold_cases_sha256=gold_digest,
        cold_s3_workflow_run=45834640325,
        cold_s3_artifact=15123052847,
        cold_s3_receipt="wikigold-cold.json",
    )
    return cases, summary, document_ids


def load_release(path):
    cases, summary, _ = load_release_with_documents(path)
    return cases, summary


def replay(args, wnut):
    routing.verify_checkpoints(args.checkpoint, args.bert)
    cases, source = load_release(args.data)
    gates = routing.run_worker(args.stock, "--probe", cases)
    arms = {}
    for arm, bert in (("without_bert", None), ("with_bert", args.bert)):
        by_label = {label: Counter() for label in PROJECTION}
        skipped_calls = Counter()
        result = routing.measure(
            "WikiGold",
            cases,
            None,
            args,
            gates,
            bert,
            projection=PROJECTION,
            prediction_aliases=PREDICTION_ALIASES,
            by_label=by_label,
            skipped_calls=skipped_calls,
            repeat_skipped_timing=True,
        )
        result["skipped_sentence_outcomes"] = dict(skipped_calls)
        result["skipped_by_label"] = {
            label: {
                metric: counts[metric]
                for metric in (
                    "gold",
                    "exact_gained",
                    "exact_lost",
                    "unmatched_added",
                    "unmatched_removed",
                )
            }
            for label, counts in by_label.items()
        }
        arms[arm] = result
    workers = {
        name: routing.file_sha(getattr(args, name))
        for name in ("stock", "candidate", "always")
    }
    return {
        "source": source,
        "checkpoint": routing.CHECKPOINT,
        "checkpoint_sha256": routing.FILES,
        "bert_checkpoint": routing.BERT_CHECKPOINT,
        "bert_sha256": routing.BERT_FILES,
        "worker_sha256": workers,
        "identical_worker_bytes_to_wnut": workers == wnut["worker_sha256"],
        "reference_wnut_baseline_revision": wnut["base_revision"],
        "reference_wnut_receipt_sha256": routing.file_sha(STUDY),
        "patch_sha256": {
            name: routing.file_sha(Path(__file__).parent / "fixtures" / name)
            for name in ("nuner-fallback-routing.patch", "nuner-always-routing.patch")
        },
        "nuner_threshold": 0.9,
        "output_threshold": 0.5,
        "projection": PROJECTION,
        "prediction_aliases": PREDICTION_ALIASES,
        "launcher_cpu_affinity": sorted(os.sched_getaffinity(0)),
        "gold_by_source_label": routing.gold_label_counts(cases, gates),
        "arms": arms,
    }


def check_measurement_protocol(wikigold, wnut):
    for key in (
        "checkpoint_sha256",
        "bert_sha256",
        "patch_sha256",
        "nuner_threshold",
        "output_threshold",
    ):
        if wikigold[key] != wnut[key]:
            raise ValueError(f"The two routing studies differ on {key}")


def tradeoff(wikigold, wnut):
    check_measurement_protocol(wikigold, wnut)
    series = []
    for dataset, arms in (
        ("WNUT-17 test", wnut["splits"]["Test"]),
        ("WikiGold", wikigold["arms"]),
    ):
        for arm, bert in (
            ("without_bert", "BERT absent"),
            ("with_bert", "BERT loaded"),
        ):
            r = arms[arm]
            c = r["skipped_subset"]
            series.append(
                {
                    "label": f"{dataset} · {bert}",
                    "observations": r["stock_skipped"],
                    "gold": c["gold"],
                    "points": [
                        {
                            "label": label,
                            "correct": c[f"{key}_exact"],
                            "unmatched": c[f"{key}_in_schema_unmatched"],
                        }
                        for key, label in (
                            ("stock", "Stock"),
                            ("candidate", "Run NuNER"),
                        )
                    ],
                }
            )
    return {
        "schema": "anno.tradeoff/v1",
        "title": "Running NuNER on stock-skipped sentences",
        "unit": "sentences",
        "series": series,
    }


def paired(wikigold, wnut):
    check_measurement_protocol(wikigold, wnut)
    rows = []
    for corpus, role, arms in (
        ("WikiGold", "Complete release · 4 scored WikiGold types", wikigold["arms"]),
        ("WNUT-17 test", "Official test · 5 mapped WNUT types", wnut["splits"]["Test"]),
        (
            "WNUT-17 development",
            "Development · 5 mapped WNUT types",
            wnut["splits"]["Development"],
        ),
    ):
        for key, label in (
            ("without_bert", "BERT absent"),
            ("with_bert", "BERT loaded"),
        ):
            arm = arms[key]
            counts = arm["skipped_subset"]
            n = arm["stock_skipped"]
            row = {
                "label": f"{corpus} · {label}",
                "role": role,
                "observations": n,
                "gained": counts["exact_gained"],
                "lost": counts["exact_lost"],
            }
            for which, prefix, calls in (
                ("baseline", "stock", 0),
                ("candidate", "candidate", n),
            ):
                timing = arm["warm_skipped_subset_latency"][prefix]
                if timing["calls"] != n:
                    raise ValueError(
                        f"Warm stack timings omit skipped sentences in {row['label']}"
                    )
                row[which] = {
                    "correct": counts[f"{prefix}_exact"],
                    "unmatched": counts[f"{prefix}_in_schema_unmatched"],
                    "calls": calls,
                    "median_ms": timing["median_ms"],
                }
            if (
                row["candidate"]["correct"] - row["baseline"]["correct"]
                != row["gained"] - row["lost"]
            ):
                raise ValueError(
                    "Paired exact gains and losses disagree with the two arms"
                )
            if (
                row["candidate"]["unmatched"] - row["baseline"]["unmatched"]
                != counts["in_schema_unmatched_added"]
                - counts["in_schema_unmatched_removed"]
            ):
                raise ValueError("Paired unmatched changes disagree with the two arms")
            rows.append(row)
    return {
        "schema": "anno.paired/v1",
        "title": "What does a skipped NuNER call buy?",
        "unit": "stock-skipped sentences",
        "baseline": "Skip NuNER",
        "candidate": "Run NuNER",
        "rows": rows,
    }


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_subparsers(dest="action", required=True)
    run = modes.add_parser(
        "replay", help="Verify the corpus and run both native stack configurations"
    )
    for name in (
        "stock",
        "candidate",
        "always",
        "data",
        "checkpoint",
        "bert",
        "output",
    ):
        run.add_argument("--" + name, type=Path, required=True)
    run.add_argument(
        "--chart", type=Path, help="Also write the native tradeoff chart input"
    )
    run.add_argument(
        "--paired", type=Path, help="Also write the paired outcomes and costs"
    )
    chart = modes.add_parser(
        "chart", help="Build the chart from the two recorded studies"
    )
    chart.add_argument("receipt", type=Path)
    chart.add_argument("output", type=Path)
    chart.add_argument("--view", choices=("tradeoff", "paired"), default="tradeoff")
    args = p.parse_args()
    wnut = json.loads(STUDY.read_text(encoding="utf-8"))
    if args.action == "chart":
        receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
        if receipt["reference_wnut_receipt_sha256"] != routing.file_sha(STUDY):
            p.error(
                "The WNUT reference file differs from the receipt used in the WikiGold replay"
            )
        write_json(
            args.output,
            paired(receipt, wnut) if args.view == "paired" else tradeoff(receipt, wnut),
        )
        return
    result = replay(args, wnut)
    write_json(args.output, result)
    if args.chart:
        write_json(args.chart, tradeoff(result, wnut))
    if args.paired:
        write_json(args.paired, paired(result, wnut))


if __name__ == "__main__":
    main()
