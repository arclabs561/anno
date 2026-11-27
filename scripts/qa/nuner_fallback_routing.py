#!/usr/bin/env python3
"""Compare the cost and accuracy of running a skipped NuNER layer with and without BERT."""
import argparse
import csv
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

from wnut_native_replay import (
    GOLD,
    MANIFEST,
    PROJECTION,
    REVISION,
    load_file,
    sentence_cases,
    surface,
)

CHECKPOINT = "deepanwa/NuNerZero_onnx@40349bb15d985ced384bed5c35115fd70069f4b9"
FILES = {
    "model.onnx": "65c04518901b4484e32c7aa80edc98ab025067e44569141bee94cc498f8096f5",
    "tokenizer.json": "44ce8be1f3118c673c7408bf903766c38bf1ea507e9b8f1e3d85b253a9a12ad1",
    "gliner_config.json": "84c03b5361224deac8f8af3ba99deab2e0fe24ae7ae953669b6c9e70460a2659",
}
BERT_CHECKPOINT = (
    "protectai/bert-base-NER-onnx@84d28ab2bc28401b4353b26033eb043c8c8a0739"
)
BERT_FILES = {
    "model.onnx": "e2165e300f6e0016c1f295bcdee36e084371e1f4f3564cd7e47cb833b58a75b7",
    "tokenizer.json": "cb26b43c98e8266ae3e99c2a583cf8315d73b33a17e6b20b4df7ff1f22392d34",
    "config.json": "e3cc09261a1c73bda961e030563585b2be563bef339a78e11c802c3377b98d66",
}
LABELS = set(PROJECTION.values())


def file_sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_worker(binary, checkpoint, cases, bert=None):
    inputs = "".join(json.dumps(text, ensure_ascii=False) + "\n" for text, _ in cases)
    command = [str(binary.resolve()), str(checkpoint)]
    if bert is not None:
        command.append(str(bert))
    proc = subprocess.run(
        command,
        input=inputs,
        stdout=subprocess.PIPE,
        text=True,
        check=True,
        env={**os.environ, "ANNO_NO_DOWNLOADS": "1", "HF_HUB_OFFLINE": "1"},
    )
    rows = [json.loads(line) for line in proc.stdout.splitlines()]
    for (text, _), row in zip(cases, rows, strict=True):
        if row["input_sha256"] != text_sha(text):
            raise ValueError("Native worker returned a different input or order")
        if len(row["spans"]) != len({tuple(s) for s in row["spans"]}):
            raise ValueError("Duplicate final typed span")
    return rows


def add_case(totals, text, source_gold, old_rows, new_rows, training_names):
    gold = {(a, b, PROJECTION[t]) for a, b, t in source_gold}
    if len(gold) != len(source_gold):
        raise ValueError("Projected gold collision")
    unseen = {v for v in gold if text[v[0] : v[1]].casefold() not in training_names}
    old, new = ({tuple(s) for s in rows} for rows in (old_rows, new_rows))
    old_fp, new_fp = ({s for s in arm - gold if s[2] in LABELS} for arm in (old, new))
    gained, lost = (new - old) & gold, (old - new) & gold
    totals.update(
        {
            "gold": len(gold),
            "unfamiliar_gold": len(unseen),
            "gold_empty_sentences": int(not gold),
            "stock_exact": len(old & gold),
            "candidate_exact": len(new & gold),
            "exact_gained": len(gained),
            "exact_lost": len(lost),
            "unfamiliar_gained": len(gained & unseen),
            "unfamiliar_lost": len(lost & unseen),
            "stock_in_schema_unmatched": len(old_fp),
            "candidate_in_schema_unmatched": len(new_fp),
            "in_schema_unmatched_added": len(new_fp - old_fp),
            "in_schema_unmatched_removed": len(old_fp - new_fp),
            "stock_outside_schema": sum(s[2] not in LABELS for s in old),
            "candidate_outside_schema": sum(s[2] not in LABELS for s in new),
            "sentences_with_changed_output": int(new != old),
        }
    )


def timing(rows):
    seconds = sorted(r["seconds"] for r in rows)
    return {
        "calls": len(seconds),
        "total_seconds": round(sum(seconds), 3),
        "median_ms": round(1000 * statistics.median(seconds), 2),
        "p95_ms": round(1000 * seconds[math.ceil(0.95 * len(seconds)) - 1], 2),
    }


def gold_label_counts(cases, gates):
    """Keep the source label inventory visible, including types NuNER never prompts."""
    all_labels, skipped_labels = Counter(), Counter()
    for (_, gold), route in zip(cases, gates, strict=True):
        labels = [label for _, _, label in gold]
        all_labels.update(labels)
        if not route["ran"]:
            skipped_labels.update(labels)
    return {"full_split": dict(all_labels), "stock_skipped": dict(skipped_labels)}


def measure(split, cases, names, args, gates, bert):
    skipped = [i for i, row in enumerate(gates) if not row["ran"]]
    routed = [i for i, row in enumerate(gates) if row["ran"]]
    controls = sorted(routed, key=lambda i: (text_sha(cases[i][0]), i))[:8]
    selected = sorted(skipped + controls)
    inputs = [cases[i] for i in selected]
    topology = "BERT present" if bert else "no BERT"
    print(
        f"{split}, {topology}: scoring {len(cases)} stock sentences and {len(inputs)} paired candidate sentences",
        file=sys.stderr,
        flush=True,
    )
    stock = run_worker(args.stock, args.checkpoint, cases, bert)
    if any(row["ran"] != gate["ran"] for row, gate in zip(stock, gates, strict=True)):
        raise ValueError("The loaded stock layers disagreed with the routing probe")
    candidate = run_worker(
        args.always if bert else args.candidate, args.checkpoint, inputs, bert
    )
    changed = dict(zip(selected, candidate, strict=True))
    if not all(row["ran"] for row in candidate):
        raise ValueError("The comparison must run NuNER on every selected sentence")
    for i in controls:
        if stock[i]["spans"] != changed[i]["spans"]:
            raise ValueError("An already-routed control changed predictions")
    # The proposed guard keeps stock routing when BERT succeeded. Check both routes on real inputs.
    if bert:
        guard_controls = (
            controls + sorted(skipped, key=lambda i: (text_sha(cases[i][0]), i))[:8]
        )
        guarded = run_worker(
            args.candidate, args.checkpoint, [cases[i] for i in guard_controls], bert
        )
        for i, row in zip(guard_controls, guarded, strict=True):
            if row["ran"] != stock[i]["ran"] or row["spans"] != stock[i]["spans"]:
                raise ValueError("BERT-backed output changed under the proposed guard")
    all_counts, skipped_counts, added_errors = Counter(), Counter(), Counter()
    for i, (text, gold) in enumerate(cases):
        # All inputs for which code paths differ have measured candidate outputs.
        old = stock[i]["spans"]
        new = changed.get(i, stock[i])["spans"]
        add_case(all_counts, text, gold, old, new, names)
        if stock[i]["ran"]:
            continue
        add_case(skipped_counts, text, gold, old, new, names)
        targets = {(a, b, PROJECTION[t]) for a, b, t in gold}
        for a, b, kind in ({tuple(s) for s in new} - {tuple(s) for s in old}) - targets:
            if kind not in LABELS:
                continue
            if any(a == start and b == end for start, end, _ in targets):
                reason = "same_span_wrong_label"
            elif any(a < end and start < b for start, end, _ in targets):
                reason = "different_overlapping_span"
            else:
                reason = "no_reference_overlap"
            added_errors[reason] += 1
    if sum(added_errors.values()) != skipped_counts["in_schema_unmatched_added"]:
        raise ValueError("Added unmatched predictions did not reconcile")
    print(
        f"{split}, {topology}: gained {skipped_counts['exact_gained']}, lost {skipped_counts['exact_lost']}",
        file=sys.stderr,
        flush=True,
    )
    return {
        "all_sentences": len(cases),
        "stock_skipped": len(skipped),
        "stock_routed": len(routed),
        "routed_control_sentences": len(controls),
        "guarded_healthy_bert_controls": len(guard_controls) if bert else 0,
        "full_split": dict(all_counts),
        "skipped_subset": dict(skipped_counts),
        "added_unmatched_by_overlap": dict(added_errors),
        "warm_skipped_subset_latency": {
            "stock": timing([stock[i] for i in skipped]),
            "candidate": timing([changed[i] for i in skipped]),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "stock",
        "candidate",
        "always",
        "data",
        "checkpoint",
        "bert",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if (args.bert / "model_quantized.onnx").exists():
        parser.error(
            "The local BERT loader would select a quantized graph instead of the pinned FP32 graph"
        )
    for directory, members in ((args.checkpoint, FILES), (args.bert, BERT_FILES)):
        for path, expected in members.items():
            if file_sha(directory / path) != expected:
                parser.error(f"Unexpected pinned checkpoint member {directory / path}")
    with MANIFEST.open(newline="", encoding="utf-8") as f:
        hashes = {
            r["path"]: r["sha256"]
            for r in csv.DictReader(f)
            if r["id"] == "Wnut17" and r["status"] == "ok" and r["revision"] == REVISION
        }
    corpora = {
        split: load_file(name, hashes[name], args.data) for split, name in GOLD.items()
    }
    train = corpora["Train"]
    names = {surface(span, train.tokens).casefold() for span in train.spans}
    result = {
        "base_revision": "75f961e54631774b4e74547a383eaf8fc79a772d",
        "checkpoint": CHECKPOINT,
        "checkpoint_sha256": FILES,
        "bert_checkpoint": BERT_CHECKPOINT,
        "bert_sha256": BERT_FILES,
        "wnut_revision": REVISION,
        "source_sha256": {name: hashes[name] for name in GOLD.values()},
        "worker_sha256": {
            k: file_sha(getattr(args, k)) for k in ("stock", "candidate", "always")
        },
        "patch_sha256": {
            name: file_sha(Path(__file__).parent / "fixtures" / name)
            for name in ("nuner-fallback-routing.patch", "nuner-always-routing.patch")
        },
        "stacks": [
            ["nuner", "regex", "heuristic"],
            ["bert-onnx", "nuner", "regex", "heuristic"],
        ],
        "nuner_threshold": 0.9,
        "output_threshold": 0.5,
        "projection": PROJECTION,
        "gold_by_source_label": {},
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "splits": {},
    }
    for split in ("Development", "Test"):
        cases = sentence_cases(corpora[split])
        gates = run_worker(args.stock, "--probe", cases)
        result["gold_by_source_label"][split] = gold_label_counts(cases, gates)
        result["splits"][split] = {
            "without_bert": measure(split, cases, names, args, gates, None),
            "with_bert": measure(split, cases, names, args, gates, args.bert),
        }
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
