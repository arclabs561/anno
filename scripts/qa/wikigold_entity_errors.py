#!/usr/bin/env python3
"""Replay stock WikiGold predictions through anno eval and plot all sentence reports.

Use the stock worker and verified FP32 NuNER checkpoint from
docs/eval/nuner-fallback-routing.md. The output directory receives an aggregate
JSON receipt, SVG, and HTML. Only temporary files contain sentence-level data.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import wikigold_nuner_routing as wikigold

# Ordered (sentence hash, final typed spans) from the verified BERT-absent stock
# run. Excluding timings lets an independently built stock worker reproduce it.
STOCK_SPANS_SHA256 = "eba388dec93c184b91800ef33d557cb30575c65aa36526efbe32e467aa639e23"
OUTCOMES = (
    "correct",
    "type_mismatch",
    "boundary_error",
    "false_negative",
    "false_positive",
)


def outputs(args, cases):
    if args.rows is not None:
        rows = json.loads(args.rows.read_text())
    else:
        for member, digest in wikigold.routing.FILES.items():
            if wikigold.routing.file_sha(args.checkpoint / member) != digest:
                raise ValueError(f"unexpected NuNER checkpoint member {member}")
        rows = wikigold.routing.run_worker(args.stock_worker, args.checkpoint, cases)
    ordered = hashlib.sha256()
    for (text, _), row in zip(cases, rows, strict=True):
        if row["input_sha256"] != wikigold.routing.text_sha(text):
            raise ValueError("stock predictions differ from the pinned sentence order")
        record = json.dumps(
            [row["input_sha256"], row["spans"]],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        ordered.update((record + "\n").encode())
    if ordered.hexdigest() != STOCK_SPANS_SHA256:
        raise ValueError(
            "final typed spans differ from the recorded BERT-absent stock run"
        )
    return rows


def eval_sentence(anno, temp, ordinal, text, gold, predicted):
    # Preserve the native worker's final span order. Eval assigns greedy partial
    # overlaps in prediction order after reserving every exact-location match.
    name = f"wikigold-sentence-{ordinal}"
    gold_file, prediction_file = temp / "gold.json", temp / "predictions.json"
    annotations = [
        {"start": a, "end": b, "label": label, "text": text[a:b]}
        for a, b, label in gold
    ]
    signals = [
        {
            "id": i,
            "location": {"Text": {"start": a, "end": b}},
            "surface": text[a:b],
            "label": label,
            "confidence": 1.0,
            "modality": "Symbolic",
            "negated": False,
        }
        for i, (a, b, label) in enumerate(predicted)
    ]
    gold_file.write_text(
        json.dumps({"id": name, "text": text, "entities": annotations})
    )
    prediction_file.write_text(
        json.dumps(
            {
                "id": name,
                "text": text,
                "signals": signals,
                "tracks": {},
                "identities": {},
            }
        )
    )
    result = subprocess.run(
        [
            anno,
            "eval",
            f"--text={text}",
            "--gold-file",
            gold_file,
            "--predictions",
            prediction_file,
            "--json",
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"anno eval failed for sentence {ordinal}: {result.stderr.decode()}"
        )
    report = json.loads(result.stdout)
    if report["predictions"] != {
        "document_id": name,
        "source_sha256": wikigold.routing.text_sha(text),
    }:
        raise ValueError(f"wrong evaluation source at occurrence {ordinal}")
    expected = len(set(gold) & {tuple(span) for span in predicted})
    if (report["correct"], report["gold_count"], report["predicted_count"]) != (
        expected,
        len(gold),
        len(predicted),
    ):
        raise ValueError(
            f"evaluation disagrees with independent exact typed spans at occurrence {ordinal}"
        )
    return report


def replay(args, output):
    cases, source = wikigold.load_release(args.data)
    rows = outputs(args, cases)
    counts = Counter(dict.fromkeys(OUTCOMES, 0))
    label_pairs, partial_labels = Counter(), Counter()
    by_gold, by_pred = defaultdict(Counter), defaultdict(Counter)
    with tempfile.TemporaryDirectory(prefix="anno-wikigold-errors-") as folder:
        temp = Path(folder)
        reports = temp / "eval.jsonl"
        with reports.open("w", encoding="utf-8") as stream:
            for ordinal, ((text, gold), row) in enumerate(
                zip(cases, rows, strict=True)
            ):
                predicted = row["spans"]
                report = eval_sentence(args.anno, temp, ordinal, text, gold, predicted)
                stream.write(json.dumps(report, separators=(",", ":")) + "\n")
                for event in report["matches"]:
                    kind = event["type"]
                    counts[kind] += 1
                    if "gold_id" in event:
                        gold_label = gold[event["gold_id"]][2]
                        by_gold[gold_label][kind] += 1
                    if "pred_id" in event:
                        pred_label = predicted[event["pred_id"]][2]
                        by_pred[pred_label][kind] += 1
                    if kind == "type_mismatch":
                        label_pairs[(gold_label, pred_label)] += 1
                    if kind == "boundary_error":
                        partial_labels[
                            (
                                "same_label"
                                if gold_label == pred_label
                                else "different_labels"
                            )
                        ] += 1

        result = subprocess.run(
            [
                args.anno,
                "viz",
                "errors",
                reports,
                "--title",
                "WikiGold entity errors under stock routing (no BERT)",
                "--format",
                "json",
            ],
            capture_output=True,
            check=True,
        )
        summary = json.loads(result.stdout)

    predicted = sum(sum(c.values()) for c in by_pred.values())
    outside = sum(
        sum(c.values()) for k, c in by_pred.items() if k not in wikigold.PROJECTION
    )
    gold = sum(sum(c.values()) for c in by_gold.values())
    expected = dict(
        exact=counts["correct"],
        same_span_wrong_label=counts["type_mismatch"],
        partial_overlap=counts["boundary_error"],
        missed_gold=counts["false_negative"],
        extra_predictions=counts["false_positive"],
        gold_count=gold,
        predicted_count=predicted,
    )
    pairs = [
        {"gold": g, "predicted": p, "pairs": n}
        for (g, p), n in sorted(
            label_pairs.items(), key=lambda item: (-item[1], item[0])
        )
    ]
    if (
        summary["reports"],
        summary["entities"],
        summary["same_span_label_confusions"],
    ) != (len(cases), expected, pairs):
        raise ValueError(
            "native pooled counts disagree with the sentence-level evaluation records"
        )
    expected_labels = {
        "by_gold_label": {
            label: {
                "exact": counts["correct"],
                "same_span_wrong_label": counts["type_mismatch"],
                "partial_overlap": counts["boundary_error"],
                "missed": counts["false_negative"],
            }
            for label, counts in by_gold.items()
        },
        "partial_labels": {
            "same": partial_labels["same_label"],
            "different": partial_labels["different_labels"],
        },
    }
    if summary.get("label_outcomes") != expected_labels:
        raise ValueError(
            "native label outcomes disagree with the independently aligned signals"
        )
    summary["metadata"] = {
        "source": source,
        "stock": {
            "model": "stock native stacked, FP32 NuNER available, BERT absent",
            "checkpoint": wikigold.routing.CHECKPOINT,
            "checkpoint_sha256": wikigold.routing.FILES,
            "ordered_final_typed_spans_sha256": STOCK_SPANS_SHA256,
        },
        "scope": "Each sentence occurrence contributes one anno eval --json report, including gold-empty occurrences. All final prediction labels are included.",
        "predicted_in_gold_inventory": predicted - outside,
        "predicted_outside_gold_inventory": outside,
        "by_gold_label": dict(by_gold),
        "by_predicted_label": dict(by_pred),
        # Preserve the independent signal-by-signal cross-check of the native aggregate.
        "partial_overlap_labels_from_aligned_signals": partial_labels,
    }
    # Have the native importer validate and serialize the annotated aggregate;
    # ordinary figure builds can render this same public JSON without model or corpus.
    seed, receipt = output / "annotated.json", output / "wikigold-entity-errors.json"
    seed.write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
    subprocess.run([args.anno, "viz", "errors", seed, "-o", receipt], check=True)
    for ext in ("svg", "html"):
        subprocess.run(
            [args.anno, "viz", "errors", receipt, "-o", receipt.with_suffix(f".{ext}")],
            check=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, required=True, help="pinned WikiGold CoNLL release"
    )
    parser.add_argument(
        "--anno", required=True, help="anno binary with viz errors support"
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--stock-worker", type=Path, help="stock stacked_nuner_routing example"
    )
    inputs.add_argument(
        "--rows", type=Path, help="reuse local verified stock worker rows (JSON array)"
    )
    parser.add_argument(
        "--checkpoint", type=Path, help="pinned FP32 NuNER directory for the worker"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if (args.stock_worker is not None) != (args.checkpoint is not None):
        parser.error("supply --checkpoint together with --stock-worker")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output_dir) as folder:
        staging = Path(folder)
        replay(args, staging)
        for ext in ("json", "svg", "html"):
            built = staging / f"wikigold-entity-errors.{ext}"
            built.replace(args.output_dir / built.name)


if __name__ == "__main__":
    main()
