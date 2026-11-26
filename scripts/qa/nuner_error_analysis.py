"""Summarize unmatched NuNER spans from the threshold-0.5 paired ledger."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def mismatch(span, alternatives):
    start, end, label = span
    overlaps = [
        (left, right, other)
        for left, right, other in alternatives
        if start < right and left < end
    ]
    if any((start, end) == (left, right) for left, right, _ in overlaps):
        return "type_only"
    if any(label == other for _, _, other in overlaps):
        return "boundary_only"
    if overlaps:
        return "boundary_and_type"
    return "no_overlap"


def diagnose(cases):
    groups = defaultdict(Counter)
    for case in cases:
        gold = set(map(tuple, case["gold"]))
        predictions = {key: set(map(tuple, case[key])) for key in ("fp32", "u8")}
        counts = groups[case["group"]]
        for key, other in [("fp32", "u8"), ("u8", "fp32")]:
            pred = predictions[key]
            for direction, errors, alternatives, other_errors in [
                ("fp", pred - gold, gold, predictions[other] - gold),
                ("fn", gold - pred, pred, gold - predictions[other]),
            ]:
                for error in errors:
                    shared = "both" if error in other_errors else "only"
                    counts[(key, direction, shared, mismatch(error, alternatives))] += 1
    return groups


def checked_errors(ledger):
    cases = ledger["cases"]
    sizes = Counter(case["group"] for case in cases)
    for model in ("fp32", "u8"):
        reported = ledger["models"][model]["groups"]
        if dict(sizes) != {group: stats["n"] for group, stats in reported.items()}:
            raise ValueError(f"case counts differ from reported {model} group sizes")
        totals = defaultdict(Counter)
        for case in cases:
            gold = set(map(tuple, case["gold"]))
            pred = set(map(tuple, case[model]))
            totals[case["group"]].update(tp=len(pred & gold), fp=len(pred - gold), fn=len(gold - pred))
        for group, observed in totals.items():
            reference = reported[group]["thresholds"]["0.5"]
            if any(n != reference[k] for k, n in observed.items()):
                raise ValueError(f"{model}/{group} span counts differ from reported metrics")

    groups = diagnose(cases)
    for group, counts in groups.items():
        for model in ("fp32", "u8"):
            reference = ledger["models"][model]["groups"][group]["thresholds"]["0.5"]
            for direction in ("fp", "fn"):
                total = sum(v for (m, d, _, _), v in counts.items() if (m, d) == (model, direction))
                if total != reference[direction]:
                    raise ValueError(f"{group} {model} {direction}: expected {reference[direction]}, got {total}")
    return groups


def public_summary(ledger, groups):
    cases = ledger["cases"]
    # Publish measurement totals; individual source offsets and predictions stay in the work area.
    summary = {key: ledger[key] for key in (
        "checkpoint", "checkpoint_revision", "dataset", "dataset_revision", "test_file_sha256",
        "labels_by_index", "packages",
    )}
    summary["paired_f1_delta_u8_minus_fp32_at_0.5"] = {
        group: {k: interval[k] for k in ("resamples", "defined", "seed", "percentile_95")}
        for group, interval in ledger["paired_f1_delta_u8_minus_fp32_at_0.5"].items()
    }
    summary["dataset_notice"] = {
        "authors": ["Simone Tedeschi", "Roberto Navigli"],
        "source": f"https://huggingface.co/datasets/{ledger['dataset']}/tree/{ledger['dataset_revision']}",
        "license": "CC-BY-NC-SA-4.0",
        "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
        "changes": "Three-class aggregation of English and Russian test annotations.",
    }
    summary["decoder"] = ledger.get("decoder", "historical raw-logit argmax and FP64 sigmoid threshold")
    summary["models"] = {}
    for model in ("fp32", "u8"):
        stats = ledger["models"][model]
        public = {k: stats[k] for k in (
            "sha256", "bytes", "ort", "provider", "machine", "cpu_affinity",
            "intra_threads", "inter_threads", "session_load_seconds", "process_peak_rss_kib",
        )}
        public["groups"] = {}
        for group, scores in stats["groups"].items():
            totals = {k: scores[k] for k in ("n", "elapsed_seconds", "median_seconds")}
            totals["thresholds"] = {
                t: {k: counts[k] for k in ("tp", "fp", "fn", "f1", "precision", "recall")}
                for t, counts in scores["thresholds"].items()
            }
            public["groups"][group] = totals
        summary["models"][model] = public
    summary["comparison"] = {
        "cases": len(cases),
        "compared_word_labels": sum(c["compared_word_labels"] for c in cases),
        "max_absolute_probability_change": max(c["max_absolute_probability_change"] for c in cases),
        "threshold_crossings": {str(t): sum(c["threshold_crossings"][str(t)] for c in cases) for t in (0.3, 0.5, 0.7)},
        "different_span_sets_at_0.5": sum(set(map(tuple, c["fp32"])) != set(map(tuple, c["u8"])) for c in cases),
    }
    counts = sum(groups.values(), Counter())
    summary["error_counts_at_0.5"] = [
        {"model": m, "direction": d, "arms": a, "category": category, "count": n}
        for (m, d, a, category), n in sorted(counts.items())
    ]
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path, help="Private case-level evaluation receipt")
    parser.add_argument("--summary", type=Path, help="Write a public aggregate-only JSON report")
    args = parser.parse_args()
    ledger = json.loads(args.receipt.read_text())
    groups = checked_errors(ledger)
    if args.summary:
        args.summary.write_text(json.dumps(public_summary(ledger, groups), indent=2) + "\n")
    else:
        print("group,model,direction,arms,category,count")
        for group, counts in groups.items():
            for key, count in sorted(counts.items()):
                print(",".join((group, *key, str(count))))
