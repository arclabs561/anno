"""Check coverage from the current CI run's matrix decision/outcome JSONL."""

import argparse
from collections import Counter, defaultdict
import json
import re
from pathlib import Path
import sys

from export_muxer_summary import WIKIGOLD
from replay_matrix_cell import FIXTURES, language_tag, replay_command


RANDOM_TASKS = ("ner", "re", "intra-coref", "temporal")


def outcomes(lines, decisions=None):
    for number, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
            if row.get("record_type") != "outcome":
                if (decisions is not None and not decisions
                        and row.get("strategy") == "worst-first" and row.get("chosen")):
                    decisions.append(row)
                continue
            strategy = row["strategy"]
            task = row["slice"].split(".", 1)[0]
            if strategy not in {"random", "worst-first"}:
                raise ValueError(f"unexpected sampling strategy {strategy!r}")
            if strategy == "random" and task not in RANDOM_TASKS:
                raise ValueError(f"unexpected random task {task!r}")
            hard_failure = row["hard_junk"]
            examples = row["cost_units"]
            if type(hard_failure) is not bool or type(examples) is not int or examples < 0:
                raise ValueError("expected boolean hard_junk and nonnegative integer cost_units")
        except (ValueError, TypeError, AttributeError, KeyError) as exc:
            raise ValueError(f"invalid matrix log at line {number}: {exc}") from exc

        yield task if strategy == "random" else strategy, row


def check_fixed_sample(lines, *, task, dataset, backend, examples, exact, counts=None, digest=None):
    rows = list(outcomes(lines))
    if len(rows) == 1:
        lane, row = rows[0]
        if dataset not in FIXTURES and (command := replay_command(row)):
            print(f"Replay this cell: `{command}`")
        score = row.get("primary_f1")
        observed = row.get("structural_counts")
        counts_match = counts is None or (isinstance(observed, dict) and all(
            type(observed.get(key)) in (float, int) and observed[key] == value
            for key, value in counts.items()))
        if (lane == task and row.get("dataset") == dataset and row.get("backend") == backend
                and (digest is None or row.get("dataset_sha256") == digest)
                and not row["hard_junk"] and row["cost_units"] == examples
                and type(score) in (float, int) and 0 <= score <= 1
                and (not exact or score == 1.0)
                and counts_match):
            print(f"{dataset} / {backend}: {examples} examples, F1 {score:.4f}.")
            return True
    print(f"Unexpected fixed sample result for {task} / {dataset} / {backend}.", file=sys.stderr)
    return False


def check_ner_smoke(lines):
    return check_fixed_sample(lines, task="ner", dataset="AnnoNerSmoke", backend="heuristic",
                              examples=1, exact=True)


def check_wikigold_bootstrap(lines):
    return check_fixed_sample(lines, task="ner", dataset="WikiGold", backend="heuristic",
                              examples=5, exact=False, digest=WIKIGOLD)


def check_temporal_smoke(lines):
    return check_fixed_sample(lines, task="temporal", dataset="AnnoTimexSmoke",
                              backend="pattern", examples=2, exact=True,
                              counts={"gold_spans": 1, "predicted_spans": 1, "matched_spans": 1,
                                      "gold_empty_sentences": 1, "gold_empty_sentences_with_predictions": 0,
                                      "unmatched_predictions_overlapping_missed_gold": 0})


def check_temporal_contrast(lines):
    return check_fixed_sample(lines, task="temporal", dataset="AnnoTimexContrast",
                              backend="pattern", examples=34, exact=False)


def report_dataset_pools(lines):
    """Report each automatic selection pass independently, including empty passes."""
    for number, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
            pool = row["pool"]
            strategy, task, phase = row["strategy"], row["slice"], row["pass"]
            excluded, rejected = pool["excluded"], pool["load_rejections"]
            examples = pool.get("rejection_examples", [])
            if not isinstance(examples, list) or len(examples) > 3:
                raise ValueError("invalid rejection examples")
            for example in examples:
                if (not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", example["dataset"])
                        or example["reason"] not in rejected):
                    raise ValueError("invalid rejection example")
            counts = (pool["catalog_count"], pool["eligible_count"], pool["backend_incompatible_count"],
                      *excluded.values(), *rejected.values())
            if (not isinstance(pool["selected"], list)
                    or any(type(n) is not int or n < 0 for n in counts)):
                raise ValueError("invalid selection or dataset count")
            selected = len(pool["selected"])
            probed = selected + sum(rejected.values())
            unchecked = pool["eligible_count"] - probed - pool["backend_incompatible_count"]
            if (row["schema_version"] != 1 or unchecked < 0
                    or pool["catalog_count"] != pool["eligible_count"] + sum(excluded.values())):
                raise ValueError("inconsistent dataset pool counts or unsupported schema")
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise ValueError(f"invalid dataset pool at line {number}: {exc}") from exc
        print(f"- {strategy} / {task} ({phase}): "
              f"{pool['eligible_count']}/{pool['catalog_count']} policy-eligible, "
              f"{probed} load checks, {selected} selected, "
              f"{pool['backend_incompatible_count']} incompatible, {unchecked} unexamined.")
        for label, counts in (("Policy exclusions", excluded), ("Load rejections", rejected)):
            if counts:
                print(f"  - {label}: " + ", ".join(f"{key} ({value})" for key, value in sorted(counts.items())))
        for example in examples:
            print(f"  - Rejected {example['dataset']} ({example['reason']}).")


def check(lines, report=False, verify_wikigold=False):
    unverified_wikigold = 0
    attempts = Counter()
    completions = Counter()
    measured_languages = Counter()
    failures = Counter()
    failure_kinds = Counter()
    failed_cells = Counter()
    inputs_by_digest = defaultdict(set)
    decisions = []
    observed = {}
    replay_commands = set()
    unavailable_commands = 0
    for lane, row in outcomes(lines, decisions):
        if report:
            digest, dataset = row.get("dataset_sha256"), row.get("dataset")
            if (isinstance(digest, str) and re.fullmatch("[a-f0-9]{64}", digest)
                    and isinstance(dataset, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", dataset)):
                inputs_by_digest[digest].add(dataset)
            command = replay_command(row)
            if command is None:
                unavailable_commands += 1
            else:
                replay_commands.add(command)
        attempts[lane] += 1
        if lane == "worst-first":
            observed[(row.get("run_id"), row.get("backend"), row.get("dataset"))] = row
        if row["hard_junk"]:
            failures[lane] += 1
            kind = row.get("fail_kind") or "unknown"
            failure_kinds[kind] += 1
            if row.get("dataset") and row.get("backend"):
                task = row["slice"].split(".", 1)[0]
                failed_cells[(task, row["dataset"], row["backend"], kind)] += 1
        elif row["cost_units"] > 0:
            completions[lane] += 1
            observed_lang = row.get("observed_language")
            if observed_lang is not None and not language_tag(observed_lang):
                raise ValueError("invalid observed language in matrix outcome")
            task = row["slice"].split(".", 1)[0]
            measured_languages[task, observed_lang or "unverified"] += 1
            if verify_wikigold and row.get("dataset") == "WikiGold" and row.get("dataset_sha256") != WIKIGOLD:
                unverified_wikigold += 1

    print("| Sample lane | Completed | Hard failures | Empty evaluations |")
    print("|---|---:|---:|---:|")
    for lane in (*RANDOM_TASKS, "worst-first"):
        empty = attempts[lane] - completions[lane] - failures[lane]
        label = lane if attempts[lane] else f"{lane} (no outcome)"
        print(f"| {label} | {completions[lane]} | {failures[lane]} | {empty} |")

    if report and measured_languages:
        print("\n| Task | Observed input language | Completed evaluations |")
        print("|---|---|---:|")
        for (task, language), count in sorted(measured_languages.items()):
            print(f"| {task} | {language} | {count} |")

    for digest, datasets in sorted(inputs_by_digest.items()):
        if len(datasets) > 1:
            print(f"Dataset IDs {', '.join(sorted(datasets))} share recorded input SHA-256 `{digest}`.")

    if failure_kinds:
        print("Hard failures by recorded kind: " + ", ".join(
            f"{kind} {count}" for kind, count in sorted(failure_kinds.items())))

    if failed_cells:
        print("Hard failures by task, dataset and backend (up to 10):")
        for (task, dataset, backend, kind), attempts_count in failed_cells.most_common(10):
            print(f"- {task} / {dataset} / {backend}: {kind} ({attempts_count})")
        if len(failed_cells) > 10:
            print(f"{len(failed_cells) - 10} additional combinations are in the decision log.")

    if decisions:
        first = decisions[0]
        backend = first["chosen"]
        candidates = first.get("top_candidates", {}).get("rows", [])
        selected = next((r for r in candidates if r["arm"] == backend), {})
        prior_calls = selected.get("calls", "unknown")
        reason = "exploration" if first.get("explore_first") else "ranked"
        retry = (first.get("worst_first_round") or {}).get("retry")
        if retry:
            reason = {"outstanding_failure": "unresolved failure",
                      "repeated_infrastructure": "probe of repeated infrastructure failure",
                      "recurrent_after_recovery": "repeated failures after recovery"}.get(retry.get("reason"), "ranked")
            print(f"Retry evidence: {retry['observation_id']} "
                  f"(risk weight {retry['risk']:.3f} over {retry['observations']} observations).")
        for dataset in first.get("datasets", []):
            row = observed.get((first.get("run_id"), backend, dataset))
            result = "no outcome"
            if row is not None:
                result = "hard failure" if row["hard_junk"] else f"{row['cost_units']} examples, F1 {row.get('primary_f1')}"
            print(f"First worst-first selection: {first['slice']} / {dataset} / {backend} ({reason}, prior calls {prior_calls}) — {result}.")
    else:
        print("No worst-first selection recorded.")

    if replay_commands:
        print("Replay the resolved cells from this checkout with the CI evaluator. "
              "External cells require the authorized dataset cache and any model assets. "
              "A loader digest, when available, identifies the input reported by that loader.")
        for command in sorted(replay_commands):
            print(f"- `{command}`")

    if unavailable_commands:
        print(f"{unavailable_commands} outcome(s) lack supported replay metadata.")

    missing = [task for task in RANDOM_TASKS if not completions[task]]
    if missing:
        print(f"No completed evaluations with examples for {', '.join(missing)}", file=sys.stderr)
    # Worst-first deliberately targets poor/failing cells. A recorded failure
    # demonstrates an attempt, but a selection decision alone does not.
    if not attempts["worst-first"]:
        print("No worst-first evaluation outcomes", file=sys.stderr)
    if unverified_wikigold:
        print(f"{unverified_wikigold} scored WikiGold outcome(s) lack the prepared corpus digest.", file=sys.stderr)
        return False
    return report or (not missing and bool(attempts["worst-first"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--verify-wikigold", action="store_true",
                        help="require the prepared release digest for scored WikiGold outcomes")
    parser.add_argument("--dataset-pools", type=Path, help="report automatic dataset selection passes from the companion JSONL")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--bootstrap", action="store_true", help="require five real WikiGold examples")
    mode.add_argument("--smoke", action="store_true", help="require the exact offline NER fixture result")
    mode.add_argument("--temporal-smoke", action="store_true",
                      help="require two scored temporal documents and perfect fixture F1")
    mode.add_argument("--temporal-contrast", action="store_true", help="require the 34-document offline temporal panel")
    mode.add_argument("--public-timex", action="store_true", help="require the full pinned TIMEX test split")
    mode.add_argument("--report", action="store_true", help="report missing external coverage without failing")
    args = parser.parse_args()
    try:
        if args.dataset_pools:
            with args.dataset_pools.open(encoding="utf-8") as pools:
                report_dataset_pools(pools)
        with args.log.open(encoding="utf-8") as stream:
            if args.temporal_contrast:
                passed = check_temporal_contrast(stream)
            elif args.public_timex:
                passed = check_fixed_sample(stream, task="temporal", dataset="TimexRecognitionSentenceOriginal",
                                            backend="pattern", examples=390, exact=False)
            elif args.temporal_smoke:
                passed = check_temporal_smoke(stream)
            elif args.smoke:
                passed = check_ner_smoke(stream)
            elif args.bootstrap:
                passed = check_wikigold_bootstrap(stream)
            else:
                passed = check(stream, args.report, args.verify_wikigold)
            sys.exit(0 if passed else 1)
    except (OSError, ValueError) as error:
        sys.exit(str(error))
