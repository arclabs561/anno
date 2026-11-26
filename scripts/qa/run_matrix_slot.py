"""Run one explicit budget slot with the existing matrix selectors and retain its receipts."""

import argparse
import json
import os
from pathlib import Path
import subprocess

from compare_matrix_picks import PLAN_TEST, build_test_binary, record
from replay_matrix_cell import language_tag
from replay_matrix_feedback import ROOT, TEST
import replay_feedback


def choose_strategy(policy, slot, period, coverage_slot):
    if slot < 1 or not 1 <= coverage_slot <= period:
        raise ValueError("slot must be positive and coverage-slot must be in 1..period")
    if policy == "random":
        return "random"
    if policy != "mixed":
        raise ValueError(f"unknown matrix policy {policy!r}")
    return "estimate" if (slot - 1) % period + 1 == coverage_slot else "worst-first"


def run_slot(binary, cache, receipts, policy, slot, period, coverage_slot, seed, plan_only=False,
             replay_history=True, coverage_cells=()):
    strategy = choose_strategy(policy, slot, period, coverage_slot)
    if not 0 <= seed < 2**64:
        raise ValueError("seed must fit an unsigned 64-bit integer")
    cells = [tuple(cell) for cell in coverage_cells]
    if cells and (policy != "mixed" or len(cells) != len(set(cells))
                  or any(not task or not language_tag(lang) for task, lang in cells)):
        raise ValueError("coverage cells require a mixed policy and distinct task/language pairs")
    target = cells[((slot - 1) // period) % len(cells)] if strategy == "estimate" and cells else None
    cache = cache.resolve()
    receipts = receipts.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    receipts.mkdir(parents=True, exist_ok=False)
    plan = dict(policy=policy, slot=slot, period=period, coverage_slot=coverage_slot,
                requested_strategy=strategy, seed=seed, plan_only=plan_only)
    if target:
        plan["coverage_target"] = {"task": target[0], "language": target[1]}
    # Only the first triage slot of each block may be spent on a reviewed case.
    replay_position = 1 if coverage_slot != 1 else 2
    if (replay_history and strategy == "worst-first"
            and (slot - 1) % period + 1 == replay_position):
        case = replay_feedback.next_case(ROOT, cache)
        if case is not None:
            plan.update(requested_strategy="reviewed-replay", selected_case=case)
            record(receipts / "schedule.json", plan)
            outcome_id = None
            if plan_only:
                record(receipts / "selection.json", plan)
            else:
                outcome = replay_feedback.observe(ROOT, cache, case["corpus"], case["case"], case)
                record(receipts / "replay.json", outcome)
                outcome_id = outcome["observation_id"]
            record(receipts / "result.json", {**plan, "observed_strategies":
                   [] if plan_only else ["reviewed-replay"], "selection_only": plan_only, "outcome_observation_id": outcome_id})
            return
    if binary is None:
        binary = build_test_binary()
    record(receipts / "schedule.json", plan)
    env = dict(os.environ, ANNO_CACHE_DIR=str(cache), ANNO_EVAL_HISTORY=str(cache / "eval-results.jsonl"),
               ANNO_SAMPLE_STRATEGY=strategy, ANNO_CI_SEED=str(seed),
               ANNO_MUXER_DECISIONS_FILE=str(receipts / "decisions.jsonl"))
    if target:
        env.update(ANNO_MATRIX_TASK=target[0], ANNO_MUXER_PIN_LANG=target[1])
    if plan_only:
        env["ANNO_SELECTION_RECEIPT"] = str(receipts / "selection.json")
    test = PLAN_TEST if plan_only else TEST
    subprocess.run([binary, test, "--exact", "--ignored", "--nocapture"],
                   cwd=ROOT, env=env, check=True)
    if plan_only:
        observed = [json.loads((receipts / "selection.json").read_text())["strategy"]]
    else:
        decisions = receipts / "decisions.jsonl"
        rows = decisions.read_text().splitlines() if decisions.is_file() else []
        evidence = [r for r in map(json.loads, rows) if r.get("record_type") == "outcome"]
        observed = sorted({r["strategy"] for r in evidence})
        if target:
            plan["target_evaluations"] = sum(
                row.get("task") == target[0] and row.get("observed_language") == target[1]
                and row.get("requested_languages") == [target[1]] and row.get("strategy") == strategy
                and row.get("hard_junk") is False and type(row.get("cost_units")) is int
                and row["cost_units"] > 0 for row in evidence
            )
    if not observed and not target:
        raise ValueError("matrix sampler emitted no outcome receipts")
    matches = observed == [strategy]
    record(receipts / "result.json", {
        **plan, "observed_strategies": observed, "strategy_matches": matches,
    })
    if target and not plan_only and not plan["target_evaluations"]:
        raise ValueError(f"no completed evaluation verified the requested task and language {target}")
    if not matches:
        raise ValueError(f"expected {strategy} in slot {slot}, observed {observed}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--receipts", type=Path, required=True, help="new directory for this attempt")
    p.add_argument("--policy", choices=("mixed", "random"), required=True)
    p.add_argument("--slot", type=int, required=True, help="one-based index in this budget")
    p.add_argument("--period", type=int, default=4)
    p.add_argument("--coverage-slot", type=int, default=2, help="one-based exploration position in each period")
    p.add_argument("--coverage-cell", nargs=2, action="append", default=[], metavar=("TASK", "LANG"),
                   help="rotate the mixed policy's estimate slots over these verified task/language cells")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--test-binary", type=Path, help="previously built anno-eval library test executable")
    p.add_argument("--plan-only", action="store_true", help="seal a selection without scoring it")
    p.add_argument("--replay-history", action=argparse.BooleanOptionalAction, default=True,
                   help="use a vetted ledger for at most one triage slot per block (default: enabled)")
    args = p.parse_args()
    os.environ.setdefault("ANNO_CACHE_DIR", str(args.cache_dir.resolve()))
    binary = args.test_binary.resolve() if args.test_binary else None
    run_slot(binary, args.cache_dir, args.receipts, args.policy, args.slot, args.period,
             args.coverage_slot, args.seed, args.plan_only, args.replay_history, args.coverage_cell)


if __name__ == "__main__":
    main()
