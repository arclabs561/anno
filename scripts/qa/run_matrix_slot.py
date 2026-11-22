"""Run one explicit budget slot with the existing matrix selectors and retain its receipts."""

import argparse
import json
import os
from pathlib import Path
import subprocess

from compare_matrix_picks import PLAN_TEST, build_test_binary, record
from replay_matrix_feedback import ROOT, TEST


def choose_strategy(policy, slot, period, coverage_slot):
    if slot < 1 or not 1 <= coverage_slot <= period:
        raise ValueError("slot must be positive and coverage-slot must be in 1..period")
    if policy == "random":
        return "random"
    if policy != "mixed":
        raise ValueError(f"unknown matrix policy {policy!r}")
    return "estimate" if (slot - 1) % period + 1 == coverage_slot else "worst-first"


def run_slot(binary, cache, receipts, policy, slot, period, coverage_slot, seed, plan_only=False):
    strategy = choose_strategy(policy, slot, period, coverage_slot)
    if not 0 <= seed < 2**64:
        raise ValueError("seed must fit an unsigned 64-bit integer")
    cache = cache.resolve()
    receipts = receipts.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    receipts.mkdir(parents=True, exist_ok=False)
    plan = dict(policy=policy, slot=slot, period=period, coverage_slot=coverage_slot,
                requested_strategy=strategy, seed=seed, plan_only=plan_only)
    record(receipts / "schedule.json", plan)
    env = dict(os.environ, ANNO_CACHE_DIR=str(cache), ANNO_EVAL_HISTORY=str(cache / "eval-results.jsonl"),
               ANNO_SAMPLE_STRATEGY=strategy, ANNO_CI_SEED=str(seed),
               ANNO_MUXER_DECISIONS_FILE=str(receipts / "decisions.jsonl"))
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
        observed = sorted({r["strategy"] for r in map(json.loads, rows) if r.get("record_type") == "outcome"})
    if not observed:
        raise ValueError("matrix sampler emitted no outcome receipts")
    matches = observed == [strategy]
    record(receipts / "result.json", {
        **plan, "observed_strategies": observed, "strategy_matches": matches,
    })
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
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--test-binary", type=Path, help="previously built anno-eval library test executable")
    p.add_argument("--plan-only", action="store_true", help="seal a selection without scoring it")
    args = p.parse_args()
    binary = args.test_binary.resolve() if args.test_binary else build_test_binary()
    run_slot(binary, args.cache_dir, args.receipts, args.policy, args.slot, args.period,
             args.coverage_slot, args.seed, args.plan_only)


if __name__ == "__main__":
    main()
