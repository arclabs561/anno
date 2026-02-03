#!/usr/bin/env python3
"""
Repeat muxer measure runs across seeds and aggregate results.

This is intentionally pragmatic: it drives the existing matrix harness (cargo test) and
then runs `muxer_agg` over the produced JSONL log.

Typical usage:
  python3 scripts/muxer_measure_repeat.py --runs 10 --seed-base 0 --log .generated/muxer_repeat.jsonl

All other env vars are inherited from your shell, so you can pin/fix facets like:
  ANNO_MUXER_MODE=measure
  ANNO_MATRIX_TASK=ner
  ANNO_MUXER_PIN_LANG=en
  ANNO_MUXER_PIN_DOMAIN=wikipedia
  ANNO_MUXER_PIN_BACKEND=crf,hmm
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], env: dict[str, str]) -> None:
    p = subprocess.run(cmd, env=env)
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--log", type=str, default=".generated/muxer_repeat.jsonl")
    ap.add_argument("--out", type=str, default=".generated/muxer_repeat_agg.json")
    ap.add_argument("--max-examples", type=int, default=None)
    args = ap.parse_args()

    if args.runs <= 0:
        print("runs must be > 0", file=sys.stderr)
        return 2

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()

    env_base = os.environ.copy()
    env_base.setdefault("ANNO_MUXER_MODE", "measure")
    env_base.setdefault("ANNO_MUXER_DECISIONS_FILE", str(log_path.resolve()))
    env_base.setdefault("ANNO_MUXER_VERBOSE", "0")
    env_base.setdefault("ANNO_MUXER_BACKENDS_PER_RUN", "1")

    if args.max_examples is not None:
        env_base["ANNO_MAX_EXAMPLES"] = str(args.max_examples)

    test_cmd = [
        "cargo",
        "test",
        "-p",
        "anno-eval",
        "--features",
        "eval-advanced",
        "test_randomized_matrix_sample",
        "--",
        "--nocapture",
    ]

    for i in range(args.runs):
        seed = args.seed_base + i
        env = env_base.copy()
        env["ANNO_CI_SEED"] = str(seed)
        print(f"[muxer-repeat] run {i+1}/{args.runs} seed={seed}", file=sys.stderr)
        run(test_cmd, env=env)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    agg_cmd = [
        "cargo",
        "run",
        "-p",
        "anno-eval",
        "--features",
        "eval-advanced",
        "--bin",
        "muxer_agg",
        "--",
        "--out",
        str(out_path),
        str(log_path),
    ]
    print(f"[muxer-repeat] aggregating -> {out_path}", file=sys.stderr)
    run(agg_cmd, env=env_base)
    print(f"[muxer-repeat] done: log={log_path} agg={out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

