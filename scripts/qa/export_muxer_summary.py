"""Replay a pinned public cohort and export its aggregate evidence as JSON."""
import argparse
from hashlib import sha256
import json
import math
import os
import re
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
BACKENDS = ("heuristic", "hmm")
SEEDS = (41, 42, 43)
EXAMPLES = 32
WIKIGOLD = "c797a64d0cf73ed058363f77671486bcfd413e70fda1726ad4a6ba624455225f"
SCORER = "anno-eval-primary-score-v1"


def summarize(journal, decisions, revision):
    """Accept only completed observations confirmed by both evaluator receipts."""
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("expected the full evaluation commit")
    outcomes = {}
    for row in decisions:
        if row.get("record_type") != "outcome":
            continue
        if row.get("schema_version") != 3 or row.get("ok") is not True or row.get("strategy") != "fixed":
            raise ValueError("unexpected outcome in public replay")
        key = row.get("observation_id")
        if not isinstance(key, str) or not key:
            raise ValueError("unidentified outcome")
        evidence = (row["dataset"], row["backend"], row["primary_f1"], row["policy_cohort_key"])
        if outcomes.setdefault(key, evidence) != evidence:
            raise ValueError("conflicting outcomes for the same observation")

    trials = {}
    seen = {}
    expected = {(b, s) for b in BACKENDS for s in SEEDS}
    for row in journal:
        key = row.get("observation_id")
        trial = (row["backend"], row["seed"])
        score = row["f1"]
        meta = json.loads(row["metadata"])
        policy = json.loads(row["policy_cohort_key"])
        prov = meta["provenance"]
        if not (
            trial in expected and row["task"] == "NER" and row["dataset"] == "WikiGold"
            and row["n"] == EXAMPLES and row["error"] is None
            and type(score) in (int, float) and 0 <= score <= 1
            and isinstance(row.get("git_commit"), str) and len(row["git_commit"]) >= 7
            and revision.startswith(row["git_commit"]) and meta["success"] is True
            and prov["observation_id"] == key and prov["schema_version"] == 1
            and prov["backend"]["effective"] == row["backend"]
            and prov["runtime"]["scheduling"]["strategy"] == "sequential"
            and prov["runtime"]["seed"] == row["seed"] and prov["runtime"]["cached_only"] is True
            and policy["schema"] == "anno-eval-policy-cohort-v1"
            and policy["max_examples"] == EXAMPLES and policy["scorer"] == SCORER
            and outcomes.get(key) == ("WikiGold", row["backend"], score, row["policy_cohort_key"])
        ):
            raise ValueError("journal row does not match the public replay protocol")
        if seen.setdefault(key, trial) != trial or trials.setdefault(trial, score) != score:
            raise ValueError("conflicting evidence for an observation or protocol trial")
    if set(trials) != expected or seen.keys() != outcomes.keys():
        raise ValueError("incomplete or unpaired public replay")

    return {
        "schema": "anno-public-muxer-summary-v1",
        "source": "pinned-offline-replay",
        "evaluation_revision": revision,
        "dataset_sha256": WIKIGOLD,
        "scorer": SCORER,
        "seeds": list(SEEDS),
        "examples_per_trial": EXAMPLES,
        "cells": [
            {
                "task": "NER", "dataset": "WikiGold", "backend": backend,
                "evaluated_trials": len(SEEDS),
                "example_evaluations": EXAMPLES * len(SEEDS),
                "mean_trial_f1": round(math.fsum(trials[backend, seed] for seed in SEEDS) / len(SEEDS), 6),
            }
            for backend in BACKENDS
        ],
    }


def replay(wikigold):
    source = wikigold.read_bytes()
    if sha256(source).hexdigest() != WIKIGOLD:
        raise ValueError("input does not match the pinned public WikiGold corpus")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    with tempfile.TemporaryDirectory() as temporary:
        cache = Path(temporary)
        datasets = cache / "datasets"
        datasets.mkdir()
        (datasets / "WikiGold.cache").write_bytes(source)
        journal = cache / "eval-results.jsonl"
        decisions = cache / "decisions.jsonl"
        env = {k: v for k, v in os.environ.items() if not k.startswith("ANNO_")}
        env.update(
            ANNO_CACHE_DIR=str(cache), ANNO_EVAL_HISTORY=str(journal),
            ANNO_MUXER_DECISIONS_FILE=str(decisions), ANNO_HISTORY_FILE=str(cache / "history.json"),
            ANNO_NO_DOWNLOADS="1", ANNO_S3_CACHE="0", ANNO_MATRIX_REQUIRE_CACHED="1",
            ANNO_MATRIX_TRY_DOWNLOAD_ON_EMPTY="0", ANNO_ML_IN_MATRIX="0",
            ANNO_MATRIX_TASK="ner", ANNO_SAMPLE_STRATEGY="random",
            ANNO_MUXER_FIXED_DATASETS="WikiGold", ANNO_MAX_EXAMPLES=str(EXAMPLES),
        )
        for backend in BACKENDS:
            for seed in SEEDS:
                subprocess.run(
                    ["cargo", "test", "--locked", "-p", "anno-eval", "--lib", "--features", "eval",
                     "muxer_matrix::test_randomized_matrix_sample", "--", "--exact", "--ignored"],
                    cwd=ROOT, env={**env, "ANNO_MUXER_FIXED_BACKEND": backend, "ANNO_CI_SEED": str(seed)},
                    stdout=sys.stderr, check=True,
                )
        return summarize(
            (json.loads(line) for line in journal.read_text().splitlines()),
            (json.loads(line) for line in decisions.read_text().splitlines()), revision,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wikigold", type=Path, help="path to the pinned WikiGold corpus")
    print(json.dumps(replay(parser.parse_args().wikigold), indent=2, allow_nan=False))
