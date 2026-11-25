"""Freeze two production first picks before scoring both held-out cells, offline."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from replay_matrix_feedback import DATASETS, ROOT, TEST, sample, sampler_env

POLICIES = ("random", "worst-first")
PLAN_TEST = "muxer_matrix::prospective_first_pick"


def build_test_binary():
    cmd = ["cargo", "test", "--offline", "--locked", "-p", "anno-eval", "--lib",
           "--no-default-features", "--features", "eval", "--no-run", "--message-format=json"]
    built = subprocess.check_output(cmd, cwd=ROOT, text=True)
    for line in built.splitlines():
        artifact = json.loads(line)
        if artifact.get("executable") and artifact["target"]["name"] == "anno_eval":
            return Path(artifact["executable"]).resolve()
    raise ValueError("Cargo did not return the anno-eval library test binary")


def record(path, obj):
    with path.open("x", encoding="utf-8") as out:
        json.dump(obj, out, indent=2)
        out.write("\n")


def fixtures(cache, text, low):
    directory = cache / "datasets"
    directory.mkdir(parents=True, exist_ok=True)
    for dataset in DATASETS:
        labeled = text.replace("-LOC", "-ORG").replace("-PER", "-ORG") if dataset == low else text
        (directory / f"{dataset}.cache").write_text(labeled, encoding="utf-8")


def first_pick(binary, cache, policy, seed):
    receipt = cache / "first-pick.json"
    env = sampler_env(cache, policy, cache / "selection-log.jsonl", seed=seed)
    env.update(ANNO_MATRIX_TASK="ner", ANNO_SELECTION_RECEIPT=str(receipt))
    history = cache / "eval-results.jsonl"
    saved = history.read_bytes()
    proc = subprocess.run([binary, PLAN_TEST, "--exact", "--ignored", "--nocapture"],
                          cwd=ROOT, env=env, capture_output=True, text=True)
    if proc.returncode or not receipt.is_file():
        raise ValueError(f"missing {policy} selection: {proc.stdout}\n{proc.stderr}")
    if history.read_bytes() != saved:
        raise ValueError("selection appended a measurement before the held-out panel was scored")
    plan = json.loads(receipt.read_text())
    if (plan["strategy"] != policy or plan["seed"] != seed or plan["tasks"] != ["ner"] or
            plan["backends"] != ["heuristic"] or len(plan["datasets"]) != 1 or plan["datasets"][0] not in DATASETS):
        raise ValueError(f"unexpected first pick: {plan}")
    return plan["datasets"][0]


def validate_pairs(picks, seeds):
    expected = {(ds, seed) for ds in DATASETS for seed in seeds}
    actual = {(pair["prior_low"], pair["seed"]) for pair in picks}
    if actual != expected or len(picks) != len(expected):
        raise ValueError("incomplete or duplicate history/seed pair")
    random_by_seed = {}
    for pair in picks:
        selected = pair["picks"]
        if selected["worst-first"] != pair["prior_low"]:
            raise ValueError("adaptive first pick did not follow the unresolved scored failure")
        if random_by_seed.setdefault(pair["seed"], selected["random"]) != selected["random"]:
            raise ValueError("the random control changed when only the observed labels were swapped")


def compare(binary, root, output, seeds):
    text = (ROOT / "crates/anno-eval/fixtures/matrix-smoke.conll").read_text()
    cmd = [binary, TEST, "--exact", "--ignored", "--nocapture"]
    picks = []
    for low in DATASETS:
        frozen = root / f"prior-{low}"
        fixtures(frozen, text, low)
        for ds in DATASETS:
            row, _ = sample(cmd, frozen, "random", frozen / f"observe-{ds}.jsonl", ds)
            if row["dataset"] != ds or row["primary_f1"] != float(ds != low):
                raise ValueError(f"training did not establish expected history: {row}")
        # Both selectors see the same neutral input. They can inspect loadability;
        # the held-out labels have never been placed in either reader's cache.
        fixtures(frozen, text, None)
        history = (frozen / "eval-results.jsonl").read_bytes()
        (output / f"history-{low}.jsonl").write_bytes(history)
        history_sha = sha256(history).hexdigest()
        for seed in seeds:
            pair = {}
            for policy in POLICIES:
                with tempfile.TemporaryDirectory(dir=root) as tmp:
                    cache = Path(tmp) / "reader"
                    shutil.copytree(frozen, cache)
                    pair[policy] = first_pick(binary, cache, policy, seed)
            picks.append({"prior_low": low, "history_sha256": history_sha, "seed": seed, "picks": pair})
    record(output / "picks.json", picks)
    validate_pairs(picks, seeds)

    # Only after both picks are sealed do independent processes score every cell.
    outcomes = []
    totals = Counter()
    for pair in picks:
        low = pair["prior_low"]
        other = next(ds for ds in DATASETS if ds != low)
        for scenario, current_low in (("persistent", low), ("switched", other), ("recovered", None)):
            scores = {}
            for ds in DATASETS:
                with tempfile.TemporaryDirectory(dir=root) as tmp:
                    cache = Path(tmp)
                    fixtures(cache, text, current_low)
                    row, _ = sample(cmd, cache, "random", cache / "measure.jsonl", ds, pair["seed"])
                if row["dataset"] != ds or row["primary_f1"] != float(ds != current_low):
                    raise ValueError(f"incomplete or unexpected held-out panel: {row}")
                scores[ds] = {"f1": row["primary_f1"], "observation_id": row["observation_id"]}
            outcomes.append({"prior_low": low, "seed": pair["seed"], "scenario": scenario, "scores": scores})
            for policy, selected in pair["picks"].items():
                totals[scenario, policy] += int(scores[selected]["f1"] == 0.0)
    record(output / "panels.json", outcomes)
    for scenario in ("persistent", "switched", "recovered"):
        print(f"{scenario}: " + ", ".join(f"{p} {totals[scenario, p]}/{len(picks)}" for p in POLICIES))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True, help="new directory for sealed picks, history and panel receipts")
    p.add_argument("--seeds", type=int, default=8, help="number of successive seeds, starting at zero")
    p.add_argument("--test-binary", type=Path, help="already built anno-eval test executable")
    args = p.parse_args()
    if args.seeds < 1:
        p.error("--seeds must be positive")
    binary = args.test_binary.resolve() if args.test_binary else build_test_binary()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory() as tmp:
        compare(binary, Path(tmp), args.output_dir, range(args.seeds))


if __name__ == "__main__":
    main()
