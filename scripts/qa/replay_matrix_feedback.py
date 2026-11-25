"""Verify that persisted evaluation results change which matrix cell runs first."""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from zipfile import ZipFile

from matrix_checkpoint import snapshot, unpack

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("WikiGold", "CoNLL2003Sample")
TEST = "muxer_matrix::test_randomized_matrix_sample"


def sampler_env(cache, strategy, output, fixed=None, seed=41):
    # A developer's defaults must not introduce another backend, seed, or history.
    env = {key: value for key, value in os.environ.items() if not key.startswith("ANNO_")}
    # Carry the build identity into Cargo alongside the isolated sampler settings.
    if "ANNO_GIT_COMMIT" in os.environ:
        env["ANNO_GIT_COMMIT"] = os.environ["ANNO_GIT_COMMIT"]
    env.update(
        ANNO_CACHE_DIR=str(cache),
        ANNO_EVAL_HISTORY=str(cache / "eval-results.jsonl"),
        ANNO_MUXER_DECISIONS_FILE=str(output),
        ANNO_SAMPLE_STRATEGY=strategy,
        ANNO_CI_SEED=str(seed),
        ANNO_MAX_EXAMPLES="1",
        ANNO_ML_IN_MATRIX="0",
        ANNO_MUXER_PIN_BACKEND="heuristic",
        ANNO_MUXER_DATASETS_PER_RUN="1",
        ANNO_MUXER_BACKENDS_PER_RUN="1",
        ANNO_MATRIX_REQUIRE_CACHED="1",
        ANNO_MATRIX_TRY_DOWNLOAD_ON_EMPTY="0",
        ANNO_NO_DOWNLOADS="1",
        ANNO_S3_CACHE="0",
        ANNO_WORST_SOFT_WEIGHT="1",
    )
    if fixed is not None:
        env.update(ANNO_MATRIX_TASK="ner", ANNO_MUXER_FIXED_DATASETS=fixed)
    return env


def sample(command, cache, strategy, output, fixed=None, seed=41):
    env = sampler_env(cache, strategy, output, fixed, seed)
    result = subprocess.run(command, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    if result.returncode or not output.is_file():
        raise ValueError(f"sampler exited {result.returncode}, log {output.name}:\n{result.stdout}")
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    outcomes = [r for r in rows if r.get("record_type") == "outcome"]
    if len(outcomes) != 1:
        raise ValueError(f"{output.name}: expected one outcome, got {len(outcomes)}")
    row = outcomes[0]
    if row["backend"] != "heuristic" or row["hard_junk"] or row["cost_units"] != 1:
        raise ValueError(f"{output.name}: expected a measured one-example heuristic evaluation: {row}")
    return row, [r for r in rows if r.get("worst_first_round")]


def replay(command, root, low, fixture, features):
    writer = root / "writer"
    reader = root / "reader"
    (writer / "datasets").mkdir(parents=True)
    for dataset in DATASETS:
        # Identical text and cost in both cells. Change only the annotation.
        labels = fixture
        if dataset == low:
            labels = fixture.replace("-LOC", "-ORG").replace("-PER", "-ORG")
        (writer / "datasets" / f"{dataset}.cache").write_text(labels)
    logs = []
    prior_obs = {}
    for dataset in DATASETS:
        log = root / f"observe-{dataset}.jsonl"
        row, _ = sample(command, writer, "random", log, dataset)
        logs.append(log)
        prior_obs[dataset] = row["observation_id"]
        if row["dataset"] != dataset or row["primary_f1"] != float(dataset != low):
            raise ValueError(f"unexpected training observation for {dataset}: {row}")
    # Transport exactly the state the Actions job publishes. Neither the
    # original dataset cache nor the writer's decision logs reach the reader.
    checkpoint = root / "checkpoint"
    if snapshot(writer, checkpoint, logs, features=features) != len(DATASETS):
        raise ValueError("checkpoint did not record both observations")
    archive = root / "checkpoint.zip"
    with ZipFile(archive, "w") as compressed:
        for file in checkpoint.iterdir():
            compressed.write(file, file.name)
    unpack(archive, reader)
    (reader / "datasets").mkdir()
    # The next process sees healthy data on both sides. Its first choice can
    # only reflect the previous outcomes, and can now record the recovery.
    for dataset in DATASETS:
        (reader / "datasets" / f"{dataset}.cache").write_text(fixture)
    row, decisions = sample(command, reader, "worst-first", root / "next-run.jsonl")
    print(f"Prior zero-F1 cell: {low}. First measured cell on restore: {row['dataset']}.")
    if row["dataset"] != low or row["primary_f1"] != 1.0:
        raise ValueError("history did not prioritize the previously low-scoring cell")
    retry = decisions[0]["worst_first_round"].get("retry") if decisions else None
    if not retry or retry["observation_id"] != prior_obs[low]:
        raise ValueError("first decision did not reference the observation that caused the retry")
    _, following = sample(command, reader, "worst-first", root / "after-recovery.jsonl")
    if not following or following[0]["worst_first_round"].get("retry"):
        raise ValueError("a recovered cell still had an outstanding retry")

    control = root / "random-control"
    unpack(archive, control)
    (control / "datasets").mkdir()
    for dataset in DATASETS:
        (control / "datasets" / f"{dataset}.cache").write_text(fixture)
    random_row, _ = sample(command, control, "random", root / "random-control.jsonl")
    return random_row["dataset"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="eval,discourse")
    args = parser.parse_args()
    command = ["cargo", "test", "--locked", "-p", "anno-eval", "--lib", "--features",
               args.features, TEST, "--", "--exact", "--ignored", "--nocapture"]
    fixture = (ROOT / "scripts/qa/fixtures/matrix-smoke.conll").read_text()
    with tempfile.TemporaryDirectory() as tmp:
        controls = [replay(command, Path(tmp) / low, low, fixture, args.features) for low in DATASETS]
        if controls[0] != controls[1]:
            raise ValueError(f"swapping history also changed the seeded random control: {controls}")
        print(f"Random control chose {controls[0]} in both histories. Both successful retries cleared.")


if __name__ == "__main__":
    main()
