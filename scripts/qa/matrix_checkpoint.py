"""Restore and publish the matrix sampler's state using trusted CI artifacts."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile


STATE_FILES = {"linucb_global_state.json", "eval-results.jsonl", "eval-history.db"}


def state_file(name):
    return Path(name).name == name and (
        name in STATE_FILES or (name.startswith("muxer_history.") and name.endswith(".json"))
    )


def select_checkpoints(artifacts, branch, repo, current_run, lookup_run):
    # A retry of an older workflow can upload a newer artifact ID. Prefer the
    # latest producing run, and never feed a run its own previous attempt.
    ordered = sorted(artifacts, key=lambda a: (a["workflow_run"]["id"], a["id"]), reverse=True)
    published_run = None
    for artifact in ordered:
        producer = artifact["workflow_run"]
        if artifact["expired"] or producer["head_branch"] != branch:
            continue
        if published_run is not None and producer["id"] >= current_run:
            continue
        run = lookup_run(producer["id"])
        if (
            run["path"] == ".github/workflows/ci.yml"
            and run["event"] in {"push", "schedule", "workflow_dispatch"}
            and run["head_branch"] == branch
            and run["head_repository"]["full_name"] == repo
        ):
            if producer["id"] < current_run:
                return artifact, published_run
            published_run = producer["id"]
    return None, published_run


def unpack(archive, root):
    with zipfile.ZipFile(archive) as source:
        names = source.namelist()
        if "eval-results.jsonl" not in names or not any(n.startswith("muxer_history.") for n in names):
            raise ValueError("checkpoint missing evaluation results or muxer history")
        if len(names) != len(set(names)) or not all(state_file(n) for n in names):
            raise ValueError("unexpected files in matrix checkpoint")
        root.mkdir(parents=True, exist_ok=True)
        for name in names:
            with source.open(name) as src, (root / name).open("xb") as dest:
                shutil.copyfileobj(src, dest)


def restore(root, name, branch, repo, current_run):
    api = ["gh", "api", f"repos/{repo}/actions/artifacts", "--method", "GET",
           "-f", f"name={name}", "-F", "per_page=100", "--paginate", "--slurp"]
    pages = json.loads(subprocess.check_output(api))
    artifacts = [a for page in pages for a in page["artifacts"] if a["name"] == name]

    def lookup(run_id):
        return json.loads(subprocess.check_output(["gh", "api", f"repos/{repo}/actions/runs/{run_id}"]))

    artifact, published_run = select_checkpoints(artifacts, branch, repo, current_run, lookup)
    if artifact is None:
        return None, published_run
    with tempfile.TemporaryFile() as tmp:
        subprocess.run(["gh", "api", f"repos/{repo}/actions/artifacts/{artifact['id']}/zip"],
                       stdout=tmp, check=True)
        unpack(tmp, root)
    return artifact["workflow_run"]["id"], published_run


def journal_size(root):
    try:
        return (root / "eval-results.jsonl").stat().st_size
    except FileNotFoundError:
        return 0


def export_state(root, stage, features):
    env = dict(os.environ, ANNO_CHECKPOINT_SOURCE=str(root.resolve()),
               ANNO_CHECKPOINT_TARGET=str(stage.resolve()))
    subprocess.run(["cargo", "test", "--locked", "-p", "anno-eval", "--lib", "--features", features,
                    "muxer_matrix::checkpoint::export", "--", "--exact", "--ignored"],
                   cwd=Path(__file__).resolve().parents[2], env=env, check=True)
    if not ((stage / "eval-results.jsonl").is_file() and (stage / "eval-history.db").is_file()
            and any(stage.glob("muxer_history.*.json"))):
        raise ValueError("incomplete projected matrix checkpoint")


def snapshot(root, target, logs, journal_start=0, features="onnx,eval,discourse"):
    # The evaluator logs an outcome even if appending its history row failed.
    # Only post-restore journal entries prove that the next run can see it.
    expected = set()
    for log in logs:
        if not log.exists():
            continue
        with log.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("record_type") != "outcome":
                    continue
                obs = row.get("observation_id")
                if not isinstance(obs, str) or not obs:
                    raise ValueError(f"{log}: outcome has no observation ID")
                expected.add(obs)
    if not expected:
        return 0
    if not 0 <= journal_start <= journal_size(root):
        raise ValueError("evaluation journal shrank after checkpoint restore")
    missing = expected.copy()
    with (root / "eval-results.jsonl").open("rb") as journal:
        journal.seek(journal_start)
        for line in journal:
            missing.discard(json.loads(line).get("observation_id"))
    if missing:
        raise ValueError(f"{len(missing)} current observations missing from the appended journal")
    if not any(root.glob("muxer_history.*.json")):
        raise ValueError("observations recorded but muxer history missing")
    if target.exists():
        raise FileExistsError(target)
    with tempfile.TemporaryDirectory(dir=target.parent) as temp:
        stage = Path(temp) / "checkpoint"
        stage.mkdir()
        export_state(root, stage, features)
        stage.rename(target)
    return len(expected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    cmd = parser.add_subparsers(dest="command", required=True)
    cmd.add_parser("restore")
    save = cmd.add_parser("snapshot")
    save.add_argument("target", type=Path)
    save.add_argument("logs", nargs="+", type=Path)
    save.add_argument("--journal-start", type=int, default=0, help="byte length at restore")
    args = parser.parse_args()
    cache = Path(os.environ["ANNO_CACHE_DIR"])
    if args.command == "restore":
        repo = os.environ["GITHUB_REPOSITORY"]
        run, published_run = restore(cache, os.environ["MATRIX_CHECKPOINT"], os.environ["DEFAULT_BRANCH"], repo, int(os.environ["GITHUB_RUN_ID"]))
        message = (f"Sampler history restored from [run {run}](https://github.com/{repo}/actions/runs/{run})."
                   if run is not None else "Sampler history starts empty. No compatible trusted checkpoint exists.")
        if published_run is not None:
            message += f" Run {published_run} already published this cohort at or after this run; this attempt is read-only."
        output = f"journal_start={journal_size(cache)}\ncan_publish={'true' if published_run is None else 'false'}\nrestored={'true' if run is not None else 'false'}"
    else:
        count = snapshot(cache, args.target, args.logs, args.journal_start)
        message = f"Matrix checkpoint contains {count} new external observations." if count else "No new external observations to checkpoint."
        output = f"observations={count}"
    print(message)
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as stream:
        print(message, file=stream)
    with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
        print(output, file=stream)
