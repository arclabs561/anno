"""Replay the CLI's default stack with only the pinned checkpoints available."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from run import ORT_SHA256, digest, verify_assets, verify_runtime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nuner_fallback_routing import BERT_CHECKPOINT, BERT_FILES, CHECKPOINT, FILES, text_sha


def stacked_assets(bert=True):
    checkpoints = [(CHECKPOINT, FILES)]
    if bert:
        checkpoints.append((BERT_CHECKPOINT, BERT_FILES))
    for checkpoint, hashes in checkpoints:
        repo, revision = checkpoint.split("@")
        yield repo.split("/")[-1], {
            "repo": repo, "revision": revision,
            "files": {name: {"remote": name, "sha256": sha} for name, sha in hashes.items()},
        }


@contextmanager
def offline_stack(models, *, bert, fetch=False):
    """Give StackedNER::default a fresh Hub cache, including only verified bytes."""
    with TemporaryDirectory(prefix="anno-stacked-hub-") as home:
        for name, spec in stacked_assets(bert):
            repo, revision = spec["repo"], spec["revision"]
            directory = models / name
            verify_assets(directory, spec, fetch)
            root = Path(home) / "hub" / ("models--" + repo.replace("/", "--"))
            snapshot = root / "snapshots" / revision
            snapshot.mkdir(parents=True)
            (root / "refs").mkdir()
            (root / "refs/main").write_text(revision)
            for name in spec["files"]:
                (snapshot / name).symlink_to((directory / name).resolve())
        hub = str(Path(home) / "hub")
        yield {**os.environ, "HF_HOME": home, "HF_HUB_CACHE": hub,
               "HUGGINGFACE_HUB_CACHE": hub, "ANNO_NO_DOWNLOADS": "1",
               "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}


def observe(worker, texts, bert, env, timeout=180, *, fallback=False):
    command = [str(worker.resolve()), "--default-with-bert" if bert else "--default-no-bert"]
    if fallback:
        command.append("--nuner-fallback")
    result = subprocess.run(command, env=env, input="".join(json.dumps(t) + "\n" for t in texts),
                            text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"Default stack exited {result.returncode}: {result.stderr}")
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    for text, row in zip(texts, rows, strict=True):
        if row["input_sha256"] != text_sha(text):
            raise ValueError("Different input digest or order")
        if len(row["spans"]) != len({tuple(s) for s in row["spans"]}):
            raise ValueError("Duplicate final typed span")
        expected = [["bert-onnx", "succeeded"]] if bert else []
        expected += [["nuner", "succeeded" if row["ran"] else "skipped"],
                     ["regex", "succeeded"], ["heuristic", "succeeded"]]
        if row["layers"] != expected:
            raise ValueError(f"Unexpected default stack: {row['layers']}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--aggregate", action="store_true", help="Write counts for a shared receipt")
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    verify_runtime(args.worker)
    verify_runtime(args.cli)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    cases = json.loads(Path(__file__).with_name("stacked-cases.json").read_text())
    report = {"revision": subprocess.check_output(
        ["git", "-C", Path(__file__).parent, "rev-parse", "HEAD"], text=True).strip(),
        "worker_sha256": digest(args.worker), "cli_sha256": digest(args.cli), "ort_sha256": ORT_SHA256,
        "nuner": CHECKPOINT, "bert": BERT_CHECKPOINT, "runs": {}}
    for bert in (False, True):
        name = "with_bert" if bert else "without_bert"
        with offline_stack(args.models, bert=bert, fetch=args.fetch) as env:
            rows = observe(args.worker, [c["text"] for c in cases], bert, env)
            for case, row in zip(cases, rows, strict=True):
                if row["ran"] != case["nuner_ran"] or sorted(row["spans"]) != sorted(case["spans"]):
                    raise ValueError(f"Default stack differs for {name}: {row}")
            fallback_rows = observe(args.worker, [c["text"] for c in cases], bert, env, fallback=True)
            for row, opt_in in zip(rows, fallback_rows, strict=True):
                if opt_in["ran"] != (row["ran"] or not bert):
                    raise ValueError(f"Unexpected NuNER fallback decision for {name}: {opt_in}")
                if opt_in["ran"] == row["ran"] and opt_in["spans"] != row["spans"]:
                    raise ValueError(f"NuNER fallback changed an identically routed input: {opt_in}")
            # Exercise the ordinary CLI entry point as well as the detailed layer observer.
            case = cases[0]
            command = [args.cli.resolve(), "extract", "--quiet", "--format", "json", "--text", case["text"]]
            record = json.loads(subprocess.check_output(command, env=env, text=True, timeout=180))
            runtime = "stacked(" + "+".join(layer for layer, _ in rows[0]["layers"]) + ")"
            if record["provenance"]["runtime_model"] != runtime:
                raise ValueError(f"Default CLI runtime differs from the layer observer: {runtime}")
            spans = []
            for entity in record["entities"]:
                start, end = entity["start"], entity["end"]
                if not 0 <= start < end <= len(case["text"]) or case["text"][start:end] != entity["text"]:
                    raise ValueError(f"CLI surface and character offsets disagree: {entity}")
                spans.append([start, end, entity["type"]])
            if sorted(spans) != sorted(case["spans"]):
                raise ValueError(f"Default CLI spans differ for {name}: {spans}")

        if args.aggregate:
            report["runs"][name] = {
                "cases_completed": len(rows), "stock_nuner_calls": sum(row["ran"] for row in rows),
                "fallback_nuner_calls": sum(row["ran"] for row in fallback_rows),
                "stock_predicted_entities": sum(len(row["spans"]) for row in rows),
                "fallback_predicted_entities": sum(len(row["spans"]) for row in fallback_rows),
                "cli_predicted_entities": len(spans),
            }
        else:
            report["runs"][name] = {"layers": rows, "fallback_layers": fallback_rows,
                                   "cli_runtime": runtime, "cli_spans": spans}
    args.receipt.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
