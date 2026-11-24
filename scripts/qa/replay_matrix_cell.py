"""Replay one resolved CI evaluation using a fresh policy history and local datasets."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from replay_matrix_feedback import ROOT, TEST, sampler_env

FIXTURES = {
    "AnnoNerSmoke": "matrix-smoke.conll",
    "AnnoTimexSmoke": "matrix-smoke-timex.jsonl",
    "AnnoTimexContrast": "timex-contrast/rows.jsonl",
}


def replay_args(row):
    """Only emit shell-safe identifiers and bounded typed values from the outcome."""
    fields = [row.get(key) for key in ("task", "dataset", "backend")]
    if not all(isinstance(v, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", v) for v in fields):
        return None
    seed, limit = row.get("seed"), row.get("max_examples")
    if type(seed) is not int or not 0 <= seed < 2**64 or type(limit) is not int or not 0 < limit < 2**64:
        return None
    args = [*fields, "--seed", str(seed), "--max-examples", str(limit)]
    digest = row.get("dataset_sha256")
    if digest is not None:
        if not isinstance(digest, str) or not re.fullmatch("[a-f0-9]{64}", digest):
            return None
        args += ["--dataset-sha256", digest]
    for key, flag in (("coref_gold", "--gold-mentions"), ("coref_english_pronouns", "--english-pronouns")):
        if type(row.get(key)) is not bool:
            return None
        if row[key]:
            args.append(flag)
    return args


def replay_command(row):
    args = replay_args(row)
    return "python3 scripts/qa/replay_matrix_cell.py " + " ".join(args) if args else None


def replay(args):
    wanted = dict(task=args.task, dataset=args.dataset, backend=args.backend, seed=args.seed,
                  max_examples=args.max_examples, dataset_sha256=args.dataset_sha256,
                  coref_gold=args.gold_mentions, coref_english_pronouns=args.english_pronouns)
    command = replay_command(wanted)
    if not command:
        raise ValueError("invalid cell identifier, seed, sample limit or dataset SHA-256")
    if args.summary:
        with args.summary.open("a", encoding="utf-8") as summary:
            print(f"Replay this cell: `{command}`", file=summary)
    with tempfile.TemporaryDirectory(prefix="anno-matrix-cell-") as temp:
        cache = Path(temp)
        datasets = cache / "datasets"
        if args.dataset in FIXTURES:
            datasets.mkdir()
            source = ROOT / "scripts/qa/fixtures" / FIXTURES[args.dataset]
            shutil.copyfile(source, datasets / f"{args.dataset}.cache")
        else:
            if not args.cache_dir or not (args.cache_dir / "datasets").is_dir():
                raise ValueError("supply an authorized local dataset cache via --cache-dir or ANNO_CACHE_DIR")
            datasets.symlink_to((args.cache_dir / "datasets").resolve(), target_is_directory=True)
        log = cache / "outcomes.jsonl"
        env = sampler_env(cache, "random", log, fixed=args.dataset, seed=args.seed)
        env.update(ANNO_MATRIX_TASK=args.task,
                   ANNO_MUXER_PIN_BACKEND=args.backend, ANNO_ML_IN_MATRIX="1",
                   ANNO_MAX_EXAMPLES=str(args.max_examples), ANNO_COREF_GOLD=str(int(args.gold_mentions)),
                   ANNO_COREF_ENGLISH_PRONOUNS=str(int(args.english_pronouns)),
                   HF_HUB_OFFLINE="1")
        subprocess.run(["cargo", "test", "--locked", "-p", "anno-eval", "--lib", "--features",
                        "onnx,eval,discourse", TEST, "--", "--exact", "--ignored", "--nocapture"],
                       cwd=ROOT, env=env, check=True)
        rows = [json.loads(line) for line in log.read_text().splitlines()] if log.is_file() else []
        outcomes = [r for r in rows if r.get("record_type") == "outcome"]
        if not outcomes:
            raise ValueError("no outcome for this cell; check the dataset, backend assets and features locally")
        if len(outcomes) != 1:
            raise ValueError("fixed replay produced more than one outcome")
        row = outcomes[0]
        if any(row.get(key) != value for key, value in wanted.items() if value is not None):
            raise ValueError("replay did not match the requested cell, sample controls or input digest")
        if args.output:
            with args.output.open("xb") as output:
                output.write(log.read_bytes())
        print(f"{args.task} / {args.dataset} / {args.backend}: examples={row['cost_units']} "
              f"primary_f1={row.get('primary_f1')} hard_failure={row['hard_junk']}")
        return 1 if row["hard_junk"] or not row["cost_units"] else 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("task", "dataset", "backend"):
        p.add_argument(name)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--max-examples", type=int, required=True)
    p.add_argument("--dataset-sha256", help="require the loader-reported dataset digest")
    p.add_argument("--gold-mentions", action="store_true")
    p.add_argument("--english-pronouns", action="store_true")
    p.add_argument("--cache-dir", type=Path, default=os.getenv("ANNO_CACHE_DIR"))
    p.add_argument("--output", type=Path, help="write a fresh JSONL receipt")
    p.add_argument("--summary", type=Path, help="append the portable command before starting the evaluation")
    try:
        sys.exit(replay(p.parse_args()))
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"Matrix replay unavailable: {exc}", file=sys.stderr)
        sys.exit(2)
