"""Compare independent, real data corruptions under matched native sampler budgets."""

import argparse
import hashlib
import csv
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory

FLAGS = {
    "coverage": ("--mode", "coverage"),
    "triage": ("--mode", "triage"),
    "random": ("--strategy", "random"),
}
PINS = {
    "WikiGold": "c797a64d0cf73ed058363f77671486bcfd413e70fda1726ad4a6ba624455225f",
    "Wnut17": "2aa79b764e56ec9264a1b30fdd9b70195bd00ff400b62edd8f399d5f13c178f0",
}
STRATEGIES = {"coverage": "estimate", "triage": "worst_first", "random": "random"}


def sha(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def fingerprint(cache):
    digest = hashlib.sha256()
    for path in sorted(cache.rglob("*")):
        if path.is_file():
            digest.update(
                str(path.relative_to(cache)).encode() + b"\x00" + bytes.fromhex(sha(path))
            )
    return digest.hexdigest()


def sampler_env(cache):
    env = {k: v for k, v in os.environ.items() if not k.startswith("ANNO_")}
    env.update(
        ANNO_CACHE_DIR=str(cache),
        ANNO_EVAL_HISTORY=str(cache / "eval-results.jsonl"),
        ANNO_ML_IN_MATRIX="0",
        ANNO_S3_CACHE="0",
        ANNO_NO_DOWNLOADS="1",
        ANNO_MATRIX_INCLUDE_NON_AUTOMATABLE="1",
        ANNO_MATRIX_TRY_DOWNLOAD_ON_EMPTY="0",
    )
    return env


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def run(args, cache, policy, seed, slots, fixed=None):
    decisions = cache / "arm-decisions.jsonl"
    history = cache / "eval-results.jsonl"
    offset = len(read_jsonl(history)) if history.exists() else 0
    command = [
        args.anno,
        "sampler",
        *FLAGS[policy],
        "run",
        "--runs",
        str(slots),
        "--seed-base",
        str(seed),
        "--task",
        "ner",
        "--require-cached",
        "--per-dataset",
        "--pin-backend",
        "stacked",
        "--datasets-per-run",
        "1",
        "--backends-per-run",
        "1",
        "--max-examples",
        "20",
        "--decisions-file",
        decisions,
    ]
    if fixed:
        command.extend(["--fixed-datasets", fixed])
    proc = subprocess.run(
        command, cwd=args.repo, env=sampler_env(cache), capture_output=True, text=True
    )
    if proc.returncode:
        raise RuntimeError(f"sampler failed for seed {seed}: {proc.stderr[-2000:]}")
    receipts = read_jsonl(decisions)
    outcomes = [r for r in receipts if r.get("record_type") == "outcome"]
    measures = read_jsonl(history)[offset:]
    assert len(outcomes) == len(measures) == slots
    rows = []
    for index, (outcome, measure) in enumerate(zip(outcomes, measures, strict=True)):
        metadata = json.loads(measure["metadata"])
        provenance = metadata["provenance"]
        assert measure["observation_id"] == outcome["observation_id"]
        assert measure["seed"] == seed + index and outcome["dataset"] in PINS
        assert measure["n"] == outcome["cost_units"] == 20
        assert metadata["success"] and measure.get("error") is None
        assert (
            measure["backend"]
            == outcome["backend"]
            == provenance["backend"]["effective"]
            == "stacked"
        )
        assert (
            provenance["dataset"]["data_source"] == "local cache"
            and provenance["runtime"]["cached_only"]
        )
        assert outcome["primary_f1"] == measure["f1"]
        assert outcome["strategy"].replace("-", "_") == STRATEGIES[policy]
        decisions_for_slot = [
            r
            for r in receipts
            if r.get("run_id") == outcome["run_id"] and r.get("chosen") == "stacked"
        ]
        if decisions_for_slot:
            assert decisions_for_slot[-1]["datasets"] == [outcome["dataset"]]
        else:
            assert policy == "random"
        rows.append(
            {
                "seed": measure["seed"],
                "slot": index + 1,
                "dataset": outcome["dataset"],
                "examples": measure["n"],
                "primary_f1": measure["f1"],
                "gold_entities": provenance["dataset"]["entity_count"],
                "observation_id": measure["observation_id"],
            }
        )
    return rows


BASES = range(140000, 140080, 10)
FAULTS = {"WikiGold": b"LOC", "Wnut17": b"person"}
ROOT = Path(__file__).resolve().parents[2]


def damage(cache, dataset):
    path = cache / "datasets" / (dataset + ".cache")
    before = path.read_bytes()
    assert sha(path) == PINS[dataset]
    # Anchor the final tag and retain tabs, spaces, newlines and all token bytes.
    regex = rb"(?m)([ \t]+)[BI]-" + FAULTS[dataset] + rb"(?=\r?$)"
    after, n = re.subn(regex, rb"\g<1>O", before)
    assert n > 0 and len(after.splitlines()) == len(before.splitlines())
    path.write_bytes(after)
    return {"changed_tags": n, "mutated_sha256": sha(path)}


def preflight(args, cache):
    output = subprocess.check_output(
        [args.anno, "sampler", "eligibility", "--task", "ner", "--require-cached"],
        env=sampler_env(cache),
        cwd=args.repo,
        text=True,
    )
    report = json.loads(output)
    pool = [r["dataset"] for r in report["datasets"] if r["eligible"]]
    assert sorted(pool) == sorted(PINS) and report["eligible_count"] == 2
    return pool


def to_csv(path, rows):
    with path.open("x", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def study(args):
    args.out.mkdir(parents=True, exist_ok=False)
    events, trials, reference, warm_rows, mutations = [], [], {}, [], {}
    with TemporaryDirectory(prefix="anno-fault-trials-") as work:
        root = Path(work)
        cold = root / "cold"
        (cold / "datasets").mkdir(parents=True)
        for dataset in FAULTS:
            source = args.cache / (dataset + ".cache")
            assert sha(source) == PINS[dataset]
            shutil.copyfile(source, cold / "datasets" / source.name)
        for dataset in FAULTS:
            template = root / dataset
            shutil.copytree(cold, template)
            warm = run(args, template, "random", 11111, 1, dataset)
            warm_rows.extend(warm)
            for base in BASES:
                control = root / "control"
                shutil.copytree(cold, control)
                records = run(args, control, "random", base, 3, dataset)
                reference.update({(r["seed"], dataset): r for r in records})
                shutil.rmtree(control)
        for fault in FAULTS:
            for prior in ("cold", *FAULTS):
                template = root / "mutant"
                shutil.copytree(root / prior, template)
                mutations[fault] = damage(template, fault)
                for index, base in enumerate(BASES):
                    policies = list(FLAGS)
                    for policy in policies[index % 3 :] + policies[: index % 3]:
                        trial = root / "trial"
                        shutil.copytree(template, trial)
                        digest = fingerprint(trial)
                        assert digest == fingerprint(template)
                        pool = preflight(args, trial)
                        rows = run(args, trial, policy, base, 3)
                        alarms = []
                        for row in rows:
                            expected = reference[(row["seed"], row["dataset"])]
                            delta = row["gold_entities"] - expected["gold_entities"]
                            if row["dataset"] != fault:
                                assert delta == 0 and row["primary_f1"] == expected["primary_f1"]
                            alarms.append(delta < 0)
                            events.append(
                                dict(
                                    fault=fault,
                                    prior=prior,
                                    base=base,
                                    policy=policy,
                                    gold_delta=delta,
                                    clean_f1=expected["primary_f1"],
                                    alarm=delta < 0,
                                    **row,
                                )
                            )
                        trials.append(
                            dict(
                                fault=fault,
                                prior=prior,
                                policy=policy,
                                base=base,
                                first=next((i + 1 for i, on in enumerate(alarms) if on), 4),
                                pool=",".join(pool),
                                initial_sha256=digest,
                            )
                        )
                        shutil.rmtree(trial)
                shutil.rmtree(template)
    ids = {row["observation_id"] for row in events}
    assert len(events) == len(ids) == 432
    assert len(reference) == 48 and len(warm_rows) == 2 and len(trials) == 144
    summary = {}
    for subset in ["all", "cold", "warm_fault", "warm_healthy"]:
        for policy in FLAGS:
            chosen = [
                r["first"]
                for r in trials
                if r["policy"] == policy
                and (
                    subset == "all"
                    or subset == "cold"
                    and r["prior"] == "cold"
                    or subset == "warm_fault"
                    and r["prior"] == r["fault"]
                    or subset == "warm_healthy"
                    and r["prior"] not in ("cold", r["fault"])
                )
            ]
            summary[subset + "/" + policy] = {
                "arms": len(chosen),
                "by_slot": [sum(v <= slot for v in chosen) for slot in (1, 2, 3)],
                "restricted_mean_slots": round(sum(chosen) / len(chosen), 4),
            }
    report = {
        "policy_selections": len(events),
        "controls": len(reference),
        "warm_ups": len(warm_rows),
        "examples_per_selection": 20,
        "arms": len(trials),
        "mutations": mutations,
        "native_binary": sha(args.anno),
        "fault_draws": sum(r["fault"] == r["dataset"] for r in events),
        "alarms": sum(r["alarm"] for r in events),
        "gold_increased": sum(r["gold_delta"] > 0 for r in events),
        "false_alarms": sum(r["alarm"] and r["dataset"] != r["fault"] for r in events),
        "summary": summary,
    }
    for file, rows in [
        ("events.csv", events),
        ("arms.csv", trials),
        ("controls.csv", reference.values()),
        ("warm.csv", warm_rows),
    ]:
        to_csv(args.out / file, list(rows))
    (args.out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, default=ROOT)
    for name in ("anno", "cache", "out"):
        p.add_argument("--" + name, type=lambda p: Path(p).resolve(), required=True)
    study(p.parse_args())
