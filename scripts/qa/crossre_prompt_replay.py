#!/usr/bin/env python3
"""Replay matched full-domain CrossRE prompt conditions with cached Anno models (Linux)."""

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

REVISION = "a58885fa760559d2dc9e176e730fe731f020d6f3"
DOMAINS = {
    "ai": "CrossREAI",
    "literature": "CrossRELiterature",
    "music": "CrossRE",
    "news": "CrossRE",
    "politics": "CrossRE",
    "science": "CrossRE",
}
MANIFEST = Path(__file__).resolve().parents[2] / "docs/eval/dataset-repo-files.csv"
RELEX = "knowledgator/gliner-relex-large-v1.0"
RELEX_ARTIFACTS = {
    "graph": "084b3873084a4173fc443c8bca691aabdf9f4357972fe032766bf6ec0c76a689",
    "tokenizer": "9dd6016a277d88072abe003d66f29e01f0ae7ab6b5be519677559dc21a5f6803",
    "config": "9ff363d306ba1edcb830343b3b79299278deb5037081d8097d06fe08a63d0e3c",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_split(path, expected):
    data = path.read_bytes()
    if digest(data) != expected:
        raise ValueError(f"unexpected file digest for {path}")
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare(data_dir, out, domain):
    dataset = DOMAINS[domain]
    with MANIFEST.open(newline="", encoding="utf-8") as source:
        rows = [
            row for row in csv.DictReader(source)
            if row["id"] == "CrossRE" and row["subset"] == domain
            and row["repository"] == "mainlp/CrossRE" and row["revision"] == REVISION
            and row["status"] == "ok" and row["split"] in ("train", "test")
        ]
    if sorted(row["split"] for row in rows) != ["test", "train"]:
        raise ValueError(f"Expected one pinned training file and one test file for {domain}")
    files = {row["split"]: row for row in rows}
    count = int(files["test"]["rows"])
    train_hash, test_hash = (files[split]["sha256"] for split in ("train", "test"))
    train = read_split(data_dir / f"{domain}-train.json", train_hash)
    test_path = data_dir / f"{domain}-test.json"
    test = read_split(test_path, test_hash)
    if len(train) != int(files["train"]["rows"]) or len(test) != count:
        raise ValueError(f"Unexpected train/test sentence counts for {domain}")
    schema = {
        "entity_types": sorted({e[2] for row in train for e in row["ner"]}),
        "relation_types": sorted({r[4] for row in train for r in row["relations"]}),
    }
    schema_path = out / f"{domain}-schema.json"
    write_json(schema_path, schema)
    corpus_cache = out / "cache" / domain / "datasets"
    corpus_cache.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(test_path, corpus_cache / f"{dataset}.cache")
    train_texts = {tuple(d["sentence"]) for d in train}
    test_texts = {tuple(d["sentence"]) for d in test}
    missing_relations = Counter()
    missing_endpoints = Counter()
    outside = 0
    for row in test:
        entity_labels = {tuple(span[:2]): span[2] for span in row["ner"]}
        for head_start, head_end, tail_start, tail_end, label, *_ in row["relations"]:
            endpoints = {entity_labels.get((head_start, head_end), "ENTITY"),
                         entity_labels.get((tail_start, tail_end), "ENTITY")}
            unknown_endpoints = endpoints.difference(schema["entity_types"])
            unknown_relation = label not in schema["relation_types"]
            missing_endpoints.update(unknown_endpoints)
            if unknown_relation:
                missing_relations[label] += 1
            outside += bool(unknown_endpoints or unknown_relation)
    return {
        "relations_outside_train_schema": outside,
        "unseen_relation_label_counts": dict(missing_relations),
        "relations_by_unseen_endpoint_type": dict(missing_endpoints),
        "dataset": dataset, "domain": domain, "train_docs": len(train), "test_docs": count,
        "train_sha256": train_hash, "test_sha256": test_hash, "schema": schema,
        "schema_sha256": digest(schema_path.read_bytes()),
        "train_test_shared_token_sequences": len(train_texts & test_texts),
        "test_unique_token_sequences": len(test_texts),
        "repeated_relations_in_sentence": sum(
            len(d["relations"]) - len({tuple(r[:5]) for r in d["relations"]}) for d in test
        ),
        "test_entities": sum(len(d["ner"]) for d in test),
        "test_relations": sum(len(d["relations"]) for d in test),
        "gold_empty_docs": sum(not d["relations"] for d in test),
    }


def measure(argv, env, cpus, output, timeout):
    # wait4 yields CPU time and peak RSS for this arm, independently of preceding arms.
    started = time.monotonic()
    with output.open("wb") as log:
        with subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT,
                              preexec_fn=lambda: os.sched_setaffinity(0, cpus)) as proc:
            while True:
                child, status, usage = os.wait4(proc.pid, os.WNOHANG)
                if child:
                    proc.returncode = os.waitstatus_to_exitcode(status)
                    break
                if time.monotonic() - started > timeout:
                    proc.kill()
                    proc.wait()
                    raise TimeoutError(f"{output} exceeded {timeout}s")
                time.sleep(0.05)
            if proc.returncode:
                raise RuntimeError(f"benchmark exited {proc.returncode}, see {output}")
    return {"wall_s": time.monotonic() - started, "user_cpu_s": usage.ru_utime,
            "sys_cpu_s": usage.ru_stime, "max_rss_kib": usage.ru_maxrss}


def validate_row(row, corpus, backend, policy):
    if not row["success"]:
        raise ValueError(row.get("error") or "benchmark did not succeed")
    metrics = row["metrics"]
    source_file = row["provenance"]["dataset"]
    if (row["dataset"] != corpus["dataset"] or source_file["data_source"] != "local cache"
            or source_file["content_sha256"] != corpus["test_sha256"]):
        raise ValueError("The scored dataset does not identify the pinned domain test file")
    sample = row["provenance"]["evaluated_sample"]
    count = corpus["test_docs"]
    if sample["unit_count"] != count or sample["gold_item_count"] != corpus["test_relations"]:
        raise ValueError("The sample counts disagree with the pinned test split")
    expected = {
        "num_docs": count, "num_gold_relations": corpus["test_relations"],
        "gold_endpoint_docs": 0, "predicted_endpoint_docs": count,
        "gold_empty_docs": corpus["gold_empty_docs"],
        "fixed_label_prompt_docs": count if policy == "fixed" else 0,
        "gold_label_prompt_docs": count if policy == "oracle" else 0,
    }
    if policy == "fixed":
        expected["gold_relations_outside_fixed_schema"] = corpus["relations_outside_train_schema"]
    for key, value in expected.items():
        if metrics[key] != value:
            raise ValueError(f"{key}: expected {value}, got {metrics[key]}")
    source = RELEX if backend == "gliner_relex" else "tplinker-heuristic"
    if sample["relation_producer_documents"] != {source: count}:
        raise ValueError(f"unexpected effective producer: {sample}")
    runtime = row["provenance"]["runtime"]
    prompt = runtime["relation_prompt"]
    expected_prompt = {"source": "fixed", **corpus["schema"]} if policy == "fixed" else {"source": "document_gold"}
    if prompt != expected_prompt or runtime["relation_threshold"] != 0.5:
        raise ValueError("unexpected prompt or confidence threshold")
    if backend == "gliner_relex":
        files = row["provenance"]["backend"]["model_artifacts"]["files"]
        hashes = {f["role"]: f["sha256"].get("value") for f in files}
        if any(hashes.get(role) != expected for role, expected in RELEX_ARTIFACTS.items()):
            raise ValueError("unexpected Relex graph, tokenizer or config")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--anno", required=True, type=Path, help="Anno CLI built with eval and onnx")
    p.add_argument("--data", required=True, type=Path, help="directory with pinned DOMAIN-{train,test}.json")
    p.add_argument("--out", required=True, type=Path, help="new directory for receipts and dataset cache")
    domains = p.add_mutually_exclusive_group()
    domains.add_argument("--domain", choices=DOMAINS, action="append",
                         help="repeat to select domains (default: ai and literature)")
    domains.add_argument("--all-domains", action="store_true", help="run all six publisher domains")
    p.add_argument("--backend", choices=["tplinker", "gliner_relex"], action="append",
                   help="repeat to select backends (default: both)")
    p.add_argument("--cpus", required=True, help="allowed Linux CPU numbers separated by commas")
    p.add_argument("--arm-timeout", type=int, default=3900, help="maximum seconds per arm")
    args = p.parse_args()
    cpus = {int(n) for n in args.cpus.split(",")}
    if not cpus or not cpus <= os.sched_getaffinity(0):
        p.error("--cpus must be contained in this process's CPU affinity")
    if args.arm_timeout <= 0:
        p.error("--arm-timeout must be positive")
    args.out.mkdir(parents=True, exist_ok=False)
    binary = args.anno.resolve()
    env = {**os.environ, "ANNO_NO_DOWNLOADS": "1", "ANNO_RELATION_ORACLE_ENTITIES": "0"}
    report = {"crossre_revision": REVISION, "binary_sha256": digest(binary.read_bytes()),
              "relex_artifacts_sha256": RELEX_ARTIFACTS, "cpus": sorted(cpus), "seed": 42,
              "sampling": "complete test split in release order", "relation_threshold": 0.5,
              "arm_timeout_seconds": args.arm_timeout, "domains": {}}
    selected_domains = DOMAINS if args.all_domains else (args.domain or ("ai", "literature"))
    for domain in selected_domains:
        corpus = prepare(args.data, args.out, domain)
        env["ANNO_CACHE_DIR"] = str((args.out / "cache" / domain).resolve())
        rows = []
        seen = set()
        for backend in args.backend or ("tplinker", "gliner_relex"):
            arm_env = env
            if backend == "tplinker":
                # A private model cache selects the study's heuristic control.
                arm_env = {**env, "XDG_CACHE_HOME": env["ANNO_CACHE_DIR"]}
            for policy in ("fixed", "oracle"):
                base = args.out / f"{domain}-{backend}-{policy}"
                argv = [str(binary), "benchmark", "--tasks", "relation", "--datasets", corpus["dataset"],
                        "--backends", backend, "--max-examples", str(corpus["test_docs"]), "--seed", "42",
                        "--cached-only", "--output", str(base.with_suffix(".md")),
                        "--output-json", str(base.with_suffix(".json"))]
                if policy == "fixed":
                    argv.extend(["--relation-schema", str(args.out / f"{domain}-schema.json")])
                print(f"Running {domain} {backend} {policy}", flush=True)
                cost = measure(argv, arm_env, cpus, base.with_suffix(".log"), args.arm_timeout)
                native = json.loads(base.with_suffix(".json").read_text())["results"][0]
                validate_row(native, corpus, backend, policy)
                sample = native["provenance"]["evaluated_sample"]
                seen.add(sample["sample_sha256"])
                if len(seen) != 1:
                    raise ValueError(f"benchmark arms used different {domain} samples")
                rows.append({"backend": backend, "prompt": policy, "sample": sample,
                             "input_sha256": native["provenance"]["dataset"]["content_sha256"],
                             "data_source": native["provenance"]["dataset"]["data_source"],
                             "metrics": native["metrics"], "cost": cost})
                report["domains"][domain] = {"corpus": corpus, "arms": rows}
                write_json(args.out / "summary.json", report)


if __name__ == "__main__":
    main()
