# SPDX-License-Identifier: MIT OR Apache-2.0
"""Compare two fixed mention budgets with one FCoref checkpoint on disjoint GAP articles."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import tempfile
import time

from coref_gap import DATA_SHA, FC_FILES, FC_REV, check_file, measure
from native_gap_possessives import VALIDATION_SHA, read_rows

DEVELOPMENT_SHA = "b9a01434fcf58d8c2f9bc762480c27e58ce466cf1ffe8b09cfecbc7a20d2d634"


def cohort(dev, test, validation):
    for path, digest in ((dev, DEVELOPMENT_SHA), (test, DATA_SHA), (validation, VALIDATION_SHA)):
        check_file(path, digest)
    prior = read_rows(test) + read_rows(validation)
    articles = {row["URL"] for row in prior}
    passages = {row["Text"] for row in prior}

    def order(row):
        return hashlib.sha256(("anno-fcoref-budget-v1/" + row["ID"]).encode()).digest()

    docs = []
    for row in sorted(read_rows(dev), key=order):
        text = row["Text"]
        if row["URL"] in articles or text in passages:
            continue
        targets = []
        for field in ("Pronoun", "A", "B"):
            start = int(row[field + "-offset"])
            end = start + len(row[field])
            if text[start:end] != row[field]:
                raise ValueError("GAP target disagrees with input text")
            targets.append((start, end))
        labels = [row[f + "-coref"] for f in ("A", "B")]
        if not set(labels) <= {"TRUE", "FALSE"}:
            raise ValueError("invalid GAP decision")
        docs.append((row["ID"], text, targets, [v == "TRUE" for v in labels]))
        articles.add(row["URL"])
        passages.add(text)
        if len(docs) == 128:
            return docs
    raise ValueError("fewer than 128 eligible articles")


def run_arm(binary, model, docs):
    env = dict(os.environ, FCOREF_MODEL_PATH=str(model), ANNO_NO_DOWNLOADS="1",
               HF_HUB_OFFLINE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    # One process per arm. The first response includes initialization.
    latencies, rows = [], []
    start = time.perf_counter()
    with subprocess.Popen([binary, "fcoref"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          text=True, encoding="utf-8", env=env) as proc:
        try:
            for doc_id, text, _, _ in docs:
                proc.stdin.write(json.dumps(dict(id=doc_id, text=text)) + "\n")
                proc.stdin.flush()
                rows.append(json.loads(proc.stdout.readline()))
                end = time.perf_counter()
                latencies.append(1000 * (end - start))
                start = end
            proc.stdin.close()
            if proc.stdout.read() or proc.wait() != 0:
                raise ValueError("unexpected FCoref process output or exit")
        finally:
            if proc.poll() is None:
                proc.kill()
    result = measure(docs, rows, "fcoref")
    result["warm_p95_ms"] = round(sorted(latencies[1:])[math.ceil(.95 * (len(latencies) - 1)) - 1], 2)
    return result


def compare(docs, old, new):
    deltas, gained, lost, extra, removed = [], 0, 0, 0, 0
    for doc, a, b in zip(docs, old["per_id"], new["per_id"], strict=True):
        truth = doc[3]
        baseline = [v == "1" for v in a[3:]]
        candidate = [v == "1" for v in b[3:]]
        deltas.append(int(candidate == truth) - int(baseline == truth))
        for gold, before, after in zip(truth, baseline, candidate, strict=True):
            gained += gold and after and not before
            lost += gold and before and not after
            extra += not gold and after and not before
            removed += not gold and before and not after
    rng = random.Random(288)
    draws = sorted(statistics.mean(rng.choices(deltas, k=len(deltas))) for _ in range(4000))
    return dict(gold_links_recovered=gained, gold_links_lost=lost,
                unmatched_links_added=extra, unmatched_links_removed=removed,
                exact_document_wins=deltas.count(1), exact_document_losses=deltas.count(-1),
                exact_document_delta_ci95=[draws[99], draws[3899]])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("development", "test", "validation", "model", "binary"):
        p.add_argument(name, type=Path)
    args = p.parse_args()
    docs = cohort(args.development, args.test, args.validation)
    # #288 used a separate verified FP32 export of the same source checkpoint.
    with (args.model / "encoder.onnx").open("rb") as source:
        encoder_sha = hashlib.file_digest(source, "sha256").hexdigest()
    files = FC_FILES
    if encoder_sha == "190101f8e3b927bc7fb5e053f63fb82a85ef7e92dc2acd3a23a5ad0dd58553f4":
        files = {**FC_FILES, "encoder.onnx": encoder_sha,
                 "config.json": "2d96a29d2ebd497d34580594f40ded241440411bcd83885b89d8fbb86189bc3b"}
    for file, digest in files.items():
        check_file(args.model / file, digest)
    base_config = json.loads((args.model / "config.json").read_text())
    arms = []
    with tempfile.TemporaryDirectory() as temp:
        for fraction in (.25, .5):
            model = Path(temp) / str(fraction)
            model.mkdir()
            for name in FC_FILES.keys() - {"config.json"}:
                (model / name).symlink_to((args.model / name).resolve())
            base_config["coref_head"]["top_lambda"] = fraction
            (model / "config.json").write_text(json.dumps(base_config))
            arms.append(run_arm(args.binary.resolve(), model, docs))
    with args.binary.open("rb") as stream:
        binary_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    out = dict(checkpoint_revision=FC_REV, model_files=files, development_sha256=DEVELOPMENT_SHA,
               binary_sha256=binary_sha,
               ordered_ids_sha256=hashlib.sha256("\n".join(d[0] for d in docs).encode()).hexdigest(),
               comparison=compare(docs, *arms))
    out["arms"] = [{"top_lambda": v, "counts": a["counts"], "warm_p95_ms": a["warm_p95_ms"]}
                   for v, a in zip((.25, .5), arms, strict=True)]
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
