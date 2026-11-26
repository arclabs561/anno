# SPDX-License-Identifier: MIT OR Apache-2.0
"""Replay a fixed-budget FCoref diagnostic with externally supplied candidate boundaries."""
import argparse
from collections import Counter
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import subprocess

from coref_gap import check_file, decode_clusters, measure
from fcoref_emitted_boundaries import MODEL_FILES, panel
from fcoref_proposal_budget import compare
from gap_receipts import digest


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "docs/eval/fcoref-emitted-boundaries.json"


def documents(rows):
    docs = []
    for row in rows:
        text = row["Text"]
        spans = []
        for name in ("Pronoun", "A", "B"):
            start = int(row[name + "-offset"])
            end = start + len(row[name])
            if not 0 <= start < end <= len(text) or text[start:end] != row[name]:
                raise ValueError("GAP source offset mismatch")
            spans.append((start, end))
        if len(set(spans)) != 3:
            raise ValueError("Expected three distinct marked spans")
        flags = [row[key + "-coref"] for key in ("A", "B")]
        if not set(flags) <= {"TRUE", "FALSE"}:
            raise ValueError("Invalid GAP link label")
        docs.append((row["ID"], text, spans, [v == "TRUE" for v in flags]))
    return docs


def run_arm(binary, env, docs, reserve):
    inputs = []
    for doc_id, text, spans, _ in docs:
        # Only source offsets reach the inference process; link labels remain in the scorer.
        marks = [[len(text[:a].encode("utf-8")), len(text[:b].encode("utf-8"))]
                 for a, b in spans] if reserve else []
        inputs.append(json.dumps(dict(id=doc_id, text=text, marked_bytes=marks)) + "\n")
    run = subprocess.run([binary, "fcoref-marked-study"], input="".join(inputs),
                         text=True, capture_output=True, env=env, check=True)
    stats = [tuple(map(int, line.split()[1:])) for line in run.stderr.splitlines()
             if line.startswith("FCOREF_RESERVE ")]
    if len(stats) != (len(docs) if reserve else 0):
        raise ValueError("Unexpected number of native candidate-budget diagnostics")
    for requested, legal, promoted, kept in stats:
        if not (requested == 3 and 0 <= promoted <= legal <= requested <= kept):
            raise ValueError("Invalid native marked-proposal diagnostics")
    predictions = [json.loads(line) for line in run.stdout.splitlines()]
    return predictions, stats


def partition_pairs(groups):
    return {pair for group in groups for pair in combinations(sorted(group), 2)}


def summarize(docs, before, after, native_stats, expected):
    stock = measure(docs, before, "fcoref")
    candidate = measure(docs, after, "fcoref")
    partitions = [[], []]
    changed = Counter()
    for (doc_id, text, targets, _), old, new in zip(docs, before, after, strict=True):
        groups = [decode_clusters(value, doc_id, text, "fcoref") for value in (old, new)]
        for accumulator, clusters in zip(partitions, groups, strict=True):
            accumulator.append((doc_id, sorted(sorted(cluster) for cluster in clusters)))
        old_pairs, new_pairs = map(partition_pairs, groups)
        marked = {tuple(sorted((targets[0], name))) for name in targets[1:]}
        changed["documents_with_pair_changes"] += old_pairs != new_pairs
        changed["added_unscored_pairs"] += len(new_pairs - old_pairs - marked)
        changed["removed_unscored_pairs"] += len(old_pairs - new_pairs - marked)
    cohort_sha = hashlib.sha256("\n".join(d[0] for d in docs).encode()).hexdigest()
    if (digest(partitions[0]) != expected["stock_partitions_sha256"]
            or stock["counts"] != expected["stock"] or cohort_sha != expected["cohort_ids_sha256"]):
        raise ValueError("Native no-hints output differs from the frozen stock panel")
    return dict(cohort_ids_sha256=cohort_sha, stock_partitions_sha256=digest(partitions[0]),
                marked_partitions_sha256=digest(partitions[1]), stock=stock["counts"],
                marked=candidate["counts"], paired=compare(docs, stock, candidate),
                output_pair_changes=dict(changed),
                proposals=dict(requested=sum(s[0] for s in native_stats), legal=sum(s[1] for s in native_stats),
                               promoted=sum(s[2] for s in native_stats),
                               retained_slots=sum(s[3] for s in native_stats),
                               documents_with_promotions=sum(bool(s[2]) for s in native_stats)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development", "test", "validation", "model", "binary"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    docs = documents(panel(args.development, args.test, args.validation))
    for name, checksum in MODEL_FILES.items():
        check_file(args.model / name, checksum)
    env = dict(os.environ, FCOREF_MODEL_PATH=str(args.model.resolve()), HF_HUB_OFFLINE="1", ANNO_NO_DOWNLOADS="1")
    before, _ = run_arm(args.binary.resolve(), env, docs, False)
    after, stats = run_arm(args.binary.resolve(), env, docs, True)
    results = summarize(docs, before, after, stats, json.loads(REFERENCE.read_text()))
    with args.binary.open("rb") as stream:
        results["diagnostic_binary_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
