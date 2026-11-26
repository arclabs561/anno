# SPDX-License-Identifier: MIT OR Apache-2.0
"""Test whether another emitted mention can supply the exact boundary of a longer one."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from coref_gap import DATA_SHA, FC_FILES, FC_REV, check_file, decode_clusters, measure
from fcoref_proposal_budget import DEVELOPMENT_SHA, cohort as budget_cohort, compare
from native_gap_possessives import VALIDATION_SHA, read_rows


MODEL_FILES = {**FC_FILES, "encoder.onnx": "190101f8e3b927bc7fb5e053f63fb82a85ef7e92dc2acd3a23a5ad0dd58553f4",
               "config.json": "2d96a29d2ebd497d34580594f40ded241440411bcd83885b89d8fbb86189bc3b"}


def panel(development, test, validation):
    # The prior budget experiment already validates all three input file pins.
    prior_ids = {doc[0] for doc in budget_cohort(development, test, validation)}
    rows = read_rows(development)
    previous = read_rows(test) + read_rows(validation) + [r for r in rows if r["ID"] in prior_ids]
    urls, texts = {r["URL"] for r in previous}, {r["Text"] for r in previous}
    selected = []
    for row in sorted(rows, key=lambda r: hashlib.sha256(("anno-fcoref-emitted-v1/" + r["ID"]).encode()).digest()):
        if row["URL"] in urls or row["Text"] in texts:
            continue
        selected.append(row)
        urls.add(row["URL"])
        texts.add(row["Text"])
        if len(selected) == 256:
            return selected
    raise ValueError("Expected 256 distinct fresh article URLs")


def reshape(text, groups):
    """Resolve proposals simultaneously; no relinking, aliases or gold inputs."""
    used = {span for group in groups for span in group}
    proposals = {}
    for group in groups:
        names = set()
        for a, b in group:
            name = text[a:b]
            words = name.split()
            if len(words) >= 2 and all(w.isalpha() and w[0].isupper() for w in words):
                names.add(name)
        for a, b in group:
            hits = set()
            for name in names:
                for match in re.finditer(r'(?<!\w)' + re.escape(name) + r'(?!\w)', text[a:b]):
                    hit = a + match.start(), a + match.end()
                    if hit != (a, b):
                        hits.add(hit)
            if len(hits) == 1:
                hit = hits.pop()
                if hit not in used:
                    proposals[a, b] = hit
    counts = Counter(proposals.values())
    replacements = {span: hit for span, hit in proposals.items() if counts[hit] == 1}
    return [[replacements.get(span, span) for span in sorted(group)] for group in groups], len(replacements)


def score(rows, predictions):
    docs, candidate = [], []
    changed_spans = changed_docs = changed_marked_spans = 0
    for row, pred in zip(rows, predictions, strict=True):
        text = row["Text"]
        groups = decode_clusters(pred, row["ID"], text, "fcoref")
        reshaped, edits = reshape(text, groups)
        candidate.append({**pred, "clusters": reshaped})
        changed_spans += edits
        changed_docs += bool(edits)
        targets = []
        for key in ("Pronoun", "A", "B"):
            start = int(row[key + "-offset"])
            end = start + len(row[key])
            if text[start:end] != row[key]:
                raise ValueError("GAP source span disagrees with its text")
            targets.append((start, end))
        original = {s for g in groups for s in g}
        updated = {s for g in reshaped for s in g}
        changed_marked_spans += sum(s in original ^ updated for s in targets)
        labels = [row[k + "-coref"] for k in ("A", "B")]
        if not set(labels) <= {"TRUE", "FALSE"}:
            raise ValueError("unexpected GAP label")
        docs.append((row["ID"], text, targets, [v == "TRUE" for v in labels]))
    old = measure(docs, predictions, "fcoref")
    new = measure(docs, candidate, "fcoref")
    return dict(stock=old["counts"], candidate=new["counts"], paired=compare(docs, old, new),
                documents_with_replacements=changed_docs, replaced_output_spans=changed_spans,
                changed_marked_span_memberships=changed_marked_spans)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("development", "test", "validation", "model", "binary"):
        p.add_argument(arg, type=Path)
    a = p.parse_args()
    rows = panel(a.development, a.test, a.validation)
    for name, digest in MODEL_FILES.items():
        check_file(a.model / name, digest)
    env = dict(os.environ, FCOREF_MODEL_PATH=str(a.model.resolve()), ANNO_NO_DOWNLOADS="1",
               HF_HUB_OFFLINE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    for opt in ("ANNO_FCOREF_DOUBLE_K", "ANNO_FCOREF_BOUNDARY_RANKS", "ANNO_EXPERIMENT_COREf_POOL"):
        env.pop(opt, None)
    inputs = "".join(json.dumps({"id": r["ID"], "text": r["Text"]}) + "\n" for r in rows)
    out = subprocess.run([a.binary.resolve(), "fcoref"], input=inputs, text=True,
                         stdout=subprocess.PIPE, check=True, env=env)
    results = score(rows, [json.loads(line) for line in out.stdout.splitlines()])
    with a.binary.open("rb") as stream:
        binary_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    print(json.dumps(dict(cohort_ids_sha256=hashlib.sha256("\n".join(r["ID"] for r in rows).encode()).hexdigest(),
                          development_sha256=DEVELOPMENT_SHA, test_sha256=DATA_SHA, validation_sha256=VALIDATION_SHA,
                          binary_sha256=binary_sha, model_revision=FC_REV, model_files=MODEL_FILES,
                          **results), indent=2))


if __name__ == "__main__":
    main()
