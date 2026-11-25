# SPDX-License-Identifier: MIT OR Apache-2.0
"""Compare the new native executable against the original study executable."""
import argparse
import json
from pathlib import Path
import tempfile

from rerank import (
    ARCHIVE, LABEL_HASH, ROR_HASH, ROR_URL, SOURCE, compiled_index, get, lookup,
    read_catalog, read_queries,
)


def audit(cache, reference, current):
    archive = get(cache, ARCHIVE, ROR_URL, ROR_HASH, False)
    labels_path = get(cache, "s2aff-gold.csv", SOURCE, LABEL_HASH, False)
    catalog, _ = read_catalog(archive)
    queries, labels = read_queries(labels_path)
    original = lookup(reference, catalog, queries)
    result = lookup(current, catalog, queries)
    counts = {"test": 0, "val": 0}
    for old, new in zip(original, result, strict=True):
        if old != new:
            raise ValueError(f"natural candidate output differs on row {old[0].source}")
        counts[labels[old[0].source].split] += 1

    # First known-target row, in file order, for each split and label multiplicity.
    samples = {}
    for q in queries:
        label = labels[q.source]
        if label.ids and label.ids.issubset(catalog):
            samples.setdefault((label.split, len(label.ids) > 1), q)
    if len(samples) != 4:
        raise ValueError("sample does not cover the four expected masked checks")
    controls = []
    for q in samples.values():
        ids = labels[q.source].ids
        filtered = {i: record for i, record in catalog.items() if i not in ids}
        rebuilt = list(lookup(reference, filtered, [q]))
        masked = list(lookup(current, catalog, [q], catalog_masks={q.source: ids}))
        if masked != rebuilt:
            raise ValueError(f"complete masked candidate output differs on row {q.source}")
        controls.append(dict(source_row=int(q.source), split=labels[q.source].split,
                             removed=len(ids), distinct_candidates=masked[0][3]))
    return {"natural_identical": counts, "masked_identical_to_physical_rebuild": controls}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cache", type=Path)
    p.add_argument("reference_binary", type=Path, help="compiled pre-experiment ngrams.rs")
    args = p.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        print(json.dumps(audit(args.cache, args.reference_binary.resolve(), compiled_index(tmp)),
                         sort_keys=True, indent=2))
