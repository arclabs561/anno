# SPDX-License-Identifier: MIT OR Apache-2.0
"""Measure target availability by swapping donor and query pages within fixed NILK test windows."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from .replay import HERE, evaluate, window_records


STRATA = {"id_sampled": "reachable", "id_unsampled": "sampled_out", "nil": "true_nil"}


def page_fold(page):
    return hashlib.sha256(("NILK-window-page-v1|" + str(page)).encode()).digest()[0] % 2


def score_windows(windows):
    targets = defaultdict(set)
    counts = defaultdict(lambda: {"exact": Counter(), "fold": Counter()})
    folds = []
    for window, records in windows.items():
        sides = ([], [])
        for row in records:
            sides[page_fold(row[0])].append(row)
        for held_out, queries in enumerate(sides):
            donors = sides[1 - held_out]
            baseline = evaluate(donors, queries)
            if baseline["shared_pages"]:
                raise ValueError("donors and queries share a source page")
            known = {qid for _, _, qid, nil in donors if not nil}
            strata = Counter()
            for _, _, qid, nil in queries:
                group = "true_nil" if nil else ("reachable" if qid in known else "sampled_out")
                targets[group].add(qid)
                strata[group] += 1
            folds.append([window, held_out, len(queries), len(known), strata])
            for policy, result in baseline["policies"].items():
                for key, value in result["links"].items():
                    label, decision = key.split(".")
                    counts[STRATA[label]][policy]["no_hit" if decision == "empty" else decision] += value
    return {"folds": folds, "by_label": counts, "targets": {k: len(v) for k, v in targets.items()}}


def page_control(observations):
    windows = defaultdict(list)
    seen = set()
    for start, key, row in observations:
        if key in seen:
            raise ValueError("duplicate test observation ID")
        seen.add(key)
        windows[start].append(row)
    rows = [r for block in windows.values() for r in block]
    pages = {r[0] for r in rows}
    return {
        "pages": len(pages), "queries": len(rows),
        "page_fold_counts": Counter(page_fold(p) for p in pages),
        "query_fold_counts": Counter(page_fold(r[0]) for r in rows),
        "within_window": score_windows(windows),
        "pooled_windows": score_windows({"pooled": rows}),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    args = parser.parse_args()
    receipt = json.loads((HERE / "receipt.json").read_text())
    pin = receipt["page_control"]
    if hashlib.sha256((HERE / "page_control_design.txt").read_bytes()).hexdigest() != pin["design_sha256"]:
        raise ValueError("page control design changed")
    result = page_control(window_records(args.cache, "test", False, receipt["sha256"]))
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if hashlib.sha256(encoded.encode()).hexdigest() != pin["result_sha256"]:
        raise ValueError("page control differs from its independent frozen receipt")
    print(encoded, end="")


if __name__ == "__main__":
    main()
