# SPDX-License-Identifier: MIT OR Apache-2.0
"""Compare locked bibliography predictions with a subsequently released reference key."""
import argparse
from collections import Counter
import json
from pathlib import Path

from .common import (ARMS, COMPONENTS, CUTS, DELIVERIES, component, doi, file_hash,
                     file_map, load_rows, require)


def index_rows(rows, fields, expected):
    result = {}
    for item in rows:
        key = tuple(doi(item[f]) if f == "notice" else item[f] for f in fields)
        require(key not in result, f"duplicate row {key}")
        result[key] = item
    require(result.keys() == expected, "missing/unexpected rows: " + str(set(result) ^ expected))
    return result


def outcomes(answer, gold, field):
    if gold.get(field) is None:
        return "unjudged_answered" if answer[field] is not None else "unjudged_abstained"
    if answer[field] is None:
        return "abstained_known"
    return "correct" if answer[field] == gold[field] else "wrong"


def arm_metrics(keys, pred, gold, arm, delivery, works):
    counter = Counter()
    wrong_works = {field: set() for field in COMPONENTS}
    for k in keys:
        r, g = pred[(*k, arm, delivery)], gold[k]
        counter["selected"] += 1
        counter["answered" if r["status"] == "available" else "abstained"] += 1
        known_joint = g.get("edition") is not None and g.get("byline") is not None
        if known_joint:
            counter["joint_reference"] += 1
            counter["joint_correct"] += r["edition"] == g["edition"] and r["byline"] == g["byline"]
        for field in COMPONENTS:
            outcome = outcomes(r, g, field)
            counter[field, outcome] += 1
            if outcome == "wrong":
                wrong_works[field].add(works[k[0]])
    result = {f: counter[f] for f in ("selected", "answered", "abstained", "joint_reference", "joint_correct")}
    for field in COMPONENTS:
        counts = {f: counter[field, f] for f in ("correct", "wrong", "abstained_known",
                                                  "unjudged_answered", "unjudged_abstained")}
        scorable = counts["correct"] + counts["wrong"]
        result[field] = {**counts, "emitted_scorable": scorable, "reference_known": scorable + counts["abstained_known"],
                         "works_with_wrong_answer": len(wrong_works[field])}
    return result


def score(document, references):
    frames = document["frame"]
    works = {doi(r["notice"]): doi(r["target"]) for r in frames}
    require(len(works) == len(frames), "duplicate notice in frame")
    keys = [(notice, cut) for notice in works for cut in CUTS]
    gold = index_rows(load_rows(references), ("notice", "cut"), set(keys))
    dimensions = ("notice", "cut", "arm", "delivery")
    expected = {(*k, arm, d) for k in keys for arm in ARMS for d in DELIVERIES}
    pred = index_rows(document["rows"], dimensions, expected)
    for k, g in gold.items():
        require(g["status"] in ("verified", "partial", "unavailable", "ambiguous"), f"reference status {k}")
        for alias in ("target", "work"):
            require(alias not in g or doi(g[alias]) == works[k[0]], "reference target mismatch")
        for field in COMPONENTS:
            component(g.get(field), field)
        if g["status"] == "verified":
            require(g.get("edition") is not None and g.get("byline") is not None, "verified reference missing component")
        if any(g.get(f) is not None for f in COMPONENTS):
            require(bool(g.get("sources")), f"uncited reference {k}")
    for k, row in pred.items():
        require(doi(row["target"]) == works[k[0]], "prediction target mismatch")
        available = row["status"] == "available"
        require(available or row["status"] == "unavailable", "unknown prediction status")
        for field in COMPONENTS:
            component(row[field], field)
            require(available == (row[field] is not None), "prediction status conflicts with " + field)
        require(available == bool(row["evidence"]), "prediction evidence conflicts with availability")

    deliveries = {}
    for d in DELIVERIES:
        arms = {}
        for arm in ARMS:
            scopes = {"all": arm_metrics(keys, pred, gold, arm, d, works)}
            scopes.update({cut: arm_metrics([k for k in keys if k[1] == cut], pred, gold, arm, d, works)
                           for cut in CUTS})
            arms[arm] = scopes
        pairs = {}
        changes = {}
        for field in ("byline", "printed_byline"):
            paired = Counter()
            for k in keys:
                if gold[k].get(field) is not None:
                    mutable = outcomes(pred[(*k, "mutable", d)], gold[k], field)
                    dated = outcomes(pred[(*k, "dated", d)], gold[k], field)
                    paired[f"{mutable}_to_{dated}"] += 1
            pairs[field] = dict(sorted(paired.items()))
            distinguishable = [n for n in works if all(gold[n, c].get(field) is not None for c in ("pre", "post"))
                               and gold[n, "pre"][field] != gold[n, "post"][field]]
            both_known = sum(all(gold[n, c].get(field) is not None for c in ("pre", "post")) for n in works)
            changes[field] = {"pre_post_known": both_known, "changed_at_notice": len(distinguishable),
                              "both_cuts_correct_on_changed": {
                                  arm: sum(all(pred[n, c, arm, d][field] == gold[n, c][field] for c in ("pre", "post"))
                                           for n in distinguishable) for arm in ARMS}}
        deliveries[d] = {"arms": arms, "paired_mutable_to_dated": pairs, "notice_changes": changes}

    disagreements = []
    for k in keys:
        for arm in ARMS:
            rows = [pred[(*k, arm, delivery)] for delivery in DELIVERIES]
            if any(rows[0].get(f) != rows[1].get(f)
                   for f in (*COMPONENTS, "status", "reason", "evidence")):
                disagreements.append({"notice": k[0], "cut": k[1], "arm": arm})
    return {"notices": len(works), "works": len(set(works.values())), "queries": len(keys),
            "reference_status": dict(sorted(Counter(g["status"] for g in gold.values()).items())),
            "deliveries": deliveries, "delivery_pairs": len(keys) * len(ARMS),
            "delivery_mismatches": disagreements}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="directory containing predictions.json and lock.json")
    parser.add_argument("--lock-sha256", required=True, help="publicly recorded SHA-256 of the pre-unblinding lock")
    parser.add_argument("--reference", required=True, type=Path)
    args = parser.parse_args()
    lock_path = args.run / "lock.json"
    require(file_hash(lock_path) == args.lock_sha256, "wrong prediction lock")
    lock = json.loads(lock_path.read_bytes())
    pred_path = args.run / "predictions.json"
    require(file_hash(pred_path) == lock["predictions_sha256"], "predictions changed after lock")
    require(file_map() == lock["code"], "code or protocol changed after lock")
    report = score(json.loads(pred_path.read_bytes()), json.loads(args.reference.read_bytes()))
    report["reference_sha256"] = file_hash(args.reference)
    report["prediction_lock_sha256"] = args.lock_sha256
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
