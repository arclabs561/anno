"""Rebuild JointIE CrossRE denominators and co-window eligibility from source annotations."""

import argparse
from collections import Counter
from hashlib import sha256
from itertools import accumulate
import json
from pathlib import Path


def check_doc(doc, record):
    words = doc["sentence"]
    text = " ".join(words)
    if doc["doc_key"] != record["document"] or sha256(text.encode()).hexdigest() != record["source_sha256"]:
        raise ValueError("source text or ID mismatch")
    bounds = list(accumulate((len(w) + 1 for w in words), initial=0))
    mentions = {}
    for a, b, kind in doc["ner"]:
        if not 0 <= a <= b < len(words) or (a, b) in mentions and mentions[a, b][0] != kind:
            raise ValueError("invalid or ambiguous gold mention")
        mentions[a, b] = kind, bounds[a], bounds[b + 1] - 1
    gold = {(mentions[a, b], label, mentions[c, d]) for a, b, c, d, label, *_ in doc["relations"]}
    windows = record["windows"]

    def coverage(h, t):
        return [i for i, w in enumerate(windows)
                if w["start"] <= min(h[1], t[1]) and max(h[2], t[2]) <= w["end"]]

    pairs = {(h, t) for h, _, t in gold}
    checks = {"gold_mentions": len(mentions), "gold_edges": len(gold), "gold_directed_pairs": len(pairs),
              "gold_edges_co_window": sum(bool(coverage(h, t)) for h, _, t in gold),
              "gold_directed_pairs_co_window": sum(bool(coverage(h, t)) for h, t in pairs)}
    if "edge_trace" in record:
        traced = set()
        for e in record["edge_trace"]:
            h, t = tuple(e["head"]), tuple(e["tail"])
            key = h, e["predicate"], t
            if key in traced or e["co_windows"] != coverage(h, t):
                raise ValueError("duplicate edge or incorrect window geometry")
            traced.add(key)
        if traced != gold:
            raise ValueError("gold edge inventory differs from trace")
        reasons = Counter("first_loss_" + e["first_loss"] for e in record["edge_trace"])
        checks.update(reasons)
        checks["gold_edges_decoded"] = reasons["first_loss_matched"]
    if "decoded_mention_keys" in record:
        predicted_mentions = {tuple(m) for m in record["decoded_mention_keys"]}
        predicted_edges = {(tuple(h), r, tuple(t)) for h, r, t in record["decoded_edge_keys"]}
        if any(h not in predicted_mentions or t not in predicted_mentions for h, _, t in predicted_edges):
            raise ValueError("decoded edge points outside returned mentions")
        if "edge_trace" in record:
            marked = {(tuple(e["head"]), e["predicate"], tuple(e["tail"])) for e in record["edge_trace"]
                      if e["first_loss"] == "matched"}
            if marked != predicted_edges & gold:
                raise ValueError("trace match differs from output span keys")
        checks.update(decoded_mentions=len(record["decoded_mention_keys"]),
                      gold_mentions_decoded=len(predicted_mentions & set(mentions.values())),
                      decoded_edges=len(predicted_edges), gold_edges_decoded=len(predicted_edges & gold))
    for k, v in checks.items():
        if record["counts"].get(k) != v:
            raise ValueError(f"{doc['doc_key']} {k}: source {v}, receipt {record['counts'].get(k)}")


def audit(source, report):
    if sha256(source).hexdigest() != report["dataset_sha256"]:
        raise ValueError("dataset hash mismatch")
    docs = [json.loads(line) for line in source.splitlines()]
    n = len(report["documents"])
    sums = Counter()
    for gold, row in zip(docs[:n], report["documents"], strict=True):
        check_doc(gold, row)
        sums.update(row["counts"])
    if dict(sums) != report["totals"]:
        raise ValueError("per-document counts differ from panel totals")
    return {"documents": n, "totals": sums}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("receipts", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.receipts:
        print(json.dumps({path.name: audit(args.source.read_bytes(), json.loads(path.read_text()))}))
