# SPDX-License-Identifier: MIT OR Apache-2.0
"""Describe every extra decoded JointIE edge against that sentence's typed gold edges."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from private_ledger import external_path, save_new


def reason(h, p, t, gold):
    if (t, p, h) in gold:
        return "same_typed_endpoints_and_predicate_reversed"
    if any((h, t) == (a, b) for a, _, b in gold):
        return "forward_typed_pair_other_predicate"
    if any((h[1:], p, t[1:]) == (a[1:], q, b[1:]) for a, q, b in gold):
        return "forward_spans_and_predicate_endpoint_type_mismatch"
    if any((h[1:], t[1:]) == (a[1:], b[1:]) for a, _, b in gold):
        return "forward_exact_spans_other_predicate_and_endpoint_type_mismatch"
    if any((h[1:], t[1:]) == (b[1:], a[1:]) for a, _, b in gold):
        return "reverse_exact_spans_with_other_label_or_type"
    spans = {s[1:] for a, _, b in gold for s in (a, b)}
    if not gold:
        return "no_gold_relations"
    if h[1:] in spans and t[1:] in spans:
        return "both_exact_spans_elsewhere"
    if h[1:] in spans or t[1:] in spans:
        return "one_exact_span_elsewhere"
    return "neither_exact_span_elsewhere"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("short", type=Path, help="external 16-word full trace")
    parser.add_argument("wide", type=Path, help="external 384-word full trace")
    parser.add_argument("typed", type=Path, help="external training-routes full trace")
    parser.add_argument("--summaries", type=Path, default=Path(__file__).resolve().parents[1] /
                        "joint_baseline/receipts", help="public aggregate receipts to verify")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output = external_path(args.output)
    results = {}
    for arm, path in (("w16", args.short), ("w384", args.wide), ("training_routes", args.typed)):
        src = json.loads(external_path(path).read_bytes())
        public = json.loads((args.summaries / f"jointie_ai64_{arm}.json").read_bytes())
        serialized = json.dumps(src, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        if hashlib.sha256(serialized.encode()).hexdigest() != public["full_receipt_sha256"]:
            raise ValueError(f"{arm}: full private trace differs from the published receipt")
        expected = public["totals"]
        counts, rows, tp = Counter(), [], 0
        for d in src["documents"]:
            gold = {
                (tuple(e["head"]), e["predicate"], tuple(e["tail"]))
                for e in d["edge_trace"]
            }
            for h, p, t in d["decoded_edge_keys"]:
                h, t = tuple(h), tuple(t)
                if (h, p, t) in gold:
                    tp += 1
                    continue
                r = reason(h, p, t, gold)
                counts[r] += 1
                rows.append([d["source_sha256"], h, p, t, r])
        if (tp, tp + len(rows)) != (expected["gold_edges_decoded"], expected["decoded_edges"]):
            raise ValueError(f"{arm}: residual inventory differs from published counts")
        results[arm] = {
            "matched": tp,
            "extra": len(rows),
            "counts": counts,
            "errors": rows,
        }
        print(arm, dict(counts))
    save_new(args.output, results)


if __name__ == "__main__":
    main()
