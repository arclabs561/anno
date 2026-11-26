"""Classify every written JEREX relation against the fixed DocRED source edition."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from aggregate_receipt import external_output

from jerex_loss import gold_clusters
from jerex_stages import DOCRED, SPLIT, digest


def endpoint_evidence(members, reference):
    exact = [i for i, other in enumerate(reference) if members == other]
    overlapping = [i for i, other in enumerate(reference) if members & other]
    unannotated = members - set().union(*(reference[i] for i in overlapping))
    return {
        "exact": exact,
        "overlapping": overlapping,
        "unannotated_mentions": len(unannotated),
    }


def predicted_outcomes(gold, predicted):
    reference, proposed = gold_clusters(gold), gold_clusters(predicted)
    endpoints = [endpoint_evidence(cluster, reference) for cluster in proposed]
    repeated = Counter(proposed)
    labels = {(e["h"], e["r"], e["t"]) for e in gold["labels"]}
    outcomes = []
    for h, r, t in sorted({(e["h"], e["r"], e["t"]) for e in predicted["labels"]}):
        a, b = endpoints[h], endpoints[t]
        item = {
            "head": h,
            "predicate": r,
            "tail": t,
            "head_alignment": a,
            "tail_alignment": b,
        }
        if not a["exact"] or not b["exact"]:
            reason = "endpoint_has_no_exact_cluster"
        elif (
            len(a["exact"]) != 1
            or len(b["exact"]) != 1
            or repeated[proposed[h]] > 1
            or repeated[proposed[t]] > 1
        ):
            reason = "ambiguous_exact_membership"
        else:
            head, tail = a["exact"][0], b["exact"][0]
            forward = sorted(p for s, p, o in labels if (s, o) == (head, tail))
            reverse = sorted(p for s, p, o in labels if (s, o) == (tail, head))
            item.update(
                forward_gold_predicates=forward, reverse_gold_predicates=reverse
            )
            if r in forward:
                reason = "matched"
            elif forward:
                reason = "different_predicate_on_annotated_pair"
            elif r in reverse:
                reason = "same_predicate_only_in_reverse"
            else:
                reason = "directed_pair_unannotated"
        item["outcome"] = reason
        outcomes.append(item)
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gold", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("stage_receipt", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    external_output(args.output)
    sys.path.insert(0, str(DOCRED))
    from score_jerex import convert

    prior = json.loads(args.stage_receipt.read_bytes())
    if digest(args.gold) != SPLIT or prior["split_sha256"] != SPLIT:
        raise ValueError("source or receipt differs from the pinned DocRED edition")
    raw = json.loads(args.predictions.read_bytes())
    gold, predicted = convert(
        json.loads(args.gold.read_bytes())[: prior["enrolled_documents"]], raw
    )
    rows = []
    totals = Counter()
    for g, p, baseline in zip(gold, predicted, prior["documents"], strict=True):
        if g["title"] != baseline["document"]:
            raise ValueError("stage receipt and predictions have different ordering")
        edges = predicted_outcomes(g, p)
        counts = Counter(e["outcome"] for e in edges)
        stage = baseline["stages"]["labeled_relations"]
        if len(edges) != stage["predicted"] or counts["matched"] != stage["matched"]:
            raise ValueError(
                f"{g['title']}: classifications disagree with exact prediction scoring"
            )
        rows.append({"document": g["title"], "predictions": edges, "outcomes": counts})
        totals.update(counts)
    receipt = {
        "source_sha256": SPLIT,
        "stage_receipt_sha256": digest(args.stage_receipt),
        "saved_predictions_sha256": digest(args.predictions),
        "outcomes": totals,
        "documents": rows,
    }
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
