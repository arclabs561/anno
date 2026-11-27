# SPDX-License-Identifier: MIT OR Apache-2.0
"""Locate exact CrossRE relation losses inside the trained GLiNER-Relex export path."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

from jointie_coverage import SCHEMA, TEST_SHA

SCRIPTS = Path(__file__).resolve().parents[2]
SOURCE_SHA = "7c5bd751e1b24e4254d70fe4355a986cd65400676ce3735f7752429fcc26960a"


def gold(doc):
    offsets = []
    end = -1
    for word in doc["sentence"]:
        start = end + 1
        end = start + len(word)
        offsets.append((start, end))
    return Counter(((offsets[a][0], offsets[b][1]), r, (offsets[c][0], offsets[d][1]))
                   for a, b, c, d, r, *_ in doc["relations"])


def classify(reference, output, labels, grid, proposed, pairs, kept):
    missed = Counter()
    for (h, r, t), n in (reference - output).items():
        if r not in labels:
            reason = "predicate_outside_schema"
        elif h not in grid or t not in grid:
            reason = "endpoint_outside_span_grid"
        elif h not in proposed or t not in proposed:
            reason = "below_mention_threshold_" + ("both" if h not in proposed and t not in proposed
                                                    else "head" if h not in proposed else "tail")
        elif (h, t) not in pairs:
            reason = "ordered_pair_not_generated"
        elif h not in kept or t not in kept:
            reason = "flat_overlap_suppression"
        elif (h, r, t) in output:
            reason = "duplicate_reference_occurrence"
        else:
            reason = "predicate_below_threshold"
        missed[reason] += n
    return missed


def suppressed_edge_scores(reference, kept, predicates, pair_scores, threshold):
    """What can changing only flat mention selection recover at the stock gate?"""
    counts = Counter({
        "recoverable_distinct_edges": 0,
        "duplicate_occurrences_of_recoverable_edges": 0,
        "disjoint_endpoints_predicate_at_or_below_gate": 0,
        "overlapping_endpoints_predicate_above_gate": 0,
        "overlapping_endpoints_predicate_at_or_below_gate": 0,
        "unavailable_predicate_score": 0,
    })
    columns = {label: i for i, label in enumerate(predicates)}
    for (head, predicate, tail), copies in reference.items():
        if head in kept and tail in kept:
            continue
        row = pair_scores.get((head, tail))
        if row is None or predicate not in columns:
            continue  # An earlier stage lost this edge.
        score = row[columns[predicate]]
        if not math.isfinite(score):
            counts["unavailable_predicate_score"] += copies
            continue
        above = score > threshold
        if head[0] < tail[1] and tail[0] < head[1]:
            key = ("overlapping_endpoints_predicate_above_gate" if above
                   else "overlapping_endpoints_predicate_at_or_below_gate")
            counts[key] += copies
        elif above:
            # The graph emits a particular directed pair and predicate once.
            counts["recoverable_distinct_edges"] += 1
            counts["duplicate_occurrences_of_recoverable_edges"] += copies - 1
        else:
            counts["disjoint_endpoints_predicate_at_or_below_gate"] += copies
    return counts


def decode(pair_scores, kept, predicates, gate):
    return Counter((head, label, tail) for (head, tail), row in pair_scores.items()
                   if head in kept and tail in kept
                   for label, score in zip(predicates, row, strict=True) if score > gate)


def score_document(model, wrapper, doc, schema, threshold):
    """Expose pre-suppression directed scores after checking the stock public predictions."""
    from export_gliner_relex_joint import prepare

    text = " ".join(doc["sentence"])
    predicates = schema["relation_types"]
    feed, (entities, relations) = prepare(model, text, schema["entity_types"], predicates, threshold)
    logits, pair_slots, scores = wrapper(*feed)
    words = list(model.data_processor.words_splitter(text))
    spans = feed[4][0].tolist()
    def chars(slot):
        start, end = spans[slot]
        return words[start][1], words[end][2]
    grid = {chars(i) for i, valid in enumerate(feed[5].tolist()) if valid}
    selected = (logits[0].flatten(0, 1).amax(-1).sigmoid() > threshold) & feed[5]
    proposed = {chars(i) for i in selected.nonzero().flatten().tolist()}
    kept = {(m["start"], m["end"]) for m in entities}
    pairs = [(chars(h), chars(t)) for h, t in pair_slots.tolist()]
    pair_scores = dict(zip(pairs, scores.tolist(), strict=True))
    if len(pair_scores) != len(pairs):
        raise ValueError("trained export returned a duplicate directed span pair")
    graph = decode(pair_scores, kept, predicates, threshold)
    expected = Counter(((r["head"]["start"], r["head"]["end"]), r["relation"],
                        (r["tail"]["start"], r["tail"]["end"])) for r in relations)
    if expected != graph:
        raise ValueError("trained export path disagrees with upstream predict_relations")
    return grid, proposed, kept, pair_scores, graph


def measure(model, wrapper, doc, schema, threshold, predicate_gate):
    predicates = schema["relation_types"]
    grid, proposed, kept, pair_scores, graph = score_document(model, wrapper, doc, schema, threshold)
    reference = gold(doc)
    matched = (graph & reference).total()
    missed = classify(reference, graph, predicates, grid, proposed, pair_scores, kept)
    if missed.total() + matched != reference.total():
        raise ValueError("stage counts do not reconcile")
    suppressed = suppressed_edge_scores(reference, kept, predicates, pair_scores, threshold)
    if suppressed.total() != missed["flat_overlap_suppression"]:
        raise ValueError("suppressed predicate scores do not reconcile to the first-loss count")
    comparison = Counter()
    if predicate_gate is not None:
        alternate = decode(pair_scores, kept, predicates, predicate_gate)
        for name, delta, previously_matched in (("added", alternate - graph, reference - graph),
                                                 ("removed", graph - alternate, reference - alternate)):
            correct = (delta & previously_matched).total()
            comparison[name + "_matched"] = correct
            comparison[name + "_unmatched"] = delta.total() - correct
        comparison.update(predicted=alternate.total(), matched=(reference & alternate).total())
    totals = Counter(gold=reference.total(), matched=matched, predicted=graph.total(), directed_pairs=len(pair_scores))
    return totals, missed, comparison, suppressed


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("source", type=Path)
    cli.add_argument("checkpoint", type=Path)
    cli.add_argument("--first-index", type=int, default=64)
    cli.add_argument("--documents", type=int, default=32)
    cli.add_argument("--predicate-threshold", type=float, help="Re-decode with only this predicate gate changed")
    args = cli.parse_args()
    if args.predicate_threshold is not None and not 0 <= args.predicate_threshold <= 1:
        cli.error("predicate threshold must be in [0, 1]")
    data = args.source.read_bytes()
    lines = data.splitlines()
    if hashlib.sha256(data).hexdigest() != TEST_SHA or not (0 <= args.first_index < len(lines)
            and 1 <= args.documents <= len(lines) - args.first_index):
        cli.error("expected a valid slice of the pinned CrossRE AI test split")
    with (args.checkpoint / "model.safetensors").open("rb") as weights:
        if hashlib.file_digest(weights, "sha256").hexdigest() != SOURCE_SHA:
            cli.error("incorrect Relex checkpoint")
    sys.path.insert(0, str(SCRIPTS))
    from export_gliner_relex_joint import JointRelex
    from gliner import GLiNER
    import torch

    model = GLiNER.from_pretrained(str(args.checkpoint), local_files_only=True, low_cpu_mem_usage=True).eval()
    wrapper = JointRelex(model.model).eval()
    totals, missed, comparison, suppressed = Counter(), Counter(), Counter(), Counter()
    schema = json.loads(SCHEMA.read_bytes())
    with torch.inference_mode():
        for line in lines[args.first_index:args.first_index + args.documents]:
            counts, losses, paired, eligible = measure(model, wrapper, json.loads(line), schema, 0.5, args.predicate_threshold)
            comparison.update(paired)
            totals.update(counts)
            missed.update(losses)
            suppressed.update(eligible)
    result = {"source_sha256": TEST_SHA, "weights_sha256": SOURCE_SHA,
                      "schema_sha256": hashlib.sha256(SCHEMA.read_bytes()).hexdigest(),
                      "first_index": args.first_index, "documents": args.documents,
                      "threshold": 0.5, "totals": totals, "missed": missed,
                      "suppressed_at_stock_threshold": suppressed}
    if args.predicate_threshold is not None:
        result.update(alternative_predicate_threshold=args.predicate_threshold, comparison=comparison)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
