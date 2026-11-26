# SPDX-License-Identifier: MIT OR Apache-2.0
"""Locate exact CrossRE relation losses inside the trained GLiNER-Relex export path."""

import argparse
from collections import Counter
import hashlib
import json
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


def measure(model, wrapper, doc, schema, threshold, predicate_gate):
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
    scored_rows = scores.tolist()
    def decode(gate):
        return Counter((h, r, t) for (h, t), row in zip(pairs, scored_rows, strict=True)
                       if h in kept and t in kept for r, score in zip(predicates, row, strict=True) if score > gate)
    graph = decode(threshold)
    expected = Counter(((r["head"]["start"], r["head"]["end"]), r["relation"],
                        (r["tail"]["start"], r["tail"]["end"])) for r in relations)
    if expected != graph:
        raise ValueError("trained export path disagrees with upstream predict_relations")
    reference = gold(doc)
    matched = (graph & reference).total()
    missed = classify(reference, graph, predicates, grid, proposed, set(pairs), kept)
    if missed.total() + matched != reference.total():
        raise ValueError("stage counts do not reconcile")
    comparison = Counter()
    if predicate_gate is not None:
        alternate = decode(predicate_gate)
        for name, delta, previously_matched in (("added", alternate - graph, reference - graph),
                                                 ("removed", graph - alternate, reference - alternate)):
            correct = (delta & previously_matched).total()
            comparison[name + "_matched"] = correct
            comparison[name + "_unmatched"] = delta.total() - correct
        comparison.update(predicted=alternate.total(), matched=(reference & alternate).total())
    return Counter(gold=reference.total(), matched=matched, predicted=graph.total(), directed_pairs=len(pairs)), missed, comparison


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
    totals, missed, comparison = Counter(), Counter(), Counter()
    schema = json.loads(SCHEMA.read_bytes())
    with torch.inference_mode():
        for line in lines[args.first_index:args.first_index + args.documents]:
            counts, losses, paired = measure(model, wrapper, json.loads(line), schema, 0.5, args.predicate_threshold)
            comparison.update(paired)
            totals.update(counts)
            missed.update(losses)
    result = {"source_sha256": TEST_SHA, "weights_sha256": SOURCE_SHA,
                      "schema_sha256": hashlib.sha256(SCHEMA.read_bytes()).hexdigest(),
                      "first_index": args.first_index, "documents": args.documents,
                      "threshold": 0.5, "totals": totals, "missed": missed}
    if args.predicate_threshold is not None:
        result.update(alternative_predicate_threshold=args.predicate_threshold, comparison=comparison)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
