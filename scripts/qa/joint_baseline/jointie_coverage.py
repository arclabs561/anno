"""Measure CrossRE gold co-window eligibility, actual scored candidates, and decoded relations."""
import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
from aggregate_receipt import external_output
import runpy

ROOT = Path(__file__).resolve().parents[3]
WORKER = ROOT / "crates/anno-cli/src/cli/commands/joint.py"
SCHEMA = ROOT / "docs/eval/crossre-ai-prompt-schema.json"
TEST_SHA = "2000a4011676319f9f3f3f6044e7de356b975d81979e080460e82c2273ed55cb"
WEIGHTS_SHA = "4ee982787ace270d4bf15dbcb28ced38e0aa201372347114ceedd6336055de2b"


def gold_graph(doc):
    tokens = doc["sentence"]
    text = " ".join(tokens)
    starts = []
    next_start = 0
    for token in tokens:
        starts.append(next_start)
        next_start += len(token) + 1

    def span(start, end):
        if not 0 <= start <= end < len(tokens):
            raise ValueError(f"invalid inclusive token span {start},{end}")
        return starts[start], starts[end] + len(tokens[end])

    labels = {}
    for start, end, label in doc["ner"]:
        key = span(start, end)
        if key in labels and labels[key] != label:
            raise ValueError(f"{doc['doc_key']}: ambiguous type for relation endpoint {key}")
        labels[key] = label
    nodes = {(label, *key) for key, label in labels.items()}
    edges = set()
    for s, e, os, oe, predicate, *_ in doc["relations"]:
        head, tail = span(s, e), span(os, oe)
        edges.add(((labels[head], *head), predicate, (labels[tail], *tail)))
    return text, nodes, edges


def measure(doc, model, schema, worker, token_limit, window, overlap, loss_trace=False, sparse_roles=False, rank_trial=False):
    text, gold_mentions, gold = gold_graph(doc)
    original_decode = model._decode
    captured = []

    def decode(candidates, *args):
        captured.append((candidates, args))
        return original_decode(candidates, *args)

    model._decode = decode
    try:
        with ExitStack() as stack:
            if loss_trace:
                from jointie_loss import DecoderTrace
                trace = DecoderTrace()
                trace.watch(stack, model)
            if sparse_roles:
                from jointie_roles import RoleTrace
                roles = RoleTrace()
                roles.watch(stack, model)
            result, windows = worker["extract_document"](model, schema, text, window, overlap, token_limit)
    finally:
        model._decode = original_decode
    (scores, decode_args), = captured
    config = decode_args[-1]

    def occurrence(key):
        label, start, end = key
        return label, scores.start_mappings[start], scores.end_mappings[end - 1]

    candidates = {occurrence(m.key) for m in scores.mentions}
    scored = {(occurrence(e.head), e.relation_type, occurrence(e.tail)) for e in scores.edges}
    graph = result.to_dict(include_confidence=True, include_spans=True)
    selected = {e["id"]: (e["type"], e["start"], e["end"]) for e in graph["entities"]}
    decoded = {(selected[e["head"]], e["type"], selected[e["tail"]]) for e in graph["relations"]}
    gold_pairs = {(h[1:], t[1:]) for h, _, t in gold}

    def eligible(head, tail):
        return any(w["start"] <= min(head[0], tail[0]) and max(head[1], tail[1]) <= w["end"]
                   for w in windows)

    counts = {
        "scored_windows": len(windows), "gold_mentions": len(gold_mentions),
        "scored_mentions": len(candidates), "gold_mentions_scored": len(gold_mentions & candidates),
        "decoded_mentions": len(selected), "gold_mentions_decoded": len(gold_mentions & set(selected.values())),
        "gold_directed_pairs": len(gold_pairs),
        "gold_directed_pairs_co_window": sum(eligible(h, t) for h, t in gold_pairs),
        "gold_edges": len(gold),
        "gold_edges_co_window": sum(eligible(h[1:], t[1:]) for h, _, t in gold),
        "gold_edges_both_mentions_scored": sum(h in candidates and t in candidates for h, _, t in gold),
        "actual_scored_edges": len(scored), "gold_edges_scored": len(gold & scored),
        "decoded_edges": len(decoded), "gold_edges_decoded": len(gold & decoded),
    }
    if not decoded <= scored:
        raise ValueError("unconstrained directed schema produced an edge absent from the scorer")
    row = {"document": doc["doc_key"], "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "windows": windows, "counts": counts}
    if loss_trace:
        row["edge_trace"] = trace.losses(gold, scores, config, windows, decoded)
        if sparse_roles:
            for edge in row["edge_trace"]:
                edge["sparse_role_stages"] = roles.classify(edge, windows)
        row["decoded_mention_keys"] = sorted(selected.values())
        row["decoded_edge_keys"] = sorted(decoded)
        counts["sparse_directed_predicate_span_pairs"] = len({(h[1:], p, t[1:]) for h,p,t in scored})
        row["feasible"] = result.feasible
        counts["feasible_documents"] = int(result.feasible)
        reasons = Counter(edge["first_loss"] for edge in row["edge_trace"])
        if sum(reasons.values()) != len(gold) or reasons["matched"] != len(gold & decoded):
            raise ValueError("gold edge outcomes do not reconcile with the model output")
        counts.update({"first_loss_" + reason: n for reason,n in sorted(reasons.items())})
    if rank_trial:
        from jointie_ranking import trial
        row["ranking_trial"] = trial(scores, config, gold, decoded, original_decode, decode_args)
    return row


def attribute_schema(rows, routes):
    """Use the declared production schema independently of geometry or model scores."""
    for row in rows:
        for edge in row["edge_trace"]:
            pair = routes.get(edge["predicate"])
            if pair is None:
                reason = "predicate_outside_schema"
            elif edge["head"][0] not in pair[0] or edge["tail"][0] not in pair[1]:
                reason = "endpoint_types_outside_schema"
            else:
                reason = None
            edge["schema_allows_gold_types"] = reason is None
            if reason is not None:
                if edge["first_loss"] == "matched":
                    raise ValueError("unexpected decoded edge outside declared schema")
                edge["first_loss"] = reason
        counts = row["counts"]
        for key in list(counts):
            if key.startswith("first_loss_"):
                del counts[key]
        for reason, n in Counter(e["first_loss"] for e in row["edge_trace"]).items():
            counts["first_loss_" + reason] = n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("crossre_test", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--check", type=Path, help="Require identical document identities, candidates, window coverage and decoded counts")
    parser.add_argument("--training-routes", type=Path, help="Restrict allowed head and tail types per predicate to their observed CrossRE AI training types")
    parser.add_argument("--loss-trace", action="store_true", help="Retain one observed outcome per gold directed typed edge")
    parser.add_argument("--first-index", type=int, default=0, help="Start of the contiguous evaluation slice (zero-based)")
    parser.add_argument("--argument-threshold", type=float, help="Experimental boundary argument gate; all other model and optimizer settings stay fixed")
    parser.add_argument("--rank-typed-alternatives", action="store_true", help="Also compare type-supported span-pair ranking at the unchanged per-predicate cap")
    parser.add_argument("--sparse-roles", action="store_true", help="Also observe routed role candidates and the actual ordered pair generator")
    parser.add_argument("--documents", type=int, default=8)
    parser.add_argument("--window-words", type=int, default=16)
    parser.add_argument("--overlap-words", type=int, default=4)
    args = parser.parse_args()
    if args.sparse_roles or args.rank_typed_alternatives or args.argument_threshold is not None:
        args.loss_trace = True
        if any((parent / ".git").is_file() or (parent / ".git/HEAD").is_file()
               for parent in args.output.resolve().parents):
            parser.error("write per-edge role diagnostics outside Git checkouts")
    if args.loss_trace or args.training_routes:
        external_output(args.output)
    import torch
    import gliner2
    from gliner2.joint_ie import JointIE, JointSchema

    if args.sparse_roles and gliner2.__version__ != "2.2.0":
        raise ValueError("role instrumentation is verified for gliner2==2.2.0")

    if args.first_index < 0 or args.documents < 1 or not 0 <= args.overlap_words < args.window_words:
        parser.error("expected a positive document count and window larger than overlap")
    data = args.crossre_test.read_bytes()
    if hashlib.sha256(data).hexdigest() != TEST_SHA:
        raise ValueError("Expected pinned CrossRE AI test file")
    worker = runpy.run_path(str(WORKER))
    hashes = worker["checkpoint_receipt"](args.checkpoint)
    if hashes["model.safetensors"] != WEIGHTS_SHA:
        raise ValueError("Expected pinned Fastino GLiNER2.5 small v1 checkpoint")
    vocab = json.loads(SCHEMA.read_bytes())
    labels = vocab["entity_types"]
    schema = JointSchema().entities(labels)
    routes = {r: (labels, labels) for r in vocab["relation_types"]}
    train_digest = None
    if args.training_routes:
        training = args.training_routes.read_bytes()
        train_digest = hashlib.sha256(training).hexdigest()
        if train_digest != "44d54f2669c2a41f613aac70095aed8a3b8f2e6befde454abb99bf97c77385d3":
            raise ValueError("Expected pinned CrossRE AI training split")
        routes = {r: (set(), set()) for r in vocab["relation_types"]}
        for line in training.splitlines():
            _, _, examples = gold_graph(json.loads(line))
            for head, name, tail in examples:
                routes[name][0].add(head[0])
                routes[name][1].add(tail[0])
        routes = {r: (sorted(heads), sorted(tails)) for r, (heads, tails) in routes.items()}
    for relation, (heads, tails) in routes.items():
        schema.relation(relation, heads, tails)
    model = JointIE.from_pretrained(str(args.checkpoint), device="cpu", local_files_only=True)
    if args.argument_threshold is not None:
        from dataclasses import replace
        if not 0 <= args.argument_threshold <= 1:
            parser.error("argument threshold must be between zero and one")
        generator = model.scorer.model.relation_pair_generator
        generator.settings = replace(generator.settings, argument_threshold=args.argument_threshold)
    schema = model.compile_schema(schema)
    limit = json.loads((args.checkpoint / "encoder_config/config.json").read_bytes())["max_position_embeddings"]
    lines = data.splitlines()
    if args.first_index + args.documents > len(lines):
        parser.error("requested document count exceeds the source split")
    rows = [json.loads(s) for s in lines[args.first_index:args.first_index + args.documents]]
    results = []
    with torch.inference_mode():
        for row in rows:
            results.append(measure(row, model, schema, worker, limit, args.window_words, args.overlap_words, args.loss_trace, args.sparse_roles, args.rank_typed_alternatives))
    if args.loss_trace:
        attribute_schema(results, routes)
    totals = Counter()
    for row in results:
        totals.update(row["counts"])
    report = {"dataset": "CrossRE AI test", "dataset_sha256": TEST_SHA, "selection": "first N in source order",
              "vocabulary_sha256": hashlib.sha256(SCHEMA.read_bytes()).hexdigest(), "checkpoint": "fastino/gliner2.5-small-v1",
              "checkpoint_revision": "b5d42bf7dc7d01c40304ae4ed43c2f13c325a75b", "checkpoint_files": hashes,
              "gliner2": gliner2.__version__, "torch": torch.__version__, "window_words": args.window_words,
              "overlap_words": args.overlap_words, "totals": dict(totals), "documents": results}
    if args.first_index:
        report.update(selection="contiguous slice in source order", first_index=args.first_index)
    if args.argument_threshold is not None:
        report["argument_threshold"] = args.argument_threshold
    if args.loss_trace:
        report.update(enrolled_documents=args.documents, scored_documents=len(results))
    if args.training_routes:
        report.update(training_sha256=train_digest, allowed_endpoint_types={r: list(pair) for r, pair in routes.items()})
    if args.check:
        expected = json.loads(args.check.read_bytes())
        fields = ["dataset_sha256", "checkpoint_files", "vocabulary_sha256", "window_words",
                  "overlap_words", "totals", "documents"]
        for field in ("first_index", "argument_threshold"):
            if report.get(field) != expected.get(field):
                raise ValueError(f"coverage replay differs for {field}")
        if args.training_routes:
            fields.extend(["training_sha256", "allowed_endpoint_types"])
        for key in fields:
            if report[key] != expected[key]:
                raise ValueError(f"coverage replay differs for {key}")
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
