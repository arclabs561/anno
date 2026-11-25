"""Measure CrossRE gold co-window eligibility, actual scored candidates, and decoded relations."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
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


def measure(doc, model, schema, worker, token_limit, window, overlap):
    text, gold_mentions, gold = gold_graph(doc)
    original_decode = model._decode
    captured = []

    def decode(candidates, *args):
        captured.append(candidates)
        return original_decode(candidates, *args)

    model._decode = decode
    try:
        result, windows = worker["extract_document"](model, schema, text, window, overlap, token_limit)
    finally:
        model._decode = original_decode
    scores, = captured

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
    return {"document": doc["doc_key"], "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "windows": windows, "counts": counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("crossre_test", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--check", type=Path, help="Require identical document identities, candidates, window coverage and decoded counts")
    parser.add_argument("--documents", type=int, default=8)
    parser.add_argument("--window-words", type=int, default=16)
    parser.add_argument("--overlap-words", type=int, default=4)
    args = parser.parse_args()
    import torch
    import gliner2
    from gliner2.joint_ie import JointIE, JointSchema

    if args.documents < 1 or not 0 <= args.overlap_words < args.window_words:
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
    for relation in vocab["relation_types"]:
        schema.relation(relation, labels, labels)
    model = JointIE.from_pretrained(str(args.checkpoint), device="cpu", local_files_only=True)
    schema = model.compile_schema(schema)
    limit = json.loads((args.checkpoint / "encoder_config/config.json").read_bytes())["max_position_embeddings"]
    rows = [json.loads(s) for s in data.splitlines()[:args.documents]]
    results = []
    with torch.inference_mode():
        for row in rows:
            results.append(measure(row, model, schema, worker, limit, args.window_words, args.overlap_words))
    totals = Counter()
    for row in results:
        totals.update(row["counts"])
    report = {"dataset": "CrossRE AI test", "dataset_sha256": TEST_SHA, "selection": "first N in source order",
              "vocabulary_sha256": hashlib.sha256(SCHEMA.read_bytes()).hexdigest(), "checkpoint": "fastino/gliner2.5-small-v1",
              "checkpoint_revision": "b5d42bf7dc7d01c40304ae4ed43c2f13c325a75b", "checkpoint_files": hashes,
              "gliner2": gliner2.__version__, "torch": torch.__version__, "window_words": args.window_words,
              "overlap_words": args.overlap_words, "totals": dict(totals), "documents": results}
    if args.check:
        expected = json.loads(args.check.read_bytes())
        for key in ("dataset_sha256", "checkpoint_files", "vocabulary_sha256", "window_words",
                    "overlap_words", "totals", "documents"):
            if report[key] != expected[key]:
                raise ValueError(f"coverage replay differs for {key}")
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
