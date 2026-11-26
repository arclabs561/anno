"""Independently score native JEREX outputs with exact, identifiable cluster endpoints."""

import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path


def gold(doc):
    offsets = [0]
    for words in doc["sents"]:
        offsets.append(offsets[-1] + len(words))
    entities = []
    types = []
    for mentions in doc["vertexSet"]:
        members = []
        for mention in mentions:
            start, end = mention["pos"]
            sent = mention["sent_id"]
            if not 0 <= start < end <= len(doc["sents"][sent]):
                raise ValueError("invalid gold mention")
            members.append((offsets[sent] + start, offsets[sent] + end))
        entities.append(frozenset(members))
        types.append(mentions[0]["type"])
    return entities, types, {(r["h"], r["t"], r["r"]) for r in doc["labels"]}


def predicted(doc):
    spans = [(m["start"], m["end"]) for m in doc["mentions"]]
    if any(not 0 <= a < b <= len(doc["tokens"]) for a, b in spans):
        raise ValueError("invalid predicted mention")
    if any(not 0 <= i < len(spans) for e in doc["entities"] for i in e["mentions"]):
        raise ValueError("invalid predicted mention index")
    groups = [frozenset(spans[i] for i in e["mentions"]) for e in doc["entities"]]
    return groups, [e["type"] for e in doc["entities"]], {
        (r["head"], r["tail"], r["type"]) for r in doc["relations"]
    }


def stages(groups, types, edges):
    if any(not g for g in groups) or any(not (0 <= h < len(groups) and 0 <= t < len(groups))
                                           for h, t, _ in edges):
        raise ValueError("empty entity or invalid relation endpoint")
    typed = list(zip(groups, types, strict=True))
    return {
        "mentions": set().union(*groups),
        "coreferent_pairs": {pair for g in groups for pair in combinations(sorted(g), 2)},
        "clusters": set(groups), "typed_clusters": set(typed),
        "directed_pairs": {(groups[h], groups[t]) for h, t, _ in edges},
        "labeled_edges": {(groups[h], groups[t], kind) for h, t, kind in edges},
        "typed_labeled_edges": {(typed[h], typed[t], kind) for h, t, kind in edges},
    }


def compare(gold_docs, pred_docs, trace=None):
    if len(gold_docs) != len(pred_docs):
        raise ValueError("number of attempted and returned documents differs")
    if trace is not None and len(trace["documents"]) != len(gold_docs):
        raise ValueError("trace covers a different number of documents")
    per_doc = []
    total = {}
    for index, (source, result) in enumerate(zip(gold_docs, pred_docs, strict=True)):
        if source["title"] != result["title"] or sum(source["sents"], []) != result["tokens"]:
            raise ValueError("predictions attached to another title or text edition")
        ref, pred = gold(source), predicted(result)
        if any(len(set(groups)) < len(groups) for groups, _, _ in [ref, pred]):
            # Converting a local entity index to a membership key would collapse distinct vertices.
            per_doc.append({"title": source["title"], "status": "ambiguous_memberships",
                            "gold_indexed_edges": len(ref[2]), "predicted_indexed_edges": len(pred[2])})
            continue
        actual, expected = stages(*pred), stages(*ref)
        # Detection also includes spans the resolver left outside every entity.
        actual["mentions"] = {(m["start"], m["end"]) for m in result["mentions"]}
        if trace is not None:
            observed = trace["documents"][index]
            if observed["document"] != source["title"] or observed["status"] != "scored":
                raise ValueError("trace title or scoring status differs")
            keys = {(e["head"], e["tail"], e["predicate"]): e for e in observed["edge_trace"]}
            if set(keys) != ref[2] or len(keys) != len(observed["edge_trace"]):
                raise ValueError("trace differs from indexed reference labels")
            for (h, t, label), item in keys.items():
                h_members, t_members = ref[0][h], ref[0][t]
                if item["head_missing_mentions"] != len(h_members - actual["mentions"]) or (
                        item["tail_missing_mentions"] != len(t_members - actual["mentions"])):
                    raise ValueError("trace differs from detected mention inventory")
                matched = (h_members, t_members, label) in actual["labeled_edges"]
                if matched != (item["first_loss"] == "matched"):
                    raise ValueError("trace match disagrees with raw entity and predicate identifiers")
        stats = {}
        for k, g in expected.items():
            p = actual[k]
            stats[k] = {"gold": len(g), "predicted": len(p), "matched": len(g & p)}
            total.setdefault(k, Counter()).update(stats[k])
        per_doc.append({"title": source["title"], "status": "scored", "counts": stats})
    return {"attempted": len(per_doc), "unique_membership_docs": sum(d["status"] == "scored" for d in per_doc),
            "totals_on_identifiable_docs": total, "documents": per_doc}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("gold", type=Path)
    cli.add_argument("predictions", type=Path)
    cli.add_argument("--first", type=int, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--trace", type=Path, help="optional corresponding observed JEREX stage receipt")
    opts = cli.parse_args()
    from joint_baseline.aggregate_receipt import external_output
    external_output(opts.output)
    source = json.loads(opts.gold.read_text())
    if not (0 < opts.first <= len(source)):
        cli.error("--first must select one or more available source documents")
    trace = json.loads(opts.trace.read_text()) if opts.trace else None
    report = compare(source[:opts.first], json.loads(opts.predictions.read_text()), trace)
    opts.output.write_text(json.dumps(report, indent=2) + "\n")
