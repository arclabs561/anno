"""Score DWIE concepts and directed relations by exact cluster endpoints.

Pass two directories of DWIE-format documents with content, one JSON file per
document. Use the upstream DWIE downloader to populate missing article content.
The ordinary report covers edges with anchored endpoints. In supplied mode the
additional all_concept metrics count all edges by their provided concept IDs.
Predicted edges involving an unanchored concept have no span-based alignment
and are reported separately. DWIE's official scorer also
offers soft, mention-level and entity-centric metrics under its own protocols.
"""

import argparse
import json
from pathlib import Path

from score import document_index, integer, score


def convert(document):
    source = document.get("content")
    if not isinstance(source, str):
        raise ValueError(f"{document['id']}: content required to verify character offsets")
    concepts = {}
    for concept in document["concepts"]:
        key = integer(concept["concept"], "concept ID")
        if key in concepts:
            raise ValueError(f"{document['id']}: duplicate concept {key}")
        concepts[key] = []
    for mention in document["mentions"]:
        start = integer(mention["begin"], "mention begin")
        end = integer(mention["end"], "mention end")
        if not start < end <= len(source) or source[start:end] != mention["text"]:
            raise ValueError(f"{document['id']}: mention disagrees with source at {start}..{end}")
        key = integer(mention["concept"], "mention concept")
        if key not in concepts:
            raise ValueError(f"{document['id']}: unknown mention concept {key}")
        concepts[key].append({"sent_id": 0, "pos": [start, end]})
    anchored = sorted(key for key, spans in concepts.items() if spans)
    index = {key: i for i, key in enumerate(anchored)}
    labels = []
    unanchored_edges = set()
    for edge in document["relations"]:
        head = integer(edge["s"], "subject concept")
        tail = integer(edge["o"], "object concept")
        label = edge["p"]
        if not isinstance(label, str) or not label:
            raise ValueError(f"{document['id']}: invalid relation label")
        if head not in concepts or tail not in concepts:
            raise ValueError(f"{document['id']}: unknown endpoint in {edge!r}")
        if head in index and tail in index:
            labels.append({"h": index[head], "t": index[tail], "r": label})
        else:
            unanchored_edges.add((head, label, tail))
    # Internal coordinate grid only. Each column is one Unicode code point.
    return {
        "title": document["id"],
        "sents": [list(source)],
        "vertexSet": [concepts[key] for key in anchored],
        "labels": labels,
        "concepts_by_id": {
            key: frozenset(tuple(m["pos"]) for m in spans)
            for key, spans in concepts.items()
        },
        "unanchored_edges": unanchored_edges,
    }


def read_directory(directory):
    files = sorted(directory.glob("*.json"))
    if not files:
        raise ValueError(f"{directory}: expected DWIE .json documents")
    return [convert(json.loads(path.read_text(encoding="utf-8"))) for path in files]


def score_concepts(gold, predictions, mode):
    report = score(gold, predictions, mode)
    predicted = document_index(predictions)
    unanchored_hits = 0
    for doc in gold:
        other = predicted[doc["title"]]
        if mode == "supplied":
            if doc["concepts_by_id"] != other["concepts_by_id"]:
                raise ValueError(f"{doc['title']}: supplied concept IDs or mention sets changed")
            unanchored_hits += len(doc["unanchored_edges"] & other["unanchored_edges"])
    for name, documents in [("gold", gold), ("predicted", predictions)]:
        report[f"unanchored_{name}_concepts"] = sum(
            sum(not spans for spans in doc["concepts_by_id"].values()) for doc in documents)
        report[f"unanchored_{name}_relations"] = sum(len(doc["unanchored_edges"]) for doc in documents)
    # The common metrics describe the anchored cohort. Supplied concept IDs
    # additionally permit direct scoring of every edge, including implicit entities.
    if mode == "supplied":
        hits = unanchored_hits + report["matched_relations"]
        denominator = (report["gold_relations"] + report["unanchored_gold_relations"]
                       + report["predicted_relations"] + report["unanchored_predicted_relations"])
        report["all_concept_matched_relations"] = hits
        report["all_concept_micro_f1"] = 2 * hits / denominator if denominator else None
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gold", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--endpoints", choices=("predicted", "supplied"), required=True)
    args = parser.parse_args()
    try:
        report = score_concepts(read_directory(args.gold), read_directory(args.predictions), args.endpoints)
    except (ValueError, TypeError, KeyError, OSError) as error:
        parser.exit(1, f"Cannot score DWIE: {error}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
