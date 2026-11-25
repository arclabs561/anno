"""Score document-level relation predictions against annotated DocRED-format JSON.

Run: python3 score.py GOLD.json PREDICTIONS.json --endpoints predicted

Both files contain DocRED document arrays (title, sents, vertexSet, labels).
Mention offsets are half-open token indices in the named sentence. Use predicted
when a system generates its own clusters. Use supplied for a relation classifier
given the complete gold cluster inventory; this mode verifies that inventory.
Unique clusters may be reordered in either mode. Predicted endpoints match only
if their complete mention sets agree uniquely, ignoring mention types. Duplicate
mention sets remain ambiguous; supplied mode uses their original vertex IDs.
Duplicate triples count once. Missing predictions should be explicit documents with empty labels.

The output measures exact cluster membership plus directed, labeled relations.
Predicted mode reports the lower and upper bounds when alignment is ambiguous.
The upper bound relaxes global consistency, permitting each edge to choose its
own alignment among same-span clusters. DocRED leaderboard scores use a different protocol. Unannotated pairs count as
absent under this benchmark's closed-world scoring.
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def integer(value, description):
    if type(value) is not int or value < 0:
        raise ValueError(f"Expected a nonnegative integer for {description}, got {value!r}")
    return value


def document_index(documents):
    if not isinstance(documents, list):
        raise ValueError("Expected a JSON array of DocRED documents")
    indexed = {}
    for document in documents:
        title = document["title"]
        if not isinstance(title, str) or not title or title in indexed:
            raise ValueError(f"Empty or duplicate document title: {title!r}")
        indexed[title] = document
    return indexed


def endpoint_inventory(document, require_entity_type=False):
    title = document["title"]
    sentences = document["sents"]
    if not isinstance(sentences, list) or any(
        not isinstance(sentence, list) or any(not isinstance(t, str) for t in sentence)
        for sentence in sentences
    ):
        raise ValueError(f"{title}: sents must be lists of tokens")
    if not isinstance(document["vertexSet"], list):
        raise ValueError(f"{title}: vertexSet must be an array")
    endpoints = []
    for cluster in document["vertexSet"]:
        if not isinstance(cluster, list):
            raise ValueError(f"{title}: vertex must be a list of mentions")
        mentions = set()
        for mention in cluster:
            sent = integer(mention["sent_id"], "sent_id")
            start, end = (integer(x, "pos") for x in mention["pos"])
            if sent >= len(sentences) or not start < end <= len(sentences[sent]):
                raise ValueError(f"{title}: mention span outside sentence: {(sent, start, end)}")
            mentions.add((sent, start, end))
        if not mentions:
            raise ValueError(f"{title}: entity cluster has no mentions")
        key = frozenset(mentions)
        if require_entity_type:
            # As in the upstream JEREX DocRED reader, the first mention
            # supplies the entity-level type even if mention labels differ.
            entity_type = cluster[0]["type"]
            if not isinstance(entity_type, str) or not entity_type:
                raise ValueError(f"{title}: expected an entity type on the first mention")
            key = (entity_type, key)
        endpoints.append(key)
    return endpoints


def relation_set(document, endpoints):
    if not isinstance(document["labels"], list):
        raise ValueError(f"{document['title']}: labels must be an array")
    triples = set()
    for relation in document["labels"]:
        head = integer(relation["h"], "h")
        tail = integer(relation["t"], "t")
        label = relation["r"]
        if head >= len(endpoints) or tail >= len(endpoints):
            raise ValueError(f"{document['title']}: relation endpoint outside vertexSet")
        if not isinstance(label, str) or not label:
            raise ValueError(f"{document['title']}: empty or invalid relation label")
        triples.add((endpoints[head], label, endpoints[tail]))
    return triples


def score(gold, predictions, endpoint_mode, *, require_entity_type=False):
    if endpoint_mode not in ("supplied", "predicted"):
        raise ValueError(f"Unknown endpoint mode {endpoint_mode!r}")
    gold_by_title = document_index(gold)
    pred_by_title = document_index(predictions)
    if gold_by_title.keys() != pred_by_title.keys():
        raise ValueError("Gold and predicted document titles must agree")
    hits = gold_count = pred_count = exact_clusters = gold_clusters = pred_clusters = 0
    ambiguous_gold_clusters = ambiguous_pred_clusters = 0
    possible_hits = ambiguous_gold_edges = ambiguous_pred_edges = 0
    for title, doc in gold_by_title.items():
        prediction = pred_by_title[title]
        if doc["sents"] != prediction["sents"]:
            raise ValueError(f"{title}: gold and predicted tokenization differ")
        expected = endpoint_inventory(doc, require_entity_type)
        actual = endpoint_inventory(prediction, require_entity_type)
        expected_counts, actual_counts = Counter(expected), Counter(actual)
        if endpoint_mode == "supplied" and actual_counts != expected_counts:
            raise ValueError(f"{title}: supplied endpoints differ from gold mention sets")
        # Some Re-DocRED vertices have identical mention spans, even identical
        # types. Keep them distinct without using the gold edge to break ties.
        ambiguous = {key for key, count in (expected_counts | actual_counts).items() if count > 1}
        ambiguous_gold_clusters += sum(expected_counts[key] for key in ambiguous)
        ambiguous_pred_clusters += sum(actual_counts[key] for key in ambiguous)
        ambiguous_gold_keys = {}
        ambiguous_pred_keys = {}
        for index, key in enumerate(expected):
            if key in ambiguous:
                expected[index] = ("gold", index, key)
                ambiguous_gold_keys[expected[index]] = key
        for index, key in enumerate(actual):
            if key in ambiguous:
                if endpoint_mode == "supplied":
                    if expected[index] != ("gold", index, key):
                        raise ValueError(f"{title}: ambiguous supplied endpoints must retain gold IDs")
                    actual[index] = expected[index]
                else:
                    actual[index] = ("prediction", index, key)
                    ambiguous_pred_keys[actual[index]] = key
        gold_clusters += len(expected)
        pred_clusters += len(actual)
        exact_clusters += len(set(expected) & set(actual))
        gold_edges = relation_set(doc, expected)
        predicted_edges = relation_set(prediction, actual)
        gold_count += len(gold_edges)
        pred_count += len(predicted_edges)
        confirmed = len(gold_edges & predicted_edges)
        hits += confirmed
        possible_hits += confirmed
        if endpoint_mode == "predicted" and ambiguous:
            def unresolved(edges, aliases):
                # Keep multiplicities: different vertex IDs can yield equal span triples.
                return Counter(
                    (aliases.get(head, head), label, aliases.get(tail, tail))
                    for head, label, tail in edges
                    if head in aliases or tail in aliases
                )
            ambiguous_gold = unresolved(gold_edges, ambiguous_gold_keys)
            ambiguous_pred = unresolved(predicted_edges, ambiguous_pred_keys)
            ambiguous_gold_edges += sum(ambiguous_gold.values())
            ambiguous_pred_edges += sum(ambiguous_pred.values())
            possible_hits += sum((ambiguous_gold & ambiguous_pred).values())

    return {
        "endpoint_mode": endpoint_mode,
        "entity_type_policy": "first_mention" if require_entity_type else "ignored",
        "documents": len(gold_by_title),
        "gold_clusters": gold_clusters,
        "predicted_clusters": pred_clusters,
        "exact_clusters": exact_clusters,
        "ambiguous_gold_clusters": ambiguous_gold_clusters,
        "ambiguous_predicted_clusters": ambiguous_pred_clusters,
        "gold_relations": gold_count,
        "predicted_relations": pred_count,
        "matched_relations": hits,
        "ambiguous_gold_relations": ambiguous_gold_edges,
        "ambiguous_predicted_relations": ambiguous_pred_edges,
        "matched_relations_upper_bound": possible_hits,
        "micro_f1_upper_bound": 2 * possible_hits / (pred_count + gold_count)
        if pred_count + gold_count else None,
        "precision": hits / pred_count if pred_count else None,
        "recall": hits / gold_count if gold_count else None,
        "micro_f1": 2 * hits / (pred_count + gold_count) if pred_count + gold_count else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gold", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--endpoints", required=True, choices=("supplied", "predicted"))
    parser.add_argument("--require-entity-type", action="store_true", help="Require the first mention’s entity type to match")
    args = parser.parse_args()
    try:
        result = score(
            json.loads(args.gold.read_text(encoding="utf-8")),
            json.loads(args.predictions.read_text(encoding="utf-8")),
            args.endpoints, require_entity_type=args.require_entity_type,
        )
    except (KeyError, TypeError, ValueError, OSError) as error:
        parser.exit(1, f"Cannot score: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
