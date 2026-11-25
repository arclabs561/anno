"""Compare JEREX joint-mode predictions.json with its ordered DocRED test split.

python3 score_jerex.py test_joint.json predictions.json

JEREX saves flat document tokens and offsets, and omits document IDs. This
adapter checks its token list against each original document before reading
the mention-cluster memberships and directed, labeled entity pairs.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from score import document_index, endpoint_inventory, integer, score


def convert(gold, predictions):
    document_index(gold)
    if not isinstance(predictions, list) or len(gold) != len(predictions):
        raise ValueError("JEREX must supply one prediction document per gold document")
    token_sources = Counter(tuple(t for sentence in doc["sents"] for t in sentence) for doc in gold)
    flat_gold, flat_predictions = [], []
    for doc, pred in zip(gold, predictions):
        # Validate sentence-local gold offsets before translating to the flat axis.
        endpoint_inventory(doc)
        tokens, offsets = [], []
        for sentence in doc["sents"]:
            offsets.append(len(tokens))
            tokens.extend(sentence)
        if "title" in pred and pred["title"] != doc["title"]:
            raise ValueError(f"{doc['title']}: JEREX prediction has a different document title")
        if "title" not in pred and token_sources[tuple(tokens)] > 1:
            raise ValueError(f"{doc['title']}: duplicate token sequence needs an explicit prediction title")
        if pred["tokens"] != tokens:
            raise ValueError(f"{doc['title']}: JEREX tokens differ from ordered gold document")
        flat_sents = [tokens]
        expected = []
        for cluster in doc["vertexSet"]:
            members = []
            for mention in cluster:
                sent_offset = offsets[mention["sent_id"]]
                flat_mention = {"sent_id": 0, "pos": [i + sent_offset for i in mention["pos"]]}
                if "type" in mention:
                    flat_mention["type"] = mention["type"]
                members.append(flat_mention)
            expected.append(members)
        flat_gold.append(dict(title=doc["title"], sents=flat_sents, vertexSet=expected, labels=doc["labels"]))
        mentions = []
        for mention in pred["mentions"]:
            start = integer(mention["start"], "mention.start")
            end = integer(mention["end"], "mention.end")
            if not start < end <= len(tokens):
                raise ValueError(f"{doc['title']}: invalid JEREX mention span {(start, end)}")
            mentions.append({"sent_id": 0, "pos": [start, end]})
        vertices = []
        for entity in pred["entities"]:
            indices = [integer(i, "entity.mentions") for i in entity["mentions"]]
            if any(i >= len(mentions) for i in indices):
                raise ValueError(f"{doc['title']}: JEREX entity references absent mention")
            members = [mentions[i].copy() for i in indices]
            if "type" in entity:
                for member in members:
                    member["type"] = entity["type"]
            vertices.append(members)
        labels = [{"h": r["head"], "t": r["tail"], "r": r["type"]} for r in pred["relations"]]
        flat_predictions.append(dict(title=doc["title"], sents=flat_sents, vertexSet=vertices, labels=labels))
    return flat_gold, flat_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gold", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--require-entity-type", action="store_true", help="Also require the endpoints’ entity types to agree")
    args = parser.parse_args()
    try:
        expected, actual = convert(
            json.loads(args.gold.read_text(encoding="utf-8")),
            json.loads(args.predictions.read_text(encoding="utf-8")),
        )
        report = score(expected, actual, "predicted", require_entity_type=args.require_entity_type)
    except (KeyError, TypeError, ValueError, OSError) as error:
        parser.exit(1, f"Cannot score JEREX: {error}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
