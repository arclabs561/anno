"""Score source-aligned coreference candidates and clusters on a person mention panel.

Usage: python scripts/qa/score_person_mentions.py GOLD.json PREDICTIONS.json
Predictions use the FCoref word-aligned receipt shape: results contain an id,
source_sha256, mentions (start, end, surface) and clusters of scalar [start,end)
pairs. The same proposed spans are tested against each annotation contract.
"""
import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

from check_span_assertions import indexed, objects, source_sha256


def span(value, text):
    if (not isinstance(value, list) or len(value) != 2
            or any(type(n) is not int for n in value)
            or not 0 <= value[0] < value[1] <= len(text)):
        raise ValueError(f"invalid Unicode scalar span {value!r}")
    return tuple(value)


def gold_span(value, surface, text, label):
    bounds = span(value, text)
    if text[slice(*bounds)] != surface:
        raise ValueError(f"{label}: recorded surface differs from source at {bounds}")
    return bounds


def annotation_maps(doc):
    text = doc["text"]
    if not isinstance(text, str):
        raise ValueError(f"{doc['id']}: expected source text")
    people = indexed(doc["people"], "people")
    mentions = indexed(doc["mentions"], "gold mentions")
    lookups = {"highlight": {}, "referential": {}}
    owners = {}
    for key, m in mentions.items():
        label = f"{doc['id']}/{key}"
        if m["person_id"] is not None and m["person_id"] not in people:
            raise ValueError(f"{label}: unknown person")
        highlight = gold_span(m["highlight"], m.get("highlight_surface"), text, label)
        refs = m["referential_spans"]
        surfaces = m.get("referential_surfaces")
        if not isinstance(refs, list) or not refs:
            raise ValueError(f"{label}: missing referential spans")
        if not isinstance(surfaces, list) or len(refs) != len(surfaces):
            raise ValueError(f"{label}: expected one recorded surface per referential span")
        references = [gold_span(r, surface, text, label) for r, surface in zip(refs, surfaces)]
        for start, end in references:
            if not start <= highlight[0] < highlight[1] <= end:
                raise ValueError(f"{label}: referential span must contain the highlight")
        for contract, spans in [("highlight", [highlight]), ("referential", references)]:
            for r in spans:
                if r in lookups[contract] or owners.setdefault(r, key) != key:
                    raise ValueError(f"{label}: ambiguous or duplicate {contract} span {r}")
                lookups[contract][r] = key

    negatives = set()
    for n in objects(doc["negative_spans"], "negative spans"):
        r = gold_span(n["span"], n.get("surface"), text, f"{doc['id']}/negative")
        if r in negatives:
            raise ValueError(f"{doc['id']}: duplicate negative span {r}")
        if any(r[0] < end and start < r[1] for start, end in owners):
            raise ValueError(f"{doc['id']}: negative span {r} overlaps a person mention")
        negatives.add(r)
    return mentions, lookups, negatives


def model_spans(doc, row):
    text = doc["text"]
    if row.get("source_sha256") != source_sha256(text):
        raise ValueError(f"{doc['id']}: prediction source_sha256 differs from gold")
    proposed = set()
    for m in objects(row["mentions"], "predicted mentions"):
        s = span([m["start"], m["end"]], text)
        if text[slice(*s)] != m["surface"] or s in proposed:
            raise ValueError(f"duplicate or misaligned candidate {s}")
        proposed.add(s)
    clusters = []
    if not isinstance(row["clusters"], list):
        raise TypeError("expected an array of clusters")
    for members in row["clusters"]:
        if not isinstance(members, list) or len(members) < 2:
            raise ValueError("expected a multi-mention cluster")
        cluster = {span(r, text) for r in members}
        if len(cluster) != len(members) or not cluster <= proposed:
            raise ValueError("cluster contains a repeated or missing candidate")
        clusters.append(cluster)
    return proposed, clusters


def selection(spans, lookup, negatives, gold_count):
    matched = spans & lookup.keys()
    # One occurrence can license more than one acceptable source range.
    return {"emitted": len(spans), "gold_mentions": gold_count,
            "matched_spans": len(matched), "covered_mentions": len({lookup[r] for r in matched}),
            "explicit_negative_spans": len(negatives), "explicit_negative_matches": len(spans & negatives),
            "other_unmatched": len(spans - matched - negatives)}


def pair_counts(mentions, clusters, lookup, proposed):
    memberships = {key: set() for key in mentions}
    for i, cluster in enumerate(clusters):
        for r in cluster & lookup.keys():
            memberships[lookup[r]].add(i)
    covered = {lookup[r] for r in proposed & lookup.keys()}
    counts = Counter(dict.fromkeys(("positive_pairs", "negative_pairs", "joined_positive",
                                    "joined_negative", "covered_positive", "covered_negative",
                                    "unresolved_mentions", "unresolved_covered", "unresolved_clustered",
                                    "unresolved_joined_to_resolved"), 0))
    resolved = [k for k, m in mentions.items() if m["person_id"] is not None]
    for a, b in combinations(resolved, 2):
        kind = "positive" if mentions[a]["person_id"] == mentions[b]["person_id"] else "negative"
        counts[kind + "_pairs"] += 1
        counts["covered_" + kind] += a in covered and b in covered
        counts["joined_" + kind] += bool(memberships[a] & memberships[b])
    resolved_clusters = set().union(*(memberships[k] for k in resolved))
    for k, m in mentions.items():
        if m["person_id"] is None:
            counts["unresolved_mentions"] += 1
            counts["unresolved_covered"] += k in covered
            counts["unresolved_clustered"] += bool(memberships[k])
            counts["unresolved_joined_to_resolved"] += bool(memberships[k] & resolved_clusters)
    return dict(counts)


def distractor_pairs(mentions, negatives, lookup, proposed, clusters):
    """Count distinct document-local occurrence pairs, including unresolved humans."""
    joined = set()
    mixed = 0
    for cluster in clusters:
        people = {lookup[r] for r in cluster & lookup.keys()}
        distractors = cluster & negatives
        if people and distractors:
            mixed += 1
            joined.update((key, r) for key in people for r in distractors)
    covered_people = {lookup[r] for r in proposed & lookup.keys()}
    return {"human_distractor_pairs": len(mentions) * len(negatives),
            "covered_pairs": len(covered_people) * len(proposed & negatives),
            "joined_pairs": len(joined), "mixed_clusters": mixed,
            "clusters": len(clusters)}


def score(gold, predictions):
    if (gold.get("offset_unit") != "unicode_scalar"
            or type(gold.get("schema_version")) is not int or gold["schema_version"] != 1):
        raise ValueError("expected version 1 person annotations with Unicode scalar offsets")
    docs = indexed(gold["documents"], "gold documents")
    rows = indexed(predictions["results"], "prediction documents")
    if docs.keys() != rows.keys() or not docs:
        raise ValueError("document IDs differ or panel is empty")
    results = []
    totals = {}
    for key, doc in docs.items():
        mentions, lookups, negatives = annotation_maps(doc)
        proposed, clusters = model_spans(doc, rows[key])
        clustered = set().union(*clusters)
        scores = {}
        for contract, lookup in lookups.items():
            scores[contract] = {
                "candidates": selection(proposed, lookup, negatives, len(mentions)),
                "cluster_members": selection(clustered, lookup, negatives, len(mentions)),
                "local_pairs": pair_counts(mentions, clusters, lookup, proposed),
                "human_distractor": distractor_pairs(mentions, negatives, lookup, proposed, clusters),
            }
            aggregate = totals.setdefault(contract, {})
            for stage, counts in scores[contract].items():
                aggregate.setdefault(stage, Counter()).update(counts)
        results.append({"id": key, "scores": scores})
    return {"totals": totals, "documents": results, "kb_linking": "not_run",
            "scope": "Person mention annotation. Candidate proposal and local cluster membership "
                     "are counted separately. Other unmatched spans do not satisfy the selected person "
                     "boundary contract and do not exactly match a marked distractor."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gold", type=Path)
    parser.add_argument("predictions", type=Path)
    args = parser.parse_args()
    try:
        inputs = [json.loads(p.read_text(encoding="utf-8")) for p in (args.gold, args.predictions)]
        result = score(*inputs)
    except (KeyError, TypeError, OSError, ValueError) as exc:
        parser.exit(2, f"invalid person mention QA run: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
