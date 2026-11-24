"""Compare compact person displays from source-bound coreference and NER receipts."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from check_span_assertions import check_range, indexed, jsonl_rows, objects, valid_label
from score_person_mentions import annotation_maps, gold_span, model_spans, span


def score_document(doc, coref, ner) -> dict[str, Any]:
    mentions, lookups, _ = annotation_maps(doc)
    text = doc["text"]
    if ner.get("source_sha256") != coref.get("source_sha256"):
        raise ValueError("NER and coreference source hashes differ")
    _, clusters = model_spans(doc, coref)
    phrases = set().union(*clusters)
    ner_people = set()
    for entity in objects(ner["entities"], "NER entities"):
        check_range(entity, text, doc["id"])
        if not valid_label(entity.get("type")):
            raise ValueError("invalid NER type")
        if entity["type"] == "PER":
            ner_people.add((entity["start"], entity["end"]))

    targets = {}
    for key, m in mentions.items():
        if m["display_highlight"] is not None:
            target = span(m["display_highlight"], text)
            if target != tuple(m["highlight"]):
                raise ValueError(
                    "display highlight differs from annotated lexical highlight"
                )
            if target in targets:
                raise ValueError("two display targets share the same source range")
            targets[target] = key
    controls = {}
    for control in objects(doc["display_controls"], "display controls"):
        r = gold_span(control["span"], control["surface"], text, "display control")
        owner = lookups["referential"].get(r)
        if r in controls or (
            owner is not None and mentions[owner]["display_highlight"] is not None
        ):
            raise ValueError("duplicate or conflicting protected display range")
        controls[r] = control["reason"]
    proposals = {
        "full": set(),
        "standalone_ner": ner_people,
        "contained_all": set(),
        "unique_contained": set(),
    }
    phrase_counts = {}
    detail: list[dict[str, Any]] = []
    for phrase in sorted(phrases):
        owner = lookups["referential"].get(phrase)
        protected = phrase in controls
        inside = {s for s in ner_people if phrase[0] <= s[0] and s[1] <= phrase[1]}
        views = {
            "full": {phrase},
            "contained_all": inside,
            "unique_contained": inside if len(inside) == 1 else set(),
        }
        target = mentions[owner]["display_highlight"] if owner is not None else None
        target = tuple(target) if target is not None and not protected else None
        row = {
            "phrase": phrase,
            "owner": owner,
            "protected": protected,
            "contained_count": len(inside),
            "views": {},
            "cluster_indices": [
                i for i, members in enumerate(clusters) if phrase in members
            ],
            "protection_reason": controls.get(phrase),
        }
        for name, displayed in views.items():
            proposals[name].update(displayed)
            known = protected or owner is not None
            wrong = displayed - {target} if known else set()
            cross_person = 0
            other_occurrence = 0
            for s in wrong:
                emitted_owner = targets.get(s)
                if (
                    owner is not None
                    and emitted_owner is not None
                    and owner != emitted_owner
                ):
                    other_occurrence += 1
                    left, right = (
                        mentions[owner]["person_id"],
                        mentions[emitted_owner]["person_id"],
                    )
                    cross_person += (
                        left is not None and right is not None and left != right
                    )
            counts = Counter(
                phrases=1,
                emitted=len(displayed),
                abstained=int(not displayed),
                eligible=int(target is not None),
                correct=int(target in displayed),
                missed=int(target is not None and target not in displayed),
                judged_wrong=len(wrong),
                unattributed_emissions=0 if known else len(displayed),
                unassessed_phrases=int(not known),
                protected_phrases=int(protected),
                protected_emissions=len(displayed) if protected else 0,
                other_occurrence=other_occurrence,
                known_wrong_person=cross_person,
            )
            phrase_counts.setdefault(name, Counter()).update(counts)
            row["views"][name] = {
                "display": sorted(displayed),
                "judged_wrong": sorted(wrong),
                "known_wrong_person": cross_person,
            }
        detail.append(row)
    recovered = [set() for _ in clusters]
    for row in detail:
        for index in row["cluster_indices"]:
            recovered[index].add(row["phrase"])
    if recovered != clusters:
        raise ValueError("display views changed source memberships")
    if not proposals["unique_contained"] <= ner_people:
        raise ValueError("projected offset outside standalone PER predictions")
    global_counts = {}
    for name, marks in proposals.items():
        hits = marks & targets.keys()
        global_counts[name] = {
            "eligible": len(targets),
            "correct": len(hits),
            "wrong": len(marks - hits),
            "missed": len(targets) - len(hits),
            "emitted": len(marks),
        }
    return {
        "id": doc["id"],
        "source_sha256": coref["source_sha256"],
        "occurrences": [
            {
                "id": key,
                "display": r,
                "covered_by": [name for name, marks in proposals.items() if r in marks],
            }
            for r, key in targets.items()
        ],
        "unmatched_displays": {
            name: sorted(marks - targets.keys()) for name, marks in proposals.items()
        },
        "pair_id": doc.get("pair_id"),
        "variant": doc.get("variant"),
        "global": global_counts,
        "phrases": phrase_counts,
        "detail": detail,
        "standalone_removed": {
            "correct": len(
                (ner_people - proposals["unique_contained"]) & targets.keys()
            ),
            "wrong": len((ner_people - proposals["unique_contained"]) - targets.keys()),
        },
        "source_references": {
            "covered_occurrences": len(
                {
                    lookups["referential"][r]
                    for r in phrases & lookups["referential"].keys()
                }
            ),
            "gold_occurrences": len(mentions),
        },
        "projection_abstentions": {
            "zero": sum(r["contained_count"] == 0 for r in detail),
            "one": sum(r["contained_count"] == 1 for r in detail),
            "multiple": sum(r["contained_count"] > 1 for r in detail),
        },
    }


def score(gold, coref, ner) -> dict[str, Any]:
    if (
        gold.get("offset_unit") != "unicode_scalar"
        or type(gold.get("schema_version")) is not int
        or gold["schema_version"] != 1
    ):
        raise ValueError("expected version 1 Unicode scalar person annotations")
    docs = indexed(gold["documents"], "gold")
    f = indexed(coref["results"], "coreference")
    n = indexed(ner, "NER")
    if not docs or docs.keys() != f.keys() or docs.keys() != n.keys():
        raise ValueError("all document IDs must match")
    rows = [score_document(d, f[k], n[k]) for k, d in docs.items()]
    totals: dict[str, Any] = {"global": {}, "phrases": {}}
    for row in rows:
        for stage, aggregate in totals.items():
            for name, counts in row[stage].items():
                aggregate.setdefault(name, Counter()).update(counts)
    scores = totals["global"]
    best = max(
        ("standalone_ner", "full"),
        key=lambda k: (scores[k]["correct"], -scores[k]["wrong"]),
    )
    candidate = scores["unique_contained"]
    new_wrong = []
    for row in rows:
        for event in row["detail"]:
            projected = event["views"]["unique_contained"]["display"]
            if (
                event["protected"]
                and projected
                and projected != event["views"]["full"]["display"]
            ):
                new_wrong.append(
                    {
                        "id": row["id"],
                        "phrase": event["phrase"],
                        "display": event["views"]["unique_contained"]["display"],
                    }
                )
    for field in ("projection_abstentions", "standalone_removed", "source_references"):
        counts = Counter()
        for row in rows:
            counts.update(row[field])
        totals[field] = counts
    gate = {
        "strict_correct_gain": candidate["correct"] > scores[best]["correct"],
        "wrong_no_higher": candidate["wrong"] <= scores[best]["wrong"],
        "no_new_protected_attribution": not new_wrong,
        "source_memberships_preserved": True,
    }
    return {
        "totals": totals,
        "documents": rows,
        "stronger_baseline": best,
        "gate": gate,
        "passes": all(gate.values()),
        "protected_per_attributions": new_wrong,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("gold", "coref", "ner"):
        parser.add_argument(arg, type=Path)
    args = parser.parse_args()
    try:
        result = score(
            json.loads(args.gold.read_text()),
            json.loads(args.coref.read_text()),
            jsonl_rows(args.ner.read_text()),
        )
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f"invalid display experiment: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
