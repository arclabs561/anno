# SPDX-License-Identifier: MIT OR Apache-2.0
"""Replay candidate policies and an empty-answer serving rule on checksum-verified AjMC pages."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from ...congress.cross_doc_partition import run_native
from ..ajmc import language, load, source_item
from .policy import CandidateIndex


FROZEN = ("native", "fold", "expanded")
VARIANTS = (*FROZEN, "empty_native")


def outcomes(hits, expected, rows):
    wrong = {b for b in hits - expected if rows[b].label.startswith("Q")}
    unknown = hits - expected - wrong
    expected_pages = {b[0] for b in expected}
    wrong_pages = {b[0] for b in wrong} - expected_pages - {b[0] for b in unknown}
    return Counter(
        {
            "queries": 1,
            "eligible": bool(expected),
            "shown": bool(hits),
            "any_correct": bool(hits & expected),
            "exact_supported": bool(expected) and hits == expected,
            "with_wrong_hit": bool(wrong),
            "with_unscored_hit": bool(unknown),
            "with_wrong_only_page": bool(wrong_pages),
            "wrong_only_pages": len(wrong_pages),
            "wrong_hits_on_supported_pages": sum(b[0] in expected_pages for b in wrong),
        }
    )


def evaluate_page(doc, gallery, rows, groups, counts, examples):
    index = CandidateIndex({b: (rows[b].name, groups[b]) for b in gallery})
    for a in sorted(b for b in rows if b[0] == doc):
        query = rows[a]
        if not query.label.startswith("Q"):
            continue
        expected = {b for b in gallery if rows[b].label == query.label}
        base = {b for b in gallery if groups[b] == groups[a]}
        before = outcomes(base, expected, rows)
        decisions = {v: index.suggest(query.name, groups[a], v) for v in FROZEN}
        # Post-hoc presentation rule evaluated on precisely the same lookup results.
        decisions["empty_native"] = (
            decisions["native"] if base else decisions["expanded"]
        )
        for variant, decision in decisions.items():
            result = base | decision.hits
            outcome = outcomes(result, expected, rows)
            tally = counts[variant]
            tally.update(outcome)
            if decision.decision != "none":
                tally[f"{decision.decision}_{decision.tier}"] += 1
            tally["gained_first_correct"] += bool(
                outcome["any_correct"] and not before["any_correct"]
            )
            tally["new_wrong_query"] += bool(
                outcome["with_wrong_hit"] and not before["with_wrong_hit"]
            )
            tally["gained_exact"] += bool(
                outcome["exact_supported"] and not before["exact_supported"]
            )
            tally["lost_exact"] += bool(
                before["exact_supported"] and not outcome["exact_supported"]
            )
            for b in sorted(decision.hits):
                row = rows[b]
                if row.label == query.label:
                    kind = "correct"
                elif row.label.startswith("Q"):
                    kind = "wrong"
                else:
                    kind = "unscored"
                tally["added_" + kind] += 1
                if row.name != query.name:
                    tally["different_spelling_" + kind] += 1
                key = variant + "/" + kind
                if key not in examples and row.name != query.name:
                    examples[key] = {
                        "query": [*a, query.name, query.label],
                        "hit": [*b, row.name, row.label],
                        "tier": decision.tier,
                    }


def empty_counts():
    return {variant: Counter() for variant in VARIANTS}


def evaluate(docs, binary):
    rows = {
        (doc, i): r for doc, mentions in docs.items() for i, r in enumerate(mentions)
    }
    all_groups = run_native(binary, docs)
    totals = {
        "all_pages": empty_counts(),
        "language": empty_counts(),
        "source_item": empty_counts(),
    }
    examples = defaultdict(dict)
    for doc in sorted(docs):
        if not docs[doc]:
            continue
        other_pages = [b for b in rows if b[0] != doc]
        evaluate_page(
            doc,
            other_pages,
            rows,
            all_groups,
            totals["all_pages"],
            examples["all_pages"],
        )
        for split, group in (("language", language), ("source_item", source_item)):
            held_out = group(doc)
            fold = f"{split}/{held_out}"
            if fold not in totals:
                totals[fold] = empty_counts()
            gallery_docs = {
                d: mentions for d, mentions in docs.items() if group(d) != held_out
            }
            gallery = [b for b in rows if b[0] in gallery_docs]
            groups = run_native(binary, {**gallery_docs, doc: docs[doc]})
            evaluate_page(doc, gallery, rows, groups, totals[fold], examples[fold])
    for split in ("language", "source_item"):
        for fold in (key for key in totals if key.startswith(split + "/")):
            for variant in VARIANTS:
                totals[split][variant].update(totals[fold][variant])
    return {"counts": totals, "examples": examples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("native_binary", type=Path)
    args = parser.parse_args()
    result = evaluate(load(args.directory), args.native_binary.resolve())
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
