# SPDX-License-Identifier: MIT OR Apache-2.0
"""Offline controls for committing one or several source affiliation references."""

from collections import Counter, defaultdict
import json
from pathlib import Path

from evaluate import Record
from ngram_retrieval import Query, decision, lookup


def policies(ranked):
    if ranked:
        first = (ranked[0][0], "first")
    else:
        first = (None, "no_candidate")
    return {"first": first, "gate": decision(ranked)}


def score(answers, truth, per_reference):
    owners = set(answers) - {None}
    gold = set(truth)
    result = {
        "answers": answers,
        "any_wrong_owner": bool(owners - gold),
        "complete": owners == gold,
        "no_link": not owners,
        "partial_correct_only": bool(owners) and owners < gold,
    }
    if per_reference:
        pairs = list(zip(answers, truth, strict=True))
        result["slot_correct"] = sum(owner == target for owner, target in pairs)
        result["slot_wrong"] = sum(owner is not None and owner != target for owner, target in pairs)
        result["slot_abstained"] = answers.count(None)
    return result


def check(binary):
    fixture = json.loads(Path(__file__).with_name("fixture.json").read_text())
    catalog = {
        owner: Record(names[0], tuple(names), "active", ())
        for owner, names in fixture["catalog"].items()
    }
    papers = fixture["papers"]
    queries = set()
    for _, _, refs in papers:
        strings = [text for text, _ in refs]
        queries.update(strings)
        queries.add("; ".join(strings))
    requests = [Query("authored", str(i), text, None) for i, text in enumerate(sorted(queries))]
    rankings = {q.name: ranked for q, _, _, _, ranked, _ in lookup(binary, catalog, requests)}
    # Source row order cannot supply tie-breaking evidence.
    reversed_rankings = {
        q.name: ranked for q, _, _, _, ranked, _ in lookup(binary, catalog, requests[::-1])
    }
    assert rankings == reversed_rankings

    evidence = {}
    totals = defaultdict(Counter)
    strata = defaultdict(lambda: defaultdict(Counter))
    for label, category, refs in papers:
        inputs = [text for text, _ in refs]
        truth = [owner for _, owner in refs]
        arms = {}
        for presentation, terms in (("flat", ["; ".join(inputs)]), ("source_fields", inputs)):
            for policy in ("first", "gate"):
                answers = [policies(rankings[term])[policy][0] for term in terms]
                arm = f"{presentation}_{policy}"
                values = score(answers, truth, presentation == "source_fields")
                arms[arm] = values
                for metric, value in values.items():
                    if metric != "answers":
                        totals[arm][metric] += int(value)
                        strata[category][arm][metric] += int(value)
        evidence[label] = {"stratum": category, "truth": truth, "arms": arms}

    def answer(case, arm="source_fields_gate"):
        return evidence[case]["arms"][arm]["answers"]

    assert answer("ambiguous_east") == answer("ambiguous_west") == [None]
    assert answer("ambiguous_east", "source_fields_first") == ["east"]
    assert answer("ambiguous_west", "source_fields_first") == ["east"]
    assert answer("absent_namesake") == ["salt"]
    assert "missing_salt" not in catalog
    assert answer("two_exact") == ["coil", "reed"]
    assert answer("three_exact") == ["mesa", "reed", "fen"]
    assert answer("two_exact", "flat_gate") == answer("three_exact", "flat_gate") == [None]
    assert answer("two_one_missing") == ["fen", None]
    assert answer("two_one_ambiguous") == ["reed", None]

    # Remove the labeled owner before retrieval and inspect the entire resulting candidate block.
    q = Query("authored", "masked_control", "Quiet Reed College", None)
    assert decision(rankings[q.name]) == ("reed", "accepted")
    (_, _, _, _, masked_ranked, block), = lookup(
        binary, catalog, [q], catalog_masks={q.source: {"reed"}}
    )
    assert "reed" not in block and decision(masked_ranked)[0] is None
    summary = {
        "n": len(papers),
        "slots": sum(len(refs) for _, _, refs in papers),
        "by_policy": dict(totals),
        "by_stratum": dict(strata),
    }
    return summary, evidence
