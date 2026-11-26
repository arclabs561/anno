# SPDX-License-Identifier: MIT OR Apache-2.0
"""Prioritize distinct directed span pairs under the pinned decoder's typed edge cap."""

from collections import defaultdict
from dataclasses import replace
from itertools import islice


def candidates(scores, config):
    support = {m.key: m.logit for m in scores.mentions}
    grouped = defaultdict(lambda: defaultdict(list))
    for edge in scores.edges:
        gate = config.relation_role_threshold if edge.candidate_threshold is None else edge.candidate_threshold
        if edge.probability >= gate:
            grouped[edge.relation_type][edge.head[1:], edge.tail[1:]].append(edge)
    cap = min(config.relation_pair_cap, config.max_edges_per_type)
    selected, stock = [], []
    for groups in grouped.values():
        tickets = []
        for alternatives in groups.values():
            # Endpoint-type expansion shares a relation logit on this boundary checkpoint.
            if any(edge.logit != alternatives[0].logit for edge in alternatives):
                raise ValueError("typed alternatives do not share a relation logit")
            alternatives.sort(key=lambda e: (-(support[e.head] + support[e.tail]), e.head, e.tail))
            tickets.extend(enumerate(alternatives[:cap]))
        tickets.sort(key=lambda entry: (entry[0], -entry[1].logit, entry[1].head[1:], entry[1].tail[1:]))
        selected.extend(edge for _, edge in islice(tickets, cap))
        flat = (edge for alternatives in groups.values() for edge in alternatives)
        stock.extend(islice(sorted(flat, key=lambda e: (-e.logit, str(e.head), str(e.tail))), cap))
    if len(stock) != len(selected):
        raise ValueError("changed the number of available typed candidate slots")
    return stock, selected


def trial(scores, config, gold, decoded, decode, args):
    stock, changed = candidates(scores, config)
    def occurrence(key):
        label, first, end = key
        return label, scores.start_mappings[first], scores.end_mappings[end - 1]
    def edges(raw):
        return {(occurrence(e.head), e.relation_type, occurrence(e.tail)) for e in raw}
    stock, alternatives = edges(stock), edges(changed)
    result = decode(replace(scores, edges=tuple(changed)), *args).to_dict(include_confidence=True, include_spans=True)
    entities = {n["id"]: (n["type"], n["start"], n["end"]) for n in result["entities"]}
    output = {(entities[e["head"]], e["type"], entities[e["tail"]]) for e in result["relations"]}
    if not decoded <= stock or not output <= alternatives:
        raise ValueError("decoder produced an edge absent from its capped input")
    def paired(before, after):
        return {"gold_before": len(before & gold), "gold_after": len(after & gold),
                "added_gold": len((after - before) & gold), "removed_gold": len((before - after) & gold),
                "added_unmatched": len((after - before) - gold), "removed_unmatched": len((before - after) - gold),
                "before": len(before), "after": len(after)}
    return {"cap": paired(stock, alternatives), "output": paired(decoded, output)}
