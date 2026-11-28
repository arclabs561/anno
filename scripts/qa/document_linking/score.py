"""Score saved stage outputs against the separate authored review."""
import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path

from run import HERE, read


def span(m):
    return m["start"], m["end"]


def tally(want, got):
    return {"matched": len(want & got), "reference": len(want), "emitted": len(got)}


def score(work):
    refs = {d["id"]: d for d in read(HERE / "reviewed.json")}
    result = {"documents": len(refs), "candidate_count": len(read(HERE / "candidates.json"))}
    ner = Counter()
    for doc_id, ref in refs.items():
        names = set(map(tuple, ref["named_spans"]))
        gold = {(s, e["type"]) for e in ref["entities"] for m in e["mentions"]
                if (s := span(m)) in names}
        detected = {(span(m), m["type"]) for m in read(work / f"{doc_id}-ner.json")["entities"]}
        ner.update(tally(gold, detected))
    result["named_span_and_type"] = dict(ner)
    decisions = {}
    for arm in ("predicted", "reviewed"):
        docs = {d["id"]: d for d in read(work / f"{arm}-documents.json")}
        relations = {d["id"]: d["relations"] for d in read(work / f"{arm}-relations.json")}
        scores = {d["id"]: d["relations"] for d in read(work / f"{arm}-scores.json")}
        link_rows = [json.loads(line) for line in (work / f"{arm}-links.jsonl").read_text().splitlines()]
        counts = {k: Counter() for k in ("mentions", "coref_pairs_on_reviewed_spans", "relations", "links")}
        facts = []
        decisions[arm] = {}
        for doc_id, ref in refs.items():
            names = set(map(tuple, ref["named_spans"]))
            gold = {span(m): e["id"] for e in ref["entities"] for m in e["mentions"]}
            gold_pairs = {tuple(sorted((a, b))) for a, b in combinations(gold, 2) if gold[a] == gold[b]}
            projected, pairs, spans = {}, set(), set()
            for e in docs[doc_id]["entities"]:
                current = {span(m) for m in e["mentions"]}
                spans.update(current)
                known = current.intersection(gold)
                pairs.update(tuple(sorted(p)) for p in combinations(known, 2))
                matches = {gold[s] for s in known}
                if len(matches) == 1:
                    projected[e["id"]] = matches.pop()
            counts["mentions"].update(tally(set(gold), spans))
            counts["coref_pairs_on_reviewed_spans"].update(tally(gold_pairs, pairs))
            expected = {tuple(r) for r in ref["relations"]}
            def edge(row):
                return projected.get(row["head"]), row["predicate"], projected.get(row["tail"])
            matched = {edge(r) for r in relations[doc_id]} & expected
            counts["relations"].update(matched=len(matched), reference=len(expected), emitted=len(relations[doc_id]))
            for relation in sorted(expected):
                edge_scores = [r["score"] for r in scores[doc_id] if edge(r) == relation]
                facts.append({"document": doc_id, "relation": list(relation),
                              "max_score": max(edge_scores) if edge_scores else None})
            seen = set()
            counts["links"]["reference"] += len(gold)
            for row in link_rows:
                if row["doc_id"] != doc_id:
                    continue
                s = span(row)
                if docs[doc_id]["text"][s[0]:s[1]] != row["text"]:
                    raise ValueError("linked source span disagrees with the original text")
                identity = row["entity_id"], s
                if identity in seen:
                    raise ValueError(f"duplicate linked occurrence {identity!r}")
                seen.add(identity)
                counts["links"]["emitted"] += 1
                counts["links"]["nil"] += row["wikidata_id"] is None
                if s in gold:
                    qid = row["wikidata_id"]
                    correct = qid == ref["wikidata"][gold[s]]
                    counts["links"]["correct_reviewed"] += correct
                    counts["links"]["nil_reviewed"] += qid is None
                    counts["links"]["wrong_id_reviewed"] += qid is not None and not correct
                    counts["links"]["assessed"] += 1
                    kind = "name" if s in names else "pronoun"
                    counts["links"][f"correct_{kind}"] += correct
                    counts["links"][f"assessed_{kind}"] += 1
                    decisions[arm][doc_id, s] = qid
            expected_links = {(e["id"], span(m)) for e in docs[doc_id]["entities"] for m in e["mentions"]}
            if expected_links != seen:
                raise ValueError("missing or unknown linked occurrences")
        result[arm] = {k: dict(v) for k, v in counts.items()}
        result[arm]["reference_fact_scores"] = facts
    common = decisions["predicted"].keys() & decisions["reviewed"].keys()
    result["paired_link_decisions"] = {
        "reviewed_occurrences_in_both": len(common),
        "different_qid_or_nil": sum(decisions["predicted"][i] != decisions["reviewed"][i] for i in common),
    }
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    print(json.dumps(score(p.parse_args().work), indent=2))
