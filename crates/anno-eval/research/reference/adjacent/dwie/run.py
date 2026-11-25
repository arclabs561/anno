"""Compare fixed surface clusters, KB-link coalescing, and joint reranking on DWIE."""
import argparse
import itertools
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

REV = "575e0be3e6f980b77152f587b26530082f94a6c1"
NIL = "<NIL>"


def surface(mention):
    return " ".join(mention["text"].casefold().split())


def concept_types(concept):
    return {s for s in concept["tags"] if s.startswith("type::")}


def learn(docs):
    links = defaultdict(Counter)
    surface_types = defaultdict(Counter)
    page_types = defaultdict(Counter)
    facts = defaultdict(set)
    for doc in docs:
        concepts = {c["concept"]: c for c in doc["concepts"]}
        for c in concepts.values():
            if c.get("link"):
                page_types[c["link"]][frozenset(concept_types(c))] += 1
        for m in doc["mentions"]:
            c = concepts[m["concept"]]
            links[surface(m)][c.get("link", NIL)] += 1
            surface_types[surface(m)][frozenset(concept_types(c))] += 1
        for r in doc["relations"]:
            a, b = (concepts[i].get("link") for i in (r["s"], r["o"]))
            if a and b:
                facts[a, b].add(r["p"])
    return links, surface_types, page_types, facts


def mode(counter, default):
    return counter.most_common(1)[0][0] if counter else default


def local_scores(mentions, prior):
    scores = Counter()
    for m in mentions:
        scores.update(dict(zip(m.get("candidates", []), m.get("scores", []), strict=True)))
    for k in scores:
        scores[k] /= len(mentions)
    if prior:
        for k, n in prior.items():
            scores[k] += n / prior.total()
    if not scores:
        scores[NIL] = 1.
    return scores


def agreement(types_a, types_b):
    a, b = types_a - {"type::entity"}, types_b - {"type::entity"}
    return len(a & b) / len(a | b) if a and b else 0.


def predict(mentions, trained, arm):
    links, surface_types, page_types, facts = trained
    groups = defaultdict(list)
    for m in mentions:
        groups[surface(m)].append(m)
    scores = {key: local_scores(group, links.get(key)) for key, group in groups.items()}
    chosen = {key: mode(weights, NIL) for key, weights in scores.items()}
    if arm in {"joint", "joint_without_relations"}:
        # A synchronous pass uses frozen, predicted, other-surface page assignments.
        original = chosen.copy()
        for key, weights in scores.items():
            others = {page for other, page in original.items() if other != key and page != NIL}
            types = mode(surface_types.get(key), frozenset())
            def utility(page):
                page_schema = mode(page_types.get(page), frozenset())
                connected = sum((page, q) in facts or (q, page) in facts for q in others)
                return (weights[page] + .15 * (page in others) + .15 * agreement(types, page_schema)
                        + (.10 * min(connected, 2) if arm == "joint" else 0.))
            chosen[key] = max(weights, key=utility)
    labels, direct, typed = [], [], set()
    for i, m in enumerate(mentions):
        key = surface(m)
        page = chosen[key]
        cluster = ("page", page) if page != NIL and arm != "pipeline" else ("surface", key)
        labels.append(cluster)
        direct.append(page)
        types = mode(page_types.get(page), mode(surface_types.get(key), frozenset()))
        typed.update((i, t) for t in types)
    return labels, direct, typed


def observed_pairs(keys):
    return {(i, j) for i, j in itertools.combinations(range(len(keys)), 2) if keys[i] == keys[j]}


def count(total, name, predicted, gold):
    total[name + "_correct"] += len(predicted & gold)
    total[name + "_predicted"] += len(predicted)
    total[name + "_gold"] += len(gold)


def evaluate(doc, trained, arm):
    mentions = doc["mentions"]
    concepts = {c["concept"]: c for c in doc["concepts"]}
    ids = [m["concept"] for m in mentions]
    members = defaultdict(list)
    for i, entity in enumerate(ids):
        members[entity].append(i)
    gold_links = [concepts[key].get("link", NIL) for key in ids]
    labels, pages, typed = predict(mentions, trained, arm)
    stats = Counter()
    stats["mentions"] = len(mentions)
    stats["link_matches_including_nil"] = sum(a == b for a,b in zip(pages,gold_links))
    count(stats,"linked", {(i,p) for i,p in enumerate(pages) if p != NIL},
          {(i,p) for i,p in enumerate(gold_links) if p != NIL})
    count(stats,"identity", observed_pairs(labels), observed_pairs(ids))
    count(stats,"typing", typed, {(i,t) for i,c in enumerate(ids) for t in concept_types(concepts[c])})
    truth = {(i,j,r["p"]) for r in doc["relations"] for i in members[r["s"]]
             for j in members[r["o"]] if i != j}
    predicted = {(i,j,r) for i,a in enumerate(pages) for j,b in enumerate(pages)
                 if i != j and a != NIL and b != NIL for r in trained[3].get((a,b), ())}
    count(stats,"relation",predicted,truth)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path)
    args = parser.parse_args()
    docs = [json.loads(p.read_text()) for p in sorted(args.annotations.glob("*.json"))]
    train = [d for d in docs if "train" in d["tags"]]
    test = [d for d in docs if "test" in d["tags"]]
    if (len(train),len(test)) != (702,100):
        raise ValueError("expected the pinned DWIE annotation split")
    trained = learn(train)
    if {d["id"] for d in train} & {d["id"] for d in test}:
        raise ValueError("overlapping document identifiers")
    per_document = {}
    result = {"revision": REV, "train_documents":len(train),"test_documents":len(test),"arms":{}}
    for arm in ("pipeline","shared_page","joint_without_relations","joint"):
        per_document[arm] = [evaluate(d, trained, arm) for d in test]
        stats = sum(per_document[arm], Counter())
        for key in ("linked","identity","typing","relation"):
            stats[key+"_micro_f1"] = 2 * stats[key+"_correct"] / (stats[key+"_gold"]+stats[key+"_predicted"])
        result["arms"][arm] = dict(stats)
    rng = random.Random(7)
    draws = [rng.choices(range(len(test)), k=len(test)) for _ in range(2000)]
    intervals = {}
    for task in ("identity", "linked", "typing", "relation"):
        def pooled_f1(arm, draw):
            rows = per_document[arm]
            hits = sum(rows[i][task + "_correct"] for i in draw)
            size = sum(rows[i][task + "_gold"] + rows[i][task + "_predicted"] for i in draw)
            return 2 * hits / size
        differences = sorted(pooled_f1("joint", draw) - pooled_f1("shared_page", draw)
                             for draw in draws)
        intervals[task] = [differences[49], differences[1949]]
    result["paired_joint_minus_shared_page_f1_bootstrap_95"] = intervals
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__ == "__main__":
    main()
