"""Rank the same source-backed suggestions on a shared predicted history."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from entity_memory_replay import Memory, link_confidence, load_conll, load_features


def support(ranked, candidate):
    """Apply the exemplar confidence rule to this target, even when it is not first."""
    distance = next(score for entity, score in ranked if entity == candidate)
    alternative = next((score for entity, score in ranked if entity != candidate), distance)
    return -distance + min(alternative - distance, 1.0)


def trace(doc, features, budget, threshold):
    exemplar, box = [Memory(policy, features.shape[1], budget) for policy in ("exemplars", "box")]
    midpoint = Memory("midpoint", features.shape[1], budget)
    latest, members, rows = {}, {}, []
    for step, (mention, query) in enumerate(zip(doc.mentions, features, strict=True)):
        surface = tuple(token.casefold() for token in doc.tokens[mention.start:mention.end])
        ranked = exemplar.rank(query, 2)
        assignment = step
        if ranked:
            target = ranked[0][0]
            confidence = link_confidence(ranked)
            rows.append((confidence, support(midpoint.rank(query, midpoint.capacity), target),
                         support(box.rank(query, box.capacity), target),
                         latest[target][0] == mention.entity, members[target] == {mention.entity},
                         latest[target][1] == surface))
            if confidence >= threshold:
                assignment = target
        latest[assignment] = mention.entity, surface
        members.setdefault(assignment, set()).add(mention.entity)
        exemplar.update(assignment, query, step)
        box.update(assignment, query, step)
        midpoint.update(assignment, query, step)
    return np.asarray(rows, dtype=float).reshape(-1, 6)


def evaluate(rows, mentions, quota):
    keep = min(int(mentions * quota), len(rows))
    ranks = [rankdata(rows[:, i]) for i in range(3)]
    scores = [ranks[0], ranks[0] + ranks[1], ranks[0] + ranks[2]]
    # Exact surface agreement takes priority, with the original score breaking ties.
    priority = rows[:, 5] * (2 * len(rows) + 1)
    scores.extend([priority + scores[0], priority + scores[2]])
    result = []
    for score in scores:
        indices = np.argsort(-score, kind="stable")[:keep]
        result.append([keep, int(keep - rows[indices, 3].sum()), int(keep - rows[indices, 4].sum())])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("receipt", "corpus", "features"):
        parser.add_argument("--" + arg, type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    if "folds" in receipt:
        groups = [(fold["exemplars"], receipt["dimension"], receipt["budget"])
                  for fold in receipt["folds"].values()]
    else:
        config = receipt["config"]
        groups = [(receipt["policies"]["exemplars"], config["dimension"], config["budget"])]
    documents = {}
    different_text = []
    quotas = (0.10, 0.20, 0.35)
    for policy, dimension, budget in groups:
        for item in policy["closed_loop_documents"]:
            name = item["document"]
            if name in documents:
                raise ValueError(f"duplicate evaluation document {name}")
            doc = load_conll(args.corpus / (name + ".conll"))
            features = load_features(doc, dimension, args.features)
            rows = trace(doc, features, budget, policy["threshold"])
            accepted = rows[:, 0] >= policy["threshold"]
            wrong = int((1 - rows[accepted, 4]).sum())
            if accepted.sum() != item.get("links", 0) or wrong != item.get("wrong_attachments", 0):
                raise ValueError(f"{name}: predicted history differs from the source replay")
            unmatched = rows[rows[:, 5] == 0]
            different_text.append(evaluate(unmatched, len(unmatched), 0.20)[:3])
            documents[name] = [evaluate(rows, len(doc.mentions), quota) for quota in quotas]
    counts = np.array(list(documents.values()))
    rng = np.random.default_rng(7)
    samples = rng.integers(len(counts), size=(2000, len(counts)))
    pooled = counts[samples].sum(axis=1)
    differences = (pooled[:, :, :1, 1:] - pooled[:, :, 1:, 1:]) / pooled[:, :, :1, :1]
    surface_gain = (pooled[:, :, 3, 1:] - pooled[:, :, 4, 1:]) / pooled[:, :, 3, :1]
    different_counts = np.array(different_text)
    sub = different_counts[samples].sum(axis=1)
    different_gain = (sub[:, 0, 1:] - sub[:, 2, 1:]) / sub[:, 0, :1]
    print(json.dumps(dict(encoder=receipt["encoder"], quotas=quotas,
                          arms=["exemplar", "exemplar+midpoint", "exemplar+box",
                                "surface/exemplar", "surface/exemplar+box"],
                          outcomes=["reviews", "wrong_latest_witness", "impure_or_wrong_target"],
                          totals=counts.sum(axis=0).tolist(),
                          baseline_minus_augmented_rate_ci95=np.quantile(
                              differences, [0.025, 0.975], axis=0).tolist(),
                          surface_minus_surface_box_ci95=np.quantile(
                              surface_gain, [0.025, 0.975], axis=0).tolist(),
                          different_text_top_fifth=dict(arms=["exemplar", "exemplar+midpoint", "exemplar+box"],
                              totals=different_counts.sum(axis=0).tolist(),
                              exemplar_minus_box_ci95=np.quantile(
                                  different_gain, [0.025, 0.975], axis=0).tolist()),
                          documents=documents), allow_nan=False, indent=1))


if __name__ == "__main__":
    main()
