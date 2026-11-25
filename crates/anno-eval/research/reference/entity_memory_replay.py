"""Compare document-local entity summaries on a causal gold-mention replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class Mention:
    start: int
    end: int
    entity: str


@dataclass
class Document:
    name: str
    tokens: list[str]
    mentions: list[Mention]


def load_conll(path: Path) -> Document:
    """Read one LitBank CoNLL document and retain nested and singleton mentions."""
    tokens: list[str] = []
    mentions: list[Mention] = []
    opened: dict[str, list[int]] = defaultdict(list)
    beginnings = endings = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            beginnings += line.startswith("#begin document")
            endings += line.startswith("#end document")
            continue
        if not line.strip():
            continue
        columns = line.split("\t")
        if len(columns) < 5:
            raise ValueError(f"{path}: expected tab-separated CoNLL columns")
        pos = len(tokens)
        tokens.append(columns[3])
        for entry in columns[-1].strip().split("|"):
            if entry in {"", "-", "_"}:
                continue
            entity = entry.strip("()")
            if entry.startswith("("):
                opened[entity].append(pos)
            if entry.endswith(")"):
                if not opened[entity]:
                    raise ValueError(f"{path}: unmatched closing span at {pos}")
                mentions.append(Mention(opened[entity].pop(), pos + 1, entity))
            if not (entry.startswith("(") or entry.endswith(")")):
                raise ValueError(f"{path}: invalid mention annotation {entry!r}")
    if beginnings != 1 or endings != 1 or any(opened.values()) or not mentions:
        raise ValueError(f"{path}: expected one complete annotated document")
    mentions.sort(key=lambda m: (m.end, m.start, m.entity))
    if len({(m.start, m.end) for m in mentions}) != len(mentions):
        raise ValueError(f"{path}: duplicate occurrence coordinates")
    return Document(path.stem, tokens, mentions)


def prefix_vector(prefix: list[str], start: int, dimension: int) -> np.ndarray:
    """Frozen lexical pilot encoder. The caller supplies only the visible prefix."""
    hashed = np.zeros(dimension, dtype=np.float32)
    for kind, words, weight in (
        (b"m", prefix[start:], 2.0),
        (b"c", prefix[max(0, start - 12):start], 1.0),
    ):
        for token in words:
            digest = hashlib.blake2b(kind + token.casefold().encode(), digest_size=8).digest()
            hashed[int.from_bytes(digest[:4], "little") % dimension] += (
                weight if digest[4] & 1 else -weight
            )
    norm = np.linalg.norm(hashed)
    return hashed / norm if norm else hashed


def encode(document: Document, dimension: int) -> np.ndarray:
    return np.stack([
        prefix_vector(document.tokens[:mention.end], mention.start, dimension)
        for mention in document.mentions
    ])


POLICIES = ("centroid", "exemplars", "box", "midpoint", "diagonal")


class Memory:
    """Array-backed LRU. Metadata and retained vector payloads count toward the cap."""

    def __init__(self, policy: str, dimension: int, budget: int):
        self.policy = policy
        vectors = 1 if policy == "centroid" else 2
        self.capacity = budget // (vectors * dimension * 4 + 3 * 8)
        if self.capacity == 0:
            raise ValueError("byte budget cannot hold one entity")
        self.vectors = np.zeros((self.capacity, vectors, dimension), dtype=np.float32)
        self.ids = np.full(self.capacity, -1, dtype=np.int64)
        self.last = np.zeros(self.capacity, dtype=np.int64)
        self.counts = np.zeros(self.capacity, dtype=np.int64)
        self.bytes = sum(a.nbytes for a in (self.vectors, self.ids, self.last, self.counts))

    def rank(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        usable = np.flatnonzero(self.ids >= 0)
        if not len(usable):
            return []
        vectors = self.vectors[usable]
        center = vectors[:, 0]
        if self.policy == "exemplars":
            distances = np.sum((vectors - query) ** 2, axis=2)
            distances[self.counts[usable] == 1, 1] = np.inf
            scores = distances.min(axis=1)
        elif self.policy in {"box", "midpoint"}:
            upper = vectors[:, 1]
            midpoint = (center + upper) * 0.5
            center_distance = np.sum((query - midpoint) ** 2, axis=1)
            scores = center_distance
            if self.policy == "box":
                outside = np.maximum(np.maximum(center - query, query - upper), 0)
                scores = np.sum(outside ** 2, axis=1) + 0.25 * center_distance
        elif self.policy == "diagonal":
            variance = vectors[:, 1] / np.maximum(self.counts[usable, None] - 1, 1)
            scale = 1 + variance * query.size
            scores = np.sum((query - center) ** 2 / scale, axis=1)
            scores += np.mean(np.log(scale), axis=1)
        else:
            scores = np.sum((query - center) ** 2, axis=1)
        order = np.lexsort((self.ids[usable], -self.last[usable], scores))[:k]
        return [(int(self.ids[usable[i]]), float(scores[i])) for i in order]

    def update(self, entity: int, query: np.ndarray, step: int) -> None:
        matches = np.flatnonzero(self.ids == entity)
        if len(matches):
            slot = int(matches[0])
            count = int(self.counts[slot])
            first = self.vectors[slot, 0]
            if self.policy == "exemplars":
                # Retain one early and one current exemplar.
                self.vectors[slot, 1] = query
            elif self.policy in {"box", "midpoint"}:
                np.minimum(first, query, out=first)
                upper = self.vectors[slot, 1]
                np.maximum(upper, query, out=upper)
            else:
                delta = query - first
                first += delta / (count + 1)
                if self.policy == "diagonal":
                    self.vectors[slot, 1] += delta * (query - first)
            self.counts[slot] += 1
        else:
            free = np.flatnonzero(self.ids < 0)
            slot = int(free[0]) if len(free) else int(np.argmin(self.last))
            self.ids[slot] = entity
            self.counts[slot] = 1
            self.vectors[slot, 0] = query
            if self.policy in {"box", "midpoint", "exemplars"}:
                self.vectors[slot, 1] = query
            elif self.policy == "diagonal":
                self.vectors[slot, 1] = 0
        self.last[slot] = step


def cluster_scores(gold: list[int], predicted: list[int]) -> dict[str, float]:
    """Singleton-inclusive MUC, B-cubed and CEAF-e over one shared mention set."""
    if len(gold) != len(predicted) or not gold:
        raise ValueError("gold and prediction must cover the same nonempty mention set")
    _, g = np.unique(gold, return_inverse=True)
    _, p = np.unique(predicted, return_inverse=True)
    overlap = np.zeros((int(g.max()) + 1, int(p.max()) + 1), dtype=np.int64)
    np.add.at(overlap, (g, p), 1)
    gs = overlap.sum(axis=1)
    ps = overlap.sum(axis=0)

    def ratio(n, d):
        return float(n / d) if d else 0.0

    def f1(precision, recall):
        return ratio(2 * precision * recall, precision + recall)

    muc_links = len(g) - np.count_nonzero(overlap)
    muc = f1(ratio(muc_links, len(g) - len(ps)), ratio(muc_links, len(g) - len(gs)))
    b3 = f1(
        float(np.sum(overlap**2 / ps) / len(g)),
        float(np.sum(overlap**2 / gs[:, None]) / len(g)),
    )
    sim = 2 * overlap / (gs[:, None] + ps)
    rows, cols = linear_sum_assignment(sim, maximize=True)
    ceaf = f1(ratio(sim[rows, cols].sum(), len(ps)), ratio(sim[rows, cols].sum(), len(gs)))
    wrong_pairs = (int(np.sum(ps * (ps - 1))) - int(np.sum(overlap * (overlap - 1)))) // 2
    return {"muc": muc, "b3": b3, "ceaf_e": ceaf, "conll_document_f1": (muc + b3 + ceaf) / 3,
            "false_merge_pairs": wrong_pairs}


def link_confidence(ranked):
    score = ranked[0][1]
    margin = ranked[1][1] - score if len(ranked) > 1 else 0.0
    return -score + min(margin, 1.0)


def fixed_coverage(trace, mentions, target):
    """Retrospective score diagnostic on a single frozen gold-history trajectory."""
    keep = min(math.floor(mentions * target), len(trace))
    # Stable order makes equal confidences independent of correctness.
    selected = sorted(trace, key=lambda item: item[0], reverse=True)[:keep]
    return {"links": keep, "wrong_attachments": sum(not correct for _, correct in selected)}


def replay(document, features, policy, budget, k, long_gap, threshold=None, trace=None):
    """Use gold history when threshold is None, else close the loop on predictions."""
    memory = Memory(policy, features.shape[1], budget)
    counts = Counter()
    identifiers: dict[str, int] = {}
    last_gold: dict[int, int] = {}
    state_gold: dict[int, set[int]] = {}
    predicted: list[int] = []
    gold: list[int] = []
    for step, (mention, query) in enumerate(zip(document.mentions, features, strict=True)):
        entity = identifiers.setdefault(mention.entity, len(identifiers))
        previous = last_gold.get(entity)
        counts["mentions"] += 1
        ranked = memory.rank(query, k)
        if previous is not None:
            counts["returning"] += 1
            if mention.end - previous >= long_gap:
                counts["long_queries"] += 1
                if entity in memory.ids and threshold is None:
                    counts["long_retained"] += 1
                    counts["long_hits"] += any(candidate == entity for candidate, _ in ranked)
        else:
            counts["first_mentions"] += 1
        last_gold[entity] = mention.end
        gold.append(entity)
        if threshold is None:
            if trace is not None and ranked:
                trace.append((link_confidence(ranked), ranked[0][0] == entity))
            memory.update(entity, query, step)
            continue
        prediction = step
        if ranked:
            if link_confidence(ranked) >= threshold:
                prediction = ranked[0][0]
                counts["links"] += 1
                counts["wrong_attachments"] += state_gold[prediction] != {entity}
            else:
                counts["abstentions"] += 1
        state_gold.setdefault(prediction, set()).add(entity)
        predicted.append(prediction)
        memory.update(prediction, query, step)
    scores = cluster_scores(gold, predicted) if threshold is not None else {}
    return {"document": document.name, "bytes": memory.bytes, "capacity": memory.capacity,
            **dict(counts), **scores}


def totals(rows):
    result = Counter()
    for row in rows:
        result.update({key: row.get(key, 0) for key in (
            "mentions", "first_mentions", "returning", "long_queries", "long_retained",
            "long_hits", "links", "wrong_attachments", "false_merge_pairs", "abstentions",
        )})
    return dict(result)


def summarize(rows):
    t = totals(rows)
    return {
        **t,
        "coverage": t["links"] / t["mentions"],
        "wrong_attachment_rate": t["wrong_attachments"] / t["links"] if t["links"] else None,
        "long_recall": t["long_hits"] / t["long_queries"] if t["long_queries"] else None,
        "mean_document_conll_f1": (
            float(np.mean([row["conll_document_f1"] for row in rows]))
            if "conll_document_f1" in rows[0] else None
        ),
        "bytes": rows[0]["bytes"],
        "capacity": rows[0]["capacity"],
    }


def fit_threshold(run, target):
    # Fit on development documents only. The closed loop can make coverage nonmonotone.
    trials = []
    for threshold in np.linspace(-4, 1.01, 31):
        result = summarize(run(float(threshold)))
        trials.append((abs(result["coverage"] - target), -float(threshold), result))
    _, negative_threshold, result = min(trials, key=lambda trial: trial[:2])
    return -negative_threshold, result


def load_features(document, dimension, features):
    if features is None:
        return encode(document, dimension)
    source = features / f"{document.name}.npz"
    with np.load(source, allow_pickle=False) as data:
        expected = [(mention.start, mention.end) for mention in document.mentions]
        vectors = data["vectors"].astype(np.float32)
        if not np.array_equal(data["spans"], expected):
            raise ValueError(f"{source}: encoded spans differ from replay order")
        if vectors.shape != (len(expected), dimension) or not np.isfinite(vectors).all():
            raise ValueError(f"{source}: expected finite vectors of the configured dimension")
        norms = np.linalg.norm(vectors, axis=1)
        if np.any((norms != 0) & ~np.isclose(norms, 1, atol=1e-5, rtol=0)):
            raise ValueError(f"{source}: vectors must be unit-normalized or zero")
    return vectors


def load_split(path, dimension, features=None):
    files = sorted(path.glob("*.conll"))
    if not files:
        raise ValueError(f"{path}: no CoNLL documents")
    documents = [load_conll(file) for file in files]
    return [(document, load_features(document, dimension, features)) for document in documents]


def compare_wrong_attachments(boxes, baseline, seed=7):
    """Paired document bootstrap for baseline rate minus box rate."""
    if [row["document"] for row in boxes] != [row["document"] for row in baseline]:
        raise ValueError("comparison documents differ")
    rng = np.random.default_rng(seed)
    samples = rng.integers(len(boxes), size=(2_000, len(boxes)))

    def rate(rows):
        wrong = np.array([row.get("wrong_attachments", 0) for row in rows])[samples].sum(axis=1)
        links = np.array([row.get("links", 0) for row in rows])[samples].sum(axis=1)
        return np.divide(wrong, links, out=np.full(links.shape, np.nan), where=links > 0)

    differences = rate(baseline) - rate(boxes)
    finite = differences[np.isfinite(differences)]
    return {
        "baseline_minus_box_95_interval": list(np.quantile(finite, [0.025, 0.975]))
        if len(finite) else None,
        "bootstrap_valid": len(finite),
    }


def run_experiment(dev, test, budget, k, gap, target):
    if {d.name for d, _ in dev} & {d.name for d, _ in test}:
        raise ValueError("development and test documents overlap")
    result = {}
    for policy in POLICIES:
        def run(split, threshold):
            return [replay(d, f, policy, budget, k, gap, threshold) for d, f in split]

        threshold, dev_stats = fit_threshold(lambda value: run(dev, value), target)
        oracle = []
        fixed = []
        for document, features in test:
            trace = []
            row = replay(document, features, policy, budget, k, gap, trace=trace)
            oracle.append(row)
            fixed.append({
                "document": document.name, "mentions": row["mentions"],
                "bytes": row["bytes"], "capacity": row["capacity"],
                **fixed_coverage(trace, row["mentions"], target),
            })
        closed = run(test, threshold)
        result[policy] = {
            "threshold": threshold, "development": dev_stats,
            "oracle_history": summarize(oracle), "closed_loop": summarize(closed),
            "fixed_coverage_gold_history": summarize(fixed),
            "oracle_documents": oracle, "closed_loop_documents": closed,
            "fixed_coverage_documents": fixed,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--features", type=Path, help="one .npz per document with spans and vectors")
    parser.add_argument("--encoder-id", help="model, revision and prefix extraction recipe for --features")
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--budget", type=int, default=12_864)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--long-gap", type=int, default=128)
    parser.add_argument("--coverage", type=float, default=0.35)
    args = parser.parse_args()
    if min(args.dimension, args.budget, args.candidates, args.long_gap) <= 0:
        parser.error("dimension, budget, candidates and long-gap must be positive")
    if not math.isfinite(args.coverage) or not 0 < args.coverage < 1:
        parser.error("coverage must be between zero and one")
    if bool(args.features) != bool(args.encoder_id):
        parser.error("--features and --encoder-id must be provided together")
    dev = load_split(args.dev, args.dimension, args.features)
    test = load_split(args.test, args.dimension, args.features)
    config = {name: getattr(args, name) for name in
              ("dimension", "budget", "candidates", "long_gap", "coverage")}
    policies = run_experiment(dev, test, args.budget, args.candidates, args.long_gap, args.coverage)
    print(json.dumps({
        "encoder": args.encoder_id or "blake2b-lexical-prefix-v1",
        "development_documents": [d.name for d, _ in dev],
        "test_documents": [d.name for d, _ in test],
        "config": config,
        "policies": policies,
        "paired_closed_loop": {
            baseline: compare_wrong_attachments(
                policies["box"]["closed_loop_documents"],
                policies[baseline]["closed_loop_documents"],
            )
            for baseline in POLICIES if baseline != "box"
        },
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
