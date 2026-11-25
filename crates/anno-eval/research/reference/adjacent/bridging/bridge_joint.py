"""Evaluate antecedent retrieval and relation typing in the GUMBridge v1 split."""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

REVISION = "bfe271bcb554aaa924c697779f2735dc16151554"
DEPTH = 32


def read_doc(path):
    mentions, links = {}, defaultdict(dict)
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.rstrip("\t").split("\t")
        position = tuple(map(int, fields[0].split("-")))
        for entry in fields[3].split("|"):
            if entry == "_":
                continue
            label, mid = re.fullmatch(r"(.+)\[(\d+)\]", entry).groups()
            if mid not in mentions:
                mentions[mid] = [position, position, label]
            mentions[mid][1] = position
        for rel, edge in zip(fields[-2].split("|"), fields[-1].split("|"), strict=True):
            if not rel.startswith("bridge:"):
                continue
            later, earlier = re.fullmatch(r"\d+-\d+\[(\d+)_(\d+)\]", edge).groups()
            links[later].setdefault(earlier, set()).update(rel[7:].split(";"))
    examples = []
    for query, gold in links.items():
        assert query in mentions and gold.keys() <= mentions.keys()
        start, _, qt = mentions[query]
        prior = [m for m, (_, end, _) in mentions.items() if end < start]
        prior.sort(key=lambda m: (mentions[m][1], mentions[m][0], int(m)), reverse=True)
        examples.append((qt, [(m, mentions[m][2]) for m in prior[:DEPTH]], gold))
    return examples


def key(q, c, mode):
    return tuple(sorted((q, c))) if mode == "symmetric" else (q, c)


class Prior:
    def __init__(self, examples, mode):
        self.exposed, self.linked = Counter(), Counter()
        self.rank_total, self.rank_link = Counter(), Counter()
        self.labels, self.by_pair = Counter(), defaultdict(Counter)
        self.mode = mode
        for qt, candidates, gold in examples:
            for rank, (mid, typ) in enumerate(candidates):
                pair = key(qt, typ, mode)
                bucket = (rank + 1).bit_length()
                self.exposed[pair] += 1
                self.rank_total[bucket] += 1
                if mid in gold:
                    self.linked[pair] += 1
                    self.rank_link[bucket] += 1
                    for label in sorted(gold[mid]):
                        self.labels[label] += 1
                        self.by_pair[pair][label] += 1
        self.global_rate = sum(self.linked.values()) / sum(self.exposed.values())

    def score(self, qt, typ, rank):
        pair = key(qt, typ, self.mode)
        bucket = (rank + 1).bit_length()
        positional = (self.rank_link[bucket] + 1) / (self.rank_total[bucket] + 2)
        # Empirical type-pair link rate shrunk with 20 pseudocandidates.
        pairwise = (self.linked[pair] + 20*self.global_rate) / (self.exposed[pair] + 20)
        local = self.by_pair[pair]
        weights = {label: (local[label] + 10 * count / self.labels.total()) /
                   (local.total() + 10) for label, count in self.labels.items()}
        label = max(sorted(weights), key=weights.get)
        return positional * pairwise, label, weights[label]


def evaluate(examples, prior):
    groups = defaultdict(Counter)
    for qt, candidates, gold in examples:
        scored = [(mid, *prior.score(qt, typ, i)) for i, (mid, typ) in enumerate(candidates)]
        for mode in ("recency", "pipeline", "joint"):
            out = groups[mode]
            by_type = out.setdefault("by_gold_subtype", {})
            for tag in set().union(*gold.values()):
                bucket = by_type.setdefault(tag, Counter())
                bucket["queries"] += 1
            out["queries"] += 1
            out["reachable"] += bool(gold.keys() & {m for m, *_ in scored})
            if not scored:
                continue
            if mode == "recency":
                ranked = scored
            else:
                ranked = sorted(scored, key=lambda row: row[1] * (row[3] if mode == "joint" else 1), reverse=True)
            chosen, _, label, _ = ranked[0]
            for tag in gold.get(chosen, ()):
                by_type[tag]["top1"] += 1
                by_type[tag]["joint_top1"] += label == tag
            out["top1"] += chosen in gold
            out["top5"] += any(m in gold for m, *_ in ranked[:5])
            out["joint_top1"] += label in gold.get(chosen, ())
            # With the true pair revealed, measure the type classifier separately.
            for mid, _, prediction, _ in scored:
                if mid in gold:
                    out["oracle_pairs"] += 1
                    out["oracle_type"] += prediction in gold[mid]
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    args = parser.parse_args()
    folds, docs = {}, {}
    for split in ("train", "dev", "test", "test2"):
        files = sorted((args.data / split).glob("*.tsv"))
        if not files:
            raise ValueError(f"Missing {split}")
        docs[split] = len(files)
        folds[split] = [example for f in files for example in read_doc(f)]
    result = {"revision": REVISION, "candidate_limit": DEPTH, "documents": docs,
              "train_gold_anaphors": len(folds["train"]), "results": {}}
    for mode in ("ordered", "symmetric"):
        prior = Prior(folds["train"], mode)
        result["results"][mode] = {s: evaluate(folds[s], prior) for s in ("dev", "test", "test2")}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
