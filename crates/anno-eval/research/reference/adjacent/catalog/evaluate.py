"""Evaluate cross-document event retrieval and reuse of verified catalog links."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
from urllib.request import urlopen

ECB = "https://huggingface.co/datasets/ahmeshaf/ecb_plus_mentions/resolve/a549d5c7c456400b0262891344300e2ca66ec58f/test.json"
ZELDA = "https://huggingface.co/datasets/naist-nlp/zelda/resolve/374abf1c2eb6d22264671ef6c357961ca92201d3/data/dev-00001-of-00001.jsonl"


def jaccard(a, b):
    return len(a & b) / len(a | b) if a or b else 0.0


def events(rows):
    # Gold locations/types expose the same entity text to both context arms.
    entities = defaultdict(set)
    for r in rows:
        if r["men_type"] == "ent":
            entities[r["doc_id"], r["sentence_id"]].add(r["mention_text"].casefold())
    mentions = []
    for r in rows:
        if r["men_type"] == "evt":
            mentions.append((r["doc_id"], r["lemma"].casefold(), set(r["sentence_tokens"]),
                             entities[r["doc_id"], r["sentence_id"]], r["gold_cluster"],
                             sha256(r["mention_id"].encode()).digest()))
    mentions.sort(key=lambda m: m[-1])
    total = Counter()
    for doc, lemma, words, arguments, gold, _ in mentions:
        total["queries"] += 1
        candidates = [m for m in mentions if m[0] != doc]
        if not any(m[4] == gold for m in candidates):
            continue
        total["reachable"] += 1
        scores = [(m[1] == lemma, jaccard(words, m[2]), jaccard(arguments, m[3]))
                  for m in candidates]
        for arm, rank in (("lemma", lambda s: s[0]), ("context", lambda s: (s[0], s[1])),
                          ("entity_context", lambda s: (s[0], s[1] + s[2]))):
            best = max(range(len(scores)), key=lambda i: rank(scores[i]))
            total[arm] += candidates[best][4] == gold
    return dict(total)


def catalog(rows):
    counts = Counter()
    for r in rows:
        counts["documents"] += 1
        seen, ids = [], set()
        for ent in sorted(r["entities"], key=lambda e: (e["start"], e["end"])):
            label = set(ent["label"])
            if not label:
                continue
            words = tuple(re.findall(r"\w+", r["text"][ent["start"]:ent["end"]].casefold()))
            counts["mentions"] += 1
            counts["previous_target"] += bool(ids & label)
            # All entries in this cache are previously verified mappings.
            for arm in ("exact", "token_overlap"):
                def score(entry):
                    earlier = entry[0]
                    if words and words == earlier:
                        return 2.0
                    return jaccard(set(words), set(earlier)) if arm == "token_overlap" else 0.0
                if seen:
                    best = max(reversed(seen), key=score)
                    if score(best) > 0:
                        counts[arm + "_offered"] += 1
                        counts[arm + "_correct"] += bool(best[1] & label)
            seen.append((words, label))
            ids.update(label)
    return dict(counts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ecb", type=Path)
    parser.add_argument("--zelda", type=Path)
    args = parser.parse_args()
    with (args.ecb.open("rb") if args.ecb else urlopen(ECB)) as source:
        scores = events(json.load(source))
    with (args.zelda.open("rb") if args.zelda else urlopen(ZELDA)) as source:
        links = catalog(json.loads(line) for _, line in zip(range(500), source))
    print(json.dumps({"ecb_test": scores, "zelda_dev_prefix": links}, indent=2))
