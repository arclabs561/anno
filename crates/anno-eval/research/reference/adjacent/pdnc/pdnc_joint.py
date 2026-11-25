"""Evaluate coupled speaker/addressee predictions on the pinned PDNC corpus (stdlib)."""
import argparse
import ast
import csv
import io
import json
import re
import random
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REV = "e507a27b89a85c1e9710a897ad0b3f5a9ad73dc4"
BASE = f"https://raw.githubusercontent.com/Priya22/project-dialogism-novel-corpus/{REV}/"
TREE = f"https://api.github.com/repos/Priya22/project-dialogism-novel-corpus/git/trees/{REV}?recursive=1"
SPEECH = re.compile(r"\b(said|asked|cried|replied|answered|whispered|shouted|called|murmured|added|continued|exclaimed|remarked|observed|returned)\b", re.I)


def fetch(url, path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(urllib.request.urlopen(url, timeout=60).read())
    return path.read_text(encoding="utf-8-sig")


def parse(value):
    return ast.literal_eval(value)


def ranked(scores):
    return sorted(scores, key=lambda key: (-scores[key], key))


def choose(speakers, listeners, coupled):
    s, a = ranked(speakers)[:2], ranked(listeners)[:2]
    if not coupled:
        return s[0], a[0]
    # Independent unaries plus a soft penalty for addressing oneself.
    return min(((x, y) for x in s for y in a),
               key=lambda pair: (-speakers[pair[0]] - listeners[pair[1]] +
                                 4 * (pair[0] == pair[1]), pair))


def predict(names, world, mentions, in_quote, prev, prev2, reciprocal, discourage_self):
    speakers = {n: 0.0 for n in names}
    listeners = speakers.copy()
    if prev2:
        speakers[prev2[0]] += 3
    if prev:
        listeners[prev[0]] += 3
        if reciprocal:
            speakers[prev[1]] += 3
    for person, weight in mentions:
        speakers[person] = speakers.get(person, 0) + weight
    for person, weight in in_quote:
        listeners[person] = listeners.get(person, 0) + weight
    for person, turn in world.items():
        speakers[person] += 0.5 / (1 + turn)
        listeners[person] += 0.5 / (1 + turn)
    return choose(speakers, listeners, discourage_self)


def run(book, cache):
    def read(name):
        return fetch(BASE + 'data/' + book + '/' + name, cache / book / name)
    roster = list(csv.DictReader(io.StringIO(read("character_info.csv"))))
    names = sorted({row['Main Name'] for row in roster if row['Main Name']})
    aliases = defaultdict(set)
    for row in roster:
        for name in set(parse(row['Aliases'])) | {row['Main Name']}:
            if name and not name.startswith('_'):
                aliases[name.lower()].add(row['Main Name'])
    aliases = {name: next(iter(ids)) for name, ids in aliases.items() if len(ids) == 1}
    pattern = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(x) for x in sorted(aliases, key=lambda x: (-len(x), x))) + r")(?!\w)", re.I)
    raw = read("novel_text.txt")
    quotes = []
    bad = 0
    for row in csv.DictReader(io.StringIO(read("quotation_info.csv"))):
        spans = parse(row['quoteByteSpans'])
        words = parse(row['subQuotationList'])
        if any(raw[l:r].strip() != w.strip() for (l, r), w in zip(spans, words, strict=True)):
            bad += 1
        quotes.append((spans[0][0], spans[-1][1], spans, row))
    quotes.sort(key=lambda item: item[0])
    history = {key: [] for key in ['pipeline', 'exclusion_only', 'reciprocal_only', 'coupled_frozen', 'coupled_feedback']}
    counts = {key: Counter() for key in history}
    world = {}
    for i, (start, end, spans, row) in enumerate(quotes):
        prev_end = min(start, quotes[i-1][1]) if i else 0
        next_start = max(end, quotes[i+1][0]) if i+1 < len(quotes) else len(raw)
        before = raw[max(prev_end, start-160):start]
        after = raw[end:min(end+160, next_start)]
        # Allow an interrupted reporting clause, excluding the quotation text.
        gaps = [raw[spans[j][1]:spans[j+1][0]][:160] for j in range(len(spans)-1)]
        mentions, vocatives = [], []
        for fragment in [before, after, *gaps]:
            verbs = list(SPEECH.finditer(fragment))
            for m in pattern.finditer(fragment):
                who = aliases[m.group().lower()]
                close = any(max(0, m.start()-v.end(), v.start()-m.end()) <= 14 for v in verbs)
                mentions.append((who, 8 if close else 1))
        for left, right in spans:
            fragment = raw[left:right]
            for m in pattern.finditer(fragment):
                if fragment[:m.start()].rstrip().endswith(',') or fragment[m.end():].lstrip().startswith(','):
                    vocatives.append((aliases[m.group().lower()], 7))
        recency = {who: i - last for who, last in world.items()}
        predictions = {}
        for arm in history:
            prevs = history[arm] if arm == 'coupled_feedback' else history['pipeline']
            predictions[arm] = predict(names, recency, mentions, vocatives,
                                       prevs[-1] if prevs else None, prevs[-2] if len(prevs)>1 else None,
                                       arm in {'reciprocal_only', 'coupled_frozen', 'coupled_feedback'},
                                       arm in {'exclusion_only', 'coupled_frozen', 'coupled_feedback'})
        speaker = row['speaker']
        audience = set(parse(row['addressees']))
        for arm, (s, a) in predictions.items():
            history[arm].append((s, a))
            tally = counts[arm]
            if speaker and audience:
                tally.update({"n": 1, "speaker": s == speaker, "one_addressee": a in audience,
                              "both": s == speaker and a in audience,
                              "single_addressee": len(audience) == 1,
                              "both_exact": s == speaker and audience == {a},
                              "implicit": row['quoteType'] == 'Implicit',
                              "implicit_both": row['quoteType'] == 'Implicit' and s == speaker and a in audience})
        world.update({who: i for who, w in mentions + vocatives})
    return book, {"quotes": len(quotes), "offset_mismatches": bad, "arms": counts}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('cache', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    tree = json.loads(fetch(TREE, args.cache / 'tree.json'))['tree']
    books = sorted(x['path'].split('/')[1] for x in tree if x['path'].startswith('data/') and x['path'].endswith('/quotation_info.csv'))
    with ThreadPoolExecutor(5) as pool:
        docs = dict(pool.map(lambda name: run(name, args.cache), books))
    total = {arm: dict(sum((Counter(v['arms'][arm]) for v in docs.values()), Counter())) for arm in next(iter(docs.values()))['arms']}
    rng = random.Random(7)
    gains = []
    for _ in range(2000):
        sampled = [docs[book]['arms'] for book in rng.choices(books, k=len(books))]
        gained = sum(x['exclusion_only']['both'] - x['pipeline']['both'] for x in sampled)
        gains.append(gained / sum(x['pipeline']['n'] for x in sampled))
    gains.sort()
    result = {'revision': REV, 'books': docs, 'total': total,
              'exclusion_minus_pipeline_joint_accuracy_book_bootstrap_95': [gains[49], gains[1949]]}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(total, 'offset mismatch', sum(v['offset_mismatches'] for v in docs.values()))


if __name__ == "__main__":
    main()
