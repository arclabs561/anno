"""Read-only development comparison using predicted dependency heads and gold mention spans.

Usage: python abstaining_heads.py DEV.parquet MODEL_DIRECTORY results.json
Use PyArrow and ufal.udpipe; obtain the pinned models listed in abstaining-head-models.json
from jwijffels/udpipe.models.ud.2.5 at 17d6bbfb7f138f9ba2cf04692f15d892becf0246.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import heapq
from itertools import groupby
import json
from pathlib import Path
import re

import pyarrow.parquet as pq
from ufal.udpipe import Model, Sentence

from corefud_retrieval import CHECKSUM, norm


def dependencies(tokens, model):
    heads = []
    for _, group in groupby(tokens, key=lambda token: token["sentence_ID"]):
        sentence = Sentence()
        base = len(heads)
        for token in group:
            sentence.addWord(token["form"])
        if not model.tag(sentence, "") or not model.parse(sentence, ""):
            raise RuntimeError("UDPipe failed to tag or parse a supplied sentence")
        for word in list(sentence.words)[1:]:
            heads.append(None if word.head == 0 else base + word.head - 1)
    return heads


def score(doc, model):
    tokens = doc["tokens"]
    governors = dependencies(tokens, model)
    mentions = []
    coords = Counter((m["onset"], m["offset"]) for m in doc["mentions"])
    counts = Counter(documents=1, source_mentions=len(doc["mentions"]))
    for m in doc["mentions"]:
        a, b = m["onset"], m["offset"]
        if a is None or b is None or not 0 <= a < b <= len(doc["text"]) or coords[a, b] != 1:
            continue
        form = norm(doc["text"][a:b])
        if not any(c.isalnum() for c in form):
            continue
        lo, hi = m["start_token"], m["end_token"]
        head = None
        if (0 <= lo <= hi < len(tokens) and tokens[lo]["char_onset"] == a
                and tokens[hi]["char_offset"] == b
                and tokens[lo]["sentence_ID"] == tokens[hi]["sentence_ID"]):
            roots = [j for j in range(lo, hi + 1) if governors[j] is None or not lo <= governors[j] <= hi]
            if len(roots) == 1:
                head = norm(tokens[roots[0]]["form"])
        counts["with_predicted_head"] += head is not None
        segments = tuple(re.findall(r"\w+|[^\w\s]", form))
        mentions.append((a, b, form, segments, head, m["COREF"]))
    mentions.sort(key=lambda m: m[:2])
    pending, seen = [], set()
    surfaces, segmented, heads = {}, {}, {}
    for i, (start, end, form, pieces, head, target) in enumerate(mentions):
        while pending and pending[0][0] <= start:
            _, j = heapq.heappop(pending)
            _, _, old_form, old_parts, old_head, old_target = mentions[j]
            surfaces[old_form] = segmented[old_parts] = j
            if old_head:
                heads[old_head] = j
            seen.add(old_target)
        heapq.heappush(pending, (end, i))
        returning = target in seen
        counts["returning" if returning else "no_antecedent"] += 1
        base = surfaces.get(form)
        for method, picked in (("surface", base), ("segments", segmented.get(pieces)),
                               ("head", surfaces.get(form, heads.get(head)))):
            if picked is None:
                counts[method + "_abstentions"] += 1
            elif mentions[picked][-1] == target:
                counts[method + "_correct"] += 1
            else:
                counts[method + ("_wrong_returning" if returning else "_wrong_first")] += 1
    return counts


def run(cache, directory, per_corpus):
    if hashlib.sha256(cache.read_bytes()).hexdigest() != CHECKSUM:
        raise ValueError("unexpected development shard")
    spec = json.loads(Path(__file__).with_name("abstaining-head-models.json").read_text())
    by_corpus = defaultdict(list)
    for batch in pq.ParquetFile(cache).iter_batches(columns=["file_name", "language"], batch_size=128):
        for d in batch.to_pylist():
            by_corpus[d["file_name"].split("-corefud-dev")[0]].append(d["file_name"])
    selected = {name for names in by_corpus.values() for name in
                sorted(names, key=lambda n: hashlib.sha256(n.encode()).digest())[:per_corpus]}
    docs = defaultdict(list)
    for batch in pq.ParquetFile(cache).iter_batches(batch_size=8):
        for d in batch.to_pylist():
            if d["file_name"] in selected:
                docs[d["language"]].append(d)
    result = {}
    for lang, group in sorted(docs.items()):
        row = {"documents": sorted(d["file_name"] for d in group)}
        result[lang] = row
        if lang not in spec:
            row["status"] = "no_matching_parser_model"
            continue
        rec = spec[lang]
        path = directory / Path(rec["path"]).name
        raw = path.read_bytes()
        if hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() != rec["git_blob"]:
            raise ValueError(f"unexpected model bytes for {lang}")
        model = Model.load(str(path))
        if model is None:
            raise ValueError(f"cannot load model for {lang}")
        counts = Counter()
        for d in group:
            counts.update(score(d, model))
        row.update(status="scored", counts=dict(counts))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("models", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--per-corpus", type=int, default=3)
    args = parser.parse_args()
    if args.per_corpus < 1:
        parser.error("--per-corpus must be positive")
    args.output.write_text(json.dumps(run(args.cache, args.models, args.per_corpus), indent=2)+"\n")
