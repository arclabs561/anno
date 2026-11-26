"""Score matched whole-sentence prefixes of the ten LitBank fold-0 test works."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from coref_gap import check_file, decode_clusters, index_partition

REVISION = "58b5fdd88d139b17ffd75365a5e1ec5b53b8dd43"
SHA256 = "eeda5569e3d86bb46e0bf867603b0ea0061c02792c07ca185f8dc3431b4bbb43"
TOKEN_BUDGET = 180


def load(path):
    import pyarrow.parquet as pq

    check_file(path, SHA256)
    docs = []
    for doc in pq.read_table(path).to_pylist():
        # The mirror's token offsets target a differently detokenized edition.
        # Match every token in sequence against the original. Only skip whitespace.
        text, cursor = doc["text"], 0
        spans = []
        count = 0
        for sentence in doc["sentences"]:
            tokens = sentence["tokens"]
            if count + len(tokens) > TOKEN_BUDGET:
                break
            count += len(tokens)
            sentence_spans = []
            for token in tokens:
                while cursor < len(text) and text[cursor].isspace():
                    cursor += 1
                surface = token["text"]
                if not text.startswith(surface, cursor):
                    raise ValueError(f"token mismatch in {doc['id']} at {cursor}")
                sentence_spans.append((cursor, cursor + len(surface)))
                cursor += len(surface)
            spans.append(sentence_spans)
        clusters = []
        for chain in doc["coref_chains"]:
            cluster = {
                (spans[s][first][0], spans[s][last][1])
                for s, first, last in chain
                if s < len(spans)
            }
            if cluster:
                clusters.append(cluster)
        if not clusters:
            raise ValueError("empty annotated excerpt")
        docs.append((doc["id"], text[:cursor], count, len(spans), clusters))
    if len(docs) != 10 or len({d[0] for d in docs}) != 10:
        raise ValueError("expected all ten test documents")
    return docs


def directional(key, response):
    """MUC and B³ numerators/denominators in one direction (missing spans split MUC)."""
    index = index_partition(response)
    muc, b3 = [0, 0], [0.0, 0]
    for group in key:
        overlaps = {}
        missing = 0
        for span in group:
            if span in index:
                i = index[span]
                overlaps[i] = overlaps.get(i, 0) + 1
            else:
                missing += 1
        muc[0] += len(group) - len(overlaps) - missing
        muc[1] += len(group) - 1
        b3[0] += sum(n * n for n in overlaps.values()) / len(group)
        b3[1] += len(group)
    return muc, b3


def score_doc(gold, predicted):
    spans, hits = set(index_partition(gold)), set(index_partition(predicted))
    # Report all emitted spans and singleton-free clustering separately.
    mention = [len(spans & hits), len(hits), len(spans)]
    gold = [c for c in gold if len(c) > 1]
    predicted = [c for c in predicted if len(c) > 1]
    gp, pp = set(index_partition(gold)), set(index_partition(predicted))
    mention_linked = [len(gp & pp), len(pp), len(gp)]
    pm, pb = directional(predicted, gold)
    rm, rb = directional(gold, predicted)
    similarities = [
        [2 * len(g & p) / (len(g) + len(p)) for p in predicted] for g in gold
    ]
    similarity = 0.0
    if min(len(gold), len(predicted)) == 1:
        similarity = max(max(scores) for scores in similarities)
    elif gold and predicted:
        from scipy.optimize import linear_sum_assignment

        row, col = linear_sum_assignment(similarities, maximize=True)
        similarity = sum(similarities[i][j] for i, j in zip(row, col))
    return {
        "mentions": mention,
        "linked_mentions": mention_linked,
        "muc": pm + rm,
        "b3": pb + rb,
        "ceafe": [similarity, len(predicted), similarity, len(gold)],
    }


def summarize(totals):
    scores = {}
    for metric, values in totals.items():
        if len(values) == 3:
            hits, pden, rden = values
            pnum = rnum = hits
        else:
            pnum, pden, rnum, rden = values
        p = pnum / pden if pden else 0.0
        r = rnum / rden if rden else 0.0
        scores[metric] = [
            round(x, 6) for x in (p, r, 2 * p * r / (p + r) if p + r else 0.0)
        ]
    scores["conll_f1"] = round(sum(scores[m][2] for m in ["muc", "b3", "ceafe"]) / 3, 6)
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", type=Path)
    parser.add_argument("binary", type=Path)
    parser.add_argument("backend", choices=["mention-ranking", "fcoref", "maverick"])
    parser.add_argument("mode", choices=["predicted", "supplied"])
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    from coref_gap import model_identity

    model = model_identity(args.backend, args.mode)
    docs = load(args.parquet)
    inputs = [
        {
            "id": k,
            "text": text,
            "mentions": sorted(index_partition(gold))
            if args.mode == "supplied"
            else None,
        }
        for k, text, tokens, sentences, gold in docs
    ]
    env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "ANNO_NO_DOWNLOADS": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    raw = subprocess.run(
        [args.binary.resolve(), args.backend],
        check=True,
        env=env,
        input="".join(json.dumps(d) + "\n" for d in inputs),
        text=True,
        stdout=subprocess.PIPE,
    ).stdout
    totals, per_doc = {}, []
    for (k, text, tokens, nsent, gold), record in zip(
        docs, raw.splitlines(), strict=True
    ):
        r = json.loads(record)
        supplied = index_partition(gold) if args.mode == "supplied" else None
        predicted = decode_clusters(r, k, text, args.backend, supplied)
        if predicted is None:
            per_doc.append(
                dict(
                    id=k,
                    text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    source_chars=len(text),
                    tokens=tokens,
                    sentences=nsent,
                    unsupported=r["unsupported"],
                )
            )
            continue
        stats = score_doc(gold, predicted)
        for metric, values in stats.items():
            before = totals.get(metric, [0] * len(values))
            totals[metric] = [a + b for a, b in zip(before, values, strict=True)]
        per_doc.append(
            dict(
                id=k,
                text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                source_chars=len(text),
                tokens=tokens,
                sentences=nsent,
                counts=stats,
            )
        )
    result = dict(
        revision=REVISION,
        sha256=SHA256,
        split="split_0/test",
        model=model,
        mode=args.mode,
        counts=totals,
        scores=summarize(totals),
        documents=per_doc,
    )
    args.output.write_text(json.dumps(result, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
