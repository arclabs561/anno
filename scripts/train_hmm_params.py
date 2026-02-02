#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets>=2.14.0",
# ]
# ///
"""
Train small HMM parameters (priors + transitions) on a redistributable NER dataset.

What we ship:
- `crates/anno/src/backends/hmm_params.json` containing:
  - label list (BIO tags)
  - initial probabilities
  - transition probabilities

What we *do not* ship:
- word emission tables (they explode in size and are dataset-specific).

Default dataset:
- WikiANN (PAN-X) via `unimelb-nlp/wikiann`, config `en`.

Usage:
  uv run scripts/train_hmm_params.py
"""

import argparse
import json
import math
import os
from datasets import load_dataset


DEFAULT_LABELS = [
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
    "B-MISC",
    "I-MISC",
]


def extract_label_names(split, tags_field="ner_tags"):
    feat = split.features.get(tags_field)
    if feat is not None and hasattr(feat, "feature") and hasattr(feat.feature, "names"):
        return list(feat.feature.names)
    return None


def normalize(v):
    s = sum(v)
    if s <= 0:
        return v
    return [x / s for x in v]


def train_initial_and_transitions(tag_seqs, labels, smoothing=1e-6):
    idx = {l: i for i, l in enumerate(labels)}
    n = len(labels)
    init = [smoothing] * n
    trans = [[smoothing] * n for _ in range(n)]

    for seq in tag_seqs:
        if not seq:
            continue
        init[idx.get(seq[0], 0)] += 1.0
        for a, b in zip(seq, seq[1:]):
            ia = idx.get(a, 0)
            ib = idx.get(b, 0)
            trans[ia][ib] += 1.0

    init = normalize(init)
    trans = [normalize(row) for row in trans]
    return init, trans


def seq_logprob(tags, init, trans, idx):
    if not tags:
        return 0.0
    p = 0.0
    a0 = idx.get(tags[0], 0)
    p += math.log(init[a0] + 1e-300)
    for a, b in zip(tags, tags[1:]):
        ia = idx.get(a, 0)
        ib = idx.get(b, 0)
        p += math.log(trans[ia][ib] + 1e-300)
    return p


def eval_markov_chain(tag_seqs, labels, init, trans):
    idx = {l: i for i, l in enumerate(labels)}
    total_tokens = 0
    total_ll = 0.0
    for s in tag_seqs:
        total_tokens += max(len(s), 1)
        total_ll += seq_logprob(s, init, trans, idx)
    nll_per_tok = -total_ll / max(total_tokens, 1)
    ppl = math.exp(nll_per_tok)
    return {"tokens": total_tokens, "nll_per_token": nll_per_tok, "perplexity": ppl}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="unimelb-nlp/wikiann")
    ap.add_argument("--config", default="en")
    ap.add_argument("--tags-field", default="ner_tags")
    ap.add_argument("--max-train-sents", type=int, default=20000)
    ap.add_argument("--smoothing", type=float, default=1e-6)
    ap.add_argument("--out", default="crates/anno/src/backends/hmm_params.json")
    args = ap.parse_args()

    ds = load_dataset(args.dataset, args.config)
    train_split = ds["train"]
    label_names = extract_label_names(train_split, args.tags_field)

    # Map dataset labels to our internal label set.
    # If dataset label set is exactly our set, keep as-is. Otherwise, map unknown labels to "O".
    labels = DEFAULT_LABELS
    if label_names is not None and set(label_names) >= set(labels):
        # Keep our ordering; map by name.
        id_to_name = {i: n for i, n in enumerate(label_names)}
        def map_seq(ids):
            return [id_to_name.get(i, "O") if id_to_name.get(i, "O") in labels else "O" for i in ids]
    else:
        # Best-effort fallback: assume CoNLL-like ids where 0..8 map to DEFAULT_LABELS.
        def map_seq(ids):
            out = []
            for i in ids:
                if 0 <= i < len(DEFAULT_LABELS):
                    out.append(DEFAULT_LABELS[i])
                else:
                    out.append("O")
            return out

    max_n = args.max_train_sents
    tag_seqs = []
    for ex in train_split:
        tag_seqs.append(map_seq(ex[args.tags_field]))
        if max_n and len(tag_seqs) >= max_n:
            break

    init, trans = train_initial_and_transitions(tag_seqs, labels, smoothing=args.smoothing)

    # Lightweight “is this sane?” evaluation: Markov-chain perplexity on heldout label sequences.
    # This does NOT evaluate extraction quality (we are not modeling emissions), but it does
    # validate that the shipped params are non-degenerate and data-derived.
    test_split = ds.get("test") or ds.get("validation")
    eval_stats = None
    if test_split is not None:
        test_seqs = []
        for ex in test_split:
            test_seqs.append(map_seq(ex[args.tags_field]))
            if len(test_seqs) >= 3000:
                break
        eval_stats = eval_markov_chain(test_seqs, labels, init, trans)

    payload = {
        "schema_version": 1,
        "dataset": args.dataset,
        "config": args.config,
        "max_train_sents": len(tag_seqs),
        "smoothing": args.smoothing,
        "labels": labels,
        "initial": init,
        "transitions": trans,
        "eval": eval_stats,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {args.out}")
    print(f"labels={len(labels)} train_sents={len(tag_seqs)}")
    print("initial[O]=", init[0])
    if eval_stats is not None:
        print("eval (markov chain):", eval_stats)


if __name__ == "__main__":
    main()

