#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "python-crfsuite>=0.9.10",
#     "datasets>=2.14.0",
# ]
# ///
"""
Train compact CRF NER weights on a redistributable dataset.

This script trains a CRF model using python-crfsuite and exports feature weights for use in the
Rust CRF backend.

Why “compact”:
- We intentionally avoid token-identity features (like `word.lower=google`) so the weights file
  stays small enough to ship in the crate.
- We keep shape/affix/casing/context and transition features, which generalize better.

Default dataset:
- `unimelb-nlp/wikiann` (WikiANN / PAN-X). The Hugging Face dataset card/discussion indicate
  Apache-2.0 for the packaged dataset. See:
  - https://huggingface.co/datasets/unimelb-nlp/wikiann
  - https://huggingface.co/datasets/unimelb-nlp/wikiann/discussions/6

Usage:
    uv run scripts/train_crf_weights.py

Output (by default):
    - crates/anno/src/backends/crf_weights.json: Feature weights for Rust CRF backend (shipped)
    - .generated/crf_model.crfsuite: Native CRFsuite model file (local artifact)
"""

import json
import os
import argparse
import pycrfsuite
from datasets import load_dataset
from collections import defaultdict
from collections import Counter


def word_shape(word: str) -> str:
    """Compute word shape (e.g., 'John' -> 'Xx', 'USA' -> 'X')."""
    shape = []
    prev = None
    for c in word:
        if c.isupper():
            ch = 'X'
        elif c.islower():
            ch = 'x'
        elif c.isdigit():
            ch = '0'
        else:
            ch = c
        if ch != prev:
            shape.append(ch)
            prev = ch
    return ''.join(shape)


def word2features(sent, i):
    """Extract features for a word at position i."""
    word = sent[i][0]
    
    features = [
        'bias',
        f'word.lower={word.lower()}',
        f'word.shape={word_shape(word)}',
        f'word.isdigit={word.isdigit()}',
        f'word.istitle={word.istitle()}',
        f'word.isupper={word.isupper()}',
    ]
    
    # Prefixes and suffixes
    if len(word) >= 2:
        features.append(f'prefix2={word[:2].lower()}')
        features.append(f'suffix2={word[-2:].lower()}')
    if len(word) >= 3:
        features.append(f'prefix3={word[:3].lower()}')
        features.append(f'suffix3={word[-3:].lower()}')
    
    # Context features
    if i > 0:
        word1 = sent[i-1][0]
        features.extend([
            f'-1:word.lower={word1.lower()}',
            f'-1:word.istitle={word1.istitle()}',
            f'-1:word.isupper={word1.isupper()}',
            f'-1:word.shape={word_shape(word1)}',
        ])
    else:
        features.append('BOS')
        
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        features.extend([
            f'+1:word.lower={word1.lower()}',
            f'+1:word.istitle={word1.istitle()}',
            f'+1:word.isupper={word1.isupper()}',
            f'+1:word.shape={word_shape(word1)}',
        ])
    else:
        features.append('EOS')
    
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, _, label in sent]


def hf_ner_to_sentences(dataset, tokens_field="tokens", tags_field="ner_tags"):
    """Convert a Hugging Face token-classification dataset split into CRFsuite sentences."""
    # Prefer the dataset-provided label names.
    tag_map = None
    try:
        feat = dataset.features[tags_field]
        # Sequence(ClassLabel(names=[...])) is typical.
        if hasattr(feat, "feature") and hasattr(feat.feature, "names"):
            names = list(feat.feature.names)
            tag_map = {i: n for i, n in enumerate(names)}
    except Exception:
        tag_map = None

    # Fallback: CoNLL-2003 tag ids (common).
    if tag_map is None:
        tag_map = {
            0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG',
            5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC', 8: 'I-MISC'
        }

    sentences = []
    for example in dataset:
        tokens = example[tokens_field]
        ner_tags = example[tags_field]
        
        sent = []
        for tok, ner in zip(tokens, ner_tags):
            label = tag_map.get(ner, 'O')
            # We keep a 3-tuple to preserve the existing sent2labels shape.
            sent.append((tok, 0, label))
        sentences.append(sent)
    
    return sentences


def train_crf(dataset_id, config, out_weights_path, out_model_path, max_train_sents, max_test_sents):
    """Train CRF model and export weights."""
    print(f"Loading dataset: {dataset_id} config={config!r} ...")
    ds = load_dataset(dataset_id, config)
    
    print("Converting to sentences...")
    train_sents = hf_ner_to_sentences(ds['train'])
    test_sents = hf_ner_to_sentences(ds['test'])

    if max_train_sents and max_train_sents > 0:
        train_sents = train_sents[:max_train_sents]
    if max_test_sents and max_test_sents > 0:
        test_sents = test_sents[:max_test_sents]
    
    print(f"Train sentences: {len(train_sents)}")
    print(f"Test sentences: {len(test_sents)}")

    # Build a bounded “shippable vocab” for word.lower features.
    # Token-identity features can explode weight count; we keep only frequent words.
    wc = Counter()
    for s in train_sents:
        for tok, _pos, _lab in s:
            wc[tok.lower()] += 1
    vocab_top_k = 2000
    vocab_min_count = 20
    keep_words = {w for (w, c) in wc.most_common(vocab_top_k) if c >= vocab_min_count}
    print(f"Shippable vocab: {len(keep_words)} words (top_k={vocab_top_k}, min_count={vocab_min_count})")
    
    # Extract features
    print("Extracting features...")
    X_train = [sent2features(s) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]
    X_test = [sent2features(s) for s in test_sents]
    y_test = [sent2labels(s) for s in test_sents]
    
    # Train CRF
    print("Training CRF model...")
    trainer = pycrfsuite.Trainer(verbose=False)
    
    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)
    
    trainer.set_params({
        'c1': 0.1,  # L1 regularization
        'c2': 0.1,  # L2 regularization
        'max_iterations': 100,
        'feature.possible_transitions': True,
    })
    
    os.makedirs(os.path.dirname(out_model_path), exist_ok=True)
    trainer.train(out_model_path)
    print(f"Model saved to {out_model_path}")
    
    # Extract weights
    print("Extracting feature weights...")
    tagger = pycrfsuite.Tagger()
    tagger.open(out_model_path)
    
    # Get state features (emission features).
    #
    # Compact export:
    # - Keep general features (shape/affix/case/context).
    # - Keep token identity only for a bounded frequent vocab (to keep size shippable).
    weights = {}
    info = tagger.info()
    
    for (attr, label), weight in info.state_features.items():
        if attr.startswith("word.lower="):
            w = attr[len("word.lower="):]
            if w not in keep_words:
                continue
        if attr.startswith("-1:word.lower="):
            # Keep context identity smaller: only keep if token is frequent.
            w = attr[len("-1:word.lower="):]
            if w not in keep_words:
                continue
        if attr.startswith("+1:word.lower="):
            w = attr[len("+1:word.lower="):]
            if w not in keep_words:
                continue
        key = f"{attr}:{label}"
        weights[key] = weight
    
    # Get transition features
    for (label_from, label_to), weight in info.transitions.items():
        key = f"trans:{label_from}->{label_to}"
        weights[key] = weight
    
    # Save weights as JSON (ship this file in the Rust crate).
    os.makedirs(os.path.dirname(out_weights_path), exist_ok=True)
    with open(out_weights_path, 'w') as f:
        json.dump(weights, f, indent=2)
    print(f"Saved {len(weights)} feature weights to {out_weights_path}")
    
    # Evaluate
    print("\nEvaluating on test set...")
    y_pred = [tagger.tag(xseq) for xseq in X_test]
    
    # Calculate F1
    correct = 0
    predicted = 0
    actual = 0
    
    for y_true_seq, y_pred_seq in zip(y_test, y_pred):
        for y_true, y_pred_tag in zip(y_true_seq, y_pred_seq):
            if y_true != 'O':
                actual += 1
            if y_pred_tag != 'O':
                predicted += 1
            if y_true != 'O' and y_true == y_pred_tag:
                correct += 1
    
    precision = correct / predicted if predicted > 0 else 0
    recall = correct / actual if actual > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    
    return weights


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="unimelb-nlp/wikiann")
    ap.add_argument("--config", default="en", help="HF config/subset (language code for WikiANN)")
    ap.add_argument("--max-train-sents", type=int, default=15000, help="0 means no limit")
    ap.add_argument("--max-test-sents", type=int, default=3000, help="0 means no limit")
    ap.add_argument(
        "--out-weights",
        default="crates/anno/src/backends/crf_weights.json",
    )
    ap.add_argument(
        "--out-model",
        default=".generated/crf_model.crfsuite",
    )
    args = ap.parse_args()
    train_crf(
        dataset_id=args.dataset,
        config=args.config,
        out_weights_path=args.out_weights,
        out_model_path=args.out_model,
        max_train_sents=args.max_train_sents,
        max_test_sents=args.max_test_sents,
    )

