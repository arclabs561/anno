"""Produce causal mention vectors from a pinned MiniLM ONNX encoder."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from entity_memory_replay import Document, Mention, load_conll


def encode_document(document, tokenizer, session, left_context):
    vectors = []
    inputs = {entry.name for entry in session.get_inputs()}
    for mention in document.mentions:
        start = max(0, mention.start - left_context)
        visible = document.tokens[start:mention.end]
        tokens = tokenizer.encode(visible, is_pretokenized=True)
        while len(tokens.ids) > 256 and start < mention.start:
            start += 1
            visible = document.tokens[start:mention.end]
            tokens = tokenizer.encode(visible, is_pretokenized=True)
        if len(tokens.ids) > 256:
            raise ValueError(f"{document.name}: mention exceeds the encoder token limit")
        positions = [
            i for i, word in enumerate(tokens.word_ids)
            if word is not None and word >= mention.start - start
        ]
        if not positions:
            raise ValueError(f"{document.name}: tokenizer dropped a mention")
        # Dynamic quantization can couple the output to other rows in a batch.
        # Score each mention alone so later mentions cannot change this vector.
        feed = {
            "input_ids": [tokens.ids],
            "attention_mask": [tokens.attention_mask],
            "token_type_ids": [tokens.type_ids],
        }
        (hidden,) = session.run(
            ["last_hidden_state"],
            {key: np.asarray(feed[key], dtype=np.int64) for key in inputs},
        )
        vector = hidden[0, positions].mean(axis=0)
        norm = np.linalg.norm(vector)
        vectors.append(vector / norm if norm else vector)
    return np.stack(vectors).astype(np.float32)


def verify_prefix_invariance(tokenizer, session, left_context):
    early = Mention(0, 1, "a")
    first = Document(
        "prefix-a", ["Ann", "saw", "her", "sister", "leave"], [early, Mention(2, 3, "b")]
    )
    second = Document(
        "prefix-b", ["Ann", "saw", "her", "astonishing", "parliamentary", "consequences"],
        [early, Mention(2, 6, "c")],
    )
    a = encode_document(first, tokenizer, session, left_context)[0]
    b = encode_document(second, tokenizer, session, left_context)[0]
    if not np.array_equal(a, b):
        raise ValueError("changing future tokens changed the early mention vector")


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--left-context", type=int, default=48)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1 or args.left_context < 0:
        parser.error("threads must be positive and left context nonnegative")
    paths = sorted(args.corpus.glob("*.conll"))
    if not paths:
        parser.error("corpus directory contains no CoNLL documents")
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    session = ort.InferenceSession(str(args.model), options, providers=["CPUExecutionProvider"])
    verify_prefix_invariance(tokenizer, session, args.left_context)
    for path in paths:
        document = load_conll(path)
        vectors = encode_document(document, tokenizer, session, args.left_context)
        spans = [(mention.start, mention.end) for mention in document.mentions]
        np.savez(args.output / f"{path.stem}.npz", spans=spans, vectors=vectors)
    print(json.dumps({
        "documents": len(paths), "model_sha256": file_digest(args.model),
        "tokenizer_sha256": file_digest(args.tokenizer), "left_context": args.left_context,
        "batch_size": 1, "onnxruntime": ort.__version__, "prefix_fixture": "bit-identical",
    }, indent=2))


if __name__ == "__main__":
    main()
