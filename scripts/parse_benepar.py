#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#     "benepar==0.2.0",
#     "spacy>=3.8,<3.9",
#     "transformers>=4.40,<4.50",
#     "torch>=2.6,<3",
# ]
# ///
"""Run benepar locally and emit constituency JSON for `anno parse`.

Model installation is explicit: pass --download once. Token offsets always
refer to the original source, including whitespace and punctuation. Python
owns inference; anno validates and consumes the resulting annotation.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from nltk.tree import Tree
    from spacy.tokens import Token


class TokenRecord(TypedDict):
    text: str
    start: int
    end: int
    tag: str | None


class ConstituentRecord(TypedDict):
    start: int
    end: int
    labels: list[str]


class SentenceRecord(TypedDict):
    tokens: list[TokenRecord]
    constituents: list[ConstituentRecord]


class ParseRecord(TypedDict):
    text: str
    sentences: list[SentenceRecord]


def encode_sentence(tree: Tree, tokens: list[Token]) -> SentenceRecord:
    """Flatten a tree, preserving unary chains and original token locations."""
    from nltk.tree import Tree

    records: list[TokenRecord] = [
        {"text": t.text, "start": t.idx, "end": t.idx + len(t.text), "tag": None}
        for t in tokens
    ]
    constituents: list[ConstituentRecord] = []
    cursor = 0

    def visit(node: Tree) -> None:
        nonlocal cursor
        start = cursor
        labels = []
        # A preterminal is the POS node immediately above a word. Keep its
        # tag separate from the phrasal labels covering the same token.
        while len(node) == 1 and isinstance(node[0], Tree):
            labels.append(node.label())
            node = node[0]
        record: ConstituentRecord = {"start": start, "end": start, "labels": labels}
        constituents.append(record)
        if len(node) == 1 and isinstance(node[0], str):
            if cursor >= len(records):
                raise ValueError("Parser returned more leaves than source tokens")
            records[cursor]["tag"] = node.label()
            cursor += 1
        else:
            labels.append(node.label())
            for child in node:
                if not isinstance(child, Tree):
                    raise TypeError("Expected POS-tagged parser leaves")
                visit(child)
        record["end"] = cursor

    visit(tree)
    if cursor != len(records):
        raise ValueError("Parser leaf count differs from source token count")
    return {"tokens": records, "constituents": constituents}


def parse_text(text: str, model: str, language: str, download: bool) -> ParseRecord:
    """Tokenize and parse; skip whitespace tokens without changing source text."""
    if download and os.environ.get("ANNO_NO_DOWNLOADS") == "1":
        raise ValueError("--download conflicts with ANNO_NO_DOWNLOADS=1")
    if not text.strip():
        return {"text": text, "sentences": []}

    import benepar
    import spacy

    if download and not benepar.download(model):
        raise RuntimeError(f"Could not download {model}")
    nlp = spacy.blank(language)
    nlp.add_pipe("sentencizer")
    doc = nlp(text)
    sentences = [[token for token in sent if not token.is_space] for sent in doc.sents]
    sentences = [tokens for tokens in sentences if tokens]
    inputs = [
        benepar.InputSentence(
            words=[token.text for token in tokens],
            space_after=[
                token.idx + len(token.text) < tokens[i + 1].idx
                if i + 1 < len(tokens)
                else bool(token.whitespace_)
                for i, token in enumerate(tokens)
            ],
        )
        for tokens in sentences
    ]
    parser = benepar.Parser(model)
    trees = list(parser.parse_sents(inputs))
    if len(trees) != len(sentences):
        raise ValueError("Parser returned a different sentence count")
    return {
        "text": text,
        "sentences": [
            encode_sentence(tree, tokens) for tree, tokens in zip(trees, sentences)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", help="Text to parse; otherwise read UTF-8 stdin")
    parser.add_argument(
        "--model", default="benepar_en3", help="Benepar model name or local path"
    )
    parser.add_argument(
        "--language", default="en", help="spaCy tokenizer language matching the model"
    )
    parser.add_argument(
        "--download", action="store_true", help="Download the named model explicitly"
    )
    args = parser.parse_args()
    text = args.text if args.text is not None else sys.stdin.read()
    # Model config/tokenizers must be cached or local unless explicitly opted in.
    if not args.download or os.environ.get("ANNO_NO_DOWNLOADS") == "1":
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    with contextlib.redirect_stdout(sys.stderr):
        result = parse_text(text, args.model, args.language, args.download)
    json.dump(result, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
