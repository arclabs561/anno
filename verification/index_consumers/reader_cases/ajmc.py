# SPDX-License-Identifier: MIT OR Apache-2.0
"""Score the frozen native PER-track resolver on local AjMC official test pages."""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import re

from ..congress.cross_doc_partition import run_native


DIGESTS = {
    "de": "0b0be2705327ae71d43816487bd4903ba912f44add22845ca34198333fec189f",
    "en": "33d8165ab7f35c92e1ec7a0b2c21e6a45cafa03e8fb83b3e00368e2c09db6f7d",
    "fr": "a8f2a1b4a021370baa49839501d79d5f9135dd849aa6e159023941ff59d56759",
}


@dataclass(frozen=True)
class Mention:
    name: str
    label: str
    secondary: bool


def mention(tokens):
    names = []
    labels = set()
    secondary = False
    for token, label, flags in tokens:
        for flag in flags:
            if flag.startswith("Partial-"):
                start, end = map(int, flag.removeprefix("Partial-").split(":"))
                if not 0 <= start < end <= len(token):
                    raise ValueError("invalid partial token span")
                token = token[start:end]
        names.append(token)
        if "NoSpaceAfter" not in flags:
            names.append(" ")
        labels.add(label)
        secondary |= "InSecondaryReference" in flags
    if len(labels) != 1:
        raise ValueError("a person span has conflicting link labels")
    label = labels.pop()
    if label not in {"NIL", "_"} and not re.fullmatch(r"Q[0-9]+", label):
        raise ValueError("unsupported link label")
    return Mention("".join(names).rstrip(), label, secondary)


def load(directory):
    docs = {}
    for lang, digest in DIGESTS.items():
        raw = (directory / f"HIPE-2022-v2.1-ajmc-test-{lang}.tsv").read_bytes()
        if sha256(raw).hexdigest() != digest:
            raise ValueError(f"{lang}: expected the pinned official test TSV")
        header, *lines = raw.decode().splitlines()
        rows = None
        tokens = []
        for line in lines:
            if line.startswith("# hipe2022:document_id = "):
                if tokens:
                    rows.append(mention(tokens))
                    tokens = []
                doc = lang + "/" + line.split(" = ", 1)[1]
                if doc in docs:
                    raise ValueError("duplicate document ID")
                rows = docs[doc] = []
            elif line and not line.startswith("#"):
                fields = dict(zip(header.split("\t"), line.split("\t"), strict=True))
                tag = fields["NE-COARSE-LIT"]
                if tag != "I-pers" and tokens:
                    rows.append(mention(tokens))
                    tokens = []
                if tag in {"B-pers", "I-pers"}:
                    if rows is None or (tag == "I-pers" and not tokens):
                        raise ValueError("person mention lacks a page or start token")
                    tokens.append(
                        (fields["TOKEN"], fields["NEL-LIT"], fields["MISC"].split("|"))
                    )
        if tokens:
            rows.append(mention(tokens))
    return docs


def score(docs, groups):
    rows = {
        (doc, i): row
        for doc, mentions in docs.items()
        for i, row in enumerate(mentions)
    }
    pair_counts = {"cross_page": Counter(), "cross_volume": Counter()}
    answers = Counter()
    labels = Counter(row.label for row in rows.values())
    id_pages = defaultdict(set)
    for (doc, _), row in rows.items():
        if row.label.startswith("Q"):
            id_pages[row.label].add(doc)
    for a, b in combinations(rows, 2):
        if a[0] == b[0]:
            continue
        left, right = rows[a], rows[b]
        if not (left.label.startswith("Q") and right.label.startswith("Q")):
            continue
        same = left.label == right.label
        predicted = groups[a] == groups[b]
        outcome = {
            (True, True): "tp",
            (False, True): "fp",
            (True, False): "fn",
            (False, False): "tn",
        }[same, predicted]
        for scope in ["cross_page", "cross_volume"]:
            if scope == "cross_volume" and a[0].split("/")[0] == b[0].split("/")[0]:
                continue
            pair_counts[scope][outcome] += 1
            pair_counts[scope]["pairs"] += 1
            pair_counts[scope]["same_id_exact_spelling"] += (
                same and left.name == right.name
            )
            pair_counts[scope]["tp_different_spelling"] += (
                same and predicted and left.name != right.name
            )
    for anchor, row in rows.items():
        if not row.label.startswith("Q"):
            continue
        other = {b for b in rows if b[0] != anchor[0]}
        expected = {b for b in other if rows[b].label == row.label}
        hits = {b for b in other if groups[anchor] == groups[b]}
        unknown_hits = {b for b in hits if not rows[b].label.startswith("Q")}
        answers["known_queries"] += 1
        answers["shown_any"] += bool(hits)
        answers["shown_with_unscorable_hit"] += bool(unknown_hits)
        answers["shown_with_wrong_qid"] += bool(hits - expected - unknown_hits)
        if expected:
            answers["with_other_page_gold"] += 1
            answers["with_other_page_exact_spelling"] += any(
                rows[b].name == row.name for b in expected
            )
            answers["retrieved_any_correct"] += bool(hits & expected)
            answers["retrieved_exact_supported_set"] += hits == expected
            answers["retrieved_all_gold_with_extra"] += expected < hits
    return {
        "pages": len(docs),
        "pages_with_people": sum(bool(mentions) for mentions in docs.values()),
        "gold_spans": len(rows),
        "assigned": len(groups),
        "qid_spans": sum(n for key, n in labels.items() if key.startswith("Q")),
        "qids": sum(key.startswith("Q") for key in labels),
        "qids_on_multiple_pages": sum(len(pages) > 1 for pages in id_pages.values()),
        "qids_in_multiple_volumes": sum(
            len({doc.split("/")[0] for doc in pages}) > 1 for pages in id_pages.values()
        ),
        "explicit_nil": labels["NIL"],
        "unspecified": labels["_"],
        "unspecified_secondary": sum(
            r.label == "_" and r.secondary for r in rows.values()
        ),
        "pairs": pair_counts,
        "reader_queries": answers,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("native_binary", type=Path)
    args = parser.parse_args()
    docs = load(args.directory)
    groups = run_native(args.native_binary.resolve(), docs)
    print(json.dumps(score(docs, groups), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
