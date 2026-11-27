"""Evaluate raw native name proposals on full pinned LitBank excerpts; publish aggregates."""

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import tempfile

from coref_corpora import litbank
from gap_receipts import digest
from native_gap_neutral_type import file_hash

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "docs/eval/litbank-name-boundaries-selection.json"


@dataclass
class Document:
    work: str
    text: str
    names: set
    annotated: set


def panel(source):
    works = litbank.load_works(source, litbank.WORKS, litbank.SOURCE_SHA256)
    documents = []
    for work in sorted(works, key=int):
        text, coref, entities, _ = works[work]
        mentions = litbank._annotations(coref, text)
        entity_types = litbank._annotations(entities, text)
        proper = {span for span, tags in mentions.items()
                  if "PROP_PER" in tags and "PER" in entity_types.get(span, ())}
        annotated = set(entity_types) | {span for span, tags in mentions.items()
                                         if any(litbank.MENTION_TYPE.fullmatch(tag) for tag in tags)}
        documents.append(Document(work, text, proper, annotated))
    return documents


def by_words(text, names):
    return Counter("multiword" if len(text[a:b].split()) >= 2 else "one_word" for a, b in names)


def selection(documents):
    manifest = []
    lengths = Counter()
    for doc in documents:
        manifest.append([doc.work, sorted(doc.names), sorted(doc.annotated)])
        lengths.update(by_words(doc.text, doc.names))
    return {
        "revision": "3e50db0ffc033d7ccbb94f4d88f6b99210328ed8",
        "litbank_brat_manifest_sha256": litbank.SOURCE_SHA256,
        "work_and_gold_sha256": digest(manifest),
        "complete_works": len(documents),
        "source_scalars": sum(len(doc.text) for doc in documents),
        "exact_proper_person_by_words": dict(lengths),
        "gold_annotated_boundaries": sum(len(doc.annotated) for doc in documents),
    }


def traced(documents, worker):
    with tempfile.TemporaryDirectory() as tmp:
        source, output = Path(tmp) / "input.jsonl", Path(tmp) / "trace.jsonl"
        with source.open("w", encoding="utf-8") as stream:
            for doc in documents:
                print(json.dumps({"id": doc.work, "text": doc.text}), file=stream)
        env = dict(os.environ, ANNO_GAP_TRACE_INPUT=str(source), ANNO_GAP_TRACE_OUTPUT=str(output))
        subprocess.run([str(worker.resolve()), "gap_trace::trace_native_gap", "--ignored"],
                       env=env, stdout=subprocess.DEVNULL, check=True)
        with output.open(encoding="utf-8") as stream:
            return [json.loads(line) for line in stream]


def inventory(doc, trace, stage):
    if trace["id"] != doc.work:
        raise ValueError("raw trace work differs from gold work")
    spans = set(map(tuple, trace[stage]))
    if any(not 0 <= a < b <= len(doc.text) for a, b in spans):
        raise ValueError("raw trace span outside source")
    return spans


def classify(span, doc):
    if span in doc.names:
        return "exact_proper_person"
    if span in doc.annotated:
        return "exact_other_annotated"
    a, b = span
    if any(a <= x < y <= b for x, y in doc.names):
        return "contains_proper_person_without_matching_annotated_boundary"
    if any(a < y and x < b for x, y in doc.names):
        return "other_overlap_with_proper_person"
    return "unassessed"


def compare(documents, stock, candidate):
    results = {}
    for stage in ("proposed", "retained"):
        counts, work_changes, added, removed = Counter(), Counter(), Counter(), Counter()
        for doc, baseline, trial in zip(documents, stock, candidate, strict=True):
            old, new = inventory(doc, baseline, stage), inventory(doc, trial, stage)
            old_hits = by_words(doc.text, old & doc.names)
            new_hits = by_words(doc.text, new & doc.names)
            for arm, hits in (("stock", old_hits), ("candidate", new_hits),
                              ("gained", by_words(doc.text, (new - old) & doc.names)),
                              ("lost", by_words(doc.text, (old - new) & doc.names))):
                for key in ("multiword", "one_word"):
                    counts[f"{arm}_exact_{key}"] += hits[key]
            for key in ("multiword", "one_word"):
                diff = new_hits[key] - old_hits[key]
                direction = "more" if diff > 0 else "fewer" if diff < 0 else "equal"
                work_changes[f"{key}_{direction}"] += 1
            counts["stock_spans"] += len(old)
            counts["candidate_spans"] += len(new)
            added.update(classify(span, doc) for span in new - old)
            removed.update(classify(span, doc) for span in old - new)
        if added.total() - removed.total() != counts["candidate_spans"] - counts["stock_spans"]:
            raise ValueError("span change counts do not reconcile")
        results[stage] = {"counts": dict(counts), "paired_works": dict(work_changes),
                          "added": dict(added), "removed": dict(removed)}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--stock", type=Path, help="stock native Rust test executable")
    parser.add_argument("--candidate", type=Path, help="trial native Rust test executable")
    args = parser.parse_args()
    if bool(args.stock) != bool(args.candidate):
        parser.error("scoring requires both --stock and --candidate")
    docs = panel(args.source)
    frozen = selection(docs)
    if args.stock:
        if frozen != json.loads(FROZEN.read_text()):
            raise ValueError("selection differs from the frozen full-work panel")
        result = {"selection": frozen, "stock_binary_sha256": file_hash(args.stock),
                  "candidate_binary_sha256": file_hash(args.candidate),
                  "stages": compare(docs, traced(docs, args.stock), traced(docs, args.candidate))}
    else:
        result = frozen
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
