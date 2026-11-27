"""Measure the guarded fallback on every otherwise-skipped WikiGold sentence."""
import argparse
from collections import Counter
from itertools import groupby
import json
import os
from pathlib import Path

from run import ORT_SHA256, verify_runtime
from stacked import CHECKPOINT, FILES, digest, observe, offline_stack, text_sha
from nuner_fallback_routing import run_worker, timing


WIKIGOLD = "c797a64d0cf73ed058363f77671486bcfd413e70fda1726ad4a6ba624455225f"
LABELS = {"PER", "LOC", "ORG", "MISC"}


def load_cases(path):
    if digest(path) != WIKIGOLD:
        raise ValueError("Expected the pinned WikiGold CoNLL file")
    cases = []
    for block in path.read_text(encoding="utf-8").strip().split("\n\n"):
        if not block.strip() or block.strip() == "-DOCSTART- O":
            continue
        pairs = [line.rsplit(None, 1) for line in block.splitlines() if line.strip()]
        words, tags = zip(*pairs)
        if "-DOCSTART-" in words:
            raise ValueError("Document marker among tokens")
        text = " ".join(words)
        gold = set()
        start = 0
        # This file uses only O and I-TYPE. Adjacent tokens with the same
        # I-TYPE form one annotated surface, including at a sentence start.
        for tag, members in groupby(zip(words, tags), key=lambda item: item[1]):
            surface = " ".join(word for word, _ in members)
            end = start + len(surface)
            if tag != "O":
                if not tag.startswith("I-") or tag[2:] not in LABELS:
                    raise ValueError(f"Unknown WikiGold tag {tag}")
                gold.add((start, end, tag[2:]))
            start = end + 1
        cases.append((text, gold))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ["data", "stock", "candidate", "models", "output"]:
        parser.add_argument("--" + arg, type=Path, required=True)
    args = parser.parse_args()
    for worker in (args.stock, args.candidate):
        verify_runtime(worker)
    cases = load_cases(args.data)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    gates = run_worker(args.stock, "--probe", cases)
    skipped = [i for i, gate in enumerate(gates) if not gate["ran"]]
    controls = sorted((i for i, gate in enumerate(gates) if gate["ran"]),
                      key=lambda i: (text_sha(cases[i][0]), i))[:16]
    indices = skipped + controls
    texts = [cases[i][0] for i in indices]
    with offline_stack(args.models, bert=False) as env:
        stock = observe(args.stock, texts, False, env, timeout=1200)
        candidate = observe(args.candidate, texts, False, env, timeout=1200)

    counts = Counter()
    for i, old, new in zip(indices, stock, candidate, strict=True):
        if old["ran"] != gates[i]["ran"] or not new["ran"]:
            raise ValueError("Loaded runtime disagreed with the expected route")
        if old["ran"]:
            if old["spans"] != new["spans"]:
                raise ValueError("An already-routed control changed output")
            continue
        gold = cases[i][1]
        old_spans = {tuple(s) for s in old["spans"]}
        new_spans = {tuple(s) for s in new["spans"]}
        for name, predictions in [("stock", old_spans), ("candidate", new_spans)]:
            counts[name + "_exact"] += len(gold & predictions)
            counts[name + "_unmatched"] += sum(s[2] in LABELS for s in predictions - gold)
            counts[name + "_outside_inventory"] += sum(s[2] not in LABELS for s in predictions)
        counts["gold"] += len(gold)
        counts["exact_gained"] += len((new_spans - old_spans) & gold)
        counts["exact_lost"] += len((old_spans - new_spans) & gold)
        counts["new_unmatched"] += sum(s[2] in LABELS for s in (new_spans - old_spans) - gold)
        counts["removed_unmatched"] += sum(s[2] in LABELS for s in (old_spans - new_spans) - gold)
    size = len(skipped)
    report = {"dataset": "WikiGold", "source_sha256": WIKIGOLD, "sentences": len(cases),
              "input_panel_sha256": text_sha("".join(text_sha(cases[i][0]) for i in indices)),
              "ort_sha256": ORT_SHA256, "checkpoint": CHECKPOINT, "checkpoint_sha256": FILES,
              "workers_sha256": {"stock": digest(args.stock), "candidate": digest(args.candidate)},
              "cpu_affinity": sorted(os.sched_getaffinity(0)),
              "stock_skipped": size, "already_routed_controls": len(controls),
              "skipped_subset": dict(counts),
              "warm_skipped_latency": {"stock": timing(stock[:size]), "candidate": timing(candidate[:size])}}
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
