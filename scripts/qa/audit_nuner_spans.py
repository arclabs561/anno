"""Check exact NuNER span scores directly from MultiNERD BIO tags and saved word logits."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import groupby
import json
from pathlib import Path
import re

import numpy as np


def reference(rows):
    parts, spans = [], set()
    offset = 0
    for row in rows:
        words = row["tokens"]
        parts.append(" ".join(words))
        tags = row["ner_tags"]
        if len(words) != len(tags):
            raise ValueError("tag count differs")
        starts = []
        for token in words:
            starts.append(offset)
            offset += len(token) + 1
        for i, tag in enumerate(tags):
            if tag in (1, 3, 5):
                last = i
                while last + 1 < len(tags) and tags[last + 1] == tag + 1:
                    last += 1
                spans.add((starts[i], starts[last] + len(words[last]), tag // 2))
            elif tag in (2, 4, 6) and (i == 0 or tags[i-1] not in (tag - 1, tag)):
                raise ValueError("orphan inside tag")
    return "\n".join(parts), spans


def word_probabilities(scores, words):
    if scores.shape != (len(words), 3) or not np.isfinite(scores).all():
        raise ValueError("non-finite scores or different word count")
    with np.errstate(over="ignore"):
        return np.float32(1) / (np.float32(1) + np.exp(-scores.astype(np.float32)))


def predictions(scores, words, text, threshold):
    return predicted_spans(word_probabilities(scores, words), words, text, threshold)


def predicted_spans(probabilities, words, text, threshold):
    labels = np.where(probabilities.max(axis=1) >= np.float32(threshold), probabilities.argmax(axis=1), -1)
    entities = set()
    for label, entries in groupby(zip(labels, words, strict=True), key=lambda entry: entry[0]):
        if label < 0:
            continue
        run = list(entries)
        start, stop = run[0][1][0], run[-1][1][1]
        surface = text[start:stop].rstrip(".,;:!?)]")
        surface = re.sub("(?:’s)+$", "", re.sub("(?:'s)+$", "", surface))
        if surface:
            entities.add((start, start + len(surface), int(label)))
    return entities


def run(directory, receipt):
    cases = json.loads((directory / "panel.json").read_text())
    if len(cases) != len(receipt["cases"]) or receipt["labels_by_index"] != ["person", "organization", "location"]:
        raise ValueError("different panel or prompt")
    sources = {}
    for lang, checksum in receipt["test_file_sha256"].items():
        data = (directory / f"test_{lang}.jsonl").read_bytes()
        if sha256(data).hexdigest() != checksum:
            raise ValueError("source hash mismatch")
        sources[lang] = list(map(json.loads, data.splitlines()))
    results = {graph: defaultdict(Counter) for graph in ("fp32", "u8")}
    thresholds = (0.3, 0.5, 0.7)
    for i, (case, saved) in enumerate(zip(cases, receipt["cases"], strict=True)):
        if any(case[key] != saved[key] for key in ("group", "language", "test_rows_zero_based")):
            raise ValueError(f"panel membership changed at case {i}")
        rows = [sources[case["language"]][j] for j in case["test_rows_zero_based"]]
        text, gold = reference(rows)
        if sha256(text.encode()).hexdigest() != saved["text_sha256"] or gold != set(map(tuple, saved["gold"])):
            raise ValueError(f"source mismatch in case {i}")
        probabilities = {}
        for graph in results:
            scores = np.load(directory / f"{graph}-{i}.npy", allow_pickle=False)
            probabilities[graph] = word_probabilities(scores, case["words"])
            for t in thresholds:
                pred = predicted_spans(probabilities[graph], case["words"], text, t)
                if t == 0.5 and pred != set(map(tuple, saved[graph])):
                    raise ValueError(f"{graph}: different span inventory on case {i}")
                stats = results[graph][case["group"], str(t)]
                stats.update(tp=len(gold & pred), fp=len(pred - gold), fn=len(gold - pred))
        crossings = {
            str(t): int(np.count_nonzero((probabilities["fp32"] >= t) != (probabilities["u8"] >= t)))
            for t in thresholds
        }
        if crossings != saved["threshold_crossings"]:
            raise ValueError(f"paired threshold crossings differ on case {i}")

    group_sizes = Counter(case["group"] for case in cases)
    output = {}
    for graph, graph_results in results.items():
        reported_groups = receipt["models"][graph]["groups"]
        if group_sizes != {group: entry["n"] for group, entry in reported_groups.items()}:
            raise ValueError(f"panel size mismatch for {graph}")
        for (group, t), stats in graph_results.items():
            reported = receipt["models"][graph]["groups"][group]["thresholds"][t]
            if any(reported[k] != n for k, n in stats.items()):
                raise ValueError(f"count mismatch for {graph}/{group}/{t}")
        output[graph] = {group: stats for (group, t), stats in graph_results.items() if t == "0.5"}
    return output


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("work", type=Path)
    cli.add_argument("receipt", type=Path)
    args = cli.parse_args()
    print(json.dumps(run(args.work, json.loads(args.receipt.read_text())), indent=2))
