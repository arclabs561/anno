"""Count DiscoTrack bridge antecedent spans retrievable from the visible prefix."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re


def tokens(text):
    return re.findall(r"\w+|[^\w\s]", text.casefold())


def run(path, marker, train):
    instances = json.loads(path.read_text())["instances"]
    counts, seen = Counter(), set()
    for row in instances:
        text, answer = row["variables"]["text"], row["answer"]
        key = (row.get("doc_id", row.get("doc")), text, answer)
        counts["rows"] += 1
        if key in seen:
            continue
        seen.add(key)
        counts["unique"] += 1
        counts["unique_train_document_overlap"] += key[0] in train
        if marker not in text:
            counts["missing_marker"] += 1
            continue
        prefix = tokens(text.split(marker, 1)[0])
        gold = tokens(answer)
        if not gold:
            raise ValueError("Empty answer")
        for width in (32, 128, "all"):
            recent = prefix if width == "all" else prefix[-width:]
            counts[f"exact_sequence_in_prior_{width}"] += any(
                recent[i:i+len(gold)] == gold for i in range(len(recent)))
    return counts


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("en", type=Path)
    p.add_argument("cs", type=Path)
    p.add_argument("gumbridge_train", type=Path)
    args = p.parse_args()
    train = {p.stem for p in args.gumbridge_train.glob("*.tsv")}
    if not train:
        raise ValueError("GUMBridge train directory is empty")
    print(json.dumps({lang: run(path, marker, train) for lang, path, marker in (
        ("en", args.en, "{"), ("cs", args.cs, "$$"))}, indent=2))
