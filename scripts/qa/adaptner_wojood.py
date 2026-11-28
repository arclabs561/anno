"""Compare native Arabic AdaptNER to all original nesting layers in SinaLab's public test sample."""

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path


def load(path):
    sentences = []
    for block in path.read_text(encoding="utf-8").strip().split("\n\n"):
        words = []
        gold = set()
        active = {}
        pos = 0
        for line in block.splitlines() + [""]:
            fields = line.split()
            word = fields[0] if fields else ""
            tags = fields[1:]
            for depth in list(active):
                kind, start, end = active[depth]
                if depth >= len(tags) or tags[depth] != "I-" + kind:
                    gold.add((kind, start, end))
                    del active[depth]
            for depth, tag in enumerate(tags):
                if tag == "O":
                    continue
                p, kind = tag.split("-", 1)
                assert p in {"B", "I"}
                if depth not in active:
                    active[depth] = (kind, pos, pos + len(word))
                else:
                    active[depth] = (kind, active[depth][1], pos + len(word))
            if fields:
                words.append(word)
            pos += len(word) + 1
        sentences.append((" ".join(words), gold))
    return sentences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gold", type=Path)
    parser.add_argument("cli", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    digest = hashlib.sha256(args.gold.read_bytes()).hexdigest()
    assert digest == "28b02687c1619df1edacbbd4f9fdc43dddfa01f4221b633300fe47459f900a42"
    docs = load(args.gold)
    lines = subprocess.run(
        [
            str(args.cli.resolve()),
            "batch",
            "--stdin",
            "--model",
            "adaptner",
            "--format",
            "jsonl",
            "--quiet",
        ],
        input="".join(
            json.dumps({"id": str(i), "text": text}) + "\n"
            for i, (text, _) in enumerate(docs)
        ),
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    stats = Counter()
    nested = Counter()
    for i, ((text, g), line) in enumerate(zip(docs, lines, strict=True)):
        row = json.loads(line)
        assert row["id"] == str(i) and row["runtime_model"] == "Arabic-AdaptNER"
        raw = row["entities"]
        assert all(text[e["start"] : e["end"]] == e["text"] for e in raw)
        p = {(e["type"], e["start"], e["end"]) for e in raw}
        assert len(p) == len(raw)
        stats.update(tp=len(g & p), fp=len(p - g), fn=len(g - p))
        for ent in g:
            kind, start, end = ent
            parents = [
                other
                for other in g
                if other[1] <= start
                and other[2] >= end
                and (other[1], other[2]) != (start, end)
            ]
            if parents:
                key = (
                    "same_type_inner"
                    if any(k == kind for k, _, _ in parents)
                    else "cross_type_inner"
                )
                nested[f"{key}_gold"] += 1
                nested[f"{key}_recovered"] += ent in p
                for parent in parents:
                    group = "same_type_pair" if parent[0] == kind else "cross_type_pair"
                    nested[f"{group}_gold"] += 1
                    nested[f"{group}_both_recovered"] += ent in p and parent in p
    stats["exact_micro_f1"] = (
        2 * stats["tp"] / (2 * stats["tp"] + stats["fp"] + stats["fn"])
    )
    result = {
        "corpus_revision": "ef3a7f4e806a2d909b9107d757c5f65545ae83dc",
        "test_sha256": digest,
        "checkpoint_revision": "c3096335b1e8d00f51451e267b8034a0cc59876f",
        "sentences": len(docs),
        "metrics": stats,
        "strictly_contained": nested,
    }
    args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
