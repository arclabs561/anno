"""Measure the heuristic relation distance/recall tradeoff on authored repeated occurrences."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import tempfile


def cases():
    short = "Marie works for Acme Inc"
    long = "Marie, following a long interview with the team, works for Acme Inc"
    negative = "Marie left before Acme Inc arrived"
    for prefix, separator in [("β", "."), ("東京", "。"), ("مَعًا", "؟"), ("नमस्ते", "।"),
                              ("文字", "！"), ("A\u030a", "?")]:
        for left, right in [(short, short), (short, negative), (negative, short), (long, short)]:
            text = f"{prefix} — {left}{separator} {right}."
            first = len(prefix) + len(" — ")
            second = first + len(left) + len(separator) + 1
            gold = set()
            for sentence, start in [(left, first), (right, second)]:
                if sentence in (short, long):
                    head = start + sentence.index("Marie")
                    tail = start + sentence.index("Acme Inc")
                    gold.add(((head, head + len("Marie"), "PER"), "WORKS_FOR",
                              (tail, tail + len("Acme Inc"), "ORG")))
            yield text, gold, second


def score(text, relations, gold, second):
    def endpoint(entity):
        start, end = entity["start"], entity["end"]
        if not 0 <= start < end <= len(text) or entity["text"] != text[start:end]:
            raise ValueError(f"invalid source span at [{start}, {end})")
        return start, end, entity["type"]

    predicted = Counter((endpoint(r["head"]), r["type"], endpoint(r["tail"])) for r in relations)
    matches = predicted & Counter(gold)
    extras = predicted - matches
    return Counter(matched=matches.total(), missed=len(gold) - matches.total(), extra=extras.total(),
                   extra_cross_sentence=sum(n for (head, _, tail), n in extras.items()
                                            if (head[0] < second) != (tail[0] < second)))


def trial(binary, backend, threshold, distance, env):
    counts = Counter(matched=0, missed=0, extra=0, extra_cross_sentence=0)
    for text, gold, second in cases():
        cmd = [binary, "extract", "--quiet", "--model", backend, "--text", text, "--format", "json",
               "--extract-relations", "--relation-types", "WORKS_FOR", "--relation-threshold",
               str(threshold), "--relation-max-span-distance", str(distance)]
        report = json.loads(subprocess.check_output(cmd, env=env, timeout=60))
        expected = "tplinker-heuristic" if backend == "tplinker" else "relation-heuristic"
        if report["provenance"]["relation_producer"]["source"] != expected:
            raise ValueError(f"unexpected relation scorer for {backend}")
        counts.update(score(text, report["relations"], gold, second))
    return dict(backend=backend, relation_threshold=threshold, max_span_distance=distance, **counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as cache:
        env = {**os.environ, "XDG_CACHE_HOME": cache, "HF_HOME": cache, "HF_HUB_OFFLINE": "1",
               "ANNO_NO_DOWNLOADS": "1"}
        results = [trial(args.binary.resolve(strict=True), backend, threshold, distance, env)
                   for backend in ("heuristic", "tplinker") for threshold in (0.1, 0.55)
                   for distance in (120, 12)]
    print(json.dumps({"documents": sum(1 for _ in cases()), "runs": results}, indent=2))


if __name__ == "__main__":
    main()
