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
            gold = {(start + sentence.index("Marie"), start + sentence.index("Acme"))
                    for sentence, start in [(left, first), (right, second)]
                    if sentence in (short, long)}
            yield text, gold, second


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
        # Score occurrence pairs including duplicate predictions, in source character offsets.
        predicted = Counter((r["head"]["start"], r["tail"]["start"]) for r in report["relations"])
        matches = predicted & Counter(gold)
        extras = predicted - matches
        counts.update(matched=matches.total(), missed=len(gold) - matches.total(),
                      extra=extras.total(), extra_cross_sentence=sum(
                          n for (h, t), n in extras.items() if (h < second) != (t < second)))
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
