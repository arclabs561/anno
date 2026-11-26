"""Replay historical aliases against the 80 frozen congressional roster editions.

Run extract_roster.py in verification/index_consumers/congress to generate the input.
Times include CLI startup, parsing both editions, lookup and JSON result decoding.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from math import ceil
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter

from reuters_holdout import lookup, write_edition

DIGEST = "f6c4246cdaff7b3094ae00cf820b9ed27462970c75bda0319460e0350290a337"


def run(roster, binary, root):
    raw = roster.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DIGEST:
        raise ValueError("unexpected roster publication interval")
    published = {}
    history = defaultdict(set)
    stats = defaultdict(Counter)
    times = defaultdict(list)
    changes = Counter()
    revisions = [json.loads(line) for line in raw.splitlines()]
    for n, revision in enumerate(revisions):
        updates = revision["upsert"]
        queries = [(r["official_full"], r["id"]) for r in updates if r["official_full"]]
        if n and queries:
            current = root / "current"
            write_edition(current, n + 1, [s for s, _ in queries])
            entries = {"default": [(s, k) for k, s in sorted(published.items()) if s],
                       "aliases": [(s, k) for k in sorted(published) for s in sorted(history[k])]}
            policies = list(entries)
            for policy in policies:
                write_edition(root / policy, n, [s for s, _ in entries[policy]])
                lookup(binary, current, root / policy, 0, [])  # one unmeasured warm-up per edition
            for i, (name, key) in enumerate(queries):
                hit = {}
                for policy in (policies if (n+i) % 2 else reversed(policies)):
                    begin = perf_counter()
                    page = lookup(binary, current, root / policy, i, [])
                    times[policy].append(1000 * (perf_counter() - begin))
                    ids = [entries[policy][c["track"]["track_id"]][1] for c in page["candidates"]]
                    count = stats[policy]
                    present = key in published
                    count["present" if present else "absent"] += 1
                    hit[policy] = key in ids
                    count["correct"] += hit[policy]
                    count["wrong"] += bool(ids) and not hit[policy]
                    count["wrong_on_absent"] += bool(ids) and not present
                    count["withheld_tie"] += page["withheld_tie"] is not None
                    count["eligible_comparisons"] += len(entries[policy])
                changes["gained"] += hit["aliases"] and not hit["default"]
                changes["lost"] += hit["default"] and not hit["aliases"]
        for key in revision["remove"]:
            del published[key]
        for row in updates:
            key, name = row["id"], row["official_full"]
            published[key] = name
            if name:
                history[key].add(name)
    for policy, samples in times.items():
        stats[policy]["median_ms"] = round(median(samples), 2)
        stats[policy]["p95_ms"] = round(sorted(samples)[ceil(len(samples) * .95)-1], 2)
    return {"editions": len(revisions), "start": revisions[0]["published_at"],
            "end": revisions[-1]["published_at"], "paired": changes, "policies": stats}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roster", type=Path)
    parser.add_argument("binary", type=Path)
    args = parser.parse_args()
    with TemporaryDirectory() as temp:
        print(json.dumps(run(args.roster, args.binary.resolve(), Path(temp)), indent=2))
