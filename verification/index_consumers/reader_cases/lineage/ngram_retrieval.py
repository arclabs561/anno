# SPDX-License-Identifier: MIT OR Apache-2.0
"""Evaluate prior-edition name lookup, suggestions and selective links on full ROR catalogs."""

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from evaluate import HERE, key, read_reference, read_snapshot


@dataclass(frozen=True)
class Query:
    cohort: str
    source: str
    name: str
    target: str | None
    excluded: str = ""


def release_queries(before, after, decisions):
    queries = []
    stale = Counter()
    for i, row in sorted(after.items()):
        old = before.get(i)
        if old is None:
            if row.status == "active":
                queries.append(Query("new_accession", i, row.display, None))
            continue
        if read_reference(before, i) != read_reference(after, i):
            stale["stale_answer"] += 1
            if old == row:
                stale["stale_through_target"] += 1
        if old.status != "active":
            continue
        if row.status == "active" and old.display != row.display:
            queries.append(Query("renamed", i, row.display, i))
        elif row.status != "active":
            targets = [s for s in row.successors if s in after and after[s].status == "active"]
            if len(targets) == 1 and targets[0] in before and before[targets[0]].status == "active":
                queries.append(Query("sole_successor", i, old.display, targets[0], i))
    for i, target in sorted(decisions.items()):
        if (
            i in before
            and target in before
            and before[i].status == before[target].status == "active"
        ):
            queries.append(Query("issue_duplicate", i, before[i].display, target, i))
    return queries, stale


def lookup(index, before, requests):
    # Every query receives the same audit output. Only its name and excluded accession enter retrieval.
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as data:
        for accession, row in before.items():
            if row.status == "active":
                fields = [
                    "I",
                    accession,
                    key(row.display),
                    *sorted({key(name) for name in row.names}),
                ]
                data.write("\t".join(fields) + "\n")
        for request in requests:
            data.write("\t".join(["Q", request.excluded, key(request.name)]) + "\n")
        data.seek(0)
        with subprocess.Popen(
            [str(index)], stdin=data, stdout=subprocess.PIPE, text=True, encoding="utf-8"
        ) as child:
            for request, line in zip(requests, child.stdout, strict=True):
                display, exact, count, ranks, block = line.rstrip("\n").split("\t")
                ranked = []
                for item in ranks.split(",") if ranks else ():
                    accession, numerator, denominator = item.split(":")
                    ranked.append((accession, int(numerator), int(denominator)))
                yield (
                    request,
                    (display.split(",") if display else []),
                    (exact.split(",") if exact else []),
                    int(count),
                    ranked,
                    (block.split(",") if block else []),
                )
            if child.wait():
                raise RuntimeError("candidate process failed")


def decision(ranked):
    if not ranked:
        return None, "no_candidate"
    winner, numerator, denominator = ranked[0]
    if 10 * numerator < 9 * denominator:
        return None, "low_score"
    other_num, other_den = ranked[1][1:] if len(ranked) > 1 else (0, 1)
    if 10 * (numerator * other_den - other_num * denominator) < denominator * other_den:
        return None, "small_margin"
    return winner, "accepted"


def outcome(answer, target):
    if answer is None:
        return ("abstain", 1) if target else ("correct_absent", 0)
    return ("correct", 0) if answer == target else ("wrong", 10)


def evaluate(catalog, requests, binary, cohorts, writer, period, edition):
    for q, displays, exact, nblocked, ranked, block in lookup(binary, catalog, requests):
        fuzzy, reason = decision(ranked)
        predictions = {
            "display": displays[0] if len(displays) == 1 else None,
            "all_names": exact[0] if len(exact) == 1 else None,
            "trigram": fuzzy,
        }
        known = q.target is not None
        top = {row[0] for row in ranked}
        c = cohorts[q.cohort]
        c["queries"] += 1
        c["blocked_ids"] += nblocked
        c["target_blocked"] += int(known and q.target in block)
        c["target_top5"] += int(q.target in top)
        c["target_top1"] += int(bool(ranked) and ranked[0][0] == q.target)
        c["target_exact_alias_set"] += int(q.target in exact)
        c["trigram." + reason] += 1
        states = {}
        costs = {}
        for arm, accepted in predictions.items():
            states[arm], costs[arm] = outcome(accepted, q.target)
            c[f"{arm}.{states[arm]}"] += 1
            c[f"{arm}.loss"] += costs[arm]
        c[f"all_names_to_trigram.{states['all_names']}.{states['trigram']}"] += 1
        writer.write(
            json.dumps(
                dict(
                    period=period,
                    edition=edition,
                    cohort=q.cohort,
                    source=q.source,
                    printed=q.name,
                    expected=q.target,
                    excluded=q.excluded,
                    display=displays,
                    aliases=exact,
                    blocked=nblocked,
                    target_blocked=(q.target in block if known else None),
                    top5=ranked,
                    answers=predictions,
                    actions=states,
                    loss=costs,
                    reason=reason,
                ),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )


def manifest(name):
    with (HERE / name).open() as f:
        return list(csv.DictReader(f))


def compiled_index(tmp, supplied=None):
    if supplied:
        return Path(supplied).resolve()
    binary = Path(tmp) / "ngrams"
    subprocess.run(
        ["rustc", "--edition", "2021", "-O", str(HERE / "ngrams.rs"), "-o", str(binary)], check=True
    )
    return binary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--binary", type=Path, help="already compiled standalone ngrams.rs")
    args = parser.parse_args()
    releases = {r["version"]: r for r in manifest("ngram-earlier.csv") + manifest("releases.csv")}
    decisions = defaultdict(dict)
    for row in manifest("duplicate_decisions.csv"):
        decisions[row["version"]][row["source"]] = row["target"]
    totals = defaultdict(lambda: defaultdict(Counter))
    stale = {}
    by_edition = {}
    with (
        tempfile.TemporaryDirectory() as tmp,
        args.ledger.open("w", encoding="utf-8", newline="\n") as writer,
    ):
        binary = compiled_index(tmp, args.binary)
        for period, versions in (
            ("uninspected_earlier", range(0, 3)),
            ("reused_later", range(10, 17)),
        ):
            before = None
            for n in versions:
                version = f"v2.{n}"
                after = read_snapshot(releases[version], args.cache, args.fetch)
                if before is not None:
                    requests, stale[version] = release_queries(before, after, decisions[version])
                    counts = defaultdict(Counter)
                    evaluate(before, requests, binary, counts, writer, period, version)
                    by_edition[version] = counts
                    for cohort, metrics in counts.items():
                        totals[period][cohort].update(metrics)
                    print(f"{version}: {len(requests)} queries", file=sys.stderr, flush=True)
                before = after
    with args.ledger.open("rb") as f:
        result = {
            "totals": totals,
            "by_edition": by_edition,
            "changed_reference_answers": stale,
            "ledger_sha256": hashlib.file_digest(f, "sha256").hexdigest(),
        }
    args.summary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    main()
