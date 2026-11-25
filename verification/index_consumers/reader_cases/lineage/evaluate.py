# SPDX-License-Identifier: MIT OR Apache-2.0
"""Replay pinned CC0 ROR snapshots and score retrieval against the preceding active catalog."""

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unicodedata
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent
MODES = ("display", "all_names")


@dataclass(frozen=True, slots=True)
class Record:
    display: str
    names: tuple
    status: str
    successors: tuple


def key(name):
    return " ".join(unicodedata.normalize("NFKC", name).casefold().split())


def keys(row, mode):
    if row.status != "active":
        return ()
    return {k for name in ((row.display,) if mode == "display" else row.names) if (k := key(name))}


def read_snapshot(spec, cache, fetch):
    path = cache / spec["file"]
    if not path.exists() and fetch:
        url = f"https://zenodo.org/api/records/{spec['record']}/files/{spec['file']}/content"
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as tmp:
            pending = Path(tmp) / spec["file"]
            urllib.request.urlretrieve(url, pending)
            with pending.open("rb") as f:
                if hashlib.file_digest(f, "sha256").hexdigest() != spec["sha256"]:
                    raise ValueError(f"download hash mismatch: {spec['file']}")
            pending.replace(path)
    with path.open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != spec["sha256"]:
            raise ValueError(f"hash mismatch: {path}")
    with zipfile.ZipFile(path) as archive:
        (filename,) = (n for n in archive.namelist() if n.endswith("-ror-data.json"))
        with archive.open(filename) as f:
            records = json.load(f)
    result = {}
    for r in records:
        accession = r["id"].removeprefix("https://ror.org/")
        (display,) = (n["value"] for n in r["names"] if "ror_display" in n["types"])
        result[accession] = Record(
            display,
            tuple(sorted({n["value"] for n in r["names"]})),
            r["status"],
            tuple(
                sorted(
                    {
                        s["id"].removeprefix("https://ror.org/")
                        for s in r["relationships"]
                        if s["type"] == "successor"
                    }
                )
            ),
        )
    if len(result) != len(records):
        raise ValueError("duplicate registry accession")
    return result


def build_names(rows):
    indexes = {mode: defaultdict(set) for mode in MODES}
    for accession, row in rows.items():
        for mode in MODES:
            for normalized in keys(row, mode):
                indexes[mode][normalized].add(accession)
    return indexes


def update_names(indexes, original, current, changed):
    for accession in changed:
        for mode in MODES:
            if accession in original:
                for k in keys(original[accession], mode):
                    ids = indexes[mode][k]
                    ids.remove(accession)
                    if not ids:
                        del indexes[mode][k]
            if accession in current:
                for k in keys(current[accession], mode):
                    indexes[mode][k].add(accession)


def read_reference(rows, accession):
    row = rows[accession]
    # The reader exposes the publisher's direct relationships including target status.
    # A relationship may have several targets or none; preserve the cited accession.
    return (
        row.status,
        tuple((s, rows[s].status if s in rows else "absent") for s in row.successors),
    )


def active_targets(state):
    return tuple(s for s, status in state[1] if status == "active")


def reverse_links(rows):
    result = defaultdict(set)
    for accession, row in rows.items():
        for s in row.successors:
            result[s].add(accession)
    return result


def name_decision(correct, candidates):
    if not candidates:
        return "empty"
    if len(candidates) > 1:
        return "ambiguous_with_target" if correct in candidates else "ambiguous_without_target"
    return "correct" if correct in candidates else "wrong"


def add_lookup(counts, cohort, accession, name, indexes, omit=None, details=None):
    counts[f"{cohort}.queries"] += 1
    for mode in MODES:
        candidates = indexes[mode].get(key(name), set())
        if omit is not None:
            candidates = candidates - {omit}
        status = name_decision(accession, candidates)
        counts[f"{cohort}.{mode}.{status}"] += 1
        if details is not None:
            details[mode] = sorted(candidates)


def transition(original, current, indexes, views, previous, version, decisions, emit):
    counts = Counter()
    changed = {i for i in original.keys() | current.keys() if original.get(i) != current.get(i)}
    counts.update(
        records_before=len(original),
        records_after=len(current),
        projected_source_rows_changed=len(changed),
        prior_ids_missing_from_release=len(original.keys() - current.keys()),
    )
    # Query before applying the next release. IDs, status and relationships never enter the name keys.
    for i, after in current.items():
        before = original.get(i)
        if after.status == "active" and before is None:
            add_lookup(counts, "first_seen_active", None, after.display, indexes)
        elif (
            after.status == "active"
            and before
            and before.status == "active"
            and before.display != after.display
        ):
            add_lookup(counts, "changed_active_name", i, after.display, indexes)
    # Closed curation issues explicitly describe some successor relationships as duplicate corrections.
    for source, (target, _) in decisions.items():
        counts["explicit_decisions"] += 1
        old, new, prior_target = original.get(source), current.get(source), original.get(target)
        if old and old.status == "active" and prior_target and prior_target.status == "active":
            add_lookup(counts, "issue_target_at_intake", target, old.display, indexes, omit=source)
        if old is None or old.status != "active":
            counts["explicit.source_previously_unavailable"] += 1
        elif new is None or new.status != "withdrawn" or target not in new.successors:
            counts["explicit.new_snapshot_disagrees"] += 1
        elif prior_target is None or prior_target.status != "active":
            counts["explicit.target_previously_unavailable"] += 1
        else:
            add_lookup(counts, "explicit_duplicate", target, old.display, indexes, omit=source)
            if target not in current or current[target].status != "active":
                counts["explicit.confirmed_target_not_active"] += 1
    affected = set(changed)
    # A target's state can change the answer for an untouched historical source.
    status_changed = {
        i
        for i in changed
        if (original[i].status if i in original else None)
        != (current[i].status if i in current else None)
    }
    old_reverse = reverse_links(original)
    new_reverse = reverse_links(current)
    for i in status_changed:
        affected.update(old_reverse.get(i, ()))
        affected.update(new_reverse.get(i, ()))
    next_views = dict(views)
    for i in affected:
        if i in current:
            next_views[i] = read_reference(current, i)
        else:
            next_views.pop(i, None)
    bulk = {i: read_reference(current, i) for i in current}
    if next_views != bulk:
        raise AssertionError(f"lineage reader diverged at {version}")
    counts["reader_rows_compared"] = len(bulk)
    for i in sorted(views.keys() & bulk.keys()):
        old_state, new_state = views[i], bulk[i]
        if old_state != new_state:
            counts["stale_answers_with_prior_state"] += 1
            counts[f"changed_answer.{old_state[0]}_to_{new_state[0]}"] += 1
            if i not in changed:
                counts["stale_via_target_change_only"] += 1
        if old_state[0] != "active" or new_state[0] == "active":
            continue
        counts["former_active"] += 1
        counts[f"former_active.{new_state[0]}"] += 1
        targets = active_targets(new_state)
        counts[
            f"former_active.active_successors.{('none', 'one', 'multiple')[min(2, len(targets))]}"
        ] += 1
        record = {
            "previous": previous,
            "edition": version,
            "id": i,
            "name": original[i].display,
            "status": new_state[0],
            "successors": list(new_state[1]),
            "active_successors": list(targets),
        }
        # Include only an unambiguous published successor that was available to retrieval.
        if len(targets) == 1:
            (target,) = targets
            before_target = original.get(target)
            if before_target is None or before_target.status != "active":
                counts["single_active_successor_absent_at_intake"] += 1
            else:
                record["prior_target_names"] = list(before_target.names)
                add_lookup(
                    counts,
                    "available_unique_successor",
                    target,
                    original[i].display,
                    indexes,
                    omit=i,
                    details=record,
                )
        if i in decisions:
            issue_target, record["curation_issues"] = decisions[i]
            record["explicit_duplicate_target"] = issue_target
            record["issue_target_previous_names"] = (
                list(original[issue_target].names) if issue_target in original else []
            )
            record["issue_target_new_status"] = (
                current[issue_target].status if issue_target in current else "absent"
            )
        if emit is not None:
            emit.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    update_names(indexes, original, current, changed)
    if indexes != build_names(current):
        raise AssertionError(f"name postings diverged at {version}")
    return next_views, counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cache", type=Path, help="directory containing the pinned releases")
    p.add_argument("--fetch", action="store_true", help="download missing archives from Zenodo")
    p.add_argument(
        "--cases", type=Path, help="optional CC0 per-transition relation and candidate rows"
    )
    p.add_argument("--out", type=Path, help="write full release and cohort counts as JSON")
    p.add_argument(
        "--verify", action="store_true", help="compare output with the pinned SHA-256 digests"
    )
    args = p.parse_args()
    with (HERE / "releases.csv").open() as f:
        releases = list(csv.DictReader(f))
    decisions = defaultdict(dict)
    with (HERE / "duplicate_decisions.csv").open() as f:
        for decision in csv.DictReader(f):
            table = decisions[decision["version"]]
            target, issues = table.setdefault(decision["source"], (decision["target"], []))
            if target != decision["target"]:
                raise ValueError(f"conflicting curation decision in {decision['version']}")
            issues.append(int(decision["issue"]))
    rows = read_snapshot(releases[0], args.cache, args.fetch)
    indexes = build_names(rows)
    views = {i: read_reference(rows, i) for i in rows}
    totals = defaultdict(Counter)
    cuts = []
    emit = args.cases.open("w") if args.cases else None
    try:
        for previous, spec in zip(releases, releases[1:]):
            current = read_snapshot(spec, args.cache, args.fetch)
            views, counts = transition(
                rows,
                current,
                indexes,
                views,
                previous["version"],
                spec["version"],
                decisions[spec["version"]],
                emit,
            )
            split = "earlier" if int(spec["version"].split(".")[1]) <= 10 else "later"
            totals[split].update(counts)
            cuts.append(
                {
                    "previous": previous["version"],
                    "edition": spec["version"],
                    "date": spec["date"],
                    "split": split,
                    "counts": dict(sorted(counts.items())),
                }
            )
            print(
                f"Checked {previous['version']} to {spec['version']}", file=sys.stderr, flush=True
            )
            rows = current
    finally:
        if emit is not None:
            emit.close()
    report = {
        "baseline_edition": releases[0]["version"],
        "last_earlier_edition": "v2.10",
        "cuts": cuts,
        "totals": {split: dict(sorted(c.items())) for split, c in totals.items()},
    }
    data = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.verify:
        with (HERE / "expected.sha256").open() as f:
            expected = {name: digest for digest, name in (line.split() for line in f)}
        if hashlib.sha256(data.encode()).hexdigest() != expected["results.json"]:
            raise ValueError("release or aggregate counts differ from the pinned result")
        if args.cases:
            with args.cases.open("rb") as f:
                if hashlib.file_digest(f, "sha256").hexdigest() != expected["cases.jsonl"]:
                    raise ValueError("per-case reader evidence differs from the pinned result")
    if args.out:
        args.out.write_text(data)
    else:
        print(data, end="")


if __name__ == "__main__":
    main()
