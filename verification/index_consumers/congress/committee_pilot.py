# SPDX-License-Identifier: MIT OR Apache-2.0
"""Replay committee printings against the Congress roster that was available at decision time."""

import argparse
import json
import sqlite3
import time
from bisect import bisect_left
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from functools import cache
from math import ceil
from pathlib import Path
from tempfile import TemporaryDirectory

DAY = timedelta(days=1)
CUTOFF = datetime(2026, 1, 1, tzinfo=timezone.utc)
SCORE_PAIRS = {}


def load(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def instant(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def normalize_token(word):
    # Match the possessive and trailing-apostrophe handling in native string_similarity.
    word = word.lower()
    while word.endswith("'s"):
        word = word[:-2]
    return word.rstrip("'")


@cache
def tokens(name):
    return frozenset(normalize_token(word) for word in name.split())


def similarity(printed, alias):
    key = (printed, alias)
    if key not in SCORE_PAIRS:
        left, right = tokens(printed), tokens(alias)
        union = left | right
        SCORE_PAIRS[key] = len(left & right) / len(union) if union else 1.0
    return SCORE_PAIRS[key]


class Directory:
    def __init__(self, records):
        self.rank = cache(self._rank)
        self.members = {}
        self.by_token = defaultdict(set)
        for identity, entry in records.items():
            aliases = {f"{entry['first']} {entry['last']}"}
            if entry["official_full"]:
                aliases.add(entry["official_full"])
            self.members[identity] = (entry["chamber"], aliases)
            for alias in aliases:
                for token in tokens(alias):
                    self.by_token[token].add(identity)

    def _rank(self, name, chamber):
        # Gold never enters this function. A shared token is required for a
        # positive native score, so the reverse index preserves all positive pairs.
        retrieved = set()
        for token in tokens(name):
            retrieved.update(self.by_token[token])
        scored = {}
        for candidate in retrieved:
            role, aliases = self.members[candidate]
            if not chamber or chamber == role:
                scored[candidate] = max(similarity(name, alias) for alias in aliases)
        if not scored:
            return None, 0.0, frozenset()
        best = max(scored.values())
        winners = [candidate for candidate, score in scored.items() if score == best]
        return winners[0] if len(winners) == 1 else None, best, frozenset(scored)


def catalogs(events):
    dates, views, records = [], [], {}
    for event in events:
        for identity in event["remove"]:
            del records[identity]
        for row in event["upsert"]:
            records[row["id"]] = row
        stamp = instant(event["published_at"])
        assert not dates or stamp > dates[-1]
        dates.append(stamp)
        views.append(Directory(records))
    return dates, views


def printed_chamber(key):
    return {"H": "rep", "S": "sen"}.get(key[0])


def next_week(at):
    # First Monday 00:00 UTC at or after publication.
    midnight = at.replace(hour=0, minute=0, second=0, microsecond=0)
    following = midnight + ((7 - at.weekday()) % 7) * DAY
    return following if following >= at else following + 7 * DAY


def quantile(values, p):
    return sorted(values)[ceil(p * len(values)) - 1] if values else None


def decide(view, name, committee, cutoff, use_chamber):
    winner, score, candidates = view.rank(
        name, printed_chamber(committee) if use_chamber else None
    )
    return winner if score >= cutoff else None, candidates


def queue_sensitivity(review, views, before):
    """Simulate capacity on archival cases unresolved at their first decision.

    Perfect review can identify a member present in the catalog at service time.
    Distinct committee appearances consume separate slots. A later catalog update
    does not cancel these fixed historical review requests.
    """
    queues = {}
    for capacity in [0, 5, 20]:
        # One token per occurrence. At most capacity oracle adjudications at
        # each UTC midnight. Unused daily capacity cannot be carried forward.
        next_free = None
        used = 0
        serviced, resolved, within_week = 0, 0, 0
        for received, _, _, _, gold in sorted(review, key=lambda row: row[:4]):
            first = received.replace(hour=0, minute=0, second=0, microsecond=0)
            if first < received:
                first += DAY
            if next_free is None or next_free < first:
                next_free, used = first, 0
            if capacity == 0:
                break
            if used == capacity:
                next_free, used = next_free + DAY, 0
            if next_free >= CUTOFF:
                break
            used += 1
            serviced += 1
            if gold in views[before(next_free)].members:
                resolved += 1
                within_week += next_free - received <= 7 * DAY
        queues[str(capacity)] = {
            "reviewed": serviced,
            "oracle_linked": resolved,
            "oracle_linked_within_7d_of_decision": within_week,
            "unresolved_at_end": len(review) - resolved,
        }
    return queues


def run(committees, roster_events):
    roster_dates, views = catalogs(roster_events)
    roster_by_revision = {event["revision"]: i for i, event in enumerate(roster_events)}

    def before(stamp):
        index = bisect_left(roster_dates, stamp) - 1
        assert index >= 0
        return index

    queries = [
        (instant(evt["published_at"]), evt["revision"], committee, name, ordinal, gold)
        for evt in committees
        for committee, name, ordinal, gold, _ in evt["upsert"]
        if gold
    ]
    results = {}
    decisions = {}
    policies = [
        ("prior", 1.0, True),
        ("prior", 0.5, True),
        ("atomic", 1.0, True),
        ("atomic", 0.5, True),
        ("weekly", 1.0, True),
        ("weekly", 0.5, True),
        # Removing the acceptance threshold still requires a unique positive maximum.
        ("prior", 0.0, True),
        ("prior", 0.5, False),
    ]
    for lag, threshold, use_chamber in policies:
        counts = Counter(
            correct=0, wrong=0, unresolved=0, repaired_wrong=0, repaired_unresolved=0
        )
        counts["occurrences"] = len(queries)
        by_period = defaultdict(Counter)
        review = []
        choices = []
        wait = []
        repaired = []
        wrong_samples = {}
        repaired_samples = {}
        missing_samples = []
        for at, revision, committee, name, ordinal, gold in queries:
            publish = next_week(at) if lag == "weekly" else at
            index = before(publish)
            if lag == "atomic" and revision in roster_by_revision:
                index = roster_by_revision[revision]
                assert roster_dates[index] == at
            view = views[index]
            chosen, retrieved = decide(view, name, committee, threshold, use_chamber)
            choices.append(chosen)
            wait.append((publish - at).total_seconds() / 3600)
            counts["in_decision_roster"] += gold in view.members
            counts["among_retrieved"] += gold in retrieved
            if chosen is None:
                outcome = "unresolved"
                review.append((publish, committee, name, ordinal, gold))
                if gold not in retrieved and len(missing_samples) < 3:
                    missing_samples.append(
                        (at.isoformat(), committee, name, gold, gold in view.members)
                    )
            elif chosen == gold:
                outcome = "correct"
            else:
                outcome = "wrong"
                wrong_samples.setdefault(
                    (name, chosen, gold),
                    (at.isoformat(), committee, name, chosen, gold),
                )
            counts[outcome] += 1
            period = "2025" if at.year == 2025 else "2021-2024"
            by_period[period][outcome] += 1
            until = publish + 30 * DAY
            if chosen == gold or until > CUTOFF:
                continue
            counts["at_risk_30d"] += 1
            # Re-query all still-incorrect cases at each new catalog edition.
            # This audit uses gold only to count first recovery, never for ranking.
            for later in range(index + 1, len(views)):
                if roster_dates[later] > until:
                    break
                final, _ = decide(views[later], name, committee, threshold, use_chamber)
                if final == gold:
                    counts[
                        "repaired_unresolved" if chosen is None else "repaired_wrong"
                    ] += 1
                    delay = (roster_dates[later] - at).total_seconds() / 3600
                    repaired.append(delay)
                    repaired_samples.setdefault(
                        (name, gold),
                        (
                            at.isoformat(),
                            committee,
                            name,
                            chosen,
                            gold,
                            roster_dates[later].isoformat(),
                        ),
                    )
                    break

        assert sum(counts[k] for k in ("correct", "wrong", "unresolved")) == len(
            queries
        )
        rule = "unique-positive" if threshold == 0 else f"{threshold:g}"
        policy_name = f"{lag}-{rule}" + ("" if use_chamber else "-no-chamber")
        decisions[policy_name] = choices
        results[policy_name] = {
            "counts": dict(counts),
            "by_commit_year": dict(by_period),
            "publication_delay_hours_p50_p95": [
                quantile(wait, 0.50),
                quantile(wait, 0.95),
            ],
            "first_recovery_hours_p50_p95": [
                quantile(repaired, 0.50),
                quantile(repaired, 0.95),
            ],
            "capacity_per_day": queue_sensitivity(review, views, before),
            "wrong_examples": list(wrong_samples.values())[:8],
            "recovery_examples": list(repaired_samples.values())[:8],
            "missing_examples": missing_samples,
        }
    names = defaultdict(set)
    for _, _, _, printed, _, gold in queries:
        names[printed].add(gold)
    return {
        "queries": len(queries),
        "atomic_weekly_disagreements": {
            rule: sum(
                left != right
                for left, right in zip(
                    decisions[f"atomic-{rule}"],
                    decisions[f"weekly-{rule}"],
                    strict=True,
                )
            )
            for rule in ("1", "0.5")
        },
        "unique_printed_name_gold_pairs": sum(
            len(targets) for targets in names.values()
        ),
        "upstream_same_name_multiple_targets": {
            n: sorted(ids) for n, ids in names.items() if len(ids) > 1
        },
        "policy": results,
    }


def source_search(committees):
    latency, active = [], {}
    with TemporaryDirectory() as tmp:
        db = Path(tmp) / "index.db"
        writer, reader = sqlite3.connect(db), sqlite3.connect(db)
        with writer:
            writer.execute(
                "CREATE TABLE printed (committee TEXT, name TEXT, ordinal INTEGER, PRIMARY KEY(committee,name,ordinal))"
            )
            writer.execute("CREATE INDEX by_name ON printed (name)")
        for event in committees:
            upsert = [tuple(row[:3]) for row in event["upsert"]]
            removed = [tuple(row) for row in event["remove"]]
            if not upsert and not removed:
                continue
            started = time.perf_counter_ns()
            with writer:
                writer.executemany(
                    "DELETE FROM printed WHERE committee=? AND name=? AND ordinal=?",
                    removed,
                )
                writer.executemany(
                    "INSERT OR REPLACE INTO printed VALUES(?,?,?)", upsert
                )
            # Read from an independent connection and compare the exact-name search
            # against the expected CURRENT committee occurrences of the selected name.
            sample = (upsert or removed)[0][1]
            actual = set(
                reader.execute(
                    "SELECT committee,name,ordinal FROM printed WHERE name=?", (sample,)
                )
            )
            elapsed = (time.perf_counter_ns() - started) / 1e6
            for k in removed:
                del active[k]
            for k in upsert:
                active[k] = None
            assert actual == {k for k in active if k[1] == sample}
            latency.append(elapsed)
        assert reader.execute("SELECT count(*) FROM printed").fetchone()[0] == len(
            active
        )
        writer.close()
        reader.close()
    return {
        "editions": len(latency),
        "sqlite": sqlite3.sqlite_version,
        "local_file_publish_and_exact_lookup_ms_p50_p95": [
            quantile(latency, 0.50),
            quantile(latency, 0.95),
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("committee", type=Path)
    parser.add_argument("roster", type=Path)
    parser.add_argument(
        "--pairs",
        type=Path,
        help="write distinct scored pairs with Python Jaccard values for native concordance",
    )
    parser.add_argument(
        "--time-search",
        action="store_true",
        help="time disk SQLite publication and exact lookup",
    )
    args = parser.parse_args()
    committees = load(args.committee)
    summary = run(committees, load(args.roster))
    if args.time_search:
        summary["source_search"] = source_search(committees)
    if args.pairs:
        with args.pairs.open("w", encoding="utf-8") as out:
            for (a, b), expected in sorted(SCORE_PAIRS.items()):
                assert all("\t" not in name and "\n" not in name for name in (a, b))
                print(a, b, expected, sep="\t", file=out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
