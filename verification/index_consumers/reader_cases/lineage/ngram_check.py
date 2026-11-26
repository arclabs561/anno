# SPDX-License-Identifier: MIT OR Apache-2.0
"""Offline tests for the as-of candidate input, distinct-owner ranking and abstention policy."""

import tempfile

from commitments.check import check as check_affiliation_commitments
from context.check import check as check_affiliation_context
from evaluate import Record
from ngram_retrieval import Query, compiled_index, decision, lookup, outcome, release_queries


def active(name, *aliases):
    return Record(name, (name, *aliases), "active", ())


def check(binary):
    earlier = {
        "a": active("Arcadia Organization", "Arcadia Foundation"),
        "b": active("Arcadia Laboratory"),
        "hidden": Record("Arcadian Foundation", ("Arcadian Foundation",), "withdrawn", ()),
        "r": active("Western Forge"),
        "target": active("Western Forge"),
    }
    for i in range(6):
        earlier[f"tie{i}"] = active("Nara Association")
    later = {
        **earlier,
        "a": active("Arcadian Foundation"),
        "r": Record("Western Forge", ("Western Forge",), "withdrawn", ("target",)),
        "new": active("New Registry"),
    }
    queries, stale = release_queries(earlier, later, {"r": "target"})
    assert stale["stale_answer"] == 1
    queries.append(Query("control", "x", "Nara Association", "tie5"))
    rows = list(lookup(binary, earlier, queries))
    cases = {
        q.cohort: (display, alias, count, rank, block)
        for q, display, alias, count, rank, block in rows
    }
    displays, aliases, _, rank, block = cases["renamed"]
    assert (
        not displays
        and not aliases
        and rank[0][0] == "a"
        and "a" in block
        and "hidden" not in block
    )
    assert (
        rank[0][1] < rank[0][2]
    )  # The future published alias would have fabricated an exact match.
    display, alias, _, rank, _ = cases["issue_duplicate"]
    assert display == alias == ["target"] and decision(rank) == ("target", "accepted")
    _, aliases, count, rank, block = cases["control"]
    assert count >= 6 and len(aliases) == 6 and "tie5" in block
    assert [a for a, _, _ in rank] == [f"tie{i}" for i in range(5)]
    assert decision(rank) == (None, "small_margin")
    assert decision([("a", 9, 10), ("b", 8, 10)]) == ("a", "accepted")
    assert decision([("a", 9, 10), ("b", 81, 100)]) == (None, "small_margin")
    assert decision([("a", 89, 100)]) == (None, "low_score")
    assert outcome("target", "target") == ("correct", 0)
    assert outcome("target", "wrong") == ("wrong", 10)
    assert outcome(None, "target") == ("abstain", 1)
    assert outcome(None, None) == ("correct_absent", 0)
    many = {str(i): active(f"abzzqx{i}") for i in range(2049)}
    request = Query("cap", "source", "¤zzq🚗", "0")
    ((_, _, _, count, ranked, block),) = lookup(binary, many, [request])
    assert count == 0 and not ranked and not block


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        binary = compiled_index(directory)
        check(binary)
        check_affiliation_context(binary)
        check_affiliation_commitments(binary)
    print("ROR candidate controls passed")
