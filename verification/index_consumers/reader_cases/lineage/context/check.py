# SPDX-License-Identifier: MIT OR Apache-2.0
"""Offline checks for phrase matching, equal inspection, and truly withheld candidates."""
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import Record
from ngram_retrieval import Query, compiled_index, decision, lookup
from context.rerank import Location, matches, panel, place, record, rerank, tokens


def check(binary):
    assert matches(tokens("Lab, ＮＥＷ-YORK."), place("New York"))
    assert not matches(tokens("Newark"), place("New"))
    assert not matches(tokens("New Metro York"), place("New York"))
    assert place("U2") == place("D.C.") == ()
    assert place("St. Gallen") == ("st", "gallen")
    assert matches(tokens("Institute, 東京大阪。"), place("東京大阪"))
    both = Location((place("New York"), place("New")), place("Arcadia"))
    empty = Location((), ())
    locations = {"a": empty, "b": both, "c": both, "d": empty}
    ranks = [("a", 9, 10), ("b", 88, 100), ("c", 88, 100)]
    ordered = rerank("New York, Arcadia.", ranks, locations)
    assert ordered == [("b", 1, 1), ("c", 1, 1), ("a", 9, 10)]
    assert decision(ordered) == (None, "small_margin")
    # Multiple matching cities earn a single city bonus, without a country match.
    assert rerank("New York", [("b", 1, 2)], locations) == [("b", 29, 50)]
    # Both arms pass the old gates at equality. Capping can retain a native tie.
    cap = [("a", 1, 1), ("b", 99, 100)]
    assert [a for a, _, _ in rerank("Arcadia", cap, locations)] == ["a", "b"]
    assert decision([("a", 9, 10), ("d", 8, 10)]) == ("a", "accepted")
    second = rerank("New York, Arcadia", [("a", 4, 5), ("b", 39, 50)], locations)
    assert decision(second) == ("b", "accepted")
    p = panel()
    for invalid in ((ranks, [*ordered, ordered[0]]), ([*ranks, ranks[0]], ordered)):
        try:
            record(p, invalid, {"b"})
        except ValueError:
            pass
        else:
            raise AssertionError("a duplicate option changed one arm's review budget")
    record(p, (ranks, ordered), {"b"})
    record(p, (ranks, ordered), {"a"})
    assert p["budget"]["1"]["gains"] == p["budget"]["1"]["losses"] == 1
    assert p["budget"]["5"]["shown_each"] == 6 and p["budget"]["5"]["losses"] == 0
    assert p["order_changed"] == 2
    masked = panel(available_target=False)
    record(masked, ([("a", 9, 10)], [("a", 9, 10)]), {"hidden"})
    assert masked["gate"]["location"]["wrong"] == 1

    def active(name, *aliases):
        return Record(name, (name, *aliases), "active", ())

    # Removing BOTH owners of a shared surface reduces its gram frequencies once.
    # The competing "zzq" gram crosses the inclusive 2,048 frequency boundary.
    catalog = {str(i): active(f"abzzqx{i}") for i in range(2048)}
    catalog["g1"] = active("abzzqxgold", "Study Circle")
    catalog["g2"] = active("abzzqxgold", "Study Circle")
    request = Query("", "1", "¤zzq🚗", None)

    def physical_check(records, requests, ids):
        masks = {q.source: ids for q in requests}
        masked = list(lookup(binary, records, requests, catalog_masks=masks))
        kept = {i: r for i, r in records.items() if i not in ids}
        assert masked == list(lookup(binary, kept, requests))
        return masked

    assert list(lookup(binary, catalog, [request]))[0][3] == 0
    assert physical_check(catalog, [request], {"g1"})[0][3] == 0
    assert physical_check(catalog, [request], {"g1", "g2"})[0][3] == 2048
    # Shared aliases survive when they retain a live owner. Ties refill beyond
    # the original top five, even when several annotated owners are absent.
    catalog = {f"a{i}": active("Common Institute", f"Institute {i}") for i in range(8)}
    names = ("Common Institute", "Institute 2", "Omega")
    queries = [Query("", str(i), name, None) for i, name in enumerate(names)]
    results = physical_check(catalog, queries, {"a0", "a1", "a4", "missing"})
    assert [a for a, _, _ in results[0][4]] == ["a2", "a3", "a5", "a6", "a7"]


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        check(compiled_index(tmp))
    print("Affiliation context controls passed")
