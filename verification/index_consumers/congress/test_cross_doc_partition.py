# SPDX-License-Identifier: MIT OR Apache-2.0
"""Controls for the cross-document pair denominator and the dated reader policy."""
import unittest

from cross_doc_partition import Observation, evaluate, reader_metrics, resolve_chamber_conflict
from transfer_rollcalls import PRIOR, QUERY, prior_rows, read_roll, score_lookup


def score(placements, latest):
    """Each occurrence supplies its withheld person ID and its assigned track group."""
    docs = {doc: [Observation("Vale", person, "venue") for person, _ in rows]
            for doc, rows in placements.items()}
    predictions = {(doc, i): group for doc, rows in placements.items()
                   for i, (_, group) in enumerate(rows)}
    return reader_metrics(docs, predictions, latest)


class CrossDocumentControls(unittest.TestCase):
    def test_unique_prior_lookup_defers_ambiguous_components(self):
        docs = {
            PRIOR: [Observation("Vale", "a", "rep"), Observation("Reed", "b", "rep"),
                    Observation("Alder", "c", "rep")],
            QUERY: [Observation("Other", "a", "rep"), Observation("Vale", "d", "rep"),
                    Observation("Reed", "b", "rep"), Observation("New", "e", "rep"),
                    Observation("Alder", "c", "rep")],
        }
        groups = {(doc, i): group for doc, assignments in ((PRIOR, [1, 2, 2]), (QUERY, [8, 1, 2, 8, 2]))
                  for i, group in enumerate(assignments)}
        result = score_lookup(docs, groups)
        self.assertEqual(result["queries"]["prior_target_present"], {
            "queries": 3, "correct": 0, "wrong": 0, "no_prior_in_component": 1, "multiple_priors_in_component": 2,
        })
        self.assertEqual(result["queries"]["prior_target_absent"], {
            "queries": 2, "correct": 0, "wrong": 1, "no_prior_in_component": 1, "multiple_priors_in_component": 0,
        })
        self.assertEqual(result["wrong_unique_with_identical_printing"], 1)
        self.assertEqual(result["pairs"], {"cross_document_pairs": 15, "tp": 2, "fp": 3, "fn": 1, "tn": 9})

    def test_heldout_filter_removes_prior_people_and_keeps_nonvoters(self):
        roll = b"""<rollcall-vote><vote-data>
            <recorded-vote><legislator name-id="a">Vale</legislator><vote>Present</vote></recorded-vote>
            <recorded-vote><legislator name-id="b">Vale</legislator><vote>Not Voting</vote></recorded-vote>
            </vote-data></rollcall-vote>"""
        rows, summary = read_roll(roll, {"a"})
        self.assertEqual([(m.name, m.person) for m in rows], [("Vale", "b")])
        self.assertEqual(summary, {"eligible": 1, "omitted": {"earlier_pilot": 1},
                                   "votes": {"Present": 1, "Not Voting": 1}})
        with self.assertRaisesRegex(ValueError, "appears twice"):
            read_roll(roll.replace(b'name-id="b"', b'name-id="a"'), {"a"})

    def test_identical_printing_can_denote_a_different_person(self):
        groups = {(PRIOR, 0): 1, (QUERY, 0): 1}
        for person, stratum, expected in (("a", "prior_target_present", "correct"),
                                          ("b", "prior_target_absent", "wrong")):
            docs = {PRIOR: [Observation("Vale", "a", "rep")], QUERY: [Observation("Vale", person, "rep")]}
            self.assertEqual(score_lookup(docs, groups)["queries"][stratum][expected], 1)

    def test_compare_prior_occurrences_across_local_group_renumbering(self):
        docs = {PRIOR: [Observation("Vale", "a", "rep"), Observation("Alder", "b", "rep")],
                QUERY: [Observation("Reed", "c", "rep"), Observation("Quay", "d", "rep")]}
        original = {(PRIOR, 0): 3, (PRIOR, 1): 8, (QUERY, 0): 3, (QUERY, 1): 8}
        renumbered = {address: 11 - group for address, group in original.items()}
        switched = dict(original)
        switched[QUERY, 0], switched[QUERY, 1] = switched[QUERY, 1], switched[QUERY, 0]
        self.assertEqual(prior_rows(docs, original), prior_rows(docs, renumbered))
        self.assertEqual(score_lookup(docs, original), score_lookup(docs, switched))
        self.assertEqual(prior_rows(docs, original), [(0,), (1,)])
        self.assertEqual(prior_rows(docs, switched), [(1,), (0,)])

    def test_prior_lookup_requires_distinct_dates(self):
        lookup = score({
            "2021-04-02/bulletin/Larch": [("a", 1), ("b", 1), ("c", 2)],
            "2021-04-02/bulletin/Silt": [("c", 2)],
            "2022-04-02/bulletin/Larch": [("b", 1), ("d", 3)],
            "2023-04-02/bulletin/Silt": [("a", 1), ("e", 2), ("c", 2), ("d", 4)],
            "2024-04-02/bulletin/Larch": [("a", 1), ("e", 2)],
        }, "2023-04-02")["prior_lookup"]
        self.assertEqual(lookup["queries"], 4)
        self.assertEqual(lookup["queries_with_history"], 3)  # e first appears at the query date.
        # Group 1 repeats the same mistaken b on two dates. Group 2 has two
        # documents on one date. Its two true hits for c disappear at the higher floor.
        self.assertEqual(lookup["minimum_dates"], {
            "1": {"shown_pages": 3, "pages_with_wrong_hits": 2,
                  "correct_hits": 3, "wrong_hits": 4, "missed_hits": 1},
            "2": {"shown_pages": 1, "pages_with_wrong_hits": 1,
                  "correct_hits": 1, "wrong_hits": 2, "missed_hits": 3},
        })

    def test_b3_counts_same_document_occurrences_and_singletons(self):
        result = score({
            "2021-04-02/bulletin/Larch": [("a", 1), ("b", 1)],
            "2022-04-02/bulletin/Silt": [("a", 1), ("c", 2)],
        }, "2022-04-02")["b3"]
        # Per-occurrence precisions are 2/3, 1/3, 2/3 and 1.
        self.assertAlmostEqual(result["precision"], 2 / 3)
        self.assertAlmostEqual(result["recall"], 1)
        self.assertAlmostEqual(result["f1"], 4 / 5)
        empty = score({}, "2022-04-02")
        self.assertEqual(empty["b3"], {"precision": 0.0, "recall": 0.0, "f1": 0.0})
        self.assertEqual(empty["prior_lookup"]["queries"], 0)

    def test_same_document_pairs_are_excluded(self):
        rows = {
            "a": [Observation("same", "one", "rep"), Observation("same", "two", "rep")],
            "b": [Observation("variant", "one", "rep"), Observation("new", "three", "rep")],
        }
        duplicate = {("a", 0): 1, ("a", 1): 1, ("b", 0): 2, ("b", 1): 1}
        counts = evaluate(rows, duplicate)
        self.assertEqual([counts[k] for k in ("cross_document_pairs", "tp", "fp", "fn", "tn")],
                         [4, 0, 2, 1, 1])

    def test_roles_can_change_between_cuts(self):
        a, b, later = "old/roster/rep", "old/roster/sen", "new/roster/sen"
        rows = {a: [Observation("N", "1", "rep")], later: [Observation("N", "1", "sen")]}
        groups = {(a, 0): 1, (later, 0): 1}
        changed_role = resolve_chamber_conflict(rows, groups)
        self.assertEqual(changed_role[a, 0], changed_role[later, 0])
        rows[b] = [Observation("N", "2", "sen")]
        groups[b, 0] = 1
        conflict = resolve_chamber_conflict(rows, groups)
        self.assertNotEqual(conflict[a, 0], conflict[b, 0])
        # A later role transition and an overlapping namesake can occur together.
        # Dividing the entire historical cluster by chamber would misgroup this future row.
        self.assertEqual(conflict[b, 0], conflict[later, 0])


if __name__ == "__main__":
    unittest.main()
