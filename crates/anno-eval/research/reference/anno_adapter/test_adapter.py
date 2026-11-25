"""Offline checks for document scope, Unicode alignment and diagnostic denominators."""

import copy
import unittest

from adapter import audit_replay, partitions, read_grounded, run_memory


class PredictedMentionAdapterTests(unittest.TestCase):
    def setUp(self):
        self.doc = {"id": "names", "text": "🦊 Léa signed. Léa left."}
        self.prediction = {
            "id": "doc-0000",
            "text": self.doc["text"],
            "coref_run": {"status": "completed", "backend": "mention-ranking"},
            "signals": [
                {"id": 7, "location": {"Text": {"start": 2, "end": 5}}, "surface": "Léa"},
                {"id": 11, "location": {"Text": {"start": 14, "end": 17}}, "surface": "Léa"},
                {"id": 19, "location": {"Discontinuous": {"segments": [[2, 4], [14, 16]]}},
                 "surface": "Lé Lé"},
            ],
            "tracks": {
                "3": {
                    "id": 3,
                    "signals": [{"signal_id": 7, "position": 0}, {"signal_id": 19, "position": 1}],
                },
            },
        }

    def test_unicode_coordinates_and_sparse_ids_are_preserved(self):
        self.assertEqual(
            read_grounded(self.doc, self.prediction, "mention-ranking"),
            ({7: (2, 5), 11: (14, 17)}, {3: [7, 19]}, [19]),
        )
        stale = copy.deepcopy(self.prediction)
        stale["signals"][0]["location"]["Text"] = {"start": 5, "end": 9}
        with self.assertRaisesRegex(ValueError, "not source-aligned"):
            read_grounded(self.doc, stale, "mention-ranking")

    def test_unknown_track_member_scope_is_rejected(self):
        stale = copy.deepcopy(self.prediction)
        stale["tracks"]["3"]["signals"][0]["doc_id"] = "different-document"
        with self.assertRaisesRegex(ValueError, "document-local"):
            read_grounded(self.doc, stale, "mention-ranking")

    def test_wrong_backend_cannot_be_counted_as_success(self):
        with self.assertRaisesRegex(ValueError, "expected completed"):
            read_grounded(self.doc, self.prediction, "fcoref")

    def test_excluded_signal_disqualifies_exact_native_cluster(self):
        locations, tracks, _ = read_grounded(self.doc, self.prediction, "mention-ranking")
        mention = {"a": {"person_id": "x"}}
        lookup = {(2, 5): "a"}
        report = partitions(tracks, locations, mention, lookup, set())
        self.assertEqual(report["cluster_details"][0]["excluded_signal_ids"], [19])
        self.assertEqual(report["clusters"]["exact_gold_clusters"], 0)
        self.assertEqual(report["clusters"]["clusters_with_excluded_signals"], 1)
        self.assertEqual(partitions({3: [7]}, locations, mention, lookup, set())
                         ["clusters"]["exact_gold_clusters"], 1)

    def test_missed_and_wrong_pairs_have_separate_denominators(self):
        locations = {7: (0, 1), 11: (2, 3), 19: (4, 5)}
        mentions = {key: {"person_id": person} for key, person in zip("abcd", "xxxy")}
        lookup = {(0, 1): "a", (2, 3): "b", (6, 7): "c", (4, 5): "d"}
        report = partitions({7: [7, 19], 11: [11]}, locations, mentions, lookup, set())
        self.assertEqual(report["pairs"]["false_merge_pairs"], 1)
        self.assertEqual(report["pairs"]["false_split_pairs_covered"], 1)
        self.assertEqual(report["pairs"]["missed_pairs_detection"], 2)
        self.assertEqual(report["clusters"]["mixed_known_identity_clusters"], 1)
        self.assertEqual(report["clusters"]["gold_entities_split_across_clusters"], 1)

    def test_causal_replay_keeps_original_ids_and_audit_does_not_choose_links(self):
        text = "same same"
        locations = {21: (0, 4), 88: (5, 9)}
        groups, rows, memory = run_memory(text, locations, "exemplars", 16, 512, 2, -3)
        self.assertEqual(groups, {21: [21, 88]})
        self.assertLessEqual(memory["bytes"], 512)
        lookup = {(0, 4): "a", (5, 9): "b"}
        matching = {"a": {"person_id": "x"}, "b": {"person_id": "x"}}
        namesakes = {"a": {"person_id": "x"}, "b": {"person_id": "y"}}
        gold = audit_replay(rows, locations, matching, lookup)
        negative = audit_replay(rows, locations, namesakes, lookup)
        self.assertEqual(gold["counts"]["accepted_gold_supported"], 1)
        self.assertEqual(negative["counts"]["known_wrong_or_impure_accepted_targets"], 1)
        self.assertEqual(negative["decisions"][1]["assigned_cluster_id"], 21)

    def test_retention_candidate_ranking_and_threshold_are_distinct(self):
        locations = {10: (0, 1), 11: (2, 3), 12: (4, 5)}
        lookup = {(0, 1): "a", (2, 3): "b", (4, 5): "c"}
        mentions = {"a": {"person_id": "x"}, "b": {"person_id": "y"}, "c": {"person_id": "x"}}

        def decision(signal, resident, candidates, assigned, accepted=False):
            return {
                "signal_id": signal, "resident_cluster_ids": resident,
                "candidate_cluster_ids": candidates, "assigned_cluster_id": assigned,
                "target_cluster_id": candidates[0] if candidates else None,
                "target_prior_signal_ids": candidates[:1], "accepted": accepted,
            }

        prior = [decision(10, [], [], 10), decision(11, [10], [10], 11)]
        cases = (
            (decision(12, [11], [11], 12), "retention_miss"),
            (decision(12, [10, 11], [11], 12), "candidate_miss"),
            (decision(12, [10, 11], [11, 10], 11, True), "ranking_miss"),
            (decision(12, [10, 11], [10, 11], 12), "threshold_abstention"),
        )
        for query, outcome in cases:
            with self.subTest(outcome=outcome):
                self.assertEqual(
                    audit_replay([*prior, query], locations, mentions, lookup)
                    ["decisions"][-1]["outcome"], outcome,
                )


if __name__ == "__main__":
    unittest.main()
