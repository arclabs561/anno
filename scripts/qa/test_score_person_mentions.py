import copy
import unittest
from typing import Any

from score_person_mentions import score, source_sha256


class DualSpanTest(unittest.TestCase):
    def setUp(self):
        text = "🦊 Dr. Jo met Jo. She left. Bell rang."
        self.doc: dict[str, Any] = {"id": "d", "text": text, "people": [{"id": "A"}, {"id": "B"}],
                    "mentions": [
                        {"id": "first", "person_id": "A", "highlight": [6, 8],
                         "referential_spans": [[2, 8], [6, 8]]},
                        {"id": "second", "person_id": "B", "highlight": [13, 15],
                         "referential_spans": [[13, 15]]},
                        {"id": "pronoun", "person_id": "A", "highlight": [17, 20],
                         "referential_spans": [[17, 20]]}],
                    "negative_spans": [{"span": [27, 31], "kind": "object"}]}
        self.gold: dict[str, Any] = {"schema_version": 1, "offset_unit": "unicode_scalar", "documents": [self.doc]}

    def row(self, spans=(), clusters=()):
        return {"id": "d", "source_sha256": source_sha256(self.doc["text"]),
                "mentions": [{"start": a, "end": b, "surface": self.doc["text"][a:b]}
                             for a, b in spans], "clusters": list(clusters)}

    def scores(self, spans=(), clusters=()):
        return score(self.gold, {"results": [self.row(spans, clusters)]})["totals"]

    def test_wider_referent_and_namesake_merge(self):
        a, b, pronoun = [2, 8], [13, 15], [17, 20]
        s = self.scores([a, b, pronoun, [27, 31], [32, 36]], [[a, b, pronoun]])
        highlight, ref = s["highlight"], s["referential"]
        self.assertEqual(highlight["candidates"]["covered_mentions"], 2)
        self.assertEqual(ref["candidates"], {
            "emitted": 5, "gold_mentions": 3, "matched_spans": 3, "covered_mentions": 3,
            "explicit_negative_spans": 1, "explicit_negative_matches": 1, "other_unmatched": 1})
        self.assertEqual(ref["cluster_members"]["emitted"], 3)
        self.assertEqual(ref["local_pairs"]["positive_pairs"], 1)
        self.assertEqual(ref["local_pairs"]["joined_positive"], 1)
        self.assertEqual(ref["local_pairs"]["negative_pairs"], 2)
        self.assertEqual(ref["local_pairs"]["joined_negative"], 2)

    def test_human_distractor_links_are_distinct_from_cross_person_merges(self):
        name, pronoun, object1, object2 = [6, 8], [17, 20], [27, 31], [32, 36]
        self.doc["negative_spans"].append({"span": object2, "kind": "nonperson"})
        nodes = [name, pronoun, object1, object2]
        pure = self.scores(nodes, [[name, pronoun], [object1, object2]])["highlight"]
        mixed = self.scores(nodes, [nodes])["highlight"]
        self.assertEqual(pure["local_pairs"], mixed["local_pairs"])
        self.assertEqual(pure["human_distractor"]["joined_pairs"], 0)
        self.assertEqual(mixed["human_distractor"], {
            "human_distractor_pairs": 6, "covered_pairs": 4, "joined_pairs": 4,
            "mixed_clusters": 1, "clusters": 1})

    def test_widening_beyond_allowed_phrase_and_disjoint_groups(self):
        wide, b, pronoun = [0, 8], [13, 15], [17, 20]
        s = self.scores([wide, b, pronoun], [[wide, pronoun]])["referential"]
        self.assertEqual(s["candidates"]["covered_mentions"], 2)
        self.assertEqual(s["local_pairs"]["joined_positive"], 0)
        self.assertEqual(s["local_pairs"]["joined_negative"], 0)

    def test_alternatives_cover_one_occurrence_and_no_invented_pairs(self):
        s = self.scores([[2, 8], [6, 8]], [[[2, 8], [6, 8]]])["referential"]
        self.assertEqual(s["candidates"]["matched_spans"], 2)
        self.assertEqual(s["candidates"]["covered_mentions"], 1)
        self.assertEqual(s["local_pairs"]["covered_positive"], 0)
        self.assertEqual(s["local_pairs"]["joined_positive"], 0)

    def test_shared_projected_span_keeps_distinct_clusters(self):
        first, second, pronoun = [6, 8], [13, 15], [17, 20]
        result = self.scores([first, second, pronoun], [[first, second], [first, pronoun]])
        s = result["highlight"]
        self.assertEqual(s["cluster_members"]["emitted"], 3)
        self.assertEqual(s["local_pairs"]["joined_positive"], 1)
        # Only first and second co-occur. A transitive merge would also link second to pronoun.
        self.assertEqual(s["local_pairs"]["joined_negative"], 1)

    def test_missing_candidates_preserve_pair_denominator(self):
        s = self.scores()["referential"]
        self.assertEqual(s["local_pairs"]["positive_pairs"], 1)
        self.assertEqual(s["local_pairs"]["covered_positive"], 0)
        self.assertEqual(s["cluster_members"]["gold_mentions"], 3)

    def test_no_people_control_and_unresolved_mention(self):
        gold = copy.deepcopy(self.gold)
        gold["documents"][0].update(people=[], mentions=[])
        scores = score(gold, {"results": [self.row([[27, 31]])]})["totals"]["highlight"]
        self.assertEqual(scores["candidates"]["explicit_negative_matches"], 1)
        self.assertEqual(scores["local_pairs"]["positive_pairs"], 0)
        self.doc["mentions"][2]["person_id"] = None
        s = self.scores([[6, 8], [17, 20]], [[[6, 8], [17, 20]]])["highlight"]
        self.assertEqual(s["local_pairs"]["positive_pairs"], 0)
        self.assertEqual(s["local_pairs"]["unresolved_joined_to_resolved"], 1)

    def test_invalid_or_stale_predictions_fail_closed(self):
        for row in [{**self.row(), "source_sha256": "0"*64},
                    self.row([[6, 8]], [[[6, 8], [13, 15]]]),
                    self.row([[6, 8], [6, 8]])]:
            with self.subTest(row=row), self.assertRaises(ValueError):
                score(self.gold, {"results": [row]})
