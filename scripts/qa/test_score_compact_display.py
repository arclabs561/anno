import copy
import unittest
from typing import Any

from check_span_assertions import source_sha256
from score_compact_display import score


class CompactDisplayTest(unittest.TestCase):
    def setUp(self):
        text = "Dr Jo met Jo and Li. She left."
        self.doc: dict[str, Any] = {
            "id": "d",
            "text": text,
            "people": [{"id": "A"}, {"id": "B"}],
            "mentions": [
                self.mention("first", "A", [3, 5], [[0, 5]], [3, 5]),
                self.mention("second", "B", [10, 12], [[10, 12]], [10, 12]),
                self.mention("li", None, [17, 19], [[17, 19]], [17, 19]),
                self.mention("she", "A", [21, 24], [[21, 24]], None),
            ],
            "display_controls": [
                {"span": [10, 19], "surface": "Jo and Li", "reason": "multiple_people"}
            ],
        }
        self.doc["negative_spans"] = []
        for m in self.doc["mentions"]:
            m["highlight_surface"] = text[slice(*m["highlight"])]
            m["referential_surfaces"] = [
                text[slice(*ref)] for ref in m["referential_spans"]
            ]
        self.f: dict[str, Any] = {
            "id": "d",
            "source_sha256": source_sha256(text),
            "mentions": [
                {"start": a, "end": b, "surface": text[a:b]}
                for a, b in [(0, 5), (10, 19), (21, 24)]
            ],
            "clusters": [[[0, 5], [10, 19], [21, 24]]],
        }
        self.ner: dict[str, Any] = {
            "id": "d",
            "source_sha256": source_sha256(text),
            "entities": [self.entity(a, b) for a, b in [(3, 5), (10, 12), (17, 19)]],
        }

    @staticmethod
    def mention(key, person, highlight, refs, display):
        return {
            "id": key,
            "person_id": person,
            "highlight": highlight,
            "referential_spans": refs,
            "display_highlight": display,
        }

    def entity(self, a, b):
        return {"start": a, "end": b, "text": self.doc["text"][a:b], "type": "PER"}

    def run_report(self):
        return score(
            {
                "schema_version": 1,
                "offset_unit": "unicode_scalar",
                "documents": [self.doc],
            },
            {"results": [self.f]},
            [self.ner],
        )

    def test_common_inventory_and_decoded_phrase_contracts(self):
        old = copy.deepcopy([self.f, self.ner])
        s = self.run_report()
        self.assertEqual(
            s["totals"]["global"]["standalone_ner"],
            {"eligible": 3, "correct": 3, "wrong": 0, "missed": 0, "emitted": 3},
        )
        self.assertEqual(s["totals"]["global"]["unique_contained"]["correct"], 1)
        self.assertFalse(s["gate"]["strict_correct_gain"])
        doc = s["documents"][0]
        self.assertEqual(
            doc["projection_abstentions"], {"zero": 1, "one": 1, "multiple": 1}
        )
        self.assertEqual(doc["phrases"]["unique_contained"]["correct"], 1)
        self.assertEqual(doc["phrases"]["contained_all"]["protected_emissions"], 2)
        self.assertEqual(old, [self.f, self.ner])

    def test_no_phrases_standalone_retains_global_opportunities(self):
        self.f["clusters"] = []
        self.ner["entities"].append(self.entity(6, 9))
        s = self.run_report()["totals"]["global"]
        self.assertEqual(
            (s["standalone_ner"]["correct"], s["standalone_ner"]["wrong"]), (3, 1)
        )
        self.assertEqual(s["unique_contained"]["missed"], 3)

    def test_protected_single_survivor_is_an_attribution_error(self):
        self.ner["entities"].pop()
        s = self.run_report()
        self.assertTrue(s["protected_per_attributions"])
        self.assertEqual(s["totals"]["global"]["unique_contained"]["correct"], 2)
        self.assertEqual(s["totals"]["phrases"]["unique_contained"]["correct"], 1)

    def test_nested_resolved_owner_differs_from_named_person(self):
        self.doc["display_controls"] = []
        self.doc["mentions"][2]["referential_spans"].append([10, 19])
        self.doc["mentions"][2]["referential_surfaces"].append("Jo and Li")
        self.doc["mentions"][2]["person_id"] = "A"
        self.ner["entities"] = [self.entity(10, 12)]
        s = self.run_report()["totals"]["phrases"]["unique_contained"]
        self.assertEqual((s["other_occurrence"], s["known_wrong_person"]), (1, 1))
        self.doc["mentions"][1]["person_id"] = None
        s = self.run_report()["totals"]["phrases"]["unique_contained"]
        self.assertEqual((s["other_occurrence"], s["known_wrong_person"]), (1, 0))

    def test_touching_ranges_and_repeated_surfaces_are_distinct_choices(self):
        first, second = self.entity(3, 5), self.entity(10, 12)
        self.ner["entities"] = [first, first.copy(), second, self.entity(12, 14)]
        # One wide source phrase contains two occurrences with the same name.
        self.f["mentions"].append({"start": 3, "end": 12, "surface": "Jo met Jo"})
        self.f["clusters"].append([[3, 12], [21, 24]])
        out = self.run_report()
        row = next(r for r in out["documents"][0]["detail"] if r["phrase"] == (3, 12))
        self.assertEqual(row["contained_count"], 2)
        self.assertEqual(row["views"]["unique_contained"]["display"], [])
        shared = next(
            r for r in out["documents"][0]["detail"] if r["phrase"] == (21, 24)
        )
        self.assertEqual(shared["cluster_indices"], [0, 1])
        self.assertEqual(out["totals"]["global"]["standalone_ner"]["correct"], 2)

    def test_relabel_same_offsets_changes_projection_without_mutating_groups(self):
        self.ner["entities"][0]["type"] = "ORG"
        self.ner["entities"].append(self.entity(3, 6))
        row = self.run_report()["documents"][0]["detail"][0]
        self.assertEqual(row["contained_count"], 0)
        self.assertEqual(row["views"]["full"]["display"], [(0, 5)])

    def test_receipt_corruption(self):
        for field, val in [("id", "bad"), ("source_sha256", "bad")]:
            orig = self.ner[field]
            self.ner[field] = val
            with self.assertRaises(ValueError):
                self.run_report()
            self.ner[field] = orig
        self.ner["entities"][0]["text"] = "bad"
        with self.assertRaises(ValueError):
            self.run_report()
