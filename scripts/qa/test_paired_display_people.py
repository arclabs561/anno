"""Check the paired holdout and its occurrence-specific display judgments."""
from collections import defaultdict
import json
from pathlib import Path
import unittest

from score_person_mentions import annotation_maps, gold_span


GROUND = Path(__file__).resolve().parents[2] / "testdata/grounded"


class PairedDisplayPeople(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = json.loads((GROUND / "paired_display_people.json").read_text(encoding="utf-8"))
        cls.docs = cls.panel["documents"]
        cls.by_id = {d["id"]: d for d in cls.docs}

    def test_paired_scenarios_and_distinct_sources(self):
        pairs = defaultdict(list)
        for doc in self.docs:
            pairs[doc["pair_id"]].append(doc)
        self.assertEqual(len(pairs), 12)
        self.assertEqual(len(self.docs), len(self.by_id))
        for pair, docs in pairs.items():
            with self.subTest(pair=pair):
                self.assertEqual(sorted(d["variant"] for d in docs), ["a", "b"])
                self.assertEqual(len({d["contrast"] for d in docs}), 1)
        source = {d["text"] for d in self.docs}
        development = json.loads((GROUND / "dual_span_people.json").read_text(encoding="utf-8"))
        self.assertEqual(len(source), 24)
        self.assertTrue(source.isdisjoint(d["text"] for d in development["documents"]))

    def test_display_ranges_and_protected_phrases(self):
        self.assertEqual(self.panel["offset_unit"], "unicode_scalar")
        for doc in self.docs:
            with self.subTest(doc=doc["id"]):
                mentions, lookups, negatives = annotation_maps(doc)
                controls = set()
                for control in doc["display_controls"]:
                    r = gold_span(control["span"], control["surface"], doc["text"], doc["id"])
                    self.assertNotIn(r, controls)
                    if r in lookups["referential"]:
                        self.assertIsNone(mentions[lookups["referential"][r]]["display_highlight"])
                    controls.add(r)
                self.assertTrue(negatives <= controls)
                for mention in mentions.values():
                    if mention["kind"] in {"pronoun", "nominal"}:
                        self.assertIsNone(mention["display_highlight"])
                    else:
                        self.assertEqual(mention["display_highlight"], mention["highlight"])

    def test_nested_possessor_and_unresolved_name(self):
        owner, assistant, _ = self.by_id["possessor_a"]["mentions"]
        self.assertNotEqual(owner["person_id"], assistant["person_id"])
        self.assertEqual(assistant["referential_surfaces"], ["Ota Lemm’s assistant"])
        self.assertIsNone(assistant["display_highlight"])
        self.assertIsNotNone(owner["display_highlight"])
        for doc in self.docs:
            if doc["pair_id"] == "namesakes":
                vocative, = (m for m in doc["mentions"] if m["kind"] == "ambiguous")
                self.assertIsNone(vocative["person_id"])
                self.assertEqual(vocative["highlight_surface"], "Henn")
                self.assertIsNotNone(vocative["display_highlight"])

    def test_unicode_and_no_person_controls(self):
        for doc in self.docs:
            if doc["pair_id"] == "unicode":
                self.assertIn("🪁", doc["text"])
                self.assertIn("Ine\u0301", doc["text"])
            if doc["pair_id"] == "empty":
                self.assertEqual(doc["mentions"], [])
                self.assertTrue(doc["display_controls"])


if __name__ == "__main__":
    unittest.main()
