"""Check the scalar offsets and one-to-one evaluation units in the person panel."""
import json
import unittest
from pathlib import Path

from score_person_mentions import score, source_sha256


FIXTURE = Path(__file__).resolve().parents[2] / "testdata/grounded/dual_span_people.json"


class DualSpanPersonPanel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = json.loads(FIXTURE.read_text(encoding="utf-8"))
        cls.docs = cls.panel["documents"]

    def test_frozen_panel_passes_the_public_scorer_with_no_proposals(self):
        rows = [{"id": d["id"], "source_sha256": source_sha256(d["text"]),
                 "mentions": [], "clusters": []} for d in self.docs]
        results = score(self.panel, {"results": rows})
        self.assertEqual(len(results["documents"]), len(self.docs))
        for contract in ("highlight", "referential"):
            self.assertEqual(results["totals"][contract]["candidates"]["gold_mentions"], 37)
            self.assertEqual(results["totals"][contract]["candidates"]["covered_mentions"], 0)

    def test_distinct_units_and_local_identity_references(self):
        self.assertEqual(len({d["id"] for d in self.docs}), len(self.docs))
        for doc in self.docs:
            with self.subTest(doc=doc["id"]):
                mentions = doc["mentions"]
                self.assertEqual(len({m["id"] for m in mentions}), len(mentions))
                people = [p["id"] for p in doc["people"]]
                self.assertEqual(len(set(people)), len(people))
                self.assertEqual(set(people), {m["person_id"] for m in mentions if m["person_id"] is not None})
                owner = {}
                for m in mentions:
                    self.assertEqual(m["person_id"] is None, m["kind"] == "ambiguous")
                    for span in [m["highlight"], *m["referential_spans"]]:
                        self.assertEqual(owner.setdefault(tuple(span), m["id"]), m["id"])
                for n in doc["negative_spans"]:
                    a, b = n["span"]
                    self.assertTrue(all(b <= c or d <= a for c, d in owner))

    def test_paired_roles_and_no_person_controls(self):
        by_id = {d["id"]: d for d in self.docs}
        roles = by_id["roles_only"]["mentions"]
        self.assertEqual([r["highlight_surface"] for r in roles], ["diver"] * 3)
        self.assertEqual([r["person_id"] for r in roles], ["A", "B", "B"])
        self.assertEqual(roles[0]["referential_surfaces"], ["The first diver"])
        self.assertEqual(roles[1]["referential_surfaces"], ["The second diver"])
        self.assertTrue(any(not d["mentions"] and d["negative_spans"] for d in self.docs))
        self.assertTrue(any(d["mentions"] and d["negative_spans"] for d in self.docs))
        self.assertTrue(any(len(d["mentions"]) == 1 for d in self.docs))
        self.assertIn("E\u0301va D’Aro", by_id["compound_punctuation"]["text"])


if __name__ == "__main__":
    unittest.main()
