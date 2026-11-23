"""Check the scalar offsets and one-to-one evaluation units in the person panel."""
import json
import unittest
from pathlib import Path


FIXTURE = Path(__file__).resolve().parents[2] / "testdata/grounded/dual_span_people.json"


class DualSpanPersonPanel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = json.loads(FIXTURE.read_text(encoding="utf-8"))
        cls.docs = cls.panel["documents"]

    def test_all_spans_recover_recorded_surfaces(self):
        self.assertEqual(self.panel["offset_unit"], "unicode_scalar")
        for doc in self.docs:
            text = doc["text"]
            for m in doc["mentions"]:
                self.assertTrue(m["referential_spans"])
                self.assertEqual(len(m["referential_spans"]), len(m["referential_surfaces"]))
                pairs = list(zip(m["referential_spans"], m["referential_surfaces"]))
                pairs.append((m["highlight"], m["highlight_surface"]))
                for (a, b), surface in pairs:
                    with self.subTest(doc=doc["id"], mention=m["id"], surface=surface):
                        self.assertTrue(0 <= a < b <= len(text))
                        self.assertEqual(text[a:b], surface)
                a, b = m["highlight"]
                for s, e in m["referential_spans"]:
                    self.assertTrue(s <= a < b <= e)
            for negative in doc["negative_spans"]:
                a, b = negative["span"]
                self.assertTrue(0 <= a < b <= len(text))
                self.assertEqual(text[a:b], negative["surface"])

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
