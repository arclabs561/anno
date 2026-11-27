# SPDX-License-Identifier: MIT OR Apache-2.0
import json
from pathlib import Path
import tempfile
import unittest

from fcoref_identity_review import compare, load_packet, packet, read_labels


class IdentityReviewTests(unittest.TestCase):
    def test_exact_highlight_and_identity_are_distinct(self):
        row = {"ID": "item", "Text": "It was Zoë's bicycle.", "A": "Zoë", "A-offset": "7"}
        entries = packet([(row, "A", (7, 20))])
        first = {entry["code"]: truth or "different" for entry, truth in entries}
        second = {entry["code"]: truth or "uncertain" for entry, truth in entries}
        result = compare(entries, first, second)
        self.assertEqual(result["exact_highlight_matches"], 0)
        self.assertEqual(result["highlight_extra_characters"]["median"], 10)
        self.assertEqual(result["reviewer_cross_tab"]["different"]["uncertain"], 1)
        self.assertEqual(result["control_correct"], [8, 8])

    def test_duplicate_missing_or_unknown_labels_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "review.jsonl"
            for data in ([("x","same"),("x","different")], [("x","same")],
                         [("x","same"),("y","unknown")]):
                p.write_text("".join(json.dumps(dict(code=k,label=v)) + "\n" for k,v in data))
                with self.subTest(data=data), self.assertRaises(ValueError):
                    read_labels(p, {"x","y"})

    def test_changed_packet_rejected_before_accepting_any_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"packet.jsonl"
            p.write_text('{"code":"substituted"}\n')
            with self.assertRaisesRegex(ValueError, "frozen study"):
                load_packet(p)
