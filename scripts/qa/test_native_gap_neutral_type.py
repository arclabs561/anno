"""Check paired link accounting and exact supplied boundaries in the GAP replay."""

import json
import unittest
from unittest.mock import patch

import native_gap_neutral_type as study


class NeutralMentionTest(unittest.TestCase):
    def test_public_receipt_matches_patch_and_reconciles_all_scored_links(self):
        path = study.ROOT / "docs/eval/native-gap-neutral-mention.json"
        receipt = json.loads(path.read_text())
        self.assertEqual(receipt["patch_sha256"], study.file_hash(study.PATCH))
        panels = receipt["panels"]
        for split, size in {
            "development": 2000,
            "test": 2000,
            "validation": 454,
        }.items():
            result = panels[split]
            self.assertEqual(size, result["documents"])
            self.assertEqual(size, sum(result["gold_patterns"].values()))
            raw = result["predicted"]
            self.assertEqual(
                raw["stock_partitions_sha256"], raw["candidate_partitions_sha256"]
            )
            table = result["supplied"]
            delta = table["paired"]
            self.assertEqual(
                table["candidate"]["tp"] - table["stock"]["tp"],
                delta["correct_gained"] - delta["correct_lost"],
            )
            self.assertEqual(
                table["candidate"]["fp"] - table["stock"]["fp"],
                delta["incorrect_added"] - delta["incorrect_removed"],
            )
            self.assertEqual(
                delta["incorrect_added"],
                delta["incorrect_added_to_distractor"]
                + delta["incorrect_added_when_neither_gold"],
            )
        fresh = panels["test_subsets"]["outside_panels_new_articles"]["documents"]
        remainder = panels["test_subsets"]["outside_panels"]["documents"]
        self.assertEqual(
            fresh + receipt["test_rows_outside_panels_sharing_a_panel_article"],
            remainder,
        )

    def test_preserves_newly_wrong_distractors_and_neither_cases(self):
        rows = [
            {"A-coref": "TRUE", "B-coref": "FALSE"},
            {"A-coref": "FALSE", "B-coref": "TRUE"},
            {"A-coref": "FALSE", "B-coref": "FALSE"},
        ]
        result = study.paired(
            rows, ["11110", "11110", "11100"], ["11101", "11101", "11111"]
        )
        self.assertEqual(result["correct_lost"], 1)
        self.assertEqual(result["correct_gained"], 1)
        self.assertEqual(result["incorrect_added_to_distractor"], 1)
        self.assertEqual(result["incorrect_added_when_neither_gold"], 2)
        self.assertEqual(result["incorrect_removed"], 1)
        self.assertEqual(result["changed_documents"], 3)

    def test_supplied_requires_each_exact_mark_even_when_all_links_are_negative(self):
        row = {
            "ID": "example",
            "Text": "Ada told Eva she came.",
            "Pronoun": "she",
            "Pronoun-offset": "13",
            "A": "Ada",
            "A-offset": "0",
            "B": "Eva",
            "B-offset": "9",
            "A-coref": "FALSE",
            "B-coref": "FALSE",
        }
        output = {
            "id": "example",
            "producer": study.PRODUCER,
            "clusters": [[[0, 3]], [[13, 16]]],
        }
        with patch.object(study.gap, "run", return_value=[output]):
            with self.assertRaisesRegex(ValueError, "supplied spans changed"):
                study.evaluate([row], "supplied", "unused")


if __name__ == "__main__":
    unittest.main()
