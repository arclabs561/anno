"""Check span scoring rules used for WNUT's published predictions."""

import csv
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_wnut_novelty import (
    compare_tokens,
    parse_bio,
    shared_outcomes,
    submission_rows,
)


class WnutSubmissionReplayTest(unittest.TestCase):
    def test_initial_and_type_changing_inside_tags_open_predicted_spans(self):
        text = "first I-person\nplace I-location\n\nname I-person"
        with self.assertRaises(ValueError):
            parse_bio(text)
        output = parse_bio(text, allow_repairs=True)
        self.assertEqual(
            output.spans,
            {
                (0, 0, 1, "person"),
                (0, 1, 2, "location"),
                (1, 0, 1, "person"),
            },
        )
        self.assertEqual(output.repairs, {0: 2, 1: 1})

    def test_exposure_counts_predicted_names_on_gold_empty_sentences(self):
        reference = parse_bio("Lima B-location\nnear O\nnew B-group\n\nLima O")
        predicted = parse_bio(
            "Lima B-location\nnear B-person\nnew B-group\n\nLima B-person"
        )
        rows = submission_rows(
            "example",
            "verbatim full test",
            {0, 1},
            [0, 0],
            reference,
            predicted,
            {"lima"},
        )
        all_rows, seen, unseen = rows
        self.assertEqual(
            (all_rows["correct"], all_rows["predicted"], all_rows["gold"]), (2, 4, 2)
        )
        self.assertEqual(
            (
                seen["correct"],
                seen["predicted"],
                seen["predictions_in_gold_empty_sentences"],
            ),
            (1, 2, 1),
        )
        self.assertEqual(
            (unseen["correct"], unseen["predicted"], unseen["gold"]), (1, 2, 1)
        )

    def test_priority_across_systems_and_exclusive_token_ends(self):
        reference = parse_bio(
            "one B-person\ntwo B-group\nthree B-product\nfour I-product\nfive B-location"
        )
        predictions = {
            (0, 0, 1, "person"),
            (0, 0, 1, "corporation"),
            (0, 1, 2, "location"),
            (0, 1, 3, "corporation"),
        }
        counts = shared_outcomes(reference, predictions, set())["all"]
        self.assertEqual(
            dict(counts), dict(correct=1, wrong_type=1, partial_overlap=1, no_overlap=1)
        )

    def test_alignment_compares_sentence_boundaries_and_token_forms(self):
        self.assertEqual(
            compare_tokens([("I", "am"), ("here",)], [("I", "m"), ("here",)]), [1, 0]
        )
        with self.assertRaises(ValueError):
            compare_tokens([("I", "am"), ("here",)], [("I",), ("am", "here")])

    def test_all_published_full_test_entity_f1_scores(self):
        # Independent check against Table 2 of Derczynski et al. (WNUT 2017).
        published = {
            "Arcada": "39.98",
            "Drexel-CCI": "26.30",
            "FLYTXT": "38.35",
            "MIC-CIS": "37.06",
            "SJTU-Adapt": "40.42",
            "SpinningBytes": "40.78",
            "UH-RiTUAL": "41.86",
        }
        receipt = (
            Path(__file__).resolve().parents[2]
            / "docs/eval/wnut-submission-results.csv"
        )
        measured = {}
        with receipt.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if row["exposure"] == "all" and row["scope"] in {
                    "verbatim full test",
                    "full test by token position",
                }:
                    denominator = int(row["predicted"]) + int(row["gold"])
                    measured[row["submission"]] = (
                        f"{200 * int(row['correct']) / denominator:.2f}"
                    )
        self.assertEqual(measured, published)


if __name__ == "__main__":
    unittest.main()
