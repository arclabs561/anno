"""Reconcile the published native error chart with the separate routing study."""

from collections import Counter
import json
from pathlib import Path
import unittest

from wikigold_entity_errors import STOCK_SPANS_SHA256

EVAL = Path(__file__).resolve().parents[2] / "docs" / "eval"


class WikiGoldEntityErrorsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((EVAL / "wikigold-entity-errors.json").read_text())
        cls.routing = json.loads((EVAL / "wikigold-nuner-routing.json").read_text())

    def test_source_and_model_match_independent_full_release_measurement(self):
        audit = self.report["metadata"]
        source = self.routing["source"]
        self.assertEqual(self.report["schema"], "anno.errors/v1")
        self.assertEqual(self.report["metric"], "entity")
        self.assertEqual(audit["source"], source)
        self.assertEqual(self.report["reports"], source["sentence_occurrences"])
        self.assertEqual(
            audit["stock"]["checkpoint_sha256"], self.routing["checkpoint_sha256"]
        )
        self.assertEqual(
            audit["stock"]["ordered_final_typed_spans_sha256"], STOCK_SPANS_SHA256
        )

        totals = self.report["entities"]
        stock = self.routing["arms"]["without_bert"]["full_split"]
        self.assertEqual(totals["gold_count"], stock["gold"])
        self.assertEqual(totals["exact"], stock["stock_exact"])
        self.assertEqual(
            audit["predicted_outside_gold_inventory"], stock["stock_outside_schema"]
        )
        self.assertEqual(
            audit["predicted_in_gold_inventory"],
            stock["stock_exact"] + stock["stock_in_schema_unmatched"],
        )
        self.assertEqual(
            totals["predicted_count"],
            audit["predicted_in_gold_inventory"]
            + audit["predicted_outside_gold_inventory"],
        )

    def test_correctness_breakdown_and_confusions_reconcile_by_label(self):
        audit = self.report["metadata"]
        totals = self.report["entities"]
        gold, pred = audit["by_gold_label"], audit["by_predicted_label"]
        self.assertEqual(
            {label: sum(counts.values()) for label, counts in gold.items()},
            self.routing["gold_by_source_label"]["full_split"],
        )
        for by_label in (gold, pred):
            self.assertEqual(
                sum(c.get("correct", 0) for c in by_label.values()), totals["exact"]
            )
            self.assertEqual(
                sum(c.get("boundary_error", 0) for c in by_label.values()),
                totals["partial_overlap"],
            )
        self.assertEqual(
            sum(c.get("false_negative", 0) for c in gold.values()),
            totals["missed_gold"],
        )
        self.assertEqual(
            sum(c.get("false_positive", 0) for c in pred.values()),
            totals["extra_predictions"],
        )

        confused_gold, confused_pred = Counter(), Counter()
        for pair in self.report["same_span_label_confusions"]:
            self.assertNotEqual(pair["gold"], pair["predicted"])
            confused_gold[pair["gold"]] += pair["pairs"]
            confused_pred[pair["predicted"]] += pair["pairs"]
        self.assertEqual(
            confused_gold,
            Counter(
                {
                    label: counts.get("type_mismatch", 0)
                    for label, counts in gold.items()
                }
            ),
        )
        self.assertEqual(
            confused_pred,
            Counter(
                {
                    label: counts.get("type_mismatch", 0)
                    for label, counts in pred.items()
                }
            ),
        )
        self.assertEqual(sum(confused_pred.values()), totals["same_span_wrong_label"])
        native_labels = self.report["label_outcomes"]
        for label, counts in gold.items():
            self.assertEqual(
                native_labels["by_gold_label"][label],
                {
                    "exact": counts.get("correct", 0),
                    "same_span_wrong_label": counts.get("type_mismatch", 0),
                    "partial_overlap": counts.get("boundary_error", 0),
                    "missed": counts.get("false_negative", 0),
                },
            )
        partial = audit["partial_overlap_labels_from_aligned_signals"]
        self.assertEqual(
            native_labels["partial_labels"],
            {"same": partial["same_label"], "different": partial["different_labels"]},
        )
        self.assertEqual(
            partial["different_labels"] + partial["same_label"],
            totals["partial_overlap"],
        )


if __name__ == "__main__":
    unittest.main()
