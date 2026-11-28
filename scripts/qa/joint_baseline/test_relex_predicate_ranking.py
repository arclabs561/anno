# SPDX-License-Identifier: MIT OR Apache-2.0
from collections import Counter
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from relex_predicate_ranking import RULES, cli_gate_report, compare_scores, suggestions, tradeoff
from relex_roles import measure


class PredicateRankingTests(unittest.TestCase):
    def test_cli_gate_replay_uses_the_original_panel_denominators_and_fingerprints(self):
        eval_dir = Path(__file__).resolve().parents[3] / "docs/eval"
        original = json.loads((eval_dir / "crossre-predicate-ranking.json").read_text())
        frozen = json.loads((eval_dir / "crossre-cli-predicate-gates.json").read_text())
        counted = {"dev": Counter(sentences=96, gold=397, matched=121, predicted=371),
                   "test": Counter(sentences=192, gold=646, matched=200, predicted=628)}
        self.assertEqual(cli_gate_report(original, counted), frozen)
        counted["test"]["sentences"] -= 1
        with self.assertRaisesRegex(ValueError, "same reference occurrences"):
            cli_gate_report(original, counted)

    def test_existing_threshold_trial_uses_the_same_scored_pairs(self):
        a, b = (0, 1), (2, 3)
        spans = {a, b}
        scores = {(a, b): [0.45], (b, a): [0.55]}
        stock = Counter({(b, "r", a): 1})
        doc = {"sentence": ["李", "A"], "relations": [[0, 0, 1, 1, "r"]]}
        with patch("relex_roles.score_document", return_value=(spans, spans, spans, scores, stock)):
            totals, losses, alternate, _ = measure(
                None, None, doc, {"relation_types": ["r"]}, 0.5, 0.4
            )
        self.assertEqual(totals, {"gold": 1, "matched": 0, "predicted": 1, "directed_pairs": 2})
        self.assertEqual(losses, {"predicate_below_threshold": 1})
        self.assertEqual(alternate["added_matched"], 1)
        self.assertEqual(alternate["predicted"], 2)

    def test_checked_in_counts_regenerate_the_tradeoff_and_conserve_reference_decisions(self):
        eval_dir = Path(__file__).resolve().parents[3] / "docs/eval"
        receipt = json.loads((eval_dir / "crossre-predicate-ranking.json").read_text())
        self.assertEqual(
            tradeoff(receipt),
            json.loads((eval_dir / "crossre-predicate-tradeoff.json").read_text()),
        )
        dev = receipt["panels"]["dev"]["additional_edges"]
        self.assertEqual(
            receipt["selected_on_development"],
            max(RULES, key=lambda name: dev[name]["correct_added"]),
        )
        self.assertEqual(len(receipt["source_sha256"]), 18)
        self.assertEqual(receipt["stock_parity"]["checked_sentences"], 320)
        for split, n in (("historical", 32), ("dev", 96), ("test", 192)):
            panel = receipt["panels"][split]
            stock = panel["stock"]
            self.assertEqual(stock["sentences"], n)
            self.assertEqual(
                stock["matched"] + sum(panel["misses_by_first_loss"].values()), stock["gold"]
            )
            self.assertEqual(stock["matched"] + stock["unmatched"], stock["predicted"])
            self.assertEqual(sum(d["selected"] for d in receipt["selection"][split].values()), n)
            for stage in panel["low_predicate_keys"].values():
                self.assertEqual(
                    stage["unique_keys"],
                    sum(
                        stage.get(k, 0)
                        for k in ("literal_first", "literal_tied_first", "literal_below_first")
                    ),
                )
                self.assertEqual(
                    stage.get("literal_below_first", 0),
                    stage["unannotated_label_higher"] + stage.get("only_annotated_labels_higher", 0),
                )
                self.assertEqual(
                    stage["unique_keys"],
                    sum(
                        stage.get(k, 0)
                        for k in (
                            "reverse_unavailable",
                            "reverse_also_correct",
                            "unannotated_reverse_higher",
                            "unannotated_reverse_tied",
                            "above_unannotated_reverse",
                        )
                    ),
                )
                self.assertGreaterEqual(stage["reference_occurrences"], stage["unique_keys"])
            control = panel["additional_edges"]["score"]
            opportunity = panel["suggestion_opportunity"]
            self.assertEqual(opportunity["sentences_with_an_eligible_edge"], control["added"])
            self.assertLessEqual(
                opportunity["one_per_sentence_upper_bound"], opportunity["eligible_gold_keys"]
            )
            self.assertLessEqual(
                opportunity["eligible_gold_keys"],
                panel["low_predicate_keys"]["retained"]["unique_keys"],
            )
            for name, candidate in panel["additional_edges"].items():
                self.assertEqual(
                    candidate["correct_added"] + candidate["unmatched_added"], candidate["added"]
                )
                self.assertEqual(
                    sum(
                        v
                        for k, v in candidate.items()
                        if k.startswith("unmatched_") and k != "unmatched_added"
                    ),
                    candidate["unmatched_added"],
                )
                if name in RULES:
                    self.assertEqual(candidate["added"], control["added"])
                    self.assertLessEqual(
                        candidate["correct_added"], opportunity["one_per_sentence_upper_bound"]
                    )
                    self.assertEqual(
                        candidate["correct_added"] - control["correct_added"],
                        candidate["correct_gained_vs_score_only"]
                        - candidate["correct_lost_vs_score_only"],
                    )

    def test_multiple_valid_labels_reverse_annotations_ties_and_missing_scores(self):
        a, b, c = (0, 1), (2, 3), (4, 5)
        labels = ["r", "q", "x"]
        ref = Counter(
            {
                (a, "r", b): 2,
                (a, "q", b): 1,
                (b, "r", a): 1,
                (a, "r", c): 1,
                (c, "missing", b): 1,
            }
        )
        scores = {
            (a, b): [0.35, 0.40, 0.10],
            (b, a): [0.45, 0.42, 0.60],
            (a, c): [0.50, 0.20, 0.50],
        }
        by_stage, confusions = compare_scores(ref, {a, b}, scores, labels)
        retained, suppressed = by_stage["retained"], by_stage["suppressed"]
        self.assertEqual((retained["unique_keys"], retained["reference_occurrences"]), (3, 4))
        self.assertEqual(
            (retained["in_suggestion_band_keys"], retained["in_suggestion_band_occurrences"]),
            (3, 4),
        )
        self.assertEqual(retained["on_multilabel_directed_pair"], 2)
        self.assertEqual(retained["only_annotated_labels_higher"], 1)
        self.assertEqual(retained["literal_first"], 1)
        self.assertEqual(retained["above_all_unannotated_labels"], 2)
        self.assertEqual(retained["reverse_also_correct"], 2)
        self.assertEqual(retained["unannotated_reverse_higher"], 1)
        self.assertEqual(retained["unannotated_label_higher"], 1)
        self.assertEqual(retained["at_least_one_unannotated_rival_higher"], 2)
        self.assertEqual(confusions, {"r → x": 1})
        self.assertEqual(suppressed["literal_tied_first"], 1)
        self.assertEqual(suppressed["unannotated_label_tied"], 1)
        self.assertEqual(suppressed["reverse_unavailable"], 1)

    def test_candidates_share_budget_bounds_and_retained_endpoints(self):
        a, b, c, d = (0, 1), (2, 3), (4, 5), (6, 7)
        scores = {
            (a, b): [0.44, 0.48, 0.55],
            (b, a): [0.49, 0.48, 0.05],
            (a, c): [0.42, 0.10, 0.06],
            (c, a): [0.01, 0.08, 0.09],
            (a, d): [0.5, 0.02, 0.02],
        }
        ranked = suggestions(scores, {a, b, c}, ["r", "q", "x"])
        self.assertEqual(ranked["score"], {(b, "r", a)})
        for rule in ("label_margin", "direction_margin", "both_margins"):
            self.assertEqual(ranked[rule], {(a, "r", c)})
        self.assertEqual(len(ranked["gate_0.40"]), 5)
        edges = suggestions({(a, b): [0.30, 0.40, 0.50]}, {a, b}, ["r", "q", "x"])
        self.assertEqual(edges["gate_0.40"], {(a, "x", b)})
        self.assertEqual(edges["score"], {(a, "x", b)})
        self.assertTrue(all(not v for v in suggestions({(a, b): [0.3]}, {a, b}, ["r"]).values()))


if __name__ == "__main__":
    unittest.main()
