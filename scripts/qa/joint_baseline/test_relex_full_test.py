# SPDX-License-Identifier: MIT OR Apache-2.0
"""Check holdout membership, individual model choices and the published chart."""

from collections import Counter
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import relex_full_test as replay
import relex_paired_figures as paired


class FullReleaseTest(unittest.TestCase):
    def test_different_correct_relations_have_zero_sentence_level_change(self):
        rows = []
        for before, after, changed in ((1, 1, 1), (1, 0, 1), (0, 1, 1), (0, 0, 0)):
            rows.append({"score": {"added": 1, "correct_added": before}, "both_margins": {
                "added": 1, "correct_added": after, "different_from_score_only": changed,
                "correct_gained_vs_score_only": after * changed, "correct_lost_vs_score_only": before * changed,
            }})
        counts = replay.paired_review_slots(rows)
        self.assertEqual(counts["both_correct_different_relation"], 1)
        self.assertEqual(counts["confidence_only_correct"], 1)
        self.assertEqual(counts["both_margins_only_correct"], 1)
        self.assertEqual(counts["neither_correct"], 1)
        unchanged = replay.paired_review_slots(rows[:1])
        interval = paired.paired_interval([unchanged])
        self.assertEqual((interval["lower"], interval["upper"]), (0, 0))

    def test_exclusion_precedence_retains_each_source_occurrence(self):
        def row(word):
            return {"sentence": [word]}

        source = {
            ("ai", "train"): [row("train")],
            ("news", "train"): [],
            ("ai", "dev"): [row("dev")],
            ("news", "dev"): [],
            ("ai", "test"): [row("old"), row("train")],
            ("news", "test"): [row("new"), row("new"), row("train"), row("dev"), row("old")],
        }
        with patch.object(replay.study, "DOMAINS", ("ai", "news")):
            excluded, supplement = replay.outside_holdout(source, [("news", 0, True)])
        self.assertEqual(excluded["ai"]["training_text"], [1])
        self.assertEqual(excluded["ai"]["historical_ai_text"], [0])
        self.assertEqual(
            excluded["news"],
            {
                "training_text": [2],
                "historical_ai_text": [4],
                "development_text": [3],
                "repeated_test_text": [1],
            },
        )
        self.assertEqual(len(supplement), 6)

    def test_candidate_probability_agreement_must_preserve_the_actual_suggestion(self):
        h, t = (0, 1), (2, 3)
        kept = {h, t}
        schema = {"entity_types": ["person"], "relation_types": ["first", "second"]}
        before = (kept, {(h, t): [0.45, 0.4499999]}, Counter(), Counter())
        after = (kept, {(h, t): [0.4499999, 0.45]}, Counter(), Counter())
        with self.assertRaisesRegex(ValueError, "suggestion differs"):
            replay.check_reference(
                {"sentence": ["A", "B"], "relations": [], "ner": []}, schema, before, after
            )

    def test_repeated_reference_audit_preserves_relation_multiplicity(self):
        triple = [0, 0, 1, 1, "rel"]
        first = {"sentence": ["A", "B"], "relations": [triple]}
        second = {"sentence": ["A", "B"], "relations": [triple, triple]}
        counts = {("ai", i): {"stock": {"matched": 1}} for i in range(2)}
        with patch.object(replay.study, "DOMAINS", ("ai",)):
            result = replay.repeated_annotations({("ai", "test"): [first, second]}, counts)
        self.assertEqual(result["different_directed_references_groups"], 1)
        self.assertEqual(result["occurrences_with_different_directed_references"], 2)
        self.assertEqual(result["different_references_and_different_stock_match_counts_groups"], 0)

    def test_an_older_aggregate_can_show_the_observed_effect_without_inventing_pairing(self):
        root = replay.BASELINE.parent
        original = json.loads(replay.BASELINE.read_text())
        full = json.loads((root / "crossre-full-test.json").read_text())
        domains = full["panels"]["additional_test"]["domains"]
        domains["ai"].pop("paired_review_slots")
        with self.assertRaisesRegex(ValueError, "every domain"):
            paired.charts(original, full)
        for domain in domains.values():
            domain.pop("paired_review_slots", None)
        row = paired.charts(original, full)["crossre-full-test-paired.json"]["rows"][0]
        self.assertNotIn("interval", row)
        self.assertEqual((row["gained"], row["lost"]), (53, 55))

    def test_frozen_release_and_chart_reconcile(self):
        root = Path(__file__).resolve().parents[3] / "docs/eval"
        receipt = json.loads((root / "crossre-full-test.json").read_text())
        self.assertEqual(
            replay.tradeoff(receipt),
            json.loads((root / "crossre-full-test-tradeoff.json").read_text()),
        )
        frozen = json.loads(replay.BASELINE.read_text())
        for name, chart in paired.charts(frozen, receipt).items():
            self.assertEqual(chart, json.loads((root / name).read_text()))
        groups = receipt["panels"]
        transitions = groups["additional_test"]["paired_review_slots"]
        self.assertEqual(transitions["both_correct_different_relation"], 14)
        self.assertEqual(transitions["confidence_only_correct"], 41)
        self.assertEqual(transitions["both_margins_only_correct"], 39)
        fresh = groups["additional_test"]
        for domain, estimates in fresh["domain_net_match_intervals"].items():
            counts = [fresh["domains"][domain]["paired_review_slots"]]
            self.assertEqual(estimates["pointwise_95"], replay.paired_interval(counts))
            adjusted = replay.paired_interval(counts, alpha=.05 / len(fresh["domains"]))
            self.assertEqual(estimates["bonferroni_family_95"], adjusted)
            self.assertLessEqual(adjusted["lower"], 0)
            self.assertGreaterEqual(adjusted["upper"], 0)
        self.assertEqual(groups["balanced_test"]["stock"], frozen["panels"]["test"]["stock"])
        self.assertEqual(
            groups["balanced_test"]["additional_edges"],
            frozen["panels"]["test"]["additional_edges"],
        )
        self.assertEqual(
            receipt["selected_on_original_development"], frozen["selected_on_development"]
        )
        checked = receipt["checks"]
        self.assertEqual(checked["source_occurrences_with_directed_stock_graph_parity"], 2446)
        self.assertEqual(checked["balanced_test_occurrences_compared_to_trained_export"], 192)
        self.assertEqual(checked["balanced_test_predicate_probabilities_checked"], 334422)
        self.assertLessEqual(checked["maximum_absolute_predicate_probability_difference"], 0.0002)
        repeated = receipt["test_text_repetition"]
        self.assertEqual(repeated["distinct_test_texts"], 2408)
        self.assertEqual(repeated["different_directed_references_groups"], 6)
        self.assertEqual(repeated["occurrences_with_different_directed_references"], 12)
        for domain, comparison in checked["native_full_release_comparison"].items():
            stock = groups["full_official_test"]["domains"][domain]["stock"]
            self.assertEqual(
                comparison["public_pytorch_minus_native_onnx"],
                {k: stock[k] - v for k, v in comparison["native_onnx"].items()},
            )
        for name, panel in groups.items():
            by_domain = sum((Counter(d["paired_review_slots"]) for d in panel["domains"].values()), Counter())
            self.assertEqual(Counter(panel["paired_review_slots"]), by_domain)
            support = panel["confidence_by_stock_pair"]
            score = panel["additional_edges"]["score"]
            self.assertEqual(set(support), {"stock_pair_absent", "stock_pair_present"})
            self.assertEqual(sum(part["suggestions"] for part in support.values()), score["added"])
            self.assertEqual(sum(part["correct"] for part in support.values()), score["correct_added"])
            totals = panel["stock"]
            self.assertEqual(totals["matched"] + totals["unmatched"], totals["predicted"])
            self.assertEqual(
                totals["gold"] - totals["matched"], sum(panel["misses_by_first_loss"].values())
            )
            self.assertEqual(
                Counter(totals),
                sum((Counter(d["stock"]) for d in panel["domains"].values()), Counter()),
            )
            for arm, counts in panel["additional_edges"].items():
                self.assertEqual(
                    counts["added"], counts["correct_added"] + counts["unmatched_added"]
                )
                if arm in replay.study.RULES:
                    self.assertEqual(counts["added"], panel["additional_edges"]["score"]["added"])
                    self.assertLessEqual(
                        counts["correct_added"],
                        panel["suggestion_opportunity"]["one_per_sentence_upper_bound"],
                    )
        for whole, left, right in (
            ("all_eligible_test", "balanced_test", "additional_test"),
            ("full_official_test", "all_eligible_test", "outside_content_holdout"),
        ):
            for doc_key in (
                "stock",
                "misses_by_first_loss",
                "suggestion_opportunity",
                "reference_structure",
                "stronger_label_confusions",
                "retained_low_scored_keys",
                "suppressed_low_scored_keys",
            ):
                self.assertEqual(
                    Counter(groups[whole][doc_key]),
                    Counter(groups[left][doc_key]) + Counter(groups[right][doc_key]),
                )
            for arm, counts in groups[whole]["additional_edges"].items():
                self.assertEqual(
                    Counter(counts),
                    Counter(groups[left]["additional_edges"][arm])
                    + Counter(groups[right]["additional_edges"][arm]),
                )
        for dom, excluded in receipt["excluded_from_content_holdout_indices"].items():
            indices = [i for values in excluded.values() for i in values]
            self.assertEqual(len(indices), len(set(indices)))
            all_n = groups["full_official_test"]["domains"][dom]["stock"]["sentences"]
            self.assertTrue(all(0 <= i < all_n for i in indices))
            self.assertEqual(
                len(indices),
                groups["outside_content_holdout"]["domains"][dom]["stock"]["sentences"],
            )


if __name__ == "__main__":
    unittest.main()
