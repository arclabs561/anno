"""Contract checks for the paired stack-routing replay."""

from collections import Counter
from contextlib import redirect_stderr
from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import nuner_fallback_routing as study


class RoutingReplayTest(unittest.TestCase):
    def test_recorded_experiment_reconciles_with_inputs_and_patches(self):
        root = Path(__file__).resolve().parents[2]
        receipt = json.loads(
            (root / "docs/eval/nuner-fallback-routing.json").read_text()
        )
        for name, expected in receipt["patch_sha256"].items():
            self.assertEqual(
                study.file_sha(root / "scripts/qa/fixtures" / name), expected
            )
        self.assertEqual(receipt["checkpoint_sha256"], study.FILES)
        self.assertEqual(receipt["bert_sha256"], study.BERT_FILES)
        for split, n, gold in (("Development", 1009, 836), ("Test", 1287, 1079)):
            source_labels = receipt["gold_by_source_label"][split]
            for topology, arm in receipt["splits"][split].items():
                self.assertEqual(arm["stock_skipped"] + arm["stock_routed"], n)
                self.assertEqual(arm["all_sentences"], n)
                self.assertEqual(arm["full_split"]["gold"], gold)
                self.assertEqual(sum(source_labels["full_split"].values()), gold)
                skipped = arm["skipped_subset"]
                self.assertEqual(
                    sum(source_labels["stock_skipped"].values()), skipped["gold"]
                )
                self.assertEqual(arm["routed_control_sentences"], 8)
                self.assertEqual(
                    arm["guarded_healthy_bert_controls"],
                    16 if topology == "with_bert" else 0,
                )
                for counts in (arm["full_split"], skipped):
                    self.assertEqual(
                        counts["candidate_exact"] - counts["stock_exact"],
                        skipped["exact_gained"] - skipped["exact_lost"],
                    )
                    self.assertEqual(
                        counts["candidate_in_schema_unmatched"]
                        - counts["stock_in_schema_unmatched"],
                        skipped["in_schema_unmatched_added"]
                        - skipped["in_schema_unmatched_removed"],
                    )
                self.assertEqual(
                    sum(arm["added_unmatched_by_overlap"].values()),
                    skipped["in_schema_unmatched_added"],
                )

    def test_paired_regressions_and_unmatched_label_scope(self):
        counts = Counter()
        study.add_case(
            counts,
            "Ada Acme Rome",
            {(0, 3, "person"), (4, 8, "corporation")},
            [(0, 3, "PER"), (4, 8, "PER"), (9, 13, "URL")],
            [(4, 8, "ORG"), (9, 13, "LOC")],
            {"ada"},
        )
        self.assertEqual(counts["exact_gained"], 1)
        self.assertEqual(counts["exact_lost"], 1)
        self.assertEqual(counts["unfamiliar_gained"], 1)
        self.assertEqual(counts["unfamiliar_lost"], 0)
        self.assertEqual(counts["in_schema_unmatched_added"], 1)
        self.assertEqual(counts["in_schema_unmatched_removed"], 1)
        self.assertEqual(counts["stock_outside_schema"], 1)

    def test_full_totals_keep_routed_controls_and_refuse_a_changed_control(self):
        cases = [("Ada", {(0, 3, "person")}), ("Bea", {(0, 3, "person")})]
        gates = [{"ran": False}, {"ran": True}]
        hit = {"ran": True, "spans": [[0, 3, "PER"]], "seconds": 0.1}
        miss = {"ran": False, "spans": [], "seconds": 0.001}
        args = SimpleNamespace(stock="stock", candidate="guard", checkpoint="weights")
        with redirect_stderr(StringIO()):
            with patch.object(
                study, "run_worker", side_effect=[[miss, hit], [hit, hit]]
            ):
                result = study.measure("development", cases, set(), args, gates, None)
            with patch.object(
                study,
                "run_worker",
                side_effect=[[miss, hit], [hit, {**hit, "spans": []}]],
            ):
                with self.assertRaisesRegex(ValueError, "already-routed control"):
                    study.measure("development", cases, set(), args, gates, None)
        self.assertEqual(result["stock_skipped"], 1)
        self.assertEqual(result["skipped_subset"]["gold"], 1)
        self.assertEqual(result["full_split"]["gold"], 2)
        self.assertEqual(result["full_split"]["candidate_exact"], 2)
        self.assertEqual(result["full_split"]["stock_exact"], 1)


if __name__ == "__main__":
    unittest.main()
