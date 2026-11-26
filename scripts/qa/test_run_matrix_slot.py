import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from run_matrix_slot import choose_strategy, run_slot


class SlotSchedule(unittest.TestCase):
    def test_one_coverage_slot_per_budget_block(self):
        expected = ["worst-first", "estimate", "worst-first", "worst-first"]
        actual = [choose_strategy("mixed", slot, 4, 2) for slot in range(1, 13)]
        self.assertEqual(actual, expected * 3)
        self.assertEqual(choose_strategy("mixed", 9, 1, 1), "estimate")

    def test_control_never_uses_targeted_evidence(self):
        for slot in range(1, 9):
            self.assertEqual(choose_strategy("random", slot, 4, 2), "random")

    def test_invalid_slot_is_not_silently_assigned(self):
        for slot, period, reserve in [(0, 4, 2), (1, 0, 1), (1, 4, 0), (1, 4, 5)]:
            with self.assertRaises(ValueError):
                choose_strategy("mixed", slot, period, reserve)


    def test_unmeasured_run_keeps_schedule_and_rejects_success_claim(self):
        with tempfile.TemporaryDirectory() as folder, patch("run_matrix_slot.subprocess.run"):
            root = Path(folder)
            receipts = root / "receipts"
            with self.assertRaisesRegex(ValueError, "no outcome"):
                run_slot(root / "fake", root / "cache", receipts, "mixed", 2, 4, 2, 9)
            self.assertEqual(json.loads((receipts / "schedule.json").read_text())["requested_strategy"], "estimate")
            self.assertFalse((receipts / "result.json").exists())
            with self.assertRaises(FileExistsError):
                run_slot(root / "fake", root / "cache", receipts, "mixed", 2, 4, 2, 9)

    def test_unexpected_native_strategy_invalidates_plans_and_measurements(self):
        for plan_only in [False, True]:
            with self.subTest(plan_only=plan_only), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                receipts = root / "receipts"
                def native(*args, **kwargs):
                    if plan_only:
                        (receipts / "selection.json").write_text(json.dumps({"strategy":"random"}))
                    else:
                        (receipts / "decisions.jsonl").write_text(json.dumps({
                            "record_type":"outcome", "strategy":"random"
                        }) + "\n")
                with patch("run_matrix_slot.subprocess.run", native):
                    with self.assertRaisesRegex(ValueError, "expected estimate"):
                        run_slot(root / "fake", root / "cache", receipts, "mixed", 2, 4, 2, 9, plan_only)
                summary = json.loads((receipts / "result.json").read_text())
                self.assertFalse(summary["strategy_matches"])
                self.assertEqual(summary["observed_strategies"], ["random"])

    def test_rotate_verified_cells_and_refuse_unmeasured_languages(self):
        cells = [("ner", "en"), ("ner", "de")]
        for slot, lang, verified in [(2, "en", True), (6, "de", True), (10, "en", False)]:
            with self.subTest(slot=slot), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                receipts = root / "receipts"
                def native(*args, env, **kwargs):
                    self.assertEqual((env["ANNO_MATRIX_TASK"], env["ANNO_MUXER_PIN_LANG"]), ("ner", lang))
                    row = dict(record_type="outcome", strategy="estimate", task="ner",
                               requested_languages=[lang], hard_junk=False, cost_units=1,
                               observed_language=lang if verified else None)
                    (receipts / "decisions.jsonl").write_text(json.dumps(row))
                with patch("run_matrix_slot.subprocess.run", native):
                    def run():
                        run_slot(root / "fake", root / "cache", receipts, "mixed", slot, 4, 2,
                                 seed=3, coverage_cells=cells)
                    if verified:
                        run()
                    else:
                        with self.assertRaisesRegex(ValueError, "no completed evaluation"):
                            run()
                summary = json.loads((receipts / "result.json").read_text())
                self.assertEqual(summary["target_evaluations"], int(verified))
