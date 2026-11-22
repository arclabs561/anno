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
