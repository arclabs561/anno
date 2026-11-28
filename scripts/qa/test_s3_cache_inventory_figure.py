"""Verify the units used in the live cache inventory figure."""

import importlib.util
from pathlib import Path
import sys
import unittest


@unittest.skipUnless(
    importlib.util.find_spec("matplotlib"), "figure requires Matplotlib"
)
class CacheInventoryFigureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        try:
            import plot_s3_cache_inventory
            import plot_s3_source_evidence
        finally:
            sys.path.pop(0)
        cls.renderer = plot_s3_cache_inventory
        cls.comparison = plot_s3_source_evidence

    def test_payload_objects_can_share_dataset_ids_and_policies(self):
        report = {
            "objects": 4,
            "unrecognized_keys": 1,
            "registered_legacy_payloads": 1,
            "registered_snapshots": 1,
            "registered_manifests": 0,
            "registered_pointers": 1,
            "registry_ids_with_payload": 1,
            "payloads_excluded_from_shared_reads": 2,
            "payloads_requiring_local_source": 2,
            "payloads_approved_for_shared_publication": 0,
            "payloads_with_digest_matched_rights_evidence": 0,
        }
        self.assertEqual(self.renderer.check_counts(report), 2)
        figure = self.renderer.draw(report)
        try:
            captions = [text.get_text() for text in figure.texts]
            self.assertTrue(any(text.startswith("0 WikiGold") for text in captions))
        finally:
            self.renderer.plt.close(figure)
        for invalid in (
            dict(report, objects=3),
            dict(report, registry_ids_with_payload=3),
            dict(report, payloads_requiring_local_source=3),
            dict(report, payloads_with_digest_matched_rights_evidence=2),
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.renderer.check_counts(invalid)

    def test_publisher_match_lanes_use_the_current_objects_reads_and_ids(self):
        receipt = {
            "objects_listed": 10,
            "objects_by_kind": {"legacy": 3, "snapshot": 3, "unrecognized": 4},
            "registry_ids_with_payload": 3,
            "registry_ids_with_own_publisher_match": 1,
            "payload_objects_by_outcome": {
                "matches_own_publisher_file": 2,
                "matches_other_registry_file_only": 1,
                "no_matching_publisher_digest": 1,
                "access_denied": 1,
                "shared_read_exclusion": 1,
            },
            "listing_started_utc": "2026-11-28",
        }
        totals, _ = self.comparison.summary_counts(receipt)
        self.assertEqual(
            totals, {"payload": 6, "read": 4, "own": 2, "skipped": 1, "failed": 1}
        )
        figure = self.comparison.draw(receipt)
        try:
            captions = " ".join(text.get_text() for text in figure.texts)
            self.assertIn(
                "6 recognized   →   4 read   →   2 own-publisher matches", captions
            )
            self.assertIn(
                "3 with a payload   →   1 with an own-publisher match", captions
            )
            self.assertIn("4 unrecognized", captions)
        finally:
            self.comparison.plt.close(figure)
        with self.assertRaisesRegex(ValueError, "known outcome"):
            self.comparison.summary_counts(
                dict(receipt, payload_objects_by_outcome={"unknown": 6})
            )
        with self.assertRaisesRegex(ValueError, "Inconsistent"):
            self.comparison.summary_counts(
                dict(receipt, registry_ids_with_own_publisher_match=3)
            )


if __name__ == "__main__":
    unittest.main()
