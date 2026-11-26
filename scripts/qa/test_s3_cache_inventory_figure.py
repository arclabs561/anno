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
        finally:
            sys.path.pop(0)
        cls.renderer = plot_s3_cache_inventory

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


if __name__ == "__main__":
    unittest.main()
