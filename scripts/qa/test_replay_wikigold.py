"""Exercise the pinned-source replay without embedding the upstream corpus."""
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import replay_wikigold as adapter


class WikiGoldReplay(unittest.TestCase):
    def test_bio_continuations_and_scalar_offsets(self):
        text, spans = adapter.annotated_sentence("🧭 O\nZé I-PER\nQuill I-PER\nVale I-LOC\nOak B-LOC")
        self.assertEqual(text, "🧭 Zé Quill Vale Oak")
        self.assertEqual(spans, [(2, 10, "PER"), (11, 15, "LOC"), (16, 19, "LOC")])

    def test_family_for_case_without_a_dataset_cache(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(adapter.family_for_case(Path.cwd(), "sentence-0"), adapter.FAMILY)
            for case in ("sentence-01", "sentence-10000", "../0", None):
                with self.subTest(case=case), self.assertRaises(ValueError):
                    adapter.family_for_case(Path.cwd(), case)

    def test_pin_and_sentence_number(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.dict(os.environ, ANNO_CACHE_DIR=temp):
            path = Path(temp) / "datasets/WikiGold.cache"
            path.parent.mkdir()
            data = b"X I-LOC\n\n-DOCSTART- O\n\nY I-ORG\n"
            path.write_bytes(data)
            with self.assertRaisesRegex(ValueError, "pinned corpus"):
                adapter.sentence("sentence-0")
            with mock.patch.object(adapter, "WIKIGOLD", hashlib.sha256(data).hexdigest()):
                self.assertEqual(adapter.sentence("sentence-1"), ("Y", [(0, 1, "ORG")]))
                for case in ("sentence-2", "sentence-01", "../0"):
                    with self.subTest(case=case), self.assertRaises(ValueError):
                        adapter.sentence(case)

    def test_wrong_types_and_clipped_ranges_fail(self):
        fixture = ("Zé Quill", [(0, 8, "PER")])
        row = {"id": "sentence-0", "source_sha256": adapter.judge.source_sha256(fixture[0]),
               "model": "stacked", "entities": []}
        with (mock.patch.object(adapter, "sentence", return_value=fixture),
              mock.patch.object(adapter.judge.shutil, "which", return_value=__file__),
              mock.patch.object(adapter.judge, "run_batch", return_value=[row])):
            failures = set()
            for entity, outcome in [
                (dict(start=0, end=8, text="Zé Quill", type="PER"), "pass"),
                (dict(start=0, end=8, text="Zé Quill", type="ORG"), "fail"),
                (dict(start=0, end=2, text="Zé", type="PER"), "fail"),
            ]:
                row["entities"] = [entity]
                result = adapter.replay(Path.cwd(), "sentence-0")
                self.assertEqual(result["outcome"], outcome)
                self.assertEqual(set(result), {"outcome", "signature", "producer_sha256"})
                if outcome == "fail":
                    self.assertRegex(result["signature"], r"^sha256:[0-9a-f]{64}$")
                    self.assertEqual(result, adapter.replay(Path.cwd(), "sentence-0"))
                    failures.add(result["signature"])
                else:
                    self.assertEqual(result["signature"], "")
            self.assertEqual(len(failures), 2)

            row["entities"][0]["text"] = "different"
            with self.assertRaisesRegex(ValueError, "does not select"):
                adapter.replay(Path.cwd(), "sentence-0")


if __name__ == "__main__":
    unittest.main()
