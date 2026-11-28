"""Exercise WikiBELA's source-coordinate contract without loading its full index."""
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "crates/anno-cli/src/cli/commands"))
import mhel


def write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class Retrieve(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.dataset = Path(folder.name)
        self.surface = "Éva\u00a0Martin"
        self.before = "𝄞 " + "a " * 105 + " ".join(["longword"] * 45) + "\r\n"
        self.text = self.surface + " " + self.before + self.surface + "  ici"
        start = self.text.rindex(self.surface)
        self.row = dict(doc_id="doc", surface=self.surface, start_pos=str(start),
                        end_pos=str(start + len(self.surface)), type="PER")
        write_rows(self.dataset / "paragraphs_test.csv",
                   [dict(doc_id="doc", text=self.text, lang="fr")])
        write_rows(self.dataset / "annotations_test.csv",
                   [self.row, dict(self.row, doc_id="absent")])

    def run_retrieval(self):
        mhel.retrieve(self.dataset, self.dataset, self.dataset, 3, "cpu")

    def test_clipping_preserves_exact_repeated_unicode_mention(self):
        for path in ("model_wiki.ckpt", "faiss.index", "index.txt", "knowledge_base.sqlite"):
            (self.dataset / path).touch()
        module = Mock()
        model = module.load_disambiguator.return_value

        # A tokenizer with a short window forces the adapter to shorten left context.
        def transform(batch):
            return {"sp_tokens_boundaries": [[[0, min(160, len(batch["texts"][0]))]]]}

        model.transform.transform.side_effect = transform
        model.get_candidates_batch.return_value = [[{"wb_id": "Q1"}]]
        out, err = io.StringIO(), io.StringIO()
        with patch.dict(sys.modules, {"src.retriever": module}), patch.object(sys, "path", sys.path[:]), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            self.run_retrieval()
        (chunk,), ((offset,),), ((length,),) = model.get_candidates_batch.call_args.args
        self.assertGreater(model.transform.transform.call_count, 1)
        self.assertEqual(chunk[offset:], self.surface + "  ici")
        self.assertTrue(chunk[:offset].endswith("longword\r\n"))
        self.assertEqual(length, len(self.surface))
        self.assertEqual(model.get_candidates_batch.call_args.kwargs, dict(k=3, lang="fr"))
        self.assertEqual(json.loads(out.getvalue()),
                         [{**self.row, "identifier": "", "candidates": [{"wb_id": "Q1"}]}])
        self.assertIn("Skipped 1 annotation rows with missing source documents", err.getvalue())

    def test_mismatch_is_rejected_before_loading_weights(self):
        write_rows(self.dataset / "annotations_test.csv", [dict(self.row, surface="Wrong")])
        with self.assertRaisesRegex(ValueError, "source mismatch in doc"):
            self.run_retrieval()

    def test_mention_larger_than_model_window_identifies_source(self):
        transform = Mock()
        transform.transform.return_value = {"sp_tokens_boundaries": [[[0, 8]]]}
        with self.assertRaisesRegex(ValueError, "in doc at 0:10 exceeds"):
            mhel.retrieval_context(transform, {"doc_id": "doc", "text": "a" * 10}, 0, 10)

    def test_missing_sqlite_file_is_not_created(self):
        write_rows(self.dataset / "annotations_test.csv", [self.row])
        for path in ("model_wiki.ckpt", "faiss.index", "index.txt"):
            (self.dataset / path).touch()
        with self.assertRaisesRegex(FileNotFoundError, "knowledge_base.sqlite"):
            self.run_retrieval()
        self.assertFalse((self.dataset / "knowledge_base.sqlite").exists())


if __name__ == "__main__":
    unittest.main()
