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


class NativeDocuments(unittest.TestCase):
    def setUp(self):
        self.doc = {"id": 7, "language": "fr", "text": "📰\r\nÉva court.", "entities": [
            {"id": eid, "type": "PER", "mentions": [{"start": 3, "end": 6, "text": "Éva"}]}
            for eid in [3, "3"]]}
        self.native = mhel.SourceDocuments([self.doc, {**self.doc, "id": "7", "entities": []}])
        self.choices = [{"wb_id": "Q42", "label": "Éva"}]
        self.entries = [self.native.candidate_output(row, self.choices)
                        for row in self.native.occurrences.values()]

    def test_same_range_different_ids_roundtrips_through_upstream_csv(self):
        def predict(args, **kwargs):
            dataset = Path(args[args.index("--dataset_path") + 1])
            paragraphs = mhel.rows(dataset / "paragraphs_test.csv")
            self.assertEqual(len(paragraphs), 2)
            self.assertEqual(paragraphs[0]["text"][3:6], "Éva")
            self.assertEqual(paragraphs[0]["publication_date"], "")
            payload = json.loads(Path(args[args.index("--json_f") + 1]).read_text())
            self.assertTrue(all(row["identifier"] == "" for row in payload))
            self.assertTrue(all(row["candidates"][0]["score"] is None for row in payload))
            write_rows(dataset / "output.csv", [
                dict(doc_id=row["doc_id"], start_pos=row["start_pos"], end_pos=row["end_pos"],
                     surface=row["surface"], type=row["type"], identifier=qid)
                for row, qid in zip(payload, ["Q42", "NIL"])])

        with tempfile.TemporaryDirectory() as tmp, patch.object(mhel.subprocess, "run", side_effect=predict):
            path = Path(tmp) / "caller-candidates.json"
            path.write_text(json.dumps(self.entries[::-1]))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                mhel.rerank(Path(tmp), Path(tmp), None, path, 5, 0., self.native)
        results = [json.loads(line) for line in out.getvalue().splitlines()]
        self.assertEqual([(r["doc_id"], r["entity_id"], r["wikidata_id"]) for r in results],
                         [(7, "3", "Q42"), (7, 3, None)])
        self.assertTrue(all(r["text"] == self.doc["text"][r["start"]:r["end"]] for r in results))

    def test_native_retrieval_preserves_identity_language_and_exact_context(self):
        module = Mock()
        model = module.load_disambiguator.return_value
        model.transform.transform.return_value = {"sp_tokens_boundaries": [[[0, len(self.doc["text"])]]]}
        model.get_candidates_batch.return_value = [self.choices]
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {"src.retriever": module}), \
                patch.object(sys, "path", sys.path[:]), contextlib.redirect_stdout(io.StringIO()) as out:
            folder = Path(tmp)
            for asset in ("model_wiki.ckpt", "faiss.index", "index.txt", "knowledge_base.sqlite"):
                (folder / asset).touch()
            mhel.retrieve(folder, folder, None, 4, "cpu", self.native)
        self.assertEqual(json.loads(out.getvalue()), self.entries)
        args, kwargs = model.get_candidates_batch.call_args
        self.assertEqual(kwargs, {"k": 4, "lang": "fr"})
        self.assertEqual(args[0][0][args[1][0][0]:][:args[2][0][0]], "Éva")

    def test_invalid_spans_candidates_and_language(self):
        for entries, error in ((self.entries[:1], "missing candidates"),
                               (self.entries + self.entries, "duplicate candidate"),
                               ([{**e, "text": "wrong"} for e in self.entries], "differs from source")):
            with self.assertRaisesRegex(ValueError, error):
                self.native.candidate_rows(entries, 0.)
        with self.assertRaisesRegex(ValueError, "scored candidates"):
            self.native.candidate_rows(self.entries, 1.)
        with self.assertRaisesRegex(ValueError, "supported language"):
            mhel.SourceDocuments([{**self.doc, "language": "xx"}])
        self.doc["entities"][0]["mentions"][0]["end"] = 5
        with self.assertRaisesRegex(ValueError, "mismatched mention"):
            mhel.SourceDocuments([self.doc])


if __name__ == "__main__":
    unittest.main()
