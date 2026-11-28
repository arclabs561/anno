"""Regressions for JEREX raw text input, independent of a reference corpus."""

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import predict_jerex as runner


class RawDocumentTests(unittest.TestCase):
    def test_unicode_repeated_tokens_and_multiword_mentions(self):
        text = "  🐈  Zoë  Lane?!»\r\nZoë wrote “3.14”. Next."
        document = runner.text_document(text, "field-notes")
        self.assertEqual(
            document["sents"],
            [
                ["🐈", "Zoë", "Lane", "?", "!", "»"],
                ["Zoë", "wrote", "“", "3", ".", "14", "”", "."],
                ["Next", "."],
            ],
        )
        tokens = [token for sentence in document["sents"] for token in sentence]
        mentions = [{"start": 1, "end": 3}, {"start": 6, "end": 7}]
        row = {"tokens": tokens, "mentions": mentions}
        runner.attach_text_offsets(row, document)
        self.assertEqual(row["source_text"], text)
        self.assertEqual([text[i:j] for i, j in row["token_char_spans"]], tokens)
        self.assertEqual(row["sentence_token_ends"], [6, 14, 16])
        self.assertEqual(
            mentions,
            [
                {"start": 1, "end": 3, "char_start": 5, "char_end": 14},
                {"start": 6, "end": 7, "char_start": 19, "char_end": 22},
            ],
        )
        self.assertEqual(text[5:14], "Zoë  Lane")

    def test_rejects_empty_text_and_unaligned_model_output(self):
        with self.assertRaisesRegex(ValueError, "non-whitespace"):
            runner.text_document("\r\n\t", "empty")
        doc = runner.text_document("A.", "note")
        with self.assertRaisesRegex(ValueError, "tokens differ"):
            runner.attach_text_offsets({"tokens": ["a", "."]}, doc)
        with self.assertRaisesRegex(ValueError, "mention token span"):
            runner.attach_text_offsets(
                {"tokens": ["A", "."], "mentions": [{"start": 1, "end": 3}]}, doc
            )

    def test_text_cli_reads_identical_bytes_from_file_and_stdin(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            text = "A.\r\nA."
            path = tmp / "note"
            path.write_bytes(text.encode())
            for source in (str(path), "-"):
                with (
                    self.subTest(source=source),
                    patch.object(
                        sys,
                        "argv",
                        [
                            "predict_jerex.py",
                            "model",
                            source,
                            "out.json",
                            "--input-format",
                            "text",
                            "--title",
                            "note",
                        ],
                    ),
                    patch.object(
                        sys, "stdin", io.TextIOWrapper(io.BytesIO(path.read_bytes()))
                    ),
                    patch.object(runner, "predict") as predict,
                ):
                    runner.main()
                    (doc,) = predict.call_args.args[1]
                    self.assertEqual(doc["source_text"], text)
                    self.assertEqual(doc["title"], "note")
                    self.assertEqual(
                        doc["token_char_spans"], [(0, 1), (1, 2), (4, 5), (5, 6)]
                    )
                    self.assertTrue(predict.call_args.kwargs["include_text_offsets"])

    def test_tokenized_json_retains_caller_boundaries(self):
        documents = [{"title": "note", "sents": [["Dr.", "A."], ["A.M."]]}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tokenized.json"
            path.write_text(json.dumps(documents), encoding="utf-8")
            with (
                patch.object(
                    sys, "argv", ["predict_jerex.py", "model", str(path), "out.json"]
                ),
                patch.object(runner, "predict") as predict,
            ):
                runner.main()
            self.assertEqual(predict.call_args.args[1], documents)
            self.assertFalse(predict.call_args.kwargs["include_text_offsets"])


if __name__ == "__main__":
    unittest.main()
