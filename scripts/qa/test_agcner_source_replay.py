"""Verify the independent character and gold-span checks used in the AgCNER study."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import agcner_source_replay as replay


class AgcnerReplayTest(unittest.TestCase):
    raw = "甲 B-G\r\n𠀀 I-G\r\n \tO\r\n乙 B-L\r\n".encode()

    def report(self, text, first_end, last_start):
        def span_digest(label, start, end):
            value = json.dumps(
                [label, [{"start": start, "end": end}]], separators=(",", ":")
            )
            return hashlib.sha256(value.encode()).hexdigest()

        return {
            "model": "heuristic",
            "gold_count": 2,
            "predicted_count": 0,
            "correct": 0,
            "pairing": {
                "document_id": "AgCNER:1",
                "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "gold_keys": {
                    "0": span_digest("G", 0, first_end),
                    "1": span_digest("L", last_start, last_start + 1),
                },
            },
        }

    def test_original_and_legacy_offsets_use_unicode_characters_and_literal_space(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cases.jsonl"
            path.write_text(json.dumps(self.report("甲𠀀 乙", 2, 3)) + "\n")
            exact = replay.check_reports(self.raw, path, legacy=False)
            self.assertEqual(exact["gold_spans_exact_source_label_and_offsets"], 2)
            self.assertEqual(
                (exact["whitespace_tokens"], exact["non_bmp_tokens"]), (1, 1)
            )
            path.write_text(json.dumps(self.report("甲 𠀀   乙", 3, 6)) + "\n")
            old = replay.check_reports(self.raw, path, legacy=True)
            self.assertEqual(
                (old["artificial_separators"], old["source_text_exact"]), (3, 0)
            )
            with self.assertRaisesRegex(ValueError, "span fingerprint differs"):
                replay.check_reports(self.raw, path, legacy=False)

    def test_independent_reader_rejects_invalid_token_and_orphan_label(self):
        with self.assertRaisesRegex(ValueError, "line 2"):
            list(replay.blocks("甲 O\n两个 O\n".encode()))
        (tokens,) = replay.blocks("甲 B-G\n乙 O\n丙 I-G\n".encode())
        with self.assertRaisesRegex(ValueError, "Orphan"):
            replay.source_case(tokens, "")


if __name__ == "__main__":
    unittest.main()
