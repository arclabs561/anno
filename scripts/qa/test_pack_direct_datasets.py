import io
import json
from pathlib import Path
import sys
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pack_direct_datasets as pack


class DirectDatasetPackTest(unittest.TestCase):
    def test_quoref_preserves_and_counts_misplaced_answers(self):
        answer = {"text": "Kate", "answer_start": 0}
        question = {"question": "Who?", "id": "q1", "answers": [answer, answer]}
        paragraph = {"context": "Meet Kate.", "qas": [question]}
        source = {"data": [{"paragraphs": [paragraph]}]}

        result = pack.measure("quoref", json.dumps(source).encode())
        self.assertEqual(result["source_offset_mismatches"], 2)
        self.assertEqual(result["questions_with_offset_mismatches"], 1)
        self.assertEqual(result["answer_spans"], 2)

        answer["answer_start"] = 5
        self.assertEqual(
            pack.measure("quoref", json.dumps(source).encode())[
                "source_offset_mismatches"
            ],
            0,
        )
        answer["answer_start"] = 9
        with self.assertRaisesRegex(ValueError, "out-of-range offset"):
            pack.measure("quoref", json.dumps(source).encode())

    def test_tatoeba_rejects_equal_totals_on_different_lines(self):
        rows = [
            dict(role="data", registry_id="LojbanTatoeba", path=name, records=2)
            for name in ("en", "jbo")
        ]
        with zipfile.ZipFile(io.BytesIO(), "w") as archive:
            archive.writestr("en", "one\n\nthree\n")
            archive.writestr("jbo", "lo pa\nlo re\n\n")
            with self.assertRaisesRegex(ValueError, "Unaligned Tatoeba"):
                pack.linked_evidence(archive, rows)

    def test_coqa_rejects_unlicensed_domains(self):
        passage = {
            "source": "mctest",
            "story": "A story",
            "questions": [],
            "answers": [],
        }
        with self.assertRaisesRegex(ValueError, "Unexpected CoQA source"):
            pack.measure("coqa", json.dumps({"data": [passage]}).encode())


if __name__ == "__main__":
    unittest.main()
