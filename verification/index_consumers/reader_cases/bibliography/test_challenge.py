# SPDX-License-Identifier: MIT OR Apache-2.0
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import challenge


HERE = Path(__file__).resolve().parent


class BibliographyChallenge(unittest.TestCase):
    def test_first_before_inclusive_and_same_day_correction(self):
        works, queries, expected = challenge.example(41)
        self.assertEqual(challenge.example(41), (works, queries, expected))
        keyed = {(q["cut"], q.get("date")): result for q, result in zip(queries, expected)}
        self.assertEqual(keyed["first", None]["edition"], 1)
        self.assertEqual(keyed["before", "2020-07-26"]["reason"], "no_edition_at_cut")
        self.assertEqual(keyed["before", "2020-08-02"]["edition"], 1)
        self.assertEqual(keyed["as_of", "2020-08-02"]["byline"][0], [["collab", "Survey group 2"]])
        self.assertEqual(keyed["before", "2020-08-13"]["edition"], 2)
        self.assertEqual(keyed["as_of", "2020-08-13"]["edition"], 4)
        self.assertEqual(challenge.failures(works, queries, expected, False), [])

    def test_key_is_independent_of_mutation_to_delivered_byline(self):
        works, queries, key = challenge.example(41)
        edition = next(e for e in works[0]["editions"] if e["edition"] == 2)
        edition["byline"].reverse()
        failures = challenge.failures(works, queries, key, False)
        self.assertTrue(failures)
        for index, observed in failures:
            self.assertEqual(key[index]["edition"], 2)
            self.assertNotEqual(observed["byline"], key[index]["byline"])

    def test_minimized_case_replays_through_both_public_clis(self):
        with tempfile.TemporaryDirectory() as tmp:
            save = Path(tmp) / "local"
            command = [sys.executable, str(HERE / "challenge.py")]
            run = subprocess.run([*command, "--seed", "41", "--fault", "--save", str(save)],
                                 capture_output=True, check=False)
            self.assertEqual(run.returncode, 1, run.stderr)
            receipt = json.loads(run.stdout)
            self.assertEqual((receipt["expected_edition"], receipt["observed_edition"]), (2, 1))
            self.assertEqual(receipt["retained_editions"], 2)
            self.assertGreater(receipt["original"]["queries"], 1)
            self.assertEqual(len(json.loads((save / "queries.json").read_bytes())), 1)
            for fault, status in (([], 0), (["--fault"], 1)):
                result = subprocess.run([*command, "--replay", str(save), *fault], capture_output=True)
                self.assertEqual(result.returncode, status, result.stderr)
                if fault:
                    replayed = json.loads(result.stdout)
                    self.assertEqual(replayed["input_sha256"], receipt["input_sha256"])
                    self.assertEqual(replayed["key_sha256"], receipt["key_sha256"])
            actual = subprocess.check_output([sys.executable, str(HERE / "reader.py"),
                                              save / "works.json", save / "queries.json"])
            self.assertEqual(json.loads(actual), json.loads((save / "expected.json").read_bytes()))

    def test_an_incomplete_process_result_is_an_error(self):
        case = challenge.example(41)
        with patch.object(challenge, "read_bulk", return_value=[]), self.assertRaisesRegex(ValueError, "incomplete"):
            challenge.failures(*case, False)
        del case[2][0]["edition"]
        with self.assertRaisesRegex(ValueError, "answer key"):
            challenge.failures(*case, False)


if __name__ == "__main__":
    unittest.main()
