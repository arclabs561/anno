# SPDX-License-Identifier: MIT OR Apache-2.0
from pathlib import Path
from subprocess import CompletedProcess
import tempfile
import unittest

import replay_bibliography as adapter
import replay_feedback as feedback


class BibliographyReplay(unittest.TestCase):
    def test_only_a_completed_comparison_records_an_outcome(self):
        def result(code, text):
            return CompletedProcess([], code, stdout=text)

        self.assertEqual(adapter.verdict(result(0, "summary\nattribution-history ok\n")), ("pass", ""))
        failed, signature = adapter.verdict(result(1, "attribution-history mismatch: prior-name\n"))
        self.assertEqual(failed, "fail")
        self.assertRegex(signature, "^" + adapter.FAIL_SIGNATURE + "$")
        for code, text in [(0, ""), (1, ""), (0, "attribution-history mismatch: prior-name"),
                           (1, "attribution-history ok"), (2, "attribution-history mismatch: prior-name")]:
            with self.subTest(code=code, text=text), self.assertRaises(ValueError):
                adapter.verdict(result(code, text))

    def test_changing_the_fixture_and_relabeling_an_answer_have_separate_pins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for path in [*adapter.FIXTURES, adapter.CHECKER]:
                full = root / path
                full.parent.mkdir(parents=True, exist_ok=True)
                full.write_text("original")
            original = adapter.describe(root, adapter.CASE)
            (root / adapter.FIXTURES[0]).write_text("different occurrence")
            changed = adapter.describe(root, adapter.CASE)
            self.assertNotEqual(original["source_sha256"], changed["source_sha256"])
            self.assertEqual(original["oracle_sha256"], changed["oracle_sha256"])
            (root / adapter.CHECKER).write_text("changed expected answer")
            relabeled = adapter.describe(root, adapter.CASE)
            self.assertEqual(changed["source_sha256"], relabeled["source_sha256"])
            self.assertNotEqual(changed["oracle_sha256"], relabeled["oracle_sha256"])
            self.assertIs(feedback.ADAPTERS["bibliography"], adapter)
            with self.assertRaises(ValueError):
                adapter.describe(root, "../local-history")


if __name__ == "__main__":
    unittest.main()
