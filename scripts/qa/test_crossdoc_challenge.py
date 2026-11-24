"""Check the private case runner's identity, outcome and receipt contract."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import replay_crossdoc_case as replay


ROOT = Path(__file__).resolve().parents[2]


class CrossDocChallengeTest(unittest.TestCase):
    def test_inputs_and_adjudication_have_separate_fingerprints(self):
        case = replay.reviewed_case(ROOT, "catalog-revision")
        initial = replay.fingerprints(case)
        changed = copy.deepcopy(case)
        changed["assertions"][0]["kind"] = "separate"
        adj = replay.fingerprints(changed)
        self.assertEqual(initial["source_sha256"], adj["source_sha256"])
        self.assertNotEqual(initial["oracle_sha256"], adj["oracle_sha256"])
        changed = copy.deepcopy(case)
        changed["documents"].reverse()
        self.assertNotEqual(initial["source_sha256"], replay.fingerprints(changed)["source_sha256"])

    def test_outcome_requires_measured_booleans(self):
        case = replay.reviewed_case(ROOT, "catalog-revision")
        with patch.object(replay.crossdoc_cases, "run_case") as run:
            import sys
            binary = Path(sys.executable)
            for cases in ([], [{"passed": 0}], [{"result": "pass"}]):
                run.return_value = {"checks": cases}
                with self.assertRaises(ValueError):
                    replay.execute(case, binary)
            run.return_value = {"checks": [{"passed": True}, {"passed": False, "observed": "private"}]}
            receipt, failures, _ = replay.execute(case, binary)
            self.assertEqual((receipt["passed"], receipt["failed"], receipt["outcome"], failures),
                             (1, 1, "fail", [1]))
            self.assertNotIn("private", json.dumps(receipt))

    def test_private_files_are_executable_without_becoming_reviewed(self):
        case = replay.reviewed_case(ROOT, "topic-scoped-homonyms")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "my-case.json"
            path.write_text(json.dumps(case))
            self.assertEqual(case, replay.load_case(path))
            with self.assertRaises(ValueError):
                replay.reviewed_case(ROOT, str(path))


if __name__ == "__main__":
    unittest.main()
