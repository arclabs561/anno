import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from zipfile import ZipFile

import crossdoc_eval_cases as runner


FIXTURES = Path(__file__).parent / "fixtures" / "crossdoc_eval"
EVAL_BIN = os.environ.get("ANNO_CDCR_BIN")


def fixture(name):
    return json.loads((FIXTURES / f"{name}.json").read_text())


class AuthoringTests(unittest.TestCase):
    def test_reproducible_archive_and_local_scope(self):
        case = fixture("topic-scoped-homonyms")
        case["documents"][0]["mentions"][0] = {"local": "A&B", "tokens": [1]}
        case["documents"][0]["sentences"] = [["A&B"]]
        blobs = []
        for _ in range(2):
            with TemporaryDirectory() as tmp:
                self.assertEqual(runner._write_panel(case, Path(tmp)), [])
                archive = Path(tmp) / "datasets/ECBPlus.cache"
                blobs.append(archive.read_bytes())
                with ZipFile(archive) as source:
                    xml = source.read("ECB+/1/1_a.xml")
                    self.assertIn(b"INTRA_DOC_COREF", xml)
                    self.assertIn(b"A&amp;B", xml)
                    self.assertNotIn(b"CROSS_DOC_COREF", xml)
        self.assertEqual(*blobs)

    def test_rejects_unknown_inputs_and_empty_assertions(self):
        case = fixture("local-seeds")
        case["assertions"][0] = {"seeds": "singletons", "threshold": 0.7}
        with self.assertRaises(ValueError):
            runner._expectations(case)
        for invalid in ({"tokens": [0]}, {"tokens": [1], "cross": "x", "local": "x"},
                        {"tokens": [1], "command": "x"}):
            with self.subTest(data=invalid), TemporaryDirectory() as tmp:
                case["documents"][0]["mentions"] = [invalid]
                with self.assertRaises(ValueError):
                    runner._write_panel(case, Path(tmp))

    @unittest.skipUnless(EVAL_BIN, "set ANNO_CDCR_BIN to the built cdcr_eval executable")
    def test_real_runner_detects_relabeling_and_selected_denominators(self):
        case = fixture("topic-scoped-homonyms")
        binary = Path(EVAL_BIN)
        self.assertTrue(all(c["passed"] for c in runner.run_case(case, binary)["checks"]))

        # An adversary changes the source to say the topic-1 homonyms are the same person.
        # The fixed reference expects distinct identities, so the process succeeds but the oracle fails.
        case["documents"][1]["mentions"][0]["cross"] = "A"
        checks = runner.run_case(case, binary)["checks"]
        failed = {c["name"] for c in checks if not c["passed"]}
        self.assertIn("singletons@0.7 gold_links", failed)
        self.assertIn("gold_local@0.7 correct_links", failed)
        self.assertNotIn("singletons@0.7 predicted_links", failed)

        case = fixture("local-seeds")
        case["selected_sentences"] = [{"topic": 1, "id": doc, "sentences": [0]} for doc in ("a", "b")]
        for row in case["assertions"]:
            row.update(gold_links=1, correct_links=1, predicted_links=1, predicted_clusters=1)
        self.assertTrue(all(c["passed"] for c in runner.run_case(case, binary)["checks"]))

    @unittest.skipUnless(EVAL_BIN, "set ANNO_CDCR_BIN to the built cdcr_eval executable")
    def test_withdrawn_mentions_can_be_adjudicated_with_gold_counters_only(self):
        case = fixture("local-seeds")
        case["documents"][1]["mentions"] = []  # The second document still exists.
        case["assertions"] = [{"seeds": "singletons", "threshold": 0.7, "gold_links": 0, "gold_clusters": 1}]
        checks = runner.run_case(case, Path(EVAL_BIN))["checks"]
        self.assertEqual({c["name"] for c in checks}, {"singletons@0.7 gold_links", "singletons@0.7 gold_clusters"})
        self.assertTrue(all(c["passed"] for c in checks))

    def test_rejects_partial_scorer_output_even_for_partial_adjudication(self):
        header = "| Local seeds | Threshold | Gold clusters | Pred clusters | Correct links | Gold links | Pred links |"
        with self.assertRaisesRegex(RuntimeError, "full seed/threshold"):
            runner._observations(header + "\n| Singletons | 0.7 | 1 | 1 | 0 | 0 | 0 |")


if __name__ == "__main__":
    unittest.main()
