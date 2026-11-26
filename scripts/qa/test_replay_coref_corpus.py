"""Offline tests for the named coreference feedback adapter and witness identity."""
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import replay_coref_corpus as adapter
import replay_feedback as feedback

CASE_ID = "1" * 16
NAME = "litbank/" + CASE_ID


class CorpusWitnessTests(unittest.TestCase):
    def setUp(self):
        tmp = self.enterContext(tempfile.TemporaryDirectory())
        self.base = Path(tmp)
        self.model = self.base / "model"
        self.model.mkdir()
        for name in ("encoder.onnx", "tokenizer.json", "scorer_weights.safetensors", "config.json"):
            (self.model / name).write_bytes(b"initial artifact")
        self.runner = self.base / "runner"
        self.runner.write_text(
            f"#!{sys.executable}\n"
            "import hashlib, json, pathlib, sys\n"
            "item, = json.loads(pathlib.Path(sys.argv[2]).read_bytes())\n"
            "assert set(item) == {'id','text'}\n"
            "print(json.dumps({'results':[{'id':item['id'],"
            "'source_sha256':hashlib.sha256(item['text'].encode()).hexdigest(),"
            "'mentions':[],'clusters':[]}]}))\n"
        )
        self.runner.chmod(0o700)
        self.fixture = self.base / "rows"
        paths = self.base / "sources.json"
        paths.write_text(json.dumps({"litbank": self.fixture.name}))
        self.enterContext(patch.dict(os.environ, {
            "ANNO_COREF_SOURCES": str(paths), "ANNO_COREF_RUNNER": str(self.runner),
            "ANNO_COREF_MODEL_DIR": str(self.model),
        }))
        self.enterContext(patch.dict(adapter.corpus.READERS, {"litbank": fixture_cases}))
        self.row = {"id": CASE_ID, "group": "private-work-canary", "text": "private-doc-canary",
                    "kind": "choice", "spans": {"target": [0, 7], "A": [8, 11]}, "gold": "A"}
        self.write_fixture()

    def write_fixture(self):
        self.fixture.write_text(json.dumps([self.row]))

    def test_real_protocol_confirms_failure_and_model_change_retires_the_witness(self):
        cache = self.base / "ledger"
        with patch.object(feedback, "stamp", return_value="b" * 40):
            original = feedback.observe(feedback.ROOT, cache, "coref_corpus", NAME)
        self.assertEqual((original["outcome"], original["signature"], original["replays"]),
                         ("fail", "wrong_marked_links:neither", 2))
        self.assertEqual(feedback.next_case(feedback.ROOT, cache)["case"], NAME)
        with sqlite3.connect(cache / feedback.DB_NAME) as db:
            raw, = db.execute("SELECT receipt FROM observations").fetchone()
        self.assertNotIn("private-", raw)
        self.assertNotIn("spans", raw)
        self.assertNotIn(str(self.base), raw)
        for name in ("config.json", "encoder_quantized.onnx", "tokenizer.json"):
            artifact = self.model / name
            before = artifact.read_bytes() if artifact.exists() else None
            artifact.write_bytes(b"changed")
            self.assertIsNone(feedback.next_case(feedback.ROOT, cache), name)
            if before is None:
                artifact.unlink()
            else:
                artifact.write_bytes(before)
        self.row["spans"]["A"] = [7, 11]
        self.write_fixture()
        self.assertIsNone(feedback.next_case(feedback.ROOT, cache))
        self.row["spans"]["A"] = [8, 11]
        self.row["gold"] = "neither"
        self.write_fixture()
        self.assertIsNone(feedback.next_case(feedback.ROOT, cache))
        self.assertEqual(adapter.replay(feedback.ROOT, NAME)["signature"], adapter.PASS_SIGNATURE)

    def test_nested_missing_endpoint_is_a_measured_failure(self):
        self.row.update(kind="nested", gold="different", spans={"inner": [0, 7], "outer": [0, 11]})
        self.write_fixture()
        code, result = feedback.replay_once(feedback.ROOT, "coref_corpus", NAME)
        self.assertEqual((code, result["signature"]), (1, "nested_endpoint_unavailable"))

    def test_unscored_and_no_offered_candidate_are_ineligible(self):
        for gold, spans in (("unscored", self.row["spans"]), ("neither", {"target": [0, 7]})):
            self.row.update(gold=gold, spans=spans)
            self.write_fixture()
            with self.assertRaises(adapter.NoMarkedComparison):
                feedback.replay_once(feedback.ROOT, "coref_corpus", NAME)

    def test_named_replay_with_unavailable_local_inputs(self):
        env = os.environ.copy()
        del env["ANNO_COREF_SOURCES"]
        result = subprocess.run(
            [sys.executable, feedback.ROOT / "scripts/qa/replay_feedback.py", "replay", "coref_corpus", NAME],
            capture_output=True, env=env,
        )
        self.assertEqual(result.returncode, 3)
        self.assertEqual(json.loads(result.stdout), {"outcome": "unavailable", "detail": adapter.LOCAL_INPUTS})
        with self.assertRaises(ValueError):
            adapter.family_for_case(None, "litbank/source-work-title")
        self.assertEqual(adapter.family_for_case(None, "phrase-detectives/pd3-" + "f"*24), "phrase-detectives")


def fixture_cases(path):
    return json.loads(path.read_bytes())


if __name__ == "__main__":
    unittest.main()
