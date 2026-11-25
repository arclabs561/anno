"""Exercise the four LitBank judgments without model weights or upstream text."""
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import replay_litbank_role as litbank
from fcoref_cgroup import replay
from replay_litbank_role import FIXTURE, prepare, score, source_sha256


class LitBankRoleTest(unittest.TestCase):
    def setUp(self):
        self.pairs = json.loads(FIXTURE.read_text())["pairs"]
        self.doc = {"id": "control", "text": "α β γ δ ε"}
        self.named = dict(zip(("pronoun", "correct_name", "role", "foil"),
                              ((0, 1), (2, 3), (4, 5), (6, 7))))

    def result(self, groups, omitted=()):
        all_spans = list(self.named.values()) + [(8, 9)]
        mentions = [{"start": a, "end": b, "surface": self.doc["text"][a:b]}
                    for a, b in all_spans if (a, b) not in omitted]
        predictions = {"results": [{"id": "control", "source_sha256": source_sha256(self.doc["text"]),
                                    "mentions": mentions, "clusters": groups}]}
        return score(self.doc, self.named, self.pairs, predictions)

    def test_correct_and_wrong_person(self):
        correct = self.result([[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9]]])
        self.assertEqual(correct["status"], "pass")
        result = self.result([[[0, 1], [4, 5], [6, 7]], [[2, 3], [8, 9]]])
        self.assertEqual(result["status"], "mismatch")
        self.assertEqual([p["observed"] for p in result["pairs"]],
                         ["different", "same", "same", "different"])

    def test_missing_or_unclustered_foil_cannot_pass_negative_judgment(self):
        for omitted in ([], [(6, 7)]):
            result = self.result([[[0, 1], [2, 3], [4, 5]]], omitted)
            self.assertEqual(result["endpoints"]["foil"]["proposed"], not omitted)
            self.assertEqual(result["pairs"][1]["observed"], "unresolved")
            self.assertEqual(result["status"], "mismatch")

    def test_cli_distinguishes_unavailable_model_and_invalid_receipt(self):
        # Inject only a toy source descriptor; exercise main's argument parsing and exit paths.
        driver = Path(__file__).with_name("replay_litbank_role.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "source.txt").write_text(self.doc["text"], encoding="utf-8")
            fixture = dict(self.doc, source_path="source.txt", source_sha256=source_sha256(self.doc["text"]),
                           excerpt=[0, len(self.doc["text"])], excerpt_sha256=source_sha256(self.doc["text"]),
                           spans=self.named, pairs=self.pairs)
            (root / "fixtures").mkdir()
            (root / "fixtures" / FIXTURE.name).write_text(json.dumps(fixture))
            cli = root / driver.name
            cli.write_bytes(driver.read_bytes())

            def run(*args):
                return subprocess.run([sys.executable, cli, root, *args], capture_output=True,
                                      env=dict(os.environ, PYTHONPATH=str(driver.parent)), text=True)

            for args in [(), ("--runner", root / "missing", "--model-dir", root),
                         ("--runner", sys.executable, "--model-dir", root / "missing")]:
                proc = run(*args)
                self.assertEqual(proc.returncode, 77, proc.stderr)
                self.assertEqual(json.loads(proc.stdout)["status"], "not_run")
            path = root / "predictions.json"
            path.write_text(json.dumps({"results": [{
                "id": self.doc["id"], "source_sha256": source_sha256(self.doc["text"]),
                "mentions": [], "clusters": [],
            }], "model": "unverified"}))
            saved = run("--predictions", path)
            self.assertEqual(saved.returncode, 1, saved.stderr)
            self.assertEqual(json.loads(saved.stdout)["prediction_source"], {
                "kind": "saved_predictions", "receipt_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
            (root / "encoder_quantized.onnx").touch()
            shadowed = run("--runner", sys.executable, "--model-dir", root)
            self.assertEqual(shadowed.returncode, 2)
            self.assertIn("override", shadowed.stderr)
            path.write_text("corrupt")
            proc = run("--predictions", path)
            self.assertEqual(proc.returncode, 2)
            self.assertIn("LitBank replay error", proc.stderr)

    def test_model_identity_rejects_encoder_substitution_and_changed_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = {}
            for name in replay.MODEL_SHA:
                data = name.encode()
                (root / name).write_bytes(data)
                expected[name] = hashlib.sha256(data).hexdigest()
            with patch.object(replay, "MODEL_SHA", expected):
                self.assertEqual(replay.pinned_fp32_hashes(root), expected)
                predictions = {"results": [{
                    "id": self.doc["id"], "source_sha256": source_sha256(self.doc["text"]),
                    "mentions": [], "clusters": [],
                }]}
                # The supplied runner is a stand-in. Verification hashes the real local files.
                with patch.object(litbank, "prepare", return_value=(self.doc, self.named)), \
                        patch.object(sys, "argv", ["litbank", str(root), "--runner", sys.executable,
                                                 "--model-dir", str(root)]), \
                        patch.object(subprocess, "check_output", return_value=json.dumps(predictions).encode()), \
                        redirect_stdout(io.StringIO()) as stdout:
                    self.assertEqual(litbank.main(), 1)
                self.assertEqual(json.loads(stdout.getvalue())["prediction_source"],
                                 {"kind": "pinned_fp32", "files_sha256": expected})
                alternate = root / "encoder_quantized.onnx"
                alternate.write_bytes(b"preferred by native loader")
                with self.assertRaisesRegex(ValueError, "override"):
                    replay.pinned_fp32_hashes(root)
                alternate.unlink()
                (root / "encoder.onnx").write_bytes(b"wrong FP32 bytes")
                with self.assertRaisesRegex(ValueError, "pinned FP32"):
                    replay.pinned_fp32_hashes(root)

    def test_source_and_scalar_offsets(self):
        source = "🐟" + self.doc["text"]
        fixture = {"id": "control", "source_path": "source.txt", "source_sha256": source_sha256(source),
                   "excerpt": [1, len(source)], "excerpt_sha256": source_sha256(self.doc["text"]),
                   "spans": {k: [a + 1, b + 1] for k, (a, b) in self.named.items()}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.txt"
            path.write_text(source, encoding="utf-8")
            self.assertEqual(prepare(fixture, path.parent), (self.doc, self.named))
            path.write_text(source + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "pinned text"):
                prepare(fixture, path.parent)


if __name__ == "__main__":
    unittest.main()
