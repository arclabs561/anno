"""Offline checks for the blind oracle CLI and replay of its frozen answers."""
from contextlib import closing, redirect_stderr, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import qa_oracle
import round_trace


def digest(data):
    return hashlib.sha256(data).hexdigest()


class OracleCliTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.task = self.root / "task.txt"
        self.task.write_bytes("Label people in half-open Unicode scalar offsets.\r\n🪶Zoë wrote.\r\n".encode())
        self.answer = self.root / "answer.json"
        self.trace = self.root / "round.sqlite3"

    def cli(self, *args, stdin=b"", code=0):
        result = subprocess.run([sys.executable, qa_oracle.__file__, *map(str, args)],
                                input=stdin, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, code, result.stderr)
        return result.stdout

    def provider(self, name, out, *args, code=0):
        cmd = [name, str(self.task), str(out), "--trace", str(self.trace), *args]
        if name == "openrouter":
            cmd.extend(["--model", "requested/model"])
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            self.assertEqual(qa_oracle.main(cmd), code, stderr.getvalue())
        return json.loads(stdout.getvalue()) if code == 0 else None

    def report(self):
        with closing(round_trace.connect(self.trace)) as db:
            return round_trace.report(db)

    def test_all_transports_use_exported_prompt_and_freeze_the_same_native_json(self):
        round_trace.initialize(self.trace, "coordinator-only candidate prediction")
        prompt = self.cli("prompt", self.task)
        self.assertEqual(prompt, self.task.read_bytes())
        self.assertNotIn(b"coordinator-only", prompt)
        reply = '[ {"label":"Zoë"} ]'
        accepted = json.loads(self.cli("accept", self.task, self.answer, "--shape", "array",
                                       stdin=reply.encode()))
        frozen = self.answer.read_bytes()
        self.assertEqual(frozen, reply.encode())
        self.assertEqual(json.loads(frozen), [{"label": "Zoë"}])
        self.assertEqual(accepted["prompt_sha256"], digest(prompt))
        self.assertEqual(accepted["answer_sha256"], digest(frozen))
        for field in ("model", "input_tokens", "output_tokens", "cost_usd"):
            self.assertIsNone(accepted[field])
        for name, transport, response in (
            ("openrouter", "oracle_gateway", (reply, "resolved/model", 9, 0, 0)),
            ("codex", "oracle_headless", (reply, None, None, None)),
        ):
            with self.subTest(provider=name):
                out = self.root / f"{name}.json"
                with patch(f"{transport}.complete", autospec=True, return_value=response) as complete:
                    receipt = self.provider(name, out, "--shape", "array")
                complete.assert_called_once()
                self.assertEqual(complete.call_args.args[0].encode(), prompt)
                if name == "openrouter":
                    self.assertEqual(complete.call_args.kwargs["shape"], "array")
                self.assertEqual(out.read_bytes(), frozen)
                self.assertEqual(receipt["prompt_sha256"], digest(prompt))
                self.assertEqual(receipt["answer_sha256"], digest(frozen))
                agent, = [s for s in self.report()["spans"] if s["kind"] == "agent" and s["tool"] == name]
                self.assertEqual((agent["status"], agent["prompt_sha256"], agent["result_sha256"]),
                                 ("ok", digest(prompt), digest(frozen)))
                self.assertEqual(agent["model"], receipt["model"])
        usage = self.report()["agent_usage"]
        self.assertEqual((usage["attempts"], usage["completed"], usage["known_model"]), (2, 2, 1))
        self.assertEqual((usage["known_tokens"], usage["reported_input_tokens"],
                          usage["reported_output_tokens"]), (1, 9, 0))
        self.assertEqual((usage["known_cost"], usage["reported_cost_usd"]), (1, 0))

    def test_single_json_fence_preserves_payload_and_raw_reply_hash(self):
        payload = ' [ {"n": 1.000} ]'.encode()
        raw = b"```json\r\n" + payload + b"\r\n```\n"
        receipt = json.loads(self.cli("accept", self.task, self.answer, "--shape", "array", stdin=raw))
        self.assertEqual(self.answer.read_bytes(), payload)
        self.assertEqual(receipt["reply_sha256"], digest(raw))
        self.assertEqual(receipt["answer_sha256"], digest(payload))
        out = self.root / "extra.json"
        for invalid in (raw + b"text", b"preface\n" + raw, raw + raw, raw.replace(b"json", b"python")):
            self.cli("accept", self.task, out, "--shape", "array", stdin=invalid, code=2)
            self.assertFalse(out.exists())

    def test_unusable_gateway_reply_keeps_reported_usage_in_failed_span(self):
        round_trace.initialize(self.trace, "test")
        with patch("oracle_gateway.complete", return_value=("invalid", "actual/model", 12, 3, 0.01)):
            self.provider("openrouter", self.answer, code=2)
        agent, = [s for s in self.report()["spans"] if s["kind"] == "agent"]
        self.assertEqual((agent["status"], agent["input_tokens"], agent["cost_usd"]), ("error", 12, 0.01))
        self.assertFalse(self.answer.exists())

    def test_codex_openrouter_attempt_is_attributed_to_the_chosen_route(self):
        round_trace.initialize(self.trace, "test")
        with patch("oracle_headless.complete", return_value=('{}', None, 12, 3)) as complete:
            receipt = self.provider("codex", self.answer, "--openrouter", "--model", "requested/model")
        self.assertTrue(complete.call_args.kwargs["openrouter"])
        self.assertEqual(receipt["via"], "codex-openrouter")
        self.assertEqual(receipt["requested_model"], "requested/model")
        self.assertIsNone(receipt["model"])
        agent, = [s for s in self.report()["spans"] if s["kind"] == "agent"]
        self.assertEqual((agent["tool"], agent["input_tokens"]), ("codex-openrouter", 12))

    def test_invalid_reply_leaves_no_file_and_allows_a_new_attempt(self):
        for invalid in (b"", b"true", b"[]", b'{"x": 1} trailing', b'{"x": {"a":1, "a":2}}',
                        b'{"n":NaN}', b'{"n":Infinity}', b'{"n":-Infinity}', b'{"n":1e999}',
                        b'{"s":"\xff"}'):
            with self.subTest(reply=invalid):
                self.cli("accept", self.task, self.answer, stdin=invalid, code=2)
                self.assertFalse(self.answer.exists())
        self.cli("accept", self.task, self.answer, stdin=b'{"native":true}')
        self.assertEqual(json.loads(self.answer.read_bytes()), {"native": True})
        if os.name == "posix":
            self.assertEqual(stat.S_IMODE(self.answer.stat().st_mode) & 0o077, 0)
        self.cli("accept", self.task, self.answer, stdin=b'{"replacement":true}', code=2)
        self.assertEqual(json.loads(self.answer.read_bytes()), {"native": True})

    def test_provider_error_closes_attempt_and_does_not_publish(self):
        round_trace.initialize(self.trace, "coordinator handoff")
        prompt = self.cli("prompt", self.task)
        with patch("oracle_gateway.complete", side_effect=ValueError("mock provider unavailable")) as complete:
            self.provider("openrouter", self.answer, code=2)
        complete.assert_called_once()
        self.assertFalse(self.answer.exists())
        report = self.report()
        attempt, = [s for s in report["spans"] if s["kind"] == "agent"]
        self.assertEqual((attempt["status"], attempt["prompt_sha256"], attempt["result_sha256"]),
                         ("error", digest(prompt), None))
        self.assertEqual(report["agent_usage"]["errors"], 1)
        for field in ("model", "input_tokens", "output_tokens", "cost_usd"):
            self.assertIsNone(attempt[field])
        self.cli("accept", self.task, self.answer, stdin=b'{}')

    def test_output_created_during_dispatch_is_preserved(self):
        round_trace.initialize(self.trace, "coordinator handoff")
        other = b"concurrent writer"

        def competing_write(*args, **kwargs):
            self.assertFalse(self.answer.exists())
            self.answer.write_bytes(other)
            return "{}", None, None, None, None

        with patch("oracle_gateway.complete", side_effect=competing_write):
            self.provider("openrouter", self.answer, code=2)
        self.assertEqual(self.answer.read_bytes(), other)
        with patch("oracle_gateway.complete") as complete:
            self.provider("openrouter", self.answer, code=2)
            complete.assert_not_called()

    def test_frozen_span_proposal_replays_on_bound_unicode_source(self):
        text = "🪶Zoë wrote."
        self.task.write_text(f"Return a fixture for PERSON spans in Unicode scalar offsets.\n{text}",
                             encoding="utf-8")
        saved = self.root / "predictions.jsonl"
        entity = dict(text="Zoë", start=1, end=4, type="PER")
        saved.write_text(json.dumps(dict(id="note", source_sha256=digest(text.encode()), entities=[entity])) + "\n")
        checker = Path(qa_oracle.__file__).with_name("check_span_assertions.py")
        for name, source, expected_type, code in (
            ("clean", text, "PER", 0),
            ("wrong-label", text, "ORG", 1),
            ("changed-source", text + " ", "PER", 2),
        ):
            with self.subTest(case=name):
                assertion = dict(kind="exact_span", text="Zoë", start=1, end=4, types=[expected_type])
                reply = json.dumps([dict(id="note", text=source, assertions=[assertion])], ensure_ascii=False)
                out = self.root / f"{name}.json"
                self.cli("accept", self.task, out, "--shape", "array", stdin=reply.encode())
                replay = subprocess.run([sys.executable, checker, "--fixture", out, "--predictions", saved],
                                        capture_output=True, timeout=10)
                self.assertEqual(replay.returncode, code, replay.stderr)
                if code == 2:
                    self.assertIn(b"source_sha256 differs", replay.stderr)
                else:
                    score = json.loads(replay.stdout)
                    self.assertEqual((score["passed"], score["total"]), (int(code == 0), 1))


if __name__ == "__main__":
    unittest.main()
