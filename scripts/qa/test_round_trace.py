import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import tempfile
import unittest

import round_trace


class TraceTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.db = self.root / "trace.sqlite3"
        self.cli = [sys.executable, round_trace.__file__, str(self.db)]
        self.handoff = "private round prompt\n"
        round_trace.initialize(self.db, self.handoff)

    def call(self, *args, code=0):
        result = subprocess.run([*self.cli, *args], capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, code, result.stderr)
        return result.stdout.strip()

    def test_brief_opt_in_matches_handoff_and_preserves_existing_file(self):
        (self.root / "schedule.json").write_text(json.dumps(
            {"slot": 2, "requested_strategy": "estimate", "seed": 9}))
        path = self.root / "another.sqlite"
        brief = [sys.executable, str(Path(round_trace.__file__).with_name("brief_round.py")),
                 str(self.root), "private question"]
        plain = subprocess.run(brief, check=True, capture_output=True)
        self.assertFalse(path.exists())
        traced = subprocess.run([*brief, "--trace", str(path)], check=True, capture_output=True)
        self.assertEqual(plain.stdout, traced.stdout)
        snapshot = path.read_bytes()
        self.assertNotIn(b"private question", snapshot)
        with sqlite3.connect(path) as db:
            digest, = db.execute("SELECT prompt_sha256 FROM spans").fetchone()
        self.assertEqual(hashlib.sha256(traced.stdout.decode().replace("\r\n", "\n").encode()).hexdigest(), digest)
        self.assertNotEqual(subprocess.run([*brief, "--trace", str(path)], capture_output=True).returncode, 0)
        self.assertEqual(path.read_bytes(), snapshot)

    def test_agent_and_replay_store_digests_and_report_partial_coverage(self):
        prompt = self.root / "private-fixture"
        result = self.root / "private-result"
        prompt.write_bytes(b"secret input")
        span = self.call("start", "--role", "scout", "--tool", "host", "--prompt-file", prompt)
        self.call("run", "--role", "replay", "--tool", "python", "--parent", span,
                  "--input-file", prompt, "--result-file", result, "--", sys.executable, "-c",
                  "import pathlib,sys;pathlib.Path(sys.argv[1]).write_bytes(b'secret output')", result)
        self.call("end", span, "--status", "ok", "--result-file", result, "--model", "observed-v1",
                  "--input-tokens", "0", "--output-tokens", "2", "--cost-usd", "0")
        failed = self.call("start", "--role", "review", "--tool", "host")
        self.call("end", failed, "--status", "error")
        retry = self.call("start", "--role", "review", "--tool", "host", "--retry-of", failed)
        report = json.loads(self.call("report"))
        data = {s["id"]: s for s in report["spans"]}
        self.assertEqual(data[span]["parent"], report["trace_id"])
        self.assertEqual(data[retry]["retry_of"], failed)
        self.assertEqual(report["failed_spans"], [failed])
        self.assertEqual(set(report["open_spans"]), {retry, report["trace_id"]})
        self.assertEqual(report["agent_usage"], dict(agents=3, attempts=3, completed=1, errors=1,
                                                    interrupted=0, open=1, assertion_failed=0,
                                                    known_model=1, known_tokens=1,
                                                    known_cost=1, reported_cost_usd=0, known_input_tokens=1,
                                                    reported_input_tokens=0, known_output_tokens=1,
                                                    reported_output_tokens=2))
        command = next(s for s in data.values() if s["kind"] == "command")
        self.assertEqual(command["parent"], span)
        self.assertEqual(command["return_code"], 0)
        self.assertGreater(command["elapsed_seconds"], 0)
        self.assertEqual(command["elapsed_basis"], "monotonic")
        self.assertEqual(data[span]["elapsed_basis"], "wall_clock")
        self.assertEqual(command["prompt_sha256"], hashlib.sha256(prompt.read_bytes()).hexdigest())
        self.assertEqual(command["result_sha256"], hashlib.sha256(result.read_bytes()).hexdigest())
        self.assertRegex(report["revision"], r"^[0-9a-f]{40}$")
        stored = self.db.read_bytes()
        for secret in (b"secret input", b"secret output", b"private-fixture", b"pathlib", b"private round prompt"):
            self.assertNotIn(secret, stored)

    def test_failed_dispatch_then_completed_retry_preserves_unknown_usage(self):
        failed = self.call("start", "--role", "oracle", "--tool", "host")
        self.call("end", failed, "--status", "error")
        retry = self.call("start", "--role", "oracle", "--tool", "host", "--retry-of", failed)
        self.call("end", retry, "--status", "ok")
        self.call("end", "round", "--status", "ok")
        report = json.loads(self.call("report"))
        self.assertEqual(report["agent_usage"], dict(
            agents=2, attempts=2, completed=1, errors=1, interrupted=0, open=0, assertion_failed=0,
            known_model=0, known_tokens=0, known_cost=0, known_input_tokens=0, known_output_tokens=0,
            reported_cost_usd=None, reported_input_tokens=None, reported_output_tokens=None))
        by_id = {s["id"]: s for s in report["spans"]}
        self.assertEqual(by_id[retry]["retry_of"], failed)

    def test_interrupted_and_assertion_failed_attempts_have_distinct_counts(self):
        for outcome in ("interrupted", "assertion_failed"):
            attempt = self.call("start", "--role", "reviewer", "--tool", "host")
            self.call("end", attempt, "--status", outcome)
        usage = json.loads(self.call("report"))["agent_usage"]
        self.assertEqual(usage["attempts"], 2)
        self.assertEqual(usage["interrupted"], 1)
        self.assertEqual(usage["assertion_failed"], 1)
        self.assertEqual(usage["completed"], 0)
        self.assertEqual(usage["errors"], 0)
        self.assertEqual(usage["open"], 0)

    def test_runner_reports_failure_and_missing_executable(self):
        self.call("run", "--role", "replay", "--tool", "test", "--", sys.executable,
                  "-c", "raise SystemExit(7)", code=7)
        self.call("run", "--role", "replay", "--tool", "missing", "--", self.root / "absent", code=127)
        report = json.loads(self.call("report"))
        self.assertEqual([s["return_code"] for s in report["spans"] if s["status"] == "error"], [7, None])

    @unittest.skipUnless(os.name == "posix", "POSIX executable permissions")
    def test_permission_denied_does_not_invent_a_child_status(self):
        script = self.root / "denied"
        script.write_text("exit 0")
        script.chmod(0o600)
        self.call("run", "--role", "replay", "--tool", "denied", "--", script, code=126)
        report = json.loads(self.call("report"))
        self.assertEqual(len(report["failed_spans"]), 1)
        self.assertIsNone(report["spans"][-1]["return_code"])

    def test_assertion_code_classification_preserves_exit_status(self):
        for code in (1, 2, 7):
            self.call("run", "--role", "replay", "--tool", "checker", "--assertion-failure-code", "1",
                      "--assertion-failure-code", "7", "--", sys.executable,
                      "-c", f"raise SystemExit({code})", code=code)
        self.call("run", "--role", "replay", "--tool", "other", "--", sys.executable,
                  "-c", "raise SystemExit(1)", code=1)
        for invalid in ("0", "-1", "126"):
            self.call("run", "--role", "replay", "--tool", "checker", "--assertion-failure-code",
                      invalid, "--", sys.executable, "-c", "pass", code=2)
        report = json.loads(self.call("report"))
        self.assertEqual(len(report["assertion_failed_spans"]), 2)
        self.assertEqual(len(report["failed_spans"]), 2)
        self.assertEqual([(s["status"], s["return_code"]) for s in report["spans"] if s["kind"] == "command"],
                         [("assertion_failed", 1), ("error", 2), ("assertion_failed", 7), ("error", 1)])

    def test_digest_failure_is_separate_from_child_status(self):
        for code in (0, 9):
            self.call("run", "--role", "replay", "--tool", "py", "--result-file", self.root / "missing",
                      "--", sys.executable, "-c", f"raise SystemExit({code})", code=code or 2)
        report = json.loads(self.call("report"))
        commands = [s for s in report["spans"] if s["kind"] == "command"]
        self.assertEqual([s["return_code"] for s in commands], [0, 9])
        self.assertEqual([s["status"] for s in commands], ["ok", "error"])
        self.assertEqual(report["recording_issues"], [s["id"] for s in commands])

    def test_concurrent_children_and_single_terminal_transition(self):
        with concurrent.futures.ThreadPoolExecutor(6) as pool:
            spans = list(pool.map(lambda _: self.call("start", "--role", "test", "--tool", "host"), range(10)))
            self.assertEqual(len(set(spans)), 10)
            self.call("end", "round", "--status", "ok", code=2)
            def finish(span):
                return subprocess.run([*self.cli, "end", span, "--status", "ok"],
                                      capture_output=True, timeout=20).returncode
            outcomes = list(pool.map(finish, spans + spans))
        self.assertEqual(outcomes.count(0), 10)
        self.assertEqual(outcomes.count(2), 10)
        self.call("end", "round", "--status", "ok")
        self.call("start", "--role", "late", "--tool", "host", code=2)
        self.assertFalse(json.loads(self.call("report"))["open_spans"])

    @unittest.skipUnless(os.name == "posix", "process-group signal test")
    def test_killed_wrapper_leaves_recoverable_open_span(self):
        cmd = [*self.cli, "run", "--role", "replay", "--tool", "sleep", "--",
               sys.executable, "-c", "import time; time.sleep(20)"]
        with subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True, start_new_session=True) as proc:
            try:
                key = proc.stderr.readline().strip().split()[-1]
            finally:
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=5)
        report = json.loads(self.call("report"))
        self.assertIn(key, report["open_spans"])
        self.assertIsNone(next(s["return_code"] for s in report["spans"] if s["id"] == key))
        self.call("end", key, "--status", "interrupted")
        self.assertIn(key, json.loads(self.call("report"))["failed_spans"])
