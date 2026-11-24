"""Synthetic Codex executable tests and optional loopback CLI contract test."""

import http.server
import json
import os
import shutil
import sys
import tempfile
import textwrap
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oracle_headless as headless


def transcript(answer="[]", usage=None):
    events = [
        {"type": "thread.started", "thread_id": "synthetic"},
        {"type": "turn.started"},
        {"type": "item.completed", "item": {"type": "agent_message", "text": answer}},
        {"type": "turn.completed", "usage": usage},
    ]
    return "\n".join(json.dumps(e) for e in events)


class HeadlessTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        (self.root / "AGENTS.md").write_text("irrelevant synthetic context")
        self.binary = self.root / "codex"
        self.env = patch.dict(os.environ, {
            "PATH": tmp.name, "HOME": tmp.name, "OPENROUTER_API_KEY": "synthetic-router-key",
            "OTHER_SECRET": "synthetic-secret", "HTTPS_PROXY": "https://example.invalid", "TMPDIR": tmp.name, "CODEX_API_KEY": "synthetic-codex-key",
        }, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def fake(self, body):
        self.binary.write_text(f"#!{sys.executable}\n" + textwrap.dedent(body))
        self.binary.chmod(0o700)

    def run_cli(self, model="example", timeout=5):
        return headless.complete("A synthetic prompt.\n[]", model=model, timeout=timeout)

    def test_private_stdin_environment_and_arguments(self):
        report = self.root / "capture"
        self.fake(f"""
            import json, os, sys
            from pathlib import Path
            Path({str(report)!r}).write_text(json.dumps({{
                "args": sys.argv[1:], "env": dict(os.environ), "cwd": os.getcwd(),
                "files": os.listdir(), "prompt": sys.stdin.read(),
                "permissions": Path.cwd().parent.stat().st_mode & 0o777,
            }}))
            print({transcript("[]", {"input_tokens": 7, "output_tokens": 0})!r})
        """)
        self.assertEqual(self.run_cli(), ("[]", None, 7, 0))
        capture = json.loads(report.read_text())
        self.assertEqual(capture["prompt"], "A synthetic prompt.\n[]")
        self.assertEqual(capture["files"], [])
        self.assertEqual(capture["permissions"], 0o700)
        self.assertFalse(Path(capture["cwd"]).exists())
        self.assertNotIn(str(self.root), capture["cwd"])
        self.assertFalse(Path(capture["env"]["CODEX_HOME"]).exists())
        self.assertEqual(capture["env"]["CODEX_API_KEY"], "synthetic-codex-key")
        self.assertNotIn("OPENROUTER_API_KEY", capture["env"])
        self.assertNotIn("OTHER_SECRET", capture["env"])
        self.assertEqual(capture["env"]["HTTPS_PROXY"], "https://example.invalid")
        self.assertNotIn(capture["prompt"], capture["args"])
        self.assertEqual(capture["args"][-3:], ["--model", "example", "-"])
        for arg in ("--ephemeral", "--ignore-user-config", "--ignore-rules", "--strict-config"):
            self.assertIn(arg, capture["args"])

    def test_explicit_openrouter_route_uses_only_its_key_and_fixed_provider(self):
        login = self.root / "saved-login"
        login.mkdir()
        (login / "AGENTS.md").write_text("SYNTHETIC_OTHER_LOGIN")
        os.environ["CODEX_HOME"] = str(login)
        os.environ["OPENAI_BASE_URL"] = "https://unselected.example"
        report = self.root / "capture"
        self.fake(f"""
            import json, os, sys
            from pathlib import Path
            assert os.environ["OPENROUTER_API_KEY"] == "synthetic-routed-key"
            assert "CODEX_API_KEY" not in os.environ and "OPENAI_BASE_URL" not in os.environ
            assert Path(os.environ["CODEX_HOME"]).parent == Path(os.environ["HOME"])
            assert "synthetic-routed-key" not in str(sys.argv)
            Path({str(report)!r}).write_text(json.dumps(sys.argv[1:]))
            print({transcript()!r})
        """)
        with patch.object(headless.oracle_gateway, "api_key", return_value="synthetic-routed-key") as key:
            with self.assertRaisesRegex(headless.OracleHeadlessError, "explicit model"):
                headless.complete("Synthetic", model=None, timeout=5, openrouter=True)
            key.assert_not_called()
            result = headless.complete("Synthetic", model="openai/example", timeout=5, openrouter=True)
            key.assert_called_once_with()
        self.assertEqual(result, ("[]", None, None, None))
        argv = json.loads(report.read_text())
        self.assertEqual(argv[-3:], ["--model", "openai/example", "-"])
        for setting in (
            'model_provider="openrouter"', 'model_providers.openrouter.base_url="https://openrouter.ai/api/v1"',
            'model_providers.openrouter.env_key="OPENROUTER_API_KEY"',
            'model_providers.openrouter.wire_api="responses"', "model_providers.openrouter.supports_websockets=false",
            "model_providers.openrouter.request_max_retries=0", "model_providers.openrouter.stream_max_retries=0",
        ):
            self.assertIn(setting, argv)

    def test_saved_login_home_is_preserved_without_reading_credentials(self):
        del os.environ["CODEX_API_KEY"]
        login = self.root / ".codex"
        login.mkdir()
        self.fake(f"""
            import os, sys
            assert os.environ['CODEX_HOME'] == {str(login)!r}
            assert '--model' not in sys.argv
            print({transcript('{}')!r})
        """)
        self.assertEqual(self.run_cli(model=None), ("{}", None, None, None))
        for name in ("AGENTS.md", "AGENTS.override.md"):
            with self.subTest(name=name):
                marker = login / name
                marker.write_text("SYNTHETIC_PRIVATE_GLOBAL")
                with self.assertRaisesRegex(headless.OracleHeadlessError, "global instructions"):
                    self.run_cli(model=None)
                marker.unlink()

    def test_timeout_nonzero_and_oversize_events_hide_content(self):
        cases = [
            ("import time\ntime.sleep(20)", "timed out", 0.1),
            ('import sys\nprint("PRIVATE", file=sys.stderr)\nsys.exit(2)', "Codex failed", 5),
            ('import sys\nprint("401 PRIVATE", file=sys.stderr)\nsys.exit(1)', "authentication failed", 5),
            ('import sys\nprint("read-only file system: PRIVATE", file=sys.stderr)\nsys.exit(1)',
             "writable login home", 5),
            ('print("PRIVATE"*1000)', "size limit", 5),
            ('print("PRIVATE")', "invalid Codex", 5),
        ]
        for body, diagnostic, seconds in cases:
            with self.subTest(diagnostic=diagnostic), patch.object(headless, "_MAX_EVENTS", 100):
                self.fake(body)
                with self.assertRaisesRegex(headless.OracleHeadlessError, diagnostic) as raised:
                    self.run_cli(timeout=seconds)
                self.assertNotIn("PRIVATE", str(raised.exception))

    def test_rejects_tool_calls_and_unfinished_turns(self):
        good = transcript().splitlines()
        for event in ("command_execution", "mcp_tool_call", "web_search", "future_tool"):
            with self.subTest(event=event):
                extra = json.dumps({"type": "item.completed", "item": {"type": event}})
                self.fake(f"print({chr(10).join([*good[:2], extra, *good[2:]])!r})")
                with self.assertRaisesRegex(headless.OracleHeadlessError, "unexpected activity"):
                    self.run_cli()
        self.fake(f"print({chr(10).join(good[:-1])!r})")
        with self.assertRaisesRegex(headless.OracleHeadlessError, "no final answer"):
            self.run_cli()

    def test_validates_limits_and_usage(self):
        self.fake(f"print({transcript('[]', {'input_tokens': True})!r})")
        with self.assertRaisesRegex(headless.OracleHeadlessError, "invalid token usage"):
            self.run_cli()
        with self.assertRaisesRegex(headless.OracleHeadlessError, "invalid token usage"):
            headless._result(transcript("[]", {"output_tokens": 2**63}).encode())
        with self.assertRaises(headless.OracleHeadlessError):
            self.run_cli(model="--trust")
        with self.assertRaises(headless.OracleHeadlessError):
            self.run_cli(timeout=float("nan"))
        with patch.object(headless, "_MAX_PROMPT", 10), self.assertRaises(headless.OracleHeadlessError):
            headless.complete("界" * 8, model=None, timeout=5)
        with patch.object(headless, "_MAX_ANSWER", 2), self.assertRaises(headless.OracleHeadlessError):
            headless._result(transcript("界").encode())


@unittest.skipUnless(
    os.environ.get("ANNO_TEST_CODEX_LOOPBACK") == "1" and shutil.which("codex"),
    "set ANNO_TEST_CODEX_LOOPBACK=1 with a recent Codex CLI to inspect the wire request",
)
class RealCliTests(unittest.TestCase):
    def test_actual_cli_sends_only_the_supplied_document_and_no_tools(self):
        requests = []
        answer = '["synthetic"]'

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                response = {"id": "resp_synthetic", "object": "response", "model": "gpt-6.2-sol"}
                item = {"type": "message", "id": "msg_synthetic", "role": "assistant",
                        "status": "completed", "content": [{"type": "output_text", "text": answer}]}
                usage = {"input_tokens": 11, "output_tokens": 3, "total_tokens": 14}
                for event in (
                    {"type": "response.created", "response": dict(response, status="in_progress")},
                    {"type": "response.output_item.added", "output_index": 0,
                     "item": dict(item, content=[], status="in_progress")},
                    {"type": "response.output_item.done", "output_index": 0, "item": item},
                    {"type": "response.completed", "response": dict(response, output=[item], usage=usage,
                                                                    status="completed")},
                ):
                    self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")

        command = headless._command
        with http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server, tempfile.TemporaryDirectory() as tmp:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            self.addCleanup(thread.join)
            # Use an isolated synthetic login directory and only a loopback
            # provider. The production CLI command has no endpoint override.
            login = Path(tmp) / ".codex"
            login.mkdir()
            (login / "config.toml").write_text('developer_instructions = "SYNTHETIC_CONFIG_CANARY"')
            (Path(tmp) / "AGENTS.md").write_text("SYNTHETIC_PROJECT_CANARY")
            settings = (
                'model_provider="fixture"', 'model_providers.fixture.name="fixture"',
                f'model_providers.fixture.base_url="http://127.0.0.1:{server.server_port}/v1"',
                'model_providers.fixture.wire_api="responses"',
                "model_providers.fixture.supports_websockets=false",
                "model_providers.fixture.request_max_retries=0",
                "model_providers.fixture.stream_max_retries=0",
            )

            def local_command(*args):
                argv = command(*args)
                return [*argv[:-1], *(arg for s in settings for arg in ("-c", s)), "-"]

            binary = shutil.which("codex")
            assert binary
            with patch.dict(os.environ, {"PATH": str(Path(binary).parent), "HOME": tmp}, clear=True):
                try:
                    with patch.object(headless, "_command", local_command):
                        result = headless.complete("SYNTHETIC_DOCUMENT", model="gpt-6.2-sol", timeout=20)
                finally:
                    server.shutdown()
            self.assertEqual(result, (answer, None, 11, 3))
        self.assertEqual(len(requests), 1)
        request = requests[0]
        self.assertEqual(request.get("tools", []), [])
        model_input = json.dumps(request["input"])
        self.assertIn("SYNTHETIC_DOCUMENT", model_input)
        self.assertNotIn("SYNTHETIC_CONFIG_CANARY", model_input)
        self.assertNotIn("SYNTHETIC_PROJECT_CANARY", model_input)


if __name__ == "__main__":
    unittest.main()
