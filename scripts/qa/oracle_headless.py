"""Run one blind QA prompt in a fresh, tool-free Codex CLI session (POSIX)."""

import argparse
import json
import os
import selectors
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path

import oracle_gateway
from round_trace import identifier


_MAX_PROMPT = 1024 * 1024
_MAX_EVENTS = 4 * 1024 * 1024
_MAX_ANSWER = 2 * 1024 * 1024

# Codex exec has no blanket --tools=none switch. The loopback test checks the
# actual Responses tool manifest. --strict-config fails on unsupported settings.
_DISABLED = (
    "apps", "browser_use", "code_mode", "code_mode_host", "code_mode_only",
    "computer_use", "context_management", "goals", "hooks", "image_generation",
    "memories", "multi_agent", "multi_agent_v2", "plugins", "request_permissions_tool",
    "shell_snapshot", "shell_tool", "skill_search", "sleep_tool", "standalone_web_search",
    "tool_suggest", "unified_exec", "view_image",
)
_CONFIG = (
    'web_search="disabled"', "agents.enabled=false", "mcp_servers={}",
    "tools.experimental_request_user_input.enabled=false", "tools.update_plan.enabled=false",
    "project_doc_max_bytes=0", "skills.include_instructions=false", "cloud.skills.enabled=false",
    "include_environment_context=false", 'history.persistence="none"',
)


_OPENROUTER_CONFIG = (
    'model_provider="openrouter"', 'model_providers.openrouter.name="OpenRouter"',
    'model_providers.openrouter.base_url="https://openrouter.ai/api/v1"',
    'model_providers.openrouter.env_key="OPENROUTER_API_KEY"',
    'model_providers.openrouter.wire_api="responses"',
    "model_providers.openrouter.supports_websockets=false",
    "model_providers.openrouter.request_max_retries=0", "model_providers.openrouter.stream_max_retries=0",
)


class OracleHeadlessError(ValueError):
    """A headless CLI failure safe to report outside the private round."""


def _environment(root, binary, openrouter):
    home = root / "home"
    home.mkdir()
    codex_home = home / ".codex"
    env = {"HOME": str(home), "PATH": os.pathsep.join((str(Path(binary).parent), os.defpath)),
           "TMPDIR": str(root)}
    # Only transport configuration is inherited alongside Codex authentication.
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy",
                 "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        if name in os.environ:
            env[name] = os.environ[name]
    if openrouter:
        env["OPENROUTER_API_KEY"] = oracle_gateway.api_key()
        codex_home.mkdir()
    elif key := os.environ.get("CODEX_API_KEY"):
        env["CODEX_API_KEY"] = key
        codex_home.mkdir()
    else:
        # Keep the CLI's own auth store, including refresh locking. Copying or
        # symlinking auth.json can strand the refreshed token in the temp dir.
        login_home = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex").absolute()
        if login_home.is_dir():
            codex_home = login_home
            # Codex includes these even with --ignore-user-config and a zero
            # project document budget. Never accept a contaminated prompt.
            if any(os.path.lexists(login_home / name) for name in ("AGENTS.md", "AGENTS.override.md")):
                raise OracleHeadlessError(
                    "Codex login home has global instructions; use CODEX_API_KEY or a dedicated CODEX_HOME"
                )
        else:
            codex_home.mkdir()
    env["CODEX_HOME"] = str(codex_home)
    return env


def _command(binary, root, model, openrouter):
    args = [
        binary, "--no-daemon", "--ask-for-approval", "never", "exec", "--strict-config",
        "--ignore-user-config", "--ignore-rules", "--ephemeral", "--skip-git-repo-check",
        "--sandbox", "read-only", "--json",
    ]
    for feature in _DISABLED:
        args.extend(("--disable", feature))
    settings = _CONFIG + _OPENROUTER_CONFIG if openrouter else _CONFIG
    for setting in (*settings, f"log_dir={json.dumps(str(root))}", f"sqlite_home={json.dumps(str(root))}"):
        args.extend(("-c", setting))
    if model is not None:
        args.extend(("--model", model))
    return [*args, "-"]


def _failure(events, diagnostics):
    # Return fixed messages only. Provider diagnostics can echo the prompt,
    # Authorization headers, account IDs, private paths or model output.
    text = (events[-65536:] + diagnostics).lower()
    for hints, message in (
        ((b"certificate", b"unknownissuer", b"invalid peer"), "Codex TLS certificate verification failed"),
        ((b"401", b"unauthorized", b"not logged in", b"refresh token", b"authentication failed"),
         "Codex authentication failed"),
        ((b"429", b"quota", b"usage limit", b"rate limit"), "Codex quota or rate limit reached"),
        ((b"403", b"forbidden", b"access denied"), "Codex provider denied access"),
        ((b"model_not_found", b"model is not supported", b"unknown model"), "Codex model unavailable"),
        ((b"unknown feature", b"unexpected argument", b"unknown field", b"error loading config",
          b"invalid value for", b"invalid type:"), "Codex configuration is unsupported by this CLI"),
        ((b"read-only file system", b"readonly database"),
         "Codex needs a writable login home. Use a writable CODEX_HOME or CODEX_API_KEY"),
        ((b"proxy", b"network", b"connect", b"error sending request", b"stream disconnected"),
         "Codex could not connect to its provider"),
    ):
        if any(hint in text for hint in hints):
            return OracleHeadlessError(message)
    return OracleHeadlessError("Codex failed. Check CLI compatibility and login")


def _run(args, cwd, env, source, timeout):
    deadline = time.monotonic() + timeout
    with subprocess.Popen(
        args, cwd=cwd, env=env, stdin=source, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, start_new_session=True,
    ) as process:
        assert process.stdout is not None and process.stderr is not None
        try:
            events = bytearray()
            diagnostics = bytearray()
            with selectors.DefaultSelector() as ready:
                ready.register(process.stdout, selectors.EVENT_READ)
                ready.register(process.stderr, selectors.EVENT_READ)
                while ready.get_map():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise OracleHeadlessError("Codex timed out")
                    for key, _ in ready.select(min(remaining, 24 * 60 * 60)):
                        chunk = os.read(key.fd, 65536)
                        if not chunk:
                            ready.unregister(key.fd)
                        elif key.fileobj is process.stderr:
                            diagnostics.extend(chunk)
                            del diagnostics[:-65536]
                        else:
                            events.extend(chunk)
                            if len(events) > _MAX_EVENTS:
                                raise OracleHeadlessError("Codex output exceeded the size limit")
            if process.wait(timeout=max(0, deadline - time.monotonic())):
                raise _failure(events, diagnostics)
            return events
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            raise


def _result(events):
    # Accept one completed turn. Refuse transcripts containing tools or unknown
    # events, including any new tool type introduced by a future CLI version.
    state = "initial"
    answer = None
    usage = None
    for line in events.splitlines():
        event = json.loads(line)
        kind = event["type"]
        if kind == "thread.started" and state == "initial":
            state = "thread"
        elif kind == "turn.started" and state == "thread":
            state = "turn"
        elif kind in ("item.started", "item.completed"):
            item = event["item"]
            if state == "thread" and item["type"] == "error":
                continue  # CLI startup diagnostic, e.g. a disabled code-mode host.
            if state != "turn" or item["type"] not in ("agent_message", "reasoning"):
                raise OracleHeadlessError("Codex returned unexpected activity in a blind session")
            if kind == "item.completed" and item["type"] == "agent_message":
                answer = item["text"]
        elif kind == "turn.completed" and state == "turn":
            usage = event.get("usage")
            state = "finished"
        else:
            raise OracleHeadlessError("Codex failed or returned an unexpected event")
    if state != "finished" or not isinstance(answer, str) or not answer.strip():
        raise OracleHeadlessError("Codex returned no final answer")
    if len(answer.encode("utf-8")) > _MAX_ANSWER:
        raise OracleHeadlessError("Codex answer exceeded the size limit")
    if usage is None:
        usage = {}
    if not isinstance(usage, dict):
        raise OracleHeadlessError("Codex returned invalid token usage")
    counts = (usage.get("input_tokens"), usage.get("output_tokens"))
    if any(v is not None and (type(v) is not int or not 0 <= v < 2**63) for v in counts):
        raise OracleHeadlessError("Codex returned invalid token usage")
    # exec JSONL reports usage but does not attest the effective provider model.
    return answer, None, *counts


def complete(
    prompt: str, *, model: str | None, timeout: float, openrouter: bool = False,
) -> tuple[str, str | None, int | None, int | None]:
    """Return the final message, observed model and observed input/output tokens."""
    if type(openrouter) is not bool or (openrouter and model is None):
        raise OracleHeadlessError("OpenRouter through Codex requires an explicit model and routing choice")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > _MAX_PROMPT:
        raise OracleHeadlessError("expected a nonempty prompt within the request size limit")
    if type(timeout) not in (int, float) or not 0 < timeout < 2**31:
        raise OracleHeadlessError("invalid Codex timeout")
    try:
        if model is not None:
            identifier(model)
        payload = prompt.encode("utf-8")
        if len(payload) > _MAX_PROMPT:
            raise OracleHeadlessError("Codex prompt exceeded the size limit")
        search_path = os.pathsep.join(
            p for p in os.get_exec_path() if os.path.isabs(p)
        )
        binary = shutil.which("codex", path=search_path)
        if not binary:
            raise OracleHeadlessError("Codex CLI is unavailable")
        # Keep the session outside the caller's project even when TMPDIR is
        # under the repository. TemporaryFile supplies private, unlinked stdin.
        with tempfile.TemporaryDirectory(prefix="anno-oracle-", dir="/tmp") as tmp:
            root = Path(tmp)
            cwd = root / "work"
            cwd.mkdir()
            env = _environment(root, binary, openrouter)
            with tempfile.TemporaryFile(dir=root) as source:
                source.write(payload)
                source.seek(0)
                events = _run(_command(binary, root, model, openrouter), cwd, env, source, timeout)
            return _result(events)
    except (OracleHeadlessError, oracle_gateway.OracleGatewayError):
        raise
    except subprocess.TimeoutExpired:
        raise OracleHeadlessError("Codex timed out") from None
    except (OSError, ValueError, KeyError, TypeError, RecursionError, argparse.ArgumentTypeError):
        raise OracleHeadlessError("invalid Codex configuration or response") from None
