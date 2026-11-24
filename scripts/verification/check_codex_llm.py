"""Run Anno extraction through a one-use, loopback Codex CLI test provider."""
import argparse
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
import threading

ROOT = Path(__file__).resolve().parents[2]
TEXT = "Ada left."
ENTITY = {"text": "Ada", "type": "person", "start": 0, "end": 3}


@contextmanager
def listener(path, respond):
    """Accept the test route only; retain handler errors for the test runner."""
    errors = []
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            if self.path != path:
                self.send_error(404)
                return
            try:
                size = int(self.headers["Content-Length"])
                if not 0 < size <= 4 * 1024 * 1024:
                    raise ValueError("unexpected request size")
                body = json.loads(self.rfile.read(size))
                content_type, result = respond(body)
                calls.append(True)
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(result)))
                self.end_headers()
                self.wfile.write(result)
            except Exception as exc:
                errors.append(str(exc))
                self.send_error(502, "test provider failed")

    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}", calls, errors
        finally:
            server.shutdown()
            thread.join()


def fixture_response(request):
    # Exercise Codex's Responses wire protocol without a paid inference call.
    if TEXT not in json.dumps(request["input"], ensure_ascii=False):
        raise ValueError("Codex request omitted the Anno fixture")
    answer = json.dumps([ENTITY])
    message = {"id": "msg_anno", "type": "message", "role": "assistant",
               "status": "completed", "content": [{"type": "output_text", "text": answer}]}
    result = {"id": "resp_anno", "object": "response", "model": "fixture",
              "status": "completed", "output": [message]}
    events = [
        {"type": "response.created", "response": {**result, "status": "in_progress", "output": []}},
        {"type": "response.output_item.added", "output_index": 0,
         "item": {**message, "status": "in_progress", "content": []}},
        {"type": "response.output_text.delta", "output_index": 0, "content_index": 0,
         "item_id": message["id"], "delta": answer},
        {"type": "response.output_item.done", "output_index": 0, "item": message},
        {"type": "response.completed", "response": result},
    ]
    wire = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
    return "text/event-stream", wire.encode()


def run(args, work, fixture_url=None):
    nonce = secrets.token_hex(16)
    result_file = work / "answer"
    cmd = [args.codex, "-a", "never", "exec", "--ephemeral", "--skip-git-repo-check",
           "--sandbox", "read-only", "--output-last-message", str(result_file)]
    if fixture_url:
        for setting in ['model_provider="anno_fixture"', 'model="fixture"',
                        'model_providers.anno_fixture.name="Anno fixture"',
                        f'model_providers.anno_fixture.base_url="{fixture_url}/v1"',
                        'model_providers.anno_fixture.wire_api="responses"']:
            cmd.extend(["-c", setting])
        cmd.append("--ignore-user-config")

    def complete(request):
        messages = request["messages"]
        prompt = "\n\n".join(m["content"] for m in messages)
        if TEXT not in prompt:
            raise ValueError("Anno request omitted the test document")
        prompt = "Answer this synthetic extraction request directly. Use no tools.\n" + prompt
        call = subprocess.run([*cmd, "-"], input=prompt, text=True, cwd=work,
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=90)
        if call.returncode:
            error = next((s for s in reversed(call.stderr.splitlines()) if s.strip()), "")
            raise RuntimeError(f"codex exec exited {call.returncode}: {error[:300]}")
        answer = result_file.read_text()
        if not answer.strip():
            raise ValueError("Codex returned an empty final message")
        data = {"choices": [{"message": {"role": "assistant", "content": answer}}]}
        return "application/json", json.dumps(data).encode()

    with listener(f"/{nonce}/v1/chat/completions", complete) as (host, calls, errors):
        env = os.environ.copy()
        for key in ["OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY"]:
            env[key] = ""  # Also suppress dotenv overrides in the child Anno process.
        env["OLLAMA_HOST"] = f"{host}/{nonce}"
        command = [args.anno] if args.anno else [
            "cargo", "run", "--quiet", "--manifest-path", str(ROOT / "Cargo.toml"),
            "-p", "anno-cli", "--no-default-features", "--features", "llm,extractor-html2text", "--bin", "anno", "--"]
        output = subprocess.run(command + ["extract", "--model", "universal-ner", "--text", TEXT,
                                "--format", "json"], env=env, cwd=work, capture_output=True, text=True)
        if errors:
            raise RuntimeError(errors[0])
        if output.returncode:
            raise RuntimeError(output.stderr[-800:])
        if len(calls) != 1:
            raise ValueError(f"expected one Anno provider request, got {len(calls)}")
        entities = json.loads(output.stdout)["entities"]
        if not any(e["text"] == "Ada" and (e["start"], e["end"]) == (0, 3)
                   and e["type"].lower() in {"per", "person"} for e in entities):
            raise ValueError(f"Anno omitted the expected person and offsets: {entities!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="use the CLI's configured model")
    parser.add_argument("--codex", default="codex", help="Codex CLI executable")
    parser.add_argument("--anno", help="prebuilt Anno binary with the llm feature")
    args = parser.parse_args()
    if args.anno:
        args.anno = str(Path(args.anno).resolve())
    with tempfile.TemporaryDirectory(prefix="anno-codex-") as tmp:
        if args.live:
            run(args, Path(tmp))
        else:
            with listener("/v1/responses", fixture_response) as (url, calls, errors):
                run(args, Path(tmp), url)
                if errors or len(calls) != 1:
                    raise ValueError(f"expected one successful fixture response: {errors}")
    print("Passed Anno → Codex CLI → " + ("configured model" if args.live else "loopback fixture"))


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        sys.exit(str(exc))
