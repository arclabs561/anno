"""Ask an independent reviewer for a JSON proposal, then replay it with the relevant QA checker."""
import argparse
from contextlib import closing, contextmanager, nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import sys
import time

import round_trace


MAX_PROMPT = 128 * 1024
MAX_REPLY = 1024 * 1024


def read_text(stream, limit):
    data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("input exceeds the byte limit")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("input must be UTF-8") from None


def make_prompt(path):
    with path.open("rb") as source:
        task = read_text(source, MAX_PROMPT)
    if not task.strip():
        raise ValueError("empty oracle task")
    return task


def answer_bytes(reply, shape):
    def unique(pairs):
        obj = dict(pairs)
        if len(obj) != len(pairs):
            raise ValueError("duplicate JSON field")
        return obj

    def invalid_constant(_):
        raise ValueError("nonfinite JSON number")

    try:
        encoded = reply.encode("utf-8")
        if len(encoded) > MAX_REPLY:
            raise ValueError("reply exceeds the byte limit")
        # Chat clients may wrap an entire JSON payload in one Markdown block.
        # Never extract snippets from prose or concatenate separate proposals.
        fenced = re.fullmatch(rb"[ \t\r\n]*```json[ \t]*\r?\n(.*?)\r?\n```[ \t\r\n]*",
                              encoded, flags=re.DOTALL)
        if fenced:
            encoded = fenced.group(1)
        value = json.loads(encoded.decode("utf-8"), object_pairs_hook=unique, parse_constant=invalid_constant)
        if not isinstance(value, dict if shape == "object" else list):
            raise ValueError("wrong root type")
        # Reject exponent overflow and escaped lone surrogates, then preserve the original bytes.
        json.dumps(value, allow_nan=False, ensure_ascii=False).encode("utf-8")
        return encoded
    except (ValueError, RecursionError):
        raise ValueError(f"expected one complete JSON {shape} with unique fields and finite numbers") from None


@contextmanager
def reserve_output(path):
    # Peers sharing a checkout may dispatch the same answer concurrently. Reserve before billing.
    if os.path.lexists(path):
        raise FileExistsError("answer path already exists")
    pending = path.with_name("." + path.name + ".pending")
    with os.fdopen(os.open(pending, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), "wb") as temp:
        try:
            if os.path.lexists(path):
                raise FileExistsError("answer path already exists")
            yield temp
            temp.flush()
            os.link(pending, path)  # Atomic publication, even if a caller creates the path meanwhile.
        finally:
            pending.unlink(missing_ok=True)


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="via", required=True)
    for name in ("prompt", "accept", "openrouter", "codex"):
        sub = commands.add_parser(name, help={
            "prompt": "write the exact reviewer prompt to stdout",
            "accept": "read a harness reply on stdin and freeze its answer",
            "openrouter": "dispatch one chat completion to OpenRouter",
            "codex": "dispatch a fresh, non-interactive Codex CLI session",
        }[name])
        sub.add_argument("task", type=Path, help="UTF-8 file containing only the contract and approved inputs")
        if name != "prompt":
            sub.add_argument("output", type=Path, help="new private answer JSON file")
            sub.add_argument("--shape", choices=("object", "array"), default="object", help="required JSON root")
        if name in ("codex", "openrouter"):
            sub.add_argument("--model", required=name == "openrouter", type=round_trace.identifier)
            sub.add_argument("--timeout", type=float, default=90, metavar="SECONDS")
            sub.add_argument("--trace", type=Path, help="existing qa-round trace database")
            sub.add_argument("--retry-of", help="failed dispatch span in that trace")
        if name == "codex":
            sub.add_argument("--openrouter", action="store_true", help="use OpenRouter as the Codex model provider")
        if name == "openrouter":
            sub.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args(argv)
    if getattr(args, "openrouter", False) and not args.model:
        parser.error("codex --openrouter requires --model")
    if getattr(args, "retry_of", None) and not args.trace:
        parser.error("--retry-of requires --trace")
    if hasattr(args, "timeout") and not (math.isfinite(args.timeout) and args.timeout > 0):
        parser.error("--timeout must be positive and finite")
    return args


def dispatch(args, prompt):
    if args.via == "accept":
        return read_text(sys.stdin.buffer, MAX_REPLY), None, None, None, None
    if args.via == "openrouter":
        import oracle_gateway
        return oracle_gateway.complete(prompt, model=args.model, max_tokens=args.max_tokens, timeout=args.timeout,
                                       shape=args.shape)
    import oracle_headless
    return (*oracle_headless.complete(prompt, model=args.model, timeout=args.timeout, openrouter=args.openrouter), None)


def run(args, prompt):
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    via = "codex-openrouter" if getattr(args, "openrouter", False) else args.via
    trace = getattr(args, "trace", None)
    with closing(round_trace.connect(trace)) if trace else nullcontext() as db:
        span = None
        status = "error"
        result = model = input_tokens = output_tokens = cost_usd = None
        started = time.monotonic_ns()
        try:
            with reserve_output(args.output) as output:
                if db is not None:
                    attempt = argparse.Namespace(parent="round", retry_of=args.retry_of,
                                                 role="oracle", tool=via)
                    span = round_trace.start(db, attempt, "agent", prompt_digest=digest)
                    print(f"QA oracle span {span}", file=sys.stderr, flush=True)
                reply, model, input_tokens, output_tokens, cost_usd = dispatch(args, prompt)
                answer = answer_bytes(reply, args.shape)
                output.write(answer)
            result = hashlib.sha256(answer).hexdigest()
            status = "ok"
        except KeyboardInterrupt:
            status = "interrupted"
            raise
        finally:
            if span:
                round_trace.finish(db, span, status, result=result, model=model, input_tokens=input_tokens,
                                   output_tokens=output_tokens, cost_usd=cost_usd,
                                   elapsed_ns=time.monotonic_ns() - started)
    return {"via": via, "prompt_sha256": digest, "answer_sha256": result,
            "reply_sha256": hashlib.sha256(reply.encode("utf-8")).hexdigest(),
            "requested_model": getattr(args, "model", None), "model": model,
            "input_tokens": input_tokens, "output_tokens": output_tokens, "cost_usd": cost_usd}


def main(argv=None):
    args = parse_args(argv)
    try:
        prompt = make_prompt(args.task)
        if args.via == "prompt":
            sys.stdout.buffer.write(prompt.encode("utf-8"))
        else:
            print(json.dumps(run(args, prompt), allow_nan=False))
    except KeyboardInterrupt:
        return 130
    except (ValueError, OSError, RuntimeError, sqlite3.Error) as exc:
        print(f"QA oracle failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
