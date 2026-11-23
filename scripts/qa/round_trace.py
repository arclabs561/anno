"""Record and inspect one local QA round. No model provider or exporter is required."""
import argparse
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import time
import uuid


REPO = Path(__file__).resolve().parents[2]
SPAN_TABLE = """CREATE TABLE spans (
    id TEXT PRIMARY KEY, parent TEXT REFERENCES spans, retry_of TEXT REFERENCES spans,
    kind TEXT NOT NULL, role TEXT NOT NULL, tool TEXT NOT NULL,
    started_ns INTEGER NOT NULL, ended_ns INTEGER, status TEXT NOT NULL DEFAULT 'open',
    prompt_sha256 TEXT, result_sha256 TEXT, return_code INTEGER, elapsed_ns INTEGER, recording_issue TEXT,
    model TEXT, input_tokens INTEGER, output_tokens INTEGER, cost_usd REAL
)"""


def connect(path):
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=rw", uri=True, timeout=15)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")
    return db


def identifier(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./:-]{0,127}", value):
        raise argparse.ArgumentTypeError("expected a short role, tool or model identifier")
    return value


def file_digest(path):
    if path is None:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def initialize(path, handoff):
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "-uno"], cwd=REPO))
    trace_id = uuid.uuid4().hex
    # Exclusive creation prevents restarting an existing round or overwriting another file.
    os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
    try:
        with closing(connect(path)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("CREATE TABLE trace (id TEXT PRIMARY KEY, revision TEXT, tracked_changes INTEGER)")
            db.execute(SPAN_TABLE)
            db.execute("INSERT INTO trace VALUES (?, ?, ?)", (trace_id, revision, dirty))
            db.execute("""INSERT INTO spans (id, kind, role, tool, started_ns, prompt_sha256)
                          VALUES (?, 'round', 'coordinator', 'qa-round', ?, ?)""",
                       (trace_id, time.time_ns(), hashlib.sha256(handoff.encode("utf-8")).hexdigest()))
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return trace_id


def span_id(db, value):
    return db.execute("SELECT id FROM trace").fetchone()[0] if value == "round" else value


def start(db, args, kind):
    prompt_digest = file_digest(args.prompt_file)
    with db:
        db.execute("BEGIN IMMEDIATE")
        parent = span_id(db, args.parent)
        state = db.execute("SELECT status FROM spans WHERE id = ?", (parent,)).fetchone()
        if state is None or state["status"] != "open":
            raise ValueError("parent must be an open span in this round")
        if args.retry_of:
            attempt = db.execute("""SELECT 1 FROM spans WHERE id=? AND parent=? AND kind=?
                                    AND role=? AND tool=? AND status IN ('error', 'interrupted')""",
                                 (args.retry_of, parent, kind, args.role, args.tool)).fetchone()
            if attempt is None:
                raise ValueError("retry must reference a failed attempt of the same operation")
        key = uuid.uuid4().hex
        db.execute("""INSERT INTO spans (id, parent, retry_of, kind, role, tool, started_ns, prompt_sha256)
                      VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                   (key, parent, args.retry_of, kind, args.role, args.tool, time.time_ns(), prompt_digest))
    return key


def finish(db, key, status, result=None, return_code=None, model=None,
           input_tokens=None, output_tokens=None, cost_usd=None, elapsed_ns=None, recording_issue=None):
    with db:
        db.execute("BEGIN IMMEDIATE")
        key = span_id(db, key)
        if db.execute("SELECT 1 FROM spans WHERE parent=? AND status='open'", (key,)).fetchone():
            raise ValueError("finish open child spans first")
        row = db.execute("""UPDATE spans SET ended_ns=?, status=?, result_sha256=?, return_code=?,
                           model=?, input_tokens=?, output_tokens=?, cost_usd=?, elapsed_ns=?, recording_issue=?
                           WHERE id=? AND status='open'""", (time.time_ns(), status, result, return_code,
                           model, input_tokens, output_tokens, cost_usd, elapsed_ns, recording_issue, key))
        if row.rowcount != 1:
            raise ValueError("span is unknown or already finished")


def run_command(db, args):
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        raise ValueError("supply the command after --")
    key = start(db, args, "command")
    print(f"QA command span {key}", file=sys.stderr, flush=True)
    started = time.monotonic_ns()
    try:
        code = subprocess.run(command).returncode
    except KeyboardInterrupt:
        finish(db, key, "interrupted", elapsed_ns=time.monotonic_ns() - started)
        return 130
    except OSError as exc:
        finish(db, key, "error", elapsed_ns=time.monotonic_ns() - started)  # No child exit status.
        print(f"Cannot launch QA command: {exc.strerror}", file=sys.stderr)
        return 127 if isinstance(exc, FileNotFoundError) else 126
    duration = time.monotonic_ns() - started
    exit_code = 128 - code if code < 0 else code
    if code in args.assertion_failure_code:
        status = "assertion_failed"
    elif code < 0:
        status = "interrupted"
    else:
        status = "ok" if code == 0 else "error"
    try:
        result = file_digest(args.result_file)
    except OSError as exc:
        finish(db, key, status, return_code=code, elapsed_ns=duration, recording_issue="unreadable_result")
        print(f"Cannot digest replay result: {exc.strerror}", file=sys.stderr)
        return exit_code or 2
    finish(db, key, status, result=result, return_code=code, elapsed_ns=duration)
    return exit_code


def utc(ns):
    return datetime.fromtimestamp(ns / 1_000_000_000, timezone.utc).isoformat()


def report(db):
    db.execute("BEGIN")  # One snapshot across the metadata and concurrent span updates.
    trace = db.execute("SELECT * FROM trace").fetchone()
    now = time.time_ns()
    spans = []
    for row in db.execute("SELECT * FROM spans ORDER BY started_ns, id"):
        span = dict(row)
        start_ns, end_ns = span.pop("started_ns"), span.pop("ended_ns")
        duration = span.pop("elapsed_ns")
        basis = "monotonic" if duration is not None else "wall_clock"
        if duration is None:
            duration = (end_ns if end_ns is not None else now) - start_ns
        span.update(started_at=utc(start_ns), ended_at=utc(end_ns) if end_ns is not None else None,
                    elapsed_seconds=duration / 1e9 if duration >= 0 else None, elapsed_basis=basis)
        spans.append(span)
    agents = [s for s in spans if s["kind"] == "agent"]
    statuses = Counter(s["status"] for s in agents)
    # Keep the historical "agents" field as an alias. Each start records one dispatch attempt.
    usage = {"agents": len(agents), "attempts": len(agents), "completed": statuses["ok"],
             "errors": statuses["error"], "interrupted": statuses["interrupted"],
             "open": statuses["open"], "assertion_failed": statuses["assertion_failed"],
             "known_model": sum(s["model"] is not None for s in agents),
             "known_tokens": sum(s["input_tokens"] is not None and s["output_tokens"] is not None
                                 for s in agents)}
    for field, known in (("cost_usd", "known_cost"), ("input_tokens", "known_input_tokens"),
                         ("output_tokens", "known_output_tokens")):
        values = [s[field] for s in agents if s[field] is not None]
        usage[known] = len(values)
        usage[f"reported_{field}"] = sum(values) if values else None
    return {"trace_id": trace["id"], "revision": trace["revision"],
            "tracked_changes": bool(trace["tracked_changes"]),
            "open_spans": [s["id"] for s in spans if s["status"] == "open"],
            "failed_spans": [s["id"] for s in spans if s["status"] in ("error", "interrupted")],
            "assertion_failed_spans": [s["id"] for s in spans if s["status"] == "assertion_failed"],
            "recording_issues": [s["id"] for s in spans if s["recording_issue"]],
            "agent_usage": usage,
            "spans": spans}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    commands = parser.add_subparsers(dest="action", required=True)
    for name in ("start", "run"):
        sub = commands.add_parser(name)
        sub.add_argument("--role", required=True, type=identifier)
        sub.add_argument("--tool", required=True, type=identifier)
        sub.add_argument("--parent", default="round")
        sub.add_argument("--retry-of")
        sub.add_argument("--prompt-file", "--input-file", type=Path)
        if name == "run":
            sub.add_argument("--result-file", type=Path)
            sub.add_argument("--assertion-failure-code", action="append", default=[], type=int, metavar="N",
                             help="classify child exit N (1..125) as a failed assertion (repeatable)")
            sub.add_argument("command", nargs=argparse.REMAINDER)
    end = commands.add_parser("end")
    end.add_argument("span")
    end.add_argument("--status", required=True, choices=("ok", "error", "interrupted", "assertion_failed"))
    end.add_argument("--result-file", type=Path)
    end.add_argument("--model", type=identifier)
    end.add_argument("--input-tokens", type=int)
    end.add_argument("--output-tokens", type=int)
    end.add_argument("--cost-usd", type=float)
    commands.add_parser("report")
    args = parser.parse_args()
    try:
        with closing(connect(args.database)) as db:
            if args.action == "start":
                print(start(db, args, "agent"))
            elif args.action == "end":
                for value in (args.input_tokens, args.output_tokens):
                    if value is not None and not 0 <= value < 2**63:
                        raise ValueError("token counts must be nonnegative 64-bit integers")
                if args.cost_usd is not None and (not math.isfinite(args.cost_usd) or args.cost_usd < 0):
                    raise ValueError("cost must be nonnegative and finite")
                finish(db, args.span, args.status, file_digest(args.result_file), model=args.model,
                       input_tokens=args.input_tokens, output_tokens=args.output_tokens, cost_usd=args.cost_usd)
            elif args.action == "run":
                if any(not 1 <= code <= 125 for code in args.assertion_failure_code):
                    raise ValueError("assertion failure codes must be in 1..125")
                return run_command(db, args)
            else:
                print(json.dumps(report(db), indent=2))
    except (OSError, ValueError, sqlite3.Error) as exc:
        parser.exit(2, f"Invalid QA trace operation: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
