"""Replay a named QA case locally or record a confirmed observation for a matrix slot."""

from contextlib import closing
import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
import re
import sqlite3
import subprocess
import sys

import replay_crossdoc_case
import replay_crossdoc_native
import replay_reviewed_kb
import replay_typed_spans
import replay_wikigold

ROOT = Path(__file__).resolve().parents[2]
ADAPTERS = {
    "crossdoc_cases": replay_crossdoc_case, "crossdoc_native": replay_crossdoc_native,
    "reviewed_kb": replay_reviewed_kb, "typed_spans": replay_typed_spans, "wikigold": replay_wikigold,
}
DB_NAME = "replay-feedback.db"
MAX_WITNESSES = 512
MAX_RECEIPT_BYTES = 2048
IDENTITY_FIELDS = ("corpus", "scope", "family", "source_sha256", "oracle_sha256")
EVENT_FIELDS = {*IDENTITY_FIELDS, "case", "schema_version", "revision", "outcome",
                "signature", "producer_sha256", "replays"}
WITNESS_FIELDS = ("id", "corpus", "family", "case_name", "source_sha256",
                  "oracle_sha256", "latest", "unresolved", "generation", "last_slot")


def stamp(root):
    subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "."], cwd=root, check=True)
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def key(data):
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def database(cache):
    cache.mkdir(parents=True, exist_ok=True)
    return create_database(cache / DB_NAME)


def create_database(path):
    db = sqlite3.connect(path, timeout=30)
    db.row_factory = sqlite3.Row
    db.executescript("""
      CREATE TABLE IF NOT EXISTS observations (id TEXT PRIMARY KEY, receipt TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS witnesses (
        id TEXT PRIMARY KEY, corpus TEXT NOT NULL, family TEXT NOT NULL, case_name TEXT NOT NULL,
        source_sha256 TEXT NOT NULL, oracle_sha256 TEXT NOT NULL, latest TEXT NOT NULL,
        unresolved INTEGER NOT NULL, generation INTEGER NOT NULL DEFAULT 1, last_slot INTEGER NOT NULL DEFAULT 0);
      CREATE INDEX IF NOT EXISTS witness_families ON witnesses(corpus, family, last_slot);
    """)
    return db


def validate_signature(adapter, outcome, signature):
    if outcome not in ("pass", "fail") or not isinstance(signature, str):
        raise ValueError("invalid replay outcome or signature")
    if outcome == "fail" and not re.fullmatch(adapter.FAIL_SIGNATURE, signature):
        raise ValueError("unexpected replay failure signature")
    if outcome == "pass" and signature != adapter.PASS_SIGNATURE:
        raise ValueError("unexpected replay success signature")


def measured(adapter, root, case, *, confirm_failure=True):
    result = adapter.replay(root, case)
    if set(result) != {"outcome", "signature", "producer_sha256"}:
        raise ValueError("unexpected oracle result fields")
    validate_signature(adapter, result["outcome"], result["signature"])
    if not isinstance(result["producer_sha256"], str) or not re.fullmatch("[a-f0-9]{64}", result["producer_sha256"]):
        raise ValueError("missing producer digest")
    if confirm_failure and result["outcome"] == "fail":
        if not result["signature"] or adapter.replay(root, case) != result:
            raise ValueError("failure was not confirmed by two matching replays")
    return result


def describe(adapter, root, case):
    # --repo can point to another checkout. Never associate one adapter's code with
    # another checkout's revision, even when both contain a seed with the same name.
    module = getattr(adapter, "__file__", None)
    if module:
        local = Path(module)
        if local.read_bytes() != (root / "scripts/qa" / local.name).read_bytes():
            raise ValueError("executed adapter differs from the selected checkout")
    description = adapter.describe(root, case)
    if set(description) != {"family", "source_sha256", "oracle_sha256"}:
        raise ValueError("unexpected oracle description fields")
    if description["family"] != adapter.family_for_case(root, case):
        raise ValueError("unreviewed replay family")
    if not all(isinstance(description[k], str) and re.fullmatch("[a-f0-9]{64}", description[k])
               for k in ("source_sha256", "oracle_sha256")):
        raise ValueError("case and oracle must be pinned by SHA-256")
    return description


def replay_once(root, corpus, case):
    """Measure the working copy without stamping a revision or modifying the replay ledger."""
    adapter = ADAPTERS[corpus]
    try:
        before = describe(adapter, root, case)
    except FileNotFoundError as exc:
        cache = os.environ.get("ANNO_CACHE_DIR")
        missing_dataset = not cache or exc.filename == str(Path(cache) / "datasets/WikiGold.cache")
        if corpus == "wikigold" and missing_dataset:
            return 3, {
                "outcome": "unavailable",
                "detail": "Set ANNO_CACHE_DIR to an authorized cache containing datasets/WikiGold.cache.",
            }
        raise
    result = measured(adapter, root, case, confirm_failure=False)
    if describe(adapter, root, case) != before:
        raise ValueError("source or oracle changed during replay")
    return int(result["outcome"] == "fail"), {
        "corpus": corpus, "case": case, "source_sha256": before["source_sha256"],
        "oracle_sha256": before["oracle_sha256"], **result,
    }


def observe(root, cache, corpus, case, selected=None):
    adapter = ADAPTERS[corpus]
    revision = stamp(root)
    scope = adapter.SCOPE
    description = describe(adapter, root, case)
    identity = {"corpus": corpus, "scope": scope, **description}
    witness = key(identity)
    if selected is not None and selected["id"] != witness:
        raise ValueError("selected witness changed before execution")
    db = database(cache)
    try:
        row = db.execute(
            "SELECT generation, latest, unresolved FROM witnesses WHERE id=?", (witness,)
        ).fetchone()
        generation = row["generation"] if row else 0
        if selected is not None and (
            not row
            or not row["unresolved"]
            or generation != selected["generation"]
            or row["latest"] != selected["observation_id"]
        ):
            raise ValueError("selected observation is stale")
    finally:
        db.close()
    result = measured(adapter, root, case)
    if stamp(root) != revision or describe(adapter, root, case) != description:
        raise ValueError("source changed during replay")
    event = {
        "schema_version": 1,
        "revision": revision,
        "case": case,
        **identity,
        **result,
        "replays": 2 if result["outcome"] == "fail" else 1,
    }
    obs_id = key(event)
    db = database(cache)
    try:
        with db:
            db.execute("BEGIN IMMEDIATE")
            current = db.execute(
                "SELECT generation FROM witnesses WHERE id=?", (witness,)
            ).fetchone()
            if (current["generation"] if current else 0) != generation:
                raise ValueError("witness changed during replay")
            db.execute(
                "INSERT OR IGNORE INTO observations VALUES (?,?)", (obs_id, json.dumps(event))
            )
            db.execute(
                """INSERT INTO witnesses
              (id,corpus,family,case_name,source_sha256,oracle_sha256,latest,unresolved)
              VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET
              generation=witnesses.generation+(witnesses.latest != excluded.latest OR ?),
              latest=excluded.latest, unresolved=excluded.unresolved, case_name=excluded.case_name""",
                (
                    witness,
                    corpus,
                    description["family"],
                    case,
                    description["source_sha256"],
                    description["oracle_sha256"],
                    obs_id,
                    int(result["outcome"] == "fail"),
                    int(selected is not None),
                ),
            )
            if selected is not None:
                db.execute(
                    """UPDATE witnesses SET last_slot=(SELECT MAX(last_slot)+1 FROM witnesses)
                              WHERE id=?""",
                    (witness,),
                )
    finally:
        db.close()
    return {"observation_id": obs_id, **event}


def current_event(root, row):
    """Admit current, bounded public metadata from the local ledger or a transfer."""
    # The old WikiGold receipt can be large and can contain full gold-span
    # differences. Recognize that identity from its index, without parsing it.
    if row["corpus"] == "wikigold":
        old = {k: row[k] for k in ("corpus", "family", "source_sha256", "oracle_sha256")}
        if key({**old, "scope": "pinned_source_contract"}) == row["id"]:
            return None

    receipt = row["receipt"]
    if not isinstance(receipt, str) or len(receipt.encode()) > MAX_RECEIPT_BYTES:
        raise ValueError("missing or oversized replay receipt")
    event = json.loads(receipt)
    if not isinstance(event, dict) or set(event) != EVENT_FIELDS:
        raise ValueError("unexpected replay receipt fields")
    adapter = ADAPTERS.get(event["corpus"])
    if adapter is None or event["scope"] != adapter.SCOPE or event["schema_version"] != 1:
        return None

    failed = event["outcome"] == "fail"
    if (event["outcome"] not in ("pass", "fail") or type(event["schema_version"]) is not int
            or type(event["replays"]) is not int or event["replays"] != (2 if failed else 1)):
        raise ValueError("invalid replay confirmation")
    validate_signature(adapter, event["outcome"], event["signature"])
    if (not isinstance(event["revision"], str)
            or not re.fullmatch(r"[a-f0-9]{40}(?:[a-f0-9]{24})?", event["revision"])
            or any(not isinstance(event[k], str) or not re.fullmatch(r"[a-f0-9]{64}", event[k])
                   for k in ("source_sha256", "oracle_sha256", "producer_sha256"))):
        raise ValueError("invalid replay digest")

    identity = {k: event[k] for k in IDENTITY_FIELDS}
    if (key(event) != row["latest"] or key(identity) != row["id"]
            or row["case_name"] != event["case"] or not isinstance(event["case"], str)
            or any(row[k] != event[k] for k in ("corpus", "family", "source_sha256", "oracle_sha256"))
            or row["unresolved"] != int(failed)
            or type(row["generation"]) is not int or not 1 <= row["generation"] < 2**63 - 1
            or type(row["last_slot"]) is not int or not 0 <= row["last_slot"] < 2**63 - 1):
        raise ValueError("replay history has an inconsistent observation")
    try:
        family = adapter.family_for_case(root, event["case"])
    except replay_typed_spans.UnknownTypedCase:
        return None
    if family != event["family"]:
        raise ValueError("replay history has an inconsistent family")
    return event


def next_case(root, cache):
    path = cache / DB_NAME
    if not path.is_file():
        return None
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        rows = db.execute("""SELECT w.*, o.receipt FROM witnesses w
            JOIN observations o ON w.latest=o.id WHERE unresolved=1
            ORDER BY (SELECT MAX(last_slot) FROM witnesses c WHERE c.corpus=w.corpus),
                     (SELECT MAX(last_slot) FROM witnesses f
                      WHERE f.corpus=w.corpus AND f.family=w.family),
                     last_slot, w.corpus, w.family, w.case_name""").fetchall()
    finally:
        db.close()
    for row in rows:
        event = current_event(root, row)
        if event is None:
            continue
        adapter = ADAPTERS[event["corpus"]]
        try:
            pins = describe(adapter, root, row["case_name"])
        except (FileNotFoundError, ValueError):
            continue
        if all(row[k] == pins[k] for k in ("family", "source_sha256", "oracle_sha256")):
            return {
                "id": row["id"],
                "observation_id": row["latest"],
                "generation": row["generation"],
                **event,
            }
    return None


def backup(source, target, root=ROOT):
    """Build a fresh checkpoint of eligible current witnesses in one read transaction."""
    with closing(sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)) as src:
        src.row_factory = sqlite3.Row
        src.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 1 << 20)
        src.execute("PRAGMA trusted_schema=OFF")
        src.execute("BEGIN")
        tables = {row[0] for row in src.execute(
            "SELECT name FROM sqlite_schema WHERE type='table' AND name IN ('witnesses','observations')"
        )}
        if tables != {"observations", "witnesses"}:
            raise ValueError("missing replay ledger tables")
        # Bound memory before reading arbitrary receipt strings. Old WikiGold
        # scopes may contain raw span lists; keep them in the original local ledger.
        columns = ",".join("w." + k for k in WITNESS_FIELDS)
        rows = src.execute(f"""SELECT {columns}, CASE WHEN length(CAST(o.receipt AS BLOB)) <= ?
                               THEN o.receipt END AS receipt FROM witnesses w
                               LEFT JOIN observations o ON w.latest=o.id ORDER BY w.id LIMIT ?""",
                           (MAX_RECEIPT_BYTES, MAX_WITNESSES + 1))
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
            staged = Path(tmp) / DB_NAME
            skipped = 0
            with closing(create_database(staged)) as dest, dest:
                for idx, row in enumerate(rows):
                    if idx >= MAX_WITNESSES:
                        raise ValueError(f"replay checkpoint exceeds {MAX_WITNESSES} witnesses")
                    event = current_event(root, row)
                    if event is None:
                        skipped += 1
                        continue
                    dest.execute("INSERT INTO observations VALUES (?,?)",
                                 (row["latest"], json.dumps(event, sort_keys=True, separators=(",", ":"))))
                    dest.execute("INSERT INTO witnesses VALUES (?,?,?,?,?,?,?,?,?,?)",
                                 tuple(row[k] for k in WITNESS_FIELDS))
            os.link(staged, target)
            if skipped:
                print(f"Omitted {skipped} witnesses with a retired case or obsolete scope", file=sys.stderr)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, help="required for replay ledger commands")
    p.add_argument("--repo", type=Path, default=ROOT)
    sub = p.add_subparsers(dest="command", required=True)
    replay = sub.add_parser("replay", help="check a named case once offline, including uncommitted changes")
    replay.add_argument("corpus", choices=sorted(ADAPTERS))
    replay.add_argument("case")
    add = sub.add_parser("observe")
    add.add_argument("corpus", choices=sorted(ADAPTERS))
    add.add_argument("case")
    sub.add_parser("next")
    sub.add_parser("snapshot").add_argument("file", type=Path)
    sub.add_parser("restore").add_argument("file", type=Path)
    args = p.parse_args()
    if args.cache_dir is not None:
        os.environ.setdefault("ANNO_CACHE_DIR", str(args.cache_dir.resolve()))
    elif args.command != "replay":
        p.error(f"--cache-dir is required for {args.command}")

    if args.command == "replay":
        root = args.repo.resolve()
        os.environ.update(ANNO_NO_DOWNLOADS="1", ANNO_S3_CACHE="0")
        local_binary = root / "target/debug" / ("anno.exe" if os.name == "nt" else "anno")
        if local_binary.is_file():
            os.environ.setdefault("ANNO_BIN", str(local_binary))
        # Child diagnostics can quote source documents. This CLI branch emits
        # only its own JSON receipt, including on native tool errors.
        with open(os.devnull, "wb") as sink:
            os.dup2(sink.fileno(), 2)
        try:
            status, receipt = replay_once(root, args.corpus, args.case)
        except FileNotFoundError:
            status, receipt = 2, {
                "outcome": "error",
                "detail": "Check ANNO_BIN (or Cargo for reviewed_kb) and the checkout files.",
            }
        except Exception:
            # At the public CLI boundary every execution or protocol error is
            # distinct from a failed assertion, even for an unexpected tool error.
            status, receipt = 2, {
                "outcome": "error",
                "detail": "Check the case name, pinned inputs and the local executable.",
            }
        print(json.dumps(receipt, sort_keys=True))
        return status
    if args.command in ("snapshot", "restore"):
        ledger = args.cache_dir / DB_NAME
        if args.command == "snapshot":
            backup(ledger, args.file, args.repo.resolve())
        else:
            backup(args.file, ledger, args.repo.resolve())
        return
    root = args.repo.resolve()
    result = (
        observe(root, args.cache_dir, args.corpus, args.case)
        if args.command == "observe"
        else next_case(root, args.cache_dir)
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    sys.exit(main())
