"""Persist vetted reproductions for QA slots without changing native evaluator history."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import subprocess

import replay_reviewed_kb

ROOT = Path(__file__).resolve().parents[2]
ADAPTERS = {"reviewed_kb": replay_reviewed_kb}
DB_NAME = "replay-feedback.db"


def stamp(root):
    subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "."], cwd=root, check=True)
    return subprocess.check_output(["git","rev-parse","HEAD"], cwd=root, text=True).strip()


def key(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def database(cache):
    cache.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(cache / DB_NAME, timeout=30)
    db.row_factory = sqlite3.Row
    db.executescript("""
      CREATE TABLE IF NOT EXISTS observations (id TEXT PRIMARY KEY, receipt TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS witnesses (
        id TEXT PRIMARY KEY, corpus TEXT NOT NULL, family TEXT NOT NULL, case_name TEXT NOT NULL,
        source_sha256 TEXT NOT NULL, oracle_sha256 TEXT NOT NULL, latest TEXT NOT NULL,
        unresolved INTEGER NOT NULL, generation INTEGER NOT NULL DEFAULT 1, last_slot INTEGER NOT NULL DEFAULT 0);
    """)
    return db


def measured(adapter, root, case):
    result = adapter.replay(root, case)
    if result["outcome"] not in ("pass", "fail") or not isinstance(result["signature"], str):
        raise ValueError("invalid oracle outcome")
    if not re.fullmatch("[a-f0-9]{64}", result["producer_sha256"]):
        raise ValueError("missing producer digest")
    if result["outcome"] == "fail":
        if not result["signature"] or adapter.replay(root, case) != result:
            raise ValueError("failure was not confirmed by two matching replays")
    return result


def observe(root, cache, corpus, case, selected=None):
    adapter = ADAPTERS[corpus]
    revision = stamp(root)
    scope = adapter.SCOPE
    description = adapter.describe(root, case)
    if not all(re.fullmatch("[a-f0-9]{64}",description[k]) for k in ("source_sha256","oracle_sha256")):
        raise ValueError("case and oracle must be pinned by SHA-256")
    identity = {"corpus": corpus, "scope": scope, **description}
    witness = key(identity)
    if selected is not None and selected["id"] != witness:
        raise ValueError("selected witness changed before execution")
    db = database(cache)
    try:
        row = db.execute("SELECT generation, latest, unresolved FROM witnesses WHERE id=?", (witness,)).fetchone()
        generation = row["generation"] if row else 0
        if selected is not None and (not row or not row["unresolved"]
                or generation != selected["generation"] or row["latest"] != selected["observation_id"]):
            raise ValueError("selected observation is stale")
    finally:
        db.close()
    result = measured(adapter, root, case)
    if stamp(root) != revision or adapter.describe(root, case) != description:
        raise ValueError("source changed during replay")
    event = {"schema_version": 1, "revision": revision, "case": case, **identity, **result,
             "replays": 2 if result["outcome"] == "fail" else 1}
    obs_id = key(event)
    db = database(cache)
    try:
        with db:
            db.execute("BEGIN IMMEDIATE")
            current = db.execute("SELECT generation FROM witnesses WHERE id=?", (witness,)).fetchone()
            if (current["generation"] if current else 0) != generation:
                raise ValueError("witness changed during replay")
            db.execute("INSERT OR IGNORE INTO observations VALUES (?,?)", (obs_id, json.dumps(event)))
            db.execute("""INSERT INTO witnesses
              (id,corpus,family,case_name,source_sha256,oracle_sha256,latest,unresolved)
              VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET
              generation=witnesses.generation+(witnesses.latest != excluded.latest OR ?),
              latest=excluded.latest, unresolved=excluded.unresolved, case_name=excluded.case_name""",
                       (witness,corpus,description["family"],case,description["source_sha256"],
                        description["oracle_sha256"],obs_id,int(result["outcome"]=="fail"),int(selected is not None)))
            if selected is not None:
                db.execute("""UPDATE witnesses SET last_slot=(SELECT MAX(last_slot)+1 FROM witnesses)
                              WHERE id=?""", (witness,))
    finally:
        db.close()
    return {"observation_id":obs_id, **event}


def next_case(root, cache):
    path = cache / DB_NAME
    if not path.is_file():
        return None
    db = sqlite3.connect(path.resolve().as_uri()+"?mode=ro",uri=True)
    db.row_factory = sqlite3.Row
    try:
        rows = db.execute("""SELECT w.*, o.receipt FROM witnesses w
            JOIN observations o ON w.latest=o.id WHERE unresolved=1
            ORDER BY (SELECT MAX(last_slot) FROM witnesses f
                      WHERE f.corpus=w.corpus AND f.family=w.family),
                     last_slot, w.corpus, w.family, w.case_name""").fetchall()
    finally:
        db.close()
    for row in rows:
        event = json.loads(row["receipt"])
        identity = {k:event[k] for k in ("corpus","scope","family","source_sha256","oracle_sha256")}
        if (key(event) != row["latest"] or key(identity) != row["id"] or event["case"] != row["case_name"]
                or event["outcome"] != "fail" or event["replays"] != 2
                or any(row[k] != event[k] for k in ("corpus","family","source_sha256","oracle_sha256"))):
            raise ValueError("replay history has an inconsistent observation")
        adapter = ADAPTERS.get(row["corpus"])
        if adapter is None:
            continue
        try:
            pins = adapter.describe(root, row["case_name"])
        except (FileNotFoundError, ValueError):
            continue
        if all(row[k] == pins[k] for k in ("family","source_sha256","oracle_sha256")):
            return {"id": row["id"], "observation_id":row["latest"],"generation":row["generation"], **event}
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir",required=True,type=Path)
    p.add_argument("--repo",type=Path,default=ROOT)
    sub = p.add_subparsers(dest="command", required=True)
    add = sub.add_parser("observe")
    add.add_argument("corpus",choices=sorted(ADAPTERS))
    add.add_argument("case")
    sub.add_parser("next")
    args = p.parse_args()
    root = args.repo.resolve()
    result = (observe(root,args.cache_dir,args.corpus,args.case) if args.command=="observe"
              else next_case(root,args.cache_dir))
    print(json.dumps(result,sort_keys=True))


if __name__ == "__main__":
    main()
