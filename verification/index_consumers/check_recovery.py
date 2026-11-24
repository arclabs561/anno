# SPDX-License-Identifier: MIT OR Apache-2.0
"""Replay index_publish_snapshots TSV from stdin into temporary SQLite files."""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile


STREAM = "synthetic-bridge"


class Rejected(Exception):
    """A publication cannot be applied at the current checkpoint."""


def load(source):
    snapshots = {0: {}}
    current = 0
    for line in source.read_text().splitlines():
        tag, *fields = line.split("\t")
        if tag == "snapshot":
            current = int(fields[0])
            assert current == len(snapshots)
            snapshots[current] = {}
        else:
            assert tag == "posting" and current > 0
            doc, edition, signal, identity = fields
            key = (doc, int(edition), int(signal))
            assert key not in snapshots[current]
            snapshots[current][key] = int(identity)
    return snapshots


def digest(rows):
    ordered = sorted((*key, value) for key, value in rows.items())
    payload = json.dumps(ordered, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def open_db(path):
    db = sqlite3.connect(path, isolation_level=None)
    db.execute(
        """CREATE TABLE IF NOT EXISTS postings (
               doc TEXT, edition INTEGER, signal INTEGER, owner INTEGER,
               PRIMARY KEY (doc, edition, signal)
           )"""
    )
    db.execute(
        """CREATE TABLE IF NOT EXISTS checkpoint (
               stream TEXT PRIMARY KEY, seq INTEGER, digest TEXT
           )"""
    )
    db.execute(
        "INSERT OR IGNORE INTO checkpoint VALUES (?, ?, ?)",
        (STREAM, 0, digest({})),
    )
    return db


def checkpoint(db):
    return db.execute(
        "SELECT seq, digest FROM checkpoint WHERE stream = ?", (STREAM,)
    ).fetchone()


def postings(db):
    return {
        (doc, edition, signal): owner
        for doc, edition, signal, owner in db.execute(
            "SELECT doc, edition, signal, owner FROM postings"
        )
    }


def expect_snapshot(db, seq, snapshots):
    assert checkpoint(db) == (seq, digest(snapshots[seq]))
    assert postings(db) == snapshots[seq]


def publish(db, snapshots, seq, *, reset=False, atomic=True, crash=None):
    """Derive a delta from native snapshots, or authoritatively replace all rows."""
    new = snapshots[seq]
    signature = digest(new)
    if atomic:
        db.execute("BEGIN IMMEDIATE")
    try:
        current, stored = checkpoint(db)
        if current == seq and stored == signature:
            if atomic:
                db.execute("COMMIT")
            return "duplicate"
        if seq <= current:
            raise Rejected("stale revision or conflicting duplicate")
        if not reset and (
            seq != current + 1 or stored != digest(snapshots[seq - 1])
        ):
            raise Rejected("missing or divergent base")

        if reset:
            db.execute("DELETE FROM postings")
            additions = new
        else:
            old = snapshots[seq - 1]
            removed = [key for key, val in old.items() if new.get(key) != val]
            db.executemany(
                "DELETE FROM postings WHERE doc = ? AND edition = ? AND signal = ?",
                removed,
            )
            additions = {key: val for key, val in new.items() if old.get(key) != val}
        if crash == "after_retract":
            os._exit(73)
        db.executemany(
            "INSERT INTO postings VALUES (?, ?, ?, ?)",
            ((*key, val) for key, val in additions.items()),
        )
        if crash == "after_add":
            os._exit(73)
        db.execute(
            "UPDATE checkpoint SET seq = ?, digest = ? WHERE stream = ?",
            (seq, signature, STREAM),
        )
        if atomic:
            db.execute("COMMIT")
        if crash == "after_commit":
            os._exit(73)  # Commit succeeded and its acknowledgement was lost.
        return "published"
    except Exception:
        if db.in_transaction:
            db.execute("ROLLBACK")
        raise


def owners(rows):
    return ",".join(f"{key[0]}={owner}" for key, owner in sorted(rows.items())) or "empty"


def check_split(snapshots):
    first, split, joined, final = (snapshots[i] for i in range(1, 5))
    assert {key[0] for key in first} == {"a", "bridge", "c"}
    assert {key[0] for key in split} == {"a", "c"}
    untouched = {key: val for key, val in first.items() if key[0] != "bridge"}
    assert split.keys() == untouched.keys()
    assert len(set(first.values())) == 1 and len(set(split.values())) == 2
    by_doc = {doc: val for (doc, _, _), val in joined.items()}
    assert by_doc["a"] != by_doc["c"] == by_doc["d"]
    assert not final
    print("Source deletion alone:", owners(untouched), "| resolved split:", owners(split))


def crash_trial(tmp, source, snapshots, atomic, phase):
    path = tmp / f"writer-{int(atomic)}-{phase}.sqlite"
    db = open_db(path)
    publish(db, snapshots, 1)
    db.close()
    # The child hard-exits without closing its database or running a rollback handler.
    writer = subprocess.run(
        [sys.executable, __file__, "crash", str(source), str(path), str(int(atomic)), phase],
        check=False,
    )
    assert writer.returncode == 73

    db = open_db(path)
    visible_revision = 2 if phase == "after_commit" else 1
    if atomic:
        expect_snapshot(db, visible_revision, snapshots)
    else:
        # Rows whose owner was preserved survive even if another posting is retracted.
        retained = {
            key: val for key, val in snapshots[1].items() if snapshots[2].get(key) == val
        }
        expected = retained if phase == "after_retract" else snapshots[2]
        assert checkpoint(db) == (1, digest(snapshots[1]))
        assert postings(db) == expected and expected != snapshots[1]
    label = "transaction" if atomic else "autocommit"
    print(f"{label} {phase}: checkpoint {visible_revision}; {owners(postings(db))}")
    if atomic:
        if phase != "after_commit":
            assert publish(db, snapshots, 2) == "published"
        expect_snapshot(db, 2, snapshots)
        writes = db.total_changes
        assert publish(db, snapshots, 2) == "duplicate"
        assert db.total_changes == writes
    db.close()


def check_gap(tmp, snapshots):
    db = open_db(tmp / "gap.sqlite")
    publish(db, snapshots, 1)
    try:
        publish(db, snapshots, 3)
    except Rejected:
        expect_snapshot(db, 1, snapshots)
    else:
        raise AssertionError("accepted a delta with a missing predecessor")
    publish(db, snapshots, 3, reset=True)
    expect_snapshot(db, 3, snapshots)
    print("Recovered gap with authoritative revision 3:", owners(postings(db)))
    publish(db, snapshots, 4, reset=True)
    expect_snapshot(db, 4, snapshots)
    try:
        publish(db, snapshots, 2, reset=True)
    except Rejected:
        expect_snapshot(db, 4, snapshots)
    else:
        raise AssertionError("older snapshot resurrected retired postings")
    print("Empty authoritative revision 4 survived delayed revision 2")
    db.close()


def main():
    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        source = tmp / "native.tsv"
        source.write_bytes(sys.stdin.buffer.read())
        snapshots = load(source)
        check_split(snapshots)
        for phase in ("after_retract", "after_add"):
            crash_trial(tmp, source, snapshots, False, phase)
            crash_trial(tmp, source, snapshots, True, phase)
        crash_trial(tmp, source, snapshots, True, "after_commit")
        check_gap(tmp, snapshots)


if __name__ == "__main__":
    if len(sys.argv) == 6 and sys.argv[1] == "crash":
        _, _, source, path, mode, phase = sys.argv
        publish(open_db(path), load(Path(source)), 2, atomic=bool(int(mode)), crash=phase)
    else:
        main()
