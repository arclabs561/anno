#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Replay authored reader cases in SQLite through the native grounded publication CLI."""
import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE source(id TEXT PRIMARY KEY, edition TEXT NOT NULL, sha256 TEXT NOT NULL);
CREATE TABLE quote(source TEXT, start INTEGER, end INTEGER, surface TEXT NOT NULL,
                   PRIMARY KEY(source, start, end));
CREATE TABLE edition(rev INTEGER PRIMARY KEY, body TEXT NOT NULL);
CREATE TABLE policy(rev INTEGER PRIMARY KEY);
CREATE TABLE hidden(policy INTEGER, assertion TEXT, PRIMARY KEY(policy, assertion));
CREATE TABLE operation(id TEXT, rev INTEGER, kind TEXT, issuer TEXT, source TEXT REFERENCES source(id),
                       target TEXT, role TEXT, previous TEXT, start INTEGER, end INTEGER,
                       PRIMARY KEY(id, rev));
CREATE INDEX operations_at ON operation(rev, id);
"""

# First choose the last version of each assertion at the requested factual edition.
# A withdrawn or hidden latest version cannot fall back to an older visible version.
LIVE = """
WITH head AS (
    SELECT id, MAX(rev) AS rev FROM operation WHERE rev <= :at GROUP BY id
), live AS (
    SELECT op.* FROM operation op JOIN head USING(id,rev)
    WHERE op.kind IS NOT NULL AND (:staff OR NOT EXISTS (
        SELECT 1 FROM hidden h WHERE h.assertion=op.id AND h.policy=:policy
    ))
), selected AS (
    SELECT * FROM live WHERE (:source IS NULL OR source=:source)
                       AND (:issuer IS NULL OR issuer=:issuer)
)
"""
SELECT = {
    "printed": """SELECT q.surface FROM selected s JOIN quote q USING(source,start,end)
                  WHERE s.kind='printed'""",
    "credit": """SELECT target FROM selected WHERE kind='credit' AND role=:role""",
    "evidence": """SELECT id FROM selected WHERE kind='credit' AND role=:role AND target=:target""",
    "reference": """SELECT COALESCE(r.target, s.target) FROM selected s LEFT JOIN live r
                    ON r.kind='redirect' AND r.previous=s.target WHERE s.kind='reference'""",
}


def call_json(args, payload=None):
    proc = subprocess.run(args, input=None if payload is None else json.dumps(payload),
                          capture_output=True, text=True, timeout=90)
    if proc.returncode:
        raise ValueError(f"{args[0]} exited {proc.returncode}: {proc.stderr.strip()}")
    return json.loads(proc.stdout)


def load_sources(db, case, anno, excerpt_ranges=()):
    # Only literal evidence gets a synthetic signal. A name or paragraph in a
    # source never becomes a credited person merely by being passed to the CLI.
    ranges = {}
    for source, start, end in excerpt_ranges:
        ranges.setdefault(source, set()).add((start, end))
    for step in case["steps"]:
        for claim in step.get("put", []):
            if claim["kind"] == "printed":
                ranges.setdefault(claim["source"], set()).add((claim["start"], claim["end"]))
    for source in case["sources"]:
        doc, edition, text = source["id"], source["edition"], source["text"]
        spans = sorted(ranges.get(doc, ()))
        signals = []
        for i, (start, end) in enumerate(spans):
            if not 0 <= start < end <= len(text):
                raise ValueError(f"invalid printed range {start}:{end} in {doc}")
            signals.append({"id": i, "location": {"Text": {"start": start, "end": end}},
                            "surface": text[start:end], "label": "PRINTED", "confidence": 1.0,
                            "modality": "Symbolic", "negated": False})
        publication = call_json([str(anno), "query", "publication", "-"], {
            "stream": case["id"], "sequence": 1, "source_revision": edition,
            "document": {"id": doc, "text": text, "signals": signals, "tracks": {}, "identities": {}},
        })
        digest = hashlib.sha256(text.encode()).hexdigest()
        if (publication["source_sha256"], publication["source_revision"]) != (digest, edition):
            raise ValueError(f"source pin differs: {case['id']}/{doc}")
        db.execute("INSERT INTO source VALUES (?,?,?)", (doc, edition, digest))
        actual = {}
        for row in publication["rows"]:
            bounds = row["location"]["Text"]
            key = bounds["start"], bounds["end"]
            if (row["document"], row["source_sha256"], row["source_revision"]) != (doc, digest, edition):
                raise ValueError(f"invalid native anchor for {doc}")
            actual[key] = row["surface"]
        if set(actual) != set(spans) or len(actual) != len(publication["rows"]):
            raise ValueError(f"native anchors did not cover the requested ranges in {doc}")
        db.executemany("INSERT INTO quote VALUES (?,?,?,?)",
                       ((doc, start, end, surface) for (start, end), surface in actual.items()))


def apply_step(db, step):
    rev = step["revision"]
    body = json.dumps(step, sort_keys=True, separators=(",", ":"))
    with db:
        seen = db.execute("SELECT body FROM edition WHERE rev=?", (rev,)).fetchone()
        if seen:
            if seen[0] != body:
                raise ValueError(f"conflicting delivery at edition {rev}")
            return False
        highest, = db.execute("SELECT MAX(rev) FROM edition").fetchone()
        expected = 1 if highest is None else highest + 1
        if rev != expected:
            raise ValueError(f"incomplete factual history: expected edition {expected}, received {rev}")
        db.execute("INSERT INTO edition VALUES (?,?)", (rev, body))
        updates = dict.fromkeys(step.get("withdraw", []))
        for c in step.get("put", []):
            kind = c.get("kind")
            if kind not in ("printed", "credit", "reference", "redirect"):
                raise ValueError("unknown assertion kind")
            required = ["id", "issuer", "source", "target"]
            if kind == "credit":
                required.append("role")
            elif kind == "redirect":
                required.append("from")
            for field in required:
                value = c.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"assertion requires a nonempty {field}")
            if not db.execute("SELECT 1 FROM source WHERE id=?", (c["source"],)).fetchone():
                raise ValueError(f"unknown assertion source {c['source']}")
            if kind == "printed" and not db.execute(
                "SELECT 1 FROM quote WHERE source=? AND start=? AND end=?",
                (c["source"], c.get("start"), c.get("end")),
            ).fetchone():
                raise ValueError("printed assertion requires a verified source range")
            updates[c["id"]] = c
        for key, value in updates.items():
            c = value or {}
            db.execute("INSERT INTO operation VALUES (?,?,?,?,?,?,?,?,?,?)",
                       (key, rev, c.get("kind"), c.get("issuer"), c.get("source"), c.get("target"),
                        c.get("role"), c.get("from"), c.get("start"), c.get("end")))
    return True


def answer(db, query):
    if query["audience"] not in ("public", "staff"):
        raise ValueError(f"unknown audience {query['audience']}")
    for table, value in [("edition", query["at"]), ("policy", query["policy"])]:
        if not db.execute(f"SELECT 1 FROM {table} WHERE rev=?", (value,)).fetchone():
            raise ValueError(f"unknown {table} {value}")
    params = {**query, "source": query.get("source"), "issuer": query.get("issuer"),
              "staff": query["audience"] == "staff"}
    ambiguous = db.execute(LIVE + """
        SELECT previous FROM live WHERE kind='redirect' GROUP BY previous HAVING COUNT(*)>1
    """, params).fetchone()
    if ambiguous:
        raise ValueError(f"multiple visible redirects from {ambiguous[0]}")
    rows = db.execute(LIVE + SELECT[query["kind"]], params)
    return sorted({row[0] for row in rows})


def install_case(db, case, anno, excerpt_ranges=()):
    db.executescript(SCHEMA)
    load_sources(db, case, anno, excerpt_ranges)
    for p in case["policies"]:
        rev = p["revision"]
        db.execute("INSERT INTO policy VALUES (?)", (rev,))
        db.executemany("INSERT INTO hidden VALUES (?,?)", ((rev, key) for key in p["hidden"]))
    # A complete fixture is a retained history. Its array order is independent
    # of numeric revision order; a single incoming patch still needs its base.
    for step in sorted(case["steps"], key=lambda s: s["revision"]):
        if not apply_step(db, step):
            raise ValueError("duplicate declared edition")
    writes = db.total_changes
    for step in reversed(case["steps"]):
        if apply_step(db, step):
            raise AssertionError("redelivery changed history")
    if db.total_changes != writes:
        raise AssertionError("redelivery wrote rows")


def read_case(case, anno, path):
    with closing(sqlite3.connect(path)) as db, db:
        install_case(db, case, anno)
    # Open a fresh reader. Its answers must survive closing the writer.
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as reader:
        results = {}
        for q in case["queries"]:
            if q["id"] in results:
                raise ValueError(f"duplicate query {q['id']}")
            results[q["id"]] = answer(reader, q)
        return results


def read_suite(cases, anno):
    if cases["version"] != 1:
        raise ValueError("unsupported case format")
    with tempfile.TemporaryDirectory() as directory:
        answers = {}
        for i, case in enumerate(cases["cases"]):
            if case["id"] in answers:
                raise ValueError(f"duplicate case {case['id']}")
            answers[case["id"]] = read_case(case, anno, Path(directory) / f"case-{i}.sqlite")
    return {"version": 1, "answers": answers}


def compare(reader, actual, expected):
    if actual == expected:
        return
    for case, questions in expected["answers"].items():
        for id_, results in questions.items():
            observed = actual.get("answers", {}).get(case, {}).get(id_)
            if observed != results:
                raise ValueError(f"{reader} {case}/{id_}: expected {results!r}, got {observed!r}")
    raise ValueError(f"{reader} returned an unexpected answer envelope or question")


def check_transport():
    # These are patches with contiguous fixture-local revisions. A complete snapshot
    # may skip revisions in anno's general publication protocol.
    with closing(sqlite3.connect(":memory:")) as db:
        db.executescript(SCHEMA)
        db.execute("INSERT INTO source VALUES ('page','first','test')")
        query = {"at": 1, "audience": "public", "policy": 1, "kind": "credit", "role": "writing"}
        db.execute("INSERT INTO policy VALUES (1)")
        claim = {"id": "wrote", "kind": "credit", "role": "writing", "issuer": "press",
                 "source": "page", "target": "author"}
        first = {"revision": 1, "put": [claim]}
        second = {"revision": 2, "withdraw": ["wrote"]}
        third = {"revision": 3, "put": []}
        apply_step(db, first)
        invalid = [(third, "incomplete factual history"),
                   ({"revision": 1, "put": []}, "conflicting delivery")]
        for malformed, reason in [
            ({k: v for k, v in claim.items() if k != "issuer"}, "nonempty issuer"),
            ({**claim, "issuer": " "}, "nonempty issuer"),
            ({**claim, "source": "unseen"}, "unknown assertion source"),
        ]:
            invalid.append(({"revision": 2, "withdraw": ["wrote"], "put": [malformed]}, reason))
        for bad, reason in invalid:
            try:
                apply_step(db, bad)
            except ValueError as e:
                assert reason in str(e)
            else:
                raise AssertionError(f"accepted {reason}")
            assert db.execute("SELECT rev FROM edition").fetchall() == [(1,)]
            assert answer(db, query) == ["author"]
            try:
                answer(db, {**query, "at": 3})
            except ValueError as e:
                assert "unknown edition" in str(e)
            else:
                raise AssertionError("exposed an incomplete factual edition")
        apply_step(db, second)
        apply_step(db, third)
        assert answer(db, {**query, "at": 3}) == []
        assert not apply_step(db, first)
        assert answer(db, {**query, "at": 3}) == []
    print("Missing predecessor, conflicting duplicate, and malformed evidence preserve the published answers")


def check_complete_history_order(suite, reference, anno, rust_reader):
    reversed_suite = {**suite, "cases": [
        {**case, "steps": list(reversed(case["steps"]))} for case in suite["cases"]
    ]}
    compare("SQLite reversed history", read_suite(reversed_suite, anno), reference)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "reversed.json"
        path.write_text(json.dumps(reversed_suite), encoding="utf-8")
        compare("Rust reversed history", call_json([rust_reader, str(path)]), reference)


def build_rust_reader():
    command = ["cargo", "build", "--locked", "-p", "anno", "--no-default-features",
               "--example", "reader_answer", "--message-format=json"]
    result = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, check=True, text=True)
    for line in result.stdout.splitlines():
        artifact = json.loads(line)
        if artifact.get("target", {}).get("name") == "reader_answer" and artifact.get("executable"):
            return Path(artifact["executable"])
    raise RuntimeError("Cargo did not report a reader_answer executable")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=HERE / "fixtures" / "cases.json")
    parser.add_argument("--answer-key", type=Path, default=HERE / "fixtures" / "answer_key.json")
    parser.add_argument("--anno", help="built anno CLI executable")
    parser.add_argument("--rust-reader", help="built reader_answer example")
    parser.add_argument("--emit", action="store_true", help="print SQLite answers without opening an answer key")
    parser.add_argument("--bounded", action="store_true", help="also replay generated short histories")
    args = parser.parse_args()
    if args.emit and args.bounded:
        parser.error("--emit and --bounded run different checks")
    if args.anno is None:
        sys.path.insert(0, str(HERE.parent / "congress"))
        from check_catalog_history import build_cli
        args.anno = build_cli()
    suite = json.loads(args.cases.read_text())
    native = read_suite(suite, args.anno)
    if args.emit:
        print(json.dumps(native, ensure_ascii=False, sort_keys=True))
        return
    key = json.loads(args.answer_key.read_text())
    compare("SQLite + anno CLI", native, key)
    rust_reader = args.rust_reader or build_rust_reader()
    rust = call_json([rust_reader, str(args.cases)])
    compare("Rust", rust, key)
    compare("independent readers", native, rust)
    check_complete_history_order(suite, key, args.anno, rust_reader)
    total = sum(map(len, native["answers"].values()))
    print(f"{len(native['answers'])} cases, {total} authored questions match (SQLite + anno CLI and independent Rust)")
    from snippets import check_snippets
    check_snippets(args.anno)
    check_transport()
    if args.bounded:
        from bounded import check_histories
        check_histories(args.anno, rust_reader)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, sqlite3.Error, subprocess.SubprocessError) as error:
        sys.exit(str(error))
