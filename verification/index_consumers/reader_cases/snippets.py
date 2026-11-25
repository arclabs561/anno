# SPDX-License-Identifier: MIT OR Apache-2.0
"""Project public excerpts and staff evidence locators under a trusted reader policy."""
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile

from check import LIVE, answer, install_case


def read_snippets(path, at, policy, audience):
    # Public assertion IDs and issuer labels are supplied independently of the
    # restricted text. Neither the digest nor offsets of the whole source are public.
    if audience == "public":
        fields = "s.id AS assertion, s.issuer, q.surface AS quote"
    elif audience == "staff":
        fields = """s.id AS assertion, s.issuer, q.surface AS quote, e.id AS source,
                    e.edition, e.sha256 AS source_sha256, q.start, q.end"""
    else:
        raise ValueError(f"unknown audience {audience}")
    query = f"""
        SELECT {fields}
        FROM selected s JOIN excerpt x ON (s.id,s.rev,s.source)=(x.id,x.rev,x.source)
        JOIN source e ON (e.id,e.edition)=(x.source,x.edition)
        JOIN quote q ON (q.source,q.start,q.end)=(e.id,x.start,x.end)
        WHERE s.kind='credit' AND s.role='writing'
        ORDER BY s.id
    """
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        db.execute("BEGIN")
        if not db.execute("SELECT 1 FROM edition WHERE rev=?", (at,)).fetchone():
            raise ValueError(f"unknown factual edition {at}")
        if not db.execute("SELECT 1 FROM policy WHERE rev=?", (policy,)).fetchone():
            raise ValueError(f"unknown snippet policy {policy}")
        rows = db.execute(LIVE + query, {
            "at": at, "policy": policy, "source": None, "issuer": None, "staff": audience == "staff",
        })
        return [dict(row) for row in rows]


def install_snippets(path, anno, first, second):
    text = first + " " + second
    start = len(first) + 1
    case = {
        "id": "excerpt-surface", "work": "leaf",
        "sources": [{"id": "proof", "edition": "original", "text": text}],
        "steps": [{"revision": 1, "put": [
            {"id": "foreword", "kind": "credit", "issuer": "press", "source": "proof",
             "target": "person-1", "role": "writing"},
            {"id": "ledger", "kind": "credit", "issuer": "editor", "source": "proof",
             "target": "person-2", "role": "writing"},
        ]}, {"revision": 2, "put": [
            {"id": "foreword", "kind": "credit", "issuer": "press", "source": "proof",
             "target": "person-3", "role": "writing"},
        ]}],
        "policies": [{"revision": 10, "hidden": ["ledger"]},
                     {"revision": 11, "hidden": ["foreword", "ledger"]}],
    }
    # The second edition will separately approve only the first word.
    ranges = [("proof", 0, len(first)), ("proof", start, len(text)),
              ("proof", 0, len(first.split()[0]))]
    with closing(sqlite3.connect(path)) as db, db:
        install_case(db, case, anno, ranges)
        db.execute("""CREATE TABLE excerpt (
            id TEXT, rev INTEGER, source TEXT, edition TEXT, start INTEGER, end INTEGER,
            PRIMARY KEY(id,rev,source))""")
        db.executemany("INSERT INTO excerpt VALUES (?,?,?,?,?,?)", [
            ("foreword", 1, "proof", "original", 0, len(first)),
            ("ledger", 1, "proof", "original", start, len(text)),
        ])


def wire(rows):
    return json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def check_snippets(anno):
    first = "Éva wrote the foreword."
    suffixes = ("Lio wrote the notes.", "Mara revised the scores.")
    authorized = {"assertion": "foreword", "issuer": "press", "quote": first}
    public_views, leaky_views, staff_proofs = [], [], []
    with tempfile.TemporaryDirectory() as directory:
        for i, second in enumerate(suffixes):
            path = Path(directory) / f"snippets-{i}.sqlite"
            install_snippets(path, anno, first, second)
            text = first + " " + second
            locator = {"source": "proof", "edition": "original",
                       "source_sha256": hashlib.sha256(text.encode()).hexdigest()}
            expected_proofs = [
                {**authorized, **locator, "start": 0, "end": len(first)},
                {"assertion": "ledger", "issuer": "editor", "quote": second,
                 **locator, "start": len(first) + 1, "end": len(text)},
            ]
            public = read_snippets(path, 1, 10, "public")
            proofs = read_snippets(path, 1, 10, "staff")
            assert public == [authorized] and proofs == expected_proofs
            # The gateway supplies policy 11 even for the original factual edition.
            historical = read_snippets(path, 1, 11, "public")
            latest = read_snippets(path, 2, 11, "public")
            assert historical == latest == []
            assert read_snippets(path, 1, 11, "staff") == expected_proofs
            # The author binding has changed but there is no approval for excerpts
            # from that revised assertion, even for a reader who can see the claim.
            with closing(sqlite3.connect(path)) as db:
                assert answer(db, {"at": 2, "policy": 10, "audience": "public", "kind": "credit",
                                   "role": "writing"}) == ["person-3"]
            unapproved = read_snippets(path, 2, 10, "public")
            assert unapproved == [], "a prior excerpt approval followed the revised assertion"
            assert read_snippets(path, 2, 10, "staff") == expected_proofs[1:]
            word = first.split()[0]
            with closing(sqlite3.connect(path)) as db, db:
                db.execute("INSERT INTO excerpt VALUES (?,?,?,?,?,?)",
                           ("foreword", 2, "proof", "original", 0, len(word)))
            approved = read_snippets(path, 2, 10, "public")
            assert approved == [{**authorized, "quote": word}]
            assert read_snippets(path, 2, 10, "staff") == [
                {**expected_proofs[0], "quote": word, "end": len(word)}, expected_proofs[1],
            ]
            assert read_snippets(path, 1, 11, "public") == []
            assert read_snippets(path, 1, 11, "staff") == expected_proofs
            public_views.append(wire([public, historical, latest, unapproved, approved]))
            staff_proofs.append(wire(proofs[0]))
            # The old public shape would confirm which withheld suffix was supplied.
            leaky_views.append(wire([{**public[0], "source_sha256": proofs[0]["source_sha256"]}]))
    assert public_views[0] == public_views[1]
    assert staff_proofs[0] != staff_proofs[1]
    assert leaky_views[0] != leaky_views[1]
    print("22 reopened snippet views: public JSON agrees across two withheld suffixes; "
          "staff proofs differ; revised claims need fresh excerpt approval")
