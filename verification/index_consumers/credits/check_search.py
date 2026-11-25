#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Check persisted credit search and autocomplete across a public access grant and revocation."""

import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile

from check_cases import build_cli, publish


CASE = "attribution-access-changes-while-printed-byline-stays-public"


def initialize(path):
    with closing(sqlite3.connect(path)) as db, db:
        db.executescript("""
            CREATE TABLE head (source TEXT PRIMARY KEY, sequence INTEGER);
            CREATE TABLE posting (source, audience, work, predicate, catalog, accession, surface);
            CREATE TABLE term (source, audience, catalog, accession, surface, folded,
                               PRIMARY KEY(source, audience, catalog, accession, surface));
        """)


def install(path, source, publication, *, retain_public_terms):
    """Install the complete trusted source snapshot and manifest in one transaction."""
    sequence = publication.checkpoint["sequence"]
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("BEGIN IMMEDIATE")
        old = db.execute("SELECT sequence FROM head WHERE source=?", (source,)).fetchone()
        if old and sequence < old[0]:
            return "stale"
        if sequence != (old[0] + 1 if old else 1):
            raise ValueError(f"{source} edition {sequence} out of order")

        db.execute("DELETE FROM posting WHERE source=?", (source,))
        if retain_public_terms:
            db.execute("DELETE FROM term WHERE source=? AND audience='staff'", (source,))
        else:
            db.execute("DELETE FROM term WHERE source=?", (source,))
        for key, (work, predicate, access) in publication.manifest["credits"].items():
            if predicate is None:
                continue
            row = publication.rows[key]
            signal, = row["signals"]
            surface = signal["surface"] if predicate == "printed_credit" else row["surface"]
            record = row["catalog_record"]
            identity = record["catalog"], record["accession"]
            for audience in ("public", "staff") if access == "public" else ("staff",):
                db.execute("INSERT INTO posting VALUES(?,?,?,?,?,?,?)",
                           (source, audience, work, predicate, *identity, surface))
                if predicate == "attributed_to":
                    db.execute("INSERT OR IGNORE INTO term VALUES(?,?,?,?,?,?)",
                               (source, audience, *identity, surface, surface.casefold()))
        db.execute("INSERT OR REPLACE INTO head VALUES(?,?)", (source, sequence))
        return "accepted"


def search(path, audience, work, subject, prefix):
    # All four endpoints read one snapshot through a new read-only connection.
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db, db:
        db.execute("BEGIN")
        matching = "FROM posting WHERE audience=? AND catalog=? AND accession=? AND predicate='attributed_to'"
        params = (audience, *subject)
        exact = [r[0] for r in db.execute("SELECT DISTINCT work " + matching + " ORDER BY work", params)]
        count, = db.execute("SELECT count(DISTINCT work) " + matching, params).fetchone()
        printed = [r[0] for r in db.execute("""
            SELECT surface FROM posting WHERE audience=? AND work=? AND predicate='printed_credit'
            ORDER BY surface""", (audience, work))]
        suggestions = db.execute("""
            SELECT DISTINCT catalog, accession, surface FROM term
            WHERE audience=? AND substr(folded, 1, length(?))=? ORDER BY catalog, accession, surface""",
            (audience, prefix.casefold(), prefix.casefold())).fetchall()
    return exact, count, printed, suggestions


def replay(path, case, publications, *, retain_public_terms=False):
    initialize(path)
    leaked = []
    subject, work = case["queries"]["person"], case["work"]
    name = (*subject, "Nivo Pell")
    for step in case["steps"]:
        source, edition = step["source"], step["edition"]
        delivery = install(path, source, publications[source, edition], retain_public_terms=retain_public_terms)
        if delivery != step.get("reject", "accepted"):
            raise AssertionError(f"unexpected {delivery} for {source} edition {edition}")
        expected = step["expect"]
        for audience, access in (("public", "public"), ("staff", "restricted")):
            visible = bool(expected[access].get("person"))
            exact, count, printed, suggestions = search(path, audience, work, subject, "ni")
            if (exact, count, printed) != ([work] if visible else [], int(visible), expected["printed"]):
                raise AssertionError(f"incorrect exact search, count, or printed byline at {source}/{edition}/{audience}")
            if suggestions == ([name] if visible else []):
                continue
            if retain_public_terms and audience == "public" and not visible and suggestions == [name]:
                leaked.append((source, edition, delivery))
            else:
                raise AssertionError(f"incorrect autocomplete at {source}/{edition}/{audience}: {suggestions}")
    return leaked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anno", type=Path, help="use an existing anno executable")
    args = parser.parse_args()
    cases = json.loads(Path(__file__).with_name("cases.json").read_text(encoding="utf-8"))
    case, = [c for c in cases["scenarios"] if c["name"] == CASE]
    publications = publish((args.anno or build_cli()).resolve(), case)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        replay(root / "correct.sqlite", case, publications)
        leaked = replay(root / "faulty.sqlite", case, publications, retain_public_terms=True)
    if leaked != [("ledger", 3, "accepted"), ("ledger", 2, "stale")]:
        raise AssertionError(f"expected stale public suggestions after revocation and stale replay, got {leaked}")
    print(f"{CASE}: {len(case['steps'])} deliveries passed; stale autocomplete control caught on both final reads")


if __name__ == "__main__":
    main()
