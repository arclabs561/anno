# SPDX-License-Identifier: MIT OR Apache-2.0
"""Publish a bibliography to one authenticated viewer using a local SQLite store.

The application supplies the viewer identity and installs complete catalog and
permission snapshots. Readers receive bytes only through online or download.
The database and staged rows are private inputs. This gate governs future
releases from this store. A recipient may retain previously downloaded bytes
and public DOI resolvers or publisher sites may serve their own bylines.
"""
from contextlib import closing, contextmanager
import json
import sqlite3

from .reader import doi_key, index_works, read_one, request, select


class Unavailable(LookupError):
    """The requested export is unavailable to this viewer at the current cut."""


def wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class Publication:
    def __init__(self, path):
        self.path = path
        with closing(sqlite3.connect(path)) as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS snapshot (
                    kind TEXT NOT NULL, owner TEXT NOT NULL, sequence INTEGER NOT NULL,
                    body TEXT NOT NULL, PRIMARY KEY(kind, owner, sequence)
                );
                CREATE TABLE IF NOT EXISTS export (
                    id INTEGER PRIMARY KEY, viewer TEXT NOT NULL, catalog INTEGER NOT NULL,
                    policy INTEGER NOT NULL, payload BLOB NOT NULL
                );
            """)

    @contextmanager
    def _transaction(self, *, writing=False):
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("BEGIN IMMEDIATE" if writing else "BEGIN")
            yield db

    def _install(self, kind, owner, sequence, value):
        if type(sequence) is not int or not 1 <= sequence <= (1 << 63) - 1:
            raise ValueError("expected a positive snapshot sequence")
        body = wire(value)
        with self._transaction(writing=True) as db:
            seen = db.execute(
                "SELECT body FROM snapshot WHERE (kind,owner,sequence)=(?,?,?)",
                (kind, owner, sequence),
            ).fetchone()
            if seen is not None:
                if seen[0] != body:
                    raise ValueError("conflicting duplicate snapshot")
                return False
            head, = db.execute(
                "SELECT MAX(sequence) FROM snapshot WHERE (kind,owner)=(?,?)",
                (kind, owner),
            ).fetchone()
            db.execute("INSERT INTO snapshot VALUES (?,?,?,?)", (kind, owner, sequence, body))
            return head is None or sequence > head

    def publish_catalog(self, sequence, works):
        """Install a complete view of the retained publisher manifests."""
        return self._install("catalog", "", sequence, index_works(works))

    def publish_policy(self, viewer, sequence, access):
        """Replace this viewer's permissions. Omitted DOIs are inaccessible."""
        if not isinstance(viewer, str) or not viewer.strip():
            raise ValueError("expected an authenticated viewer")
        if not isinstance(access, dict):
            raise ValueError("expected permissions keyed by DOI")
        levels = {}
        for doi, level in access.items():
            key = doi_key(doi)
            if level not in ("metadata", "full") or key in levels:
                raise ValueError("expected one metadata or full permission per DOI")
            levels[key] = level
        return self._install("policy", viewer, sequence, levels)

    @staticmethod
    def _cut(db, viewer):
        if not isinstance(viewer, str) or not viewer.strip():
            raise Unavailable("bibliography unavailable")
        current = dict(db.execute(
            """SELECT kind, MAX(sequence) FROM snapshot
               WHERE (kind='catalog' AND owner='') OR (kind='policy' AND owner=?)
               GROUP BY kind""", (viewer,),
        ))
        if "catalog" not in current or "policy" not in current:
            raise Unavailable("bibliography unavailable")
        return current["catalog"], current["policy"]

    @staticmethod
    def _body(db, kind, owner, sequence):
        body, = db.execute(
            "SELECT body FROM snapshot WHERE (kind,owner,sequence)=(?,?,?)",
            (kind, owner, sequence),
        ).fetchone()
        return json.loads(body)

    def _inputs(self, db, viewer):
        catalog_rev, policy_rev = self._cut(db, viewer)
        catalog = self._body(db, "catalog", "", catalog_rev)
        access = self._body(db, "policy", viewer, policy_rev)
        return (catalog_rev, policy_rev), catalog, access

    @staticmethod
    def _row(catalog, access, query):
        query = request(query)
        doi = query["doi"]
        level = access.get(doi)
        work = catalog.get(doi) if level else None
        if work is None or level == "full":
            return read_one(work, query)
        # For metadata-only requests, even the choice of status must depend
        # solely on the trusted publication metadata. Two receipts can disagree
        # about protected evidence while agreeing on the edition and its date.
        releases = work.get("editions")
        if isinstance(releases, list):
            releases = [
                {k: version.get(k) for k in ("edition", "published_on")}
                if isinstance(version, dict) else None for version in releases
            ]
        edition, reason = select({"through": work.get("through"), "editions": releases}, query)
        if reason:
            return {**query, "status": "unavailable", "reason": reason}
        return {**query, "status": "available", "edition": edition["edition"],
                "published_on": edition["published_on"]}

    def online(self, viewer, query):
        with self._transaction() as db:
            _, catalog, access = self._inputs(db, viewer)
            return self._row(catalog, access, query)

    def export(self, viewer, queries):
        """Stage rows at one cut. Publish a handle only if that cut is still current."""
        with self._transaction() as db:
            cut, catalog, access = self._inputs(db, viewer)
        rows = [self._row(catalog, access, q) for q in queries]
        payload = wire(rows).encode("utf-8")
        with self._transaction(writing=True) as db:
            if self._cut(db, viewer) != cut:
                raise Unavailable("bibliography unavailable")
            artifact = db.execute(
                "INSERT INTO export(viewer,catalog,policy,payload) VALUES (?,?,?,?)",
                (viewer, *cut, payload),
            )
            return artifact.lastrowid

    def download(self, viewer, artifact):
        """Authorize release of the cached bytes against the current committed snapshots."""
        if type(artifact) is not int or not 1 <= artifact <= (1 << 63) - 1:
            raise Unavailable("bibliography unavailable")
        with self._transaction() as db:
            cut = self._cut(db, viewer)
            row = db.execute(
                "SELECT payload FROM export WHERE (id,viewer,catalog,policy)=(?,?,?,?)",
                (artifact, viewer, *cut),
            ).fetchone()
            if row is None:
                raise Unavailable("bibliography unavailable")
            return row[0]
