# SPDX-License-Identifier: MIT OR Apache-2.0
"""Exercise release of original synthetic bibliographies across catalog and policy changes."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from .publication import Publication, Unavailable


DOI = "10.0000/plate"
OTHER = "10.0000/list"
FIRST = {"doi": DOI, "cut": "first"}
LATEST = {"doi": DOI.upper(), "cut": "as_of", "date": "2024-03-01"}


def edition(number, date, author, digest):
    return {
        "edition": number, "published_on": date,
        "source_url": f"https://publisher.example/{author}/v{number}.xml",
        "source_sha256": digest * 64, "license": "https://creativecommons.org/licenses/by/4.0/",
        "byline": [[["given-names", "Éva"], ["surname", author]]],
    }


def work(doi, editions):
    return {"doi": doi, "through": "2024-03-01", "editions": editions}


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "publication.sqlite"
        self.service = Publication(self.path)
        self.writer = Publication(self.path)
        self.first = edition(1, "2024-01-10", "First", "a")
        self.later = edition(2, "2024-02-10", "Revised", "b")
        self.works = [work(DOI, [self.first, self.later]), work(OTHER, [self.first])]
        self.rights = {DOI: "full", OTHER: "metadata"}
        self.writer.publish_catalog(3, self.works)
        self.writer.publish_policy("reader", 3, self.rights)

    def test_online_and_bulk_preserve_order_and_scope(self):
        other = {"doi": OTHER, "cut": "first"}
        queries = [LATEST, other, FIRST, LATEST, {"doi": "10.0000/unknown", "cut": "first"}]
        online = [self.service.online("reader", query) for query in queries]
        artifact = self.service.export("reader", queries)
        actual = json.loads(Publication(self.path).download("reader", artifact))
        self.assertEqual(actual, online)
        self.assertEqual([actual[n]["byline"] for n in (0, 2, 3)],
                         [self.later["byline"], self.first["byline"], self.later["byline"]])
        self.assertEqual(set(actual[1]), {"doi", "cut", "date", "status", "edition", "published_on"})
        self.assertEqual(actual[-1]["status"], "unavailable")
        self.writer.publish_policy("other-viewer", 1, self.rights)
        with self.assertRaises(Unavailable):
            self.service.download("other-viewer", artifact)
        self.assertEqual(json.loads(self.service.download("reader", artifact)), online)

    def test_duplicate_late_and_conflicting_events(self):
        artifact = self.service.export("reader", [LATEST])
        self.assertFalse(self.writer.publish_catalog(3, list(reversed(self.works))))
        self.assertFalse(self.writer.publish_policy("reader", 3, {DOI.upper(): "full", OTHER: "metadata"}))
        self.assertFalse(self.writer.publish_catalog(1, []))
        self.assertFalse(self.writer.publish_policy("reader", 1, {}))
        self.assertFalse(self.writer.publish_policy("reader", 1, {}))
        with self.assertRaisesRegex(ValueError, "conflicting duplicate"):
            self.writer.publish_policy("reader", 1, self.rights)
        with self.assertRaisesRegex(ValueError, "conflicting duplicate"):
            self.writer.publish_catalog(3, [])
        self.assertEqual(json.loads(self.service.download("reader", artifact))[0]["edition"], 2)
        self.writer.publish_catalog(4, [self.works[1]])
        with self.assertRaises(Unavailable):
            self.service.download("reader", artifact)
        self.assertEqual(self.service.online("reader", FIRST)["status"], "unavailable")

    def test_revocation_during_generation_never_publishes_partial_bytes(self):
        def interrupted():
            yield FIRST
            self.writer.publish_policy("reader", 4, {DOI: "metadata"})
            yield LATEST

        with self.assertRaises(Unavailable):
            self.service.export("reader", interrupted())
        with closing(sqlite3.connect(self.path)) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM export").fetchone(), (0,))
        refreshed = self.service.export("reader", [FIRST, LATEST])
        rows = json.loads(self.service.download("reader", refreshed))
        self.assertEqual([r["edition"] for r in rows], [1, 2])
        self.assertTrue(all("byline" not in row for row in rows))

    def test_cached_bytes_require_current_authorization(self):
        artifact = self.service.export("reader", [LATEST])
        allowed_bytes = self.service.download("reader", artifact)
        self.writer.publish_policy("reader", 4, {DOI: "metadata"})
        # Negative control: releasing the cached payload by handle alone would
        # expose the contributor and the locator even after permissions changed.
        with closing(sqlite3.connect(self.path)) as db:
            unsafe, = db.execute("SELECT payload FROM export WHERE id=?", (artifact,)).fetchone()
        self.assertEqual(allowed_bytes, unsafe)
        self.assertIn(b"Revised", unsafe)
        self.assertIn(self.later["source_sha256"].encode(), unsafe)
        with self.assertRaises(Unavailable):
            self.service.download("reader", artifact)
        self.writer.publish_policy("reader", 5, self.rights)
        with self.assertRaises(Unavailable):
            self.service.download("reader", artifact)

    def test_hidden_work_and_unknown_work_have_identical_output(self):
        self.writer.publish_policy("reader", 4, {DOI: "metadata"})
        denied = {"doi": OTHER, "cut": "first"}
        before = self.service.online("reader", denied)
        self.writer.publish_catalog(5, [self.works[0]])
        after = json.loads(self.service.download("reader", self.service.export("reader", [denied])))
        self.assertEqual(after, [before])
        self.assertEqual(before["status"], "unavailable")

    def test_metadata_only_bytes_do_not_depend_on_protected_evidence(self):
        # These are independent source stores. The visible publication metadata
        # for the requested edition agrees in all of them.
        changed = edition(2, self.later["published_on"], "Different", "e")
        missing = {"edition": 2, "published_on": self.later["published_on"]}
        invalid = {**changed, "byline": [], "source_sha256": "invalid", "license": None}
        variants = [[self.later], [changed], [missing], [invalid], [self.later, changed]]
        outputs = []
        for index, versions in enumerate(variants):
            db = Publication(Path(self.temp.name) / f"catalog-{index}.sqlite")
            db.publish_catalog(1, [work(DOI, [self.first, *versions])])
            db.publish_policy("reader", 1, {DOI: "metadata"})
            rows = [db.online("reader", q) for q in (FIRST, LATEST)]
            self.assertEqual([r["edition"] for r in rows], [1, 2])
            public = db.download("reader", db.export("reader", [FIRST, LATEST]))
            self.assertEqual(json.loads(public), rows)
            outputs.append(public)
        self.assertEqual(len(set(outputs)), 1)
        safe = {"doi", "cut", "date", "status", "edition", "published_on"}
        self.assertTrue(all(set(row) == safe for row in json.loads(outputs[0])))


if __name__ == "__main__":
    unittest.main()
