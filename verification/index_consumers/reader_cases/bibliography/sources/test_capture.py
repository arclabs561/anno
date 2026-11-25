# SPDX-License-Identifier: MIT OR Apache-2.0
"""Offline checks for missing editions, inconsistent chronology and source rights."""
import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from verification.index_consumers.reader_cases.bibliography.sources.capture import (
    HERE, boundary, source,
)
from verification.index_consumers.reader_cases.bibliography.sources.verify_print import verify

STAMP = datetime(2024, 1, 4, tzinfo=timezone.utc)


def version(n, hour=0):
    return {"status": "vor", "id": "12", "doi": "10.7554/eLife.12", "version": n,
            "versionDate": f"2024-01-01T{hour:02}:00:00Z"}


class SourceChecks(unittest.TestCase):
    def test_same_day_and_missing_edition(self):
        cut, excluded = boundary({"versions": [version(2, 11), {"status": "preprint"}, version(1, 9)]}, "12")
        self.assertEqual(([r[0] for r in cut], excluded), ([1, 2], 1))
        with self.assertRaises(ValueError):
            boundary({"versions": [version(2)]}, "12")
        with self.assertRaises(ValueError):
            boundary({"versions": [version(1), version(1)]}, "12")

    def test_announced_but_unretrieved(self):
        with TemporaryDirectory() as tmp:
            cache = Path(tmp)
            meta, wire, _ = source(cache, "12", None, False, STAMP)
            self.assertEqual((meta["status"], wire), ("index_unavailable", None))
            (cache / "12-versions.json").write_text(json.dumps({"versions": [version(1), version(2)]}))
            meta, wire, rights = source(cache, "12", None, False, STAMP)
            self.assertEqual(wire["through"], "2024-01-03")
            self.assertEqual(wire["editions"], [{"edition": n, "published_on": "2024-01-01"} for n in (1, 2)])
            self.assertEqual([r["status"] for r in meta["editions"]], ["version_record_unavailable"] * 2)
            self.assertEqual(rights, [])

    def test_disagreeing_rights(self):
        with TemporaryDirectory() as tmp:
            cache = Path(tmp)
            v = version(1)
            (cache / "12-versions.json").write_text(json.dumps({"versions": [v]}))
            v["xml"] = "https://cdn.elifesciences.org/articles/12/elife-12-v1.xml"
            v["copyright"] = {"license": "CC-BY-4.0"}
            (cache / "12-v1.json").write_text(json.dumps(v))
            (cache / "12-v1.xml").write_text(
                '<article xmlns:xlink="http://www.w3.org/1999/xlink"><front><article-meta>'
                '<article-id pub-id-type="doi">10.7554/eLife.12</article-id>'
                '<permissions><license xlink:href="https://creativecommons.org/publicdomain/zero/1.0/"/>'
                '</permissions></article-meta></front></article>'
            )
            meta, wire, rights = source(cache, "12", None, False, STAMP)
            self.assertEqual(meta["editions"][0]["status"], "xml_identity_rights_or_relation_unverified")
            self.assertNotIn("byline", wire["editions"][0])
            self.assertEqual(rights, [])

    def test_pdf_cache_stays_outside_the_worktree(self):
        path = HERE / "publisher-inputs"
        with self.assertRaisesRegex(ValueError, "external cache"):
            verify([], path, True)
        self.assertFalse(path.exists())
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            verify([], directory / "pdf-cache", False)
            self.assertTrue((directory / "pdf-cache").is_dir())

    def test_pinned_selection_and_coverage(self):
        frame = json.loads((HERE / "frame.json").read_bytes())
        receipt = json.loads((HERE / "receipt.json").read_bytes())
        original = (HERE / frame["frame_source"]).read_bytes()
        self.assertEqual(sha256(original).hexdigest(), frame["frame_sha256"])
        self.assertEqual(sha256((HERE / "frame.json").read_bytes()).hexdigest(), receipt["frame_sha256"])
        manifest = json.loads(original)
        self.assertEqual(frame["rows"], manifest["eligible"][12:])
        self.assertEqual(len(frame["rows"]), 17)
        for row in frame["rows"]:
            editions = receipt["sources"][row["target"]]["editions"]
            self.assertEqual([e["edition"] for e in editions], list(range(1, len(editions) + 1)))
            self.assertTrue(all(e["status"] == "supported" and "byline" not in e for e in editions))
            self.assertTrue(any(e["published_on"] < row["day"] for e in editions))
            self.assertEqual(editions[-1]["published_on"], row["day"])
            notices = receipt["sources"][row["notice"]]["editions"]
            self.assertTrue(all(e["corrected_doi"].lower() == f"10.7554/elife.{row['target']}" for e in notices))


if __name__ == "__main__":
    unittest.main()
