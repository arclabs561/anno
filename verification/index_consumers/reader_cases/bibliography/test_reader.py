# SPDX-License-Identifier: MIT OR Apache-2.0
"""Development cases use original fictitious works. --anno PATH requires the native test."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from reader import doi_key, json_text, publish_byline, read_bulk, read_one, verify_source


DOI = "10.0000/fixture"
LICENSE = "https://creativecommons.org/licenses/by/4.0/"
HERE = Path(__file__).resolve().parent
FIRST = [[["given-names", "Aï"], ["surname", "Xu"]], [["collab", "Tide Survey"]],
         [["given-names", "Aï"], ["surname", "Xu"]]]
SECOND = [[["string-name", "É. Raad"]], *FIRST]
THIRD = [*FIRST[:1], [["given-names", "Aï"], ["surname", "Xu"], ["suffix", "Jr."]]]


def edition(version, day, authors):
    body = f"original fabricated source {version}".encode()
    return {"edition": version, "published_on": day, "byline": authors,
            "source_url": f"https://journal.example/{version}",
            "source_sha256": hashlib.sha256(body).hexdigest(), "license": LICENSE}


def query(cut, day=None, doi=DOI):
    q = {"doi": doi, "cut": cut}
    if day is not None:
        q["date"] = day
    return q


class BibliographyReaderTest(unittest.TestCase):
    def setUp(self):
        self.v1 = edition(1, "2021-03-14", FIRST)
        self.v2 = edition(2, "2021-03-21", SECOND)
        self.v3 = edition(3, "2021-03-21", THIRD)
        self.work = {"doi": DOI.upper(), "through": "2021-04-01",
                     "editions": [self.v3, self.v1, self.v2]}

    def test_three_cuts_order_and_provenance(self):
        for q, selected in [(query("first"), self.v1),
                            (query("before", "2021-03-21"), self.v1),
                            (query("as_of", "2021-03-21"), self.v3)]:
            self.assertEqual(read_one(self.work, q),
                             {**q, "date": q.get("date"), "status": "available", **selected})
        selected = read_one(self.work, query("first"))
        selected["byline"][0].append(["suffix", "changed externally"])
        self.assertEqual(self.v1["byline"], FIRST)
        self.assertEqual(selected["byline"][2], FIRST[2])

    def test_dates_and_certified_coverage(self):
        probes = [(query("before", "2021-03-14"), "no_edition_at_cut"),
                  (query("as_of", "2021-03-13"), "no_edition_at_cut"),
                  (query("as_of", "2021-04-02"), "outside_coverage"),
                  (query("before", "2021-04-03"), "outside_coverage")]
        for q, reason in probes:
            self.assertEqual(read_one(self.work, q), {**q, "status": "unavailable", "reason": reason})
        self.assertEqual(read_one(self.work, query("before", "2021-04-02"))["edition"], 3)
        self.assertEqual(read_one(self.work, query("as_of", "2021-03-14"))["byline"], FIRST)

    def test_unknown_unreadable_and_malformed_release(self):
        self.assertEqual(read_one(None, query("first"))["reason"], "unknown_work")
        q = query("as_of", "2021-03-21")
        for replacement in [{"byline": []}, {"byline": [[["role", "senior author"]]]},
                            {"license": None}, {"source_sha256": "abc"}, {"source_url": None}]:
            changed = {**self.work, "editions": [self.v1, self.v2, {**self.v3, **replacement}]}
            self.assertEqual(read_one(changed, q), {**q, "status": "unavailable",
                                                    "reason": "selected_evidence_unavailable"})
            self.assertEqual(read_one(changed, query("first"))["edition"], 1)

    def test_partial_history_and_contradictions(self):
        gap = {**self.work, "editions": [self.v1, self.v3]}
        # v2's day is unknown, so v1 cannot be declared the latest strictly
        # before v3's day. v3 is nevertheless provably latest at that day's end.
        self.assertEqual(read_one(gap, query("before", "2021-03-21"))["reason"], "incomplete_history")
        self.assertEqual(read_one(gap, query("as_of", "2021-03-21"))["edition"], 3)
        prefix_missing = {**self.work, "editions": [self.v2, self.v3]}
        self.assertEqual(read_one(prefix_missing, query("first"))["reason"], "incomplete_history")
        self.assertEqual(read_one(prefix_missing, query("as_of", "2021-03-21"))["edition"], 3)
        no_watermark = {**self.work, "through": None}
        self.assertEqual(read_one(no_watermark, query("first"))["edition"], 1)
        self.assertEqual(read_one(no_watermark, query("before", "2021-03-21"))["edition"], 1)
        self.assertEqual(read_one(no_watermark, query("as_of", "2021-03-21"))["reason"],
                         "incomplete_history")
        self.work["editions"].append(dict(self.v1))
        self.assertEqual(read_one(self.work, query("first"))["edition"], 1)
        self.work["editions"][-1]["source_sha256"] = "f" * 64
        self.assertEqual(read_one(self.work, query("first"))["reason"], "invalid_chronology")
        for malformed in ({"edition": True}, {"published_on": "2021-03-01"},
                          {"published_on": "2021-04-02"}):
            work = {**self.work, "editions": [self.v1, {**self.v2, **malformed}]}
            self.assertEqual(read_one(work, query("first"))["reason"], "invalid_chronology")

    def test_input_and_wire(self):
        self.assertEqual(doi_key(DOI.upper()), DOI)
        for q in (query("yesterday"), query("first", "2021-01-01"), query("as_of"),
                  query("as_of", "2021-02-29"), query("before", "20210314")):
            with self.assertRaises(ValueError):
                read_one(self.work, q)
        qs = [query("first"), query("first", doi="10.0000/unknown"), query("first")]
        rows = read_bulk([self.work], qs)
        self.assertEqual(rows, [read_one(self.work, q) for q in qs])
        with self.assertRaisesRegex(ValueError, "duplicate work"):
            read_bulk([self.work, self.work], qs)
        with tempfile.TemporaryDirectory() as temp:
            works, queries = Path(temp) / "works.json", Path(temp) / "queries.json"
            works.write_text(json.dumps({"works": [self.work]}))
            queries.write_text(json.dumps(qs))
            command = [sys.executable, str(HERE / "reader.py"), str(works)]
            bulk = subprocess.run([*command, str(queries)], capture_output=True, text=True, check=True)
            online = subprocess.run([*command, "--ask"], input=json.dumps(qs[0]),
                                    capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(bulk.stdout), rows)
            self.assertEqual(json.loads(online.stdout), rows[0])

    def test_reopen_frozen_source(self):
        first = read_one(self.work, query("first"))
        original = b"original fabricated source 1"
        verify_source(first, self.v1["source_url"], original)
        with self.assertRaisesRegex(ValueError, "source bytes"):
            verify_source(first, self.v1["source_url"], b"original fabricated source 2")
        with self.assertRaisesRegex(ValueError, "locator"):
            verify_source(first, self.v2["source_url"], original)

    def test_native_anno_named_editions_and_occurrences(self):
        anno = os.environ.get("ANNO_BIBLIO_CLI")
        if not anno:
            self.skipTest("pass --anno PATH to require the native anno publication round trip")
        self.assertTrue(os.access(anno, os.X_OK), f"required anno executable unavailable: {anno}")
        # Identical byline bytes in separately named editions retain separate
        # source revisions. UTF-8 multibyte letters exercise scalar offsets.
        self.work["editions"] = [self.v1, {**self.v2, "byline": FIRST}]
        native = []
        for q in (query("first"), query("as_of", "2021-03-21")):
            row = read_one(self.work, q)
            publication = publish_byline(row, anno)
            native.append(publication)
            self.assertNotEqual(publication["source_sha256"], row["source_sha256"])
            ordered = sorted(publication["rows"], key=lambda r: r["signal_id"])
            self.assertEqual([json.loads(r["surface"]) for r in ordered], FIRST)
            for posting in ordered:
                self.assertEqual(posting["label"], "JATS_AUTHOR")
                span = posting["location"]["Text"]
                self.assertEqual(json_text(FIRST)[span["start"]:span["end"]], posting["surface"])
        self.assertEqual(native[0]["source_sha256"], native[1]["source_sha256"])
        self.assertEqual([p["source_revision"] for p in native], ["v1", "v2"])
        self.assertNotEqual(native[0]["checkpoint"], native[1]["checkpoint"])


if __name__ == "__main__":
    options = argparse.ArgumentParser(add_help=False)
    options.add_argument("--anno")
    args, unknown = options.parse_known_args()
    if args.anno:
        os.environ["ANNO_BIBLIO_CLI"] = args.anno
    unittest.main(argv=[sys.argv[0], *unknown])
