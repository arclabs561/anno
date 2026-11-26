"""Storage and attachment contracts for locally materialized corpus windows."""
import json
import stat
import tempfile
import unittest
from pathlib import Path

from coref_corpus import export_inputs, source_sha256


class ExportTests(unittest.TestCase):
    def setUp(self):
        tmp = self.enterContext(tempfile.TemporaryDirectory())
        self.path = Path(tmp) / "panel.json"
        self.sidecar = self.path.with_name("panel.json.provenance.json")
        self.case = {"id": "opaque-id", "text": "José called.", "group": "work",
                     "gold": "unscored", "provenance": {"notice": "fixture terms", "publisher_id": "key"}}

    def export(self):
        export_inputs(self.path, [self.case], self.path.parent / "source.bin", "fixture", {"revision": "pin"})

    def test_private_pair_binds_text_and_original_source(self):
        self.export()
        self.assertEqual(json.loads(self.path.read_bytes()), [{"id": "opaque-id", "text": "José called."}])
        lineage = json.loads(self.sidecar.read_bytes())
        self.assertEqual(lineage["source_record"]["revision"], "pin")
        self.assertEqual(lineage["cases"][0]["source_sha256"], source_sha256("José called."))
        self.assertEqual(lineage["cases"][0]["provenance"], self.case["provenance"])
        for f in (self.path, self.sidecar):
            self.assertEqual(stat.S_IMODE(f.stat().st_mode), 0o600)

    def test_existing_artifacts_survive_export_collision(self):
        for existing, other in ((self.path, self.sidecar), (self.sidecar, self.path)):
            with self.subTest(path=existing):
                existing.write_bytes(b"previous export")
                with self.assertRaises(FileExistsError):
                    self.export()
                self.assertEqual(existing.read_bytes(), b"previous export")
                self.assertFalse(other.exists())
                existing.unlink()
