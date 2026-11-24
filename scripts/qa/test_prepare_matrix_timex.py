from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import prepare_matrix_timex as timex


class PrepareTimexTest(unittest.TestCase):
    def test_local_source_replaces_only_its_own_manifest_entry(self):
        data = b"verified test input"
        with TemporaryDirectory() as tmp, patch.object(timex, "SIZE", len(data)), \
                patch.object(timex, "DIGEST", sha256(data).hexdigest()):
            directory = Path(tmp)
            source = directory / "licensed-local-input"
            source.write_bytes(data)
            manifest = directory / "manifest.json"
            manifest.write_text(json.dumps({"version": 1, "entries": {
                timex.NAME: {"source_url": "old"}, "other.cache": {"unrelated": True}}}))
            timex.prepare(directory, source)
            corpus = directory / timex.NAME
            self.assertEqual(corpus.read_bytes(), data)
            self.assertEqual(json.loads(manifest.read_text())["entries"], {"other.cache": {"unrelated": True}})
            original_inode = corpus.stat().st_ino
            timex.prepare(directory, corpus)
            self.assertEqual(corpus.stat().st_ino, original_inode)

    def test_invalid_source_leaves_existing_cache_alone(self):
        with TemporaryDirectory() as tmp, patch.object(timex, "SIZE", 3):
            directory = Path(tmp)
            source = directory / "incoming"
            source.write_bytes(b"bad")
            corpus = directory / timex.NAME
            corpus.write_bytes(b"existing")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                timex.prepare(directory, source)
            self.assertEqual(corpus.read_bytes(), b"existing")
