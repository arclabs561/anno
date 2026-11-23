from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import prepare_matrix_timex as timex


class PrepareTimexTest(unittest.TestCase):
    def test_stale_manifest_and_repeated_preparation(self):
        data = b"verified test input"
        digest = sha256(data).hexdigest()

        def download(args, *, check):
            self.assertTrue(check)
            Path(args[args.index("-o") + 1]).write_bytes(data)

        with TemporaryDirectory() as tmp, patch.object(timex, "SIZE", len(data)), \
                patch.object(timex, "DIGEST", digest), patch.object(timex.subprocess, "run", download):
            directory = Path(tmp)
            manifest = directory / "manifest.json"
            manifest.write_text(json.dumps({"version": 1, "entries": {
                timex.NAME: {"source_url": "old"}, "other.cache": {"unrelated": True}}}))
            timex.prepare(directory)
            self.assertEqual((directory / timex.NAME).read_bytes(), data)
            self.assertEqual(json.loads(manifest.read_text())["entries"], {"other.cache": {"unrelated": True}})
            with patch.object(timex.subprocess, "run", side_effect=AssertionError("unnecessary download")):
                timex.prepare(directory)

    def test_rejects_corrupt_download(self):
        def corrupt(args, *, check):
            Path(args[args.index("-o") + 1]).write_bytes(b"bad")

        with TemporaryDirectory() as tmp, patch.object(timex, "SIZE", 3), \
                patch.object(timex.subprocess, "run", corrupt):
            directory = Path(tmp)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                timex.prepare(directory)
            self.assertEqual(list(directory.iterdir()), [])
