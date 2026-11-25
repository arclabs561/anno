"""Publication may skip existing filenames. Only identical bytes satisfy the check."""

from pathlib import Path
import tempfile
import unittest

from verify_pypi import missing_files, release_inventory


class ReleaseInventoryTest(unittest.TestCase):
    def test_existing_files(self):
        expected = {"example.whl": "aaa", "source.tar.gz": "bbb"}
        rows = [{"filename": name, "digests": {"sha256": digest}, "yanked": False}
                for name, digest in expected.items()]
        self.assertEqual(missing_files(expected, rows), [])
        self.assertEqual(missing_files(expected, rows[:1]), ["source.tar.gz"])
        for changes in ({"digests": {"sha256": "ccc"}}, {"yanked": True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                missing_files(expected, [rows[0], {**rows[1], **changes}])

    def test_release_requires_source_and_matching_wheel(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / "anno_py-0.2.0-cp39-abi3-linux_x86_64.whl").write_bytes(b"wheel")
            with self.assertRaises(ValueError):
                release_inventory(directory)
            sdist = directory / "anno_py-0.1.0.tar.gz"
            sdist.write_bytes(b"sdist")
            with self.assertRaises(ValueError):
                release_inventory(directory)
            sdist.rename(directory / "anno_py-0.2.0.tar.gz")
            version, hashes = release_inventory(directory)
            self.assertEqual(version, "0.2.0")
            self.assertEqual(len(hashes), 2)
