# SPDX-License-Identifier: MIT OR Apache-2.0
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

    def test_release_requires_all_four_platforms_and_matching_source(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            platforms = [
                "manylinux_2_17_x86_64.manylinux2014_x86_64",
                "manylinux_2_17_aarch64", "macosx_11_0_arm64", "win_amd64",
            ]
            for tag in platforms:
                (directory / f"anno_py-0.2.0-cp39-abi3-{tag}.whl").write_bytes(b"wheel")
            with self.assertRaises(ValueError):
                release_inventory(directory)
            sdist = directory / "anno_py-0.1.0.tar.gz"
            sdist.write_bytes(b"sdist")
            with self.assertRaises(ValueError):
                release_inventory(directory)
            sdist.rename(directory / "anno_py-0.2.0.tar.gz")
            for ref in ("refs/tags/anno-py-v0.2.0", "refs/heads/main"):
                with self.subTest(ref=ref):
                    self.assertEqual(release_inventory(directory, ref)[0], "0.2.0")
            with self.assertRaisesRegex(ValueError, "Tag .* does not match"):
                release_inventory(directory, "refs/tags/anno-py-v0.1.0")
            version, hashes = release_inventory(directory)
            self.assertEqual(version, "0.2.0")
            self.assertEqual(len(hashes), 5)

            other_job = directory / "another-job"
            other_job.mkdir()
            duplicate = other_job / f"anno_py-0.2.0-cp39-abi3-{platforms[0]}.whl"
            duplicate.write_bytes(b"another wheel")
            with self.assertRaisesRegex(ValueError, "Duplicate Linux x86_64"):
                release_inventory(directory)
            duplicate.unlink()
            (directory / "anno_py-0.2.0-cp39-abi3-win_amd64.whl").unlink()
            with self.assertRaisesRegex(ValueError, "Missing release wheels: Windows x64"):
                release_inventory(directory)

            mixed = directory / "anno_py-0.2.0-cp39-abi3-win_amd64.macosx_11_0_arm64.whl"
            mixed.write_bytes(b"wheel")
            with self.assertRaisesRegex(ValueError, "Unexpected platform tags"):
                release_inventory(directory)
