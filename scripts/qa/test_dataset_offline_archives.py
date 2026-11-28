"""Cover partial Drive downloads and archive integrity on restore."""

from pathlib import Path
import os
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from offline_dataset_archives import check, local_path, restore, split_archive


class ArchiveTest(unittest.TestCase):
    def test_restore_checks_each_part_and_whole_without_clobbering(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "sample.zip"
            original = b"ordered contents of the original archive"
            source.write_bytes(original)
            record = split_archive(root, source.name, 7)
            self.assertGreater(len(record["parts"]), 1)
            source.unlink()
            last = local_path(root, record["parts"][-1]["path"])
            tail = last.read_bytes()
            last.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "Size or SHA-256"):
                restore(root, record)
            self.assertFalse(source.exists())
            last.write_bytes(tail)
            with self.assertRaisesRegex(ValueError, "Size or SHA-256"):
                restore(root, dict(record, sha256="0" * 64))
            self.assertFalse(source.exists())
            restore(root, record)
            self.assertEqual(source.read_bytes(), original)
            check(source, record)
            source.write_bytes(b"existing")
            with self.assertRaises(ValueError):
                restore(root, record)
            self.assertEqual(source.read_bytes(), b"existing")

    def test_restore_checks_a_destination_published_by_another_writer(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            dest = root / "sample.zip"
            dest.write_bytes(b"archive")
            record = split_archive(root, dest.name, 3)
            link = os.link
            for content in (b"archive", b"another"):
                dest.unlink()
                def publish_first(source, target):
                    dest.write_bytes(content)
                    link(source, target)

                with patch("offline_dataset_archives.os.link", side_effect=publish_first):
                    if content == b"archive":
                        restore(root, record)
                    else:
                        with self.assertRaisesRegex(ValueError, "Size or SHA-256"):
                            restore(root, record)
                self.assertEqual(dest.read_bytes(), content)

    def test_receipt_paths_are_relative(self):
        with self.assertRaises(ValueError):
            local_path(Path("."), "../elsewhere")
        with self.assertRaises(ValueError):
            local_path(Path("."), "/elsewhere")


if __name__ == "__main__":
    unittest.main()
