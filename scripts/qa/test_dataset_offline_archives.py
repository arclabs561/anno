"""Cover integrity, retries and concurrent transfers on local storage."""

from concurrent.futures import ThreadPoolExecutor
import errno
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Barrier
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import offline_dataset_archives as archives
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

    def test_receipt_paths_are_relative(self):
        for path in [
            "../elsewhere",
            "/elsewhere",
            r"C:\elsewhere",
            r"..\elsewhere",
            "dir/C:elsewhere",
        ]:
            with self.subTest(path=path), self.assertRaises(ValueError):
                local_path(Path("."), path)

    def test_simultaneous_split_and_restore_without_hard_links(self):
        with TemporaryDirectory() as temp, ThreadPoolExecutor(max_workers=2) as workers:
            root = Path(temp)
            path = root / "sample.zip"
            original = b"1234567890" * 10
            path.write_bytes(original)
            publish = archives.publish
            barrier = Barrier(2, timeout=10)

            def racing_publish(staged, dest, record):
                if dest.name in {path.name, path.name + ".part001"}:
                    barrier.wait()
                publish(staged, dest, record)

            with patch.object(archives, "publish", racing_publish), patch(
                "os.link", side_effect=OSError(errno.EPERM, "hard links unavailable")
            ):
                results = [
                    workers.submit(split_archive, root, path.name, 30) for _ in range(2)
                ]
                first, second = [result.result() for result in results]
                self.assertEqual(first, second)
                path.unlink()
                results = [workers.submit(restore, root, first) for _ in range(2)]
                for result in results:
                    result.result()
            self.assertEqual(path.read_bytes(), original)

    def test_conflicting_simultaneous_restores_preserve_the_winner(self):
        with TemporaryDirectory() as temp, ThreadPoolExecutor(max_workers=2) as workers:
            root = Path(temp)
            publish = archives.publish
            barrier = Barrier(2, timeout=10)

            def racing_publish(staged, dest, record):
                barrier.wait()
                publish(staged, dest, record)

            records = []
            for name in ["first", "other"]:
                raw = name.encode()
                (root / name).write_bytes(raw)
                digest = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
                records.append(
                    dict(**digest, path="result.zip", parts=[dict(path=name, **digest)])
                )
            with patch.object(archives, "publish", racing_publish):
                results = [workers.submit(restore, root, record) for record in records]
                errors = [result.exception() for result in results]
            self.assertEqual(sum(err is None for err in errors), 1)
            self.assertIsInstance(
                next(err for err in errors if err is not None), ValueError
            )
            check(root / "result.zip", records[errors.index(None)])

    def test_windows_rename_collision_validates_existing_contents(self):
        # Exercise Windows' no-overwrite rename contract on any test host.
        with TemporaryDirectory() as temp:
            root = Path(temp)
            staged, dest = root / "prepared", root / "output"
            staged.write_bytes(b"expected")
            dest.write_bytes(b"expected")
            record = dict(bytes=8, sha256=hashlib.sha256(b"expected").hexdigest())
            with patch.object(archives, "os", wraps=os) as platform, patch.object(
                Path, "rename", side_effect=FileExistsError
            ):
                platform.name = "nt"
                archives.publish(staged, dest, record)
                dest.write_bytes(b"existing")
                with self.assertRaisesRegex(ValueError, "Size or SHA-256"):
                    archives.publish(staged, dest, record)
            self.assertEqual(dest.read_bytes(), b"existing")

    def test_abrupt_exit_leaves_partial_work_retryable(self):
        # Exit the child without running `finally`, as with a killed transfer.
        child = """
import json, os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import offline_dataset_archives as archives
root = Path(sys.argv[2])
stop_at = sys.argv[3]
publish = archives.publish
def interrupt(staged, dest, receipt):
    if dest.name == stop_at:
        os._exit(23)
    publish(staged, dest, receipt)
archives.publish = interrupt
if stop_at == 'sample.zip':
    archives.restore(root, json.loads((root / 'receipt.json').read_text()))
else:
    archives.split_archive(root, 'sample.zip', 5)
"""
        with TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "sample.zip"
            original = b"whole archive contents"
            path.write_bytes(original)
            args = [
                sys.executable,
                "-c",
                child,
                str(Path(archives.__file__).parent),
                str(root),
            ]
            self.assertEqual(
                subprocess.run([*args, "sample.zip.part002"]).returncode, 23
            )
            self.assertEqual((root / "sample.zip.part001").read_bytes(), original[:5])
            self.assertFalse((root / "sample.zip.part002").exists())
            record = split_archive(root, path.name, 5)
            (root / "receipt.json").write_text(json.dumps(record))
            path.unlink()
            self.assertEqual(subprocess.run([*args, path.name]).returncode, 23)
            self.assertFalse(path.exists())
            restore(root, record)
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
