"""Offline regressions for model preparation, repair and shared Hub cache reads."""

import hashlib
import shutil
import sys
import tempfile
import types
import unittest
from contextlib import nullcontext
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _prepare_snapshot import pinned_inputs, prepare


def sha(data):
    return hashlib.sha256(data).hexdigest()


def blob(data):
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


class PrepareSnapshotTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.dest = self.root / "model"
        self.files = {
            "config.json": {"git_blob": blob(b"config"), "size": 6},
            "nested/weights": {"sha256": sha(b"weights")},
        }

    def run_prepare(self, fetch, cached=None, **kwargs):
        hub = types.SimpleNamespace(
            snapshot_download=fetch,
            try_to_load_from_cache=cached or Mock(return_value=None),
        )
        modules = {"huggingface_hub": hub}
        if find_spec("filelock") is None:
            modules["filelock"] = types.SimpleNamespace(
                FileLock=lambda path: nullcontext()
            )
        with patch.dict(sys.modules, modules):
            prepare(self.dest, "owner/model", "a" * 40, self.files, **kwargs)

    def test_corrupt_fetch_does_not_publish_a_partial_directory(self):
        def fetch(*, local_dir, **kwargs):
            (local_dir / "nested").mkdir(parents=True)
            (local_dir / "config.json").write_bytes(b"config")
            (local_dir / "nested/weights").write_bytes(b"damaged")
            return str(local_dir)

        with self.assertRaisesRegex(ValueError, "does not match"):
            self.run_prepare(fetch)
        self.assertFalse(self.dest.exists())
        self.assertFalse(any(p.is_dir() for p in self.root.iterdir()))

    def test_interrupted_repair_preserves_valid_and_preexisting_files(self):
        (self.dest / "nested").mkdir(parents=True)
        (self.dest / "config.json").write_bytes(b"config")
        weights = self.dest / "nested/weights"
        weights.write_bytes(b"old")
        notes = self.dest / "notes"
        notes.write_bytes(b"keep")

        def interrupted(*, local_dir, allow_patterns, **kwargs):
            self.assertEqual(allow_patterns, ["nested/weights"])
            (local_dir / "nested").mkdir(parents=True)
            (local_dir / "nested/weights").write_bytes(b"unfinished")
            raise ConnectionError("interrupted")

        with self.assertRaises(ConnectionError):
            self.run_prepare(interrupted)
        self.assertEqual(weights.read_bytes(), b"old")
        self.assertEqual(notes.read_bytes(), b"keep")

        def retry(*, local_dir, allow_patterns, **kwargs):
            self.assertEqual(allow_patterns, ["nested/weights"])
            (local_dir / "nested").mkdir(parents=True)
            (local_dir / "nested/weights").write_bytes(b"weights")
            return str(local_dir)

        self.run_prepare(retry)
        self.assertEqual(weights.read_bytes(), b"weights")
        self.assertEqual(notes.read_bytes(), b"keep")
        fetch = Mock(side_effect=AssertionError("unexpected fetch"))
        self.run_prepare(fetch)
        fetch.assert_not_called()

    def test_offline_hub_cache_is_preserved_and_published_files_are_regular(self):
        cache = self.root / "cache"
        (cache / "nested").mkdir(parents=True)
        (cache / "blob").write_bytes(b"weights")
        (cache / "nested/weights").symlink_to("../blob")
        (cache / "config.json").write_bytes(b"config")
        fetch = Mock(side_effect=AssertionError("unexpected fetch"))
        self.run_prepare(fetch, cached=lambda repo, name, revision: str(cache / name))
        fetch.assert_not_called()
        self.assertTrue((cache / "nested/weights").is_symlink())
        self.assertEqual((cache / "config.json").read_bytes(), b"config")
        shutil.rmtree(cache)
        self.assertFalse((self.dest / "nested/weights").is_symlink())
        self.assertEqual((self.dest / "nested/weights").read_bytes(), b"weights")

    def test_another_otter_variant_survives_unchanged(self):
        self.dest.mkdir()
        (self.dest / "config.json").write_bytes(b"other")
        fetch = Mock(side_effect=AssertionError("unexpected fetch"))
        with self.assertRaisesRegex(ValueError, "another model variant"):
            self.run_prepare(fetch, other_configs=[{"git_blob": blob(b"other")}])
        self.assertEqual((self.dest / "config.json").read_bytes(), b"other")

    def test_repair_rejects_a_conflicting_subdirectory_before_replacing_files(self):
        self.dest.mkdir()
        (self.dest / "config.json").write_bytes(b"old")
        (self.dest / "nested").symlink_to(self.root)
        cache = self.root / "cache"
        (cache / "nested").mkdir(parents=True)
        (cache / "config.json").write_bytes(b"config")
        (cache / "nested/weights").write_bytes(b"weights")
        with self.assertRaisesRegex(ValueError, "Cannot install"):
            self.run_prepare(
                Mock(), cached=lambda repo, name, revision: str(cache / name)
            )
        self.assertEqual((self.dest / "config.json").read_bytes(), b"old")
        self.assertFalse((self.root / "weights").exists())

    def test_export_inputs_exclude_stale_optional_sidecars(self):
        self.dest.mkdir()
        (self.dest / "config.json").write_bytes(b"config")
        (self.dest / "adapter_config.json").write_bytes(b"unrelated adapter")
        (self.dest / "nested").mkdir()
        (self.dest / "nested/weights").symlink_to("../config.json")
        files = {name: sha(b"config") for name in self.files}
        stage = pinned_inputs(self.dest, self.root / "staging", files)
        self.assertFalse((stage / "adapter_config.json").exists())
        self.assertEqual((stage / "nested/weights").read_bytes(), b"config")
        self.assertFalse((stage / "nested/weights").is_symlink())
        (self.dest / "config.json").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "Unexpected SHA-256"):
            pinned_inputs(self.dest, self.root / "second-staging", files)


if __name__ == "__main__":
    unittest.main()
