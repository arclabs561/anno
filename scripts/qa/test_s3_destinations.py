"""S3 maintenance commands require an operator-selected destination; no network."""

import importlib.util
import os
import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "registry_enrichment", ROOT / "scripts/apply_registry_enrichment.py"
)
ENRICHMENT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ENRICHMENT)


class S3DestinationTests(unittest.TestCase):
    def test_registry_urls_follow_configured_bucket(self):
        for bucket in ("example-dataset-cache", "s3://example-dataset-cache/"):
            with (
                self.subTest(bucket=bucket),
                patch.dict(os.environ, {"ANNO_S3_BUCKET": bucket}),
            ):
                content, count = ENRICHMENT.update_registry(
                    '\n    Example { categories: ["ner"] }',
                    {"Example": {"s3_path": "datasets/example.json"}},
                    {},
                )
                self.assertEqual(count, 1)
                self.assertIn(
                    "s3://example-dataset-cache/datasets/example.json", content
                )

    def test_registry_requires_destination_only_for_s3_updates(self):
        with patch.dict(os.environ, {"ANNO_S3_BUCKET": ""}):
            with self.assertRaisesRegex(ValueError, "ANNO_S3_BUCKET"):
                ENRICHMENT.update_registry(
                    "", {"Example": {"s3_path": "datasets/example.json"}}, {}
                )
            self.assertEqual(
                ENRICHMENT.update_registry("unchanged", {}, {}), ("unchanged", 0)
            )

    def test_spot_reader_uses_shared_filters_and_only_syncs_datasets(self):
        from shared_dataset_cache import excluded_patterns

        for cli, command in (("aws", "sync-ds"), ("s5cmd", "sync-datasets")):
            with self.subTest(cli=cli), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                cache = root / "local cache"
                calls = root / "arguments"
                for tool in ("bash", "dirname", "mkdir", "python3"):
                    (root / tool).symlink_to(shutil.which(tool))
                program = root / cli
                program.write_text(
                    '#!/bin/sh\nprintf "%s\\0" "$@" >> "$CALLS"\nexit "${FAKE_STATUS:-0}"\n'
                )
                program.chmod(0o755)
                env = {
                    "PATH": str(root),
                    "HOME": str(root),
                    "CALLS": str(calls),
                    "ANNO_SPOT_BUCKET": "example-cache",
                    "ANNO_CACHE_DIR": str(cache),
                }
                bridge = [str(ROOT / "scripts/spot/trainctl-bridge.sh"), command]
                subprocess.run(
                    bridge, env=env, check=True, capture_output=True, timeout=10
                )
                args = calls.read_bytes().decode().split("\0")[:-1]
                filters = [
                    arg for pattern in excluded_patterns() for arg in ("--exclude", pattern)
                ]
                if cli == "aws":
                    expected = [
                        "s3", "sync", "s3://example-cache/datasets/",
                        str(cache / "datasets"), "--size-only", *filters,
                    ]
                else:
                    expected = [
                        "sync", "--size-only", *filters,
                        "s3://example-cache/datasets/*", f"{cache}/datasets/",
                    ]
                self.assertEqual(args, expected)
                failed = subprocess.run(
                    bridge, env={**env, "FAKE_STATUS": "7"}, capture_output=True, timeout=10
                )
                self.assertEqual(failed.returncode, 7)

    def test_source_upload_uses_only_explicit_destination(self):
        # Execute the repository recipe body with its declared Bash interpreter.
        # CI runs this suite without just installed; no upload logic is duplicated.
        recipe = textwrap.dedent(
            (ROOT / "justfile")
            .read_text()
            .split("spot-upload-src:\n", 1)[1]
            .split("\n\n", 1)[0]
        )
        for bucket in (
            None,
            "",
            "example-source-bucket",
            "s3://example-source-bucket/",
        ):
            with (
                self.subTest(bucket=bucket),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                calls = root / "calls"
                for name in ("git", "aws"):
                    program = root / name
                    program.write_text(
                        '#!/bin/sh\nprintf "%s\\n" "$0 $*" >> "$CALLS"\n'
                    )
                    program.chmod(0o755)
                environment = dict(os.environ)
                environment.pop("ANNO_S3_BUCKET", None)
                environment.update(
                    PATH=f"{root}:{environment['PATH']}", CALLS=str(calls)
                )
                if bucket is not None:
                    environment["ANNO_S3_BUCKET"] = bucket
                completed = subprocess.run(
                    ["bash", "-c", recipe],
                    env=environment,
                    cwd=root,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                if bucket:
                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    self.assertIn(
                        "aws s3 cp /tmp/anno-src.tar.gz "
                        "s3://example-source-bucket/src/anno-src.tar.gz",
                        calls.read_text(),
                    )
                else:
                    self.assertNotEqual(completed.returncode, 0)
                    self.assertIn("ANNO_S3_BUCKET", completed.stderr)
                    self.assertFalse(
                        calls.exists(), "Must fail before git or AWS executes"
                    )


if __name__ == "__main__":
    unittest.main()
