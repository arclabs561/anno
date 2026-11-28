"""S3 maintenance commands require an operator-selected destination; no network."""

import base64
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import sys
import tarfile
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

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.calls_path = self.root / "calls"
        self.env = {
            "HOME": str(self.root), "PATH": f"{self.bin}:{os.defpath}",
            "CALLS": str(self.calls_path), "PYTHONPATH": str(ROOT / "scripts/qa"),
        }
        # Always mask cloud CLIs, Cargo and metadata requests.
        for tool in ("aws", "cargo", "trainctl", "curl", "uv"):
            self.fake(tool)

    def fake(self, name, code=""):
        path = self.bin / name
        path.write_text('#!/bin/sh\nprintf "%s %s\\n" "${0##*/}" "$*" >> "$CALLS"\n' + code)
        path.chmod(0o755)

    def run_tool(self, args, **overrides):
        return subprocess.run(
            args, env={**self.env, **overrides}, cwd=self.root, text=True,
            capture_output=True, timeout=15,
        )

    def test_s3_actions_require_destination_before_external_calls(self):
        actions = {
            "sync_datasets_s3.sh": ["upload", "download", "download-datasets", "sync", "ls", "size", "upload-script"],
            "sync_models_s3.sh": ["upload", "download", "ls"],
            "upload_safetensors_s3.sh": [""],
            "spot/trainctl-bridge.sh": ["sync-res", "sync-ds", "upload-src", "cache-upload", "cache-download"],
            "spot/setup.sh": [""], "spot/worker.sh": [""], "spot/upload-source.sh": [""],
            "spot/launch.sh": [""], "spot/launch-trainctl.sh": [""],
        }
        for script, commands in actions.items():
            for cmd in commands:
                for value in ("", "s3://", "s3://example-bucket/objects"):
                    with self.subTest(script=script, cmd=cmd, bucket=value):
                        result = self.run_tool(["bash", ROOT / "scripts" / script, *cmd.split()], ANNO_S3_BUCKET=value)
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn("ANNO_S3_BUCKET", result.stderr)
        orchestrator = ROOT / "scripts/spot/orchestrate.py"
        for cmd in ("generate", "launch", "results", "status", "full"):
            result = self.run_tool([sys.executable, orchestrator, cmd])
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("ANNO_S3_BUCKET", result.stderr)
        missing = self.run_tool([sys.executable, ROOT / "scripts/spot/aggregate.py", "--download"])
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn("ANNO_S3_BUCKET", missing.stderr)
        self.assertFalse(self.calls_path.exists())

    def test_helpers_and_local_commands_run_offline(self):
        for script in ("sync_datasets_s3.sh", "sync_models_s3.sh", "upload_safetensors_s3.sh", "spot/setup.sh",
                       "spot/worker.sh", "spot/launch.sh", "spot/launch-trainctl.sh", "spot/trainctl-bridge.sh",
                       "spot/upload-source.sh"):
            result = self.run_tool(["bash", ROOT / "scripts" / script, "--help"])
            self.assertEqual(result.returncode, 0, (script, result.stderr))
        for script in ("audit_dataset_health.py", "enrich_dataset_registry.py", "spot/orchestrate.py", "spot/aggregate.py"):
            result = self.run_tool([sys.executable, ROOT / "scripts" / script, "--help"])
            self.assertEqual(result.returncode, 0, (script, result.stderr))
        for script in ("sync_datasets_s3.sh", "sync_models_s3.sh"):
            result = self.run_tool(["bash", ROOT / "scripts" / script, "status"])
            self.assertEqual(result.returncode, 0, result.stderr)
        result = self.run_tool([sys.executable, ROOT / "scripts/spot/orchestrate.py", "generate", "--dry-run",
                                "--backends", "heuristic", "--datasets", "WikiGold", "--seeds", "7"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Generated 1 tasks", result.stdout)
        result = self.run_tool([sys.executable, ROOT / "scripts/spot/aggregate.py"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.calls_path.exists())

    def test_spot_source_selection_and_archive_inputs(self):
        recipe = (ROOT / "justfile").read_text().split("spot-upload-src:\n", 1)[1].split("\n\n", 1)[0].strip().lstrip("@")
        script = ROOT / recipe
        self.assertTrue(script.is_file())
        self.fake("git")
        for mode in ("aws", "trainctl"):
            for generic, spot, selected in (("s3://example-source-bucket/", "", "example-source-bucket"),
                                             ("example-other", "example-spot", "example-spot")):
                result = self.run_tool(["bash", script, mode], ANNO_S3_BUCKET=generic, ANNO_SPOT_BUCKET=spot)
                self.assertEqual(result.returncode, 0, result.stderr)
                calls = self.calls_path.read_text().splitlines()
                self.assertEqual(len(calls), 2)
                self.assertTrue(calls[0].startswith("git -C "))
                self.assertIn("HEAD -- Cargo.toml Cargo.lock README.md ", calls[0])
                self.assertTrue(calls[1].endswith(f" s3://{selected}/src/anno-src.tar.gz"))
                self.assertTrue(calls[1].startswith(f"{mode} s3 "))
                self.calls_path.unlink()

    def test_spot_archive_excludes_corpora_but_includes_build_assets(self):
        # Write blobs and a tree to a private temp object store. The current
        # working tree may contain uncommitted build inputs. No commits or index changes.
        raw_git = shutil.which("git")
        def git(*args, env=None):
            return subprocess.run([raw_git, "-C", ROOT, *args], env=env, text=True, capture_output=True, check=True).stdout.strip()
        objects = self.root / "objects"
        objects.mkdir()
        repo_objects = Path(git("rev-parse", "--path-format=absolute", "--git-common-dir")) / "objects"
        env = {**self.env, "GIT_INDEX_FILE": str(self.root / "scratch-index"),
               "GIT_OBJECT_DIRECTORY": str(objects), "GIT_ALTERNATE_OBJECT_DIRECTORIES": str(repo_objects)}
        git("read-tree", "HEAD", env=env)
        git("add", "--", "scripts/s3_bucket.sh", "crates/anno-eval/examples/public_benchmark_report.rs", env=env)
        tree = git("write-tree", env=env)
        archive = self.root / "source.tar.gz"
        result = self.run_tool(
            ["bash", "-c", 'source "$1"; archive_worker_source "$2" "$3"', "_",
             ROOT / "scripts/spot/upload-source.sh", archive, tree], **env,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        with tarfile.open(archive) as source:
            names = {item.name for item in source if item.isfile()}
        for required in ("Cargo.toml", "Cargo.lock", "scripts/spot/worker.sh", "scripts/s3_bucket.sh",
                         "crates/anno-cli/src/main.rs", "crates/anno/benches/backends.rs", "crates/anno/src/viz/report.css",
                         "crates/anno-eval/src/eval/shared_cache_exclusions.tsv",
                         "crates/anno-eval/examples/public_benchmark_report.rs"):
            self.assertIn(required, names)
        self.assertFalse(any("/tests/" in name or "/fixtures/" in name or
                             name.startswith(("docs/", "testdata/", "fuzz/")) for name in names))

    def test_no_dataset_publication_including_linked_raw_sources(self):
        cache = self.root / "local cache"
        for category in ("datasets", "models", "scripts"):
            (cache / category).mkdir(parents=True)
        raw = cache / "datasets" / "PreCo.cache"
        raw.write_bytes(b"private source")
        nested = cache / "datasets" / "by-sha256" / ("1" * 64)
        nested.mkdir(parents=True)
        (nested / "WikiGold.cache").write_bytes(b"private source 2")
        (cache / "models" / "test-model.bin").write_bytes(b"model")
        (cache / "scripts" / "tool.py").write_text("pass")
        for cli in ("aws", "s5cmd"):
            self.fake(cli)
            result = self.run_tool(["bash", ROOT / "scripts/sync_datasets_s3.sh", "upload"],
                                   ANNO_CACHE_DIR=str(cache), ANNO_S3_BUCKET="s3://example-dataset-cache/")
            self.assertEqual(result.returncode, 0, result.stderr)
            calls = self.calls_path.read_text()
            self.assertNotIn("datasets/", calls)
            self.assertIn("s3://example-dataset-cache/models/", calls)
            manifest = json.loads((cache / "manifest.json").read_text())
            self.assertEqual(manifest["artifacts"]["datasets"], [])
            self.calls_path.unlink()
            link = cache / "scripts" / "disguised.py"
            link.symlink_to(raw)
            result = self.run_tool(["bash", ROOT / "scripts/sync_datasets_s3.sh", "upload"],
                                   ANNO_CACHE_DIR=str(cache), ANNO_S3_BUCKET="example-dataset-cache")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("link outside scripts", result.stderr)
            self.assertFalse(any(" sync " in line or " cp " in line
                                 for line in self.calls_path.read_text().splitlines()))
            link.unlink()
            self.calls_path.unlink()

    def test_worker_template_and_iam_match_selected_bucket(self):
        setup = ROOT / "scripts/spot/setup.sh"
        command = ('source "$1"; SETUP_DIR="$2"; BUCKET=example-fleet; ROLE_NAME=worker-role; INSTANCE_PROFILE=worker-profile; ACCOUNT_ID=123456789012; '
                   'QUEUE_NAME=example-queue; QUEUE_URL=https://sqs.example/queue; '
                   'LAUNCH_TEMPLATE_NAME="anno-eval-worker-${BUCKET}"; create_iam_role; create_launch_template')
        result = self.run_tool(["bash", "-c", command, "_", setup, self.root])
        self.assertEqual(result.returncode, 0, result.stderr)
        policy = json.loads((self.root / "role-policy.json").read_text())["Statement"]
        self.assertEqual([s["Resource"] for s in policy if s["Action"] == "s3:PutObject"],
                         ["arn:aws:s3:::example-fleet/results/*"])
        for stmt in policy:
            if isinstance(stmt["Action"], str) and stmt["Action"].startswith("s3:"):
                self.assertIn("arn:aws:s3:::example-fleet", str(stmt["Resource"]))
        template = json.loads((self.root / "launch-template.json").read_text())
        self.assertEqual(template["LaunchTemplateName"], "anno-eval-worker-example-fleet")
        user_data = base64.b64decode(template["LaunchTemplateData"]["UserData"], validate=True).decode()
        self.assertIn("export ANNO_SPOT_BUCKET=example-fleet", user_data)
        self.assertIn("export ANNO_SPOT_QUEUE_URL=https://sqs.example/queue", user_data)
        self.assertIn('s3://$ANNO_SPOT_BUCKET/src/anno-src.tar.gz', user_data)
        self.assertIn('s3://$ANNO_SPOT_BUCKET/models/ner/', user_data)
        syntax = subprocess.run(["bash", "-n"], input=user_data, text=True, capture_output=True)
        self.assertEqual(syntax.returncode, 0, syntax.stderr)

    def test_saved_config_priority_and_explicit_collection(self):
        spec = importlib.util.spec_from_file_location("spot_storage", ROOT / "scripts/spot/storage.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config = self.root / "config.env"
        config.write_text("ANNO_SPOT_BUCKET=s3://saved-bucket/\n")
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                module.selected_bucket(config_file=config)
        config.write_text("ANNO_SPOT_CONFIG_VERSION=2\nANNO_SPOT_BUCKET=s3://saved-bucket/\n")
        for generic, spot, expected in (("", "", "saved-bucket"), ("generic-bucket", "", "generic-bucket"),
                                         ("generic-bucket", "s3://spot-bucket/", "spot-bucket")):
            with patch.dict(os.environ, {"ANNO_S3_BUCKET": generic, "ANNO_SPOT_BUCKET": spot}, clear=True):
                self.assertEqual(module.selected_bucket(config_file=config), expected)
                self.assertEqual(module.selected_bucket("s3://flag-bucket/", config), "flag-bucket")
            command = 'source "$1"; anno_spot_config "$2"; anno_bucket_name "$ANNO_SPOT_BUCKET" ANNO_SPOT_BUCKET'
            shell = self.run_tool(["bash", "-c", command, "_", ROOT / "scripts/s3_bucket.sh", config],
                                  ANNO_S3_BUCKET=generic, ANNO_SPOT_BUCKET=spot)
            self.assertEqual(shell.stdout.strip(), expected, shell.stderr)
        config.write_text(config.read_text() + "ANNO_SPOT_QUEUE=existing-tasks\nANNO_SPOT_QUEUE_URL=https://sqs.example/existing-tasks\n")
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(module.fleet_settings("saved-bucket", config)[1:], ("existing-tasks", "https://sqs.example/existing-tasks"))
            self.assertEqual(module.fleet_settings("other.bucket", config)[1:], ("anno-eval-other_bucket", ""))
        with patch.dict(os.environ, {"ANNO_SPOT_QUEUE": "operator-tasks"}, clear=True):
            self.assertEqual(module.fleet_settings("saved-bucket", config)[1:], ("operator-tasks", ""))
            self.assertEqual(module.fleet_settings("other.bucket", config)[1:], ("operator-tasks", ""))
        aggregate = ROOT / "scripts/spot/aggregate.py"
        result = self.run_tool([sys.executable, aggregate, "--download", "--bucket", "s3://flag-bucket/"],
                               ANNO_SPOT_BUCKET="other-bucket")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("aws s3 ls s3://flag-bucket/results/ --recursive", self.calls_path.read_text())

    def test_spot_publishes_only_aggregate_receipts_and_classified_failures(self):
        capture = self.root / "uploads"
        capture.mkdir()
        self.fake("aws", """case "$3" in
            -) cat > "$CAPTURE/result.error" ;;
            *) cp "$3" "$CAPTURE/result.${4##*.}" ;;
        esac""")
        target = self.root / "target"
        projector = target / "release/examples/public_benchmark_report"
        projector.parent.mkdir(parents=True)
        actual = os.environ.get("ANNO_TEST_PUBLIC_BENCHMARK_REPORT")
        if actual:
            projector.symlink_to(Path(actual).resolve(strict=True))
        else:
            # This suite also runs before Cargo. The typed projector has its own
            # crate tests. A binary path enables the full worker round trip.
            projector.write_text("""#!/bin/sh
                jq -e '.results[0] | .dataset == "WikiGold"' "$1" >/dev/null || exit 1
                printf '| WikiGold | heuristic | 25.0 | 50.0 | 16.7 | 10 | 15 |\\n' > "$2"
                jq 'del(.results[0].backend_display, .results[0].metrics.PRIVATE_PASSAGE)' "$1" > "$3"
            """)
            projector.chmod(0o755)
        private = {"results": [{"task": "NER", "dataset": "WikiGold", "backend": "heuristic", "seed": 7,
                                "backend_display": "PRIVATE_PASSAGE", "success": True,
                                "metrics": {"f1": 0.25, "precision": 0.5, "recall": 0.167, "PRIVATE_PASSAGE": 9},
                                "num_examples": 10, "duration_ms": 15}],
                   "summary": {"total_combinations": 1, "successful": 1, "failed": 0, "skipped": 0,
                               "tasks": ["NER"], "datasets": ["WikiGold"], "backends": ["heuristic"]}}
        work = self.root / "local.receipt"
        work.mkdir()
        (work / "report.json").write_text(json.dumps(private))
        (work / "report.md").write_text("PRIVATE_PASSAGE")
        command = ('source "$1"; BUCKET=example-fleet; CARGO_TARGET_DIR="$2"; '
                   'publish_task_result "$3" 0 1')
        worker = ROOT / "scripts/spot/worker.sh"
        result = self.run_tool(["bash", "-c", command, "_", worker, target, work], CAPTURE=str(capture))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(work.exists())
        self.assertEqual({p.suffix for p in capture.iterdir()}, {".md", ".json"})
        for path in capture.iterdir():
            self.assertNotIn("PRIVATE_PASSAGE", path.read_text())
        self.assertEqual(json.loads((capture / "result.json").read_text())["results"][0]["metrics"]["f1"], .25)
        self.assertIn("_meta: backend=heuristic dataset=WikiGold seed=7", (capture / "result.md").read_text())
        with patch.object(sys, "path", [str(ROOT), *sys.path]):
            from scripts.spot.aggregate import parse_result_file
            from scripts.spot.orchestrate import expand_result_artifact
        record = parse_result_file(capture / "result.md")
        self.assertEqual((record["backend"], record["dataset"], record["seed"], record["f1"]),
                         ("heuristic", "WikiGold", 7, 25.0))
        native = expand_result_artifact(json.loads((capture / "result.json").read_text()))[0]
        self.assertEqual((native["_meta"]["seed"], native["metrics"]["f1"]), (7, .25))
        work.mkdir()
        (work / "report.json").write_text("PRIVATE_PASSAGE")
        result = self.run_tool(["bash", "-c", command, "_", worker, target, work],
                               CAPTURE=str(capture), ANNO_SPOT_RETAIN_FAILURE_ARTIFACTS="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((work / "report.json").is_file())
        self.assertEqual(json.loads((capture / "result.error").read_text()),
                         {"error": "projection_failed", "exit_code": 0})
        calls = self.calls_path.read_text()
        self.assertNotIn("PRIVATE_PASSAGE", calls)
        self.assertNotIn("/report.json", calls)
        self.assertNotIn("/report.md", calls)
        self.assertIn("/results/failures/", calls)

    def test_health_inventory_distinguishes_unmeasured_and_absent(self):
        data = self.root / "url-health.json"
        inventory = self.root / "objects.txt"
        output = self.root / "report.json"
        data.write_text(json.dumps({"health_results": {"WikiGold": {"primary": {"status": "ok"}},
                                                       "Other": {"primary": {"status": "ok"}}}}))
        inventory.write_text("2026-01-01 09:00:00 23 datasets/WikiGold.cache\n")
        args = [sys.executable, ROOT / "scripts/audit_dataset_health.py", "--enrichment", data, "--output", output]
        self.assertEqual(self.run_tool(args).returncode, 0)
        report = json.loads(output.read_text())
        self.assertEqual(report["summary"]["cache_unknown"], 2)
        self.assertIsNone(report["cache_unknown"][0]["in_s3"])
        self.assertEqual(self.run_tool([*args, "--s3-inventory", inventory]).returncode, 0)
        report = json.loads(output.read_text())
        self.assertEqual([r["name"] for r in report["healthy_cached"]], ["WikiGold"])
        self.assertEqual([r["name"] for r in report["healthy_not_cached"]], ["Other"])
        self.assertFalse(self.calls_path.exists())


if __name__ == "__main__":
    unittest.main()
