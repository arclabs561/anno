import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import replay_typed_spans as adapter

ROOT = Path(__file__).resolve().parents[2]
CASE = "message-channel"


class TypedReplay(unittest.TestCase):
    def test_native_metadata_required_for_replay(self):
        case = adapter._case(ROOT, CASE)
        good = {"id": CASE, "entities": [], "model": "stacked",
                "source_sha256": adapter.judge.source_sha256(case["text"])}
        with patch.object(adapter, "binary_and_digest", return_value=(Path("anno"), "a" * 64)), \
                patch.object(adapter.judge.subprocess, "run") as run:
            run.return_value.stdout = json.dumps(good)
            self.assertEqual(adapter.replay(ROOT, CASE)["outcome"], "pass")
            for label, row in [
                ("wrong backend", {**good, "model": "heuristic"}),
                ("missing backend", {k: v for k, v in good.items() if k != "model"}),
                ("missing source", {k: v for k, v in good.items() if k != "source_sha256"}),
                ("wrong source", {**good, "source_sha256": "0" * 64}),
            ]:
                with self.subTest(label=label), self.assertRaises(ValueError):
                    run.return_value.stdout = json.dumps(row)
                    adapter.replay(ROOT, CASE)

    def test_changed_adapter_invalidates_oracle(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            scripts = Path("scripts/qa")
            for path in ["replay_typed_spans.py", "check_span_assertions.py", "domain_coverage.py",
                         "fixtures/typed-span-assertions.json"]:
                dest = root / scripts / path
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / scripts / path, dest)
            before = adapter.describe(root, CASE)
            changed = root / scripts / "replay_typed_spans.py"
            with changed.open("ab") as stream:
                stream.write(b"\n# Different adapter revision\n")
            with self.assertRaisesRegex(ValueError, "adapter"):
                adapter.describe(root, CASE)
            with patch.object(adapter, "__file__", str(changed)):
                after = adapter.describe(root, CASE)
            self.assertEqual(before["source_sha256"], after["source_sha256"])
            self.assertNotEqual(before["oracle_sha256"], after["oracle_sha256"])
