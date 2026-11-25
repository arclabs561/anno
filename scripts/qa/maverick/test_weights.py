"""Offline smoke tests for the Maverick and generic weight converters."""
import hashlib
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import runpy
import os

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import convert_pytorch_to_safetensors as generic
import download_maverick_weights as maverick


class ConverterTests(unittest.TestCase):
    def test_pinned_hub_download_and_tensor_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weights, config = root / "weights.ckpt", root / "config.yaml"
            tensor = torch.tensor([1.5])
            # The converter must ignore training objects and retain parameter prefixes
            # below the Lightning wrapper, including integer buffers.
            torch.save({"state_dict": {"model.encoder.layer": tensor, "model.ids": torch.tensor([7])},
                        "training_metadata": Path("training-output")}, weights)
            config.write_text("# fixture")
            reference = root / "fixture.safetensors"
            subprocess.run(
                ["cargo", "run", "--quiet", "--locked", "-p", "anno", "--no-default-features",
                 "--features", "candle", "--example", "maverick_export", "--", weights, reference],
                cwd=maverick.ROOT, check=True,
            )
            info = dict(maverick.MODELS["ontonotes"], source_sha256=maverick.sha256_file(weights),
                        prepared_sha256=maverick.sha256_file(reference))

            def download(*, repo_id, revision, filename, cache_dir):
                self.assertEqual((repo_id, revision), (info["repo"], info["revision"]))
                return {"weights.ckpt": weights, "config.yaml": config}[filename]

            with patch.dict(maverick.MODELS, {"ontonotes": info}), patch.object(maverick, "verify_encoder"), patch.object(
                maverick, "hf_hub_download", side_effect=download
            ):
                output = maverick.download_model("ontonotes", root)
                self.assertTrue(torch.equal(load_file(output)["encoder.layer"], tensor))
                self.assertEqual(load_file(output)["ids"].tolist(), [7])
                with patch.dict(info, {"prepared_sha256": "0" * 64}):
                    with self.assertRaisesRegex(ValueError, "Prepared checkpoint digest mismatch"):
                        maverick.download_model("ontonotes", root)
                weights.write_bytes(b"incorrect file")
                with self.assertRaisesRegex(ValueError, "digest mismatch"):
                    maverick.download_model("ontonotes", root)
                self.assertTrue(torch.equal(load_file(output)["encoder.layer"], tensor))

    def test_worker_resolves_prepared_ids_and_rejects_unknown_architectures(self):
        worker_path = Path(__file__).resolve().parents[3] / "crates/anno/src/backends/coref/maverick_worker.py"
        resolve = runpy.run_path(str(worker_path))["prepared_checkpoint"]
        models = {name: info.copy() for name, info in maverick.MODELS.items()}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"ANNO_CACHE_DIR": tmp}):
            path = Path(tmp) / "models/maverick/maverick-ontonotes.safetensors"
            path.parent.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "prepare"):
                resolve("sapienzanlp/maverick-mes-ontonotes", models)
            with self.assertRaisesRegex(ValueError, "prepared MES"):
                resolve(str(Path(tmp) / "weights.ckpt"), models)
            save_file({"fixture": torch.ones(1)}, path)
            with self.assertRaisesRegex(ValueError, "expected prepared"):
                resolve(str(path), models)
            save_file({"fixture": torch.ones(1)}, path,
                      metadata={"anno_architecture": "maverick-mes-deberta-v3-large-v1"})
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                resolve("sapienzanlp/maverick-mes-ontonotes", models)
            models["ontonotes"]["prepared_sha256"] = maverick.sha256_file(path)
            self.assertEqual(resolve("sapienzanlp/maverick-mes-ontonotes", models), path)
            self.assertEqual(resolve(str(path), models), path)

    def test_encoder_isolation_and_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            allowed = root / "config.json"
            allowed.write_bytes(b"fixture config")
            (root / "model.safetensors").write_bytes(b"unreviewed optional file")
            staged = root / "staged"
            staged.mkdir()
            pins = {"ENCODER": "test/encoder", "ENCODER_REVISION": "fixture-revision",
                    "ENCODER_FILES": {allowed.name: hashlib.sha256(allowed.read_bytes()).hexdigest()}}
            with patch("huggingface_hub.hf_hub_download", return_value=allowed) as fetch:
                with patch.dict(os.environ, {"ANNO_NO_DOWNLOADS": "1"}):
                    maverick.verify_encoder(pins, directory=staged)
                fetch.assert_called_once_with("test/encoder", "config.json", revision="fixture-revision",
                                              cache_dir=None, local_files_only=True)
                self.assertEqual(list(staged.iterdir()), [staged / allowed.name])
                self.assertEqual((staged / allowed.name).read_bytes(), allowed.read_bytes())
                allowed.write_bytes(b"different bytes")
                with self.assertRaisesRegex(ValueError, "digest mismatch"):
                    maverick.verify_encoder(pins)

    def test_panel_checks_prepared_weights_against_shared_pins(self):
        panel = runpy.run_path(str(Path(__file__).resolve().parents[1] / "coref_gap.py"))
        with patch.dict(os.environ, {"MAVERICK_PYTHON": "fixture", "MAVERICK_MODEL_PATH": "prepared",
                                     "HF_HUB_CACHE": "fixture-hub", "TRANSFORMERS_CACHE": "fixture-hub"}):
            # Reuse the frozen environment declarations from the panel recipe.
            reqs = (Path(__file__).parent / "requirements-cpu-py312.txt").read_text().splitlines()
            pinned = dict(line.split("==", 1) for line in reqs if "==" in line)
            def versions(args, **kwargs):
                from types import SimpleNamespace
                import json
                return SimpleNamespace(stdout=json.dumps({name: pinned[name] for name in args[3:]}))
            from unittest.mock import Mock
            checks = Mock()
            with patch.dict(panel["model_identity"].__globals__, {"check_file": checks}), patch(
                "subprocess.run", side_effect=versions
            ):
                identity = panel["model_identity"]("maverick", "predicted")
            self.assertEqual(identity["source_sha256"], maverick.MODELS["ontonotes"]["source_sha256"])
            self.assertEqual(identity["sha256"], maverick.MODELS["ontonotes"]["prepared_sha256"])
            checks.assert_any_call("prepared", identity["sha256"])

    def test_restricted_loaders_refuse_object_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint, output = root / "weights.bin", root / "model.safetensors"
            torch.save({"fixture_metadata": Path("training-output")}, checkpoint)
            with self.assertRaises(SystemExit) as err:
                generic.convert_pytorch_to_safetensors(checkpoint, output)
            self.assertIn("restricted loader", str(err.exception))
            self.assertFalse(output.exists())
