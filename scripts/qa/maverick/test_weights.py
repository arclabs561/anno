"""Offline smoke tests for the Maverick and generic weight converters."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load_file

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
            repo, revision, _ = maverick.MODELS["ontonotes"]
            info = (repo, revision, hashlib.sha256(weights.read_bytes()).hexdigest())

            def download(*, repo_id, revision, filename, cache_dir):
                self.assertEqual((repo_id, revision), info[:2])
                return {"weights.ckpt": weights, "config.yaml": config}[filename]

            with patch.dict(maverick.MODELS, {"ontonotes": info}), patch.object(
                maverick, "hf_hub_download", side_effect=download
            ):
                output = maverick.download_model("ontonotes", root)
                self.assertTrue(torch.equal(load_file(output)["encoder.layer"], tensor))
                self.assertEqual(load_file(output)["ids"].tolist(), [7])
                weights.write_bytes(b"incorrect file")
                with self.assertRaisesRegex(ValueError, "digest mismatch"):
                    maverick.download_model("ontonotes", root)
                self.assertTrue(torch.equal(load_file(output)["encoder.layer"], tensor))

    def test_restricted_loaders_refuse_object_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint, output = root / "weights.bin", root / "model.safetensors"
            torch.save({"fixture_metadata": Path("training-output")}, checkpoint)
            with self.assertRaises(SystemExit) as err:
                generic.convert_pytorch_to_safetensors(checkpoint, output)
            self.assertEqual(err.exception.code, 1)
            self.assertFalse(output.exists())
