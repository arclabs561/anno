import hashlib
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from run import verify_assets


class CheckpointCache(unittest.TestCase):
    def test_cold_retry_warm_offline_and_corruption(self):
        payload = b"checkpoint"
        filename = "fp32_v2/encoder.onnx"
        spec = {"repo": "publisher/model", "revision": "a" * 40,
                "files": {filename: {"remote": filename, "sha256": hashlib.sha256(payload).hexdigest()}}}
        url = f"https://huggingface.co/publisher/model/resolve/{spec['revision']}/{filename}"
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("run.urlopen", side_effect=[HTTPError(url, 503, "Unavailable", {}, None),
                                                   io.BytesIO(payload)]) as request, patch("run.time.sleep"):
                verify_assets(root, spec, fetch=True)
                self.assertEqual(request.call_count, 2)
                request.assert_called_with(url, timeout=180)
            with patch("run.urlopen", side_effect=AssertionError("unexpected network access")):
                verify_assets(root, spec, fetch=False)
                (root / filename).write_bytes(b"damaged")
                with self.assertRaisesRegex(ValueError, "Missing or incorrect"):
                    verify_assets(root, spec, fetch=False)
            with patch("run.urlopen", return_value=io.BytesIO(b"bad response")):
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    verify_assets(root, spec, fetch=True)
                self.assertEqual((root / filename).read_bytes(), b"damaged")
