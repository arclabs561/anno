"""Offline contract tests for the HuggingFace dataset cache downloader."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT_PATH = Path(__file__).with_name("download_hf_datasets.py")


def load_downloader():
    spec = importlib.util.spec_from_file_location("download_hf_datasets", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeHttpResponse:
    def __init__(self, content: bytes, status: int = 200, headers=None):
        self.content = content
        self.status = status
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _size=-1):
        return self.content


class DownloadHfDatasetsTests(unittest.TestCase):
    def test_verified_conll2003_fixture_writes_canonical_cache_and_manifest(self):
        downloader = load_downloader()
        requests = []

        def urlopen(request, **_kwargs):
            requests.append(request.full_url)
            return FakeHttpResponse(b'[{"tokens":["JAPAN","CHINA"],"tags":[5,1]}]')

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(downloader, "CACHE_DIR", Path(directory)),
            patch.object(downloader, "urlopen", urlopen),
        ):
            self.assertTrue(
                downloader.download_dataset(
                    "conll2003_sample", downloader.DATASETS["conll2003_sample"]
                )
            )

            self.assertEqual(
                requests,
                [
                    "https://huggingface.co/datasets/tner/conll2003/resolve/b18612dee0007b1f7129731dbf2f5f2ed4039ad3/dataset/test.json"
                ],
            )

            cache_path = Path(directory) / "CoNLL2003Sample.cache"
            self.assertEqual(
                cache_path.read_text(encoding="utf-8"),
                "JAPAN\tB-LOC\nCHINA\tB-ORG\n\n",
            )
            manifest = json.loads(
                (Path(directory) / "manifest.json").read_text(encoding="utf-8")
            )
            entry = manifest["entries"]["CoNLL2003Sample.cache"]
            self.assertEqual(
                entry["source_url"],
                "https://huggingface.co/datasets/tner/conll2003",
            )
            self.assertEqual(
                entry["resolved_url"],
                "https://huggingface.co/datasets/tner/conll2003/resolve/b18612dee0007b1f7129731dbf2f5f2ed4039ad3/dataset/test.json",
            )
            self.assertEqual(entry["sentence_count"], 1)
            self.assertEqual(entry["entity_count"], 2)

    def test_verified_jnlpba_fixture_writes_canonical_cache_and_manifest(self):
        downloader = load_downloader()
        requests = []

        def urlopen(request, **_kwargs):
            requests.append(request.full_url)
            return FakeHttpResponse(b'[{"tokens":["HUVECs","released"],"tags":[1,0]}]')

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(downloader, "CACHE_DIR", Path(directory)),
            patch.object(downloader, "urlopen", urlopen),
        ):
            self.assertTrue(
                downloader.download_dataset("jnlpba", downloader.DATASETS["jnlpba"])
            )

            self.assertEqual(
                requests,
                [downloader.DATASETS["jnlpba"]["resolved_url"]],
            )

            cache_path = Path(directory) / "JNLPBA.cache"
            self.assertEqual(
                cache_path.read_text(encoding="utf-8"), "HUVECs\tB-DNA\nreleased\tO\n\n"
            )
            manifest = json.loads(
                (Path(directory) / "manifest.json").read_text(encoding="utf-8")
            )
            entry = manifest["entries"]["JNLPBA.cache"]
            self.assertEqual(
                entry["source_url"],
                downloader.DATASETS["jnlpba"]["registry_source_url"],
            )
            self.assertEqual(
                entry["resolved_url"], downloader.DATASETS["jnlpba"]["resolved_url"]
            )
            self.assertEqual(entry["sentence_count"], 1)
            self.assertEqual(entry["entity_count"], 1)

    def test_unverified_sources_do_not_write_canonical_cache_or_manifest(self):
        downloader = load_downloader()
        rejected = ["bc2gm_full", "uner", "biomner", "craft", "msner"]
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(downloader, "CACHE_DIR", Path(directory)),
        ):
            for name in rejected:
                config = downloader.DATASETS[name]
                with self.subTest(name=name):
                    self.assertFalse(config["identity_verified"])
                    self.assertFalse(downloader.download_dataset(name, config))
                    self.assertFalse(
                        (Path(directory) / config["cache_filename"]).exists()
                    )
                    self.assertFalse((Path(directory) / "manifest.json").exists())

    def test_only_identity_verified_dataset_is_eligible_for_a_canonical_cache(self):
        downloader = load_downloader()

        self.assertEqual(
            {
                name: config["cache_filename"]
                for name, config in downloader.DATASETS.items()
                if config.get("identity_verified")
            },
            {
                "conll2003_sample": "CoNLL2003Sample.cache",
                "jnlpba": "JNLPBA.cache",
            },
        )

    def test_unknown_numeric_tags_reject_canonical_output_before_manifest_write(self):
        downloader = load_downloader()

        with self.assertRaisesRegex(ValueError, "outside verified label mapping"):
            downloader.normalise_tags(
                [99], downloader.DATASETS["jnlpba"]["label_names"]
            )
        with self.assertRaisesRegex(
            ValueError, "require an explicit verified label mapping"
        ):
            downloader.normalise_tags([1], None)

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(downloader, "CACHE_DIR", Path(directory)),
            patch.object(
                downloader,
                "urlopen",
                return_value=FakeHttpResponse(b'[{"tokens":["HUVECs"],"tags":[99]}]'),
            ),
        ):
            self.assertFalse(
                downloader.download_dataset("jnlpba", downloader.DATASETS["jnlpba"])
            )
            self.assertFalse((Path(directory) / "JNLPBA.cache").exists())
            self.assertFalse((Path(directory) / "manifest.json").exists())

    def test_token_tag_length_mismatch_rejects_canonical_output(self):
        downloader = load_downloader()

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(downloader, "CACHE_DIR", Path(directory)),
            patch.object(
                downloader,
                "urlopen",
                return_value=FakeHttpResponse(
                    b'[{"tokens":["HUVECs","released"],"tags":[1]}]'
                ),
            ),
        ):
            self.assertFalse(
                downloader.download_dataset("jnlpba", downloader.DATASETS["jnlpba"])
            )
            self.assertFalse((Path(directory) / "JNLPBA.cache").exists())
            self.assertFalse((Path(directory) / "manifest.json").exists())

    def test_proxy_rejection_happens_before_fetching_artifact(self):
        downloader = load_downloader()
        calls = []
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(downloader, "CACHE_DIR", Path(directory)),
            patch.object(
                downloader, "urlopen", side_effect=lambda *_args: calls.append("fetch")
            ),
        ):
            self.assertFalse(
                downloader.download_dataset("uner", downloader.DATASETS["uner"])
            )
            self.assertEqual(calls, [])
            self.assertFalse((Path(directory) / "UNER.cache").exists())

    def test_pinned_artifact_rejects_non_ok_and_oversized_responses(self):
        downloader = load_downloader()
        with (
            self.assertRaisesRegex(ValueError, "HTTP 404"),
            patch.object(
                downloader, "urlopen", return_value=FakeHttpResponse(b"", status=404)
            ),
        ):
            downloader.load_pinned_json_artifact("https://example.invalid/test.json")

        with (
            self.assertRaisesRegex(ValueError, "exceeds"),
            patch.object(
                downloader,
                "urlopen",
                return_value=FakeHttpResponse(
                    b"[]",
                    headers={"Content-Length": str(downloader.MAX_ARTIFACT_BYTES + 1)},
                ),
            ),
        ):
            downloader.load_pinned_json_artifact("https://example.invalid/test.json")

    def test_pinned_artifact_accepts_jsonl(self):
        downloader = load_downloader()
        with patch.object(
            downloader,
            "urlopen",
            return_value=FakeHttpResponse(
                b'{"tokens":["first"]}\n{"tokens":["second"]}\n'
            ),
        ):
            self.assertEqual(
                downloader.load_pinned_json_artifact(
                    "https://example.invalid/test.json"
                ),
                [{"tokens": ["first"]}, {"tokens": ["second"]}],
            )


if __name__ == "__main__":
    unittest.main()
