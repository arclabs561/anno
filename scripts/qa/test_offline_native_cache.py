import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_offline_native_cache as native


class NativeCacheTest(unittest.TestCase):
    def test_agcner_keeps_original_bio_tags_and_character_order(self):
        raw = "稻 B-PET\r\n虫 I-PET\r\n\r\n新 O\r\n".encode()
        rows = [
            json.loads(line)
            for line in native.native_bytes("agcner-jsonl", raw).splitlines()
        ]
        self.assertEqual(
            rows[0], {"tokens": ["稻", "虫"], "ner_tags": ["B-PET", "I-PET"]}
        )
        self.assertEqual(rows[1], {"tokens": ["新"], "ner_tags": ["O"]})

    def test_histnero_preserves_empty_labelled_tokens_and_metadata(self):
        original = [
            {
                "id": "9418",
                "doc_id": "example",
                "region": "test",
                "tokens": ["", "loc", ""],
                "ner_tags": [7, 5, 6],
            }
        ]
        payload = json.loads(
            native.native_bytes("histnero-rows", json.dumps(original).encode())
        )
        self.assertEqual(payload["rows"], [{"row_idx": 0, "row": original[0]}])
        self.assertEqual(payload["num_rows_total"], 1)
        labels = payload["features"][0]["type"]["feature"]["names"]
        self.assertEqual(
            [labels[x] for x in original[0]["ner_tags"]], ["B-PROD", "B-LOC", "I-LOC"]
        )

    def test_pos_keeps_sentence_boundaries_and_every_word(self):
        payload = json.loads(
            native.native_bytes("masakhapos-rows", b"word NOUN\nword VERB\n\nnew ADJ\n")
        )
        self.assertEqual(payload["num_rows_total"], 2)
        self.assertEqual(
            payload["rows"][0]["row"],
            {"tokens": ["word", "word"], "tags": ["B-NOUN", "B-VERB"]},
        )
        self.assertEqual(payload["rows"][1]["row_idx"], 1)

    def test_norne_reads_name_attribute_only_for_word_rows(self):
        source = b"# source reference\n1-2\txy\t_\t_\t_\t_\t_\t_\t_\t_\n1\tx\t_\tX\t_\t_\t_\t_\t_\tSpaceAfter=No|name=B-PER\n2\ty\t_\tX\t_\t_\t_\t_\t_\tname=I-PER\n"
        self.assertEqual(
            native.native_bytes("norne-bio", source), b"x\tB-PER\ny\tI-PER\n"
        )
        with self.assertRaisesRegex(ValueError, "cannot be projected"):
            native.native_bytes("norne-bio", source.replace(b"name=I-PER", b"_"))

    def test_corrupt_and_missing_sources_fail_without_fetch(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            item = {
                "sha256": native.digest(b"real bytes"),
                "url": "https://example.org/release",
            }
            with self.assertRaises(FileNotFoundError):
                native.source_bytes(item, [cache], False)
            (cache / item["sha256"]).write_bytes(b"different bytes")
            with self.assertRaisesRegex(ValueError, "digest"):
                native.source_bytes(item, [cache], False)

    def test_verify_uses_pinned_profiles_even_when_zip_is_internally_consistent(self):
        data = json.loads(native.INDEX.read_text())["datasets"][0]
        counts = {"sentences": 1, "tokens": 1, "native_labels": 1}
        raw = b"ID\tURL\ncase-1\thttps://en.wikipedia.org/wiki/Example\n"
        item = {
            "variant": "test",
            "article_url_column": "URL",
            "default": True,
            "counts": counts,
            "url": "https://example.org/test",
            "sha256": native.digest(raw),
        }
        data.update(usage="general", notices=[], inputs=[item])
        index = {"snapshot_date": "2026-11-28", "datasets": [data]}
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(native, "stats", return_value=counts),
        ):
            root = Path(tmp)
            (root / item["sha256"]).write_bytes(raw)
            archive = native.build(
                index, root, sys.executable, [root], False, "general"
            )
            native.verify(archive, sys.executable, index, "general")
            with zipfile.ZipFile(archive) as z:
                members = {name: z.read(name) for name in z.namelist()}
            for case, message in [
                ("empty", "pinned general collection"),
                ("omitted", "Missing pinned profiles"),
                ("duplicate", "duplicate profile"),
                ("manifest", "Restored manifest differs"),
                ("replacement", "differs from the index"),
                ("missing articles", "article attribution"),
                ("wrong articles", "article attribution"),
            ]:
                with self.subTest(case=case):
                    catalog = json.loads(members["sources.json"])
                    changed = dict(members)
                    if case in {"empty", "omitted"}:
                        catalog["inputs"] = []
                        if case == "empty":
                            catalog["file_count"] = 0
                    elif case == "duplicate":
                        catalog["inputs"] *= 2
                    elif case in {"missing articles", "wrong articles"}:
                        path = native.article_notice_path(data["id"], item["variant"])
                        if case == "missing articles":
                            del changed[path]
                        else:
                            changed[path] = changed[path].replace(
                                b"wiki/Example", b"wiki/Other"
                            )
                    elif case == "manifest":
                        path = "cache/datasets/manifest.json"
                        manifest = json.loads(changed[path])
                        manifest["entries"][data["id"] + ".cache"][
                            "sentence_count"
                        ] += 1
                        changed[path] = native.json_bytes(manifest)
                    else:
                        record = catalog["inputs"][0]
                        replacement = b"two\n"
                        record["sha256"] = record["cache_sha256"] = native.digest(
                            replacement
                        )
                        changed[record["path"]] = replacement
                        manifest_path = "cache/datasets/manifest.json"
                        manifest = json.loads(changed[manifest_path])
                        manifest["entries"][data["id"] + ".cache"]["sha256"] = record[
                            "sha256"
                        ]
                        changed[manifest_path] = native.json_bytes(manifest)
                    changed["sources.json"] = native.json_bytes(catalog)
                    with zipfile.ZipFile(archive, "w") as z:
                        for name, payload in changed.items():
                            z.writestr(name, payload)
                    with self.assertRaisesRegex(ValueError, message):
                        native.verify(archive, sys.executable, index, "general")

    def test_rejects_duplicate_defaults_and_unsafe_zip_names(self):
        data = json.loads(native.INDEX.read_text())["datasets"][0]
        with self.assertRaisesRegex(ValueError, "one default"):
            native.validate_index(
                {"datasets": [{**data, "inputs": [data["inputs"][0]] * 2}]}
            )
        with (
            zipfile.ZipFile(io.BytesIO(), "w") as archive,
            self.assertRaisesRegex(ValueError, "Unsafe archive path"),
        ):
            native.member(archive, "notices/../outside", b"x")


if __name__ == "__main__":
    unittest.main()
