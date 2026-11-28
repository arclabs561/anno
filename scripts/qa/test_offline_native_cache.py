import csv
import io
import json
import sys
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
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

    def test_agcner_preserves_explicit_whitespace_and_supplementary_characters(self):
        raw = "𠀀 B-PET\r\n  I-PET\r\n\t O\r\n\u2028 O".encode()
        row = json.loads(native.native_bytes("agcner-jsonl", raw))
        self.assertEqual(row["tokens"], ["𠀀", " ", "\t", "\u2028"])
        self.assertEqual(row["ner_tags"], ["B-PET", "I-PET", "O", "O"])

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
                ("ghost file", "cache files differ"),
                ("ghost entry", "manifest inventory differs"),
                ("wrong source", "Restored manifest differs"),
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
                    elif case == "ghost file":
                        changed["cache/datasets/Other.cache"] = raw
                    elif case in {"ghost entry", "wrong source"}:
                        path = "cache/datasets/manifest.json"
                        manifest = json.loads(changed[path])
                        entry = manifest["entries"][data["id"] + ".cache"]
                        if case == "ghost entry":
                            manifest["entries"]["Other.cache"] = dict(
                                entry, dataset_id="Other.cache"
                            )
                        else:
                            entry["source_url"] = "https://example.org/another-corpus"
                        changed[path] = native.json_bytes(manifest)
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

    def test_cli_fetches_only_selected_variant_and_restores_it_as_default(self):
        counts = {"sentences": 1, "tokens": 1, "native_labels": 1}
        payloads = {
            "test": b"Ann B-PER\n",
            "train": b"Ivy B-PER\n",
            "NOTICE": b"Publisher terms",
        }

        def source(name):
            return {
                "url": f"https://example.org/{name}",
                "sha256": native.digest(payloads[name]),
            }

        data = {
            "id": "SECFilingsNER",
            "usage": "general",
            "registry_url": "https://example.org/test",
            "attribution": "test publisher",
            "license": "CC BY 3.0",
            "interpretation": "BIO",
            "inputs": [
                {
                    "variant": split,
                    "default": split == "test",
                    "counts": counts,
                    **source(split),
                }
                for split in ("test", "train")
            ],
            "notices": [{"name": "NOTICE", **source("NOTICE")}],
        }
        with tempfile.TemporaryDirectory() as scratch:
            directory = Path(scratch)
            recipe = directory / "index.json"
            recipe.write_text(
                json.dumps(
                    {
                        "snapshot_date": "2026-11-28",
                        "datasets": [data, {**data, "id": "Other"}],
                    }
                )
            )
            group_archive = directory / "anno-native-cache-general.zip"
            group_archive.write_bytes(b"existing complete package")
            args = [
                "script",
                str(directory),
                "--index",
                str(recipe),
                "--anno",
                sys.executable,
                "--dataset",
                "SECFilingsNER",
                "--variant",
                "train",
            ]
            artifact = directory / "anno-native-cache-dataset-SECFilingsNER-train.zip"
            requested = []

            def fetch(request, timeout):
                requested.append(request.full_url)
                return io.BytesIO(payloads[request.full_url.rsplit("/", 1)[1]])

            with (
                patch.object(native, "stats", return_value=counts),
                patch.object(
                    native.subprocess, "check_output", return_value="anno 1.0"
                ),
                redirect_stdout(io.StringIO()),
            ):
                with (
                    patch.object(sys, "argv", [*args, "--fetch"]),
                    patch.object(native.urllib.request, "urlopen", side_effect=fetch),
                ):
                    native.main()
                self.assertEqual(
                    requested, [source("NOTICE")["url"], source("train")["url"]]
                )
                self.assertEqual(
                    {p.name for p in (directory / "source-cache").iterdir()},
                    {source("train")["sha256"], source("NOTICE")["sha256"]},
                )
                first = artifact.read_bytes()
                with zipfile.ZipFile(artifact) as z:
                    self.assertEqual(
                        z.read("cache/datasets/SECFilingsNER.cache"), payloads["train"]
                    )
                    catalog = json.loads(z.read("sources.json"))
                    self.assertEqual(
                        catalog["selection"],
                        {
                            "dataset": "SECFilingsNER",
                            "indexed_variants": 2,
                            "variant": "train",
                        },
                    )
                    (item,) = catalog["inputs"]
                    self.assertTrue(item["default"])
                    self.assertFalse(item["index_default"])
                    self.assertEqual(item["counts"], counts)
                    self.assertIn(
                        "1 of 2 indexed variants (train)", z.read("README.md").decode()
                    )
                with (
                    patch.object(
                        native.urllib.request,
                        "urlopen",
                        side_effect=AssertionError("unexpected HTTP"),
                    ),
                    patch.object(sys, "argv", args),
                ):
                    native.main()
                self.assertEqual(artifact.read_bytes(), first)
                recipe.unlink()
                with patch.object(
                    sys,
                    "argv",
                    [
                        "script",
                        str(artifact),
                        "--anno",
                        sys.executable,
                        "--index",
                        str(recipe),
                        "--verify-only",
                    ],
                ):
                    native.main()
                self.assertEqual(
                    group_archive.read_bytes(), b"existing complete package"
                )

    def test_selection_validates_scope_and_names_whole_dataset_differently(self):
        index = json.loads(native.INDEX.read_text())
        selected = native.select_index(index, "SECFilingsNER", None)
        self.assertEqual(
            [i["variant"] for i in selected["datasets"][0]["inputs"]], ["test", "train"]
        )
        self.assertEqual(
            native.archive_stem(selected, "general"),
            "anno-native-cache-dataset-SECFilingsNER",
        )
        self.assertEqual(
            native.archive_stem(index, "general"), "anno-native-cache-general"
        )
        for dataset, variant, error in (
            (None, "test", "requires --dataset"),
            ("Unknown", None, "Unknown --dataset"),
            ("SECFilingsNER", "invalid", "Available: test, train"),
        ):
            with (
                self.subTest(dataset=dataset, variant=variant),
                self.assertRaisesRegex(ValueError, error),
            ):
                native.select_index(index, dataset, variant)

    def test_inventory_counts_distinct_profiles_across_overlapping_zips(self):
        with tempfile.TemporaryDirectory() as scratch:
            directory = Path(scratch)
            for name, variants in (
                ("general", ["test", "train"]),
                ("dataset-SECFilingsNER-test", ["test"]),
            ):
                inputs = [
                    {"id": "SECFilingsNER", "variant": variant} for variant in variants
                ]
                with zipfile.ZipFile(
                    directory / f"anno-native-cache-{name}.zip", "w"
                ) as archive:
                    archive.writestr("sources.json", json.dumps({"inputs": inputs}))
            registry = directory / "registry.csv"
            registry.write_text(
                "id,access,url\nSECFilingsNER,Public,https://example.org\n"
            )
            native.write_inventory({}, registry, directory)
            with (directory / "native-scope.csv").open() as handle:
                (row,) = csv.DictReader(handle)
            self.assertEqual(row["included_profiles"], "2")
            self.assertEqual(
                set(row["archive"].split(" | ")),
                {
                    "anno-native-cache-general.zip",
                    "anno-native-cache-dataset-SECFilingsNER-test.zip",
                },
            )

    def test_verifier_requires_registry_ids_and_canonical_cache_paths(self):
        for ident, path, message in (
            ("invalid id", "cache/datasets/invalid id.cache", "Invalid dataset ID"),
            ("SECFilingsNER", "other/file", "path disagrees"),
        ):
            item = {
                "id": ident,
                "variant": "test",
                "default": True,
                "sha256": "0" * 64,
                "path": path,
            }
            catalog = {
                "schema": "anno.offline-native-cache/v1",
                "dataset_ids": 1,
                "file_count": 1,
                "distinct_payloads": 1,
                "inputs": [item],
            }
            with tempfile.TemporaryDirectory() as scratch, self.subTest(ident=ident):
                artifact = Path(scratch) / "offered.zip"
                with zipfile.ZipFile(artifact, "w") as archive:
                    archive.writestr("sources.json", json.dumps(catalog))
                with self.assertRaisesRegex(ValueError, message):
                    native.verify(artifact, sys.executable)


if __name__ == "__main__":
    unittest.main()
