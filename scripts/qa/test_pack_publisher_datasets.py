import csv
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pack_publisher_datasets as pack


class PublisherPackTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.args = SimpleNamespace(
            manifest=root / "source.csv",
            terms=root / "terms.json",
            exclusions=root / "local.tsv",
            cache_dir=root / "cache",
            notice_cache=root / "notices",
            output=root / "output",
            fetch_missing=False,
        )
        self.args.cache_dir.mkdir()
        self.args.exclusions.write_text("")
        repo, rev = "example/labels", "c" * 40
        template = dict(
            id="Survey",
            repository=repo,
            revision=rev,
            path="",
            split="test",
            subset="",
            url="",
            sha256="",
            metric="BIO",
            rows="",
            tokens="",
            tagged="",
            status="",
            reason="",
        )
        self.rows = []
        for name, raw, status in [
            ("ok", b"One\tB-ORG\n", "ok"),
            ("broken", b"bad record\n", "error"),
            ("restricted", b"Two\tO\n", "ok"),
        ]:
            row = dict(
                template,
                path=name + ".tsv",
                url=pack.pinned_url(repo, rev, name + ".tsv"),
                sha256=pack.sha(raw),
                status=status,
            )
            if status == "ok":
                row.update(rows="1", tokens="1", tagged=str(int(name == "ok")))
            else:
                row["reason"] = "Unsupported or missing BIO label at line 1"
            self.rows.append(row)
            (self.args.cache_dir / pack.sha(row["url"].encode())).write_bytes(raw)
        self.rows.append(
            dict(template, id="Alias", status="alias", reason="Included in Survey.")
        )
        self.args.manifest.write_bytes(pack.csv_bytes(self.rows, list(template)))
        notice = b"Example notice\n"
        path = self.args.notice_cache / repo / rev / "files" / "LICENSE"
        path.parent.mkdir(parents=True)
        path.write_bytes(notice)
        policy = {
            "ledger_sha256": pack.sha(self.args.manifest.read_bytes()),
            "repositories": {
                repo: {
                    "revision": rev,
                    "rules": [
                        {
                            "paths": ["restricted.tsv"],
                            "bundle": "",
                            "terms": "Locator only.",
                        },
                        {
                            "bundle": "general",
                            "terms": "Example grant.",
                            "notices": ["LICENSE"],
                        },
                    ],
                    "notice_sha256": {"LICENSE": pack.sha(notice)},
                    "review_notice": "LICENSE",
                }
            },
        }
        self.args.terms.write_text(json.dumps(policy))

    def test_preserves_bad_bytes_and_notice_and_never_downloads_exclusions(self):
        with patch.object(
            pack, "urlopen", side_effect=AssertionError("unexpected network")
        ), redirect_stdout(io.StringIO()):
            pack.build(self.args)
            archive = self.args.output / "publisher-general.zip"
            digest = pack.sha(archive.read_bytes())
            pack.verify(archive, True)
            with ZipFile(archive) as z:
                entries = pack.read_rows(z.read("manifest.csv"))
                self.assertEqual([r["status"] for r in entries], ["ok", "error"])
                self.assertEqual(z.read(entries[1]["member"]), b"bad record\n")
                self.assertFalse(
                    any(n.endswith("restricted.tsv") for n in z.namelist())
                )
            missing = self.args.cache_dir / pack.sha(self.rows[2]["url"].encode())
            missing.unlink()
            self.args.fetch_missing = True
            pack.build(self.args)
            self.assertEqual(pack.sha(archive.read_bytes()), digest)
            coverage = list(
                csv.DictReader(
                    io.StringIO(
                        (self.args.output / "publisher-coverage.csv").read_text()
                    )
                )
            )
            self.assertEqual(coverage[2]["receipt_check"], "not fetched (excluded)")
            self.assertEqual(coverage[3]["receipt_check"], "no file")

    def test_corrupt_input_and_changed_ledger_stop_release(self):
        cached = self.args.cache_dir / pack.sha(self.rows[0]["url"].encode())
        cached.write_bytes(b"different")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            pack.build(self.args)
        self.assertFalse(list(self.args.output.glob("*.zip")))
        self.args.manifest.write_bytes(self.args.manifest.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "ledger changed"):
            pack.build(self.args)

    def test_annotation_only_rule_rejects_unmasked_words(self):
        policy = json.loads(self.args.terms.read_bytes())
        policy["repositories"]["example/labels"]["rules"][-1]["require_mask"] = "[TOK]"
        self.args.terms.write_text(json.dumps(policy))
        with self.assertRaisesRegex(ValueError, "Unmasked source content"):
            pack.build(self.args)
        self.assertFalse(list(self.args.output.glob("*.zip")))


if __name__ == "__main__":
    unittest.main()
