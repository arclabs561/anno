# SPDX-License-Identifier: MIT OR Apache-2.0
"""Exercise reviewed edition/page lookup using two original one-page PDFs."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from .open_page import open_page


DOI = "10.7554/eLife.99999"


def blank_pdf(note):
    data = bytearray(b"%PDF-1.4\n%" + note + b"\n")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Resources << >> >>",
    ]
    offsets = []
    for number, obj in enumerate(objects, 1):
        offsets.append(len(data))
        data.extend(f"{number} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = len(data)
    data.extend(b"xref\n0 4\n0000000000 65535 f \n")
    for offset in offsets:
        data.extend(f"{offset:010d} 00000 n \n".encode())
    data.extend(f"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    return bytes(data)


class OpeningPages(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.files = []
        self.index = {"pages": []}
        for version, note in enumerate((b"first", b"revision"), 1):
            path = self.directory / f"cached # {version}.pdf"
            data = blank_pdf(note)
            path.write_bytes(data)
            self.files.append(path)
            self.index["pages"].append({
                "doi": DOI, "edition": version, "page": 1,
                "url": f"https://cdn.elifesciences.org/articles/99999/elife-99999-v{version}.pdf",
                "sha256": sha256(data).hexdigest(),
            })

    def test_exact_page_and_escaped_local_path(self):
        path = self.files[1]
        record = self.index["pages"][1]
        result = open_page(self.index, DOI.upper(), 2, path)
        self.assertEqual(result, {
            "doi": DOI.lower(), "edition": 2, "page": 1, "pdf_sha256": record["sha256"],
            "publisher_page": record["url"] + "#page=1", "local_page": path.as_uri() + "#page=1",
        })

    def test_missing_changed_or_other_edition_never_falls_back(self):
        for doi, version, path in [
            (DOI, 1, self.files[1]), (DOI, 3, self.files[1]), (DOI, True, self.files[0]),
            ("10.7554/eLife.11111", 2, self.files[1]),
        ]:
            with self.subTest(doi=doi, version=version), self.assertRaises(ValueError):
                open_page(self.index, doi, version, path)
        self.files[0].write_bytes(blank_pdf(b"changed at same path"))
        with self.assertRaises(ValueError):
            open_page(self.index, DOI, 1, self.files[0])
        self.files[0].unlink()
        with self.assertRaises(FileNotFoundError):
            open_page(self.index, DOI, 1, self.files[0])

    def test_tampered_or_ambiguous_index(self):
        record, other = self.index["pages"]
        for records in [
            [{**record, "doi": "10.7554/eLife.12345"}],
            [{**other, "edition": 1}], [{**record, "sha256": other["sha256"]}],
            [{**record, "page": 2}], [{**record, "page": True}], [record, record],
        ]:
            with self.subTest(records=records), self.assertRaises(ValueError):
                open_page({"pages": records}, DOI, 1, self.files[0])

    def test_cli_returns_only_verified_locator(self):
        receipt = self.directory / "pages.json"
        receipt.write_text(json.dumps(self.index), encoding="utf-8")
        command = [sys.executable, "-m", open_page.__module__, DOI, "2"]
        for file, success in [(self.files[1], True), (self.files[0], False)]:
            result = subprocess.run([*command, str(file), "--index", str(receipt)],
                                    capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode == 0, success)
            if success:
                self.assertEqual(json.loads(result.stdout)["edition"], 2)
            else:
                self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
