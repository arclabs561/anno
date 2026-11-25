# SPDX-License-Identifier: MIT OR Apache-2.0
"""Locate an independently reviewed opening PDF page for one eLife edition.

Run as a module with DOI, edition number and the local PDF path. The returned
local URI identifies the bytes verified at lookup time. Revalidate after a
file replacement. The curator supplies the physical page number and reviewed
source digest; the caller supplies the publisher edition selected for reading.
"""
import argparse
import json
from pathlib import Path
import re

from .reader import doi_key
from .sources.verify_print import file_hash


DEFAULT_INDEX = Path(__file__).with_name("sources") / "opening_pages.json"


def open_page(index, doi, edition, pdf):
    """Return local and publisher page locators after verifying the exact cached PDF."""
    doi = doi_key(doi)
    if type(edition) is not int or edition < 1:
        raise ValueError("expected a positive publisher edition")
    records = [
        row for row in index["pages"]
        if doi_key(row["doi"]) == doi and type(row["edition"]) is int and row["edition"] == edition
    ]
    if len(records) != 1:
        raise ValueError("expected one reviewed source page for the selected DOI and edition")
    record = records[0]
    article = re.fullmatch(r"10\.7554/elife\.([0-9]+)", doi)
    if not article:
        raise ValueError("expected a reviewed eLife DOI")
    number = article.group(1)
    expected_url = f"https://cdn.elifesciences.org/articles/{number}/elife-{number}-v{edition}.pdf"
    if record["url"] != expected_url:
        raise ValueError("reviewed PDF URL differs from the selected work or edition")
    if type(record["page"]) is not int or record["page"] != 1:
        raise ValueError("expected a reviewed physical opening page")
    digest = record["sha256"]
    if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise ValueError("invalid reviewed PDF digest")
    path = Path(pdf).resolve(strict=True)
    if file_hash(path) != digest:
        raise ValueError("cached PDF bytes differ from the selected edition's receipt")
    fragment = f"#page={record['page']}"
    return {
        "doi": doi, "edition": edition, "page": record["page"], "pdf_sha256": digest,
        "publisher_page": record["url"] + fragment, "local_page": path.as_uri() + fragment,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("doi")
    parser.add_argument("edition", type=int)
    parser.add_argument("pdf", type=Path, help="local cached PDF to verify")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX, help="trusted public source-page index")
    args = parser.parse_args()
    try:
        index = json.loads(args.index.read_bytes())
        result = open_page(index, args.doi, args.edition, args.pdf)
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"Source page unavailable: {error}\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
