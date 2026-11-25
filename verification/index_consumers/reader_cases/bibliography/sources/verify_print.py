#!/usr/bin/env python3
"""Replay the 45 manually located eLife opening-PDF bylines without printing names."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import NamedTemporaryFile
import unicodedata
from urllib.request import urlopen

DEFAULT_REFERENCE = Path(__file__).resolve().parents[1] / "scoring/receipts/reference.json"


def file_hash(path):
    with path.open("rb") as stream:
        digest = sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cached_pdf(directory, source, fetch):
    url = source["url"]
    match = re.fullmatch(
        r"https://cdn\.elifesciences\.org/articles/([0-9]+)/elife-\1-v([1-9][0-9]*)\.pdf",
        url,
    )
    if not match:
        raise ValueError(f"unexpected PDF location {url}")
    path = directory / url.rsplit("/", 1)[1]
    if fetch and not path.exists():
        with NamedTemporaryFile(dir=directory, delete=False) as output:
            temporary = Path(output.name)
        try:
            with urlopen(url, timeout=120) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
            if file_hash(temporary) != source["sha256"]:
                raise ValueError(f"retrieved PDF differs from frozen bytes: {url}")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    if file_hash(path) != source["sha256"]:
        raise ValueError(f"cached PDF differs from frozen bytes: {path}")
    return path, match.groups()


def printed_fingerprint(path, source, work, version, stage):
    # Each range was selected from this exact PDF before the structured comparison.
    if source["page"] != 1:
        raise ValueError(f"unexpected opening page for {path.name}")
    decoded = subprocess.check_output(
        ["pdftotext", "-f", "1", "-l", "1", "-layout", "-enc", "UTF-8", str(path), "-"],
        stderr=subprocess.PIPE,
    ).decode("utf-8")
    start, end = source["layout_lines"]
    lines = decoded.splitlines()[start - 1:end]
    if stage == "poa":
        # The nine accepted manuscripts include printed manuscript line numbers.
        lines = [re.sub(r"^\s*\d+\s*", "", line) for line in lines]
    text = " ".join(" ".join(lines).split()).replace("-\u00ad", "-")
    if (work, version) == ("81453", "1"):
        # The final hyphen on the first line joins a family name across the line break.
        text = text.replace("- ", "-")
    if (work, version) == ("80254", "1"):
        # The superscript before the final author has no following separator comma.
        text = text.replace("# ", "#, ")
    text = re.sub(r"\d+(?:,\d+)*|[*#‡†§¶]", "", text)
    if (work, version) in {("79863", "1"), ("78649", "1")}:
        text = text.replace(" and ", ", ")
    names = [
        unicodedata.normalize("NFC", " ".join(name.split()))
        for name in text.split(",") if name.strip()
    ]
    if not names or any(not any(char.isalpha() for char in name) for name in names):
        raise ValueError(f"invalid PDF name span for {path.name}")
    literal = json.dumps(names, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return {"count": len(names), "sha256": sha256(literal).hexdigest()}


def verify(rows, directory, fetch):
    directory.mkdir(parents=True, exist_ok=True)
    observed = {}
    for row in rows:
        source, = (s for s in row["sources"] if s["role"] == "publisher-opening-pdf")
        footprint = (source["url"], source["sha256"], source["page"],
                     tuple(source["layout_lines"]), row["edition_status"])
        if footprint not in observed:
            path, (work, version) = cached_pdf(directory, source, fetch)
            observed[footprint] = printed_fingerprint(
                path, source, work, version, row["edition_status"]
            )
        if observed[footprint] != row["printed_byline"]:
            raise ValueError(f"printed byline differs at {row['notice']} {row['cut']}")
    print(f"Verified {len(rows)} printed-byline receipts from {len(observed)} distinct opening PDFs.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True, help="external PDF directory")
    parser.add_argument("--fetch", action="store_true", help="download missing versioned PDFs")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    args = parser.parse_args()
    verify(json.loads(args.reference.read_text())["rows"], args.cache, args.fetch)


if __name__ == "__main__":
    main()
