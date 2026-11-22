"""Check exact pattern DATE spans against the map links in NASA's lunar eclipse catalogs."""

import argparse
from collections import Counter
from datetime import date
import hashlib
from html import unescape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import subprocess
import sys
from threading import Thread
from urllib.request import urlopen

CATALOGS = (
    ("LE1901-2000.html", 229, "972e37819ec3f8e8b3f48fd38a06984636f6733812a1fa2557e25600d2204532"),
    ("LE2001-2100.html", 228, "0f640913669a640418f05c173cea0bdeca194fcac6b5b196b6b9ab1b08122792"),
)
ROOT = "https://eclipse.gsfc.nasa.gov/LEcat5/"
MAP_LINK = re.compile(r'^<a href="[^"]*/LE(\d{4}-\d{2}-\d{2})[TPN]\.gif"')
MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()


def catalog_input(html):
    rows, spans = [], []
    offset = 0
    for line in html.splitlines():
        link = MAP_LINK.match(line)
        if not link:
            continue
        event = date.fromisoformat(link[1])
        row = unescape(re.sub(r"<[^>]+>", "", line))
        cell = re.match(r"^\d{5}  (.{11})  ", row)
        if not cell or cell[1] != f"{event.year} {MONTHS[event.month - 1]} {event.day:02d}":
            raise ValueError(f"calendar column disagrees with map {link[1]}")
        spans.append((offset + cell.start(1), offset + cell.end(1), cell[1]))
        rows.append(row)
        offset += len(row) + 1
    return "\n".join(rows), spans


def check_spans(output, expected):
    actual = Counter((e["start"], e["end"], e["text"]) for e in output["entities"] if e["type"] == "DATE")
    gold = Counter(expected)
    if actual != gold:
        raise ValueError(f"{sum((actual & gold).values())}/{len(expected)} exact dates; "
                         f"first missing: {next(iter(gold - actual), None)}, "
                         f"first unexpected: {next(iter(actual - gold), None)}")


def check_rendered_catalog(doc, table, gold):
    """Locate calendar cells by their row IDs in the text returned by the real URL CLI."""
    text = doc["text"]
    dates = Counter()
    for signal in doc["signals"]:
        if signal["label"] == "DATE":
            span = signal["location"]["Text"]
            start, end = span["start"], span["end"]
            if not 0 <= start < end <= len(text) or text[start:end] != signal["surface"]:
                raise ValueError(f"DATE surface disagrees with rendered text at {start}..{end}")
            dates[start, end, signal["surface"]] += 1
    expected = []
    for row, (_, _, calendar) in zip(table.splitlines(), gold, strict=True):
        # html2text displays linked row IDs as [09651][19]. Plain text also occurs.
        row_id = re.escape(row[:5])
        identifier = rf"(?:{row_id}|\[{row_id}\](?:\[\d+\])?)"
        pattern = rf"(?m)^[ \t]*{identifier}[ \t]+({re.escape(calendar)})(?=[ \t]|$)"
        matches = list(re.finditer(pattern, text))
        if len(matches) != 1:
            raise ValueError(f"expected one rendered calendar cell for row {row[:5]}, found {len(matches)}")
        expected.append((*matches[0].span(1), calendar))
    # Headers and other columns can legitimately contain additional dates.
    # Score only predicted DATE spans that intersect an annotated calendar cell.
    scoped = Counter({span: count for span, count in dates.items()
                      if any(span[0] < end and start < span[1] for start, end, _ in expected)})
    if scoped != Counter(expected):
        hits = sum((scoped & Counter(expected)).values())
        raise ValueError(f"{hits}/{len(expected)} exact rendered calendar dates; "
                         f"first missing: {next(iter(Counter(expected) - scoped), None)}")


def extract_page(anno_bin, raw):
    class CatalogPage(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), CatalogPage) as server:
        worker = Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            output = subprocess.check_output([
                anno_bin, "extract", "--url", f"http://127.0.0.1:{server.server_port}/catalog.html",
                "--model", "pattern", "--format", "grounded",
            ], timeout=120)
            return json.loads(output)
        finally:
            server.shutdown()
            worker.join()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anno-bin", default="./target/debug/anno")
    parser.add_argument("--source-dir", type=Path, help="read the two named HTML files locally, without network access")
    parser.add_argument("--web", action="store_true", help="also test HTML-to-text and model extraction through the URL CLI")
    args = parser.parse_args()
    for filename, count, digest in CATALOGS:
        if args.source_dir:
            raw = (args.source_dir / filename).read_bytes()
        else:
            with urlopen(ROOT + filename, timeout=30) as response:
                raw = response.read()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"{filename}: source SHA-256 changed")
        text, expected = catalog_input(raw.decode("utf-8"))
        if len(expected) != count:
            raise ValueError(f"{filename}: expected {count} catalog rows, read {len(expected)}")
        output = subprocess.check_output([args.anno_bin, "extract", "--model", "pattern", "--format", "json"],
                                         input=text, text=True, encoding="utf-8")
        try:
            check_spans(json.loads(output), expected)
        except ValueError as error:
            raise ValueError(f"{filename}: {error}") from error
        print(f"{filename}: {count}/{count} exact dates in isolated rows")
        if args.web:
            check_rendered_catalog(extract_page(args.anno_bin, raw), text, expected)
            print(f"{filename}: {count}/{count} exact dates at rendered page coordinates")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
