# Public-source pattern regressions

Each text file contains a brief UTF-8 excerpt with a final newline. The CLI test in
`tests/public_sources.rs` checks all structured entities expected from the
pattern backend on these excerpts. Offsets are half-open Unicode scalar indices.
People, places and other learned labels are outside this test's scope.

- **boe.txt** — Spanish Constitution, repeal provision §2, from the sentence ending in “Vizcaya” through its following sentence. The line break represents the paragraph boundary. The date follows accented letters on both lines. [Official text, BOE-A-1978-31229](https://www.boe.es/buscar/act.php?id=BOE-A-1978-31229&p=19781229&tn=1), [BOE reuse terms](https://www.boe.es/informacion/aviso_legal/).
- **aozora.txt** — Natsume Sōseki, *吾輩は猫である* (*I Am a Cat*), opening line. This public-domain literary excerpt includes the full-width indentation and Aozora's inline pronunciation markup from the original CP932 transcription. It has no structured entities. [Aozora edition 789](https://www.aozora.gr.jp/cards/000148/card789.html), [pinned transcription](https://github.com/aozorahack/aozorabunko_text/blob/0984f7dc18a2270bdfd2ff99bf14fd1b3ebb4f65/cards/000148/files/789_ruby_5639/789_ruby_5639.txt).
- **aozora-footer.html.cp932** — The same pinned edition’s final colophon sentence in its original CP932 encoding, inside a minimal HTML paragraph for offline HTTP replay. The source encloses the library URL in fullwidth parentheses immediately followed by Japanese prose.
- **tzdb.txt** and **tzdb-midnight.txt** — IANA Time Zone Database release 2025b, release header and Iran correction respectively. In the latter, +0330 is a UTC offset and 24:00 is local end-of-day. Both excerpts are from the public-domain release notes. [NEWS](https://github.com/eggert/tz/blob/2025b/NEWS), [license](https://github.com/eggert/tz/blob/2025b/LICENSE).
- **nasa-eclipse.txt** — Catalog row 09651, the lunar eclipse of January 9, 2001. The first column is its catalog number. Its map links to `LE2001-01-09T.gif`, independently specifying the calendar date. [Fred Espenak, NASA/GSFC, Catalog of Lunar Eclipses](https://eclipse.gsfc.nasa.gov/LEcat5/LE2001-2100.html).

With an Anno debug binary built, run `python3 scripts/qa/check_nasa_dates.py` to
score all 457 NASA calendar cells. It verifies each source file's SHA-256 and
uses each eclipse map's ISO date to check the visible date and character offsets.
For offline replay, pass `--source-dir DIR` containing the original
`LE1901-2000.html` and `LE2001-2100.html`. Select another binary with `--anno-bin`.

To exercise full HTML ingestion, add `--web` and use a CLI built with URL
support, for example:

```sh
cargo build --locked -p anno-cli --no-default-features --features eval,extractor-html2text
python3 scripts/qa/check_nasa_dates.py --web --anno-bin target/debug/anno
```

The script serves the hash-checked HTML on loopback, then locates each date
by catalog row ID in the CLI's rendered source. It scores exact character
ranges within the 457 calendar cells. Dates in headers and other columns are
outside this panel. Pass `--source-dir DIR` to reuse downloaded snapshots.
