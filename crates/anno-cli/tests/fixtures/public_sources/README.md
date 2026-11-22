# Public-source pattern regressions

Each UTF-8 file contains a brief excerpt with a final newline. The CLI test in
`tests/public_sources.rs` checks all structured entities expected from the
pattern backend on these excerpts. Offsets are half-open Unicode scalar indices.
People, places and other learned labels are outside this test's scope.

- **boe.txt** — Spanish Constitution, repeal provision §2, from the sentence ending in “Vizcaya” through its following sentence. The line break represents the paragraph boundary. The date follows accented letters on both lines. [Official text, BOE-A-1978-31229](https://www.boe.es/buscar/act.php?id=BOE-A-1978-31229&p=19781229&tn=1), [BOE reuse terms](https://www.boe.es/informacion/aviso_legal/).
- **aozora.txt** — Natsume Sōseki, *吾輩は猫である* (*I Am a Cat*), opening line. This public-domain literary excerpt includes the full-width indentation and Aozora's inline pronunciation markup from the original CP932 transcription. It has no structured entities. [Aozora edition 789](https://www.aozora.gr.jp/cards/000148/card789.html), [pinned transcription](https://github.com/aozorahack/aozorabunko_text/blob/0984f7dc18a2270bdfd2ff99bf14fd1b3ebb4f65/cards/000148/files/789_ruby_5639/789_ruby_5639.txt).
- **tzdb.txt** and **tzdb-midnight.txt** — IANA Time Zone Database release 2025b, release header and Iran correction respectively. In the latter, +0330 is a UTC offset and 24:00 is local end-of-day. Both excerpts are from the public-domain release notes. [NEWS](https://github.com/eggert/tz/blob/2025b/NEWS), [license](https://github.com/eggert/tz/blob/2025b/LICENSE).
