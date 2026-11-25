# Read a bibliography at a publisher edition

A bibliography client can ask who was credited at first publication, strictly before a correction, or through a specified UTC date. The reader returns the edition and the ordered author fields in the publisher's JATS XML. Its source URL and digest identify the original XML. A request for the author's spelling as it appeared on the page requires the dated PDF. [Publisher sources and terms](sources/README.md) · [Printed-page references](sources/curation.md)

## Ask a question

Run these commands from the repository root. Capture the inputs once in an external cache. Keep the generated attribution file with the prepared bylines. For repeated use, omit `--fetch` to verify the cached input against the recorded source digests.

```sh
dir=verification/index_consumers/reader_cases/bibliography
python3 "$dir/sources/capture.py" --fetch --cache /tmp/anno-elife-inputs --output /tmp/anno-elife-reader
python3 "$dir/reader.py" /tmp/anno-elife-reader/reader_input.json --ask <<'JSON'
{"doi":"10.7554/eLife.74094","cut":"before","date":"2023-06-12"}
JSON
```

Use `{"doi":"10.7554/eLife.74094","cut":"first"}` for the first numbered journal edition, or `"cut":"as_of"` with a date to include releases on that day. For a bulk export, pass a JSON array of queries in place of `--ask`. The result preserves query order and repeated queries. A missing release that could change the selection, or an unavailable selected byline, yields an unavailable answer. Numbered releases establish ordering within the day. Publisher release chronology supplies no subscriber arrival timestamp.

Add `--anno PATH_TO_ANNO` to submit each distinct available byline to `anno query publication`. The CLI checks the projected text digest, edition, author positions and Unicode ranges. This projection keeps each author's typed fields in occurrence order. Its text digest identifies different bytes from those referenced by the publisher XML receipt.

## Reopen the printed page

For an edition selected by the reader, the [page opener](open_page.py) validates a cached PDF against a names-free receipt for the DOI and edition and returns the local and publisher page locators. For example, after downloading the publisher's v1 PDF for work 21221 to the path below, run

```sh
python3 -m verification.index_consumers.reader_cases.bibliography.open_page \
  10.7554/eLife.21221 1 /tmp/elife-21221-v1.pdf
```

The receipt covers all 48 observed work editions. The file must match the expected SHA-256 at lookup time. The operation was added after scoring so that a caller could inspect the exact artifact when a rendering differs from structured metadata. It does not supply author text to the index or change the recorded score. [Versioned PDF](https://cdn.elifesciences.org/articles/21221/elife-21221-v1.pdf) · [Verification method](sources/curation.md)

## What the holdout established

The remaining 17 works in the frozen eLife correction frame supplied 51 work-and-date questions. An independent curator recorded the ordered names on dated PDFs before inspecting the structured JATS names. The scorer opened that key after freezing the reader, reference protocol and predictions.

| Method | Correct edition | Exact XML author fields | Exact names on the PDF |
| --- | ---: | ---: | ---: |
| Read the edition selected by the question | 51/51 | 51/51 | 43/51 |
| Use the latest captured edition for every question | 17/51 | 42/51 | 37/51 |

Both methods answered every question. All eight dated-reader PDF discrepancies concerned first-edition accepted manuscripts. All had the expected author count. Two differed only in punctuation after initials and six also differed in the name text. The XML and PDF originate from the same publisher. These comparisons establish what two representations of a particular credit contained. They provide no labels for linking a person across works. [Locked predictions, reference and score](scoring/README.md)

Run the preserved score offline with `python3 -m verification.index_consumers.reader_cases.bibliography.scoring.replay`. All 102 online/bulk comparisons agreed after receiving the same evidence. The local publication checks separately deliver late and duplicate snapshots and exercise revocation while generating and reopening an export.

## Publish an authorized export

The [SQLite example](publication.py) takes complete catalog and per-viewer policy snapshots from an application. Install the snapshots, export and download through the same store. The caller supplies an authenticated viewer. A metadata-only permission releases the selected edition and publication day while withholding byline values, source URLs and whole-document digests.

```python
from verification.index_consumers.reader_cases.bibliography.publication import Publication

store = Publication(database_path)
store.publish_catalog(1, works)
store.publish_policy(viewer, 1, {work_doi: "full"})
question = {"doi": work_doi, "cut": "first"}
row = store.online(viewer, question)
handle = store.export(viewer, [question])
export_bytes = store.download(viewer, handle)
```

The store checks both revisions before publishing a handle and again before serving cached bytes. A permission change can prevent a future download. A recipient can retain an earlier copy. This local example assumes its database and source cache are private to the serving application.

## Challenge a date cut

```sh
dir=verification/index_consumers/reader_cases/bibliography
python3 "$dir/challenge.py" --seed 41 --rounds 256
```

The shared seed ran 4,450 synthetic queries. The deliberate one-day error at seed 41 reduces to two editions and one question. To export this witness, run `python3 "$dir/challenge.py" --seed 41 --fault --save /tmp/new-biblio-witness` with a new path. The faulty caller exits with status 1. Replay the saved inputs and independent key with `python3 "$dir/challenge.py" --replay /tmp/new-biblio-witness`; the dated reader passes. A proposer can keep new seeds local until the minimized case, expected answer and reuse terms are ready for shared replay.
