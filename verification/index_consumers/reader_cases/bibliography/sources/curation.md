# Reference for the printed byline

The question is which ordered names a reader can see on the opening page of the specified publisher edition. Seventeen correction notices produced 51 queries and 45 distinct PDFs. Six editions each serve two cuts. Thirty-six PDFs are versions of record and nine are publisher-released accepted manuscripts. All 45 have a readable byline on page one. The public reference includes each opening-page link, credit count, fingerprint, source hash, page, text-line span and article license. [Reference receipts](../scoring/receipts/reference.json)

## What fixes an edition

The frame fixes the correction notice DOI, target DOI and notice day. An independent curator retrieved each target’s publisher release index and selected dated, numbered editions of that DOI. The first cut takes the earliest publisher release. The pre cut takes the latest release before the start of the UTC notice day. The post cut takes the latest release before the start of the following UTC day. External preprints have their own identifiers. The original publication date can repeat across revisions, so the comparison uses the publisher’s edition-specific `versionDate`. [Frame](frame.json) · [Example publisher release history](https://api.elifesciences.org/articles/74094/versions)

For every pre cut, the curator also retrieved the repository-wide XML snapshot from the latest available commit before the UTC notice day and checked the target DOI, edition-specific PDF self-link, license, and archived XML bytes. All 17 historical XML files matched the corresponding edition file in the pinned archive byte for byte. The commit establishes an archived copy before the notice. The publisher’s release index supplies its effective release date. A versioned PDF retrieved for this study establishes what that PDF displays when retrieved. It cannot date an earlier download of the identical PDF bytes. [Publisher XML archive](https://github.com/elifesciences/elife-article-xml/tree/68e2b8dcb1591b86bd58cbb85b0178fc9b68b9b6) · [Per-work archival commits](../scoring/receipts/reference.json)

## How the printed reference was made

A separate Codex agent acted as curator. It used neither reader predictions nor the publisher API’s name projection to construct the printed key. It read the opening page of every selected PDF using Poppler’s layout-preserving text extraction, located each complete byline between the title and affiliations, and reviewed the ordered names for every edition. It also inspected rendered crops of five editions with difficult typography or line breaks. These observations received no separate human annotation. The publisher supplies the PDF, the release chronology and the structured XML.

The transcription retains case, accents, hyphens, initial periods, order, duplicates, and any group names or suffixes. Manuscript line numbers, affiliation/correspondence superscripts and separators lie outside the names. In one manuscript a hyphenated surname crossed a line. In another, two authors were separated by a superscript without a comma. A discretionary character next to a visibly printed hyphen was removed from the PDF text layer. The public verifier encodes these decisions for these particular pages. It uses the curator’s selected line spans and does not discover arbitrary bylines. The printed fingerprint hashes a compact UTF-8 JSON array of strings after NFC and whitespace normalization. The structured fingerprint hashes typed JATS name components, which are available from the publisher archive. A PDF cannot in general establish the boundary between family names and given names.

The curator froze the full private PDF transcription at SHA-256 `f8dbc1399e4541ab5ce6d961d6af2aef3789198c06eb36e45e3a5061a41ba9ac` before examining these JATS byline fields. It then checked the structured author components using its own XML reader and delivered the separate key at SHA-256 `a85997443eca7a35c7ca3c56440aade47894d822b6ab66dddc4aa180edf022d5`. The lead authorized comparison after the versioned reader and latest-state predictions were sealed. [Scoring protocol](../scoring/protocol.txt)

## What the eight first-edition differences contain

Eight accepted-manuscript opening pages differ from their edition’s structured author display. All eight have the same number of credited names in both representations. Two works differ only in periods on initials. Six others also differ in the presence or expansion of given-name text. These differences include one accent difference and one family-name spelling difference. The original edition links below show the visible text. These are observations about the publisher’s records. They provide no identity labels or account of who made a change.

| Differences observed in the opening PDF and JATS | First-edition sources |
| --- | --- |
| Periods on initials only | [60563](https://cdn.elifesciences.org/articles/60563/elife-60563-v1.pdf#page=1), [79863](https://cdn.elifesciences.org/articles/79863/elife-79863-v1.pdf#page=1) |
| Given-name components added, omitted or expanded, sometimes with punctuation differences | [65092](https://cdn.elifesciences.org/articles/65092/elife-65092-v1.pdf#page=1), [66393](https://cdn.elifesciences.org/articles/66393/elife-66393-v1.pdf#page=1), [74094](https://cdn.elifesciences.org/articles/74094/elife-74094-v1.pdf#page=1), [78649](https://cdn.elifesciences.org/articles/78649/elife-78649-v1.pdf#page=1) |
| Given-name differences plus an accent difference | [80254](https://cdn.elifesciences.org/articles/80254/elife-80254-v1.pdf#page=1) |
| Given-name differences plus a family-name spelling difference | [81453](https://cdn.elifesciences.org/articles/81453/elife-81453-v1.pdf#page=1) |

## Replaying the PDF evidence

With Python 3 and Poppler’s `pdftotext` on the path, run these commands from the repository root. The first downloads about 550 MiB of PDF material into an external directory. The second verifies the same source bytes and transcription hashes offline. The recorded text lines were extracted with Poppler 25.03.0. Changed PDF bytes or different layout extraction can prevent verification.

```sh
python3 verification/index_consumers/reader_cases/bibliography/sources/verify_print.py --cache /tmp/anno-elife-byline-pdfs --fetch
python3 verification/index_consumers/reader_cases/bibliography/sources/verify_print.py --cache /tmp/anno-elife-byline-pdfs
```

The PDFs and full bylines stay outside Git. The original authors and eLife provide the rendered articles and version-specific credit lists at the source URLs. Each PDF receipt carries the license recorded in that edition’s JATS permissions. Of the 45 distinct editions, 43 declare CC BY 4.0 and two declare CC0 1.0. [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) · [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)

## Opening pages outside the reference cuts

The names-free opening-page index covers all 48 target editions in the source input. Forty-five were already opened for the sealed reference. After the prediction lock, the curator separately downloaded and opened three intermediate PDFs, confirmed a readable title and credited byline on physical page one, and recorded each complete PDF's SHA-256. These extra editions supply openable locations only and add zero scored queries. [Opening-page index](opening_pages.json) · [65092 v2](https://cdn.elifesciences.org/articles/65092/elife-65092-v2.pdf#page=1) · [78649 v2](https://cdn.elifesciences.org/articles/78649/elife-78649-v2.pdf#page=1) · [81453 v2](https://cdn.elifesciences.org/articles/81453/elife-81453-v2.pdf#page=1)
