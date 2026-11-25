# Publisher source capture for historical byline retrieval

This source pack records which eLife editions could be supplied to a reader for each of the final 17 correction notices in the existing 2023 selection frame. It contains the identifiers, version times, source URLs, per-artifact SHA-256 values, and rights of the inputs. Ordered author names and full source documents are materialized only in an external cache. The scorer obtains its own historical answer key.

## Intake

The 29 candidate notices were ordered by Crossref publication date then DOI. The earlier study had used the first 12. [frame.json](frame.json) freezes positions 13–29 of [the existing registration frame](../../publisher/sample.json) before inspecting those notices. It supplies the notice day and target as published in that frame. Prefix each numeric `notice` or `target` with `10.7554/eLife.` to obtain its DOI. The frame source SHA-256 and the SHA-256 of the selected rows (key-sorted compact UTF-8 JSON followed by LF) allow independent checks of the selection. [Original selection policy](../../publisher/sampling.txt)

For each notice and target, the collector opened the publisher's version index. It retained every contiguous numbered eLife version, including publications before final editing (`poa`) and versions of record (`vor`). Twelve unnumbered external preprint entries fall outside that coverage. The per-edition JSON supplied its `versionDate`, DOI, license and XML link. The collector opened the linked JATS and verified the article's own DOI and license. For each notice it also verified the explicit JATS correction relation against the frozen target. [eLife developer API](https://api.elifesciences.org/) · [eLife publishing policies](https://elifesciences.org/terms)

| Observed at 2026-11-25 22:52 UTC | Target articles | Correction notices |
|---|---:|---:|
| Complete publisher version indexes | 17 | 17 |
| Numbered publisher editions | 48 | 17 |
| Version-specific API records and verified linked XML | 48 | 17 |
| Editions declaring CC BY 4.0 | 46 | 17 |
| Editions declaring CC0 1.0 | 2 | 0 |

All 17 targets have a numbered first release, an edition strictly before the notice day, and an edition on the notice day. At this capture, the latest available edition for every target also fell on the notice day. In one target's history, two earlier editions shared a UTC day. The input retains both numbers and the receipt retains the full UTC timestamps. The `through` certificate ends on **2026-11-24**, the last complete UTC day before capture. The `published_on` wire field is the UTC calendar date of that **edition's versionDate**. The article's original publication date can repeat across versions. A current version index reconstructs publisher chronology; it cannot establish when a live subscriber saw the change.

## Rebuild the reader input

From the repository root, use a fresh external cache and output directory. The first command fetches the public version indexes, version metadata and XML bodies. The second checks an already populated cache offline. A successful run verifies the pinned [receipt](receipt.json) and produces `reader_input.json`, `receipt.json` and `attributions.json` outside the checkout.

```sh
dir=verification/index_consumers/reader_cases/bibliography/sources
python3 "$dir/capture.py" --fetch --cache /tmp/anno-elife-inputs --output /tmp/anno-elife-reader
python3 "$dir/capture.py" --cache /tmp/anno-elife-inputs --output /tmp/anno-elife-reader
```

The publisher index and edition API are live endpoints. A later change, missing response, or altered XML byte stream causes the pinned replay to fail. It writes an external `unverified-receipt.json` on a receipt mismatch. To record a distinct observation, use a new empty external cache and output directory and pass `--observe-at` with the actual UTC capture timestamp. Existing cached responses are reused, so retaining one cache per observation matters.

The wire is an object containing a `works` array. A work contains its DOI, the inclusive chronology certificate and its ordered editions. A supported edition contains an edition number, UTC day, selected XML URL, XML hash, license URL and ordered byline. Each author is represented by ordered component pairs. Person parts preserve prefix, given names, surname and suffix. Simple literal collaboration and string names retain their respective kinds. Names use NFC and whitespace normalization; ordering, repeated positions and case remain significant. No person identifier or role is inferred. The implementation ignores editor and reviewer entries and restricts extraction to direct JATS byline authors.

If an announced edition's source or rights cannot be verified, that numbered edition stays in the feed with its date and no byline. The source receipt records why. The reader can then return an unavailable answer at the affected cut. An absent or inconsistent index cannot certify any cut; the receipt keeps its status and the affected work receives no complete feed entry. A missing notice source is reported independently of the availability of the target history. Every selected source and target was available in the pinned capture.

## Attribution

The observation records identify source articles by DOI and editions by exact publisher URLs. eLife and the original authors supply the works. The public receipt contains identifiers, dates, rights URIs, and source hashes. External prepared text is subject to the individual source's declared terms. The generated `attributions.json` records the DOI, edition, rights holder supplied by the publisher where present, URL and license. Keep that sidecar with any reuse of prepared bylines, and identify the bylines as extracted JATS components with normalized whitespace and Unicode. The notices and articles can have different rights. [eLife terms](https://elifesciences.org/terms) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) · [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)
