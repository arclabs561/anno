# Scoring a dated bibliography export

Each selected notice contributes three queries about the credited byline on its target work. The selectors address the first edition, the last edition before the notice day and the last edition through the notice day. An ordered author list can stay the same across two different editions, so the scorer measures the edition, structured author list and printed author list separately. The printed reference comes from the independently read PDF wherever the curator could recover the full list. [Frozen scoring protocol](protocol.txt)

## Frozen holdout results

Both readers answered all 51 queries about 17 distinct works. The dated reader selected the reference edition for all 51 and reproduced the structured JATS byline for all 51. Its author strings matched the independently transcribed printed PDF on 43 queries. The latest-edition baseline matched 17 editions, 42 structured bylines and 37 printed bylines. The actual latest edition at capture fell on the notice day for all 17 works. [Original numerical report](receipts/official.json)

| Correct answer / 17 per cut | First edition | Before notice | Through notice |
| --- | ---: | ---: | ---: |
| Dated, edition | 17 | 17 | 17 |
| Latest, edition | 0 | 0 | 17 |
| Dated, structured author list | 17 | 17 | 17 |
| Latest, structured author list | 11 | 14 | 17 |
| Dated, printed author list | 9 | 17 | 17 |
| Latest, printed author list | 6 | 14 | 17 |

Three works changed their bylines across the notice. The dated reader reproduced both sides of all three changes in both reference formats. Across all cuts, it corrected six of the latest-edition reader’s printed-byline errors. Eight printed answers remained wrong under both policies. At least one printed answer was wrong on 8/17 works for the dated reader and 11/17 works for the latest-edition reader. Repeated cuts on the same work share evidence.

All eight dated mismatches occurred on first-edition accepted manuscripts. Their PDFs and the selected structured data listed the same numbers of authors. Two bylines differed only in periods after initials. Six differed in given-name text, including full versus abbreviated forms and additional or omitted given names. Differences also included an accent and a family-name spelling. A correct version number and a correct structured author list therefore did not guarantee exact reproduction of the printed bibliography on these releases. The independent curator reviewed these differences after unblinding; the original references and scores stand. [Reference with edition-specific PDF locators and source hashes](receipts/reference.json)

All 102 individual-versus-bulk response pairs agreed, including evidence locations and hashes, after receiving the same snapshot. These reads use the same lookup function. Running them in two modes checks interface consistency and supplies no independent accuracy estimates. This holdout covers one publisher's consecutive correction notices, with verified references for every cut. It does not measure ordinary publication frequency, source capture delay, reader time or unique personal identity.

## Re-score the public receipt offline

The bundled prediction receipt stores each distinct result once. Replay reconstructs the original prediction bytes, verifies their pre-unblinding digest against the original lock, checks the frozen scorer and reader, verifies the public reference digest, and computes the complete official report again. Public reference records retain source identifiers, licensing references where present, PDF pages and line ranges, source hashes and ordered-byline fingerprints. They omit author names and publisher content. The original [reader source](receipts/reader.py.txt) is archived byte for byte. New predictions use the current reader. The published PDF and XML remain at the cited publisher and archive URLs. The official score retains the SHA-256 of the original sealed key. The optional second command checks that the public fields match a local copy of its original bytes. [Original prediction lock](receipts/lock.json) · [Compact predictions](receipts/predictions.json)

```sh
python3 -m verification.index_consumers.reader_cases.bibliography.scoring.replay
python3 -m verification.index_consumers.reader_cases.bibliography.scoring.replay --original-key /path/to/sealed/key.json
```

## Reproduce a prediction lock

From the repository root, provide the collected work input and a new output directory. The prediction command does not take a reference key. It records both input digests, the code digests and every response, including abstentions. Its online arm issues individual calls against the supplied snapshot. Its bulk arm submits the same requests in an ordered batch. Both use the same reader logic. The mutable baseline substitutes the publisher's latest edition at the catalog capture date for every requested cut.

```sh
python3 -m verification.index_consumers.reader_cases.bibliography.scoring.predict \
  --frame verification/index_consumers/reader_cases/bibliography/sources/frame.json \
  --works /path/to/reader_input.json --out /path/to/new_run
```

Record the printed lock digest before unblinding. The prediction file holds the source locator and digest, edition number, and fingerprints for the structured and displayed author lists. It omits the literal author names. An available query whose reference is ambiguous or unavailable still counts as answered and remains unjudged for the affected component.

## Score an authorized, released key

```sh
python3 -m verification.index_consumers.reader_cases.bibliography.scoring.score \
  --run /path/to/new_run --lock-sha256 "$PREDICTION_LOCK_SHA256" \
  --reference /path/to/released_key.json
python3 -m unittest verification.index_consumers.reader_cases.bibliography.scoring.test_score
```

The report includes all 51 work–cut queries, counts of distinct affected works, independent printed-evidence coverage, historical answers recovered or spoiled by changing arms, and discrepancies between the two delivery modes. A missing, extra or duplicate row fails validation. Available and scorable have separate denominators. A single underlying work can contribute multiple queries and the two delivery modes repeat each one. The checks measure the supplied publisher evidence and API behavior. They provide no estimate of subscriber lag, reviewer time, or personal identity.
