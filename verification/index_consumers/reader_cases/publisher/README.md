# eLife historical byline holdout

For one of twelve consecutive correction notices, substituting the corrected article for the earlier edition changes the answer to “who appeared on the byline at the earlier date?” All twelve works kept the same number of authors. This fixture checks the ordered, publisher-credited byline at an edition cut. Identity equivalence, contributor roles and elapsed operator time each need a different reference.

| Historical query arm | Answered / selected works | Answers with a dated reference | Wrong ordered bylines |
| --- | ---: | ---: | ---: |
| Substitute the correction-day byline | 12/12 | 12 | 1/12 |
| Read the last earlier edition | 12/12 | 12 | 0/12 |

For the changed work, the third author's suffix appeared in v5 on 2023-01-11. The v4 byline dated 2022-01-10 omitted the suffix. The notice describes other corrections. The versioned publisher API author line and the typeset first pages independently confirm how the publisher displayed the two bylines. The publisher supplies all three formats. These observations establish an actual historical display difference and leave the person's identity and the reason for the edit open. [Correction](https://doi.org/10.7554/eLife.86017) · [Version index](https://api.elifesciences.org/articles/70062/versions) · [v4 printed page](https://cdn.elifesciences.org/articles/70062/elife-70062-v4.pdf) · [v5 printed page](https://cdn.elifesciences.org/articles/70062/elife-70062-v5.pdf)

## Selection and reference

[sampling.txt](sampling.txt) freezes the rule before notice inspection. Crossref returned 1,871 distinct eLife work registrations with a 2023 publication date, including 29 correction notices. Eligibility required a correction relation or a title beginning with “Correction”, “Erratum” or “Corrigendum”. All 29 had a correction relation with one target. We sorted by publication date then DOI and kept the first 12, spanning January 3 to May 18. All 12 selected titles, correction links, target DOIs and source pairs were present. They describe corrections to content other than the byline. The other 17 remain in the published selection frame and have no scored cases. [Crossref eLife journal works](https://api.crossref.org/journals/2050-084X/works) · [eLife public archive](https://github.com/elifesciences/elife-article-xml/tree/68e2b8dcb1591b86bd58cbb85b0178fc9b68b9b6)

The version-aware reader selects the latest target release strictly before the notice day. Its comparator takes the unique release on the notice day for that same DOI. The read uses the publisher's UTC version date. The original publication date repeats across revisions. None of the 12 cuts had multiple releases on the same day. With only day precision, ambiguous within-day chronology requires stronger evidence.

For each query we also retrieved the earlier article XML from a repository commit timestamped before the notice, and recorded its hash, timestamp and revision. Eleven copies came from the latest eligible commit touching that file. For notice 88685 the path-history query returned HTTP 500. The earlier edition was present in the repository-wide commit on April 18 before the April 19 notice. All 12 historical XML files matched the corresponding retained older editions byte for byte. This is an earlier snapshot of the same publisher record. Agreement measures recovery of that record. The experiment cannot estimate real-time capture delay, human decision quality or correctness about people in the world.

We compared the ordered names of immediate JATS byline authors in document order, including the literal prefix, given names, surname and suffix where present. Whitespace and Unicode normalization remove formatting differences. Case, accents, order, duplicate names and printed suffixes remain significant. The reader also handles simple literal collaboration and string names. Ambiguous or unsupported name structures yield an unknown byline. The 12 earlier editions contain 107 author positions in total. The experimental unit is the work. Eleven unchanged bylines are controls for this query only. They supply no independently labeled different-person pairs. One disagreement in this chronological window cannot support a population error rate, a model comparison or a deployment threshold.

## Replay

Run the receipt comparison without network access from the repository root.

```sh
python3 verification/index_consumers/reader_cases/publisher/check.py
```

Recompute each ordered fingerprint, work DOI, correction target, license, source hash and release cut against original publisher material. The fetch mode populates an external directory. The cache mode reuses it completely offline.

```sh
python3 verification/index_consumers/reader_cases/publisher/check.py --fetch /tmp/anno-elife-xml
python3 verification/index_consumers/reader_cases/publisher/check.py --cache /tmp/anno-elife-xml
```

To reproduce selection independently, cache the registrar pages and compare them with the full 29-row eligible frame. Later deposit changes can alter a live registry's results. The checker reports a difference in that case.

```sh
mkdir -p /tmp/anno-elife-registry
for offset in 0 1000; do
  curl --fail --silent --show-error --get 'https://api.crossref.org/journals/2050-084X/works' \
    --data-urlencode 'filter=from-pub-date:2023-01-01,until-pub-date:2023-12-31' \
    --data-urlencode 'select=DOI,title,published,update-to' \
    --data-urlencode 'sort=published' --data-urlencode 'order=asc' \
    --data-urlencode 'rows=1000' --data-urlencode "offset=$offset" \
    --output "/tmp/anno-elife-registry/year_$offset.json"
done
python3 verification/index_consumers/reader_cases/publisher/check.py --registry-cache /tmp/anno-elife-registry
```

## Source rights

[sample.json](sample.json) stores DOI suffixes under `10.7554/eLife.`, dates, revisions, source hashes, normalized byline hashes, counts and original review labels. eLife and the article authors supply the originals. Each XML read checks its own license URI. Across 48 reads (including historical references), 45 declare CC BY 4.0 and three declare CC0 1.0. The fixture ships bibliographic facts and derived fingerprints. Readers can retrieve the original credits, notices, article text and case-specific license from the pinned sources. [eLife terms](https://elifesciences.org/terms) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) · [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) · [Crossref API](https://www.crossref.org/documentation/retrieve-metadata/rest-api/)
