# Two independent temporal QA panels

Use MEANTIME to measure exact temporal spans and false alarms on naturally occurring news.
Use the Microsoft suite to probe specific expression types and lookalike negatives.
Keep their scores separate. Their inputs, annotation policies and sampling differ.

| Source | Rows | Gold spans | Gold-empty rows | Sampling unit |
|---|---:|---:|---:|---|
| NewsReader MEANTIME English v1.0 | 71 | 53 | 34 | 24 Wikinews articles across four topics |
| Microsoft Recognizers-Text English full-model spec | 56 | 40 | 16 | Eight positive expression types plus unequivocal negatives |

## Reproduce the panels

Get the English **manually annotated** zip from NewsReader. For Microsoft, download the English
default full-model JSON and the license from the pinned commit. These three files are the only
external inputs. [NewsReader download and license](https://cltl.github.io/newsreader.io/results/data/wikinews/) ·
[Microsoft JSON](https://raw.githubusercontent.com/microsoft/Recognizers-Text/da7edcff59f669b2a460ab9d400e36298f0d658e/Specs/DateTime/English/DateTimeModel.json) ·
[Microsoft license](https://raw.githubusercontent.com/microsoft/Recognizers-Text/da7edcff59f669b2a460ab9d400e36298f0d658e/LICENSE)

From the repository root, point the commands at those local files. Preparation runs offline with
Python's standard library and verifies the SHA-256 of every input and both resulting gold panels.

```sh
panel_dir=$(mktemp -d)
python3 scripts/qa/temporal_sources/prepare.py meantime ./meantime_newsreader_english.zip "$panel_dir/news"
python3 scripts/qa/temporal_sources/prepare.py msrt ./DateTimeModel.json ./LICENSE "$panel_dir/spec"
```

Each output directory contains model inputs (`inputs.jsonl`), separate gold (`panel.jsonl`),
the upstream license (`LICENSE.txt`), attribution and changes (`NOTICE.txt`), and a receipt with
hashes and source identifiers. Keep both text files with either JSONL when sharing it. The news
notice lists article credits without revealing any gold spans. For native Anno scoring,
follow the isolated-cache replay with the desired panel's `panel.jsonl` and its row count from the
table. For per-row inference, use `inputs.jsonl` and join predictions to gold by `id`.
[Native temporal replay](../fixtures/timex-contrast/README.md)

## What the labels cover

**MEANTIME.** Minard et al. manually annotated English Wikinews at document level for the
headline and the first four sentences. Later text has limited annotation coverage. Sample six
article IDs per topic by a fixed hash and retain body sentences with upstream IDs 2–4, after the
headline and dateline. Each retained sentence has document-level markables. Join the XML tokens
with single spaces, take only anchored TIMEX3 mentions and preserve the source span, type and ID.
One duplicate sentence occurs in two articles and is counted once. The resulting spans comprise
46 dates, five durations and two times. The receipt names each article, archive member and
Wikinews source URL. Estimate uncertainty by article. These four business topics limit transfer
to other domains. The corpus and its annotations are © 2015 Fondazione Bruno Kessler, VU
University Amsterdam and Universidad Del Pais Vasco, distributed under CC BY 4.0. The underlying
article text is credited to Wikinews. Four articles predate September 25, 2005 and fall under its
public-domain policy. The remaining twenty are from the CC BY 2.5 period. Each credit in the
generated notice identifies the corresponding article and its terms. [Minard et al., LREC 2016](https://aclanthology.org/L16-1699/) · [Corpus license](https://creativecommons.org/licenses/by/4.0/) · [Wikinews copyright policy](https://en.wikinews.org/w/index.php?title=Wikinews:Copyright&oldid=4991371) · [CC BY 2.5](https://creativecommons.org/licenses/by/2.5/)

**Microsoft.** The source is a human-maintained software specification. Remove repeated inputs,
require explicit spans with matching surfaces, and sample five unique cases in each of date,
time, datetime, duration, recurrence, date range, time range and datetime range. The sixteen
negatives are sampled from 39 upstream empty cases screened for clear non-temporal readings.
Other upstream empties include real dates handled by optional calendar modes. Convert inclusive
UTF-16 offsets to half-open Unicode scalar offsets and recover the original capitalization from
the input. Gold reflects the source's treatment of compound intervals. Its balanced proportions
carry no production prevalence estimate. © Microsoft Corporation, MIT license.
[Pinned specification](https://github.com/microsoft/Recognizers-Text/blob/da7edcff59f669b2a460ab9d400e36298f0d658e/Specs/DateTime/English/DateTimeModel.json)

Both panels were frozen before candidate outputs were examined. An overlap audit compared them
with each other, the 390-row public news test split, the 34-row contrast panel and the 20-row
weekday probe. There were zero normalized exact matches and zero word-trigram Jaccard matches
at 0.8 or above. Model pretraining exposure is unknown. After inspecting scores, treat these
rows as regression evidence. Reserve different source documents or cases for the next prospective
decision. When attributing gains, report exact matches, missed and extra spans, and the number
of gold-empty rows that received a prediction.
