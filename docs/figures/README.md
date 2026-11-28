# Figures used in the Anno guides

The guides use these figures to describe datasets, demonstrate extraction behavior and compare model results. A regular Git checkout includes the images and the measurements used to draw them. Native Anno charts also include a standalone HTML report with the counts, and an SVG for the guide. The dataset explorer links registry IDs to measured files and recorded backend runs. [Open the offline dataset explorer](../eval/dataset-explorer.html)

| Figure | Guide and reader’s question | Generator or native chart input |
|---|---|---|
| [Coverage by domain](dataset-coverage.png) | [Which sources did the survey measure?](../eval/dataset-coverage.md) | [Coverage](../../scripts/plot_measurement_coverage.py) |
| [Dated S3 publisher matches](s3-publisher-matches.png) | [Which cached inputs match a measured publisher file?](../eval/dataset-coverage.md#what-the-s3-cache-establishes) | [S3 match outcomes](../../scripts/plot_s3_source_evidence.py) |
| [Direct-reader selection](anno-selection.png) | [Which formats and languages reached the direct reader?](../eval/dataset-coverage.md#what-the-readers-actually-opened) | [Coverage](../../scripts/plot_measurement_coverage.py) |
| [Biomedical NER density](dataset-density-biomedical.png) | [How dense are the annotations in comparable biomedical NER corpora?](../eval/dataset-coverage.md#compare-annotation-density-across-datasets) | [Density](../../scripts/plot_density_ranges.py) |
| [Density reference for all 81 measured IDs](dataset-density.png) | [What share of tokens carries an annotation?](../eval/dataset-coverage.md#compare-annotation-density-across-datasets) | [Density](../../scripts/plot_density_ranges.py) |
| [WikiANN split estimates](wikiann-split-density.png) | [Can my training split estimate what reviewers will see highlighted?](../QA.md#which-datasets-do-the-token-counts-describe) | [WikiANN](../../scripts/plot_wikiann_split_density.py) |
| [Benchmark chooser](benchmark-chooser.png) | [Which labels does my task need?](../QA.md#find-a-dataset-or-add-a-case) | [Worked examples](../../scripts/plot_benchmark_examples.py) |
| [GENIA nested mentions](anno-genia-nesting.png) | [Which annotated mentions disappear in HTML highlights?](../EXTRACTION_WORKFLOWS.md#review-overlapping-highlights) | [GENIA](../../scripts/plot_genia_nesting.py) |
| [GENIA span overlap](genia-span-overlap.svg) | [Which labels and how much sentence text participate in overlapping gold annotations?](../EXTRACTION_WORKFLOWS.md#review-overlapping-highlights) | [Full release overlap counts](../eval/genia-span-overlap.json) |
| [WNUT unfamiliar names](wnut-novelty.png) | [Do the systems miss the mention or assign the wrong type?](../DOMAIN_QA.md#test-recognition-of-unfamiliar-names) | [WNUT](../../scripts/plot_wnut_novelty.py) |
| [Native Anno on WNUT-17](wnut-native-comparison.png) | [Did the release candidate recover more unfamiliar names?](../DOMAIN_QA.md#compare-native-anno-on-unfamiliar-names) | [Native comparison](../../scripts/plot_wnut_native.py) |
| [Running a skipped NuNER call](nuner-routing-tradeoff.png) | [What does the extra inference recover when BERT is present or missing?](../eval/nuner-fallback-routing.md) | [Routing comparison](../../scripts/plot_nuner_routing.py) |
| [NuNER on WikiGold and WNUT](wikigold-routing-paired.svg) | [Does the routing decision transfer to encyclopedia text?](../eval/wikigold-nuner-routing.md) | [Paired calls, suggestions and latency](../eval/wikigold-routing-paired.json) |
| [WikiGold entity errors](wikigold-entity-errors.svg) | [How much of the remaining error is a label or boundary mismatch?](../eval/wikigold-nuner-routing.md#which-types-benefit) | [Full release evaluation counts](../eval/wikigold-entity-errors.json) |
| [A fourth NuNER prompt](nuner-misc-prompt-paired.svg) | [Which MISC mentions did the prompt recover, and which previously correct mentions did it displace?](../eval/wikigold-nuner-routing.md#does-a-fourth-prompt-recover-misc) | [WikiGold and WNUT paired outcomes](../eval/nuner-misc-prompt-paired.json) |
| [Correct pronoun links retained](native-gap-retention-paired.svg) | [How many wrong-person links does abstention avoid at a chosen correct-link retention?](../eval/native-gap-link-retention.md) | [Selected policy across splits and supplied boundaries](../eval/native-gap-retention-paired.json) |
| [CrossRE prompts](crossre-prompts.png) | [How much does revealing test labels change relation F1?](../eval/crossre-full-prompt-comparison.md) | [Worked examples](../../scripts/plot_benchmark_examples.py) |
| [CrossRE rule selection](crossre-predicate-paired.svg) | [Which predicate scores justify outputting a directed relation?](../eval/crossre-predicate-ranking.md) | [Paired development and original test results](../eval/crossre-predicate-paired.json) |
| [CrossRE at equal review budgets](crossre-full-test-paired.svg) | [How many correct relations does an extra review slot recover across the official test release?](../eval/crossre-predicate-ranking.md#do-the-rules-transfer-to-the-remaining-test-sentences) | [Fresh holdout, overlaps and release total](../eval/crossre-full-test-paired.json) |
| [CrossRE endpoint confidence](crossre-endpoint-paired.svg) | [Does weighting both endpoints improve a smaller fixed review queue?](../eval/crossre-predicate-ranking.md) | [Calibration, unused development texts and previously examined official test data](../eval/crossre-endpoint-paired.json) |
| [NER across languages and lengths](ner-language-length.png) | [Which groups support changing the backend?](../eval/ner-language-length.md#paired-glinerel-and-heuristic-results) | [Multilingual NER](../../scripts/plot_ner_language_length.py) |

The detailed operating-point sweeps remain available for [GAP](native-gap-margin-tradeoff.svg), [WikiGold and WNUT](wikigold-routing-tradeoff.svg), [original CrossRE](crossre-predicate-tradeoff.svg) and [full-release CrossRE](crossre-full-test-tradeoff.svg). Each names settings directly and reports how many suggestions they produce.

## Where new visualizations belong

Put an operation in `anno viz` when a developer can apply it to their own inputs. Counting exact matches gained and lost, measuring overlapping spans, and rendering a portable report each pass that test. A new backend or threshold can be tested with the existing commands.

Keep the study design with its data. Choosing CrossRE's holdout, mapping WikiGold labels, verifying S3 bytes and drawing a corpus-specific Matplotlib figure belong in the research scripts. Publish the counts and guide through the external catalog. Reject a new CLI flag whose only job is to reproduce one paper figure. [Generators and their measurement inputs](../../scripts/README.md#reproduce-the-dataset-figures)

## Reproduce and download

From an Anno checkout or unpacked figure artifact, use `anno viz list --dataset WikiGold` to discover the available native studies, then `anno viz render wikigold-routing-paired -o routing.html`. To redraw a portable report from a file or pipe, use `anno viz render`. To use a particular snapshot from another directory, pass `--catalog PATH` or set `ANNO_FIGURE_CATALOG`. [Portable study catalog](../eval/figure-catalog.json) [Native chart formats and CLI examples](../QUICKSTART.md#visualize-evaluation-results)

`just dataset-figures` rebuilds the full collection, and `just dataset-figures --update` refreshes the published outputs. CI saves every SVG and PNG, the native study inputs and HTML reports, the searchable dataset reference and its derived CSVs in the **dataset-figures** Actions artifact for 30 days. [Regeneration and source refresh](../../scripts/README.md#reproduce-the-dataset-figures)

A manual **Dataset figures** run on `main` also builds an additional figure from the live, authenticated S3 inventory. Its **s3-dataset-cache-figures** artifact contains the PNG, SVG and aggregate verification results. The live figure describes the cache at the time of that run. [Run the workflow and download its artifacts](https://github.com/arclabs561/anno/actions/workflows/dataset-figures.yml) · [S3 verification scope](../S3_CACHE_CI.md#verify-the-live-dataset-cache)

The coverage guide embeds the dated comparison from its checked-in receipt. The separate **cache-source-evidence** artifact compares eligible S3 payload bytes with files used in the surveys. Its dated PNG, SVG, CSV and JSON distinguish verified matches from reads without a matching publisher file, and account for skipped payloads. It refreshes when measured-file evidence changes on `main` and is included in a manual **Dataset figures** run. [Publisher match workflow](https://github.com/arclabs561/anno/actions/workflows/cache-source-evidence.yml)
