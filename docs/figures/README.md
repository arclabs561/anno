# Figures used in the Anno guides

The guides use these figures to describe datasets, demonstrate extraction behavior and compare model results. The documentation embeds the PNGs from this directory. A regular Git checkout includes the images and the measurements used to draw them.

| Figure | Guide and reader’s question | Generator |
|---|---|---|
| [Coverage by domain](dataset-coverage.png) | [Which sources did the survey measure?](../eval/dataset-coverage.md) | [Coverage](../../scripts/plot_measurement_coverage.py) |
| [Dated S3 publisher matches](s3-publisher-matches.png) | [Which cached inputs match a measured publisher file?](../eval/dataset-coverage.md#what-the-s3-cache-establishes) | [S3 match outcomes](../../scripts/plot_s3_source_evidence.py) |
| [Direct-reader selection](anno-selection.png) | [Which formats and languages reached the direct reader?](../eval/dataset-coverage.md#what-the-readers-actually-opened) | [Coverage](../../scripts/plot_measurement_coverage.py) |
| [Annotation density](dataset-density.png) | [What share of tokens carries an annotation?](../eval/dataset-coverage.md#compare-annotation-density-across-datasets) | [Density](../../scripts/plot_density_ranges.py) |
| [WikiANN split estimates](wikiann-split-density.png) | [Can my training split estimate what reviewers will see highlighted?](../QA.md#which-datasets-do-the-token-counts-describe) | [WikiANN](../../scripts/plot_wikiann_split_density.py) |
| [Benchmark chooser](benchmark-chooser.png) | [Which labels does my task need?](../QA.md#find-a-dataset-or-add-a-case) | [Worked examples](../../scripts/plot_benchmark_examples.py) |
| [GENIA nested mentions](anno-genia-nesting.png) | [Which annotated mentions disappear in HTML highlights?](../EXTRACTION_WORKFLOWS.md#review-overlapping-highlights) | [GENIA](../../scripts/plot_genia_nesting.py) |
| [WNUT unfamiliar names](wnut-novelty.png) | [Do the systems miss the mention or assign the wrong type?](../DOMAIN_QA.md#test-recognition-of-unfamiliar-names) | [WNUT](../../scripts/plot_wnut_novelty.py) |
| [Native Anno on WNUT-17](wnut-native-comparison.png) | [Did the release candidate recover more unfamiliar names?](../DOMAIN_QA.md#compare-native-anno-on-unfamiliar-names) | [Native comparison](../../scripts/plot_wnut_native.py) |
| [CrossRE prompts](crossre-prompts.png) | [How much does revealing test labels change relation F1?](../eval/crossre-full-prompt-comparison.md) | [Worked examples](../../scripts/plot_benchmark_examples.py) |
| [NER across languages and lengths](ner-language-length.png) | [Which groups support changing the backend?](../eval/ner-language-length.md#paired-glinerel-and-heuristic-results) | [Multilingual NER](../../scripts/plot_ner_language_length.py) |

Use `just dataset-figures` to rebuild previews or `just dataset-figures --update` to refresh the published images and derived CSVs. Pull requests and merges to `main` that change figure inputs or generators run the build and save the eleven PNGs, eleven SVGs and five CSVs in the **dataset-figures** Actions artifact for 30 days. [Regeneration and source refresh](../../scripts/README.md#reproduce-the-dataset-figures)

A manual **Dataset figures** run on `main` also builds an additional figure from the live, authenticated S3 inventory. Its **s3-dataset-cache-figures** artifact contains the PNG, SVG and aggregate verification results. The live figure describes the cache at the time of that run. [Run the workflow and download its artifacts](https://github.com/arclabs561/anno/actions/workflows/dataset-figures.yml) · [S3 verification scope](../S3_CACHE_CI.md#verify-the-live-dataset-cache)

The coverage guide embeds the dated comparison from its checked-in receipt. The separate **cache-source-evidence** artifact compares eligible S3 payload bytes with files used in the surveys. Its dated PNG, SVG, CSV and JSON distinguish verified matches from reads without a matching publisher file, and account for skipped payloads. It refreshes when measured-file evidence changes on `main` and is included in a manual **Dataset figures** run. [Publisher match workflow](https://github.com/arclabs561/anno/actions/workflows/cache-source-evidence.yml)
