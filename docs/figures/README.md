# Figures used in the Anno guides

The guides use these nine figures to describe datasets, demonstrate extraction behavior and compare model results. The documentation embeds the PNGs from this directory. A regular Git checkout includes the images and the measurements used to draw them.

| Figure | Guide and reader’s question | Generator |
|---|---|---|
| [Coverage by domain](dataset-coverage.png) | [Which sources did the survey measure?](../eval/dataset-coverage.md) | [Coverage](../../scripts/plot_measurement_coverage.py) |
| [Direct-reader selection](anno-selection.png) | [Which formats and languages reached the direct reader?](../eval/dataset-coverage.md#what-the-readers-actually-opened) | [Coverage](../../scripts/plot_measurement_coverage.py) |
| [Annotation density](dataset-density.png) | [What share of tokens carries an annotation?](../eval/dataset-coverage.md#compare-annotation-density-across-datasets) | [Density](../../scripts/plot_density_ranges.py) |
| [WikiANN split estimates](wikiann-split-density.png) | [Can training annotations estimate review volume in my language?](../eval/dataset-coverage.md#estimate-review-volume-for-a-wikiann-language) | [WikiANN](../../scripts/plot_wikiann_split_density.py) |
| [Benchmark chooser](benchmark-chooser.png) | [Which labels does my task need?](../QA.md#find-a-dataset-or-add-a-case) | [Worked examples](../../scripts/plot_benchmark_examples.py) |
| [GENIA nested mentions](anno-genia-nesting.png) | [Which annotated mentions disappear in HTML highlights?](../EXTRACTION_WORKFLOWS.md#review-overlapping-highlights) | [GENIA](../../scripts/plot_genia_nesting.py) |
| [WNUT unfamiliar names](wnut-novelty.png) | [Do the systems miss the mention or assign the wrong type?](../DOMAIN_QA.md#test-recognition-of-unfamiliar-names) | [WNUT](../../scripts/plot_wnut_novelty.py) |
| [CrossRE prompts](crossre-prompts.png) | [How much does revealing test labels change relation F1?](../eval/crossre-full-prompt-comparison.md) | [Worked examples](../../scripts/plot_benchmark_examples.py) |
| [NER across languages and lengths](ner-language-length.png) | [Which groups support changing the backend?](../eval/ner-language-length.md#paired-glinerel-and-heuristic-results) | [Multilingual NER](../../scripts/plot_ner_language_length.py) |

Use `just dataset-figures` to rebuild previews or `just dataset-figures --update` to refresh the published images and derived CSVs. Pull requests that change figure inputs or generators run the build and save the nine PNGs, nine SVGs and four CSVs in the **dataset-figures** Actions artifact for 30 days. [Regeneration and source refresh](../../scripts/README.md#reproduce-the-dataset-figures)

A manual **Dataset figures** run on `main` also builds a tenth figure from the live, authenticated S3 inventory. Its **s3-dataset-cache-figures** artifact contains the PNG, SVG and aggregate verification results. The live figure describes the cache at the time of that run. [Run the workflow and download its artifacts](https://github.com/arclabs561/anno/actions/workflows/dataset-figures.yml) · [S3 verification scope](../S3_CACHE_CI.md#verify-the-live-dataset-cache)
