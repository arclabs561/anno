# `scripts/` (development + ops helpers)

This directory contains **repo-local helper scripts** used by `justfile`, CI, and ad-hoc debugging.
Treat these as **tools**, not as a stable public interface.

## Prefer `just`

Most common entrypoints are wrapped in `justfile` so you don’t need to remember flags/paths:

- **Health**: `just check`, `just ci`, `just ci-eval`
- **Eval**: `just matrix`, `just eval-quick`, `just eval-wide`
- **Docs**: `just docs-audit`, `just readme-preview`
- **Static analysis**: `just static-analysis`, `just validate-setup`

## Python scripts

These scripts are not part of a Python package. Many are intended to be run with `uv`:

- Example: `uv run scripts/eval_comprehensive.py --max-examples 20`

When a script needs extra dependencies, prefer using `uv`’s inline metadata format (see
the generated `convert_pytorch_to_safetensors.py` written by `sync_datasets_s3.sh`).

### Reproduce the dataset figures

The guide figures are tracked in `docs/figures/`. Each has a PNG. Native Anno charts also have an SVG for vector export and a standalone HTML report for reading the counts at different screen widths. `docs/eval/` holds the CSV and JSON measurements. The figure index links each plot to its investigation and input or generator. [Figure index](../docs/figures/README.md) · [Coverage and exclusions](../docs/eval/dataset-coverage.md)

Run these commands from the repository root with `just`, `uv` and Cargo installed. On first use `uv` installs the pinned plotting dependencies. Cargo builds the checkout's CLI with the minimal HTML extractor. Rendering uses the checked-in measurements and runs offline once dependencies are cached.

```sh
just dataset-figures           # Preview images, native chart inputs and CSVs in .generated/dataset-figures/
just dataset-figures --update  # Refresh the published images and derived summaries
just dataset-figures --check   # Check the published images and counts
```

To call the entry point directly, run `uv run --python 3.12 --script scripts/refresh_dataset_figures.py --check`. Use `--output DIR` for another preview directory or `--anno PATH` to render with an existing Anno binary. [Build entry point and dependencies](refresh_dataset_figures.py)

The build discovers study inputs under `docs/eval/`. Name each file `<study>-<view>.json`, using the matching native view in the table. Add its dataset IDs and guide to `docs/eval/figure-studies.csv` so readers can find it with `anno viz list --dataset NAME`. The generator writes the same native inputs and dataset evidence to `docs/eval/figure-catalog.json` and the CI download. A figure refresh can update this catalog independently of a CLI release. It calls the public CLI and rasterizes the SVG with resvg and the bundled DejaVu Sans fonts. Embed the SVG with the same file stem. [Chart formats and CLI examples](../docs/QUICKSTART.md#visualize-evaluation-results)

| View | Portable input | What the report shows |
|---|---|---|
| `paired` | `anno.paired/v1`, or two `anno eval --json` streams with `--against` | Exact matches gained, lost and retained on the same inputs, total suggestions, and measured calls or latency. |
| `tradeoff` | `anno.tradeoff/v1` | Correct predictions at different budgets. Use `--x-axis suggestions` for total queue size or the default unmatched count for error analysis. |
| `errors` | `anno.errors/v1` | Exact matches, wrong labels, boundary mismatches, missed gold and extra predictions from labeled evaluations. |
| `overlap` | `anno.overlap/v1` | How many supplied occurrences and source characters share annotation spans. |

```sh
anno viz paired docs/eval/wikigold-routing-paired.json --output wikigold.html
```

For new data or model runs, regenerate the relevant measurements, run `--update`, review the figure and its caption, then commit the inputs, code and changed outputs together. CI compares PNG pixels, native SVG and HTML reports, the offline dataset explorer, derived CSVs and the numerical sections of the coverage guide. [CI workflow](../.github/workflows/dataset-figures.yml)

Each build saves PNGs, SVGs, HTML reports, portable chart JSON and derived CSVs in the `dataset-figures` Actions artifact for 30 days. Start a manual run on the default branch to also verify the live cache in the trusted S3 job. The `s3-dataset-cache-figures` artifact contains the live inventory figure and WikiGold verification. The `cache-source-evidence` artifact contains the broader publisher-file matches, with their figure and counts. That comparison also refreshes after changes to the measured-file ledgers on `main`. [Actions runs and downloads](https://github.com/arclabs561/anno/actions/workflows/dataset-figures.yml)

To publish a new dated S3 figure in the guide, download `cache-source-evidence` from the trusted run, copy `cache-source-coverage.json` to `docs/eval/s3-source-coverage.json` and `publisher-file-matches.csv` to `docs/eval/s3-publisher-file-matches.csv`, then run `just dataset-figures --update`. The generator updates the run link and counts in the coverage guide. The checked-in receipt and image remain available after the corresponding CI artifact expires.

```sh
gh workflow run dataset-figures.yml --ref main
gh run download RUN_ID --name dataset-figures --dir .generated/ci-figures
gh run download RUN_ID --name s3-dataset-cache-figures --dir .generated/ci-s3
gh run download RUN_ID --name cache-source-evidence --dir .generated/ci-source-matches
```

The source refresh examples below use these paths.

```sh
out=.generated/dataset-figures
ledgers=docs/eval
mkdir -p "$out"
```

**Recompute from publisher annotations.** WNUT reads the three gold splits and all seven archived submissions. GENIA verifies its three split files against their recorded SHA-256 digests. CrossRE verifies all 18 domain/split files and compares each held-out split with the same domain’s training labels.

```sh
python3 scripts/plot_wnut_novelty.py "$out"
cmp "$out/wnut-submission-results.csv" "$ledgers/wnut-submission-results.csv"
cargo build --locked -p anno-eval --example genia_export
python3 scripts/plot_genia_nesting.py "$out" --native-export "${CARGO_TARGET_DIR:-target}/debug/examples/genia_export"
cmp "$out/genia-native-exports.json" "$ledgers/genia-native-exports.json"
cmp "$out/genia-nesting-counts.json" "$ledgers/genia-nesting-counts.json"
python3 scripts/genia_grounded_overlap.py "$out/genia-span-overlap.json"
cmp "$out/genia-span-overlap.json" "$ledgers/genia-span-overlap.json"
python3 scripts/survey_crossre_prompts.py "$out/crossre-prompt-scope.csv"
cmp "$out/crossre-prompt-scope.csv" "$ledgers/crossre-prompt-scope.csv"
```

For offline WNUT replay, add `--data-dir PATH_TO_CHECKOUT` at upstream commit `e52a2d2a71ac1a051ca2d532eef28653c5c02603`. For GENIA or CrossRE, add `--cache-dir PATH_TO_POPULATED_REPO_FILE_CACHE`. The GENIA overlap replay uses `anno` on PATH. Pass `--anno PATH` for a particular binary. The nesting study also accepts `--input PATH_TO_GENIA_TEST_IOB2` for a test-only calculation. [WNUT source](https://github.com/leondz/emerging_entities_17/tree/e52a2d2a71ac1a051ca2d532eef28653c5c02603) · [GENIA and CrossRE file manifest](../docs/eval/dataset-repo-files.csv)

[Native WNUT-17 replay](../docs/DOMAIN_QA.md#compare-native-anno-on-unfamiliar-names) builds the selected Anno revisions, runs the same 1,287 test sentences through both binaries and writes the aggregate used by the native comparison figure. [Six-domain CrossRE replay](../docs/eval/crossre-full-prompt-comparison.md) records the model, corpus files and two prompt conditions.

**Refresh source measurements.** The direct survey also generates the task-tag, PII fixture, WikiGold/WNUT-16, token-density and MIT query figures. Install the system libarchive and the Python `libarchive-c` package to read the FINER RAR release. The Hugging Face scan and the Quoref and CHIA direct readers use `pyarrow`. Replaying CHIA’s seeded split also uses NumPy. The GitHub source and archived CI surveys use an authenticated `gh` CLI. Run the following with network access and the same `out` and `ledgers` variables as above.

```sh
python3 scripts/plot_datasets.py "$out"
python3 scripts/survey_repo_files.py --manifest "$ledgers/dataset-repo-files.csv" \
  --output "$out/dataset-repo-files.csv"
python3 scripts/survey_hf_splits.py "$out/anno-all.csv" "$out/dataset-hf-splits.csv"
python3 scripts/survey_github_sources.py --inventory "$out/anno-all.csv" \
  --output "$out/dataset-github-sources.csv"
python3 scripts/survey_registry_pointers.py "$out/anno-all.csv" "$out/dataset-registry-pointers.csv"
python3 scripts/survey_matrix_evidence.py --limit 100 --output "$out/dataset-matrix-evidence.csv"
```

To rebuild coverage for this refresh, use `$out/anno-all.csv` as the inventory and the new ledgers in `$out`. The direct reader accepts `--source-cache PATH` to reuse downloaded bytes. For the 15 GB medieval charter release it reads the ZIP directory and all four annotated text members in about 7.5 MB of verified ranges. For replay, `survey_repo_files.py` checks files against the supplied manifest. Add `--pins "$ledgers/dataset-github-sources.csv"` to the GitHub survey to select the recorded repository revisions. The following commands verify previously measured Hugging Face splits and rerun the recorded CI prechecks, subject to log availability.

```sh
python3 scripts/survey_hf_splits.py --replay "$ledgers/dataset-hf-splits.csv" "$out/hf-replay.csv"
awk -F, 'NR > 1 && !seen[$1]++ {print "--run", $1}' "$ledgers/dataset-matrix-evidence.csv" | \
  xargs python3 scripts/survey_matrix_evidence.py --output "$out/ci-replay.csv"
```

### Official multilingual coreference scores

For one aligned CorefUD key/response dataset, run the [CRAC 2026 scorer](https://ufal.mff.cuni.cz/corefud/shared-task/2026#unconstr-evaluation) on the original CoNLL-U files. This preserves mention heads, discontinuous spans and empty-node identities. The local clone supplies the [pinned scorer and its fixtures](https://github.com/ufal/corefud-scorer/tree/4fd7b0e0c661aeeff88bc60c19ef507b84d1b590).

```sh
git clone https://github.com/ufal/corefud-scorer.git /tmp/corefud-scorer
uv run scripts/eval_corefud.py --scorer /tmp/corefud-scorer --check
uv run scripts/eval_corefud.py --scorer /tmp/corefud-scorer key.conllu response.conllu
```

The default uses head matching, excludes singletons, and matches zeros by dependency. The scorer prints MUC, B³, CEAF-e, zero-anaphor P/R/F1 and the CoNLL score in percent. Use `--mode partial`, `--mode exact` or `--mode singletons` for the three secondary settings. To compare a multilingual system to the primary leaderboard, score each evaluation dataset separately and take the unweighted mean of the dataset CoNLL scores.

The native `anno dataset eval --task coref` command matches exact character ranges, retains singleton chains supplied by the loader or resolver, and reports the arithmetic mean of document scores. With one gold cluster `{A,B,C}` and response clusters `{A,B}, {C}`, this convention yields 63.81%. The official primary mode yields 69.40% on the same one-token mentions after singleton exclusion. The difference comes from B³ and CEAF-e. Native library callers can select `CorefEvaluation::compute_without_singletons` to apply that filter with character-span matching.

## Shell scripts

Most bash scripts are standalone entrypoints and should use strict mode:

```bash
set -euo pipefail
```

If a script is meant to be sourced (rare here), it should avoid setting shell options.

## Organization (informal)

- **Eval + regression hunting**: `eval-*.sh`, `check-*-patterns.sh`
- **Dataset registry + generation**: `generate_*`, `*_registry*`, `verify_dataset_urls.py`
- **S3 cache + artifacts**: `sync_datasets_s3.sh`, `prepare_datasets_s3.py`, `upload_*_s3.sh`
- **CI helpers**: `benchmark-static-analysis.sh`, `validate-static-analysis-setup.sh`
- **Spot orchestration**: `spot/` (AWS spot eval harness tooling)

If you add a new script, prefer:

- one clear entrypoint (`main()` for python; `case "$1"` for bash)
- being runnable from the repo root
- printing a short `--help`/usage string
