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

Run from the repository root with Python 3 and Matplotlib 3.6 or later. The coverage, density and benchmark charts read checked-in ledgers and examples offline. Each registry ID counts once in the combined coverage chart even when several sources cover it. [Coverage findings and source ledgers](../docs/eval/dataset-coverage.md)

```sh
out=.generated/dataset-figures
ledgers=docs/eval
mkdir -p "$out"
python3 scripts/plot_measurement_coverage.py "$ledgers/dataset-figure-inventory.csv" "$out" \
  --direct-files "$ledgers/dataset-direct-files.csv" \
  --repo-files "$ledgers/dataset-repo-files.csv" \
  --hf-splits "$ledgers/dataset-hf-splits.csv" \
  --github-sources "$ledgers/dataset-github-sources.csv" \
  --ci-prechecks "$ledgers/dataset-matrix-evidence.csv"
python3 scripts/plot_density_ranges.py "$ledgers/dataset-figure-inventory.csv" "$out" \
  --direct-files "$ledgers/dataset-direct-files.csv" \
  --repo-files "$ledgers/dataset-repo-files.csv" \
  --hf-splits "$ledgers/dataset-hf-splits.csv" \
  --github-sources "$ledgers/dataset-github-sources.csv"
python3 scripts/plot_benchmark_examples.py "$out"
```

The coverage generator writes `dataset-coverage.csv`, the combined `dataset-coverage` figure and the direct-reader `anno-selection` figure. The density generator writes `dataset-density-ranges.csv` and the `dataset-density` figure. The benchmark generator renders the JEREX benchmark chooser and the CrossRE prompt comparison. All three write PNG and SVG.

**Annotation-level figures.** These commands read the pinned WNUT-17 and GENIA inputs. WNUT compares the development and registry-designated test mentions with training. GENIA verifies all three split files against their recorded SHA-256 digests.

```sh
python3 scripts/plot_wnut_novelty.py "$out"
python3 scripts/plot_genia_nesting.py "$out"
```

For offline WNUT replay, add `--data-dir PATH_TO_CHECKOUT` at upstream commit `e52a2d2a71ac1a051ca2d532eef28653c5c02603`. For GENIA, add `--cache-dir PATH_TO_POPULATED_REPO_FILE_CACHE` for all three splits or `--input PATH_TO_GENIA_TEST_IOB2` for the original test-only calculation. [WNUT source](https://github.com/leondz/emerging_entities_17/tree/e52a2d2a71ac1a051ca2d532eef28653c5c02603) · [GENIA file manifest](../docs/eval/dataset-repo-files.csv)

**Refresh source measurements.** The direct survey also generates the task-tag, PII fixture, WikiGold/WNUT-16, token-density and MIT query figures. Install the system libarchive and the Python `libarchive-c` package to read the FINER RAR release. The Hugging Face scan requires `pyarrow`. The GitHub source and archived CI surveys use an authenticated `gh` CLI. Run the following with network access and the same `out` and `ledgers` variables as above.

```sh
python3 scripts/plot_datasets.py "$out"
python3 scripts/survey_repo_files.py --manifest "$ledgers/dataset-repo-files.csv" \
  --output "$out/dataset-repo-files.csv"
python3 scripts/survey_hf_splits.py "$out/anno-all.csv" "$out/dataset-hf-splits.csv"
python3 scripts/survey_github_sources.py --inventory "$out/anno-all.csv" \
  --output "$out/dataset-github-sources.csv"
python3 scripts/survey_matrix_evidence.py --limit 100 --output "$out/dataset-matrix-evidence.csv"
```

To rebuild coverage for this refresh, use `$out/anno-all.csv` as the inventory and the five new ledgers in `$out`. For replay, `survey_repo_files.py` already checks files against the supplied manifest. Add `--pins "$ledgers/dataset-github-sources.csv"` to the GitHub survey to select the recorded repository revisions. The following commands verify previously measured Hugging Face splits and rerun the recorded CI prechecks, subject to log availability.

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
