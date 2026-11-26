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

Every published dataset figure is in **`docs/figures/`** as a regular Git-tracked PNG. The figure index links each image to its guide and Python generator. The checked-in CSV and JSON measurements live in `docs/eval/`. [Figure index](../docs/figures/README.md) · [Coverage and exclusions](../docs/eval/dataset-coverage.md)

Run these commands from the repository root with `just` and `uv` installed. On first use, `uv` installs the plotting dependencies. The render reads only files in the checkout.

```sh
just dataset-figures           # Preview all PNGs, SVGs and four derived CSVs in .generated/dataset-figures/
just dataset-figures --update  # Refresh the published PNGs and derived CSVs after a change
just dataset-figures --check   # Verify the published pixels and CSV contents
```

With `uv` alone, run `uv run --python 3.12 --script scripts/refresh_dataset_figures.py --check`. Pass `--output PATH` for a separate preview directory. Each figure generator also runs independently with Python and Matplotlib. [Single entry point and renderer dependencies](refresh_dataset_figures.py)

**When the figures change.** For new source data or model results, first refresh the relevant pinned measurements using the source commands below. For a change to the chart design, edit its generator. Run `--update`, review the figures and captions, then commit the changed inputs, code and outputs together. CI renders the figures on Linux for pull requests that touch figure inputs, generators or published PNGs. It compares image dimensions and pixels, checks the four derived CSVs, and names any stale result. Maintainers can also run the check on demand. [CI workflow](../.github/workflows/dataset-figures.yml)

Each figure build saves its PNGs, SVGs and four derived CSVs as the `dataset-figures` Actions artifact for 30 days. Start a manual run on the default branch to also verify the live cache in the trusted S3 job. Its separate `s3-dataset-cache-figures` artifact contains the live inventory figure and its verification results. [Actions runs and downloads](https://github.com/arclabs561/anno/actions/workflows/dataset-figures.yml)

```sh
gh workflow run dataset-figures.yml --ref main
gh run download RUN_ID --name dataset-figures --dir .generated/ci-figures
gh run download RUN_ID --name s3-dataset-cache-figures --dir .generated/ci-s3
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
python3 scripts/plot_genia_nesting.py "$out"
cmp "$out/genia-nesting-counts.json" "$ledgers/genia-nesting-counts.json"
python3 scripts/survey_crossre_prompts.py "$out/crossre-prompt-scope.csv"
cmp "$out/crossre-prompt-scope.csv" "$ledgers/crossre-prompt-scope.csv"
```

For offline WNUT replay, add `--data-dir PATH_TO_CHECKOUT` at upstream commit `e52a2d2a71ac1a051ca2d532eef28653c5c02603`. For GENIA or CrossRE, add `--cache-dir PATH_TO_POPULATED_REPO_FILE_CACHE`. GENIA also accepts `--input PATH_TO_GENIA_TEST_IOB2` for a test-only calculation. [WNUT source](https://github.com/leondz/emerging_entities_17/tree/e52a2d2a71ac1a051ca2d532eef28653c5c02603) · [GENIA and CrossRE file manifest](../docs/eval/dataset-repo-files.csv)

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

To rebuild coverage for this refresh, use `$out/anno-all.csv` as the inventory and the five new ledgers in `$out`. The direct reader accepts `--source-cache PATH` to reuse downloaded bytes. For the 15 GB medieval charter release it reads the ZIP directory and all four annotated text members in about 7.5 MB of verified ranges. For replay, `survey_repo_files.py` checks files against the supplied manifest. Add `--pins "$ledgers/dataset-github-sources.csv"` to the GitHub survey to select the recorded repository revisions. The following commands verify previously measured Hugging Face splits and rerun the recorded CI prechecks, subject to log availability.

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
