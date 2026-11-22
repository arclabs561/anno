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

### Official multilingual coreference scores

For one aligned CorefUD key/response dataset, run the [CRAC 2026 scorer](https://ufal.mff.cuni.cz/corefud/shared-task/2026#unconstr-evaluation) on the original CoNLL-U files. This preserves mention heads, discontinuous spans and empty-node identities. The local clone supplies the [pinned scorer and its fixtures](https://github.com/ufal/corefud-scorer/tree/4fd7b0e0c661aeeff88bc60c19ef507b84d1b590).

```sh
git clone https://github.com/ufal/corefud-scorer.git /tmp/corefud-scorer
uv run scripts/eval_corefud.py --scorer /tmp/corefud-scorer --check
uv run scripts/eval_corefud.py --scorer /tmp/corefud-scorer key.conllu response.conllu
```

The default uses head matching, excludes singletons, and matches zeros by dependency. The scorer prints MUC, B³, CEAF-e, zero-anaphor P/R/F1 and the CoNLL score in percent. Use `--mode partial`, `--mode exact` or `--mode singletons` for the three secondary settings. To compare a multilingual system to the primary leaderboard, score each evaluation dataset separately and take the unweighted mean of the dataset CoNLL scores. Anno's native span-based evaluator uses a different matching protocol.

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
