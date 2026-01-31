# `scripts/` (development + ops helpers)

This directory contains **repo-local helper scripts** used by `justfile`, CI, and ad-hoc debugging.
Treat these as **tools**, not as a stable public interface.

## Prefer `just`

Most common entrypoints are wrapped in `justfile` so you don’t need to remember flags/paths:

- **Health**: `just check`, `just ci`, `just ci-eval`
- **Eval**: `just matrix`, `just eval-quick`, `just eval-wide`
- **Docs**: `just docs-audit`, `just serve-readme`, `just e2e-readme-test`
- **Static analysis**: `just static-analysis`, `just validate-static-analysis-setup`

## Python scripts

These scripts are not part of a Python package. Many are intended to be run with `uv`:

- Example: `uv run scripts/eval_comprehensive.py --max-examples 20`

When a script needs extra dependencies, prefer using `uv`’s inline metadata format (see
the generated `convert_pytorch_to_safetensors.py` written by `sync_datasets_s3.sh`).

## Shell scripts

Most bash scripts are standalone entrypoints and should use strict mode:

```bash
set -euo pipefail
```

If a script is meant to be sourced (rare here), it should avoid setting shell options.

## Organization (informal)

- **Eval + regression hunting**: `eval-*.sh`, `check-*-patterns.sh`, `summarize-failures.sh`
- **Dataset registry + generation**: `generate_*`, `*_registry*`, `verify_dataset_urls.py`
- **S3 cache + artifacts**: `sync_datasets_s3.sh`, `prepare_datasets_s3.py`, `upload_*_s3.sh`
- **CI helpers**: `benchmark-static-analysis.sh`, `validate-static-analysis-setup.sh`
- **Spot orchestration**: `spot/` (AWS spot eval harness tooling)

If you add a new script, prefer:

- one clear entrypoint (`main()` for python; `case "$1"` for bash)
- being runnable from the repo root
- printing a short `--help`/usage string
