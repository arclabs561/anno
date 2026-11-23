#!/usr/bin/env bash
# Run the same reviewed-assignment checks as the required minimal CI job.
set -euo pipefail
cd "$(dirname "$0")/../.."

z3 -version
cargo test --locked -p anno --no-default-features \
  --test reviewed_kb_assignment --test reviewed_kb_sequences --test reviewed_kb_seeds
python3 scripts/verification/check_review_boundary.py
python3 verification/reviewed_kb/check_model.py
python3 scripts/verification/check_review_invalidation.py
