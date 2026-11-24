#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
set -euo pipefail
cd "$(dirname "$0")/../.."

cargo test --locked -p anno --no-default-features --test index_reconciliation -- --nocapture
cargo test --locked -p anno --no-default-features --example index_link_policy
cargo run --quiet --locked -p anno --no-default-features --example index_component_repair
cargo test --locked -p anno --no-default-features --test citation_successors
cargo run --quiet --locked -p anno --no-default-features --example rag_citation
cargo run --quiet --locked -p anno --no-default-features --example index_publish_snapshots \
  | python3 verification/index_consumers/check_recovery.py
python3 verification/temporal_identity/check_replay.py
python3 verification/index_consumers/check_projection.py
python3 verification/index_consumers/congress/test_committee_pilot.py
python3 verification/index_consumers/congress/test_cross_doc_partition.py
python3 verification/index_consumers/congress/check_catalog_history.py
