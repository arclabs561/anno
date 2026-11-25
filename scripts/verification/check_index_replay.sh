#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
set -euo pipefail
cd "$(dirname "$0")/../.."

cargo test --locked -p anno --no-default-features --test index_reconciliation --test membership_publication -- --nocapture
cargo test --locked -p anno-cli --no-default-features --features "eval extractor-readability" --test membership_queries
cargo test --locked -p anno --no-default-features --example index_link_policy
cargo run --quiet --locked -p anno --no-default-features --example index_component_repair
cargo test --locked -p anno --no-default-features --test citation_successors
cargo run --quiet --locked -p anno --no-default-features --example rag_citation
cargo run --quiet --locked -p anno --no-default-features --example index_publish_snapshots \
  | python3 verification/index_consumers/check_recovery.py
python3 verification/temporal_identity/check_replay.py
python3 verification/index_consumers/check_projection.py
python3 verification/index_consumers/reader_cases/lineage/check.py
python3 verification/index_consumers/reader_cases/publisher/check.py
python3 -m unittest verification.index_consumers.reader_cases.test_ajmc
python3 -m unittest verification.index_consumers.reader_cases.candidates.test_policy \
  verification.index_consumers.reader_cases.candidates.test_read \
  verification.index_consumers.reader_cases.open_world.test_replay
python3 verification/index_consumers/reader_cases/confirmation/check.py
python3 verification/index_consumers/reader_cases/lineage/ngram_check.py
python3 -m unittest discover -s verification/index_consumers/reader_cases/catalog_pilots -p 'test_*.py'
python3 verification/index_consumers/reader_cases/catalog_pilots/native.py --verify
python3 verification/index_consumers/congress/test_committee_pilot.py
python3 verification/index_consumers/congress/test_cross_doc_partition.py
python3 verification/index_consumers/congress/check_catalog_history.py
python3 verification/index_consumers/check_editions.py --review-example
python3 verification/index_consumers/credits/check_cases.py
python3 verification/index_consumers/credits/check_search.py
python3 verification/index_consumers/reader_cases/check.py --bounded
python3 verification/index_consumers/reader_cases/formal/check_support.py
python3 -m unittest discover -s scripts/qa -p 'test_replay_reader_cases.py'
