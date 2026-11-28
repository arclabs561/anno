#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Run with Kani 0.68.0 and Lean 4.19.0 on PATH.
set -euo pipefail
cd "$(dirname "$0")/../.."

kani verification/mention_spans/check.rs
kani verification/pii/check.rs
lean verification/mention_spans/MentionSpans.lean

bash verification/pii/check_traversal.sh
