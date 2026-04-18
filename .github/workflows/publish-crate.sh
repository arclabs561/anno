#!/usr/bin/env bash
# Publish one workspace crate to crates.io, tolerating "already uploaded" as success.
#
# Used by .github/workflows/publish.yml. The workflow runs each crate in dep
# order; crates published at the current version in a previous attempt are
# treated as no-ops so the workflow can be re-run without manual cleanup.
#
# After any successful (or skipped-as-duplicate) publish, sleeps 30s so the
# crates.io index propagates before the next dependent crate is published.
set -euo pipefail

crate="${1:?usage: publish-crate.sh CRATE}"
log=$(mktemp)
trap 'rm -f "$log"' EXIT

if cargo publish -p "$crate" --allow-dirty 2>&1 | tee "$log"; then
    echo "::notice::published $crate"
    sleep 30
    exit 0
fi

# cargo publish failed. Treat "already uploaded at the current version" as
# success so the workflow is idempotent across retries.
if grep -qE "crate version .* is already uploaded" "$log" \
    || grep -qE "already exists on crates\.io" "$log"; then
    echo "::notice::$crate is already published at the current version, skipping"
    sleep 5
    exit 0
fi

echo "::error::publish failed for $crate"
exit 1
