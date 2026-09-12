#!/usr/bin/env bash
# Validate cargo publish --dry-run for the crates.io workspace packages.

set -euo pipefail

work_dir=$(mktemp -d "${TMPDIR:-/tmp}/anno-publish-validation.XXXXXX")
trap 'rm -rf "$work_dir"' EXIT

echo '=== Publish Validation ==='
echo
echo '## Workspace Structure'
cargo metadata --format-version 1 | jq -r '.workspace_members[]' | sort
echo

log_file="$work_dir/publish.log"
echo '## Publish readiness'
if cargo publish --dry-run --workspace --exclude anno-py >"$log_file" 2>&1; then
    echo '[ok] cargo publish --dry-run passed for anno, anno-eval, and anno-cli'
    cat "$log_file"
else
    echo '[error] cargo publish --dry-run failed'
    cat "$log_file"
    exit 1
fi

echo
echo '## Crates.io status'
for crate in anno anno-eval anno-cli; do
    if version=$(curl --fail --silent --show-error --user-agent 'anno-publish-validation/1.0' \
        "https://crates.io/api/v1/crates/$crate" | jq -r '.crate.max_version // "not published"'); then
        echo "[info] $crate latest: $version"
    else
        echo "[warning] could not query crates.io for $crate; publish validation still passed"
    fi
done
