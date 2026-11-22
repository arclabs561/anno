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

echo "Cargo home: ${CARGO_HOME:-$HOME/.cargo}"
echo "Staging target: $work_dir/target"
echo "Checkout commit: $(git rev-parse --verify HEAD 2>/dev/null || echo unavailable)"
if [[ -n "${PR_HEAD_SHA:-}" ]]; then
    echo "PR head commit: $PR_HEAD_SHA"
fi
log_file="$work_dir/publish.log"
echo '## Publish readiness'
# Cargo stages workspace crates in a local registry under the target directory.
# A fresh registry path makes Cargo unpack the current source at the same version.
if cargo publish --dry-run --workspace --exclude anno-py --target-dir "$work_dir/target" >"$log_file" 2>&1; then
    echo '[ok] cargo publish --dry-run passed for anno, anno-eval, and anno-cli'
    status=0
else
    status=$?
    echo '[error] cargo publish --dry-run failed'
fi
cat "$log_file"

# These are the candidate tarballs served by Cargo's temporary workspace
# registry, including those already staged when a later crate fails to build.
for archive in "$work_dir"/target/package/tmp-registry/*.crate; do
    [[ -f "$archive" ]] || continue
    sha256sum "$archive"
done
if ((status != 0)); then
    exit "$status"
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
