#!/usr/bin/env bash
# Run the production validator on two workspace crates at unchanged versions.
set -euo pipefail

validator="$(cd "$(dirname "$0")" && pwd)/validate-publish.sh"
fixture=$(mktemp -d)
trap 'rm -rf "$fixture"' EXIT
cd "$fixture"
export CARGO_TARGET_DIR="$fixture/reused-target"
mkdir -p api/src caller/src
printf '[workspace]\nmembers = ["api", "caller"]\nresolver = "2"\n' > Cargo.toml
for crate in api caller; do
    printf '[package]\nname = "anno-repro-%s"\nversion = "0.1.0"\nedition = "2021"\n' "$crate" > "$crate/Cargo.toml"
done
printf '[dependencies]\nanno-repro-api = { path = "../api", version = "=0.1.0" }\n' >> caller/Cargo.toml
printf 'pub fn original() {}\n' > api/src/lib.rs
printf 'pub fn call() { anno_repro_api::original(); }\n' > caller/src/lib.rs
cargo publish --dry-run --workspace

# Remove an API from the candidate dependency. A reused staging registry can
# satisfy the caller with the old package and report success.
printf 'pub fn replacement() {}\n' > api/src/lib.rs
if bash "$validator" > broken.log 2>&1; then
    cat broken.log
    echo 'Publish validation accepted a caller of a removed API.' >&2
    exit 1
fi
cat broken.log
grep -F 'cannot find function `original`' broken.log
for crate in api caller; do
    grep -Eq "^[0-9a-f]{64}  .*/anno-repro-${crate}-0[.]1[.]0[.]crate$" broken.log
done
