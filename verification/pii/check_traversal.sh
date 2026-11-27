#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Verify the production span traversal with Verus and its matching rustc.
set -euo pipefail
cd "$(dirname "$0")/../.."

version=0.2026.11.22.1ce6a59
if ! rustup run 1.100.0 rustc --version >/dev/null 2>&1; then
    rustup toolchain install 1.100.0 --profile minimal
fi

verifier=${VERUS_BIN:-$(command -v verus || true)}
if [[ -z "$verifier" ]]; then
    if [[ "$(uname -sm)" != "Linux x86_64" ]]; then
        echo "Install Verus $version for this platform and set VERUS_BIN to its executable." >&2
        exit 1
    fi
    scratch=$(mktemp -d)
    trap 'rm -rf "$scratch"' EXIT
    archive="verus-$version-x86-linux.zip"
    curl --fail --silent --show-error --location --retry 3 \
        "https://github.com/verus-lang/verus/releases/download/release/$version/$archive" \
        --output "$scratch/$archive"
    (cd "$scratch" && echo "c955d5958877f4f907cce80aa996d4239a481984829d82e325938c7bf242b0db  $archive" | sha256sum --check)
    unzip -q "$scratch/$archive" -d "$scratch"
    verifier="$scratch/verus-x86-linux/verus"
fi

if [[ "$("$verifier" --version | awk '/^  Version:/ {print $2}')" != "$version" ]]; then
    echo "This proof uses Verus $version." >&2
    exit 1
fi
"$verifier" --crate-type lib verification/pii/traversal.rs
