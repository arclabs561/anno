#!/usr/bin/env bash
#
# static-analysis-common.sh
#
# Single source of truth for static-analysis tool invocations that depend on repo layout.
#
# Usage:
#   ./scripts/static-analysis-common.sh machete
#   ./scripts/static-analysis-common.sh machete-cmd   # prints command (shell-escaped)
#
set -euo pipefail

repo_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P
}

require_tool() {
  local tool="$1"
  local hint="$2"
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "error: missing tool: $tool" >&2
    echo "hint: $hint" >&2
    return 1
  fi
}

# cargo-machete starts at the directory containing the package manifest.
machete=(cargo machete crates/anno)

run_machete() {
  require_tool cargo "install Rust toolchain"
  require_tool cargo-machete "cargo install cargo-machete"
  local root
  root="$(repo_root)"
  (cd "$root" && "${machete[@]}")
}

case "${1:-}" in
  machete)
    run_machete
    ;;
  machete-cmd)
    echo "${machete[*]}"
    ;;
  *)
    echo "usage: $0 {machete|machete-cmd}" >&2
    exit 2
    ;;
esac

