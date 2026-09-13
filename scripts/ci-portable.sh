#!/usr/bin/env bash
# Provider-neutral CI commands and receipts.
#
# Providers own checkout, credentials, and artifact upload. This script owns the
# command boundary and leaves enough evidence under ANNO_ARTIFACT_DIR to compare
# local and hosted runs without interpreting provider-specific logs.

set -euo pipefail

usage() {
    printf '%s\n' 'Usage: scripts/ci-portable.sh {check|blocking|cache-stats}' >&2
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly script_dir
repo_root="$(cd -- "${script_dir}/.." && pwd)"
readonly repo_root

if [[ $# -ne 1 ]]; then
    usage
    exit 2
fi

readonly lane="$1"
case "${lane}" in
    check|blocking|cache-stats) ;;
    *)
        usage
        exit 2
        ;;
esac

artifact_dir="${ANNO_ARTIFACT_DIR:-${repo_root}/reports/ci-${lane}}"
if [[ "${artifact_dir}" != /* ]]; then
    artifact_dir="${repo_root}/${artifact_dir}"
fi
readonly artifact_dir
readonly log_dir="${artifact_dir}/logs"

mkdir -p "${log_dir}"

write_metadata() {
    {
        printf 'lane=%s\n' "${lane}"
        printf 'source_revision=%s\n' "$(git -C "${repo_root}" rev-parse HEAD)"
        printf 'source_dirty=%s\n' "$(git -C "${repo_root}" status --porcelain | wc -l | tr -d ' ')"
        printf 'cargo=%s\n' "$(cargo --version)"
        printf 'rustc=%s\n' "$(rustc --version)"
        printf 'rustc_wrapper_env=%s\n' "${RUSTC_WRAPPER:-unset (Cargo configuration may supply a wrapper)}"
        printf 'cargo_incremental=%s\n' "${CARGO_INCREMENTAL:-unset}"
        printf 'sccache_dir=%s\n' "${SCCACHE_DIR:-unset}"
    } >"${artifact_dir}/metadata.txt"
    cp "${repo_root}/Cargo.lock" "${artifact_dir}/Cargo.lock"
}

write_cache_stats() {
    local cache_command=()

    if [[ -n "${RUSTC_WRAPPER:-}" ]]; then
        cache_command=("${RUSTC_WRAPPER}")
    elif command -v sccache >/dev/null 2>&1; then
        printf '%s\n' 'RUSTC_WRAPPER is unset; plain sccache statistics may differ from a Cargo-configured wrapper.' >&2
        cache_command=(sccache)
    else
        printf '%s\n' 'sccache unavailable: configure RUSTC_WRAPPER or install sccache.' \
            >"${artifact_dir}/sccache-stats.txt"
        return
    fi

    if ! "${cache_command[@]}" --show-adv-stats >"${artifact_dir}/sccache-stats.txt" 2>&1; then
        printf '%s\n' 'Unable to read sccache statistics for the configured wrapper.' \
            >>"${artifact_dir}/sccache-stats.txt"
    fi
}

finish() {
    local status=$?
    write_cache_stats
    printf 'exit_status=%s\n' "${status}" >>"${artifact_dir}/metadata.txt"
    exit "${status}"
}

run_step() {
    local name="$1"
    shift
    printf '==> %s\n' "${name}"
    "$@" 2>&1 | tee "${log_dir}/${name}.log"
}

trap finish EXIT
cd "${repo_root}"
write_metadata

case "${lane}" in
    check)
        # The existing GitHub "Check" assertion, exposed through the same
        # provider-neutral receipt contract as the CodeBuild pilot.
        run_step cargo-check cargo check --workspace --all-targets
        ;;
    blocking)
        # Inventory of unique assertions: check-cached owns docs/fmt/clippy and
        # the eval+discourse workspace tests; the next commands retain
        # the all-target compilation and no-default-features library boundaries.
        run_step check-cached just check-cached
        run_step qa-panel-contract python3 scripts/qa/test_run_panel.py
        run_step portable-failure-contract python3 scripts/qa/test_ci_portable.py
        run_step cargo-check cargo check --workspace --all-targets
        run_step minimal-library cargo test --package anno --no-default-features --lib
        ;;
    cache-stats)
        # The EXIT trap writes the receipt. Keep this mode useful for a provider
        # post-build/finally phase without running another compiler command.
        ;;
esac
