#!/usr/bin/env bash
# Publish the checked-in inputs for the Spot worker's cargo build.
set -euo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/../s3_bucket.sh"

archive_worker_source() {
    # Restrict data-bearing extensions to the individual production assets the
    # crates embed. Research inputs, caches and test fixtures stay local.
    git -C "$script_dir/../.." archive --format=tar.gz -o "$1" "${2:-HEAD}" -- \
        Cargo.toml Cargo.lock README.md 'LICENSE*' ':(glob)crates/*/Cargo.toml' \
        ':(glob)crates/*/src/**/*.rs' ':(glob)crates/*/examples/**/*.rs' ':(glob)crates/*/benches/**/*.rs' \
        crates/anno/src/viz/report.css crates/anno/src/backends/{hmm_params,crf_weights}.json \
        crates/anno/src/backends/coref/{maverick_worker.py,maverick_models.json} \
        crates/anno-cli/src/cli/commands/joint.py \
        crates/anno-eval/src/eval/{litbank_attribution.json,shared_cache_exclusions.tsv} \
        scripts/spot/worker.sh scripts/s3_bucket.sh
}

upload_worker_source() {
    case "${1:-aws}" in
        help|-h|--help) echo "Usage: $0 [aws|trainctl]"; return ;;
        aws|trainctl) ;;
        *) echo "Usage: $0 [aws|trainctl]" >&2; return 2 ;;
    esac
    anno_spot_config "$script_dir/config.env"
    local bucket destination
    bucket=$(anno_bucket_name "${ANNO_SPOT_BUCKET:-}" "ANNO_SPOT_BUCKET or ANNO_S3_BUCKET")
    destination="s3://$bucket/src/anno-src.tar.gz"
    local scratch_dir
    scratch_dir=$(mktemp -d)
    ( trap 'rm -rf -- "$scratch_dir"' EXIT
      archive_worker_source "$scratch_dir/anno-src.tar.gz"
      if [[ ${1:-aws} == trainctl ]]; then
          "${TRAINCTL:-trainctl}" s3 upload "$scratch_dir/anno-src.tar.gz" "$destination"
      else
          aws s3 cp "$scratch_dir/anno-src.tar.gz" "$destination"
      fi )
    printf 'Worker source uploaded to %s\n' "$destination"
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then upload_worker_source "$@"; fi
