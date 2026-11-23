#!/usr/bin/env bash
set -euo pipefail

dataset_dir="${ANNO_CACHE_DIR:?}/datasets"
input="$dataset_dir/WikiGold.cache"
digest=c797a64d0cf73ed058363f77671486bcfd413e70fda1726ad4a6ba624455225f

mkdir -p "$dataset_dir"
if [[ -f "$input" ]]; then
    if echo "$digest  $input" | sha256sum --check --status; then exit 0; fi
    # PR sampling continues if this step fails. Keep rejected bytes out of its cache.
    rm -- "$input"
fi

temporary=$(mktemp "$dataset_dir/.wikigold.XXXXXX")
trap 'rm -f -- "$temporary"' EXIT
curl --fail --location --retry 2 --max-time 30 --max-filesize 1000000 \
    https://raw.githubusercontent.com/juand-r/entity-recognition-datasets/8522f4b39cf9ed323b77f2d8a5005628f8861908/data/wikigold/CONLL-format/data/wikigold.conll.txt \
    -o "$temporary"
echo "$digest  $temporary" | sha256sum --check
mv -- "$temporary" "$input"
