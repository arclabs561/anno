#!/usr/bin/env bash
# Fetch mutable source captures for local extraction checks.
set -euo pipefail

dir=testdata/real_world
mkdir -p "$dir"
stage=$(mktemp -d "$dir/.fetch.XXXXXX")
trap 'rm -rf -- "$stage"' EXIT

urls=(
    "https://climate.ec.europa.eu/eu-action/european-climate-law_en"
    "https://www.consilium.europa.eu/en/policies/climate-change/"
    "https://openai.com/policies/usage-policies/"
    "https://news.un.org/en/story/2024/09/1154541"
    "https://arxiv.org/abs/2309.14084"
    "https://arxiv.org/pdf/2309.14084"
    "https://arxiv.org/pdf/2101.00884"
    "https://arxiv.org/abs/2101.00884"
)

failed=0
for url in "${urls[@]}"; do
    name=${url#https://}
    name=${name//\//_}
    name=${name//./_}
    case "$url" in
        https://arxiv.org/pdf/*|*.pdf) ext=pdf ;;
        *) ext=html ;;
    esac
    destination="$dir/$name.$ext"
    capture="$stage/$name.$ext"
    echo "Fetching $url"
    if ! status=$(curl --fail --location --silent --show-error --output "$capture" \
        --write-out '%{http_code}' "$url"); then
        echo "Download failed for $url" >&2
        failed=1
        continue
    fi
    if [[ "$status" != 2[0-9][0-9] ]]; then
        echo "Unexpected HTTP status $status for $url" >&2
        failed=1
        continue
    fi
    if [[ ! -s "$capture" ]]; then
        echo "Empty response for $url" >&2
        failed=1
        continue
    fi
    bytes=$(wc -c < "$capture")
    if (( bytes < 100 )); then
        echo "Short response ($bytes bytes) for $url" >&2
        failed=1
        continue
    fi
    if [[ "$ext" == pdf ]] && ! head -c 5 "$capture" | grep -Fqx '%PDF-'; then
        echo "Missing PDF header for $url" >&2
        failed=1
        continue
    fi
    mv -- "$capture" "$destination"
    printf 'Saved %s (%d bytes)\n' "$destination" "$bytes"
done
exit "$failed"
