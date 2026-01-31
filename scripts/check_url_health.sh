#!/usr/bin/env bash
# Check URL health for all datasets in the registry
# Reports broken URLs, timeouts, and suggests alternatives

set -euo pipefail

echo "=== Dataset URL Health Check ==="
echo ""

# Extract all URLs from dataset registry
REGISTRY_FILE="crates/anno-eval/src/eval/dataset_registry.rs"

if [[ ! -f "$REGISTRY_FILE" ]]; then
    echo "Error: Registry file not found: $REGISTRY_FILE"
    exit 1
fi

# Create temporary file for URLs
TMP_URLS=$(mktemp)
trap "rm -f $TMP_URLS" EXIT

# Extract URLs (look for url: "http" patterns)
rg -o 'url:\s*"([^"]+)"' "$REGISTRY_FILE" | \
    sed 's/url: "\(.*\)"/\1/' | \
    grep -E '^https?://' > "$TMP_URLS" || true

TOTAL_URLS=$(wc -l < "$TMP_URLS" | tr -d ' ')
echo "Found $TOTAL_URLS URLs to check"
echo ""

VALID=0
BROKEN=0
TIMEOUT=0
SSL_ERROR=0
AUTH_REQUIRED=0

while IFS= read -r url; do
    if [[ -z "$url" ]]; then
        continue
    fi

    # Check URL with timeout
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 --connect-timeout 5 "$url" 2>&1 || echo "000")
    
    case "$HTTP_CODE" in
        200|301|302|303|307|308)
            ((VALID++))
            ;;
        401|403)
            ((AUTH_REQUIRED++))
            echo "AUTH: $url"
            ;;
        404|410)
            ((BROKEN++))
            echo "BROKEN: $url"
            ;;
        000)
            # Check if it's SSL error or timeout
            if curl -s --max-time 5 "$url" 2>&1 | grep -q "SSL"; then
                ((SSL_ERROR++))
                echo "SSL_ERROR: $url"
            else
                ((TIMEOUT++))
                echo "TIMEOUT: $url"
            fi
            ;;
        *)
            ((BROKEN++))
            echo "ERROR ($HTTP_CODE): $url"
            ;;
    esac
done < "$TMP_URLS"

echo ""
echo "=== Summary ==="
echo "Valid: $VALID"
echo "Broken: $BROKEN"
echo "Timeout: $TIMEOUT"
echo "SSL Error: $SSL_ERROR"
echo "Auth Required: $AUTH_REQUIRED"
echo "Total: $TOTAL_URLS"
