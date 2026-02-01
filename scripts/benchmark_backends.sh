#!/usr/bin/env bash
# Benchmark anno backends on real data
# Usage: ./scripts/benchmark_backends.sh [--quick]
#
# Outputs:
#   - Performance timing for each backend
#   - Entity counts
#   - Sample predictions for quality inspection

set -euo pipefail

ANNO="${ANNO:-./target/release/anno}"
QUICK="${1:-}"

if [[ ! -x "$ANNO" ]]; then
    echo "Building anno in release mode..."
    cargo build -p anno-cli --release --features onnx --bin anno
fi

echo "=============================================="
echo "    ANNO Backend Benchmark"
echo "=============================================="
echo ""
echo "Binary: $ANNO"
echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

# Test data
NEWS_DIR="hack/real_data/news"
if [[ ! -d "$NEWS_DIR" ]]; then
    echo "Warning: $NEWS_DIR not found, using sample text"
    NEWS_DIR=""
fi

# Backends to test
if [[ "$QUICK" == "--quick" ]]; then
    BACKENDS="pattern heuristic stacked"
else
    BACKENDS="pattern heuristic stacked ensemble gliner"
fi

# Sample texts for inline testing
SAMPLES=(
    "Tim Cook is the CEO of Apple Inc. in Cupertino, California."
    "The White House announced that President Biden met with EU leaders."
    "Dr. Jane Smith from MIT published research on BRCA1 mutations."
    "Microsoft acquired GitHub for \$7.5 billion in October 2018."
)

echo "=== 1. Performance Timing ==="
echo ""
echo "| Backend    | Time (s) | Entities |"
echo "|------------|----------|----------|"

for backend in $BACKENDS; do
    if [[ -n "$NEWS_DIR" ]]; then
        start=$(date +%s.%N)
        entity_count=0
        for f in "$NEWS_DIR"/*.txt; do
            count=$("$ANNO" extract --model "$backend" --format json --file "$f" 2>/dev/null | jq '.entities | length' 2>/dev/null || echo 0)
            entity_count=$((entity_count + count))
        done
        end=$(date +%s.%N)
        elapsed=$(echo "$end - $start" | bc)
        printf "| %-10s | %8.3f | %8d |\n" "$backend" "$elapsed" "$entity_count"
    else
        # Fallback to sample texts
        start=$(date +%s.%N)
        entity_count=0
        for sample in "${SAMPLES[@]}"; do
            count=$("$ANNO" extract --model "$backend" --format json "$sample" 2>/dev/null | jq '.entities | length' 2>/dev/null || echo 0)
            entity_count=$((entity_count + count))
        done
        end=$(date +%s.%N)
        elapsed=$(echo "$end - $start" | bc)
        printf "| %-10s | %8.3f | %8d |\n" "$backend" "$elapsed" "$entity_count"
    fi
done

echo ""
echo "=== 2. Sample Predictions ==="
echo ""

SAMPLE="Tim Cook is the CEO of Apple Inc. in Cupertino, California."
echo "Text: '$SAMPLE'"
echo ""

for backend in $BACKENDS; do
    echo "--- $backend ---"
    "$ANNO" extract --model "$backend" "$SAMPLE" 2>/dev/null || echo "(failed)"
    echo ""
done

echo "=== 3. Multi-Backend Agreement ==="
echo ""
echo "Testing ensemble weighted voting..."

"$ANNO" extract --model ensemble --format json "$SAMPLE" 2>/dev/null | jq -r '.entities[] | "\(.type) \"\(.text)\" conf=\(.confidence | tostring | .[0:4]) source=\(.source)"' || echo "(failed)"

echo ""
echo "=== Benchmark Complete ==="

