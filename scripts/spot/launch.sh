#!/bin/bash
# Launch spot fleet for evaluation
#
# Convenience wrapper around orchestrate.py for manual control.
#
# Usage:
#   ./scripts/spot/launch.sh              # Launch with default settings
#   ./scripts/spot/launch.sh 8            # Launch with 8 instances
#   ./scripts/spot/launch.sh 4 quick      # Launch 4 instances, quick backends only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLEET_SIZE="${1:-4}"
PROFILE="${2:-full}"

# Load config if exists
if [[ -f "$SCRIPT_DIR/config.env" ]]; then
    source "$SCRIPT_DIR/config.env"
fi

REGION="${ANNO_SPOT_REGION:-us-east-1}"

echo "========================================"
echo "  Anno Spot Fleet Launcher"
echo "========================================"
echo ""
echo "Region:     $REGION"
echo "Fleet size: $FLEET_SIZE"
echo "Profile:    $PROFILE"
echo ""

case "$PROFILE" in
    quick)
        BACKENDS="pattern,heuristic,stacked"
        DATASETS="WikiGold,Wnut17"
        SEEDS="42"
        ;;
    ml)
        BACKENDS="gliner,nuner,w2ner,gliner_multitask,bert_onnx,gliner_candle"
        DATASETS="WikiGold,Wnut17,MitMovie,CoNLL2003Sample"
        SEEDS="42,123"
        ;;
    full)
        # All backends and datasets (defined in orchestrate.py)
        BACKENDS=""
        DATASETS=""
        SEEDS=""
        ;;
    *)
        echo "Unknown profile: $PROFILE"
        echo "Available: quick, ml, full"
        exit 1
        ;;
esac

# Generate tasks
echo "Generating evaluation tasks..."
if [[ -n "$BACKENDS" ]]; then
    uv run "$SCRIPT_DIR/orchestrate.py" generate \
        --backends "$BACKENDS" \
        --datasets "$DATASETS" \
        --seeds "$SEEDS"
else
    uv run "$SCRIPT_DIR/orchestrate.py" generate
fi

# Launch fleet
echo ""
echo "Launching spot fleet..."
uv run "$SCRIPT_DIR/orchestrate.py" launch --fleet-size "$FLEET_SIZE"

echo ""
echo "========================================"
echo "  Fleet Launched"
echo "========================================"
echo ""
echo "Monitor progress:"
echo "  just spot-status"
echo ""
echo "View results when complete:"
echo "  just spot-results"
echo ""
echo "Cancel fleet:"
echo "  just spot-teardown"

