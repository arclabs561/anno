#!/bin/bash
# Launch spot evaluation using trainctl
#
# Uses trainctl (../trainctl) for robust spot instance management.
# trainctl provides: SSM integration, EBS optimization, monitoring dashboard.
#
# Usage:
#   ./scripts/spot/launch-trainctl.sh              # Launch single worker
#   ./scripts/spot/launch-trainctl.sh 4            # Launch 4 workers
#   ./scripts/spot/launch-trainctl.sh 4 quick      # 4 workers, quick profile

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
NUM_WORKERS="${1:-1}"
PROFILE="${2:-full}"

# Load config if exists
source "$SCRIPT_DIR/../s3_bucket.sh"
anno_spot_config "$SCRIPT_DIR/config.env"

if [[ ${1:-} == --help || ${1:-} == -h || ${1:-} == help ]]; then
    sed -n '2,/^$/s/^# *//p' "${BASH_SOURCE[0]}"
    exit 0
fi
REGION="${ANNO_SPOT_REGION:-us-east-1}"
BUCKET=$(anno_bucket_name "${ANNO_SPOT_BUCKET:-}" "ANNO_SPOT_BUCKET or ANNO_S3_BUCKET")
export ANNO_SPOT_BUCKET="$BUCKET"
INSTANCE_TYPE="${ANNO_SPOT_INSTANCE_TYPE:-c7i.xlarge}"

# Check for trainctl
TRAINCTL=""
if command -v trainctl &>/dev/null; then
    TRAINCTL="trainctl"
elif [[ -f "$REPO_ROOT/../trainctl/target/release/trainctl" ]]; then
    TRAINCTL="$REPO_ROOT/../trainctl/target/release/trainctl"
elif [[ -f "$REPO_ROOT/../trainctl/target/debug/trainctl" ]]; then
    TRAINCTL="$REPO_ROOT/../trainctl/target/debug/trainctl"
else
    echo "trainctl not found. Build it:"
    echo "  cd ../trainctl && cargo build --release"
    exit 1
fi

echo "========================================"
echo "  Anno Evaluation Fleet (via trainctl)"
echo "========================================"
echo ""
echo "trainctl:      $TRAINCTL"
echo "Workers:       $NUM_WORKERS"
echo "Instance type: $INSTANCE_TYPE"
echo "Profile:       $PROFILE"
echo "Region:        $REGION"
echo ""

# Generate tasks first
case "$PROFILE" in
    quick)
        BACKENDS="pattern,heuristic,stacked"
        DATASETS="WikiGold,Wnut17"
        SEEDS="42"
        ;;
    ml)
        BACKENDS="gliner,nuner,w2ner,gliner_multitask,bert_onnx"
        DATASETS="WikiGold,Wnut17,MitMovie,CoNLL2003Sample"
        SEEDS="42,123"
        ;;
    full)
        BACKENDS=""
        DATASETS=""
        SEEDS=""
        ;;
    *)
        echo "Unknown profile: $PROFILE (available: quick, ml, full)"
        exit 1
        ;;
esac

echo "Generating evaluation tasks..."
if [[ -n "$BACKENDS" ]]; then
    uv run "$SCRIPT_DIR/orchestrate.py" generate \
        --backends "$BACKENDS" \
        --datasets "$DATASETS" \
        --seeds "$SEEDS"
else
    uv run "$SCRIPT_DIR/orchestrate.py" generate
fi

# Launch workers
INSTANCE_IDS=()
echo ""
echo "Launching $NUM_WORKERS spot worker(s)..."

for i in $(seq 1 "$NUM_WORKERS"); do
    echo ""
    echo "--- Worker $i/$NUM_WORKERS ---"
    
    # Create spot instance via trainctl
    RESULT=$("$TRAINCTL" aws create "$INSTANCE_TYPE" --spot 2>&1)
    INSTANCE_ID=$(echo "$RESULT" | grep -oE 'i-[a-z0-9]+' | head -1)
    
    if [[ -z "$INSTANCE_ID" ]]; then
        echo "Failed to create instance $i: $RESULT"
        continue
    fi
    
    echo "Created: $INSTANCE_ID"
    INSTANCE_IDS+=("$INSTANCE_ID")
    
    # Wait for SSM to be ready
    echo "Waiting for SSM agent..."
    sleep 30
    
    # Sync anno repo and run worker
    echo "Starting worker on $INSTANCE_ID..."
    
    # Use SSM to run the worker setup and start
    {
        printf 'export ANNO_SPOT_BUCKET=%q ANNO_SPOT_REGION=%q\n' "$BUCKET" "$REGION"
        cat <<'WORKER_SETUP'
set -e

# Install dependencies
sudo dnf install -y git gcc openssl-devel pkg-config || \
sudo yum install -y git gcc openssl-devel pkg-config || \
sudo apt-get update && sudo apt-get install -y git build-essential libssl-dev pkg-config

# Install Rust if not present
if ! command -v cargo &>/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source $HOME/.cargo/env
fi

# Use the same checked-in worker source as the launch-template path.
mkdir -p "$HOME/anno"
archive=$(mktemp)
aws s3 cp "s3://$ANNO_SPOT_BUCKET/src/anno-src.tar.gz" "$archive" --region "$ANNO_SPOT_REGION"
tar xzf "$archive" -C "$HOME/anno"
rm -f "$archive"
cd "$HOME/anno"

# Set up environment
export ANNO_CACHE_DIR=/tmp/anno-cache
export CARGO_HOME=/tmp/cargo
mkdir -p $ANNO_CACHE_DIR $CARGO_HOME

# Build anno
cargo build --locked --release -p anno-cli --bin anno --features "eval onnx" 2>&1 | tail -5
cargo build --locked --release -p anno-eval --example public_benchmark_report --features "eval onnx" 2>&1 | tail -5

echo "Worker ready on $(hostname)"
WORKER_SETUP
    } | "$TRAINCTL" aws ssh "$INSTANCE_ID"

    # Start the actual worker process in background
    echo "Launching worker process on $INSTANCE_ID..."
    {
        printf 'export ANNO_SPOT_REGION=%q ANNO_SPOT_BUCKET=%q ANNO_SPOT_QUEUE=%q ANNO_SPOT_QUEUE_URL=%q\n' \
            "$REGION" "$BUCKET" "${ANNO_SPOT_QUEUE:-anno-eval-${BUCKET//./_}}" "${ANNO_SPOT_QUEUE_URL:-}"
        cat <<'WORKER_RUN'
cd "$HOME/anno"
export ANNO_SOURCE_DIR="$PWD" ANNO_CACHE_MOUNT=/tmp/anno-cache CARGO_TARGET_DIR="$PWD/target"
nohup ./scripts/spot/worker.sh > /tmp/worker.log 2>&1 &
echo "Worker PID: $!"
WORKER_RUN
    } | "$TRAINCTL" aws ssh "$INSTANCE_ID" --background

done

# Summary
echo ""
echo "========================================"
echo "  Fleet Launched"
echo "========================================"
echo ""
echo "Instance IDs:"
for id in "${INSTANCE_IDS[@]}"; do
    echo "  - $id"
done
echo ""
echo "Commands:"
echo "  Monitor:   $TRAINCTL aws processes <instance-id> --watch"
echo "  Dashboard: $TRAINCTL aws top <instance-id>"
echo "  Logs:      $TRAINCTL aws logs <instance-id>"
echo "  SSH:       $TRAINCTL aws ssh <instance-id>"
echo "  Status:    just spot-status"
echo "  Terminate: $TRAINCTL aws terminate <instance-id>"
echo ""

# Save instance IDs
printf '%s\n' "${INSTANCE_IDS[@]}" > "$SCRIPT_DIR/instance_ids.txt"
echo "Instance IDs saved to: scripts/spot/instance_ids.txt"

