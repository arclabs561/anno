#!/bin/bash
# Spot instance worker script
#
# Runs on each spot instance:
# 1. Mounts EBS cache volume
# 2. Syncs datasets/models from S3
# 3. Pulls tasks from SQS, runs eval, pushes results
# 4. Monitors for spot interruption
#
# Usually launched via SSM RunCommand or in user-data.

set -euo pipefail

# Ensure HOME is set (SSM/cloud-init may not set it)
export HOME="${HOME:-/root}"

# Load config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/config.env" ]]; then
    source "$SCRIPT_DIR/config.env"
fi

REGION="${ANNO_SPOT_REGION:-us-east-1}"
BUCKET="${ANNO_SPOT_BUCKET:-arc-anno-data}"
QUEUE_URL="${ANNO_SPOT_QUEUE_URL:-}"
# Use /opt/cache on root volume (100 GiB) instead of separate EBS
CACHE_MOUNT="${ANNO_CACHE_MOUNT:-/opt/cache}"
# Source is extracted by userdata to /root/anno
ANNO_DIR="${ANNO_SOURCE_DIR:-/root/anno}"

# Logging
LOG_GROUP="/aws/anno-eval/workers"
# Use IMDSv2 (token required)
IMDS_TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 300" 2>/dev/null || true)
if [[ -n "$IMDS_TOKEN" ]]; then
    INSTANCE_ID=$(curl -s -H "X-aws-ec2-metadata-token: $IMDS_TOKEN" http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo "local")
else
    INSTANCE_ID="local"
fi

log() {
    local level="$1"
    shift
    echo "[$(date -uIs)] [$level] [$INSTANCE_ID] $*"
}

log_info() { log "INFO" "$@"; }
log_warn() { log "WARN" "$@"; }
log_error() { log "ERROR" "$@"; }

# ============================================================================
# Cache/Volume Setup
# ============================================================================

setup_cache_volume() {
    log_info "Setting up cache directory at $CACHE_MOUNT..."
    
    # Create directory structure on root volume (100 GiB, no separate EBS)
    mkdir -p "$CACHE_MOUNT"/{cargo,rustup,sccache,target,datasets,models}
    chown -R "$(whoami):$(whoami)" "$CACHE_MOUNT" 2>/dev/null || true
    
    log_info "Cache directory ready"
}

setup_rust_env() {
    log_info "Setting up Rust environment..."
    
    # Use env vars from userdata if set, otherwise use defaults
    export CARGO_HOME="${CARGO_HOME:-$HOME/.cargo}"
    export RUSTUP_HOME="${RUSTUP_HOME:-$HOME/.rustup}"
    export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$CACHE_MOUNT/target}"
    export ANNO_CACHE_DIR="${ANNO_CACHE_DIR:-$CACHE_MOUNT}"
    
    # Add cargo to PATH if not already
    export PATH="$CARGO_HOME/bin:$PATH"
    
    # Verify cargo is available (should be installed by userdata)
    if ! command -v cargo &>/dev/null; then
        log_warn "Cargo not found, installing..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
            sh -s -- -y --default-toolchain stable --no-modify-path
    fi
    
    log_info "Rust $(rustc --version 2>/dev/null || echo 'unknown')"
}

sync_from_s3() {
    # Skip full sync - ANNO_S3_CACHE=1 fetches datasets on-demand
    # This avoids a large upfront data transfer step.
    log_info "S3 cache enabled - datasets/models fetched on-demand"
    export ANNO_S3_CACHE=1
    export ANNO_S3_BUCKET="$BUCKET"
}

clone_and_build() {
    log_info "Setting up anno build..."
    
    # Source is already extracted by userdata to ANNO_DIR
    if [[ ! -d "$ANNO_DIR" ]]; then
        log_error "Anno source not found at $ANNO_DIR - userdata must extract it"
        exit 1
    fi
    
    cd "$ANNO_DIR"
    
    # Set HF_HOME for model loading (synced by userdata)
    export HF_HOME="${CACHE_MOUNT}/models"
    
    # Check if already built
    if [[ -x "$CARGO_TARGET_DIR/release/anno" ]]; then
        log_info "Using cached build at $CARGO_TARGET_DIR/release/anno"
        return 0
    fi
    
    # Build release with ONNX support for ML backends
    log_info "Building anno with ONNX (this may take a while on fresh instances)..."
    cargo build --release -p anno-cli --bin anno --features "eval onnx" 2>&1 | tail -5
    
    log_info "Build complete: $($CARGO_TARGET_DIR/release/anno --version 2>/dev/null || echo 'built')"
}

# ============================================================================
# Task Processing
# ============================================================================

receive_task() {
    # Poll SQS for next task
    local response
    response=$(aws sqs receive-message \
        --queue-url "$QUEUE_URL" \
        --region "$REGION" \
        --max-number-of-messages 1 \
        --wait-time-seconds 20 \
        --visibility-timeout 600 \
        --output json 2>/dev/null)
    
    if [[ -z "$response" ]] || [[ "$(echo "$response" | jq -r '.Messages')" == "null" ]]; then
        return 1
    fi
    
    echo "$response"
}

process_task() {
    local task_json="$1"
    
    local body
    body=$(echo "$task_json" | jq -r '.Messages[0].Body')
    local receipt
    receipt=$(echo "$task_json" | jq -r '.Messages[0].ReceiptHandle')
    
    local backend dataset seed max_examples task_type
    backend=$(echo "$body" | jq -r '.backend')
    dataset=$(echo "$body" | jq -r '.dataset')
    seed=$(echo "$body" | jq -r '.seed // 42')
    max_examples=$(echo "$body" | jq -r '.max_examples // 500')
    task_type=$(echo "$body" | jq -r '.task // "ner"')
    
    log_info "Processing: backend=$backend dataset=$dataset seed=$seed"
    
    local output_file="/tmp/result_${backend}_${dataset}_${seed}.json"
    local start_time
    start_time=$(date +%s)
    
    cd "$ANNO_DIR"
    
    # Enable S3 fallback for dataset loading
    export ANNO_S3_CACHE=1
    export ANNO_S3_BUCKET="$BUCKET"
    
    # Set HF_HOME for ONNX model loading
    export HF_HOME="${CACHE_MOUNT}/models"
    
    # Run evaluation using anno benchmark (uses TaskEvaluator)
    # Monitor memory usage and detect OOM (exit code 137 = SIGKILL)
    local exit_code=0
    local anno_bin="$CARGO_TARGET_DIR/release/anno"
    
    # Check available memory before starting
    local mem_available
    mem_available=$(free -m | awk '/^Mem:/{print $7}')
    log_info "Available memory: ${mem_available}MB before evaluation"
    
    # Run with timeout and memory monitoring
    if ! timeout 1800 "$anno_bin" benchmark \
        --datasets "$dataset" \
        --backends "$backend" \
        --tasks "$task_type" \
        --max-examples "$max_examples" \
        --seed "$seed" \
        --output "$output_file" 2>&1 | tee /tmp/eval_output.log; then
        exit_code=${PIPESTATUS[0]}
    fi
    
    # Check if OOM occurred (exit code 137 = 128 + 9 SIGKILL)
    if [[ $exit_code -eq 137 ]]; then
        log_error "OOM detected (exit 137) for $backend/$dataset"
        # Reduce max_examples for retry if too high
        if [[ $max_examples -gt 50 ]]; then
            log_warn "Reducing max_examples from $max_examples to 30 for OOM-prone dataset"
            max_examples=30
        fi
    fi
    
    local end_time
    end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # Log memory after evaluation
    local mem_after
    mem_after=$(free -m | awk '/^Mem:/{print $7}')
    log_info "Available memory: ${mem_after}MB after evaluation (used: $((mem_available - mem_after))MB)"
    
    # Upload result with metadata in filename
    local timestamp
    timestamp=$(date +%Y%m%d_%H%M%S)
    local result_key="results/${backend}/${dataset}/seed_${seed}_${timestamp}.md"
    
    if [[ -f "$output_file" && -s "$output_file" ]]; then
        # Output is markdown, upload directly with metadata header
        {
            echo "<!-- _meta: backend=$backend dataset=$dataset seed=$seed instance=$INSTANCE_ID duration=${duration}s exit_code=$exit_code -->"
            cat "$output_file"
        } > "${output_file}.wrapped"
        
        # Upload to S3
        aws s3 cp "${output_file}.wrapped" "s3://$BUCKET/$result_key" --region "$REGION"
        log_info "Result uploaded: s3://$BUCKET/$result_key"
    else
        # Upload error marker
        echo "{\"error\": \"evaluation failed\", \"exit_code\": $exit_code, \"backend\": \"$backend\", \"dataset\": \"$dataset\"}" | \
            aws s3 cp - "s3://$BUCKET/${result_key}.error" --region "$REGION"
        log_error "Evaluation failed for $backend/$dataset (exit $exit_code)"
    fi
    
    # Delete message from queue (task complete)
    aws sqs delete-message \
        --queue-url "$QUEUE_URL" \
        --receipt-handle "$receipt" \
        --region "$REGION"
    
    log_info "Task complete: $backend/$dataset in ${duration}s"
}

# ============================================================================
# Interruption Handling
# ============================================================================

check_spot_interruption() {
    # Check for spot termination notice
    local action
    action=$(curl -s -f http://169.254.169.254/latest/meta-data/spot/instance-action 2>/dev/null || true)
    
    if [[ -n "$action" ]]; then
        log_warn "SPOT INTERRUPTION: $action"
        return 0
    fi
    return 1
}

cleanup() {
    log_info "Cleaning up..."
    
    # Stop sccache to flush
    sccache --stop-server 2>/dev/null || true
    
    # Sync any remaining data
    sync
    
    log_info "Cleanup complete"
}

trap cleanup EXIT

# ============================================================================
# Main Loop
# ============================================================================

main() {
    log_info "Starting anno evaluation worker..."
    
    # Setup
    setup_cache_volume
    setup_rust_env
    sync_from_s3
    clone_and_build
    
    log_info "Worker ready, polling for tasks..."
    
    local idle_count=0
    local max_idle=30  # Exit after ~10 minutes of no tasks (30 * 20s polls)
    
    while true; do
        # Check for spot interruption
        if check_spot_interruption; then
            log_warn "Exiting due to spot interruption notice"
            break
        fi
        
        # Try to get a task
        local task
        if task=$(receive_task); then
            idle_count=0
            process_task "$task"
        else
            ((idle_count++))
            log_info "No tasks available (idle $idle_count/$max_idle)"
            
            if [[ $idle_count -ge $max_idle ]]; then
                log_info "Max idle reached, shutting down"
                break
            fi
        fi
    done
    
    log_info "Worker shutting down"
}

# Run if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi

