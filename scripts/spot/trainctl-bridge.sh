#!/usr/bin/env bash
# trainctl-bridge.sh - Bridge between anno spot eval and trainctl
#
# Uses trainctl for:
# 1. Interactive dashboard monitoring (ratatui)
# 2. Faster S3 operations (native Rust parallel transfers)
# 3. Process monitoring with diagnostics
# 4. EBS optimization
#
# Usage:
#   ./trainctl-bridge.sh dashboard        # Launch interactive dashboard
#   ./trainctl-bridge.sh sync-datasets    # Download datasets using the shared filters
#   ./trainctl-bridge.sh sync-results     # Download results using trainctl s3
#   ./trainctl-bridge.sh processes <id>   # Show processes on instance
#   ./trainctl-bridge.sh top <id>         # Interactive top for instance

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.env" 2>/dev/null || true

# trainctl's recursive sync cannot apply the shared dataset exclusions.
case "${1:-}" in
    sync-datasets|sync-ds)
        ANNO_S3_BUCKET="s3://${ANNO_SPOT_BUCKET:-arc-anno-data}" \
        ANNO_CACHE_DIR="${ANNO_CACHE_DIR:-$HOME/.cache/anno}" \
            exec "$SCRIPT_DIR/../sync_datasets_s3.sh" download-datasets
        ;;
esac

# Find trainctl binary
find_trainctl() {
    if command -v trainctl &>/dev/null; then
        echo "trainctl"
    elif [ -f "$HOME/Documents/dev/trainctl/target/release/trainctl" ]; then
        echo "$HOME/Documents/dev/trainctl/target/release/trainctl"
    elif [ -f "../trainctl/target/release/trainctl" ]; then
        echo "../trainctl/target/release/trainctl"
    else
        echo ""
    fi
}

TRAINCTL=$(find_trainctl)

if [ -z "$TRAINCTL" ]; then
    echo "trainctl not found. Build it:"
    echo "  cd ../trainctl && cargo build --release"
    exit 1
fi

REGION="${ANNO_SPOT_REGION:-us-east-1}"
BUCKET="${ANNO_SPOT_BUCKET:-arc-anno-data}"

case "${1:-help}" in
    dashboard|dash)
        # Launch trainctl interactive dashboard
        echo "Launching trainctl dashboard..."
        $TRAINCTL monitor --dashboard --update-interval 5
        ;;
        
    sync-results|sync-res)
        # Download evaluation results
        OUTPUT_DIR="${2:-reports/spot-results}"
        mkdir -p "$OUTPUT_DIR"
        echo "Downloading results from s3://$BUCKET/results/ to $OUTPUT_DIR/"
        $TRAINCTL s3 sync "$OUTPUT_DIR" "s3://$BUCKET/results" --direction down
        echo "Results downloaded to $OUTPUT_DIR/"
        ;;
        
    upload-src)
        # Upload source code using trainctl's fast S3
        echo "Archiving source..."
        git archive --format=tar.gz HEAD -o /tmp/anno-src.tar.gz
        echo "Uploading to s3://$BUCKET/src/anno-src.tar.gz..."
        $TRAINCTL s3 upload /tmp/anno-src.tar.gz "s3://$BUCKET/src/anno-src.tar.gz"
        echo "Source uploaded."
        ;;
        
    processes|ps)
        # Show processes on instance
        INSTANCE_ID="${2:-}"
        if [ -z "$INSTANCE_ID" ]; then
            # Find first anno-eval-worker instance
            INSTANCE_ID=$(aws ec2 describe-instances \
                --filters "Name=tag:Name,Values=anno-eval-worker" "Name=instance-state-name,Values=running" \
                --query 'Reservations[0].Instances[0].InstanceId' \
                --output text --region "$REGION" 2>/dev/null || echo "")
        fi
        if [ -z "$INSTANCE_ID" ] || [ "$INSTANCE_ID" = "None" ]; then
            echo "No running anno-eval-worker instances found"
            exit 1
        fi
        echo "Processes on $INSTANCE_ID:"
        $TRAINCTL aws processes "$INSTANCE_ID"
        ;;
        
    top)
        # Interactive top for instance
        INSTANCE_ID="${2:-}"
        if [ -z "$INSTANCE_ID" ]; then
            INSTANCE_ID=$(aws ec2 describe-instances \
                --filters "Name=tag:Name,Values=anno-eval-worker" "Name=instance-state-name,Values=running" \
                --query 'Reservations[0].Instances[0].InstanceId' \
                --output text --region "$REGION" 2>/dev/null || echo "")
        fi
        if [ -z "$INSTANCE_ID" ] || [ "$INSTANCE_ID" = "None" ]; then
            echo "No running anno-eval-worker instances found"
            exit 1
        fi
        echo "Interactive top for $INSTANCE_ID (Ctrl+C to exit):"
        $TRAINCTL aws processes "$INSTANCE_ID" --watch
        ;;
        
    exec)
        # Execute command on instance via SSM
        INSTANCE_ID="${2:-}"
        CMD="${3:-}"
        if [ -z "$INSTANCE_ID" ] || [ -z "$CMD" ]; then
            echo "Usage: $0 exec <instance-id> <command>"
            exit 1
        fi
        $TRAINCTL aws exec "$INSTANCE_ID" "$CMD"
        ;;
        
    logs)
        # Tail worker logs
        INSTANCE_ID="${2:-}"
        if [ -z "$INSTANCE_ID" ]; then
            INSTANCE_ID=$(aws ec2 describe-instances \
                --filters "Name=tag:Name,Values=anno-eval-worker" "Name=instance-state-name,Values=running" \
                --query 'Reservations[0].Instances[0].InstanceId' \
                --output text --region "$REGION" 2>/dev/null || echo "")
        fi
        if [ -z "$INSTANCE_ID" ] || [ "$INSTANCE_ID" = "None" ]; then
            echo "No running anno-eval-worker instances found"
            exit 1
        fi
        echo "Logs from $INSTANCE_ID:"
        $TRAINCTL aws exec "$INSTANCE_ID" "tail -100 /var/log/anno-worker.log"
        ;;
        
    cost)
        # Show current fleet cost
        $TRAINCTL resources
        ;;
        
    help|*)
        cat <<EOF
trainctl-bridge: Use trainctl for enhanced spot evaluation

Commands:
  dashboard         Launch interactive ratatui dashboard
  sync-datasets     Download permitted datasets from S3 (s5cmd or aws)
  sync-results      Download evaluation results from S3
  upload-src        Upload source code to S3
  processes <id>    Show processes on instance (default: first worker)
  top <id>          Interactive top for instance (default: first worker)
  exec <id> <cmd>   Execute command on instance via SSM
  logs <id>         Tail worker logs (default: first worker)
  cost              Show current resource costs

trainctl location: $TRAINCTL
EOF
        ;;
esac

