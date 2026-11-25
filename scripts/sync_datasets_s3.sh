#!/bin/bash
# Sync anno artifacts to/from S3 cache bucket
#
# Publishes models and scripts. Reads permitted datasets from existing owner-provisioned storage.
#
# Usage:
#   ./scripts/sync_datasets_s3.sh upload    # Upload local models, scripts and manifest to S3
#   ./scripts/sync_datasets_s3.sh download  # Download from S3
#   ./scripts/sync_datasets_s3.sh download-datasets  # Only permitted dataset objects
#   ./scripts/sync_datasets_s3.sh sync      # Bidirectional sync (local wins)
#   ./scripts/sync_datasets_s3.sh status    # Show what's cached
#   ./scripts/sync_datasets_s3.sh ls        # List S3 contents
#   ./scripts/sync_datasets_s3.sh size      # Show S3 bucket size
#   ./scripts/sync_datasets_s3.sh upload-script  # Upload safetensor conversion script
#
# Environment Variables:
#   ANNO_CACHE_DIR       Local cache directory (default: ~/.cache/anno or ~/Library/Caches/anno)
#   ANNO_S3_BUCKET       S3 bucket URL (default: s3://anno-cache)
#   ANNO_CACHE_SOURCE    Cache source priority: "local", "s3", "local+s3" (default: local+s3)
#
# S3 CLI Selection (in order of preference):
#   1. s5cmd - fastest, parallel transfers (https://github.com/peak/s5cmd)
#   2. aws s3 - standard AWS CLI

set -euo pipefail

# Configurable S3 bucket for all anno artifacts (datasets, models, scripts)
# Naming: arc (global namespace) -> anno (project) -> data
S3_BUCKET="${ANNO_S3_BUCKET:-s3://arc-anno-data}"
MANIFEST_FILE="manifest.json"

# Platform-aware cache directory
if [[ "$OSTYPE" == "darwin"* ]]; then
    LOCAL_CACHE="${ANNO_CACHE_DIR:-$HOME/Library/Caches/anno}"
else
    LOCAL_CACHE="${ANNO_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/anno}"
fi

# Subdirectories for different artifact types
DATASETS_DIR="$LOCAL_CACHE/datasets"
MODELS_DIR="$LOCAL_CACHE/models"
SCRIPTS_DIR="$LOCAL_CACHE/scripts"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_debug() { [[ "${ANNO_DEBUG:-}" == "1" ]] && echo -e "${BLUE}[DEBUG]${NC} $1" || true; }

# Detect best S3 CLI tool
S3_CMD=""
detect_s3_cmd() {
    if command -v s5cmd &>/dev/null; then
        S3_CMD="s5cmd"
        log_info "Using s5cmd (fast parallel transfers)"
    elif command -v aws &>/dev/null; then
        S3_CMD="aws"
        log_info "Using aws cli (s5cmd not found, consider installing for faster transfers)"
    else
        log_error "No S3 CLI found. Install 'aws' or 's5cmd'."
        exit 1
    fi
}

check_aws() {
    detect_s3_cmd
    if [[ "$S3_CMD" == "aws" ]]; then
        if ! aws sts get-caller-identity &>/dev/null; then
            log_error "AWS credentials not configured. Run 'aws configure' first."
            exit 1
        fi
    elif [[ "$S3_CMD" == "s5cmd" ]]; then
        # s5cmd uses AWS credentials from environment or ~/.aws/
        # Try a simple ls to verify access
        if ! s5cmd ls "${S3_BUCKET}/" &>/dev/null 2>&1; then
            # Bucket might not exist yet, that's ok for first upload
            log_warn "Bucket ${S3_BUCKET} may not exist yet or credentials issue"
        fi
    fi
}

generate_manifest() {
    local manifest_path="$1"
    log_info "Generating manifest..."
    
    # Create manifest with metadata about all artifacts
    cat > "$manifest_path" << MANIFEST_HEADER
{
  "version": "2.0",
  "created": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "created_by": "${USER:-unknown}@$(hostname -s)",
  "anno_version": "$(cargo pkgid -p anno 2>/dev/null | cut -d'#' -f2 || echo 'unknown')",
  "bucket": "$S3_BUCKET",
  "description": "Anno NER artifacts cache (datasets, models, scripts)",
  "artifacts": {
    "datasets": [
MANIFEST_HEADER

    # No raw dataset package has been approved for shared publication.
    local first=true
    cat >> "$manifest_path" << MID_SECTION
    ],
    "models": [
MID_SECTION

    # Add models
    first=true
    if [ -d "$MODELS_DIR" ]; then
        for f in "$MODELS_DIR"/*; do
            if [ -f "$f" ]; then
                local basename=$(basename "$f")
                local size=$(stat -f%z "$f" 2>/dev/null || stat --format=%s "$f" 2>/dev/null || echo 0)
                
                if [ "$first" = true ]; then
                    first=false
                else
                    echo "," >> "$manifest_path"
                fi
                
                cat >> "$manifest_path" << ENTRY
      {"filename": "$basename", "size_bytes": $size}
ENTRY
            fi
        done
    fi
    
    cat >> "$manifest_path" << MID_SECTION2
    ],
    "scripts": [
MID_SECTION2

    # Add scripts
    first=true
    if [ -d "$SCRIPTS_DIR" ]; then
        for f in "$SCRIPTS_DIR"/*; do
            if [ -f "$f" ]; then
                local basename=$(basename "$f")
                local size=$(stat -f%z "$f" 2>/dev/null || stat --format=%s "$f" 2>/dev/null || echo 0)
                
                if [ "$first" = true ]; then
                    first=false
                else
                    echo "," >> "$manifest_path"
                fi
                
                cat >> "$manifest_path" << ENTRY
      {"filename": "$basename", "size_bytes": $size}
ENTRY
            fi
        done
    fi
    
    cat >> "$manifest_path" << MANIFEST_FOOTER
    ]
  }
}
MANIFEST_FOOTER
    
    log_info "Manifest generated: $manifest_path"
}

upload_all() {
    log_info "Uploading all artifacts to S3..."
    
    mkdir -p "$DATASETS_DIR" "$MODELS_DIR" "$SCRIPTS_DIR"
    
    log_info "Skipping raw datasets: no reviewed publication package. Existing objects are preserved."
    # Generate manifest before upload
    local local_manifest="$LOCAL_CACHE/$MANIFEST_FILE"
    generate_manifest "$local_manifest"
    
    if [[ "$S3_CMD" == "s5cmd" ]]; then
        # s5cmd is much faster for bulk operations
        [[ -d "$MODELS_DIR" ]] && s5cmd sync --size-only "$MODELS_DIR/" "$S3_BUCKET/models/" 2>/dev/null || true
        [[ -d "$SCRIPTS_DIR" ]] && s5cmd sync --size-only "$SCRIPTS_DIR/" "$S3_BUCKET/scripts/" 2>/dev/null || true
        s5cmd cp "$local_manifest" "$S3_BUCKET/$MANIFEST_FILE"
    else
        # aws cli fallback
        [[ -d "$MODELS_DIR" ]] && aws s3 sync "$MODELS_DIR" "$S3_BUCKET/models/" --size-only
        [[ -d "$SCRIPTS_DIR" ]] && aws s3 sync "$SCRIPTS_DIR" "$S3_BUCKET/scripts/" --size-only
        aws s3 cp "$local_manifest" "$S3_BUCKET/$MANIFEST_FILE"
    fi
    
    log_info "Upload complete."
}

download_datasets() {
    local cache_policy pattern
    local -a dataset_filters=()
    # Filter legacy names and content-addressed paths in pre-provisioned stores.
    cache_policy="$(python3 "$(dirname "$0")/qa/shared_dataset_cache.py")"
    while IFS= read -r pattern; do
        dataset_filters+=(--exclude "$pattern")
    done <<< "$cache_policy"
    mkdir -p "$DATASETS_DIR"
    if [[ "$S3_CMD" == "s5cmd" ]]; then
        s5cmd sync --size-only "${dataset_filters[@]}" "$S3_BUCKET/datasets/*" "$DATASETS_DIR/"
    else
        aws s3 sync "$S3_BUCKET/datasets/" "$DATASETS_DIR" --size-only "${dataset_filters[@]}"
    fi
}

download_all() {
    log_info "Downloading permitted artifacts from S3..."
    download_datasets
    mkdir -p "$MODELS_DIR" "$SCRIPTS_DIR"
    if [[ "$S3_CMD" == "s5cmd" ]]; then
        s5cmd sync --size-only "$S3_BUCKET/models/*" "$MODELS_DIR/" 2>/dev/null || true
        s5cmd sync --size-only "$S3_BUCKET/scripts/*" "$SCRIPTS_DIR/" 2>/dev/null || true
    else
        aws s3 sync "$S3_BUCKET/models/" "$MODELS_DIR" --size-only
        aws s3 sync "$S3_BUCKET/scripts/" "$SCRIPTS_DIR" --size-only
    fi
    
    log_info "Download complete."
    show_local_summary
}

sync_all() {
    log_info "Bidirectional sync (local wins on conflicts)..."
    download_all
    upload_all
    log_info "Sync complete."
}

show_status() {
    echo "=== Anno Cache Status ==="
    echo ""
    echo "S3 Bucket:    $S3_BUCKET"
    echo "Local Cache:  $LOCAL_CACHE"
    echo "S3 CLI:       $S3_CMD"
    echo "Cache Source: ${ANNO_CACHE_SOURCE:-local+s3}"
    echo ""
    
    echo "--- S3 Contents ---"
    if [[ "$S3_CMD" == "s5cmd" ]]; then
        echo "Datasets:"
        s5cmd ls "$S3_BUCKET/datasets/*" 2>/dev/null | head -10 || echo "  (none)"
        echo "Models:"
        s5cmd ls "$S3_BUCKET/models/*" 2>/dev/null | head -5 || echo "  (none)"
        echo "Scripts:"
        s5cmd ls "$S3_BUCKET/scripts/*" 2>/dev/null | head -5 || echo "  (none)"
    else
        aws s3 ls "$S3_BUCKET/" --recursive --human-readable 2>/dev/null | head -20 || echo "(empty)"
    fi
    echo ""
    
    show_local_summary
}

show_local_summary() {
    echo "--- Local Contents ---"
    echo "Datasets: $(ls -1 "$DATASETS_DIR" 2>/dev/null | wc -l | tr -d ' ') files"
    [[ -d "$DATASETS_DIR" ]] && du -sh "$DATASETS_DIR" 2>/dev/null || echo "  (none)"
    echo "Models: $(ls -1 "$MODELS_DIR" 2>/dev/null | wc -l | tr -d ' ') files"
    [[ -d "$MODELS_DIR" ]] && du -sh "$MODELS_DIR" 2>/dev/null || echo "  (none)"
    echo "Scripts: $(ls -1 "$SCRIPTS_DIR" 2>/dev/null | wc -l | tr -d ' ') files"
    [[ -d "$SCRIPTS_DIR" ]] && du -sh "$SCRIPTS_DIR" 2>/dev/null || echo "  (none)"
}

list_s3() {
    echo "=== S3 Bucket Contents ==="
    if [[ "$S3_CMD" == "s5cmd" ]]; then
        s5cmd ls "$S3_BUCKET/**" 2>/dev/null || echo "(empty or no access)"
    else
        aws s3 ls "$S3_BUCKET/" --recursive --human-readable 2>/dev/null || echo "(empty or no access)"
    fi
}

show_size() {
    echo "=== S3 Bucket Size ==="
    if [[ "$S3_CMD" == "s5cmd" ]]; then
        # s5cmd doesn't have a direct size command, count files and estimate
        local count=$(s5cmd ls "$S3_BUCKET/**" 2>/dev/null | wc -l | tr -d ' ')
        echo "Files: $count"
        s5cmd du "$S3_BUCKET/" 2>/dev/null || echo "Size calculation not available with s5cmd"
    else
        aws s3 ls "$S3_BUCKET/" --recursive --summarize --human-readable 2>/dev/null | tail -2
    fi
}

upload_script() {
    log_info "Uploading safetensor conversion script..."
    mkdir -p "$SCRIPTS_DIR"
    
    local script_path="$SCRIPTS_DIR/convert_pytorch_to_safetensors.py"
    
    # Create the script if it doesn't exist
    if [ ! -f "$script_path" ]; then
        cat > "$script_path" << 'SCRIPT'
#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["torch>=2.0", "safetensors>=0.4"]
# ///
"""Convert PyTorch model weights to SafeTensors format.

Usage: uv run convert_pytorch_to_safetensors.py <input.pt> <output.safetensors>

SafeTensors is a faster, safer format for storing tensors:
- Memory-mapped loading (instant load, low memory)
- No arbitrary code execution (unlike pickle)
- Cross-framework compatible
"""
import sys
from pathlib import Path

def convert(input_path: str, output_path: str) -> None:
    import torch
    from safetensors.torch import save_file

    print(f"Loading {input_path}...")
    state_dict = torch.load(input_path, map_location="cpu", weights_only=True)
    
    # Handle nested state dicts
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif "model" in state_dict:
        state_dict = state_dict["model"]
    
    # Ensure all tensors are contiguous
    state_dict = {k: v.contiguous() for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
    
    print(f"Saving {len(state_dict)} tensors to {output_path}...")
    save_file(state_dict, output_path)
    
    # Verify
    from safetensors import safe_open
    with safe_open(output_path, framework="pt") as f:
        print(f"Verified: {len(f.keys())} tensors")
    
    print("Done!")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
SCRIPT
        log_info "Created script: $script_path"
    fi
    
    # Upload to S3
    if [[ "$S3_CMD" == "s5cmd" ]]; then
        s5cmd cp "$script_path" "$S3_BUCKET/scripts/convert_pytorch_to_safetensors.py"
    else
        aws s3 cp "$script_path" "$S3_BUCKET/scripts/convert_pytorch_to_safetensors.py"
    fi
    
    log_info "Script uploaded to $S3_BUCKET/scripts/"
}

# Main
case "${1:-status}" in
    upload)
        check_aws
        upload_all
        ;;
    download-datasets)
        detect_s3_cmd
        download_datasets
        ;;
    download)
        check_aws
        download_all
        ;;
    sync)
        check_aws
        sync_all
        ;;
    status)
        check_aws
        show_status
        ;;
    ls)
        check_aws
        list_s3
        ;;
    size)
        check_aws
        show_size
        ;;
    upload-script)
        check_aws
        upload_script
        ;;
    *)
        echo "Anno Cache Sync Tool"
        echo ""
        echo "Usage: $0 {upload|download|download-datasets|sync|status|ls|size|upload-script}"
        echo ""
        echo "Commands:"
        echo "  upload        Upload local cache to S3"
        echo "  download      Download S3 cache to local"
        echo "  download-datasets  Download permitted datasets only"
        echo "  sync          Bidirectional sync (local wins)"
        echo "  status        Show cache status"
        echo "  ls            List S3 bucket contents"
        echo "  size          Show S3 bucket size"
        echo "  upload-script Upload safetensor conversion script"
        echo ""
        echo "Environment:"
        echo "  ANNO_S3_BUCKET     S3 bucket URL (default: s3://anno-cache)"
        echo "  ANNO_CACHE_DIR     Local cache dir"
        echo "  ANNO_CACHE_SOURCE  Priority: local, s3, local+s3"
        exit 1
        ;;
esac
