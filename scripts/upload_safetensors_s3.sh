#!/bin/bash
# Convert and upload safetensors to S3 for GLiNER-Candle backend
# This pre-converts models so CI/users don't need Python+torch
#
# Usage:
#   ./scripts/upload_safetensors_s3.sh
#
# Prerequisites:
#   - s5cmd or aws cli
#   - Python with torch, safetensors (for conversion)
#   - Models already downloaded to HF cache

set -e

S3_BUCKET="${ANNO_S3_BUCKET:-arc-anno-data}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub"
CONVERT_SCRIPT="$(dirname "$0")/convert_pytorch_to_safetensors.py"

echo "=== Uploading safetensors to s3://$S3_BUCKET/safetensors/ ==="
echo ""

# Models that need safetensors conversion for Candle
# Format: "hf_model_name" "s3_key"
GLINER_MODELS=(
    "knowledgator--gliner-x-small|gliner/gliner-x-small"
    "knowledgator--GLiNER-Small|gliner/gliner-small"
    "knowledgator--GLiNER-Medium|gliner/gliner-medium"
    "knowledgator--GLiNER-Large|gliner/gliner-large"
    "urchade--gliner_base|gliner/gliner-base"
    "urchade--gliner_small-v2.1|gliner/gliner-small-v2.1"
    "urchade--gliner_medium-v2.1|gliner/gliner-medium-v2.1"
    "urchade--gliner_large-v2.1|gliner/gliner-large-v2.1"
)

# Check for upload tool
if command -v s5cmd &> /dev/null; then
    S3_CMD="s5cmd"
elif command -v aws &> /dev/null; then
    S3_CMD="aws"
else
    echo "ERROR: Neither s5cmd nor aws cli found"
    echo "Install: brew install peak/tap/s5cmd"
    exit 1
fi

uploaded=0
skipped=0
converted=0
failed=0

for entry in "${GLINER_MODELS[@]}"; do
    model="${entry%%|*}"
    s3_key="${entry##*|}"
    
    # Convert model name to HF cache path format
    local_path="$HF_CACHE/models--${model}"
    
    # Find snapshot directory
    if [ -d "$local_path/snapshots" ]; then
        snapshot_dir=$(ls -t "$local_path/snapshots" 2>/dev/null | head -1)
        model_dir="$local_path/snapshots/$snapshot_dir"
    else
        echo "SKIP: $model (not in local cache)"
        ((skipped++)) || true
        continue
    fi
    
    echo ""
    echo "=== Processing: $model ==="
    
    # Check for existing safetensors
    safetensors_file="$model_dir/model.safetensors"
    pytorch_file="$model_dir/pytorch_model.bin"
    
    # If safetensors already exists, upload it
    if [ -f "$safetensors_file" ]; then
        echo "  Found existing: model.safetensors"
    elif [ -f "$pytorch_file" ]; then
        # Convert to safetensors
        echo "  Converting: pytorch_model.bin -> model.safetensors"
        if uv run "$CONVERT_SCRIPT" "$pytorch_file" "$safetensors_file"; then
            echo "  Conversion successful"
            ((converted++)) || true
        else
            echo "  ERROR: Conversion failed"
            ((failed++)) || true
            continue
        fi
    else
        echo "  SKIP: No pytorch_model.bin or model.safetensors found"
        ((skipped++)) || true
        continue
    fi
    
    # Upload to S3
    s3_dest="s3://$S3_BUCKET/safetensors/$s3_key/model.safetensors"
    
    echo "  Uploading to: $s3_dest"
    
    if [ "$S3_CMD" = "s5cmd" ]; then
        if s5cmd cp "$safetensors_file" "$s3_dest" 2>&1; then
            echo "  OK"
            ((uploaded++)) || true
        else
            echo "  FAILED"
            ((failed++)) || true
        fi
    else
        if aws s3 cp "$safetensors_file" "$s3_dest" 2>&1; then
            echo "  OK"
            ((uploaded++)) || true
        else
            echo "  FAILED"
            ((failed++)) || true
        fi
    fi
    
    # Also upload config.json if present
    config_file="$model_dir/config.json"
    if [ -f "$config_file" ]; then
        config_dest="s3://$S3_BUCKET/safetensors/$s3_key/config.json"
        echo "  Uploading config.json"
        if [ "$S3_CMD" = "s5cmd" ]; then
            s5cmd cp "$config_file" "$config_dest" 2>/dev/null || true
        else
            aws s3 cp "$config_file" "$config_dest" 2>/dev/null || true
        fi
    fi
    
    # Upload tokenizer files if present
    for tok_file in tokenizer.json tokenizer_config.json vocab.txt special_tokens_map.json; do
        src="$model_dir/$tok_file"
        if [ -f "$src" ]; then
            echo "  Uploading $tok_file"
            dest="s3://$S3_BUCKET/safetensors/$s3_key/$tok_file"
            if [ "$S3_CMD" = "s5cmd" ]; then
                s5cmd cp "$src" "$dest" 2>/dev/null || true
            else
                aws s3 cp "$src" "$dest" 2>/dev/null || true
            fi
        fi
    done
done

echo ""
echo "=== Summary ==="
echo "Uploaded:  $uploaded"
echo "Converted: $converted"
echo "Skipped:   $skipped"
echo "Failed:    $failed"

# Generate manifest
echo ""
echo "=== Generating safetensors manifest ==="

manifest_file="/tmp/safetensors_manifest.json"
cat > "$manifest_file" << EOF
{
  "generated_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "bucket": "s3://$S3_BUCKET",
  "description": "Pre-converted safetensors for GLiNER-Candle backend",
  "models": [
EOF

first=true
for entry in "${GLINER_MODELS[@]}"; do
    model="${entry%%|*}"
    s3_key="${entry##*|}"
    
    s3_path="s3://$S3_BUCKET/safetensors/$s3_key/model.safetensors"
    
    # Check if file exists in S3
    if [ "$S3_CMD" = "s5cmd" ]; then
        exists=$(s5cmd ls "$s3_path" 2>/dev/null | wc -l)
    else
        exists=$(aws s3 ls "$s3_path" 2>/dev/null | wc -l)
    fi
    
    if [ "$exists" -gt 0 ]; then
        if [ "$first" = true ]; then
            first=false
        else
            echo "," >> "$manifest_file"
        fi
        cat >> "$manifest_file" << EOF
    {
      "model": "$model",
      "s3_key": "$s3_key",
      "path": "$s3_path"
    }
EOF
    fi
done

cat >> "$manifest_file" << EOF

  ]
}
EOF

# Upload manifest
if [ "$S3_CMD" = "s5cmd" ]; then
    s5cmd cp "$manifest_file" "s3://$S3_BUCKET/manifests/safetensors.json"
else
    aws s3 cp "$manifest_file" "s3://$S3_BUCKET/manifests/safetensors.json"
fi

echo "Manifest uploaded to s3://$S3_BUCKET/manifests/safetensors.json"
cat "$manifest_file"
