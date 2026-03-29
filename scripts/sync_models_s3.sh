#!/usr/bin/env bash
# Sync model weights to/from S3 cache bucket
#
# Usage:
#   ./scripts/sync_models_s3.sh upload    # Upload local HF cache to S3
#   ./scripts/sync_models_s3.sh download  # Download from S3 to local HF cache
#   ./scripts/sync_models_s3.sh status    # Show what's cached locally vs S3
#   ./scripts/sync_models_s3.sh ls        # List S3 model contents
#
# Environment Variables:
#   ANNO_S3_BUCKET       S3 bucket name (default: arc-anno-data)
#   HF_HOME              HuggingFace cache root (default: ~/.cache/huggingface)
#
# After ONNX auto-export (GLiNER bi-encoder etc.), run `upload` to persist
# the exported ONNX files to S3 so other machines can skip the export step.

set -euo pipefail

S3_BUCKET="${ANNO_S3_BUCKET:-arc-anno-data}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub"
ACTION="${1:-status}"

# Models to sync (HF cache dirname -> S3 path)
declare -A MODELS=(
    # NER - ONNX backends
    ["protectai--bert-base-NER-onnx"]="ner/bert-ner-onnx"
    ["onnx-community--gliner_small-v2.1"]="ner/gliner/gliner-small-v2.1-onnx"
    ["knowledgator--gliner-x-small"]="ner/gliner/gliner-x-small"
    ["juampahc--gliner_multi-v2.1-onnx"]="ner/gliner/gliner-multi-v2.1"

    # NER - GLiNER bi-encoder (2026, arXiv:2602.18487)
    ["knowledgator--gliner-bi-large-v2.0"]="ner/gliner/gliner-bi-large-v2.0"
    ["knowledgator--gliner-bi-base-v2.0"]="ner/gliner/gliner-bi-base-v2.0"

    # NER - NuNER
    ["deepanwa--NuNerZero_onnx"]="ner/nuner/nuner-zero-onnx"
    ["numind--NuNER_Zero"]="ner/nuner/nuner-zero"
    ["numind--NuNER_Zero-4k"]="ner/nuner/nuner-zero-4k"

    # NER - BERT variants
    ["dslim--bert-base-NER"]="ner/bert-base-ner"
    ["dbmdz--bert-large-cased-finetuned-conll03-english"]="ner/bert-large-conll03"

    # Encoders (shared by multiple backends)
    ["answerdotai--ModernBERT-base"]="encoders/modernbert-base"
    ["BAAI--bge-large-en-v1.5"]="encoders/bge-large-en"

    # NER - PII
    ["knowledgator--gliner-pii-edge-v1.0"]="ner/gliner/gliner-pii-edge"

    # NER - DeBERTa
    ["deberta-v3-ner"]="ner/deberta-v3-ner"

    # Relation Extraction
    ["jackboyla--glirel-large-v0"]="re/glirel-large"
    ["knowledgator--gliner-relex-large-v1.0"]="re/gliner-relex-large"

    # Coreference Resolution
    ["shtoshni--longformer_coreference_ontonotes"]="coref/longformer-ontonotes"
    ["shtoshni--spanbert_coreference_large"]="coref/spanbert-large"
    ["shtoshni--spanbert_coreference_base"]="coref/spanbert-base"
    ["sapienzanlp--maverick-mes-ontonotes"]="coref/maverick-ontonotes"
    ["hsiehpinghan--Qwen2-0.5B-Instruct-Coreference-Resolution"]="coref/qwen2-0.5b-coref"
    ["biu-nlp--f-coref"]="coref/f-coref"
)

# Detect S3 CLI
S3CMD=""
if command -v s5cmd &>/dev/null; then
    S3CMD="s5cmd"
elif command -v aws &>/dev/null; then
    S3CMD="aws"
else
    echo "ERROR: Neither s5cmd nor aws CLI found."
    echo "  Install s5cmd: brew install peak/tap/s5cmd"
    echo "  Or AWS CLI:    brew install awscli"
    exit 1
fi

s3_sync_up() {
    local src="$1" dst="$2"
    if [[ "$S3CMD" == "s5cmd" ]]; then
        s5cmd sync "$src/*" "$dst/"
    else
        aws s3 sync "$src" "$dst" --quiet
    fi
}

s3_sync_down() {
    local src="$1" dst="$2"
    if [[ "$S3CMD" == "s5cmd" ]]; then
        s5cmd sync "$src/*" "$dst/"
    else
        aws s3 sync "$src" "$dst" --quiet
    fi
}

s3_ls() {
    local path="$1"
    if [[ "$S3CMD" == "s5cmd" ]]; then
        s5cmd ls "$path" 2>/dev/null
    else
        aws s3 ls "$path" --recursive 2>/dev/null
    fi
}

case "$ACTION" in
upload)
    echo "=== Uploading models to s3://$S3_BUCKET/models/ ==="
    uploaded=0; skipped=0; failed=0

    for model in "${!MODELS[@]}"; do
        local_path="$HF_CACHE/models--$model"
        s3_path="s3://$S3_BUCKET/models/${MODELS[$model]}"

        if [[ ! -d "$local_path" ]]; then
            echo "SKIP: $model (not in local cache)"
            ((skipped++)) || true
            continue
        fi

        local_size=$(du -sh "$local_path" 2>/dev/null | cut -f1)
        echo "UP: $model ($local_size) -> $s3_path/"

        if s3_sync_up "$local_path" "$s3_path" 2>&1; then
            echo "  OK"
            ((uploaded++)) || true
        else
            echo "  FAILED"
            ((failed++)) || true
        fi
    done

    echo ""
    echo "=== Summary: $uploaded uploaded, $skipped skipped, $failed failed ==="
    ;;

download)
    echo "=== Downloading models from s3://$S3_BUCKET/models/ ==="
    downloaded=0; skipped=0; failed=0

    for model in "${!MODELS[@]}"; do
        local_path="$HF_CACHE/models--$model"
        s3_path="s3://$S3_BUCKET/models/${MODELS[$model]}"

        if [[ -d "$local_path" ]]; then
            echo "SYNC: $model (exists, pulling updates)"
        else
            echo "DOWN: $model <- $s3_path/"
            mkdir -p "$local_path"
        fi

        if s3_sync_down "$s3_path" "$local_path" 2>&1; then
            echo "  OK"
            ((downloaded++)) || true
        else
            echo "  FAILED (may not exist in S3 yet)"
            ((failed++)) || true
        fi
    done

    echo ""
    echo "=== Summary: $downloaded synced, $skipped skipped, $failed failed ==="
    ;;

status)
    echo "=== Model cache status ==="
    printf "%-50s %-10s\n" "MODEL" "LOCAL"
    printf "%-50s %-10s\n" "-----" "-----"

    for model in $(echo "${!MODELS[@]}" | tr ' ' '\n' | sort); do
        local_path="$HF_CACHE/models--$model"

        local_status="--"
        if [[ -d "$local_path" ]]; then
            local_status=$(du -sh "$local_path" 2>/dev/null | cut -f1)
        fi

        printf "%-50s %-10s\n" "$model" "$local_status"
    done
    ;;

ls)
    echo "=== S3 model contents ==="
    s3_ls "s3://$S3_BUCKET/models/"
    ;;

*)
    echo "Usage: $0 {upload|download|status|ls}"
    exit 1
    ;;
esac
