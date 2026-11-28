#!/usr/bin/env bash
# Evaluate the pretrained Edge v2.0 checkpoint with Anno's native WikiGold scorer.
set -euo pipefail

output=${1:?usage: bench_gliner_edge.sh NEW_OUTPUT_DIRECTORY}
mkdir -- "$output"
output=$(realpath "$output")
cd "$(dirname "$0")/../.."

model=knowledgator/gliner-bi-edge-v2.0
revision=3a401e4902d93df04971c795fb1f42f37801aced
export ANNO_CACHE_DIR="$output/cache"
export GLINER_POLY_MODEL_PATH="$output/export"

bash scripts/qa/prepare_matrix_wikigold.sh
uvx --from huggingface-hub hf download "$model" --revision "$revision" --local-dir "$output/source"
uv run --with gliner==0.2.30 --with transformers==5.22.2 scripts/export_gliner_poly_onnx.py --model "$output/source" \
    --strategy manual --output "$GLINER_POLY_MODEL_PATH"
cargo run --locked -p anno-cli --features onnx,eval --bin anno -- benchmark \
    --tasks ner --datasets WikiGold --backends gliner_poly --cached-only --seed 42 --max-examples 100 \
    --output-json "$output/results.json" --output "$output/results.md"

# The evaluator can exit successfully when a requested backend is skipped.
jq -e '.results | length == 1 and (.[0] | .success and .backend == "gliner_poly"
    and .num_examples == 100 and (.metrics.f1 | type == "number"))' "$output/results.json" >/dev/null
(cd "$GLINER_POLY_MODEL_PATH" && sha256sum model.onnx label_encoder.onnx) > "$output/graphs.sha256"
jq --arg model "$model" --arg revision "$revision" --arg anno "$(git rev-parse HEAD)" \
    '{model: $model, model_revision: $revision, anno_commit: $anno,
      result: (.results[0] | {dataset, backend, seed, num_examples,
        dataset_sha256: .provenance.dataset.content_sha256,
        metrics: (.metrics | {num_gold, num_predicted, precision, recall, f1})})}' \
    "$output/results.json" > "$output/benchmark.json"
