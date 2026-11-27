#!/usr/bin/env bash
# Run against the verified FP32 export used by .github/workflows/fcoref-cli.yml.
set -euo pipefail

repo=$(git rev-parse --show-toplevel)
: "${FCOREF_MODEL_PATH:?set to the pinned FP32 export}"
: "${ORT_DYLIB_PATH:?set to the ONNX Runtime shared library}"
model=$(realpath "$FCOREF_MODEL_PATH")
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

caller="$work/aurelius"
git init --quiet "$caller"
git -C "$caller" fetch --quiet --depth 1 https://github.com/arlyon/aurelius.git 1319bcdcee28ee5b668b41a2a4f85dd0722b460f
git -C "$caller" checkout --quiet --detach FETCH_HEAD
lanes="$caller/lanes"
# The pinned caller normally resolves exactly 0.13.0 from its vendor directory.
python3 - "$lanes/Cargo.toml" "$repo/crates/anno" <<'PY'
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
original = manifest.read_text()
pin = 'anno = { version = "=0.13.0",'
vendor = '[patch.crates-io]\nanno = { path = "vendor/anno" }'
assert original.count(pin) == original.count(vendor) == 1
manifest.write_text(original.replace(vendor, '').replace(pin, f'anno = {{ path = {json.dumps(sys.argv[2])},'))
PY

mkdir "$work/model"
for name in config.json encoder.onnx scorer_weights.safetensors tokenizer.json; do
  ln -s "$model/$name" "$work/model/$name"
done
curl --fail --silent --show-error --location --retry 3 \
  https://github.com/bminixhofer/nlprule/releases/download/0.6.4/en_tokenizer.bin.gz \
  --output "$work/en_tokenizer.bin.gz"
echo "b500dd208ace9ba218f6b52f8cdab63d4c09d6f2967e9bd8f917bf5984d4468a  $work/en_tokenizer.bin.gz" | sha256sum --check
gzip -dc "$work/en_tokenizer.bin.gz" > "$work/model/en_tokenizer.bin"

export AURELIUS_TEST_LANGUAGE_MODEL_DIR="$work/model" ANNO_NO_DOWNLOADS=1
cd "$lanes"
cargo test --no-default-features --features 'language postgres' --test language
git -C "$caller" apply "$repo/scripts/qa/aurelius/document-budget.patch"
cp "$repo/scripts/qa/aurelius/word_budget.rs" tests/anno_document_budget.rs
cargo test --locked --no-default-features --features 'language postgres' --test anno_document_budget
