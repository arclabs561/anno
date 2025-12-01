# Model Downloads Reference

## Overview

All models in the anno codebase are downloaded from **HuggingFace Hub** (`hf_hub` crate). There are no other model sources (no local models, no other repositories).

## Models Downloaded by Backend

### ONNX Backends (with `onnx` feature)

#### BertNEROnnx
- **Default**: `protectai/bert-base-NER-onnx`
- **Files**: `model.onnx`, `tokenizer.json`, `config.json`
- **Size**: ~400MB

#### GLiNEROnnx
- **Default**: `onnx-community/gliner_small-v2.1`
- **Alternatives**:
  - `onnx-community/gliner_medium-v2.1` (~110M params)
  - `onnx-community/gliner_large-v2.1` (~340M params)
  - `onnx-community/gliner-multitask-large-v0.5`
- **Files**: `onnx/model.onnx` (or `model.onnx`), `tokenizer.json`, `config.json`
- **Size**: ~200MB (small), ~400MB (medium), ~1.3GB (large)

#### GLiNER2Onnx
- **Default**: `fastino/gliner2-base-v1`
- **Alternatives**:
  - `knowledgator/gliner-multitask-large-v0.5`
- **Files**: `model.onnx`, `tokenizer.json`, `config.json`
- **Size**: ~400MB

#### NuNER
- **Default**: `deepanwa/NuNerZero_onnx`
- **Alternatives**:
  - `numind/NuNER_Zero` (original, may need conversion)
  - `numind/NuNER_Zero_4k` (4K context)
- **Files**: `onnx/model.onnx` (or `model.onnx`), `tokenizer.json`
- **Size**: ~200MB

#### W2NER
- **Default**: `ljynlp/w2ner-bert-base`
- **Note**: Requires authentication (401 error if not authenticated)
- **Files**: `model.onnx` (or `onnx/model.onnx`), `tokenizer.json`
- **Size**: ~400MB

#### T5Coref (Coreference Resolution)
- **Model**: T5-based coreference model
- **Files**: ONNX model, tokenizer
- **Size**: ~500MB

### Candle Backends (with `candle` feature)

#### CandleNER
- **Default**: `dslim/bert-base-NER`
- **Alternatives**:
  - `dbmdz/bert-large-cased-finetuned-conll03-english`
- **Files**: `model.safetensors` (or `pytorch_model.bin` + conversion), `config.json`, `tokenizer.json` (or `vocab.txt`)
- **Size**: ~400MB

#### GLiNERCandle
- **Default**: `knowledgator/modern-gliner-bi-large-v1.0`
- **Alternatives**:
  - `urchade/gliner_small-v2.1` (requires PyTorch→Safetensors conversion)
  - `knowledgator/gliner-x-small` (may need conversion)
- **Files**: `model.safetensors` (or `pytorch_model.bin` + conversion), `tokenizer.json`, `config.json`
- **Size**: ~1.3GB (large)

#### GLiNER2Candle
- **Model**: Uses same models as GLiNER2Onnx
- **Files**: `model.safetensors`, `tokenizer.json`, `config.json`
- **Size**: ~400MB

#### CandleEncoder (for GLiNER/Candle backends)
- **Models**:
  - `answerdotai/ModernBERT-base` (default, ModernBERT)
  - `google-bert/bert-base-uncased` (BERT)
  - `microsoft/deberta-v3-base` (DeBERTa-v3)
- **Files**: `model.safetensors`, `tokenizer.json`, `config.json`
- **Size**: ~400MB (BERT), ~500MB (DeBERTa), ~600MB (ModernBERT)

## Complete Model List

### NER Models
1. `protectai/bert-base-NER-onnx` - BERT ONNX
2. `onnx-community/gliner_small-v2.1` - GLiNER small (default)
3. `onnx-community/gliner_medium-v2.1` - GLiNER medium
4. `onnx-community/gliner_large-v2.1` - GLiNER large
5. `onnx-community/gliner-multitask-large-v0.5` - GLiNER multitask
6. `fastino/gliner2-base-v1` - GLiNER2 (default)
7. `knowledgator/gliner-multitask-large-v0.5` - GLiNER2 alternative
8. `deepanwa/NuNerZero_onnx` - NuNER (default)
9. `numind/NuNER_Zero` - NuNER original
10. `numind/NuNER_Zero_4k` - NuNER 4K context
11. `ljynlp/w2ner-bert-base` - W2NER (requires auth)
12. `dslim/bert-base-NER` - Candle BERT (default)
13. `dbmdz/bert-large-cased-finetuned-conll03-english` - Candle BERT alternative
14. `knowledgator/modern-gliner-bi-large-v1.0` - GLiNER Candle (default)
15. `urchade/gliner_small-v2.1` - GLiNER Candle alternative

### Encoder Models (for Candle backends)
16. `answerdotai/ModernBERT-base` - ModernBERT (default)
17. `google-bert/bert-base-uncased` - BERT
18. `microsoft/deberta-v3-base` - DeBERTa-v3

### Coreference Models
19. T5-based coreference model (ONNX)

## Total Download Size

**First run (all models):**
- ONNX models: ~2.5GB
- Candle models: ~3.5GB
- **Total**: ~6GB (if both features enabled)

**Typical CI run (one feature):**
- ONNX only: ~2.5GB
- Candle only: ~3.5GB

## Caching Strategy

All models are cached in `~/.cache/huggingface` by the `hf_hub` crate.

**CI Cache:**
- Path: `~/.cache/huggingface`
- Key: `hf-models-${{ runner.os }}-v2`
- Restore keys: `hf-models-${{ runner.os }}-` (partial cache hits)

**After first run:**
- Models persist in cache
- Subsequent runs use cached models (fast)
- Only new models are downloaded

## Model Sources

**100% HuggingFace Hub** - No other sources:
- ❌ No local model files
- ❌ No other model repositories
- ❌ No HTTP downloads (except via HF Hub)
- ✅ All models via `hf_hub::api::sync::Api`

## Which Models Are Actually Downloaded in CI?

### Test (ONNX backend)
Downloads when tests run:
- `protectai/bert-base-NER-onnx` (if BertNEROnnx tests run)
- `onnx-community/gliner_small-v2.1` (if GLiNEROnnx tests run)
- `deepanwa/NuNerZero_onnx` (if NuNER tests run)
- `fastino/gliner2-base-v1` (if GLiNER2Onnx tests run)
- `ljynlp/w2ner-bert-base` (if W2NER tests run, may fail due to auth)

### Test (Candle backend)
Downloads when tests run:
- `dslim/bert-base-NER` (if CandleNER tests run)
- `knowledgator/modern-gliner-bi-large-v1.0` (if GLiNERCandle tests run)
- `answerdotai/ModernBERT-base` (encoder for GLiNER models)
- `fastino/gliner2-base-v1` (if GLiNER2Candle tests run)

**Note**: Tests may not download all models - only models used by tests that actually run.

## Recommendations

1. **Cache is working** - Models are cached after first download
2. **First run is slow** - Unavoidable, must download models
3. **Subsequent runs are fast** - Uses cache
4. **Consider model selection** - Only download models you need
5. **Feature flags** - Use `onnx` OR `candle`, not both, to reduce download size

