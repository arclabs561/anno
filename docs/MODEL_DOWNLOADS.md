# Model Downloads Reference

## Overview

This document describes where model weights come from and how models are used in this codebase.

## Trained vs Off-the-Shelf Models

### Pre-Trained Models (Downloaded from HuggingFace)

**All NER and encoder models are pre-trained and downloaded from HuggingFace Hub.** These models were trained by their original authors (GLiNER team, NuMind, etc.) and we only run inference.

| Model Type | Source | Training | Weights Location |
|------------|--------|----------|------------------|
| **BERT NER** | `protectai/bert-base-NER-onnx` | Trained by ProtectAI | HuggingFace Hub |
| **GLiNER** | `onnx-community/gliner_small-v2.1` | Trained by GLiNER authors | HuggingFace Hub |
| **GLiNER2** | `fastino/gliner2-base-v1` | Trained by Fastino Labs | HuggingFace Hub |
| **NuNER** | `deepanwa/NuNerZero_onnx` | Trained by NuMind | HuggingFace Hub |
| **W2NER** | `ljynlp/w2ner-bert-base` | Trained by W2NER authors | HuggingFace Hub |
| **ModernBERT** | `answerdotai/ModernBERT-base` | Trained by Answer.AI | HuggingFace Hub |
| **DeBERTa** | `microsoft/deberta-v3-base` | Trained by Microsoft | HuggingFace Hub |

**What we do:**
- Download pre-trained weights from HuggingFace
- Run inference via ONNX Runtime or Candle
- Cache models locally after first download

**What we don't do:**
- Train these models
- Modify their weights
- Fine-tune them (though the infrastructure could support it)

### Trainable Models (Implemented in This Codebase)

**Box Embeddings for Coreference Resolution** are the only trainable models in this codebase.

**Note**: This implementation is related to the **matryoshka-box** research project (not yet published). Box embeddings combine geometric representations (hyperrectangles) with logical invariants for coreference resolution.

| Model Type | Training Code | Training Data | Weights Location |
|------------|---------------|---------------|------------------|
| **Box Embeddings** | N/A (inference only) | N/A | N/A |

**Note**: Box embedding **training** is in `anno` (`src/backends/box_embeddings_training.rs`). The [matryoshka-box](https://github.com/arclabs561/matryoshka-box) research project extends this with matryoshka-specific features (variable dimensions, etc.).

**Training in anno:**
- Training code: `src/backends/box_embeddings_training.rs`
- Training examples: `examples/box_training.rs`, `examples/box_training_real_data.rs`
- Standard box embedding training with full evaluation integration

**Research extensions in matryoshka-box:**
- Extends `anno`'s training with matryoshka features (variable dimensions, hierarchical reasoning)
- See: https://github.com/arclabs561/matryoshka-box

**Inference in anno:**
- Box embeddings can be created from vectors: `BoxEmbedding::from_vector()`
- Coreference resolution: `BoxCorefResolver::resolve_with_boxes()`
- Trained boxes can be saved/loaded for inference

## Using Your Own Models (Bring Your Own)

### Backends with Local Path Support

**W2NER** and **T5Coref** support loading models from local file paths:

#### W2NER

```rust
use anno::backends::w2ner::W2NER;

// Use local model directory
let w2ner = W2NER::from_pretrained("/path/to/local/w2ner-model")?;

// Or use HuggingFace model ID (default)
let w2ner = W2NER::from_pretrained("ljynlp/w2ner-bert-base")?;
```

**Local directory structure:**
```
/path/to/local/w2ner-model/
├── model.onnx          # ONNX model file (or onnx/model.onnx)
└── tokenizer.json      # Tokenizer file
```

**How it works:**
- If the path exists as a directory, loads from local files
- Otherwise, downloads from HuggingFace Hub
- Automatically detects which to use

#### T5Coref (Coreference Resolution)

```rust
use anno::backends::coref_t5::{T5Coref, T5CorefConfig};

let config = T5CorefConfig::default();
let coref = T5Coref::from_path("/path/to/t5-coref-model", config)?;
```

**Local directory structure:**
```
/path/to/t5-coref-model/
├── encoder_model.onnx  # Encoder ONNX model
├── decoder_model.onnx  # Decoder ONNX model
└── tokenizer.json      # Tokenizer file
```

### Other Backends (HuggingFace Only)

Most other backends (BERT, GLiNER, NuNER, GLiNER2, etc.) currently only support HuggingFace model IDs. To use your own models:

1. **Upload to HuggingFace**: Upload your model to HuggingFace Hub, then use the model ID:
   ```rust
   let ner = GLiNEROnnx::new("your-username/your-custom-model")?;
   ```

2. **Use HuggingFace cache**: Models downloaded from HuggingFace are cached in `~/.cache/huggingface/hub/`. You can manually place files there, but this is not officially supported and may break with cache updates.

3. **Extend the backend**: The backends use `hf_hub::api::sync::Api` which could be extended to support local paths (similar to W2NER's implementation). See `src/backends/w2ner.rs` lines 202-206 for the pattern.

### Box Embeddings (Custom Training)

Box embeddings are trained in this codebase and can be saved/loaded:

```rust
use anno::backends::box_embeddings_training::{BoxEmbeddingTrainer, TrainingConfig};

// Train
let mut trainer = BoxEmbeddingTrainer::new(config, dim, None);
trainer.train(&examples);

// Save trained boxes
trainer.save_boxes("/path/to/boxes.json")?;

// Load later
let trainer = BoxEmbeddingTrainer::load_boxes("/path/to/boxes.json")?;
```

## Model Sources

**Default**: All pre-trained models are downloaded from **HuggingFace Hub** (`hf_hub` crate):
- ✅ HuggingFace model IDs (e.g., `"onnx-community/gliner_small-v2.1"`)
- ✅ Automatic caching in `~/.cache/huggingface/hub/`
- ✅ Authentication via `HF_TOKEN` environment variable (for gated models)

**Custom Models**:
- ✅ W2NER: Supports local file paths
- ✅ Box embeddings: Trained in-code, can be saved/loaded
- ⚠️ Other backends: HuggingFace Hub only (can be extended)

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

