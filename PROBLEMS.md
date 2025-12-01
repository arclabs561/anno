# Remaining Issues to Fix

## Problem 1: GLiNERCandle - PyTorch to Safetensors Conversion

**Status:** ✅ **FIXED** - Automatic conversion using Python script with PEP 723 dependencies (via `uv run --script`)

**Root Cause:**
- GLiNER models (urchade/gliner_small-v2.1, knowledgator/gliner-x-small) only provide `pytorch_model.bin`
- Candle's `VarBuilder::from_mmaped_safetensors()` ONLY accepts safetensors format
- Cannot load PyTorch `.bin` files directly in Candle

**Location:**
- `src/backends/gliner_candle.rs:312-407` (conversion function)
- `scripts/convert_pytorch_to_safetensors.py` (PEP 723 script with inline dependencies)

**Solution Implemented:**
- ✅ Automatic conversion using `uv run --script` (falls back to `python3` if uv not available)
- ✅ Conversion script uses PEP 723 format with inline dependencies (torch, safetensors, packaging)
- ✅ Conversion is cached - subsequent loads use cached safetensors file
- ✅ All Candle backends now support conversion: GLiNERCandle, CandleNER, GLiNER2Candle, CandleEncoder
- ✅ Comprehensive e2e tests in `tests/conversion_e2e.rs`

**Implementation Details:**
- Uses Python's `torch.load()` and `safetensors.torch.save_file()` for conversion
- Script is self-contained with PEP 723 inline dependencies (no manual pip install needed)
- Conversion happens automatically when a model only has `pytorch_model.bin`
- Cached conversions stored as `model_converted.safetensors` in model cache directory
- Falls back to `python3` if `uv` is not available

**Backends with Conversion Support:**
1. **GLiNERCandle** - ✅ Automatic conversion
2. **CandleNER** - ✅ Automatic conversion
3. **GLiNER2Candle** - ✅ Automatic conversion
4. **CandleEncoder** - ✅ Automatic conversion

**Recommended Solutions (in order of preference):**
1. **Use GLiNEROnnx (ONNX backend)** - works with all GLiNER models, no conversion needed
2. **Automatic conversion** - Candle backends now automatically convert `pytorch_model.bin` to safetensors
3. **Use models with safetensors already available** - e.g., `knowledgator/modern-gliner-bi-large-v1.0`
4. **Manual conversion** - `uv run --script scripts/convert_pytorch_to_safetensors.py <input.bin> <output.safetensors>`

---

## Problem 2: W2NER - Authentication Required (401 Error)

**Status:** ✅ **FIXED** - Better error handling and documentation added

**Root Cause:**
- Model `ljynlp/w2ner-bert-base` requires authentication
- Returns HTTP 401 when accessing `onnx/model.onnx`
- Model may be private or gated

**Location:**
- `src/backends/w2ner.rs:214-234` (error handling)

**Solution Implemented:**
- ✅ Enhanced error message that detects 401 authentication errors
- ✅ Provides clear instructions for users with access:
  1. Get access to the model on HuggingFace
  2. Set `HF_TOKEN` environment variable
  3. Or use an alternative nested NER model
- ✅ Documents that W2NER is not available in public benchmarks due to authentication

**Error Message (Improved):**
```
W2NER model 'ljynlp/w2ner-bert-base' requires HuggingFace authentication (401 Unauthorized).
This model may be private or gated.
To use W2NER:
 1. Get access to the model on HuggingFace
 2. Set HF_TOKEN environment variable with your HuggingFace token
 3. Or use an alternative nested NER model.
Note: W2NER is currently not available in public benchmarks due to authentication requirements.
```

**Note:** The `hf-hub` crate should automatically use `HF_TOKEN` if set. The improved error message guides users on how to authenticate.

---

## Problem 3: NuNER Evaluation Returns 0% F1

**Status:** ✅ **FIXED** - Zero-shot backends now use dataset labels

**Root Cause:**
- NuNER model loads successfully and manual extraction works
- Evaluation framework was calling `Model::extract_entities()` which uses default labels: `["person", "organization", "location"]`
- Datasets use different labels (e.g., WikiGold uses `["PER", "LOC", "ORG", "MISC"]`)
- Label mismatch caused 0% F1 even though model was extracting entities correctly

**Location:**
- `src/eval/task_evaluator.rs:328-423` (`evaluate_ner_task`)
- `src/eval/task_evaluator.rs:465-563` (zero-shot backend creation and extraction)

**Solution Implemented:**
- ✅ Extracts entity types from dataset: `dataset.entity_types()`
- ✅ Maps dataset labels to model-compatible labels (e.g., "PER" → "person", "ORG" → "organization")
- ✅ For zero-shot backends (NuNER, GLiNER, GLiNER2, GLiNERCandle), creates cached backend instance and calls `extract(text, labels, threshold)` with dataset labels
- ✅ Caches backend instances to avoid recreating ONNX sessions for each sentence (fixes ONNX errors)

**Supported Zero-Shot Backends:**
- `nuner` - Uses `NuNER::extract(text, labels, threshold)`
- `gliner_onnx` / `gliner` - Uses `GLiNEROnnx::extract(text, labels, threshold)`
- `gliner2` - Uses `GLiNER2Onnx::extract(text, schema)` with `TaskSchema::new().with_entities(labels)`
- `gliner_candle` - Uses `GLiNERCandle::extract(text, labels, threshold)`

**Label Mapping:**
- Handles common variations: "PER"/"PERSON" → "person", "ORG"/"ORGANIZATION" → "organization", etc.
- See `map_dataset_labels_to_model()` function for full mapping logic

**See**: `docs/BENCHMARK_ANALYSIS.md` for full analysis

---

## Problem 4: Performance - Sequential Processing Bottleneck

**Status:** ✅ **FIXED** - Parallel processing added via `eval-parallel` feature

**Root Cause:**
- Evaluation framework processed sentences sequentially in a `for` loop
- No parallelization despite multi-core systems
- ONNX inference is CPU-bound but single-threaded
- Performance: ~0.7-0.9 examples/sec for ML backends

**Location:**
- `src/eval/task_evaluator.rs:362-392` (evaluation loop)

**Solution Implemented:**
- ✅ Added `rayon` dependency for parallel processing
- ✅ Created `eval-parallel` feature flag
- ✅ Implemented parallel sentence processing using `par_iter()`
- ✅ Maintains backward compatibility (sequential fallback when feature disabled)
- ✅ Added progress reporting (sentence count, percentage)

**Performance Improvement:**
- Expected 2-4x speedup on multi-core systems
- Parallel processing only when `eval-parallel` feature enabled
- Thread-safe: Backend sessions use Mutex (already thread-safe)

**Dependencies Added:**
- `rayon = "1"` (optional, via `eval-parallel` feature)

**Usage:**
```bash
# Build with parallel evaluation
cargo build --features onnx,eval-parallel

# Run benchmark (will use parallel processing)
./target/debug/anno benchmark --tasks ner --backends bert_onnx --datasets wikigold
```

---

## Problem 5: NuNER ONNX Missing Input: span_mask

**Status:** ✅ **FIXED** - Dynamic input detection and span tensor generation

**Root Cause:**
- Some NuNER ONNX models require `span_mask` and `span_idx` inputs
- Code only provided 4 inputs: `input_ids`, `attention_mask`, `words_mask`, `text_lengths`
- Missing inputs caused "Missing Input: span_mask" ONNX errors

**Location:**
- `src/backends/nuner.rs:313-325` (ONNX inference call)

**Solution Implemented:**
- ✅ Added `make_span_tensors()` static method (similar to GLiNER)
- ✅ Dynamic input detection: checks model requirements on load
- ✅ Generates span tensors only when model requires them
- ✅ Falls back to 4-input token mode if model doesn't need spans
- ✅ Added `MAX_SPAN_WIDTH` constant (12, matching GLiNER)

**Implementation Details:**
- Checks `session.inputs` to detect required inputs
- Generates `span_idx` and `span_mask` tensors when needed
- Maintains backward compatibility with token-only models

**See**: `tests/test_nuner_span_tensors.rs` for test coverage

---

## Summary & Action Plan

### ✅ All Issues Fixed:
1. **GLiNERCandle**: ✅ Automatic PyTorch to Safetensors conversion using Python script (via `uv run --script`)
2. **W2NER**: ✅ Better error handling and documentation for 401 authentication errors
3. **NuNER**: ✅ Evaluation fixed - zero-shot backends now use dataset labels (NuNER, GLiNER, GLiNER2, GLiNERCandle)

### Implementation Details:

**GLiNERCandle Conversion:**
- Uses Python script with PEP 723 inline dependencies (torch, safetensors)
- Calls `uv run --script` (or `python3` fallback) to convert PyTorch `.bin` to safetensors
- Script is self-contained - no manual pip install needed
- Caches converted files

**W2NER Authentication:**
- Enhanced error messages detect 401 errors
- Provides clear instructions for authentication
- Documents limitation in error message

**NuNER/Zero-Shot Evaluation:**
- Extracts entity types from datasets
- Maps dataset labels to model-compatible labels
- Caches backend instances to avoid ONNX session recreation
- Supports all zero-shot backends: NuNER, GLiNER, GLiNER2, GLiNERCandle

### Previously Fixed:
       - ✅ Error conversion for HuggingFace API errors (`From<ApiError>` implemented)
       - ✅ `task_evaluator.rs` feature gate issue (now handles both `eval` and `eval-advanced` features)
       - ✅ README updated to reflect coreference resolution capabilities
       - ✅ Compilation errors resolved
       - ✅ HybridNER removed (replaced with StackedNER)
       - ✅ Comprehensive benchmark completed (see `docs/BENCHMARK_ANALYSIS.md`)
       - ✅ Performance: Parallel processing added (2-4x speedup expected)
       - ✅ NuNER ONNX: Dynamic span tensor generation for models that require it

### Files Modified:
1. ✅ `src/backends/gliner_candle.rs` - Automatic conversion using Python script (uv run --script)
2. ✅ `src/backends/w2ner.rs` - Enhanced 401 error handling
3. ✅ `src/eval/task_evaluator.rs` - Zero-shot backend support with dataset labels
4. ✅ `scripts/convert_pytorch_to_safetensors.py` - PEP 723 script with inline dependencies

---

## Problem 6: Backend Harmonization - Inconsistent Trait Coverage

**Status:** ✅ **FIXED** - All backends now have consistent trait implementations

**Root Cause:**
- Many backends could implement capability traits (BatchCapable, StreamingCapable, GpuCapable) but didn't
- Catalog had incorrect status/feature flags
- Dead traits (CalibratedConfidence, VisualCapable) were defined but never used
- Encoder traits were duplicated without clear documentation

**Location:**
- Multiple backend files
- `src/lib.rs` (trait definitions)
- `src/backends/catalog.rs` (status tracking)

**Solution Implemented:**
- ✅ **GpuCapable**: Implemented for all Candle backends (GLiNERCandle, CandleNER, GLiNER2Candle)
- ✅ **BatchCapable**: Added to NuNER, W2NER, BertNEROnnx, CandleNER, HeuristicNER
- ✅ **StreamingCapable**: Added to NuNER, W2NER, BertNEROnnx, CandleNER, HeuristicNER
- ✅ **DiscontinuousNER**: Verified existing implementation for W2NER
- ✅ **Catalog fixes**: GLiNERCandle (WIP → Beta), NuNER (feature flag and status corrected)
- ✅ **Dead traits removed**: CalibratedConfidence and VisualCapable (documented for future re-addition)
- ✅ **Encoder trait documentation**: Created `docs/ENCODER_TRAIT_DESIGN.md` explaining dual trait design
- ✅ **Comprehensive tests**: Added `tests/trait_harmonization_tests.rs` with 20+ property-based and integration tests

**Trait Coverage Matrix (Final State):**

| Backend | Model | BatchCapable | StreamingCapable | GpuCapable | DynamicLabels | RelationCapable | DiscontinuousNER |
|---------|-------|--------------|------------------|------------|---------------|-----------------|------------------|
| PatternNER | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| HeuristicNER | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| StackedNER | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| GLiNEROnnx | ✅ | ✅ | ✅ | ❌* | ✅ | ❌ | ❌ |
| GLiNERCandle | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| GLiNER2Onnx | ✅ | ✅ | ✅ | ❌* | ✅ | ✅ | ❌ |
| GLiNER2Candle | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| NuNER | ✅ | ✅ | ✅ | ❌* | ✅ | ❌ | ❌ |
| W2NER | ✅ | ✅ | ✅ | ❌* | ❌ | ✅ | ✅ |
| BertNEROnnx | ✅ | ✅ | ✅ | ❌* | ❌ | ❌ | ❌ |
| CandleNER | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |

*ONNX backends could implement GpuCapable by checking execution providers (deferred)

**Test Coverage:**
- ✅ 20+ property-based tests (proptest) for trait invariants
- ✅ Integration tests for trait combinations
- ✅ Edge case tests (empty strings, large offsets, batch size consistency)
- ✅ All tests passing (20/20)

**Files Modified:**
1. ✅ `src/backends/catalog.rs` - Fixed status and feature flags
2. ✅ `src/backends/gliner_candle.rs` - Added GpuCapable
3. ✅ `src/backends/candle.rs` - Added GpuCapable, BatchCapable, StreamingCapable
4. ✅ `src/backends/gliner2.rs` - Added GpuCapable for Candle version
5. ✅ `src/backends/nuner.rs` - Added BatchCapable, StreamingCapable
6. ✅ `src/backends/w2ner.rs` - Added BatchCapable, StreamingCapable
7. ✅ `src/backends/onnx.rs` - Added BatchCapable, StreamingCapable
8. ✅ `src/backends/heuristic.rs` - Added BatchCapable, StreamingCapable
9. ✅ `src/lib.rs` - Removed dead traits (CalibratedConfidence, VisualCapable)
10. ✅ `tests/trait_harmonization_tests.rs` - New comprehensive test suite
11. ✅ `docs/HARMONIZATION_PLAN.md` - Design document
12. ✅ `docs/HARMONIZATION_SUMMARY.md` - Implementation summary
13. ✅ `docs/ENCODER_TRAIT_DESIGN.md` - Encoder trait documentation
14. ✅ `docs/BACKEND_INTERFACE_REVIEW.md` - Comprehensive review

**See**: `docs/HARMONIZATION_SUMMARY.md` for full details


