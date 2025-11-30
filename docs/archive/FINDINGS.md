# Research Findings: Models & Datasets

## GLiNER Safetensors Models Found

### ✅ Available with Safetensors:
1. **knowledgator/modern-gliner-bi-large-v1.0**
   - Has `model.safetensors` (confirmed via HuggingFace)
   - Large model, ModernBERT-based
   - **Status**: Updated as default in `DEFAULT_GLINER_CANDLE_MODEL`

2. **knowledgator/gliner-x-small**
   - May have safetensors (checking)
   - Smaller model, faster inference
   - **Status**: Alternative fallback

### ⚠️ Need Conversion:
- `urchade/gliner_small-v2.1` - Only pytorch_model.bin
- `urchade/gliner_medium-v2.1` - Only pytorch_model.bin
- `urchade/gliner_large-v2.1` - Only pytorch_model.bin

**Action**: Updated `DEFAULT_GLINER_CANDLE_MODEL` to use `knowledgator/modern-gliner-bi-large-v1.0`

---

## W2NER Alternatives

### Current Issue:
- `ljynlp/w2ner-bert-base` - Returns 401 (private/authenticated)

### Research Findings:
- W2NER paper: "Unified Named Entity Recognition as Word-Word Relation Classification" (AAAI 2022)
- Architecture: Word-word relation classification for nested/discontinuous NER
- No public ONNX alternatives found yet

### Potential Solutions:
1. **Search for reimplementations**: Look for community ports
2. **Alternative nested NER models**: 
   - T2-NER (two-stage span-based)
   - Triaffine-nested-ner
   - Other word-word relation models
3. **Document limitation**: W2NER requires authentication

**Action**: Keep graceful skip, document in benchmarks

---

## Nested/Discontinuous NER Datasets

### Standard Benchmarks (Already in Codebase):
- ✅ **GENIA**: Already supported (`DatasetId::GENIA`)
  - 2000 MEDLINE abstracts
  - Biomedical nested NER
  - Available via HuggingFace datasets-server

### Missing Benchmarks (Need to Add):

#### ACE 2004 / ACE 2005
- **Status**: LDC Licensed (requires purchase)
- **Use**: Standard nested NER benchmark
- **Alternative**: Some papers use processed versions
- **Action**: Document limitation, note LDC requirement

#### CADEC (Clinical Adverse Drug Events)
- **Status**: Research dataset, may be available
- **Use**: Discontinuous NER in clinical text
- **Size**: ~1000 training, ~379 test records
- **Action**: Search for public version or HuggingFace dataset

#### ShARe13 / ShARe14
- **Status**: Clinical entity recognition shared tasks
- **Use**: Discontinuous NER benchmarks
- **Action**: Search for public versions

### Dataset Sources Found:
- HuggingFace datasets-server API (for some datasets)
- Research paper repositories (processed versions)
- LDC catalog (licensed datasets)

---

## Recommended Next Steps

### Immediate (Models):
1. ✅ Update GLiNERCandle to use `knowledgator/modern-gliner-bi-large-v1.0`
2. ✅ Add fallback to `knowledgator/gliner-x-small`
3. ✅ Updated benchmark with fallback logic
4. ⏳ Search for W2NER alternatives or document limitation

### Short-term (Datasets):
1. ✅ Found CADEC on HuggingFace: `KevinSpaghetti/cadec` (ready to add)
2. ⏳ Add CADEC dataset support to DatasetLoader
3. ⏳ Add ShARe13/14 support (if public versions found)
4. ⏳ Document ACE 2004/2005 LDC requirement

### Long-term:
1. Support HuggingFace token authentication for private models
2. Add dataset conversion scripts for LDC datasets (if user has license)
3. Create evaluation suite specifically for nested/discontinuous NER

---

## Code Changes Made

1. **src/lib.rs**: Updated `DEFAULT_GLINER_CANDLE_MODEL` to `knowledgator/modern-gliner-bi-large-v1.0`
2. **benches/ner.rs**: Added fallback logic for GLiNERCandle
3. **examples/download_models.rs**: Updated Candle model list
4. **src/backends/gliner_candle.rs**: Updated error messages

---

## Testing Status

- ✅ Error conversion fixes working
- ✅ GLiNERCandle conversion logic ready
- ⏳ Need to test with `knowledgator/modern-gliner-bi-large-v1.0` (has safetensors)
- ⏳ W2NER: Document limitation or find alternative

