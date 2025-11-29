# Remaining Issues to Fix

## Problem 1: GLiNERCandle - PyTorch to Safetensors Conversion

**Root Cause:**
- GLiNER models (urchade/gliner_small-v2.1, knowledgator/gliner-x-small) only provide `pytorch_model.bin`
- Candle's `VarBuilder::from_mmaped_safetensors()` ONLY accepts safetensors format
- Cannot load PyTorch `.bin` files directly in Candle

**Location:**
- `src/backends/gliner_candle.rs:395-402`
- `src/backends/gliner_candle.rs:294-364` (conversion function)

**Current Code:**
```rust
let weights_path = repo
    .get("model.safetensors")
    .or_else(|_| repo.get("gliner_model.safetensors"))
    .or_else(|_| {
        // Workaround: Try to convert pytorch_model.bin to safetensors
        // Error conversion is now fixed (From<ApiError> impl exists)
        let pytorch_path = repo.get("pytorch_model.bin")?;
        convert_pytorch_to_safetensors(&pytorch_path)
    })
```

**Status:**
- ✅ Error conversion fixed: `From<hf_hub::api::sync::ApiError>` implemented in `src/error.rs:117`
- ⚠️ Conversion function exists but requires Python dependencies (`torch`, `safetensors`)

**Conversion Function (Already Written):**
- Location: `src/backends/gliner_candle.rs:294-364`
- Uses Python with `safetensors` and `torch` libraries
- Converts `pytorch_model.bin` → `model_converted.safetensors`
- Caches result to avoid re-conversion

**Solutions:**

### Option A: Install Python Dependencies
The conversion function is already implemented. Just install required Python packages:
```bash
pip install torch safetensors
```

### Option B: Standalone Conversion Script
Create `scripts/convert_pytorch_to_safetensors.py`:
```python
#!/usr/bin/env python3
"""Convert pytorch_model.bin to model.safetensors"""
import sys
import torch
from safetensors.torch import save_file
from pathlib import Path

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: convert_pytorch_to_safetensors.py <input.bin> <output.safetensors>")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    
    state_dict = torch.load(input_path, map_location="cpu")
    save_file(state_dict, output_path)
    print(f"Converted {input_path} -> {output_path}")
```

Then call from Rust:
```rust
Command::new("python3")
    .arg("scripts/convert_pytorch_to_safetensors.py")
    .arg(&pytorch_path)
    .arg(&safetensors_path)
    .status()?;
```

### Option C: Pre-convert and Host
- Convert GLiNER models once
- Host safetensors versions somewhere accessible
- Update model paths to use converted versions

### Option D: Pure Rust Conversion
- Research if `safetensors` Rust crate can read PyTorch format
- Likely requires `pickle` parser (complex)

**Recommended:** Option A (install Python deps) - conversion logic already works

---

## Problem 2: W2NER - Authentication Required (401 Error)

**Root Cause:**
- Model `ljynlp/w2ner-bert-base` requires authentication
- Returns HTTP 401 when accessing `onnx/model.onnx`
- Model may be private or gated

**Location:**
- `src/backends/w2ner.rs:214-217`
- `benches/ner.rs:36-37` (model constant)

**Current Code:**
```rust
const W2NER_MODEL: &str = "ljynlp/w2ner-bert-base";

let model_file = repo
    .get("model.onnx")
    .or_else(|_| repo.get("onnx/model.onnx"))
    .map_err(|e| Error::Retrieval(format!("Failed to download model: {}", e)))?;
```

**Error Message:**
```
[Bench] W2NER failed to load: Retrieval error: Failed to download model: 
request error: https://huggingface.co/ljynlp/w2ner-bert-base/resolve/main/onnx/model.onnx: 
status code 401 (skipping)
```

**Solutions:**

### Option A: Find Public Alternative
Search for:
- Public W2NER ONNX models
- Alternative nested/discontinuous NER models
- Models with similar architecture (word-word relations)

**Search Queries:**
- "W2NER ONNX model huggingface public"
- "nested NER ONNX model"
- "word-word relation NER ONNX alternative"

### Option B: Support HuggingFace Authentication
Add token support to `hf-hub` API:
```rust
let api = Api::new()?;
// If HF_TOKEN env var set, use it
if let Ok(token) = std::env::var("HF_TOKEN") {
    // Configure API with token
}
```

### Option C: Document Limitation
- Mark W2NER as "requires authentication"
- Provide instructions for users with access
- Skip in benchmarks (already done)

**Recommended:** Option A (find alternative) + Option C (document)

---

## Summary & Action Plan

### Active Issues:
1. **GLiNERCandle**: PyTorch to Safetensors conversion needed (Python dependencies required)
2. **W2NER**: Authentication required (401 error) - find alternative or document limitation

### Recently Fixed:
- ✅ Error conversion for HuggingFace API errors (`From<ApiError>` implemented)
- ✅ `task_evaluator.rs` feature gate issue (now handles both `eval` and `eval-advanced` features)
- ✅ README updated to reflect coreference resolution capabilities
- ✅ Compilation errors resolved

### Files to Modify:
1. `src/backends/gliner_candle.rs` - Conversion function exists, requires Python deps
2. `benches/ner.rs` - W2NER already skipped in benchmarks (handled gracefully)
3. `examples/download_models.rs` - Update model lists if needed


