# Examples

### Running

```bash
# Pattern extraction (no model downloads)
cargo run --example quickstart

# With evaluation framework
cargo run --example quickstart --features eval
cargo run --example eval_basic --features eval   # Simple backend comparison
cargo run --example eval --features eval         # Full evaluation suite

# Transformer models (downloads ~400MB on first run)
cargo run --example bert --features onnx
cargo run --example models --features onnx

# Full benchmark suite
cargo run --example benchmark --features eval-full

# Discourse analysis (abstract anaphora, events)
cargo run --example abstract_anaphora_eval --features discourse
cargo run --example discourse_pipeline --features discourse

# GLiNER2 multi-task extraction (NER, classification, relations)
cargo run --example gliner2_multitask --features onnx

# Production setup (async, session pooling)
cargo run --example production --features production

# Grounded entities (Signal → Track → Identity)
cargo run --example grounded
```

### What's here

| Example | Features | Description |
|---------|----------|-------------|
| `quickstart` | — | Basic span extraction with RegexNER |
| `eval_basic` | eval | **Start here!** Simple backend comparison |
| `eval` | eval | Full evaluation on synthetic and real datasets |
| `bert` | onnx | BERT-based NER |
| `models` | onnx | All backends including zero-shot GLiNER |
| `coref` | eval | Coreference metrics (MUC, B³, CEAF, LEA) |
| `bias` | eval-bias | Gender and demographic bias analysis |
| `benchmark` | eval-full | Combined quality/bias/robustness evaluation |
| `advanced` | eval | Discontinuous spans, relation extraction |
| `hybrid` | onnx | Combining transformer + pattern backends |
| `candle` | candle | Pure Rust inference backend |
| `abstract_anaphora_eval` | discourse | Abstract anaphora resolution evaluation |
| `discourse_pipeline` | discourse | Event extraction + shell noun analysis |
| `gliner2_multitask` | onnx | Multi-task extraction: NER, classification, relations |
| `production` | production | Async inference, session pooling, warmup |
| `gliner_candle` | candle | GLiNER via pure Rust Candle backend |
| `download_models` | onnx | Download and cache HuggingFace models |
| `grounded` | — | Signal → Track → Identity entity hierarchy |

### Available Synthetic Datasets

The library includes 25+ synthetic datasets for testing and development:

**Core Domains:**
- `news_dataset()` - News articles (CoNLL-2003 style)
- `biomedical_dataset()` - Medical/clinical text
- `financial_dataset()` - Finance and markets
- `legal_dataset()` - Legal documents
- `scientific_dataset()` - Research papers
- `social_media_dataset()` - Tweets and posts

**Industry-Specific:**
- `technology_dataset()` - AI/tech companies
- `healthcare_dataset()` - Medical entities
- `manufacturing_dataset()` - Semiconductor/industrial
- `automotive_dataset()` - EV/automotive
- `energy_dataset()` - Energy/climate
- `aerospace_dataset()` - Aerospace/defense

**Specialized:**
- `sports_dataset()`, `politics_dataset()`, `ecommerce_dataset()`
- `travel_dataset()`, `weather_dataset()`, `food_dataset()`
- `cybersecurity_dataset()`, `real_estate_dataset()`, `academic_dataset()`
- `multilingual_dataset()`, `globally_diverse_dataset()`

```rust
use anno::eval::synthetic::{all_datasets, technology_dataset, Domain};

// Get all datasets
let all = all_datasets();  // ~200+ examples

// Or specific domains
let tech = technology_dataset();
```

### Zero-shot NER

Standard NER models only recognize entity types from their training data. GLiNER lets you specify types at runtime:

```rust
use anno::GLiNEROnnx;

let ner = GLiNEROnnx::new("onnx-community/gliner_small-v2.1")?;

let entities = ner.extract(
    "Patient presents with hypertension, prescribed lisinopril",
    &["condition", "medication"],
    0.4,
)?;
```

See `models.rs` for more examples.
