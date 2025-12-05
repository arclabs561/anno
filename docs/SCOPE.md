# Scope

This crate does named entity recognition and related tasks. The hierarchy of what's implemented:

```
                    Knowledge Graphs
                          ↑
                   Relation Extraction
                    ↙           ↘
            Coreference    Event Extraction
                    ↘           ↙
              Named Entity Recognition
                          ↑
                   Pattern Matching
```

### What's implemented

| Task | Status |
|------|--------|
| Span detection + entity typing | Mature. Multiple backends. |
| Pattern extraction (dates, money, etc.) | Mature. Regex-based. |
| Coreference metrics | Stable. MUC, B³, CEAF, LEA, BLANC. |
| Coreference resolution | Basic rule-based resolver. |
| Discontinuous NER | Stable. W2NER-style grid decoding. |
| Relation extraction | TPLinker placeholder (heuristic-based). Full ONNX model pending. |

### What's not implemented

Event extraction, knowledge graph construction, nested NER (overlapping spans), and document-level coreference are out of scope for now.

### Trait hierarchy

```rust
// Base trait - all backends implement this
trait Model {
    fn extract_entities(&self, text: &str, lang: Option<&str>) -> Result<Vec<Entity>>;
}

// Zero-shot: entity types specified at runtime (type hints)
trait ZeroShotNER: Send + Sync {
    fn extract_with_types(&self, text: &str, entity_types: &[&str], threshold: f32) -> Result<Vec<Entity>>;
    fn extract_with_descriptions(&self, text: &str, descriptions: &[&str], threshold: f32) -> Result<Vec<Entity>>;
}

// Relation extraction: joint entity + relation
trait RelationExtractor: Send + Sync {
    fn extract_with_relations(
        &self,
        text: &str,
        entity_types: &[&str],
        relation_types: &[&str],
        threshold: f32,
    ) -> Result<ExtractionWithRelations>;
}

// Coreference: mention clustering
trait CoreferenceResolver: Send + Sync {
    fn resolve(&self, entities: &[Entity]) -> Vec<Entity>;
}

// Discontinuous spans (W2NER-style)
trait DiscontinuousNER: Send + Sync {
    fn extract_discontinuous(&self, text: &str, labels: &[&str]) -> Result<Vec<DiscontinuousEntity>>;
}

// Gazetteer/lexicon lookup (separate from NER models)
trait Lexicon: Send + Sync {
    fn lookup(&self, text: &str) -> Option<(EntityType, f64)>;
    fn contains(&self, text: &str) -> bool;
    fn source(&self) -> &str;
}
```

**Type hints vs Gazetteers:**
- **Type hints** (`ZeroShotNER::extract_with_types`): Tell model WHAT types to extract (semantic matching). Example: `["person", "organization"]` → model extracts only those types.
- **Gazetteers** (`Lexicon` trait): Exact-match lookup of known entities. Example: `"AAPL"` → `Organization`. Currently defined but not integrated into NER pipeline (see `docs/LEXICON_DESIGN.md`).

### Backend philosophy

1. **Zero-dependency default**: `RegexNER` and `StackedNER` require no model downloads.
2. **ONNX for production**: Cross-platform, widely tested, good performance.
3. **Candle for pure Rust**: Metal/CUDA without Python dependencies.

### Research basis

This library primarily implements existing research. See [RESEARCH.md](RESEARCH.md) for a detailed breakdown of what's novel versus implementation.

| Paper | What we use |
|-------|-------------|
| GLiNER | Bi-encoder for zero-shot span classification |
| W2NER | Word-word grid for discontinuous spans |
| UniversalNER | Cross-domain type normalization (placeholder) |

**Note**: The ONNX backends are integration work, not novel implementations. Our main contributions are architectural design and unified evaluation framework integration.

### Ecosystem positioning

Other Rust NER libraries:
- [rust-bert](https://github.com/guillaume-be/rust-bert): Full transformer implementations via tch-rs, many NLP tasks
- [gline-rs](https://github.com/fbilhaut/gline-rs): Focused GLiNER inference with detailed pipeline documentation

This library's niche:
- Unified trait across regex/heuristics/ML (swap backends without code changes)
- Zero-dependency baselines for fast iteration
- Evaluation framework (unique in Rust NER)
- Coreference metrics (MUC, B³, CEAF, LEA)

The ONNX backends are integration work, not novel implementations.

### Non-goals

- **Training**: This is inference-only. Train your models in Python.
- **Tokenization**: We use HuggingFace tokenizers, not custom implementations.
- **Document parsing**: Feed us text, not PDFs or HTML.

### Maturity levels

| Level | Meaning |
|-------|---------|
| Mature | Stable API, well tested |
| Stable | Works, API may evolve |
| Experimental | Limited testing |
| Stub | Types/traits only |

**Mature**: RegexNER, StackedNER, evaluation framework.  
**Stable**: GLiNER, NuNER, W2NER, coref metrics, BIO adapter.  
**Experimental**: Candle backend, LLM prompting.  
**Stub**: RelationExtractor trait.

### Roadmap

**v0.2 (current)**: W2NER, NuNER, GLiNER, coreference metrics/resolver, bias analysis, calibration evaluation, BIO adapter, TypeMapper.

**v0.3**: Relation extraction models, event extraction types.

**v0.4+**: Multi-modal NER, streaming extraction.

### Contributing

Useful areas:
- More annotated test data
- Dataset loaders (BRAT, WebAnno formats)  
- Benchmark reproductions
- Documentation improvements
