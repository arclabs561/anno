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

// Zero-shot: entity types specified at runtime
trait ZeroShotNER: Model {
    fn extract_with_labels(&self, text: &str, labels: &[&str], threshold: f32) -> Result<Vec<Entity>>;
}

// Relation extraction: joint entity + relation
trait RelationExtractor: Model {
    fn extract_with_relations(
        &self,
        text: &str,
        entity_types: &[&str],
        relation_types: &[&str],
        threshold: f32,
    ) -> Result<ExtractionWithRelations>;
}

// Coreference: mention clustering
trait CoreferenceResolver {
    fn resolve(&self, text: &str) -> Result<Vec<CorefChain>>;
}

// Discontinuous spans (W2NER-style)
trait DiscontinuousNER {
    fn extract_discontinuous(&self, text: &str, labels: &[&str]) -> Result<Vec<DiscontinuousEntity>>;
}
```

### Backend philosophy

1. **Zero-dependency default**: `PatternNER` and `StackedNER` require no model downloads.
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

**Mature**: PatternNER, StackedNER, evaluation framework.  
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
