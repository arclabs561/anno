# anno - Scope

NER for Rust. Also: coreference metrics, relation traits.

Task hierarchy:

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

## Scope

### Core Tasks (Implemented or In-Progress)

| Task | Status | Description |
|------|--------|-------------|
| **NER** | Mature | Extract entity mentions with type labels |
| **Pattern Extraction** | Mature | Regex-based structured entities (dates, money, emails) |
| **Coreference** | Metrics + Resolver | Link mentions to same real-world entity |
| **Relation Extraction** | Traits only | Extract (head, relation, tail) triples |
| **Discontinuous NER** | Stable | Handle non-contiguous spans (W2NER) |

### Future Tasks (Designed but not implemented)

| Task | Description |
|------|-------------|
| **Event Extraction** | Triggers + arguments for event types |
| **Knowledge Graphs** | Populate and query graph structures |
| **Multi-modal NER** | Entity extraction from images/documents |
| **Temporal NER** | Time expressions and temporal reasoning |
| **Nested NER** | Overlapping entity spans |

## Architecture

### Trait Hierarchy

```rust
// Base: Text → Spans
trait Model {
    fn extract_entities(&self, text: &str, lang: Option<&str>) -> Result<Vec<Entity>>;
}

// Zero-shot: Custom labels at runtime
trait ZeroShotNER: Model {
    fn extract_with_labels(&self, text: &str, labels: &[&str], threshold: f32) -> Result<Vec<Entity>>;
}

// Relation extraction
trait RelationExtractor: Model {
    fn extract_with_relations(&self, text: &str) -> Result<ExtractionWithRelations>;
}

// Coreference
trait CoreferenceResolver {
    fn resolve(&self, text: &str) -> Result<Vec<CorefChain>>;
}

// Discontinuous spans
trait DiscontinuousNER {
    fn extract_discontinuous(&self, text: &str, labels: &[&str]) -> Result<Vec<DiscontinuousEntity>>;
}

// Lexicon/Gazetteer (exact match)
trait Lexicon {
    fn lookup(&self, text: &str) -> Option<(EntityType, f64)>;
    fn source(&self) -> &str;
}
```

### Knowledge Sources

| Source | Purpose | When to Use |
|--------|---------|-------------|
| **TypeMapper** | Label normalization | Domain-specific datasets |
| **Lexicon** | Exact entity lookup | Closed domains |
| **SemanticRegistry** | Embedding-based lookup | Zero-shot NER |

### Backend Philosophy

1. **Zero-dependency baseline**: Always works, no external models
2. **ONNX for production**: Widely tested, cross-platform
3. **Candle for pure Rust**: Metal/CUDA without Python

### Evaluation Philosophy

1. **Task-specific metrics**: F1 for NER, MUC/B³/CEAF for coref
2. **Multiple modes**: Strict, Exact, Partial, Type-only
3. **Real datasets**: CoNLL-2003, WikiGold, WNUT-17, GAP, etc.
4. **Stratified sampling**: Proportional entity type coverage

## Research

| Paper | Concept | Implementation |
|-------|---------|----------------|
| GLiNER | Bi-encoder | `ZeroShotNER` |
| W2NER | Word-word | `DiscontinuousNER` |
| ModernBERT | RoPE | `TextEncoder` |
| UniversalNER | Cross-domain | `TypeMapper` |

## Non-Goals

- **Training**: We're inference-only. Use Python/JAX for training.
- **Tokenization**: We use HuggingFace tokenizers, not custom implementations.
- **Document parsing**: PDF/HTML extraction is out of scope. Feed us text.
- **LLM orchestration**: We don't wrap GPT-4 for extraction. Use dedicated tools.

## Maturity Levels

| Level | Meaning | Examples |
|-------|---------|----------|
| **Mature** | Battle-tested, stable API | PatternNER, StatisticalNER, StackedNER, Evaluation framework |
| **Stable** | Works, API may evolve | GLiNER, NuNER, W2NER, Coref metrics+resolver, BIO adapter, TypeMapper |
| **Experimental** | Functional, limited testing | Candle backend, LLM prompting, Demonstration selection |
| **Stub** | Traits/types only | RelationExtractor (traits), Event Extraction (types) |

## Roadmap

### v0.2 (Current - Released)
- [x] W2NER decoder implementation (ONNX backend with from_pretrained, grid decoding)
- [x] NuNER zero-shot token NER (ONNX backend with BIO decoding)
- [x] GLiNER span-based zero-shot NER (ONNX backend)
- [x] Coreference metrics (MUC, B³, CEAF-e/m, LEA, BLANC, CoNLL F1)
- [x] Rule-based coreference resolver (gender-aware, neopronoun support)
- [x] LLM prompting for NER (CodeNER-style prompts)
- [x] CMAS demonstration selection for few-shot learning
- [x] Improved synthetic dataset (115+ examples with strict offset validation)
- [x] Stratified evaluation harness with TypeMapper support
- [x] TypeMapper for domain-specific dataset normalization
- [x] 5 new datasets (FewNERD, CrossNER, UniversalNER, DocRED, Re-TACRED)
- [x] Lexicon trait and HashMapLexicon implementation
- [x] ExtractionMethod refinement (Pattern/Neural/Lexicon/SoftLexicon/GatedEnsemble)
- [x] Sealed Model trait pattern for API stability
- [x] Comprehensive bias analysis (gender, demographic, temporal, length)
- [x] Advanced evaluation (calibration, robustness, OOD detection, active learning)
- [x] BIO/IOB/IOBES adapter for sequence labeling compatibility

### v0.3 (Medium-term)
- [ ] Relation extraction with joint models
- [ ] Event extraction types and traits
- [ ] Knowledge graph builder
- [ ] Multi-document coreference

### v0.4+ (Long-term)
- [ ] Multi-modal NER (ColPali-style)
- [ ] Incremental/streaming extraction
- [ ] Active learning integration
- [ ] Custom model fine-tuning export

## Contributing

Focus areas for contributions:
1. **More synthetic test data** - High-quality annotated examples
2. **Dataset loaders** - New formats (BRAT, WebAnno, etc.)
3. **Benchmark results** - Reproduce academic benchmarks
4. **Documentation** - Usage examples, tutorials

