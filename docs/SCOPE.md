# anno - Scope & Roadmap

## What This Library Does

**anno** extracts structured data from text. Primarily NER, with support for related tasks.

Text annotation tasks form a natural hierarchy:

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

Each layer builds on the previous. NER finds entity mentions; coreference links mentions
to real-world entities; relation extraction connects entities; knowledge graphs organize
it all into queryable structures.

## Scope

### Core Tasks (Implemented or In-Progress)

| Task | Status | Description |
|------|--------|-------------|
| **NER** | Mature | Extract entity mentions with type labels |
| **Pattern Extraction** | Mature | Regex-based structured entities (dates, money, emails) |
| **Coreference** | Metrics only | Link mentions to same real-world entity |
| **Relation Extraction** | Traits only | Extract (head, relation, tail) triples |
| **Discontinuous NER** | Stub | Handle non-contiguous entity spans |

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

### Knowledge Sources (Research-Aligned)

| Source | Purpose | When to Use | Research |
|--------|---------|-------------|----------|
| **TypeMapper** | Label normalization | Domain-specific datasets | — |
| **Lexicon** | Exact entity lookup | Closed domains only | Song et al. 2020 |
| **SemanticRegistry** | Embedding-based lookup | Zero-shot NER | GLiNER 2024 |

See `docs/LEXICON_DESIGN.md` for detailed research context on lexicon integration.

### Backend Philosophy

1. **Zero-dependency baseline**: Always works, no external models
2. **ONNX for production**: Widely tested, cross-platform
3. **Candle for pure Rust**: Metal/CUDA without Python

### Evaluation Philosophy

1. **Task-specific metrics**: F1 for NER, MUC/B³/CEAF for coref
2. **Multiple modes**: Strict, Exact, Partial, Type-only
3. **Real datasets**: CoNLL-2003, WikiGold, WNUT-17, GAP, etc.
4. **Stratified sampling**: Proportional entity type coverage

## Research Papers

The trait system draws from recent NER/IE research:

| Paper | Year | Concept | Implementation |
|-------|------|---------|----------------|
| GLiNER (NAACL) | 2024 | Bi-encoder matching | `BiEncoder`, `LateInteraction` traits |
| ModernBERT | 2024 | RoPE, unpadding | `RaggedBatch`, encoder configs |
| W2NER (AAAI) | 2022 | Word-word relations | `HandshakingMatrix`, `DiscontinuousNER` |
| UniversalNER (ICLR) | 2024 | Cross-domain zero-shot | `ZeroShotNER`, `SemanticRegistry` |
| ColPali | 2024 | Vision-native retrieval | `ModalityInput`, `VisualCapable` |
| **ReasoningNER** | 2025 | CoT + GRPO for NER | Future: `ReasoningNER` trait |
| **CMAS** | 2025 | Multi-agent zero-shot | Future: agent orchestration |
| GEMNET (NAACL) | 2021 | Gated gazetteer | `GatedEnsemble`, `Lexicon` trait |
| Soft Gazetteers (ACL) | 2020 | Embedding lookup | `SoftLexicon` (planned) |
| Familiarity | 2024 | Label shift bias | `LabelShift` type |

### Key 2025 Findings (bleeding edge)

| Paper | F1 | Key Insight |
|-------|-----|-------------|
| ReasoningNER | 85.2 | CoT reasoning improves NER by 1.3+ F1 points |
| BioClinical ModernBERT | SOTA | Domain adaptation + long context = gains |
| NER Retriever | — | Retrieval-based approach for ad-hoc entity types |

## Non-Goals

- **Training**: We're inference-only. Use Python/JAX for training.
- **Tokenization**: We use HuggingFace tokenizers, not custom implementations.
- **Document parsing**: PDF/HTML extraction is out of scope. Feed us text.
- **LLM orchestration**: We don't wrap GPT-4 for extraction. Use dedicated tools.

## Maturity Levels

| Level | Meaning | Examples |
|-------|---------|----------|
| **Mature** | Battle-tested, stable API | NER, PatternNER, StackedNER |
| **Stable** | Works, API may evolve | Evaluation, synthetic datasets, TypeMapper |
| **Experimental** | Functional, limited testing | ONNX backends, Candle backend, coref metrics |
| **Stub** | Traits/types only | RelationExtractor, DiscontinuousNER, W2NER |

## Roadmap

### v0.2 (Near-term)
- [ ] Full W2NER decoder implementation
- [ ] Coreference model integration (via ONNX)
- [x] Improved synthetic dataset (115+ examples with strict offset validation)
- [x] Stratified evaluation harness with TypeMapper support
- [x] TypeMapper for domain-specific dataset normalization
- [x] 5 new datasets (FewNERD, CrossNER, UniversalNER, DocRED, Re-TACRED)
- [x] Lexicon trait and HashMapLexicon implementation
- [x] ExtractionMethod refinement (Pattern/Neural/Lexicon/SoftLexicon/GatedEnsemble)
- [x] Sealed Model trait pattern for API stability

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

