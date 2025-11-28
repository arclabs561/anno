# Multimodal & Future-Looking Trait Design: Gap Analysis and Roadmap

This document analyzes the current state of multimodal and advanced trait implementations
in the `anno` crate, identifies gaps in end-to-end tests and evaluations, and proposes
a concrete implementation roadmap.

## Executive Summary

| Trait | Definition | Implementation | Unit Tests | E2E/Eval | Status |
|-------|------------|----------------|------------|----------|--------|
| `Model` | `lib.rs` | All backends | Extensive | Harness | Mature |
| `ZeroShotNER` | `inference.rs` | GLiNER (ONNX) | Basic | **Added** | Stable |
| `RelationExtractor` | `inference.rs` | Heuristic only | Integration | **Added** | Stub |
| `DiscontinuousNER` | `inference.rs` | W2NER | Property tests | **Added** | Stable |
| `VisualCapable` | `lib.rs` | None | `ImageFormat` only | **Added** | Scaffold |
| `BiEncoder` | `inference.rs` | None | None | None | Design only |
| `TextEncoder` | `inference.rs` | None | None | None | Design only |
| `LabelEncoder` | `inference.rs` | None | None | None | Design only |
| `LateInteraction` | `inference.rs` | DotProduct, MaxSim | Property tests | None | Partial |

**Update (Nov 2024)**: Evaluation modules and synthetic datasets have been added for
discontinuous NER, relation extraction, and visual NER. See [Implemented Evaluation Modules](#implemented-evaluation-modules).

## Implemented Evaluation Modules

The following evaluation infrastructure has been added:

### Discontinuous NER Evaluation (`eval/discontinuous.rs`)

```rust
pub fn evaluate_discontinuous_ner(
    gold: &[DiscontinuousGold],
    pred: &[DiscontinuousEntity],
    config: &DiscontinuousEvalConfig,
) -> DiscontinuousNERMetrics;
```

**Metrics**:
- Exact F1 (all spans must match exactly)
- Entity Boundary F1 (head and tail tokens correct)
- Partial Span F1 (overlap-based matching)
- Per-type breakdown

**Synthetic Data** (`eval/dataset/synthetic/discontinuous.rs`):
- Easy: "X and Y Z" coordination patterns
- Medium: Multiple coordinations
- Hard: Nested and complex structures
- Biomedical domain (left/right ventricle, etc.)
- Legal domain (paragraphs, sections)

### Relation Extraction Evaluation (`eval/relation.rs`)

```rust
pub fn evaluate_relations(
    gold: &[RelationGold],
    pred: &[RelationPrediction],
    config: &RelationEvalConfig,
) -> RelationMetrics;
```

**Metrics**:
- Boundary F1 (Rel): Entity spans + relation type
- Strict F1 (Rel+): Exact span + type match
- Per-relation breakdown

**Synthetic Data** (`eval/dataset/synthetic/relations.rs`):
- FOUNDED, WORKS_FOR, LOCATED_IN, CEO_OF relations
- Business, scientific, biographical domains
- Easy (single relation), Medium (multiple), Hard (implicit)

### Visual NER Evaluation (`eval/visual.rs`)

```rust
pub fn evaluate_visual_ner(
    gold: &[VisualGold],
    pred: &[VisualPrediction],
    config: &VisualEvalConfig,
) -> VisualNERMetrics;
```

**Metrics**:
- Text F1 (text content match, ignoring boxes)
- Box IoU (Intersection-over-Union of bounding boxes)
- End-to-End F1 (correct text AND box)
- Per-type breakdown

**Synthetic Data**: Invoice, receipt, and document examples with bounding boxes.

### Advanced Evaluator Infrastructure (`eval/advanced_evaluator.rs`)

Unified evaluator interface for all task types:

```rust
pub fn evaluator_for_task(task: &EvalTask) -> Box<dyn TaskEvaluator>;

// Unified results enum
pub enum EvalResults {
    NER { ... },
    Discontinuous(DiscontinuousNERMetrics),
    Relation(RelationMetrics),
    Coreference { ... },
    Event { ... },
}
```

### Test Coverage

All evaluation modules have:
- Unit tests for metric calculations
- Property tests for invariants
- Integration tests in `tests/advanced_trait_tests.rs`

## Current State Analysis

### 1. VisualCapable Trait (Critical Gap)

**Location**: `src/lib.rs` (lines 117-123)

```rust
pub trait VisualCapable: Model {
    fn extract_from_image(&self, image_data: &[u8], format: &str) -> Result<Vec<Entity>>;
}
```

**Status**: Trait defined, no implementations.

**Supporting Types** (in `inference.rs`):
- `ModalityInput<'a>` - Text/Image/Hybrid enum
- `ImageFormat` - PNG/JPEG/WebP/Unknown
- `VisualPosition` - Bounding box for token locations

**Test Coverage**: Only `ImageFormat::default()` has a unit test.

**Research Alignment**: The design targets ColPali-style visual document understanding,
but no actual ColPali integration exists.

### 2. ZeroShotNER Trait (Partial Implementation)

**Location**: `src/backends/inference.rs` (lines 893-923)

**Implementations**:
- `GLiNER` via `gliner_onnx.rs` - Uses `extract_with_types()` method
- `GLiNERPipeline` via `gliner_pipeline.rs` - Wrapper interface

**Test Coverage**:
- Unit tests for supporting types (`SpanLabelScore`, `EncoderOutput`)
- Integration tests in `tests/backend_nuner_w2ner.rs` are `#[ignore]`
- No dedicated zero-shot evaluation framework

**Gap**: Need evaluation on standard zero-shot benchmarks (MIT Movie, MIT Restaurant,
CrossNER, etc.) with familiarity analysis per arXiv:2412.10121.

### 3. RelationExtractor Trait (Minimal Implementation)

**Location**: `src/backends/inference.rs` (lines 1002-1020)

**Current Implementation**: Only heuristic pattern matching in `extract_relations()`:
- Detects triggers like "founded", "works for", "located in"
- Requires entity types to match expected patterns (Person-Organization, etc.)
- No learned model integration

**Test Coverage**: One integration test in `inference_tests.rs`.

**Gap**: No W2NER-style joint entity-relation extraction despite `HandshakingMatrix`
being fully implemented.

### 4. DiscontinuousNER Trait (Good Coverage, Missing Eval)

**Location**: `src/backends/inference.rs` (lines 1110-1126)

**Implementation**: `W2NER` in `src/backends/w2ner.rs` implements this trait.

**Test Coverage**:
- Property tests for `DiscontinuousEntity` in `inference_tests.rs`
- Integration tests for W2NER decoding in `backend_nuner_w2ner.rs`
- Tests include single-token, multi-token, nested, and invalid cases

**Gap**: `EvalTask::DiscontinuousNER` is defined but no:
- Synthetic data generator for discontinuous entities
- Metrics calculation for discontinuous F1
- Benchmark datasets (CADEC, ShARe13, ShARe14)

### 5. Bi-Encoder Architecture (Design Only)

**Traits Defined** (in `inference.rs`):
- `TextEncoder` - Encode text → embeddings
- `LabelEncoder` - Encode labels → embeddings  
- `BiEncoder` - Combine text + label encoding
- `LateInteraction` - Compute span-label similarity

**Status**: Well-documented with research references, but no implementations.
Only `DotProductInteraction` and `MaxSimInteraction` have basic implementations.

## Evaluation Framework Coverage

### EvalTask Coverage (Updated)

| Task | Defined | Metrics | Synthetic Data | Benchmark |
|------|---------|---------|----------------|-----------|
| `NER` | Yes | Full | Yes | CoNLL, WikiGold |
| `Coreference` | Yes | Full | Yes | GAP |
| `RelationExtraction` | Yes | **Full** | **Yes** | DocRED (planned) |
| `DiscontinuousNER` | Yes | **Full** | **Yes** | CADEC (planned) |
| `EventExtraction` | Yes | Stub | None | None |
| Visual NER | **Yes** | **Full** | **Yes** | FUNSD (planned) |

### Remaining Evaluation Gaps

1. **Relation Extraction Metrics**
   - Boundary evaluation (Rel): entity boundaries + relation type
   - Strict evaluation (Rel+): exact entity match + relation type
   - Per-relation-type breakdown

2. **Discontinuous NER Metrics**
   - Entity Boundary F1 (EBF) per W2NER paper
   - Multi-span overlap handling
   - Partial credit for partial span recovery

3. **Visual NER Metrics**
   - Bounding box IoU for visual entities
   - OCR quality impact analysis
   - Cross-modal grounding accuracy

## Implementation Roadmap

### Phase 1: Evaluation Infrastructure (Priority: High)

#### 1.1 Discontinuous NER Evaluation

```rust
// src/eval/discontinuous.rs
pub struct DiscontinuousNERMetrics {
    pub entity_boundary_f1: f64,
    pub exact_match_f1: f64,
    pub partial_span_recall: f64,
    pub per_type: HashMap<String, TypeMetrics>,
}

pub fn evaluate_discontinuous_ner(
    gold: &[DiscontinuousGold],
    pred: &[DiscontinuousEntity],
) -> DiscontinuousNERMetrics;
```

**Datasets to support**:
- CADEC (Adverse Drug Events)
- ShARe13/ShARe14 (Clinical NER)
- Synthetic coordination structures

#### 1.2 Relation Extraction Evaluation

```rust
// src/eval/relation.rs
pub struct RelationMetrics {
    pub boundary_f1: f64,   // Rel evaluation
    pub strict_f1: f64,     // Rel+ evaluation
    pub per_relation: HashMap<String, TypeMetrics>,
}

pub fn evaluate_relations(
    gold: &[RelationGold],
    pred: &[RelationTriple],
    require_entity_match: bool,
) -> RelationMetrics;
```

**Datasets to support**:
- DocRED (document-level RE)
- TACRED (sentence-level RE)
- SciERC (scientific RE)

### Phase 2: Synthetic Data Generation (Priority: High)

#### 2.1 Discontinuous Entity Generator

```rust
// src/eval/dataset/synthetic/discontinuous.rs
pub fn generate_discontinuous_examples(count: usize) -> Vec<AnnotatedExample> {
    // Templates:
    // - "X and Y Z" → ["X Z", "Y Z"] (coordination)
    // - "X, Y, and Z W" → ["X W", "Y W", "Z W"]
    // - Medical: "pain in left and right arm"
}
```

#### 2.2 Relation Extraction Generator

```rust
// src/eval/dataset/synthetic/relations.rs
pub fn generate_relation_examples(count: usize) -> Vec<RelationExample> {
    // Templates:
    // - "{PERSON} founded {ORG}" → FOUNDED relation
    // - "{PERSON} works at {ORG}" → EMPLOYED_BY relation
    // - "{ORG} is headquartered in {LOC}" → LOCATED_IN relation
}
```

### Phase 3: Backend Implementations (Priority: Medium)

#### 3.1 GLiNER Bi-Encoder Integration

Complete the bi-encoder architecture with actual model loading:

```rust
// src/backends/gliner_biencoder.rs
pub struct GLiNERBiEncoder {
    text_encoder: Box<dyn TextEncoder>,
    label_encoder: Box<dyn LabelEncoder>,
    span_rep: SpanRepresentationLayer,
    registry: SemanticRegistry,
}

impl BiEncoder for GLiNERBiEncoder { ... }
impl ZeroShotNER for GLiNERBiEncoder { ... }
```

#### 3.2 W2NER Joint Extraction

Wire up `HandshakingMatrix` to `RelationExtractor`:

```rust
impl RelationExtractor for W2NER {
    fn extract_with_relations(...) -> Result<ExtractionWithRelations> {
        // Use handshaking matrix to decode both entities and relations
    }
}
```

### Phase 4: Visual/Multimodal (Priority: Low, v0.4+)

#### 4.1 ColPali Integration Design

```rust
// src/backends/colpali.rs
pub struct ColPali {
    encoder: OrtSession,
    tokenizer: Tokenizer,
    image_processor: ImageProcessor,
}

impl Model for ColPali { ... }
impl VisualCapable for ColPali {
    fn extract_from_image(&self, image_data: &[u8], format: &str) -> Result<Vec<Entity>> {
        // 1. Process image into patches
        // 2. Encode patches with vision encoder
        // 3. Late interaction with label embeddings
        // 4. Return entities with bounding boxes
    }
}
```

#### 4.2 Visual NER Evaluation

```rust
// src/eval/visual.rs
pub struct VisualNERMetrics {
    pub text_f1: f64,           // Text content match
    pub bbox_iou: f64,          // Bounding box IoU
    pub grounding_accuracy: f64, // Correct region identification
}

pub fn evaluate_visual_ner(
    gold: &[VisualEntity],
    pred: &[VisualEntity],
) -> VisualNERMetrics;
```

**Datasets**:
- FUNSD (form understanding)
- CORD (receipt understanding)
- DocVQA (document visual QA)

## Testing Strategy

### Property-Based Tests (Existing Pattern)

Continue using `proptest` for invariants:
- Discontinuous spans: `start < end` for each segment
- Relation triples: `head_idx != tail_idx`
- Visual positions: coordinates in `[0.0, 1.0]`
- Confidence scores: bounded to `[0.0, 1.0]`

### Integration Tests

```rust
// tests/advanced_tasks_tests.rs
#[test]
fn test_discontinuous_ner_e2e() {
    let model = W2NER::new();
    let text = "New York and Los Angeles airports";
    let entities = model.extract_discontinuous(text, &["LOC"], 0.5)?;
    
    // Verify discontinuous entity detection
    assert!(entities.iter().any(|e| !e.is_contiguous()));
}

#[test]
fn test_relation_extraction_e2e() {
    let model = W2NER::with_config(W2NERConfig { ... });
    let text = "Steve Jobs founded Apple in California.";
    let result = model.extract_with_relations(text, ...)?;
    
    // Verify relation detection
    assert!(result.relations.iter().any(|r| r.relation_type == "FOUNDED"));
}
```

### Benchmark Tests

```rust
// benches/advanced_tasks_bench.rs
#[bench]
fn bench_discontinuous_extraction(b: &mut Bencher) {
    let model = W2NER::from_pretrained("...")?;
    b.iter(|| model.extract_discontinuous(TEXT, &LABELS, 0.5));
}
```

## Research References

### Zero-Shot NER
- GLiNER: arXiv:2311.08526 (NAACL 2024)
- UniversalNER: arXiv:2308.03279 (ICLR 2024)
- Familiarity bias: arXiv:2412.10121

### Discontinuous NER
- W2NER: arXiv:2112.10070 (AAAI 2022)
- Evaluation: CADEC, ShARe13, ShARe14 datasets

### Relation Extraction
- DocRED: arXiv:1906.06127
- TACRED: arXiv:1705.08028
- Joint extraction: arXiv:2203.05412 (OneRel)

### Visual Document Understanding
- ColPali: arXiv:2407.01449
- LayoutLM: arXiv:1912.13318
- FUNSD: arXiv:1905.13538

## Conclusion

The trait design in `inference.rs` is well-aligned with bleeding-edge research, but
the implementations lag significantly behind the design. Priority should be:

1. **Immediate**: Add evaluation infrastructure for `DiscontinuousNER` and `RelationExtractor`
2. **Short-term**: Synthetic data generators for these tasks
3. **Medium-term**: Complete bi-encoder implementation for GLiNER
4. **Long-term**: ColPali integration for visual document understanding

The modular design of `eval/` makes it straightforward to add new evaluation modules
following the existing patterns (see `ARCHITECTURE.md`).

