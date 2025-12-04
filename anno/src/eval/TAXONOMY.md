# Evaluation Taxonomy

This document defines the tasks, metrics, datasets, and entity types supported by Anno.

## Research Context (Why This Matters)

Recent papers fundamentally changed how we should think about NER:

| Paper | Key Finding | Our Response |
|-------|-------------|--------------|
| **Familiarity** (arXiv:2412.10121) | Zero-shot benchmarks are biased by label overlap | `LabelShift` type to quantify |
| **Literary Coref** (arXiv:2401.00238) | Single CoNLL F1 is misleading | `CorefChainStats` for stratified eval |
| **GLiNER** (NAACL 2024, arXiv:2311.08526) | NER as span-label matching | `BiEncoder`, `LateInteraction` traits |
| **UniversalNER** (ICLR 2024, arXiv:2308.03279) | Distillation outperforms teacher LLM | 43 datasets, 9 domains |
| **ReasoningNER** (arXiv:2511.11978, Nov 2025) | Chain-of-thought improves NER F1 by 1.3+ | Future: `ReasoningNER` trait |
| **ModernBERT** (arXiv:2412.13663, Dec 2024) | 3x faster encoder with 8K context | `TextEncoder` trait |
| **CMAS** (arXiv:2502.18702, Feb 2025) | Multi-agent zero-shot NER | Future: agent orchestration |

### 2025 State of the Art (as of Nov 2025)

| Model | Approach | Zero-Shot Avg F1 | Notes |
|-------|----------|------------------|-------|
| **ReasoningNER** | CoT + GRPO | 85.2 | Best on 11/20 datasets |
| **GLiNER-bi** | Bi-encoder | 80-84 | Fastest inference |
| **UniversalNER-7B** | Distilled LLM | 84.8 | Best balance |
| **GPT-4** | In-context | 77-80 | Slow, expensive |

## Tasks

| Task | Input | Output | Example |
|------|-------|--------|---------|
| **NER** | Text | `Vec<Entity>` | "John works at Apple" → [Person("John"), Org("Apple")] |
| **Relation Extraction** | Text | `Vec<(Entity, Relation, Entity)>` | "John founded Apple" → (John, FOUNDED, Apple) |
| **Coreference Resolution** | Text | `Vec<CorefChain>` | "John said he..." → [[John, he]] |
| **Discontinuous NER** | Text | `Vec<DiscontinuousEntity>` | "pain...in abdomen" → [pain + abdomen] |

## Metrics by Task

### NER Metrics

| Metric | What it measures | When to use |
|--------|------------------|-------------|
| **Strict F1** | Exact span + exact type | Standard benchmark |
| **Exact F1** | Exact span, any type | Span quality only |
| **Partial F1** | Overlapping span + type | Tolerant evaluation |
| **Type F1** | Correct type, ignoring span | Type classification quality |

### Coreference Metrics

| Metric | Focus | Pros | Cons |
|--------|-------|------|------|
| **MUC** | Links | Simple | Ignores singletons |
| **B³** | Mentions | Per-mention | Inflated by singletons |
| **CEAF-e** | Entities | Optimal alignment | Expensive to compute |
| **CEAF-m** | Mentions | Optimal alignment | Sensitive to cluster size |
| **LEA** | Links+Entities | Balanced | Less common |
| **BLANC** | All pairs | Most discriminative | Quadratic complexity |
| **CoNLL F1** | Composite | Standard | Average of MUC, B³, CEAF-e |

### Relation Extraction Metrics

| Metric | Requires |
|--------|----------|
| **Strict F1** | Correct head + tail + relation type |
| **Boundary F1** | Correct head + tail spans |
| **Type F1** | Correct relation type only |

## Entity Type Taxonomy

```
EntityCategory (high-level)
├── Agent           (ML-required)
│   └── Person, NORP
├── Organization    (ML-required)
│   └── Organization, Facility
├── Place           (ML-required)
│   └── Location, GPE
├── Creative        (ML-required)
│   └── WorkOfArt, Event, Product
├── Temporal        (Pattern-detectable)
│   └── Date, Time
├── Numeric         (Pattern-detectable)
│   └── Money, Percent, Quantity, Cardinal, Ordinal
├── Contact         (Pattern-detectable)
│   └── Email, URL, Phone
├── Relation        (ML-required, for KG)
│   └── CEO_OF, WORKS_FOR, FOUNDED, LOCATED_IN
└── Misc            (Catch-all)
    └── Custom types, domain-specific
```

## Datasets

### NER Datasets (Standard)

| Dataset | Domain | Entity Types | Size | Quality |
|---------|--------|--------------|------|---------|
| **CoNLL-2003** | News | PER, ORG, LOC, MISC | 20k ents | Gold standard |
| **WikiGold** | Wikipedia | PER, ORG, LOC, MISC | 3.5k ents | Good |
| **WNUT-17** | Social media | 6 types | 2k ents | Noisy, challenging |
| **OntoNotes** | Mixed | 18 types | 18k ents | Comprehensive |
| **MultiNERD** | Wikipedia | 15+ types | 100k+ ents | Large scale |

### NER Datasets (Domain-Specific)

| Dataset | Domain | Entity Types | Use Case |
|---------|--------|--------------|----------|
| **BC5CDR** | Biomedical | Disease, Chemical | Medical NER |
| **NCBI Disease** | Biomedical | Disease | Medical NER |
| **MIT Movie** | Entertainment | Actor, Director, Genre, etc. | Slot filling |
| **MIT Restaurant** | Restaurants | Cuisine, Location, Price, etc. | Slot filling |

### NER Datasets (Few-Shot / Cross-Domain)

| Dataset | Focus | Entity Types | Use Case |
|---------|-------|--------------|----------|
| **Few-NERD** | Few-shot learning | 8 coarse + 66 fine | Few-shot evaluation |
| **CrossNER** | Domain transfer | 5 domains | Cross-domain generalization |
| **UniversalNER** | Zero-shot | Open | Zero-shot evaluation |

### Relation Extraction Datasets

| Dataset | Focus | Relations | Use Case |
|---------|-------|-----------|----------|
| **DocRED** | Document-level | 96 types | Multi-sentence reasoning |
| **Re-TACRED** | Sentence-level | 41 types | Standard RE benchmark |

### Coreference Datasets

| Dataset | Domain | Size | Features |
|---------|--------|------|----------|
| **GAP** | Wikipedia | 8.9k pairs | Gender-balanced pronouns |
| **PreCo** | Reading | 38k docs | Includes singletons |
| **LitBank** | Literature | 100 works | Literary coreference |
| **OntoNotes Coref** | Mixed | 3k docs | Standard benchmark (not yet impl) |

### Synthetic Datasets

| Dataset | Purpose | Examples | Entity Types |
|---------|---------|----------|--------------|
| **Easy** | Baseline | ~23 | Standard NER |
| **Medium** | General | ~69 | + ambiguous cases |
| **Hard** | Challenging | ~11 | + nested, long |
| **Adversarial** | Edge cases | ~12 | + unicode, malformed |

Total: 115+ examples with strict offset validation (text must match substring at offset).

## Test Organization (Proposed)

```
tests/
├── unit/                       # Fast, isolated
│   ├── entity_test.rs          # Entity struct tests
│   ├── offset_test.rs          # Offset conversion tests
│   └── type_test.rs            # Type system tests
├── backend/                    # Per-backend tests
│   ├── pattern_test.rs         # RegexNER
│   ├── heuristic_test.rs       # HeuristicNER
│   ├── stacked_test.rs         # StackedNER
│   ├── bert_onnx_test.rs       # BertNEROnnx
│   └── gliner_test.rs          # GLiNER variants
├── task/                       # Per-task tests
│   ├── ner_strict_test.rs      # Strict NER eval
│   ├── ner_partial_test.rs     # Partial NER eval
│   ├── coref_test.rs           # Coreference eval
│   └── relation_test.rs        # Relation extraction
├── dataset/                    # Per-dataset tests
│   ├── conll2003_test.rs       # CoNLL-2003 specific
│   ├── wikigold_test.rs        # WikiGold specific
│   └── synthetic_test.rs       # Synthetic data
├── property/                   # Property-based tests
│   ├── invariants_test.rs      # Core invariants
│   ├── offset_props_test.rs    # Offset properties
│   └── coref_props_test.rs     # Coref properties
└── integration/                # End-to-end tests
    ├── pipeline_test.rs        # Full pipeline
    └── benchmark_test.rs       # Performance
```

## Current Gaps

1. **Missing datasets**: OntoNotes coref, ACE 2005, TAC-KBP
2. **Missing tests**: Explicit relation extraction evaluation
3. **Test naming**: Inconsistent (some use `test_`, some descriptive)
4. ~~**Type mapping**: Domain datasets (MIT Movie) need type normalization~~ (Done: `TypeMapper` added)
5. ~~**Label shift quantification**: Need to measure training/eval overlap~~ (Done: `LabelShift` added)
6. ~~**Stratified coref metrics**: Need per-chain-length breakdown~~ (Done: `CorefChainStats` added)

## Recent Additions

- **17 datasets total**: 12 NER, 2 RE, 3 Coreference
- **TypeMapper**: Normalizes domain-specific types to standard NER types
- **EvalTask enum**: Clarifies what capability is being evaluated
- **EvalConfig.normalize_types**: Toggle type normalization in evaluation
- **Extended synthetic dataset**: 115+ high-quality examples with strict validation
- **Offset validation tests**: Entity text must match substring at specified offset
- **TokenSpan + OffsetMapping**: Subword token alignment for transformer models
- **LabelShift**: Quantifies training/eval label overlap (Familiarity paper)
- **CorefChainStats**: Stratified metrics by chain length (Literary Coref paper)
- **sampling module**: Stratified sampling maintaining entity type proportions

