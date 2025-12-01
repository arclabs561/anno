# Tasks × Models × Datasets × Evals × Tests Matrix

Comprehensive review of the entire evaluation and testing infrastructure.

## Overview

This document provides a complete matrix of:
- **Tasks**: What can be evaluated (NER, Coreference, Relations, etc.)
- **Models**: All backends available (9 backends)
- **Datasets**: All datasets supported (30+ datasets)
- **Evals**: Evaluation metrics and types
- **Tests**: Test suites and coverage (2,400+ tests)

## 1. Tasks (5 Task Types)

### 1.1 NER (Named Entity Recognition)
- **Input**: Text
- **Output**: `Vec<Entity>` (span + type)
- **Metrics**: Precision, Recall, F1 (Strict/Exact/Partial/Type modes)
- **Supported by**: All 9 backends

### 1.2 Relation Extraction
- **Input**: Text
- **Output**: `Vec<(Entity, Relation, Entity)>`
- **Metrics**: Relation F1 (with/without entity correctness)
- **Supported by**: GLiNER2 (Onnx, Candle)

### 1.3 Coreference Resolution
- **Input**: Text
- **Output**: `Vec<CorefChain>`
- **Metrics**: MUC, B³, CEAF-e/m, LEA, BLANC, CoNLL F1
- **Supported by**: T5Coref, DiscourseAwareResolver

### 1.4 Discontinuous NER
- **Input**: Text
- **Output**: `Vec<DiscontinuousEntity>`
- **Metrics**: Same as NER but with discontinuous span matching
- **Supported by**: W2NER

### 1.5 Event Extraction
- **Input**: Text
- **Output**: Events with trigger and argument spans
- **Metrics**: Trigger F1, Argument F1, Event F1
- **Supported by**: EventExtractor (discourse feature)

## 2. Models/Backends (9 Backends)

| Backend | Feature | Zero-Shot | Nested | Discontinuous | Relations | Status | Tests |
|---------|---------|-----------|--------|---------------|-----------|--------|-------|
| **RegexNER** | - | No | No | No | No | ✅ Stable | ✅ 100+ |
| **HeuristicNER** | - | No | No | No | No | ✅ Stable | ✅ 50+ |
| **StackedNER** | - | No | No | No | No | ✅ Stable | ✅ 100+ |
| **BertNEROnnx** | `onnx` | No | No | No | No | ✅ Stable | ✅ 30+ |
| **GLiNEROnnx** | `onnx` | **Yes** | No | No | No | ✅ Stable | ✅ 50+ |
| **NuNER** | `onnx` | **Yes** | No | No | No | ✅ Stable | ✅ 20+ |
| **W2NER** | `onnx` | No | **Yes** | **Yes** | No | ✅ Stable | ✅ 10+ |
| **CandleNER** | `candle` | No | No | No | No | ✅ Stable | ✅ 20+ |
| **GLiNERCandle** | `candle` | **Yes** | No | No | No | ✅ Beta | ✅ 20+ |
| **GLiNER2Onnx** | `onnx` | **Yes** | No | No | **Yes** | ✅ Stable | ✅ 70+ |
| **GLiNER2Candle** | `candle` | **Yes** | No | No | **Yes** | ✅ Beta | ✅ 10+ |

**Total**: 11 backends (9 NER-only, 2 multi-task)

## 3. Datasets (30+ Datasets)

### 3.1 NER Datasets (25 datasets)

| Dataset | Domain | Entity Types | Size | Source | Tests |
|---------|--------|--------------|------|--------|-------|
| **WikiGold** | Wikipedia | PER, LOC, ORG, MISC | ~3.5k entities | GitHub | ✅ |
| **WNUT-17** | Social Media | person, location, etc. | ~5k tweets | GitHub | ✅ |
| **MIT Movie** | Movies | actor, director, genre, etc. | Domain-specific | MIT | ✅ |
| **MIT Restaurant** | Restaurants | amenity, cuisine, dish, etc. | Domain-specific | MIT | ✅ |
| **CoNLL-2003** | News | PER, LOC, ORG, MISC | ~22k sentences | GitHub | ✅ |
| **OntoNotes** | General | 18 entity types | Public sample | GitHub | ✅ |
| **MultiNERD** | Multilingual | 15+ types | ~50k examples | HF Direct | ✅ |
| **BC5CDR** | Biomedical | disease, chemical | ~1.5k docs | GitHub | ✅ |
| **NCBIDisease** | Biomedical | disease | ~800 docs | GitHub | ✅ |
| **GENIA** | Biomedical | gene, protein, cell | 2000 abstracts | HF API | ✅ |
| **AnatEM** | Biomedical | 12 anatomical types | ~1.3k docs | HF API | ✅ |
| **BC2GM** | Biomedical | gene, protein | ~20k sentences | HF API | ✅ |
| **BC4CHEMD** | Biomedical | chemical | ~88k sentences | HF API | ✅ |
| **TweetNER7** | Social Media | 7 types | ~11k tweets | HF Direct | ✅ |
| **BroadTwitterCorpus** | Social Media | Various | Stratified | HF Direct | ✅ |
| **FabNER** | Manufacturing | 12 types | Domain-specific | HF API | ✅ |
| **FewNERD** | General | 8 coarse + 66 fine | 188k sentences | HF API | ✅ |
| **CrossNER** | Cross-domain | 5 domains | Domain-specific | HF API | ✅ |
| **UniversalNERBench** | General | Various | Test subset | MIT | ✅ |
| **WikiANN** | Multilingual | PER, LOC, ORG | 282 languages | HF API | ✅ |
| **MultiCoNER** | Multilingual | 33 types, 12 langs | Complex entities | HF API | ✅ |
| **MultiCoNERv2** | Multilingual | 36 types, 12 langs | Noisy web text | HF API | ✅ |
| **WikiNeural** | Multilingual | 9 languages | Silver annotations | HF API | ✅ |
| **PolyglotNER** | Multilingual | 40 languages | Wikipedia+Freebase | HF API | ✅ |
| **UniversalNER** | Multilingual | 19 datasets, 13 langs | Gold standard | HF API | ✅ |

### 3.2 Relation Extraction (2 datasets)

| Dataset | Domain | Relations | Size | Source | Tests |
|---------|--------|-----------|------|--------|-------|
| **DocRED** | General | 96 types | Multi-sentence | GitHub | ✅ |
| **ReTACRED** | General | 41 types | ~106k examples | GitHub | ✅ |

### 3.3 Discontinuous NER (1 dataset)

| Dataset | Domain | Format | Size | Source | Tests |
|---------|--------|--------|------|--------|-------|
| **CADEC** | Clinical | Discontinuous | Clinical text | HF Direct | ✅ |

### 3.4 Coreference (3 datasets)

| Dataset | Domain | Format | Size | Source | Tests |
|---------|--------|--------|------|--------|-------|
| **GAP** | Wikipedia | Pronoun-name pairs | 8.9k examples | GitHub | ✅ |
| **PreCo** | General | Large-scale | 10x OntoNotes | GitHub | ✅ |
| **LitBank** | Literary | Fiction works | 100 works (1719-1922) | GitHub | ✅ |

### 3.5 Synthetic Data (Generated, No Download)

- **Domains**: news, scientific, financial, legal, biomedical, social_media, entertainment, specialized, relations, discontinuous, misc
- **Generated on-demand**: Fast, no cache needed
- **Tests**: ✅ Extensive property-based tests

**Total**: 31 real datasets + unlimited synthetic

## 4. Evaluation Types/Metrics

### 4.1 NER Metrics

| Metric | Description | Mode | Tests |
|--------|-------------|------|-------|
| **Precision** | TP / (TP + FP) | All modes | ✅ |
| **Recall** | TP / (TP + FN) | All modes | ✅ |
| **F1** | 2 × (P × R) / (P + R) | All modes | ✅ |
| **Strict** | Exact span + type match | Default | ✅ |
| **Exact** | Exact span match (type ignored) | Optional | ✅ |
| **Partial** | Overlapping span match | Optional | ✅ |
| **Type** | Type match (span ignored) | Optional | ✅ |
| **Per-Type** | Metrics per entity type | All modes | ✅ |
| **Micro/Macro** | Aggregation methods | All modes | ✅ |

### 4.2 Relation Extraction Metrics

| Metric | Description | Tests |
|--------|-------------|-------|
| **Relation F1** | Relation-level F1 | ✅ |
| **Entity Match Required** | Option to require correct entities | ✅ |
| **Per-Relation** | Metrics per relation type | ✅ |

### 4.3 Coreference Metrics

| Metric | Description | Tests |
|--------|-------------|-------|
| **MUC** | Link-based metric | ✅ |
| **B³ (B-cubed)** | Mention-based metric | ✅ |
| **CEAF-e** | Entity-based alignment | ✅ |
| **CEAF-m** | Mention-based alignment | ✅ |
| **LEA** | Link-based entity-aware | ✅ |
| **BLANC** | Rand-index based | ✅ |
| **CoNLL F1** | Average of MUC, B³, CEAF-e | ✅ |

### 4.4 Advanced Evaluation (eval-advanced feature)

| Type | Description | Tests |
|------|-------------|-------|
| **Error Analysis** | Confusion matrix, error categorization | ✅ |
| **Bias Analysis** | Gender, demographic, temporal, length bias | ✅ |
| **Calibration** | ECE, MCE, Brier score, reliability diagrams | ✅ |
| **Threshold Analysis** | Precision-recall curves, optimal thresholds | ✅ |
| **Long Tail** | Rare entity analysis | ✅ |
| **Few-Shot** | Few-shot learning evaluation | ✅ |
| **Robustness** | Perturbation testing | ✅ |
| **OOD Detection** | Out-of-distribution detection | ✅ |

## 5. Test Suites (2,400+ Tests)

### 5.1 Backend Tests

| Test File | Backends Tested | Tests | Description |
|-----------|----------------|-------|-------------|
| `backend_candle.rs` | CandleNER, GLiNERCandle | 10+ | Candle backend tests |
| `backend_nuner_w2ner.rs` | NuNER, W2NER | 58+ | Zero-shot and nested NER |
| `backend_comparison.rs` | All backends | 16+ | Cross-backend comparison |
| `gliner1_tests.rs` | GLiNEROnnx | 56+ | GLiNER v1 tests |
| `gliner2_tests.rs` | GLiNER2Onnx, GLiNER2Candle | 72+ | GLiNER2 multi-task |
| `gliner_candle_bug_tests.rs` | GLiNERCandle | 22+ | Candle-specific bugs |

### 5.2 Trait Tests

| Test File | Traits Tested | Tests | Description |
|-----------|---------------|-------|-------------|
| `trait_harmonization_tests.rs` | All traits | 21 | Property-based trait tests |
| `trait_integration_tests.rs` | All traits | 16 | Integration trait tests |
| `advanced_trait_tests.rs` | Advanced traits | 37+ | Advanced trait coverage |

### 5.3 Integration Tests

| Test File | Scope | Tests | Description |
|-----------|-------|-------|-------------|
| `integration.rs` | Basic pipeline | 8+ | Simple integration |
| `integration_comprehensive.rs` | Full pipeline | 34+ | Comprehensive integration |
| `integration_full_pipeline.rs` | NER→Coref→Discourse | 24+ | End-to-end pipeline |
| `integration_eval.rs` | Evaluation integration | 28+ | Eval framework integration |
| `eval_integration.rs` | Eval framework | 32+ | Full eval integration |

### 5.4 Dataset Tests

| Test File | Datasets | Tests | Description |
|-----------|----------|-------|-------------|
| `real_datasets.rs` | All real datasets | 30+ | Real dataset evaluation |
| `zero_shot_eval_tests.rs` | Zero-shot datasets | 6+ | Zero-shot evaluation |

### 5.5 Quality Tests

| Test File | Focus | Tests | Description |
|-----------|-------|-------|-------------|
| `e2e_quality_comprehensive.rs` | E2E quality | 33+ | End-to-end quality |
| `comprehensive_ner_tests.rs` | NER quality | 128+ | Comprehensive NER |
| `ner_tests.rs` | Basic NER | 8+ | Basic NER tests |
| `ner_comprehensive.rs` | Advanced NER | 12+ | Advanced NER |
| `relation_extraction_quality.rs` | Relations | 38+ | Relation quality |

### 5.6 Property-Based Tests (Proptest)

| Test File | Properties | Tests | Description |
|-----------|------------|-------|-------------|
| `eval_proptest.rs` | Eval properties | 16+ | Eval property tests |
| `corpus_proptest.rs` | Corpus properties | 4+ | Corpus properties |
| `invariant_tests.rs` | Invariants | 23+ | Invariant preservation |

### 5.7 Fuzz Tests

| Test File | Target | Tests | Description |
|-----------|--------|-------|-------------|
| `fuzz_edge_cases.rs` | Edge cases | 33+ | Edge case fuzzing |
| `offset_fuzz_tests.rs` | Offsets | 10+ | Offset fuzzing |
| `schema_fuzz_tests.rs` | Schemas | 2+ | Schema fuzzing |
| `entity_builder_fuzz_tests.rs` | Entity building | 5+ | Entity builder fuzzing |
| `entity_validation_fuzz_tests.rs` | Validation | 7+ | Validation fuzzing |
| `lang_detection_fuzz_tests.rs` | Language detection | 9+ | Lang detection fuzzing |
| `grounded_fuzz_tests.rs` | Grounded entities | 2+ | Grounded entity fuzzing |
| `similarity_fuzz_tests.rs` | Similarity | 11+ | Similarity fuzzing |

### 5.8 Edge Case Tests

| Test File | Focus | Tests | Description |
|-----------|-------|-------|-------------|
| `edge_cases.rs` | General edge cases | 44+ | General edge cases |
| `offset_edge_cases.rs` | Offset edge cases | 18+ | Offset-specific |
| `offset_bug_tests.rs` | Offset bugs | 30+ | Offset bug fixes |
| `discontinuous_span_tests.rs` | Discontinuous | 55+ | Discontinuous spans |
| `bounds_validation.rs` | Bounds | 10+ | Bounds validation |

### 5.9 Bug Fix Tests

| Test File | Bugs | Tests | Description |
|-----------|------|-------|-------------|
| `bug_fixes.rs` | General bugs | 30+ | Bug fixes |
| `subtle_bugs_tests.rs` | Subtle bugs | 52+ | Subtle bug detection |
| `w2ner_auth_tests.rs` | W2NER auth | 4+ | Authentication issues |
| `test_nuner_span_tensors.rs` | NuNER spans | - | Span tensor handling |

### 5.10 Specialized Tests

| Test File | Focus | Tests | Description |
|-----------|-------|-------|-------------|
| `domain_specific.rs` | Domain-specific | 29+ | Domain-specific NER |
| `multilingual_ner_tests.rs` | Multilingual | 38+ | Multilingual NER |
| `discourse_comprehensive.rs` | Discourse | 46+ | Discourse analysis |
| `coref_integration.rs` | Coreference | 36+ | Coreference integration |
| `pattern_statistical_detailed.rs` | Pattern stats | 74+ | Pattern statistics |
| `lang_detection_tests.rs` | Language detection | 60+ | Language detection |
| `type_mapper_tests.rs` | Type mapping | 4+ | Type mapping |
| `schema_mapping_tests.rs` | Schema mapping | 56+ | Schema mapping |

### 5.11 Performance Tests

| Test File | Focus | Tests | Description |
|-----------|-------|-------|-------------|
| `corpus_performance.rs` | Corpus performance | 6+ | Corpus performance |
| `concurrency_tests.rs` | Concurrency | 11+ | Concurrent execution |
| `test_parallel_evaluation.rs` | Parallel eval | - | Parallel evaluation |
| `slow_benchmarks.rs` | Benchmarks | 16+ | Slow benchmarks |

### 5.12 Regression Tests

| Test File | Focus | Tests | Description |
|-----------|-------|-------|-------------|
| `regression_f1.rs` | F1 regression | 10+ | F1 score regression |
| `seqeval_comparison.rs` | SeqEval comparison | 14+ | SeqEval compatibility |

### 5.13 CLI Tests

| Test File | Focus | Tests | Description |
|-----------|-------|-------|-------------|
| `cli_integration.rs` | CLI | 114+ | CLI integration |

### 5.14 Die Hard Tests

| Test File | Focus | Tests | Description |
|-----------|-------|-------|-------------|
| `die_hard.rs` | Stress tests | 180+ | Stress testing |

**Total Test Count**: ~2,400+ tests across 79 test files

## 6. Coverage Matrix

### 6.1 Task × Model Coverage

| Task | RegexNER | HeuristicNER | StackedNER | BertNEROnnx | GLiNEROnnx | NuNER | W2NER | CandleNER | GLiNERCandle | GLiNER2 |
|------|------------|--------------|------------|-------------|------------|-------|-------|------------|--------------|---------|
| **NER** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Zero-Shot NER** | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ |
| **Nested NER** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| **Discontinuous NER** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| **Relation Extraction** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Coreference** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

### 6.2 Task × Dataset Coverage

| Task | NER Datasets | Relation Datasets | Discontinuous Datasets | Coref Datasets |
|------|--------------|-------------------|------------------------|----------------|
| **NER** | ✅ 25 datasets | ❌ | ❌ | ❌ |
| **Zero-Shot NER** | ✅ 25 datasets | ❌ | ❌ | ❌ |
| **Nested NER** | ✅ 25 datasets | ❌ | ❌ | ❌ |
| **Discontinuous NER** | ❌ | ❌ | ✅ 1 dataset (CADEC) | ❌ |
| **Relation Extraction** | ❌ | ✅ 2 datasets | ❌ | ❌ |
| **Coreference** | ❌ | ❌ | ❌ | ✅ 3 datasets |

### 6.3 Model × Dataset Coverage

**All NER models** can evaluate on **all 25 NER datasets** (with appropriate label mapping).

**GLiNER2** can evaluate on:
- All 25 NER datasets (NER task)
- 2 relation datasets (Relation task)

**W2NER** can evaluate on:
- All 25 NER datasets (NER task)
- 1 discontinuous dataset (CADEC)

### 6.4 Evaluation × Task Coverage

| Evaluation Type | NER | Relations | Coreference | Discontinuous | Events |
|-----------------|-----|-----------|-------------|--------------|--------|
| **Basic Metrics** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Error Analysis** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Bias Analysis** | ✅ | ❌ | ✅ | ❌ | ❌ |
| **Calibration** | ✅ | ✅ | ❌ | ✅ | ❌ |
| **Threshold Analysis** | ✅ | ✅ | ❌ | ✅ | ❌ |
| **Long Tail** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Few-Shot** | ✅ | ❌ | ❌ | ❌ | ❌ |

## 7. Test Coverage Summary

### 7.1 By Backend

| Backend | Unit Tests | Integration Tests | Dataset Tests | Total |
|---------|------------|-------------------|---------------|-------|
| RegexNER | ✅ 100+ | ✅ 50+ | ✅ 30+ | ~180+ |
| HeuristicNER | ✅ 50+ | ✅ 50+ | ✅ 30+ | ~130+ |
| StackedNER | ✅ 100+ | ✅ 50+ | ✅ 30+ | ~180+ |
| BertNEROnnx | ✅ 30+ | ✅ 20+ | ✅ 30+ | ~80+ |
| GLiNEROnnx | ✅ 50+ | ✅ 30+ | ✅ 30+ | ~110+ |
| NuNER | ✅ 20+ | ✅ 10+ | ✅ 30+ | ~60+ |
| W2NER | ✅ 10+ | ✅ 5+ | ✅ 10+ | ~25+ |
| CandleNER | ✅ 20+ | ✅ 10+ | ✅ 20+ | ~50+ |
| GLiNERCandle | ✅ 20+ | ✅ 10+ | ✅ 20+ | ~50+ |
| GLiNER2Onnx | ✅ 70+ | ✅ 30+ | ✅ 30+ | ~130+ |
| GLiNER2Candle | ✅ 10+ | ✅ 5+ | ✅ 10+ | ~25+ |

### 7.2 By Task

| Task | Tests | Coverage |
|------|-------|----------|
| **NER** | ~1,500+ | ✅ Comprehensive |
| **Zero-Shot NER** | ~200+ | ✅ Good |
| **Nested NER** | ~50+ | ✅ Good |
| **Discontinuous NER** | ~60+ | ✅ Good |
| **Relation Extraction** | ~100+ | ✅ Good |
| **Coreference** | ~100+ | ✅ Good |
| **Event Extraction** | ~50+ | ✅ Basic |

### 7.3 By Dataset

| Dataset Category | Datasets | Tests | Coverage |
|------------------|----------|-------|----------|
| **General NER** | 6 | ✅ | Comprehensive |
| **Biomedical NER** | 5 | ✅ | Good |
| **Social Media NER** | 2 | ✅ | Good |
| **Multilingual NER** | 6 | ✅ | Good |
| **Specialized NER** | 6 | ✅ | Good |
| **Relation Extraction** | 2 | ✅ | Good |
| **Discontinuous NER** | 1 | ✅ | Good |
| **Coreference** | 3 | ✅ | Good |

## 8. Gaps and Recommendations

### 8.1 Coverage Gaps

1. **Event Extraction**: Limited dataset coverage (synthetic only)
2. **Relation Extraction**: Only 2 datasets (DocRED, ReTACRED)
3. **Discontinuous NER**: Only 1 dataset (CADEC)
4. **Coreference**: Limited to 3 datasets
5. **Multilingual**: Some datasets use proxies (not original)

### 8.2 Test Gaps

1. **GLiNER2Candle**: Fewer tests than GLiNER2Onnx
2. **W2NER**: Limited test coverage (auth issues)
3. **Event Extraction**: Basic tests only
4. **Advanced Eval**: Some metrics not tested on all tasks

### 8.3 Recommendations

1. **Add more relation datasets**: TACRED, SemEval, etc.
2. **Add more discontinuous datasets**: ShARe/CLEF, etc.
3. **Expand event extraction**: ACE 2005, etc.
4. **Increase GLiNER2Candle tests**: Match GLiNER2Onnx coverage
5. **Add cross-task evaluation**: NER + Relations + Coref combined
6. **Expand multilingual**: More original datasets (not proxies)

## 9. Summary Statistics

- **Tasks**: 5 task types
- **Models**: 11 backends (9 NER-only, 2 multi-task)
- **Datasets**: 31 real datasets + unlimited synthetic
- **Evaluation Types**: 10+ metric categories
- **Tests**: ~2,400+ tests across 79 test files
- **Coverage**: Comprehensive for NER, good for other tasks

## 10. Quick Reference

### Run All Tests
```bash
cargo test --all-features
```

### Run Specific Test Suite
```bash
cargo test --test real_datasets --features eval-advanced
cargo test --test backend_nuner_w2ner --features onnx
cargo test --test gliner2_tests --features onnx
```

### Run Evaluation
```bash
cargo test --test eval_integration --features eval-advanced
```

### Run Benchmarks
```bash
cargo test --test slow_benchmarks --features eval-advanced -- --ignored
```

