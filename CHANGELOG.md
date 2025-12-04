# Changelog

## [Unreleased]

### Changed
- **BREAKING**: Refactored into workspace structure with 5 crates:
  - `anno-core`: Core types (Entity, GroundedDocument, Corpus, GraphDocument)
  - `anno`: Main NER library (backends, evaluation, document processing)
  - `anno-coalesce`: Cross-document entity coalescing
  - `anno-strata`: Hierarchical clustering (Leiden, RAPTOR)
  - `anno-cli`: Unified CLI binary
- Updated README to document workspace structure and crate organization
- All types from `anno-core` are re-exported in `anno` for backward compatibility

### Added
- **GLiNER2**: Multi-task extraction (NER + classification + relations) via ONNX or Candle
- **Coreference Resolution**: T5-based coreference resolver (`T5Coref`) and rule-based resolver (`SimpleCorefResolver`)
- **Graph RAG**: `GraphDocument` for exporting entities and relations to Neo4j/NetworkX
- **Grounded Entity Representation**: Signal → Track → Identity hierarchy for multimodal NER
- **Task Evaluation System**: Comprehensive task-dataset-backend evaluation framework
- **Backend Factory**: Dynamic backend creation based on feature flags
- **Discourse Analysis**: Abstract anaphora resolution, event extraction, shell noun detection
- **CLI Tools**: `anno` and `anno-eval` binaries for command-line usage
- **Justfile**: Task runner with common development commands
- **GLiNER Prompt Cache**: LRU cache for prompt encodings, ~44x speedup for repeated entity types

### Changed
- **BREAKING**: Renamed `PatternNER` → `RegexNER` for clarity (module: `pattern.rs` → `regex.rs`)
- Updated README to reflect coreference resolution capabilities (not just NER)
- Library description now emphasizes "Information extraction: NER, coreference resolution, and evaluation"
- Fixed `task_evaluator.rs` to handle `eval-advanced` feature gate correctly
- Improved documentation across all modules
- Backend factory now accepts "regex"/"regexner" in addition to "pattern"/"patternner" (backward compatible)

### Fixed
- Compilation error in `task_evaluator.rs` when `eval-advanced` feature not enabled
- Clippy warnings for double `#[must_use]` attributes
- Error conversion for HuggingFace API errors (`From<ApiError>` implementation)
- Documentation link warnings (modules exist, links were correct)

## [0.2.0] - 2025-11-27

### Added
- **StackedNER**: Composable layered extraction with conflict strategies
- **HeuristicNER**: Zero-dependency heuristic NER for Person/Org/Location
- **ConflictStrategy**: Priority, LongestSpan, HighestConf, Union
- **EvalMode**: SemEval-style evaluation (Strict, Exact, Partial, Type)
- **Coreference metrics**: MUC, B³, CEAF-e, CEAF-m, LEA, BLANC, CoNLL F1
- **offset module**: Robust byte/char offset conversion with SpanConverter
- **DiscontinuousSpan**: Support for non-contiguous entity spans
- **Provenance tracking**: model_version, timestamp fields
- **Dataset loader**: Download/cache for CoNLL-2003, WikiGold, WNUT-17, etc.
- **NuNER**: Token-based zero-shot NER via ONNX (BIO tagging)
- **W2NER**: Nested/discontinuous entity extraction via ONNX
- **CandleNER**: Pure Rust BERT encoder with Metal/CUDA support
- **GLiNERCandle**: Pure Rust zero-shot NER with span classification
- **anno::auto()**: Automatic backend selection based on available features
- **Capability traits**: BatchCapable, GpuCapable, DynamicLabels, etc.
- 887 tests total

### Changed
- Renamed `GLiNERv2` → `GLiNER`, `GLiNERNER` → `GLiNEROnnx`
- Renamed `LayeredNER`/`TieredNER` → `StackedNER`
- `Entity` now uses byte offsets consistently
- `EntityType::is_structured()` helper method
- Improved README with feature flag documentation

### Fixed
- Clippy `implicit_saturating_sub` in `Relation::span_distance()`
- Clippy `type_complexity` in benchmark templates
- Unicode byte/char offset handling throughout

### Removed
- Orphaned `bytes.rs` and `span.rs` files

## [0.1.2] - 2025-11-27

### Fixed
- Clippy `implicit_saturating_sub` in `Relation::span_distance()`
- Clippy `type_complexity` in benchmark templates

## [0.1.1] - 2025-11-26

### Changed
- Fixed CI: align feature names (`onnx` instead of `ml-ner-onnx`)
- Added property tests for Entity and RegexNER
- Added `#[must_use]` to pure functions
- Fixed all doc tests (6/6 passing)
- Fixed clippy lints

## [0.1.0] - 2025-11-26

Initial release.

### Added

- `Model` trait for NER backends
- `RegexNER`: DATE, MONEY, PERCENT extraction (always available)
- `RuleBasedNER`: Gazetteer-based (deprecated)
- `BertNEROnnx`: BERT via ONNX Runtime (feature: `onnx`)
- `GLiNERNER`: Zero-shot NER via ONNX (feature: `onnx`)
- `CandleNER`: Rust-native BERT (feature: `candle`)
- Evaluation framework with `GoldEntity`, metrics, validation
- CoNLL-2003, JSON/JSONL dataset loading

