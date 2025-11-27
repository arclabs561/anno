# Changelog

## [0.2.0] - 2025-11-27

### Added
- **StackedNER**: Composable layered extraction with conflict strategies
- **StatisticalNER**: Zero-dependency heuristic NER for Person/Org/Location
- **ConflictStrategy**: Priority, LongestSpan, HighestConf, Union
- **EvalMode**: SemEval-style evaluation (Strict, Exact, Partial, Type)
- **Coreference metrics**: MUC, B³, CEAF-e, CEAF-m, LEA, BLANC, CoNLL F1
- **offset module**: Robust byte/char offset conversion with SpanConverter
- **DiscontinuousSpan**: Support for non-contiguous entity spans
- **Provenance tracking**: model_version, timestamp fields
- **Dataset loader**: Download/cache for CoNLL-2003, WikiGold, WNUT-17, etc.
- **NuNER, W2NER**: Placeholder backends for future implementation
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
- Added property tests for Entity and PatternNER
- Added `#[must_use]` to pure functions
- Fixed all doc tests (6/6 passing)
- Fixed clippy lints

## [0.1.0] - 2025-11-26

Initial release.

### Added

- `Model` trait for NER backends
- `PatternNER`: DATE, MONEY, PERCENT extraction (always available)
- `RuleBasedNER`: Gazetteer-based (deprecated)
- `BertNEROnnx`: BERT via ONNX Runtime (feature: `onnx`)
- `GLiNERNER`: Zero-shot NER via ONNX (feature: `onnx`)
- `CandleNER`: Rust-native BERT (feature: `candle`)
- Evaluation framework with `GoldEntity`, metrics, validation
- CoNLL-2003, JSON/JSONL dataset loading

