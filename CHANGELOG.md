# Changelog

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

