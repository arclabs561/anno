# Changelog

## [0.1.0] - 2024-11-26

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

