# Changelog

## [Unreleased]

### Changed
- Workspace refactor: 5 crates (`anno-core`, `anno`, `anno-coalesce`, `anno-strata`, `anno-cli`)
- `PatternNER` → `RegexNER`

### Added
- GLiNER2: Multi-task extraction (NER + classification)
- Coreference resolution: T5-based and rule-based
- Graph RAG: Neo4j/NetworkX export
- Grounded entity representation: Signal → Track → Identity
- Task evaluation system
- Discourse analysis: Abstract anaphora, events

## [0.2.0] - 2025-11-27

### Added
- StackedNER: Composable layered extraction
- HeuristicNER: Zero-dependency NER
- Coreference metrics: MUC, B³, CEAF, LEA, BLANC
- NuNER, W2NER, CandleNER, GLiNERCandle
- 887 tests

### Changed
- `GLiNERv2` → `GLiNER`, `GLiNERNER` → `GLiNEROnnx`
- `LayeredNER`/`TieredNER` → `StackedNER`
- Entity uses byte offsets consistently

## [0.1.0] - 2025-11-26

Initial release.
