# Changelog

## [Unreleased]

### Changed (breaking)
- Renamed `gliner2` backend to `gliner_multitask` to match what it actually loads (`onnx-community/gliner-multitask-large-v0.5`, the GLiNER v1 multi-task model from Stepanov & Shtopko 2024). The `gliner2` name was a misnomer that conflated this backend with the unrelated fastino-ai GLiNER2 architecture (Zaratiana et al. 2025), which is not yet supported. Affects:
  - Constants: `models::GLINER2` → `models::GLINER_MULTITASK`; `DEFAULT_GLINER2_MODEL` → `DEFAULT_GLINER_MULTITASK_MODEL`.
  - Types: `GLiNER2Onnx` → `GLiNERMultitaskOnnx`; `GLiNER2Candle` → `GLiNERMultitaskCandle`; `GLiNER2` → `GLiNERMultitask`.
  - Module: `anno::backends::gliner2` → `anno::backends::gliner_multitask`.
  - CLI: `--model gliner2` → `--model gliner_multitask`; `--model gliner2_candle` → `--model gliner_multitask_candle`. Backend strings in eval harness, scripts, and docs updated similarly.
  - Eval harness: `register_gliner2(_default)` → `register_gliner_multitask(_default)`. `register_gliner_multitask_default` now correctly defaults to `DEFAULT_GLINER_MULTITASK_MODEL` (was previously calling itself with `fastino/gliner2-base-v1`, which never loaded).
  - References: README.md and `docs/REFERENCES.md` re-attributed to Stepanov & Shtopko 2024 (arXiv:2406.12925); fastino-ai GLiNER2 (Zaratiana et al. 2025, arXiv:2507.18546) listed as distinct architecture not yet wired.

### Fixed
- `NuNER::as_zero_shot` no longer fails to compile under `--no-default-features` (cfg-gating mismatch with the `onnx`-gated `ZeroShotNER` impl).
- Cleaned up several pre-existing warnings: unused `EntityCategory` import in `tplinker.rs` (now `cfg`-gated), unused-macro warning on `define_feature_stub`, missing-docs on stub `GLiREL`/`GLiNERPoly` structs, dead-code on `local_model_cache_candidates` (now `cfg`-gated).

### Added
- `CONTRIBUTING.md`: dev setup, workspace layout, where backends live, feature flag map, style and PR guidance.

## [0.6.0] - 2026-04-16

### Added
- **Active learning** (`anno::active`): score and rank texts by model uncertainty for annotation prioritization; `anno-eval` bridge functions and `select_for_annotation` helper.
- **Optional `slabs` integration** for text chunking (behind `slabs` feature).
- **Publishing infrastructure** for `anno-eval`, `anno-graph`, and `anno-cli`.

### Fixed
- Muxer 0.5 API changes and downstream clippy.
- `lru` bump to 0.16 (RUSTSEC-2026-0002 soundness fix).
- Test relaxations for HMM recall and PII false-positive coverage.
- Fixture gitignore: `press_release.html` now tracked (was shadowed by `*.html`).
- Clippy under Rust 1.95 stable: `unnecessary_sort_by`, `collapsible_match`, `unnecessary_min_or_max`.

### Changed
- Integration test binaries consolidated from 14 into 1 (46s wall time, down from >10 min).
- Integration tests now package-target `anno-lib` with `discourse` feature.

## [0.5.0] - 2026-04-01

### Added
- **Tag-triggered publish workflow** (`v*` tags, OIDC trusted publishing).
- **Biomedical, GLiNER-PII, GLiNER-RelEx backends** wired up; DeBERTa-v3 upgrade.
- **Neural f-coref** coreference backend with heuristic fallback (in `anno-cli`).
- **86 backend unit tests** across 11 low-coverage modules.
- BertNEROnnx local directory loading.

### Fixed
- `FantasyCoref` URL (branch main -> master).
- GLiNER ONNX for token-level models (PII Edge) and variable vocab.
- Subtract pass: removed thin wrappers, inlined deps, shrank API surface.
- 31 doc link errors.

### Changed
- MSRV bumped from 1.85 to 1.88 (workspace, CI, justfile).
- TPLinker cached via `LazyLock` in tests (89s → 9.6s).
- Doc audit: stale README claims fixed, catalog updated after wrapper removal.

## [0.4.0] - 2026-03-15

### Added
- **Discourse module** restored: centering theory, dialogue, events, uncertain reference (feature-gated).
- **NuNER Zero / Zero-4k** backends registered; ONNX export scripts.
- **B2NER** backend registered.
- **ONNX export scripts** documented in BACKENDS.md.

### Changed
- Muxer 0.3.12 → 0.4.0 migration.
- `clump` 0.4 → 0.5, `innr` 0.1 → 0.2.
- Discourse feature wired through eval, CLI, and justfile.

## [0.3.0] - 2026-02-19

### Changed
- Workspace refactor: split into `anno-core`, `anno`, `anno-eval`, `anno-cli`, `anno-metrics`, `anno-graph`
- `PatternNER` → `RegexNER`
- Lib target renamed to `anno` (package name remains `anno-lib`); fixes `use anno::` in doctests and integration tests

### Added
- **GLiNER2** (`backends::gliner2`): Multi-task information extraction — NER, classification, structure extraction, and task composition via `TaskSchema`; ONNX and Candle backends
- **Coreference resolution**: T5-based seq2seq scaffold (`T5Coref`), graph-based iterative refinement (`GraphCoref`), mention-ranking (`MentionRankingCoref`), and rule-based fallback
- **Graph RAG** (`anno-core::graph`): `GraphDocument` with Neo4j Cypher and NetworkX/JSON-LD export; unified fractured-graph resolution via coref
- **Grounded entity representation** (`anno-core::grounded`): three-level Signal → Track → Identity hierarchy; `GroundedDocument` with dual spatial/chain indexes, HTML rendering, eval comparison
- **Task evaluation system** (`anno-eval`): dataset loaders, multi-objective LinUCB routing, regression detection, quality matrix, git-tagged scoring
- **Discourse analysis** (`anno::discourse`): centering theory, uncertain reference (ε-terms), abstract anaphora, shell nouns, event extraction
- `anno-metrics` crate: shared CorefChainStats and cluster evaluation primitives
- `anno-graph` crate: adapters between `anno-core` and `lattix` graph substrates
- Contextual backend routing via muxer 0.1.2 (LinUCB + objective manifold)

### Notes
- `T5Coref::resolve()` currently uses a rule-based heuristic fallback; full encoder/decoder ONNX loop is scaffolded but not yet wired
- Publishing to crates.io is paused pending resolution of internal path deps; see `docs/PUBLISH_STATUS.md`

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
- Entity uses character offsets consistently

## [0.1.0] - 2025-11-26

Initial release.
