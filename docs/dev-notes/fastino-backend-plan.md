# Fastino GLiNER2 backend (design plan)

Work-in-progress design notes for a future `gliner2_fastino` backend in
`crates/anno/src/backends/`. See issue arclabs561/anno#17 for surface context.

## Why a separate backend

`anno::backends::gliner_multitask` loads `onnx-community/gliner-multitask-large-v0.5`
(GLiNER v1 with task-conditioned label prompts; Stepanov & Shtopko 2024,
arXiv:2406.12925). The fastino-ai GLiNER2 architecture (Zaratiana et al. 2025,
arXiv:2507.18546) is unrelated:

- Different special-token vocabulary: `[P]`, `[E]`, `[C]`, `[L]`, `[R]`,
  `[SEP_STRUCT]`, `[SEP_TEXT]`. Tokens are string-added at load-time;
  integer IDs are read from `tokenizer.json`, not hardcoded.
- Different head structure: count-predictor MLP from `[P]` embedding plus
  occurrence ID embeddings for per-instance-attribute decoding.
- Different encoder configuration: the `counting_layer` config field
  switches between `count_lstm` (base), `count_lstm_moe` (large), and
  `count_lstm_v2` (multi). Parameters are not interchangeable across
  fastino model variants.

The hardcoded special-token IDs in `gliner_multitask`
(`<<ENT>>=128002`, `<<SEP>>=128003`) are the gliner-multitask-large-v0.5
vocabulary. fastino models will not load through the existing backend.

## Existing prior art (NOT dependency candidates)

| Repo | License | crates.io? | Why not a dep |
|------|---------|------------|---------------|
| `SemplificaAI/gliner2-rs` | Apache-2.0 | no (private workspace member) | private; ort version mismatch (rc.9 vs anno's rc.12); 3-tuple return shape; token offsets not char offsets |
| `paul-english/gliner2_rs` | Apache-2.0 | yes (`gliner2 = "0.1.3"`) | API contract mismatch (`ExtractionOutput` with recursive `TaskValue`); wraps Candle + tch-rs only |
| `fbilhaut/gline-rs` | Apache-2.0 | yes (`gline-rs = "1"`) | GLiNER v1 only; issue #15 stalled since 2025-12; same ort version mismatch |
| `fastino-ai/GLiNER2` | Apache-2.0 | n/a (Python) | reference impl; DeBERTa-v2 SentencePiece tokenizer; single-pass schema concatenation |
| `lmoe/gliner2-onnx` | (TBD) | n/a | community ONNX export script; covers `gliner2-large-v1` and `gliner2-multi-v1` only (NOT `gliner2-base-v1`); no structured-extraction support |

Port. Do not depend.

## Implementation plan

### Phase 1: NER + classification heads (~2 weeks)

1. New module `crates/anno/src/backends/gliner2_fastino/`.
2. ONNX export route: either ship a script mirroring `lmoe/gliner2-onnx`,
   or pin to a known-good external pre-export
   (`SemplificaAI/gliner2-multi-v1-onnx`).
3. Port `processor.rs` from `SemplificaAI/gliner2-rs` (Apache-2.0,
   attribute in source comments). Special-token registration from
   `tokenizer.json`. Prompt format:
   `( [P] task_name ( [E] label1 [E] label2 ) ) [SEP_TEXT] words...`
4. ONNX session loading via `ort = rc.12` (anno's pin).
5. NER head: span scoring via dot-product similarity (Eq. 1 of the
   GLiNER2 paper). Implement `ZeroShotNER`.
6. Classification head: MLP over `[L]` token embeddings. New
   `TextClassifier` trait or internal-only for now.
7. Tests against `fastino/gliner2-multi-v1`: load + extract on fixture
   text. Mark `#[ignore]` if model download is required.

### Phase 2: Structure extraction (~2 weeks)

8. Count-predictor head from `[P]` embedding (20-class MLP for 0-19
   instances).
9. Occurrence ID embeddings plus per-instance-attribute span scoring.
10. JSON output schema mapping. Surface as `extract_structure(text, schema)`.

### Phase 3: V2 IOBinding pipeline (~1 week, perf only)

11. Port the 8-session IOBinding chain from gliner2-rs `lib_v2.rs` for
    zero-copy inference between sessions.
12. OS-aware artifact selection: `_iobinding` variants on Linux/Windows,
    `fp16` fallback on macOS.

## Improvement ideas surfaced by the research (not blocked on fastino)

These are worth applying to anno more broadly. Each is an independent
follow-up:

1. **Per-label thresholds.** Each entity type gets an optional override
   for the global threshold. Reference pattern: paul-english/gliner2_rs
   `ExtractionMetadata` (`metadata.get(&name).and_then(|m| m.threshold)
   .unwrap_or(global)`).
2. **Label descriptions.** Free-text descriptions per label produce a
   documented quality boost in the GLiNER paper. Currently `ZeroShotNER::
   extract_with_types(text, &[&str], threshold)` accepts labels only;
   adding `extract_with_descriptions(text, &[(&str, &str)], threshold)`
   exposes the feature without breaking the existing API.
3. **Streaming batch with callback.** Large-file workloads benefit from
   incremental output. Reference: paul-english/gliner2_rs
   `batch_extract_streaming` with `on_batch` callback.
4. **`PerSample` batch schema.** Each text in a batch can have its own
   label set. Reference: `BatchSchemaMode::Shared | PerSample`.
5. **Parity test** between `gliner_onnx` and `gliner_candle` (and later
   gliner2_fastino + Python reference). `max_abs_diff < 5e-3` bound on
   score vectors.
6. **Macro-based backend method sharing.** Avoid duplication between
   `*Onnx`-style and `*Candle`-style structs without trait objects.
   Reference: paul-english/gliner2_rs `impl_gliner2_api!`.
7. **Backend env var override.** Pick among compiled-in features at
   runtime (`ANNO_BACKEND=candle`).
8. **README benchmark tables** with explicit reproduction commands and
   sentinel-block auto-regeneration.

## License attribution

Source comments must attribute upstream Apache-2.0 work, e.g.:

```rust
// Adapted from SemplificaAI/gliner2-rs (Apache-2.0):
// https://github.com/SemplificaAI/gliner2-rs/blob/main/rust_component/src/processor.rs
```

## References

- Issue: arclabs561/anno#17
- Related closed PR: arclabs561/anno#16 (config-file fallback in
  `GLiNERMultitaskCandle`, landed independently 2026-04-25)

Papers:

- Zaratiana et al. 2024, "GLiNER: Generalist Model for NER",
  arXiv:2311.08526
- Stepanov & Shtopko 2024, "GLiNER multi-task: Generalist Lightweight
  Model for Various Information Extraction Tasks", arXiv:2406.12925
- Zaratiana et al. 2025, "GLiNER2: An Efficient Multi-Task Information
  Extraction System with Schema-Driven Interface", arXiv:2507.18546
  (EMNLP 2025 System Demonstrations)

External implementations:

- github.com/urchade/GLiNER (Python ref, GLiNER v1 plus multi-task)
- github.com/fastino-ai/GLiNER2 (Python ref, GLiNER2)
- github.com/SemplificaAI/gliner2-rs (Rust port, not on crates.io)
- github.com/paul-english/gliner2_rs (Rust crate, on crates.io)
- github.com/fbilhaut/gline-rs (Rust GLiNER v1 only)
- github.com/lmoe/gliner2-onnx (community ONNX export tooling)
