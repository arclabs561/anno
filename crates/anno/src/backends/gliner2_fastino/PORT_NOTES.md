# Phase 3 port notes (2026-05-05)

Source: github.com/SemplificaAI/gliner2-rs (Apache-2.0).
Specifically: `rust_component/src/lib_v2.rs` — the v2 IOBinding-ready engine.

## Why v2 not v1

v1 has 5 sessions (encoder, span_rep, count_lstm, count_pred, classifier)
with token-gathering, schema-gathering, ArgMax over count logits, and
Einsum-style scoring done in Rust. v2 fuses those ops into 3 additional
ONNX graphs (token_gather, schema_gather, count_pred_argmax,
count_lstm_fixed) for IOBinding-friendly zero-copy chaining. The
SemplificaAI pin (`SemplificaAI/gliner2-multi-v1-onnx`, commit
51d4a15c) ships the v2 graphs in `fp32/` and `fp32_v2/` subdirs. We
port v2.

## Standard mode vs IOBinding mode

Phase 3 ships standard mode (extract_standard at lib_v2.rs:660-897).
The pipeline runs the 8 sessions sequentially with `Tensor::from_array`
+ `try_extract_tensor` round trips between Rust ndarray and ort tensors.

IOBinding mode (extract_iobinding at lib_v2.rs:285-660) keeps tensors
in a single ort allocator across session boundaries (typically GPU
device memory). 2-3× speedup, no functional difference. Phase 3.5
follow-up — separate plan.

## Mechanical transformations applied during port

| Upstream | This crate |
|---|---|
| `anyhow::Result<_>` | `Result<_, super::errors::Error>` (or `crate::Result`) |
| `ExtractedEntity { score, label, text, start_tok, end_tok }` | `crate::Entity` (char offsets, not token offsets) |
| `ExtractedClassification { task_name, label, score }` | tuple `(String, f32)` for the internal `classify` API |
| `ExtractedRelation` | not ported in Phase 3 (relations are a separate Track / Phase 2.5) |
| `Gliner2Config { max_width, ... }` | hardcoded `max_width: 8` per spec; max_count 20 |
| `tokenizer.encode(.., false)` panics | propagate as `Error::Tokenizer` |
| `RwLock<ExecutionMode>` | not needed (we ship standard mode only) |
