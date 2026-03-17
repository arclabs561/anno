//! NER backend implementations.
//!
//! Each backend implements the `Model` trait for consistent usage.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────┐
//! │ Layer 3: ML Backends (feature-gated)                │
//! │                                                     │
//! │  Zero-Shot NER (any entity type):                   │
//! │   - GLiNER: Bi-encoder span classification          │
//! │   - NuNER: Token classification (arbitrary length)  │
//! │                                                     │
//! │  Complex Structures (nested/discontinuous):         │
//! │   - W2NER: Word-word relation grids                 │
//! │                                                     │
//! │  Traditional (fixed types):                         │
//! │   - BertNEROnnx: Sequence labeling                  │
//! ├─────────────────────────────────────────────────────┤
//! │ Layer 2: HeuristicNER (zero deps)                   │
//! │   Person/Org/Location via heuristics                │
//! ├─────────────────────────────────────────────────────┤
//! │ Layer 1: RegexNER (zero deps)                     │
//! │   Date/Time/Money/Email/URL/Phone                   │
//! └─────────────────────────────────────────────────────┘
//! ```
//!
//! # Backend Comparison
//!
//! | Backend | Feature | Zero-Shot | Relations | Notes |
//! |---------|---------|-----------|-----------|-------|
//! | `StackedNER` | - | No | No | Composable with any backend |
//! | `EnsembleNER` | - | No | No | Weighted voting across backends |
//! | `RegexNER` | - | No | No | Structured entities only |
//! | `HeuristicNER` | - | No | No | Heuristic baseline |
//! | `CrfNER` | - | No | No | CRF statistical baseline |
//! | `HmmNER` | - | No | No | HMM statistical baseline |
//! | `LexiconNER` | - | No | No | Dictionary lookup |
//! | `GLiNEROnnx` | `onnx` | Yes | No | Span-based zero-shot |
//! | `GLiNER2Onnx` | `onnx` | Yes | Yes | Multi-task (NER + RE) |
//! | `NuNER` | `onnx` | Yes | No | Token-based zero-shot |
//! | `W2NER` | `onnx` | No | No | Grid-based, nested entities |
//! | `BertNEROnnx` | `onnx` | No | No | Traditional fixed-label NER |
//! | `TPLinker` | `onnx` | No | Yes | Handshaking matrix RE |
//! | `UniversalNER` | `llm` | Yes | No | LLM-based extraction |
//! | `CandleNER` | `candle` | No | No | Pure-Rust inference |
//! | `GLiNERCandle` | `candle` | Yes | No | Pure-Rust GLiNER |
//! | `BiLstmCrfNER` | - | No | No | BiLSTM-CRF baseline |
//!
//! # When to Use What
//!
//! - **Default choice**: `StackedNER::default()` - cascading: ML first (if available), then heuristic + pattern
//! - **Hybrid approach**: `StackedNER` with ML backends - combine ML accuracy with pattern speed
//! - **Custom types**: `GLiNER` or `NuNER` - zero-shot, any entity type
//! - **Nested entities**: `W2NER` - handles overlapping spans
//! - **Structured data**: `RegexNER` - dates, emails, money
//!
//! # Backend Combination Design Space
//!
//! Two approaches for combining multiple backends:
//!
//! | Combiner | Execution | Conflict Resolution | Best For |
//! |----------|-----------|---------------------|----------|
//! | [`StackedNER`] | Sequential (cascade) | Priority/LongestSpan/HighestConf | Production, latency |
//! | [`EnsembleNER`] | Parallel (all) | Weighted voting + agreement | Maximum accuracy |
//!
//! **StackedNER** runs backends in layer order. Earlier layers claim spans first.
//! Good for: fast execution, structured patterns + ML fill-in.
//!
//! **EnsembleNER** runs ALL backends, groups overlapping spans into conflict clusters,
//! and resolves via weighted voting with type-conditioned weights and agreement bonuses.
//! Good for: maximum accuracy when latency allows.
//!
//! Both accept any `Model` implementation - they're fully composable with ML backends.
//!
//! # Quick Start
//!
//! Zero-dependency default (Pattern + Heuristic):
//!
//! ```rust
//! use anno::{Model, StackedNER};
//!
//! let ner = StackedNER::default();
//! let entities = ner.extract_entities("Dr. Smith charges $100/hr", None).unwrap();
//! ```
//!
//! Custom stack with pattern + heuristic:
//!
//! ```rust
//! use anno::{Model, RegexNER, HeuristicNER, StackedNER};
//! use anno::backends::stacked::ConflictStrategy;
//!
//! let ner = StackedNER::builder()
//!     .layer(RegexNER::new())
//!     .layer(HeuristicNER::new())
//!     .strategy(ConflictStrategy::LongestSpan)
//!     .build();
//! ```
//!
//! **StackedNER is fully composable** - you can combine ML backends with pattern/heuristic layers:
//!
//! ```rust,no_run
//! #[cfg(feature = "onnx")]
//! {
//! use anno::{Model, StackedNER, GLiNEROnnx, RegexNER, HeuristicNER};
//! use anno::backends::stacked::ConflictStrategy;
//!
//! // ML-first: ML runs first, then patterns fill gaps
//! let ner = StackedNER::with_ml_first(
//!     Box::new(GLiNEROnnx::new("onnx-community/gliner_small-v2.1").unwrap())
//! );
//!
//! // ML-fallback: patterns/heuristics first, ML as fallback
//! let ner = StackedNER::with_ml_fallback(
//!     Box::new(GLiNEROnnx::new("onnx-community/gliner_small-v2.1").unwrap())
//! );
//!
//! // Custom stack: any combination of backends
//! let ner = StackedNER::builder()
//!     .layer(RegexNER::new())           // High-precision structured entities
//!     .layer_boxed(Box::new(GLiNEROnnx::new("onnx-community/gliner_small-v2.1").unwrap()))  // ML layer
//!     .layer(HeuristicNER::new())       // Quick named entities
//!     .strategy(ConflictStrategy::HighestConf)  // Resolve conflicts by confidence
//!     .build();
//! }
//! ```

/// Macros for generating feature-gated backend stubs.
#[macro_use]
pub(crate) mod macros;

/// Shared HuggingFace model loading and ONNX session construction utilities.
#[cfg(any(feature = "onnx", feature = "candle"))]
pub(crate) mod hf_loader;

/// Coreference resolution backends (trait, neural, heuristic).
pub mod coref;

// Always available (zero deps beyond std)
/// BiLSTM + CRF NER - neural baseline from 2015-2018.
///
/// Bidirectional LSTM with Conditional Random Field output layer.
/// The dominant neural NER architecture before BERT/transformers.
pub mod bilstm_crf;
pub mod catalog;
pub mod crf;
pub mod heuristic;
pub mod inference;
pub mod lexicon;
pub mod nuner;
pub(crate) mod ort_compat;
pub mod regex;
pub mod stacked;
pub mod tplinker;
pub mod w2ner;

/// Ensemble NER - weighted voting across multiple backends.
///
/// Unlike `StackedNER` (priority-based layers), `EnsembleNER` collects
/// candidates from ALL backends and resolves conflicts via weighted voting
/// with agreement bonuses.
pub mod ensemble;

/// Hidden Markov Model NER - classical statistical approach.
///
/// Implements HMM-based sequence labeling, the dominant approach from the 1990s
/// before CRFs. Useful as a baseline and for understanding NER history.
pub mod hmm;

/// Chunked extraction and overlap deduplication for long text.
pub mod chunking;

/// Map a backend name (stable ID used in stacked/ensemble compositions) to an
/// [`ExtractionMethod`](anno_core::ExtractionMethod).
///
/// Shared by `StackedNER` and `EnsembleNER` so the mapping stays consistent.
pub(crate) fn method_for_backend_name(name: &str) -> anno_core::ExtractionMethod {
    match name {
        // Stable IDs used by built-in compositions.
        "regex" => anno_core::ExtractionMethod::Pattern,
        "heuristic" => anno_core::ExtractionMethod::Heuristic,
        // Everything else: treat as neural by default.
        _ => anno_core::ExtractionMethod::Neural,
    }
}

// =============================================================================
// Tests for shared utilities
// =============================================================================

#[cfg(test)]
mod tests {
    use super::method_for_backend_name;
    use anno_core::ExtractionMethod;

    // -------------------------------------------------------------------------
    // method_for_backend_name: exact stable IDs
    // -------------------------------------------------------------------------

    #[test]
    fn regex_maps_to_pattern() {
        assert_eq!(
            method_for_backend_name("regex"),
            ExtractionMethod::Pattern,
            "\"regex\" must map to Pattern"
        );
    }

    #[test]
    fn heuristic_maps_to_heuristic() {
        assert_eq!(
            method_for_backend_name("heuristic"),
            ExtractionMethod::Heuristic,
            "\"heuristic\" must map to Heuristic"
        );
    }

    // -------------------------------------------------------------------------
    // method_for_backend_name: wildcard / unknown IDs -> Neural
    // -------------------------------------------------------------------------

    #[test]
    fn unknown_ids_map_to_neural() {
        // Any backend name that is not one of the three stable IDs must fall
        // back to Neural.  This covers ML backends (gliner, bert, nuner, candle)
        // as well as completely arbitrary strings.
        let neural_names = [
            "gliner",
            "gliner-candle",
            "GLiNER-ONNX",
            "bert-ner-onnx",
            "bert-onnx",
            "nuner",
            "NuNER",
            "w2ner",
            "llm",
            "custom-backend",
            "",
            "REGEX",     // case-sensitive: must NOT match "regex"
            "Heuristic", // case-sensitive: must NOT match "heuristic"
        ];
        for name in neural_names {
            assert_eq!(
                method_for_backend_name(name),
                ExtractionMethod::Neural,
                "\"{}\" should map to Neural (unknown/wildcard ID)",
                name
            );
        }
    }

    #[test]
    fn ensemble_nested_id_maps_to_neural() {
        // An EnsembleNER's transparent name looks like "ensemble(regex|heuristic)".
        // The composite string is not one of the stable keys, so it must map to
        // Neural — the caller is responsible for extracting inner provenance.
        let name = "ensemble(regex|heuristic)";
        assert_eq!(
            method_for_backend_name(name),
            ExtractionMethod::Neural,
            "composite ensemble ID '{}' should map to Neural",
            name
        );
    }

    #[test]
    fn stacked_id_maps_to_neural() {
        let name = "stacked(regex|heuristic)";
        assert_eq!(
            method_for_backend_name(name),
            ExtractionMethod::Neural,
            "composite stacked ID '{}' should map to Neural",
            name
        );
    }

    // -------------------------------------------------------------------------
    // Stability: the two exact matches remain stable under whitespace
    // -------------------------------------------------------------------------

    #[test]
    fn match_is_exact_no_trimming() {
        // Whitespace-padded variants are NOT trimmed; they should fall through
        // to Neural, confirming the match is exact byte-equality.
        for padded in &[" regex", "regex ", " heuristic", "heuristic\n"] {
            assert_eq!(
                method_for_backend_name(padded),
                ExtractionMethod::Neural,
                "whitespace-padded name '{}' should not match a stable ID",
                padded
            );
        }
    }
}

// Advanced backends
#[cfg(feature = "onnx")]
pub mod albert;
#[cfg(feature = "onnx")]
pub mod deberta_v3;
pub mod gliner_poly;
/// GLiREL: Zero-shot relation extraction via ONNX.
///
/// Uses a DeBERTa-v3 encoder with relation scoring head from the GLiREL family.
/// Export models with `scripts/export_glirel_onnx.py`.
pub mod glirel;
pub mod universal_ner;

// LLM client abstraction (config, providers, mock)
pub(crate) mod llm_client;

// LLM-based NER prompting (CodeNER-style)
pub(crate) mod llm_prompt;

// GLiNER via ONNX (uses same feature as other ONNX models)

// ONNX implementations
#[cfg(feature = "onnx")]
pub mod gliner_onnx;

#[cfg(feature = "onnx")]
pub mod onnx;

// Pure Rust via Candle
#[cfg(feature = "candle")]
pub mod candle;

#[cfg(feature = "candle")]
pub mod encoder_candle;

#[cfg(feature = "candle")]
pub mod gliner_candle;

// GLiNER2 multi-task extraction (ONNX or Candle)
#[cfg(any(feature = "onnx", feature = "candle"))]
pub mod gliner2;

// Re-exports (always available)
pub use bilstm_crf::BiLstmCrfNER;
pub use crf::CrfNER;
pub use ensemble::EnsembleNER;
pub use heuristic::HeuristicNER;
pub use lexicon::LexiconNER;
pub use nuner::NuNER;
pub use regex::RegexNER;
pub use stacked::{ConflictStrategy, StackedNER};

pub use tplinker::TPLinker;
pub use w2ner::{W2NERConfig, W2NERRelation, W2NER};

// Advanced backends
#[cfg(feature = "onnx")]
pub use albert::ALBERTNER;
#[cfg(feature = "onnx")]
pub use deberta_v3::DeBERTaV3NER;
pub use gliner_poly::GLiNERPoly;
pub use glirel::GLiREL;
pub use universal_ner::UniversalNER;

// Re-exports (feature-gated)
#[cfg(feature = "onnx")]
pub use gliner_onnx::GLiNEROnnx;

#[cfg(feature = "onnx")]
pub use onnx::BertNEROnnx;

#[cfg(feature = "candle")]
pub use candle::CandleNER;

#[cfg(feature = "candle")]
pub use encoder_candle::{EncoderArchitecture, EncoderConfig};

#[cfg(feature = "candle")]
pub use gliner_candle::GLiNERCandle;

// GLiNER2 multi-task model
#[cfg(any(feature = "onnx", feature = "candle"))]
pub use gliner2::{
    ClassificationResult, ClassificationTask, EntityTask, ExtractedStructure, ExtractionResult,
    FieldType, GLiNER2, StructureTask, StructureValue, TaskSchema,
};

#[cfg(feature = "onnx")]
pub use gliner2::GLiNER2Onnx;

#[cfg(feature = "candle")]
pub use gliner2::GLiNER2Candle;

// CorefCluster is always available (lives in coref::resolve, not feature-gated).
pub use coref::CorefCluster;

// T5 coreference
#[cfg(feature = "onnx")]
pub use coref::t5::{T5Coref, T5CorefConfig};

// F-coref neural coreference
#[cfg(feature = "onnx")]
pub use coref::fcoref::{FCoref, FCorefConfig};

// Config re-exports (for quantization control)
#[cfg(feature = "onnx")]
pub use gliner_onnx::GLiNERConfig;

#[cfg(feature = "onnx")]
pub use onnx::BertNERConfig;

// Coreference resolution trait (from anno-core, always available)
pub use anno_core::CoreferenceResolver;

// Unified coref backend trait
pub use coref::resolve::CorefBackend;

// Classical HMM NER (zero deps)
pub use hmm::{HmmConfig, HmmNER};

// Chunking and overlap deduplication
pub use chunking::{deduplicate_overlapping, ChunkConfig, OverlapStrategy};

// Simple rule-based coreference resolvers.
#[cfg(feature = "analysis")]
pub use coref::simple::{CorefConfig, SimpleCorefResolver};
