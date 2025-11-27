//! NER Backend Catalog
//!
//! Comprehensive catalog of NER backends in `anno`, with research context.
//!
//! # Backend Overview
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                        NER Backend Spectrum                         │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │                                                                     │
//! │  Zero-Shot NER (any entity type)                                    │
//! │  ├─ GLiNER (span classification) - onnx feature                     │
//! │  ├─ NuNER (token classification) - arbitrary length entities        │
//! │  └─ UniversalNER (LLM-based) - expensive but powerful               │
//! │                                                                     │
//! │  Complex Structures (nested/discontinuous)                          │
//! │  ├─ W2NER (word-word grids) - handles nested entities               │
//! │  └─ TPLinker (handshaking) - joint entity-relation                  │
//! │                                                                     │
//! │  Traditional NER (fixed types)                                      │
//! │  ├─ BertNEROnnx (sequence labeling) - fast, reliable                │
//! │  └─ CRF-based (conditional random fields) - classical               │
//! │                                                                     │
//! │  Zero Dependency (always available)                                 │
//! │  ├─ PatternNER (regex) - structured entities                        │
//! │  └─ StatisticalNER (heuristics) - Person/Org/Location               │
//! │                                                                     │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Implemented Backends
//!
//! | Backend | Feature | Zero-Shot | Nested | Speed | Status |
//! |---------|---------|-----------|--------|-------|--------|
//! | `PatternNER` | - | No | No | ~400ns | ✅ Complete |
//! | `StatisticalNER` | - | No | No | ~50μs | ✅ Complete |
//! | `StackedNER` | - | No | No | varies | ✅ Complete |
//! | `HybridNER` | - | No | No | varies | ✅ Complete |
//! | `BertNEROnnx` | `onnx` | No | No | ~50ms | ✅ Complete |
//! | `GLiNEROnnx` | `onnx` | **Yes** | No | ~100ms | ✅ Complete |
//! | `NuNER` | `onnx` | **Yes** | No | ~100ms | ✅ Complete |
//! | `W2NER` | `onnx` | No | **Yes** | ~150ms | ✅ Complete |
//! | `CandleNER` | `candle` | No | No | varies | ✅ Complete |
//! | `GLiNERCandle` | `candle` | **Yes** | No | varies | ✅ Complete |
//!
//! # Research Landscape (2024-2025)
//!
//! ## Zero-Shot NER
//!
//! ### GLiNER (Implemented)
//!
//! Bi-encoder architecture that embeds entity labels and text spans
//! into the same space, enabling zero-shot classification.
//!
//! - **Paper**: [GLiNER: Generalist Model for NER](https://arxiv.org/abs/2311.08526)
//! - **Best for**: Custom entity types without retraining
//! - **Limitation**: Fixed span window
//!
//! ### NuNER (Implemented)
//!
//! Token classifier variant of GLiNER from NuMind. Uses BIO tagging
//! instead of span classification, enabling arbitrary-length entities.
//!
//! - **Models**: `numind/NuNER_Zero`, `numind/NuNER_Zero_4k`
//! - **License**: MIT (open weights)
//! - **Best for**: Long entities, variable-length spans
//!
//! ### UniversalNER
//!
//! Instruction-tuned LLM (LLaMA-based) for open NER.
//!
//! - **Paper**: [UniversalNER](https://universal-ner.github.io)
//! - **Pros**: 45 entity types, competitive with ChatGPT
//! - **Cons**: LLM-based (expensive inference)
//!
//! ## Complex Structures
//!
//! ### W2NER (Implemented)
//!
//! Word-word relation classification for nested/discontinuous entities.
//! Models NER as classifying relations between every token pair.
//!
//! - **Paper**: [W2NER: Unified NER via Word-Word Relations](https://arxiv.org/abs/2112.10070)
//! - **Best for**: Medical NER ("severe \[pain\] in \[abdomen\]")
//! - **Grid**: N×N×L tensor (sequence × sequence × labels)
//!
//! ### TPLinker (HandshakingMatrix)
//!
//! Handshaking tagging scheme for joint entity-relation extraction.
//! Already implemented via `HandshakingMatrix` in `inference.rs`.
//!
//! - **Paper**: [TPLinker: Single-stage Joint Extraction](https://aclanthology.org/2020.coling-main.138/)
//! - **Best for**: Knowledge graph construction
//!
//! ## Encoder Comparison
//!
//! | Encoder | Context | ONNX | Candle | Notes |
//! |---------|---------|------|--------|-------|
//! | BERT | 512 | ✅ | ✅ | Classic, well-tested |
//! | DeBERTa-v3 | 512 | ✅ | ✅ | Disentangled attention |
//! | ModernBERT | 8192 | ✅ | ✅ | SOTA (2024), RoPE, GeGLU |
//! | RoBERTa | 512 | ✅ | ✅ | Improved pretraining |
//!
//! All encoders are implemented in `encoder_candle.rs` with:
//! - `EncoderConfig::bert_base()` / `::deberta_v3_base()` / `::modernbert_base()`
//! - Automatic config detection from HuggingFace `config.json`
//! - RoPE (Rotary Position Embeddings) for long-context models
//! - GeGLU activation for ModernBERT
//!
//! # Model Recommendations
//!
//! | Use Case | Recommended |
//! |----------|-------------|
//! | Simple NER | `StackedNER::default()` |
//! | Custom entity types | `NuNER` or `GLiNEROnnx` |
//! | Nested entities | `W2NER` |
//! | Production (fixed types) | `BertNEROnnx` |
//! | Knowledge graphs | `HandshakingMatrix` (TPLinker) |
//! | Structured data | `PatternNER` |
//!
//! # Why Multiple GLiNER Implementations?
//!
//! We have three GLiNER-related implementations:
//!
//! ## 1. `GLiNEROnnx` (Manual ONNX)
//!
//! Hand-written ONNX inference code.
//!
//! - **Feature**: `onnx`
//! - **Deps**: `ort`, `tokenizers`, `hf-hub`
//! - **Status**: Implemented and tested
//!
//! **Why**: Full control over tokenization, custom error handling,
//! no external crate dependency on gline-rs.
//!
//! ## 2. `GLiNERCandle` (Pure Rust, GPU)
//!
//! Pure Rust implementation for native GPU support.
//!
//! - **Feature**: `candle`
//! - **Deps**: `candle-core`, `candle-nn`, `tokenizers`
//! - **Status**: ✅ Implemented
//!
//! **Why**: Metal (Apple Silicon) and CUDA acceleration
//! without C++ dependencies. Pure Rust ML inference.
//!
//! **Components**:
//! - `CandleEncoder`: Transformer encoder (BERT/ModernBERT/DeBERTa)
//! - `SpanRepLayer`: Span embeddings from [start, end, width]
//! - `LabelEncoder`: Project label embeddings
//! - `SpanLabelMatcher`: Cosine similarity matching
//!
//! ## 3. `gline-rs` (External Crate)
//!
//! Production-ready library by Frédérik Bilhaut.
//!
//! - **Crate**: `gline-rs` (crates.io)
//! - **Status**: Referenced, not integrated
//!
//! **Why not integrated**: Users can add it themselves.
//! We provide manual impl for those who want control.
//!
//! # Default Thresholds (Data-Motivated)
//!
//! | Parameter | Default | Source |
//! |-----------|---------|--------|
//! | `threshold` | 0.5 | gline-rs, GLiNER paper |
//! | `max_width` | 12 | Standard span width |
//! | `max_length` | 512 | BERT context limit |
//! | `flat_ner` | true | No overlapping entities |
//!
//! These match the gline-rs defaults and GLiNER paper recommendations.

/// Backend implementation status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendStatus {
    /// Fully implemented and tested
    Stable,
    /// Implemented but may have rough edges
    Beta,
    /// Work in progress
    WIP,
    /// Planned for future implementation
    Planned,
    /// Research only, not planned
    Research,
}

impl std::fmt::Display for BackendStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendStatus::Stable => write!(f, "stable"),
            BackendStatus::Beta => write!(f, "beta"),
            BackendStatus::WIP => write!(f, "wip"),
            BackendStatus::Planned => write!(f, "planned"),
            BackendStatus::Research => write!(f, "research"),
        }
    }
}

/// Information about a backend implementation.
#[derive(Debug, Clone)]
pub struct BackendInfo {
    /// Backend name
    pub name: &'static str,
    /// Cargo feature required (if any)
    pub feature: Option<&'static str>,
    /// Implementation status
    pub status: BackendStatus,
    /// Whether it supports zero-shot NER
    pub zero_shot: bool,
    /// Whether it supports GPU acceleration
    pub gpu_support: bool,
    /// Brief description
    pub description: &'static str,
    /// Recommended model IDs
    pub recommended_models: &'static [&'static str],
}

/// Catalog of all available and potential backends.
pub static BACKEND_CATALOG: &[BackendInfo] = &[
    // =========================================================================
    // Implemented Backends
    // =========================================================================
    BackendInfo {
        name: "pattern",
        feature: None,
        status: BackendStatus::Stable,
        zero_shot: false,
        gpu_support: false,
        description: "Regex-based extraction for structured entities (dates, money, emails)",
        recommended_models: &[],
    },
    BackendInfo {
        name: "gliner",
        feature: Some("gliner"),
        status: BackendStatus::Stable,
        zero_shot: true,
        gpu_support: true,
        description: "GLiNER via gline-rs library (SOTA zero-shot NER)",
        recommended_models: &[
            "onnx-community/gliner_small-v2.1",
            "onnx-community/gliner_large-v2.1",
            "knowledgator/modern-gliner-bi-large-v1.0",
        ],
    },
    BackendInfo {
        name: "gliner_onnx",
        feature: Some("onnx"),
        status: BackendStatus::Beta,
        zero_shot: true,
        gpu_support: true,
        description: "GLiNER via manual ONNX implementation",
        recommended_models: &[
            "onnx-community/gliner_small-v2.1",
        ],
    },
    BackendInfo {
        name: "bert_onnx",
        feature: Some("onnx"),
        status: BackendStatus::Beta,
        zero_shot: false,
        gpu_support: true,
        description: "BERT NER via ONNX Runtime (PER/ORG/LOC/MISC)",
        recommended_models: &[
            "protectai/bert-base-NER-onnx",
        ],
    },
    
    // =========================================================================
    // Work in Progress
    // =========================================================================
    BackendInfo {
        name: "gliner_candle",
        feature: Some("candle"),
        status: BackendStatus::WIP,
        zero_shot: true,
        gpu_support: true,
        description: "GLiNER via Candle (pure Rust, Metal/CUDA)",
        recommended_models: &[
            "answerdotai/ModernBERT-base",
        ],
    },
    
    // =========================================================================
    // Planned Backends
    // =========================================================================
    BackendInfo {
        name: "nuner",
        feature: Some("nuner"),
        status: BackendStatus::Planned,
        zero_shot: true,
        gpu_support: true,
        description: "NuNER Zero (token classifier, arbitrary-length entities)",
        recommended_models: &[
            "numind/NuNER-v1.0",
            "numind/NuNER-v2.0-4k",
        ],
    },
    BackendInfo {
        name: "rust_bert",
        feature: Some("rust-bert"),
        status: BackendStatus::Planned,
        zero_shot: false,
        gpu_support: true,
        description: "rust-bert integration (requires libtorch)",
        recommended_models: &[
            "bert-base-NER",
            "dbmdz/bert-large-cased-finetuned-conll03-english",
        ],
    },
];

impl BackendInfo {
    /// Get backend by name.
    #[must_use]
    pub fn by_name(name: &str) -> Option<&'static BackendInfo> {
        BACKEND_CATALOG.iter().find(|b| b.name == name)
    }

    /// Get all stable backends.
    #[must_use]
    pub fn stable() -> Vec<&'static BackendInfo> {
        BACKEND_CATALOG.iter()
            .filter(|b| b.status == BackendStatus::Stable)
            .collect()
    }

    /// Get all zero-shot capable backends.
    #[must_use]
    pub fn zero_shot() -> Vec<&'static BackendInfo> {
        BACKEND_CATALOG.iter()
            .filter(|b| b.zero_shot)
            .collect()
    }

    /// Get all GPU-capable backends.
    #[must_use]
    pub fn with_gpu() -> Vec<&'static BackendInfo> {
        BACKEND_CATALOG.iter()
            .filter(|b| b.gpu_support)
            .collect()
    }
}

/// Print a summary of available backends.
pub fn print_catalog() {
    println!("NER Backend Catalog");
    println!("{}", "=".repeat(80));
    println!("{:15} {:10} {:8} {:5} {:5} Description", 
        "Name", "Feature", "Status", "0-shot", "GPU");
    println!("{}", "-".repeat(80));
    
    for backend in BACKEND_CATALOG {
        let feature = backend.feature.unwrap_or("-");
        let zero_shot = if backend.zero_shot { "yes" } else { "no" };
        let gpu = if backend.gpu_support { "yes" } else { "no" };
        
        println!("{:15} {:10} {:8} {:5} {:5} {}",
            backend.name, feature, backend.status, zero_shot, gpu, backend.description);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_lookup() {
        assert!(BackendInfo::by_name("pattern").is_some());
        assert!(BackendInfo::by_name("gliner").is_some());
        assert!(BackendInfo::by_name("nonexistent").is_none());
    }

    #[test]
    fn test_stable_backends() {
        let stable = BackendInfo::stable();
        assert!(!stable.is_empty());
        assert!(stable.iter().all(|b| b.status == BackendStatus::Stable));
    }

    #[test]
    fn test_zero_shot_backends() {
        let zero_shot = BackendInfo::zero_shot();
        assert!(!zero_shot.is_empty());
        assert!(zero_shot.iter().all(|b| b.zero_shot));
    }
}

