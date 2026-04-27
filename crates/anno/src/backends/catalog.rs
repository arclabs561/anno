//! NER Backend Catalog
//!
//! This module is documentation-only: it describes the set of backends that live
//! under `crate::backends` and gives a few “where to start” pointers.
//!
//! Keep in mind:
//! - Many backends are **feature-gated** (`onnx`, `candle`, etc.).
//! - Any “speed” or “quality” comparisons belong in the eval harness, not in
//!   rustdoc prose.
//!
//! Paper pointers (context only):
//! - GLiNER: arXiv:2311.08526
//! - UniversalNER: arXiv:2308.03279
//! - W2NER: arXiv:2112.10070
//! - TPLinker: `https://aclanthology.org/2020.coling-main.138/`
//!
//! Common configuration knobs you will see across GLiNER-like implementations:
//! - `threshold`: score cutoff for accepting a span
//! - `max_width`: maximum span width considered
//! - `max_length`: maximum input length per window/chunk
//! - `flat_ner`: whether to enforce non-overlapping entities

/// Backend implementation status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendStatus {
    /// Fully implemented and tested
    Stable,
    /// Implemented but may have rough edges
    Beta,
    /// Work in progress
    WIP,
}

impl std::fmt::Display for BackendStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendStatus::Stable => write!(f, "stable"),
            BackendStatus::Beta => write!(f, "beta"),
            BackendStatus::WIP => write!(f, "wip"),
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
        name: "heuristic",
        feature: None,
        status: BackendStatus::Stable,
        zero_shot: false,
        gpu_support: false,
        description: "Heuristic NER baseline (capitalization + context)",
        recommended_models: &[],
    },
    BackendInfo {
        name: "stacked",
        feature: None,
        status: BackendStatus::Stable,
        zero_shot: false,
        gpu_support: false,
        description: "Stacked NER (pattern + heuristic; default no-ML baseline)",
        recommended_models: &[],
    },
    BackendInfo {
        name: "ensemble",
        feature: None,
        status: BackendStatus::Beta,
        zero_shot: false,
        gpu_support: false,
        description: "Ensemble NER (weighted voting across backends)",
        recommended_models: &[],
    },
    BackendInfo {
        name: "crf",
        feature: None,
        status: BackendStatus::Stable,
        zero_shot: false,
        gpu_support: false,
        description: "CRF sequence labeling baseline (optional trained weights)",
        recommended_models: &[],
    },
    BackendInfo {
        name: "hmm",
        feature: None,
        status: BackendStatus::Stable,
        zero_shot: false,
        gpu_support: false,
        description: "HMM sequence labeling baseline (optional bundled params)",
        recommended_models: &[],
    },
    BackendInfo {
        name: "heuristic_crf",
        feature: None,
        status: BackendStatus::Stable,
        zero_shot: false,
        gpu_support: false,
        description: "CRF sequence labeling with heuristic emission features (capitalization, word shape, gazetteer)",
        recommended_models: &[],
    },
    BackendInfo {
        name: "tplinker",
        feature: None,
        status: BackendStatus::Beta,
        zero_shot: false,
        gpu_support: true,
        description: "Joint entity-relation extraction via handshaking tagging (Wang et al., COLING 2020; ONNX neural with onnx feature, heuristic fallback otherwise)",
        recommended_models: &[],
    },
    BackendInfo {
        name: "universal_ner",
        feature: Some("llm"),
        status: BackendStatus::Beta,
        zero_shot: true,
        gpu_support: true,
        description: "UniversalNER (LLM-backed zero-shot via OpenRouter/Anthropic/Groq/Ollama; configurable model)",
        recommended_models: &[
            "google/gemini-2.5-flash-lite",
            "anthropic/claude-haiku-4.5",
            "deepseek/deepseek-v3.2",
            "llama-3.3-70b-versatile",
        ],
    },
    BackendInfo {
        name: "gliner",
        feature: Some("onnx"),
        status: BackendStatus::Stable,
        zero_shot: true,
        gpu_support: true,
        // Smart-default alias: routes to `gliner_onnx` when the `onnx` feature
        // is on (preferred), falls back to `gliner_candle` when only `candle`
        // is on. Pick this when you want GLiNER and don't care which backend
        // resolves it. Pick `gliner_onnx` or `gliner_candle` to force one.
        description: "GLiNER zero-shot NER (smart default; prefers ONNX, falls back to Candle)",
        recommended_models: &[crate::models::GLINER, "onnx-community/gliner_large-v2.1"],
    },
    BackendInfo {
        name: "gliner_onnx",
        feature: Some("onnx"),
        status: BackendStatus::Beta,
        zero_shot: true,
        gpu_support: true,
        // Note: knowledgator/gliner-bi-*-v2.0 bi-encoder models need ONNX export
        // (not yet available as pre-converted ONNX).
        description: "GLiNER via manual ONNX implementation",
        recommended_models: &[crate::models::GLINER],
    },
    BackendInfo {
        name: "bert_onnx",
        feature: Some("onnx"),
        status: BackendStatus::Beta,
        zero_shot: false,
        gpu_support: true,
        description: "BERT NER via ONNX Runtime (PER/ORG/LOC/MISC)",
        recommended_models: &[crate::models::BERT_ONNX],
    },
    BackendInfo {
        name: "gliner_multitask",
        feature: Some("onnx"),
        status: BackendStatus::Beta,
        zero_shot: true,
        gpu_support: true,
        description: "GLiNER multi-task (NER + heuristic relations + structure)",
        recommended_models: &[crate::models::GLINER_MULTITASK],
    },
    BackendInfo {
        name: "w2ner",
        feature: Some("onnx"),
        status: BackendStatus::Beta,
        zero_shot: false,
        gpu_support: true,
        description: "W2NER nested entity extraction (grid-based)",
        recommended_models: &[crate::models::W2NER],
    },
    BackendInfo {
        name: "deberta_v3",
        feature: Some("onnx"),
        status: BackendStatus::Beta,
        zero_shot: false,
        gpu_support: true,
        description: "DeBERTa-v3 NER via BertNEROnnx (export: uv run scripts/export_deberta_ner_to_onnx.py)",
        recommended_models: &[crate::models::DEBERTA_V3],
    },
    BackendInfo {
        name: "biomedical",
        feature: Some("onnx"),
        status: BackendStatus::Beta,
        zero_shot: false,
        gpu_support: true,
        description: "Biomedical NER via BertNEROnnx (Disease, Chemical, Drug, Gene, Species)",
        recommended_models: &[crate::models::BIOMEDICAL],
    },
    // =========================================================================
    // Implemented Backends (Beta)
    // =========================================================================
    BackendInfo {
        name: "gliner_candle",
        feature: Some("candle"),
        status: BackendStatus::Beta,
        zero_shot: true,
        gpu_support: true,
        description: "GLiNER via Candle (pure Rust, Metal/CUDA)",
        recommended_models: &[
            crate::models::GLINER_CANDLE,
            crate::models::GLINER_BI_BASE,
            crate::models::GLINER_BI_LARGE,
        ],
    },
    BackendInfo {
        name: "nuner",
        feature: Some("onnx"),
        status: BackendStatus::Stable,
        zero_shot: true,
        gpu_support: true,
        description: "NuNER Zero (token classifier, arbitrary-length entities)",
        // First entry is the source PyTorch repo; the runtime loader uses
        // crate::models::NUNER (deepanwa/NuNerZero_onnx), a community ONNX
        // export of the same weights.
        recommended_models: &[
            crate::models::NUNER_ZERO,
            crate::models::NUNER_ZERO_4K,
            crate::models::NUNER_ZERO_SPAN,
        ],
    },
    BackendInfo {
        name: "candle_ner",
        feature: Some("candle"),
        status: BackendStatus::Beta,
        zero_shot: false,
        gpu_support: true,
        description: "BERT NER via Candle (pure Rust; Metal/CUDA)",
        recommended_models: &[crate::models::CANDLE],
    },
    BackendInfo {
        name: "glirel",
        feature: Some("onnx"),
        status: BackendStatus::Beta,
        zero_shot: true,
        gpu_support: true,
        description: "GLiREL zero-shot relation extraction (DeBERTa encoder + scoring head)",
        recommended_models: &["jackboyla/glirel-large-v0"],
    },
    BackendInfo {
        name: "gliner_poly",
        feature: Some("onnx"),
        status: BackendStatus::Beta,
        zero_shot: true,
        gpu_support: true,
        description: "GLiNER Poly-encoder for zero-shot NER with inter-label attention fusion",
        // The `knowledgator/gliner-poly-*-v1.0` model cards on HuggingFace exist
        // but ship no weights (per the export script's docstring). The actual
        // loadable models for this backend are the `gliner-bi-*-v1.0` /
        // `modern-gliner-bi-*-v1.0` family.
        recommended_models: &[crate::models::GLINER_POLY],
    },
    BackendInfo {
        name: "gliner_pii",
        feature: Some("onnx"),
        status: BackendStatus::Beta,
        zero_shot: true,
        gpu_support: true,
        description: "GLiNER PII Edge: 60+ PII categories, zero-shot detection",
        recommended_models: &[crate::models::GLINER_PII],
    },
    BackendInfo {
        name: "gliner_relex",
        feature: Some("onnx"),
        status: BackendStatus::Beta,
        zero_shot: true,
        gpu_support: true,
        description: "GLiNER-RelEx: joint NER + relation extraction, zero-shot",
        recommended_models: &[crate::models::GLINER_RELEX],
    },
];

impl BackendInfo {
    /// Get backend by name.
    #[must_use]
    pub fn by_name(name: &str) -> Option<&'static BackendInfo> {
        BACKEND_CATALOG.iter().find(|b| b.name == name)
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
    fn all_entries_are_implemented() {
        // After removing Planned/Research, every catalog entry should be
        // Stable, Beta, or WIP.
        for info in BACKEND_CATALOG {
            assert!(
                matches!(
                    info.status,
                    BackendStatus::Stable | BackendStatus::Beta | BackendStatus::WIP
                ),
                "{} has unexpected status {:?}",
                info.name,
                info.status
            );
        }
    }

    #[test]
    fn test_backend_status_display() {
        assert_eq!(BackendStatus::Stable.to_string(), "stable");
        assert_eq!(BackendStatus::Beta.to_string(), "beta");
        assert_eq!(BackendStatus::WIP.to_string(), "wip");
    }

    #[test]
    fn test_catalog_no_duplicate_names() {
        let mut names: Vec<&str> = BACKEND_CATALOG.iter().map(|b| b.name).collect();
        let original_len = names.len();
        names.sort();
        names.dedup();
        assert_eq!(
            names.len(),
            original_len,
            "catalog has duplicate backend names"
        );
    }

    #[test]
    fn test_catalog_all_have_descriptions() {
        for info in BACKEND_CATALOG {
            assert!(
                !info.description.is_empty(),
                "{} has empty description",
                info.name
            );
        }
    }

    #[test]
    fn test_catalog_feature_gated_backends() {
        // ML backends should require a feature
        let ml_names = [
            "bert_onnx",
            "gliner",
            "nuner",
            "gliner_multitask",
            "w2ner",
            "candle",
        ];
        for name in ml_names {
            if let Some(info) = BackendInfo::by_name(name) {
                assert!(info.feature.is_some(), "{} should be feature-gated", name);
            }
        }
    }

    #[test]
    fn test_catalog_always_available_backends() {
        // Statistical/heuristic backends should not require features
        let always_names = ["pattern", "heuristic", "crf", "hmm"];
        for name in always_names {
            if let Some(info) = BackendInfo::by_name(name) {
                assert!(
                    info.feature.is_none(),
                    "{} should be always available (no feature gate)",
                    name
                );
            }
        }
    }

    #[test]
    fn test_catalog_recommended_models_nonempty_for_non_wip() {
        // For Beta/Stable backends, recommended_models[0] (when present) must be a
        // non-empty model id. Stops empty/whitespace-only string regressions.
        for info in BACKEND_CATALOG {
            if matches!(info.status, BackendStatus::WIP) {
                continue;
            }
            if let Some(first) = info.recommended_models.first() {
                assert!(
                    !first.trim().is_empty(),
                    "{}: recommended_models[0] is empty/whitespace",
                    info.name
                );
            }
        }
    }

    #[test]
    fn test_catalog_aligned_with_lib_constants() {
        // For backends whose default model is encoded as a `crate::models::*` constant,
        // the catalog's first recommended model must match. Replacing the literal in
        // the catalog with the const ref guarantees this at compile time. This test
        // documents (and detects regressions in) the small set of backends where the
        // catalog still uses literals because the constant points at a different repo.
        let pairs: &[(&str, &str)] = &[
            ("gliner", crate::models::GLINER),
            ("gliner_onnx", crate::models::GLINER),
            ("bert_onnx", crate::models::BERT_ONNX),
            ("gliner_multitask", crate::models::GLINER_MULTITASK),
            ("w2ner", crate::models::W2NER),
            ("deberta_v3", crate::models::DEBERTA_V3),
            ("biomedical", crate::models::BIOMEDICAL),
            ("gliner_candle", crate::models::GLINER_CANDLE),
            ("nuner", crate::models::NUNER_ZERO),
            ("candle_ner", crate::models::CANDLE),
            ("gliner_poly", crate::models::GLINER_POLY),
            ("gliner_pii", crate::models::GLINER_PII),
            ("gliner_relex", crate::models::GLINER_RELEX),
        ];
        for (name, expected) in pairs {
            let info = BackendInfo::by_name(name)
                .unwrap_or_else(|| panic!("backend '{}' missing from catalog", name));
            let first = info
                .recommended_models
                .first()
                .unwrap_or_else(|| panic!("'{}': empty recommended_models", name));
            assert_eq!(
                *first, *expected,
                "'{}': catalog recommended_models[0] does not match crate::models constant",
                name
            );
        }
    }

    #[test]
    fn test_catalog_zero_shot_backends() {
        // Known zero-shot backends
        let zs_names = [
            "gliner",
            "nuner",
            "gliner_multitask",
            "gliner_poly",
            "gliner_pii",
        ];
        for name in zs_names {
            if let Some(info) = BackendInfo::by_name(name) {
                assert!(info.zero_shot, "{} should be zero-shot", name);
            }
        }
    }

    #[test]
    fn test_catalog_recommended_models_not_empty_for_ml() {
        for info in BACKEND_CATALOG {
            if info.feature.is_some() && info.status != BackendStatus::WIP {
                assert!(
                    !info.recommended_models.is_empty(),
                    "{} (status={}) should have recommended models",
                    info.name,
                    info.status
                );
            }
        }
    }

    #[test]
    fn test_by_name_returns_correct_entry() {
        let gliner = BackendInfo::by_name("gliner").unwrap();
        assert!(gliner.zero_shot);
        assert_eq!(gliner.feature, Some("onnx"));
    }
}
