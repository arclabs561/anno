//! Backend Name Enum
//!
//! Type-safe backend identifiers replacing string-based names.
//!
//! # Example
//!
//! ```rust
//! use anno::eval::backend_name::BackendName;
//!
//! let backend = BackendName::Stacked;
//! let name: &str = backend.as_str();
//! ```

use serde::{Deserialize, Serialize};
use std::fmt;

/// Type-safe backend identifier.
///
/// Replaces string-based backend names with compile-time checked enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum BackendName {
    // Always available
    Pattern,
    Heuristic,
    Stacked,

    // ONNX backends
    #[cfg(feature = "onnx")]
    BertOnnx,
    #[cfg(feature = "onnx")]
    GLiNEROnnx,
    #[cfg(feature = "onnx")]
    NuNER,
    #[cfg(feature = "onnx")]
    W2NER,
    #[cfg(feature = "onnx")]
    GLiNER2,
    #[cfg(feature = "onnx")]
    GLiNERPoly,
    #[cfg(feature = "onnx")]
    DeBERTaV3,
    #[cfg(feature = "onnx")]
    ALBERT,
    #[cfg(feature = "onnx")]
    TPLinker,

    // Candle backends
    #[cfg(feature = "candle")]
    CandleNER,
    #[cfg(feature = "candle")]
    GLiNERCandle,
    #[cfg(feature = "candle")]
    GLiNER2Candle,

    // Coreference
    CorefResolver,

    // Universal
    UniversalNER,
}

impl BackendName {
    /// Get string representation of backend name.
    ///
    /// Returns the canonical string identifier for this backend.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            BackendName::Pattern => "pattern",
            BackendName::Heuristic => "heuristic",
            BackendName::Stacked => "stacked",
            #[cfg(feature = "onnx")]
            BackendName::BertOnnx => "bert_onnx",
            #[cfg(feature = "onnx")]
            BackendName::GLiNEROnnx => "gliner_onnx",
            #[cfg(feature = "onnx")]
            BackendName::NuNER => "nuner",
            #[cfg(feature = "onnx")]
            BackendName::W2NER => "w2ner",
            #[cfg(feature = "onnx")]
            BackendName::GLiNER2 => "gliner2",
            #[cfg(feature = "onnx")]
            BackendName::GLiNERPoly => "gliner_poly",
            #[cfg(feature = "onnx")]
            BackendName::DeBERTaV3 => "deberta_v3",
            #[cfg(feature = "onnx")]
            BackendName::ALBERT => "albert",
            #[cfg(feature = "onnx")]
            BackendName::TPLinker => "tplinker",
            #[cfg(feature = "candle")]
            BackendName::CandleNER => "candle_ner",
            #[cfg(feature = "candle")]
            BackendName::GLiNERCandle => "gliner_candle",
            #[cfg(feature = "candle")]
            BackendName::GLiNER2Candle => "gliner2_candle",
            BackendName::CorefResolver => "coref_resolver",
            BackendName::UniversalNER => "universal_ner",
        }
    }

    /// Parse backend name from string.
    ///
    /// Returns `None` if the string doesn't match any known backend.
    #[must_use]
    pub fn try_parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "pattern" | "patternner" | "regex" | "regexner" => Some(BackendName::Pattern),
            "heuristic" | "heuristicner" => Some(BackendName::Heuristic),
            "stacked" | "stackedner" => Some(BackendName::Stacked),
            #[cfg(feature = "onnx")]
            "bert_onnx" | "bertneronnx" => Some(BackendName::BertOnnx),
            #[cfg(feature = "onnx")]
            "gliner_onnx" | "glineronnx" => Some(BackendName::GLiNEROnnx),
            #[cfg(feature = "onnx")]
            "nuner" | "nunerzero" => Some(BackendName::NuNER),
            #[cfg(feature = "onnx")]
            "w2ner" | "w2ner" => Some(BackendName::W2NER),
            #[cfg(feature = "onnx")]
            "gliner2" | "gliner2onnx" => Some(BackendName::GLiNER2),
            #[cfg(feature = "onnx")]
            "gliner_poly" | "glinerpoly" => Some(BackendName::GLiNERPoly),
            #[cfg(feature = "onnx")]
            "deberta_v3" | "debertav3" => Some(BackendName::DeBERTaV3),
            #[cfg(feature = "onnx")]
            "albert" | "albert_ner" => Some(BackendName::ALBERT),
            #[cfg(feature = "onnx")]
            "tplinker" => Some(BackendName::TPLinker),
            #[cfg(feature = "candle")]
            "candle_ner" | "candlener" => Some(BackendName::CandleNER),
            #[cfg(feature = "candle")]
            "gliner_candle" | "glinercandle" => Some(BackendName::GLiNERCandle),
            #[cfg(feature = "candle")]
            "gliner2_candle" | "gliner2candle" => Some(BackendName::GLiNER2Candle),
            "coref_resolver" | "corefresolver" | "simplecorefresolver" => {
                Some(BackendName::CorefResolver)
            }
            "universal_ner" | "universal-ner" | "universalner" => Some(BackendName::UniversalNER),
            _ => None,
        }
    }

    /// Get all available backends (based on enabled features).
    ///
    /// Returns a vector of all `BackendName` variants that are available
    /// given the current feature flags.
    #[must_use]
    pub fn all_available() -> Vec<Self> {
        let mut backends = vec![
            BackendName::Pattern,
            BackendName::Heuristic,
            BackendName::Stacked,
            BackendName::CorefResolver,
            BackendName::UniversalNER,
        ];

        #[cfg(feature = "onnx")]
        {
            backends.extend(&[
                BackendName::BertOnnx,
                BackendName::GLiNEROnnx,
                BackendName::NuNER,
                BackendName::W2NER,
                BackendName::GLiNER2,
                BackendName::GLiNERPoly,
                BackendName::DeBERTaV3,
                BackendName::ALBERT,
                BackendName::TPLinker,
            ]);
        }

        #[cfg(feature = "candle")]
        {
            backends.extend(&[BackendName::CandleNER, BackendName::GLiNERCandle]);
        }

        #[cfg(all(feature = "candle", feature = "onnx"))]
        {
            backends.push(BackendName::GLiNER2Candle);
        }

        backends
    }
}

impl fmt::Display for BackendName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl std::str::FromStr for BackendName {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::try_parse(s).ok_or_else(|| format!("Unknown backend: '{}'", s))
    }
}

impl From<BackendName> for String {
    fn from(name: BackendName) -> Self {
        name.as_str().to_string()
    }
}
