//! Parser for fastino GLiNER2 `config.json`. Captures the `counting_layer`
//! enum that selects the encoder counting architecture; Phase 1 doesn't use
//! it (it's a Phase 2 head), but we read it at load time so Phase 2 can
//! dispatch without re-parsing.
#![allow(missing_docs)] // implementation internals; public API is on GLiNER2Fastino in mod.rs

use serde::Deserialize;

/// `counting_layer` field in fastino `config.json` — selects the encoder
/// counting architecture variant. Parameters are not interchangeable across
/// these.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CountingLayer {
    /// `fastino/gliner2-base-v1`
    CountLstm,
    /// `fastino/gliner2-large-v1`
    CountLstmMoe,
    /// `fastino/gliner2-multi-v1`
    CountLstmV2,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)] // counting_layer + max_seq_length reserved for Phase 2 dispatch
pub struct FastinoConfig {
    /// Hidden size of the encoder (e.g. 768 base, 1024 large).
    pub hidden_size: usize,
    /// Counting head architecture (Phase 2; ignored in Phase 1 loading).
    #[serde(default)]
    pub counting_layer: Option<CountingLayer>,
    /// Maximum sequence length supported by the encoder.
    #[serde(default = "default_max_len")]
    pub max_seq_length: usize,
}

fn default_max_len() -> usize {
    512
}

impl FastinoConfig {
    pub fn from_path(path: &std::path::Path) -> Result<Self, super::errors::Error> {
        let s = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&s)?)
    }
}

impl Default for FastinoConfig {
    /// Defaults appropriate for `fastino/gliner2-multi-v1`: 768-dim DeBERTa-v2
    /// encoder, 512-token max, count_lstm_v2 head. Used when config.json is
    /// not shipped (e.g., SemplificaAI's pre-export).
    fn default() -> Self {
        Self {
            hidden_size: 768,
            counting_layer: Some(CountingLayer::CountLstmV2),
            max_seq_length: 512,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_minimal_config() {
        let json = r#"{"hidden_size": 768, "counting_layer": "count_lstm_v2"}"#;
        let cfg: FastinoConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.hidden_size, 768);
        assert_eq!(cfg.counting_layer, Some(CountingLayer::CountLstmV2));
        assert_eq!(cfg.max_seq_length, 512);
    }

    #[test]
    fn parses_all_three_counting_variants() {
        for (s, expected) in [
            ("count_lstm", CountingLayer::CountLstm),
            ("count_lstm_moe", CountingLayer::CountLstmMoe),
            ("count_lstm_v2", CountingLayer::CountLstmV2),
        ] {
            let json = format!(r#"{{"hidden_size": 768, "counting_layer": "{s}"}}"#);
            let cfg: FastinoConfig = serde_json::from_str(&json).unwrap();
            assert_eq!(cfg.counting_layer, Some(expected));
        }
    }

    #[test]
    fn missing_counting_layer_is_optional_for_phase1() {
        let json = r#"{"hidden_size": 768}"#;
        let cfg: FastinoConfig = serde_json::from_str(json).unwrap();
        assert!(cfg.counting_layer.is_none());
    }
}
