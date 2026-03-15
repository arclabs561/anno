//! Unified trait for coreference resolution backends.
//!
//! [`CorefBackend`] provides a common interface for all within-document coreference
//! resolvers. Unlike the sealed [`Model`](crate::Model) trait (NER), this trait is
//! intentionally **open**: external crates can implement it for custom coref backends
//! (REST APIs, Python models via PyO3, domain-specific heuristics, etc.).
//!
//! # Implementors
//!
//! | Backend | Feature gate | Notes |
//! |---------|-------------|-------|
//! | [`FCoref`](super::fcoref::FCoref) | `onnx` | Neural, exact signature match |
//! | [`T5Coref`](super::t5::T5Coref) | `onnx` | Seq2seq, exact signature match |
//! | [`MentionRankingCoref`](super::mention_ranking::MentionRankingCoref) | - | Heuristic, adapter converts MentionCluster |
//! | `SimpleCorefResolver` | `analysis` | Takes `&[Entity]` not `&str`; incompatible (no impl) |

use crate::Result;

/// A coreference cluster: a group of mentions referring to the same entity.
///
/// This is the common return type for [`CorefBackend::resolve`]. It mirrors the
/// struct in `coref_t5` but lives here so the trait is available without the `onnx`
/// feature gate.
#[derive(Debug, Clone)]
pub struct CorefCluster {
    /// Cluster ID.
    pub id: u32,
    /// Member mention texts.
    pub mentions: Vec<String>,
    /// Member mention spans as `(char_start, char_end)`.
    pub spans: Vec<(usize, usize)>,
    /// Canonical name (longest / most informative mention).
    pub canonical: String,
}

/// Unified interface for within-document coreference resolution.
///
/// Accepts raw text and returns clusters of co-referring mentions.
///
/// # Open trait
///
/// This trait is deliberately **not sealed**. External coref backends are welcome --
/// implement `CorefBackend` and use `Box<dyn CorefBackend>` for dynamic dispatch.
///
/// # Example
///
/// ```rust,ignore
/// use anno::backends::coref::resolve::CorefBackend;
///
/// fn run_coref(backend: &dyn CorefBackend, text: &str) {
///     if !backend.is_available() {
///         eprintln!("{} is not available", backend.name());
///         return;
///     }
///     let clusters = backend.resolve(text).unwrap();
///     for c in &clusters {
///         println!("[{}] {} mentions, canonical: {}", c.id, c.mentions.len(), c.canonical);
///     }
/// }
/// ```
pub trait CorefBackend: Send + Sync {
    /// Resolve coreferences in `text`, returning clusters of co-referring mentions.
    ///
    /// Each [`CorefCluster`] contains the mention texts, their character-offset spans,
    /// and a canonical (most informative) mention string.
    fn resolve(&self, text: &str) -> Result<Vec<CorefCluster>>;

    /// Human-readable backend name (e.g., `"fcoref"`, `"coref-t5"`).
    fn name(&self) -> &'static str;

    /// Whether the backend is ready for inference.
    ///
    /// Returns `false` when required model artifacts are missing or failed to load.
    fn is_available(&self) -> bool;
}

// =============================================================================
// Implementations for existing backends
// =============================================================================

// FCoref: returns Vec<CorefCluster> directly (same type via re-export).
#[cfg(feature = "onnx")]
impl CorefBackend for super::fcoref::FCoref {
    fn resolve(&self, text: &str) -> Result<Vec<CorefCluster>> {
        self.resolve(text)
    }

    fn name(&self) -> &'static str {
        "fcoref"
    }

    fn is_available(&self) -> bool {
        true // FCoref is available once constructed (loading validates artifacts).
    }
}

// T5Coref: returns Vec<CorefCluster> directly (same type via re-export).
#[cfg(feature = "onnx")]
impl CorefBackend for super::t5::T5Coref {
    fn resolve(&self, text: &str) -> Result<Vec<CorefCluster>> {
        self.resolve(text)
    }

    fn name(&self) -> &'static str {
        "coref-t5"
    }

    fn is_available(&self) -> bool {
        true // T5Coref is available once constructed.
    }
}

// MentionRankingCoref: returns Vec<MentionCluster>, not Vec<CorefCluster>.
// We adapt by converting MentionCluster -> CorefCluster.
impl CorefBackend for super::mention_ranking::MentionRankingCoref {
    fn resolve(&self, text: &str) -> Result<Vec<CorefCluster>> {
        let clusters = self.resolve(text)?;
        Ok(clusters
            .into_iter()
            .enumerate()
            .map(|(i, mc)| {
                let mentions: Vec<String> = mc.mentions.iter().map(|m| m.text.clone()).collect();
                let spans: Vec<(usize, usize)> =
                    mc.mentions.iter().map(|m| (m.start, m.end)).collect();
                // Canonical: longest mention text (same heuristic as other backends).
                let canonical = mentions
                    .iter()
                    .max_by_key(|t| t.len())
                    .cloned()
                    .unwrap_or_default();
                CorefCluster {
                    id: i as u32,
                    mentions,
                    spans,
                    canonical,
                }
            })
            .collect())
    }

    fn name(&self) -> &'static str {
        "mention-ranking"
    }

    fn is_available(&self) -> bool {
        true // Heuristic backend, always available once constructed.
    }
}

// SimpleCorefResolver (eval/analysis): NOT implemented.
//
// Its signature is `resolve(&self, entities: &[Entity]) -> Vec<Entity>` -- it operates
// on pre-extracted entities, not raw text. Adapting it would require coupling an NER
// model into the trait call, which defeats the purpose of a simple unified interface.
// Callers needing SimpleCorefResolver should use it directly via its inherent API.

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal test backend that returns a fixed set of clusters.
    struct StubCoref {
        available: bool,
    }

    impl CorefBackend for StubCoref {
        fn resolve(&self, text: &str) -> Result<Vec<CorefCluster>> {
            if text.is_empty() {
                return Ok(vec![]);
            }
            Ok(vec![CorefCluster {
                id: 0,
                mentions: vec!["John".to_string(), "He".to_string()],
                spans: vec![(0, 4), (30, 32)],
                canonical: "John".to_string(),
            }])
        }

        fn name(&self) -> &'static str {
            "stub-coref"
        }

        fn is_available(&self) -> bool {
            self.available
        }
    }

    #[test]
    fn trait_object_dispatch() {
        let backend: Box<dyn CorefBackend> = Box::new(StubCoref { available: true });
        let clusters = backend
            .resolve("John went to the store. He bought milk.")
            .unwrap();
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].canonical, "John");
        assert_eq!(clusters[0].mentions.len(), 2);
    }

    #[test]
    fn trait_object_empty_input() {
        let backend: Box<dyn CorefBackend> = Box::new(StubCoref { available: true });
        let clusters = backend.resolve("").unwrap();
        assert!(
            clusters.is_empty(),
            "empty input should produce no clusters"
        );
    }

    #[test]
    fn trait_object_name_and_availability() {
        let available: Box<dyn CorefBackend> = Box::new(StubCoref { available: true });
        let unavailable: Box<dyn CorefBackend> = Box::new(StubCoref { available: false });

        assert_eq!(available.name(), "stub-coref");
        assert!(available.is_available());
        assert!(!unavailable.is_available());
    }

    #[test]
    fn heterogeneous_vec_of_trait_objects() {
        // Verify that multiple different backends can coexist in a Vec<Box<dyn CorefBackend>>.
        let backends: Vec<Box<dyn CorefBackend>> = vec![
            Box::new(StubCoref { available: true }),
            Box::new(StubCoref { available: false }),
        ];

        let available_count = backends.iter().filter(|b| b.is_available()).count();
        assert_eq!(available_count, 1);

        // Only invoke resolve on available backends.
        for b in &backends {
            if b.is_available() {
                let clusters = b.resolve("test text").unwrap();
                assert!(!clusters.is_empty());
            }
        }
    }
}
