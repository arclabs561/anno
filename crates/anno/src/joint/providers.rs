//! Score provider implementations for pluggable unary factors.
//!
//! These implementations allow the joint model to use existing anno
//! backends for NER, coreference, and entity linking.

use std::sync::Arc;

use crate::backends::box_embeddings::BoxEmbedding;
use crate::linking::candidate::CandidateGenerator;
use crate::linking::linker::{EntityLinker, Mention};
use anno_core::EntityType;

use super::types::{
    AntecedentValue, CorefScoreProvider, JointMention, LinkScoreProvider, NerScoreProvider,
};

// =============================================================================
// EntityLinker-based Link Score Provider
// =============================================================================

/// A `LinkScoreProvider` that uses the existing `EntityLinker` infrastructure.
///
/// This adapter bridges the joint model with anno's entity linking system.
///
/// # Example
///
/// ```rust,ignore
/// use anno::joint::{JointModelBuilder, EntityLinkerProvider};
/// use anno::linking::EntityLinker;
/// use std::sync::Arc;
///
/// let linker = EntityLinker::builder()
///     .with_max_candidates(20)
///     .build()?;
///
/// let provider = EntityLinkerProvider::new(Arc::new(linker));
/// ```
pub struct EntityLinkerProvider {
    linker: Arc<EntityLinker>,
    max_candidates: usize,
}

impl EntityLinkerProvider {
    /// Create a new provider wrapping an entity linker.
    pub fn new(linker: Arc<EntityLinker>) -> Self {
        Self {
            linker,
            max_candidates: 20,
        }
    }

    /// Set maximum candidates to return.
    pub fn with_max_candidates(mut self, max: usize) -> Self {
        self.max_candidates = max;
        self
    }
}

impl LinkScoreProvider for EntityLinkerProvider {
    fn link_candidates(&self, mention: &JointMention, text: &str) -> Vec<(String, f64)> {
        // Convert JointMention to linking::Mention
        let linking_mention = Mention::new(&mention.text, mention.start, mention.end);
        let linking_mention = if let Some(ref entity) = mention.entity {
            linking_mention.with_type(entity.entity_type.clone())
        } else {
            linking_mention
        };

        // Use EntityLinker to get candidates
        let result = self.linker.link(&[linking_mention], text);

        if result.entities.is_empty() {
            return vec![("NIL".to_string(), 0.0)];
        }

        let linked = &result.entities[0];

        // Collect candidates with scores
        let mut candidates: Vec<(String, f64)> = linked
            .alternatives
            .iter()
            .take(self.max_candidates - 1)
            .map(|alt| (alt.kb_id.clone(), alt.score.ln().max(-100.0)))
            .collect();

        // Add the top candidate
        if let Some(ref kb_id) = linked.kb_id {
            candidates.insert(0, (kb_id.clone(), linked.confidence.ln().max(-100.0)));
        }

        // Always include NIL option
        candidates.push(("NIL".to_string(), (-2.0_f64).ln())); // ~0.13 prior for NIL

        candidates
    }
}

// =============================================================================
// Box embedding based Coref Score Provider
// =============================================================================

/// A lightweight `CorefScoreProvider` that scores antecedents using box
/// embeddings' mutual overlap (`coreference_score`). Boxes are derived
/// deterministically from mention text to avoid needing a trained encoder.
///
/// This is a stopgap adapter to let the joint model consume box-based
/// coreference cues without wiring a full box-training pipeline.
#[allow(dead_code)] // Future: wire up box-based coref in joint model
pub struct BoxCorefProvider {
    /// Half-width of the constructed boxes in each dimension.
    pub radius: f32,
}

impl Default for BoxCorefProvider {
    fn default() -> Self {
        Self { radius: 0.1 }
    }
}

impl BoxCorefProvider {
    /// Convert a mention into a deterministic 2D box embedding.
    ///
    /// The hash is mapped into [0,1]² and expanded by `radius` in each dim.
    #[allow(dead_code)] // struct is currently not wired into main joint path
    fn mention_to_box(&self, mention: &JointMention) -> BoxEmbedding {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        mention.text.hash(&mut hasher);
        mention.start.hash(&mut hasher);
        let h = hasher.finish();
        let v1 = ((h & 0xFFFF) as f32) / 65535.0;
        let v2 = (((h >> 16) & 0xFFFF) as f32) / 65535.0;
        let radius = self.radius.max(1e-3);
        BoxEmbedding::new(
            vec![v1 - radius, v2 - radius],
            vec![v1 + radius, v2 + radius],
        )
    }
}

impl CorefScoreProvider for BoxCorefProvider {
    fn antecedent_scores(
        &self,
        mention: &JointMention,
        candidates: &[&JointMention],
        _text: &str,
    ) -> Vec<(AntecedentValue, f64)> {
        let m_box = self.mention_to_box(mention);

        // Score each candidate via box overlap and convert to log-score
        let mut scores: Vec<(AntecedentValue, f64)> = candidates
            .iter()
            .map(|cand| {
                let c_box = self.mention_to_box(cand);
                let s = m_box.coreference_score(&c_box).max(1e-6);
                (AntecedentValue::Mention(cand.idx), s.ln() as f64)
            })
            .collect();

        // NEW cluster prior (mild)
        scores.push((AntecedentValue::NewCluster, (-1.0_f64).ln()));
        scores
    }
}

// =============================================================================
// Dictionary-based Link Score Provider
// =============================================================================

/// A simpler `LinkScoreProvider` that uses a dictionary for candidate generation.
///
/// This is faster than the full EntityLinker but less accurate.
pub struct DictionaryLinkProvider {
    generator: Arc<dyn CandidateGenerator>,
    max_candidates: usize,
}

impl DictionaryLinkProvider {
    /// Create a new dictionary-based provider.
    pub fn new(generator: Arc<dyn CandidateGenerator>) -> Self {
        Self {
            generator,
            max_candidates: 20,
        }
    }

    /// Set maximum candidates to return.
    pub fn with_max_candidates(mut self, max: usize) -> Self {
        self.max_candidates = max;
        self
    }
}

impl LinkScoreProvider for DictionaryLinkProvider {
    fn link_candidates(&self, mention: &JointMention, text: &str) -> Vec<(String, f64)> {
        let entity_type_str = mention.entity.as_ref().map(|e| e.entity_type.to_string());

        let mut candidates = self.generator.generate(
            &mention.text,
            text,
            entity_type_str.as_deref(),
            self.max_candidates,
        );

        // Compute scores and convert to log-space
        let results: Vec<(String, f64)> = candidates
            .iter_mut()
            .map(|c| {
                c.compute_score();
                (c.kb_id.clone(), c.score.ln().max(-100.0))
            })
            .collect();

        if results.is_empty() {
            vec![("NIL".to_string(), 0.0)]
        } else {
            let mut results = results;
            results.push(("NIL".to_string(), (-2.0_f64).ln()));
            results
        }
    }
}

// =============================================================================
// Model-based NER Score Provider
// =============================================================================

/// A `NerScoreProvider` that uses any `Model` implementation.
///
/// This allows using GLiNER, NuNER, or any other NER backend
/// to provide type scores.
pub struct ModelNerProvider {
    model: Arc<dyn crate::Model>,
    /// Supported entity types
    entity_types: Vec<EntityType>,
}

impl ModelNerProvider {
    /// Create a new provider wrapping an NER model.
    pub fn new(model: Arc<dyn crate::Model>) -> Self {
        let entity_types = model.supported_types();
        Self {
            model,
            entity_types,
        }
    }

    /// Override entity types to consider.
    pub fn with_entity_types(mut self, types: Vec<EntityType>) -> Self {
        self.entity_types = types;
        self
    }
}

impl NerScoreProvider for ModelNerProvider {
    fn type_scores(&self, mention: &JointMention, text: &str) -> Vec<(EntityType, f64)> {
        // If the mention already has an entity with a type, use that as a strong prior
        if let Some(ref entity) = mention.entity {
            let prior_type = entity.entity_type.clone();
            let confidence = entity.confidence;

            return self
                .entity_types
                .iter()
                .map(|et| {
                    if et == &prior_type {
                        (et.clone(), confidence.ln().max(-100.0))
                    } else {
                        (et.clone(), (1.0 - confidence).ln().max(-100.0) - 2.0)
                    }
                })
                .collect();
        }

        // Otherwise, run NER on the mention span
        // Extract a context window around the mention
        let context_start = mention.start.saturating_sub(50);
        let context_end = (mention.end + 50).min(text.chars().count());
        let context: String = text
            .chars()
            .skip(context_start)
            .take(context_end - context_start)
            .collect();

        match self.model.extract_entities(&context, None) {
            Ok(entities) => {
                // Find entity overlapping with the mention
                let mention_in_context_start = mention.start - context_start;
                let mention_in_context_end = mention.end - context_start;

                let matching_entity = entities.iter().find(|e| {
                    e.start <= mention_in_context_end && e.end >= mention_in_context_start
                });

                match matching_entity {
                    Some(e) => self
                        .entity_types
                        .iter()
                        .map(|et| {
                            if et == &e.entity_type {
                                (et.clone(), e.confidence.ln().max(-100.0))
                            } else {
                                (et.clone(), (1.0 - e.confidence).ln().max(-100.0) - 1.0)
                            }
                        })
                        .collect(),
                    None => {
                        // Uniform distribution as fallback
                        let uniform = (-(self.entity_types.len() as f64)).ln();
                        self.entity_types
                            .iter()
                            .map(|et| (et.clone(), uniform))
                            .collect()
                    }
                }
            }
            Err(_) => {
                // Fallback to uniform distribution
                let uniform = (-(self.entity_types.len() as f64)).ln();
                self.entity_types
                    .iter()
                    .map(|et| (et.clone(), uniform))
                    .collect()
            }
        }
    }
}

// =============================================================================
// Heuristic Coref Score Provider
// =============================================================================

/// A simple heuristic `CorefScoreProvider` based on string matching.
///
/// This is a lightweight alternative to neural mention-ranking models.
pub struct HeuristicCorefProvider {
    /// Weight for exact match
    exact_match_weight: f64,
    /// Weight for substring match
    substring_weight: f64,
    /// Weight for same head word
    head_match_weight: f64,
    /// Distance penalty per mention
    distance_penalty: f64,
}

impl Default for HeuristicCorefProvider {
    fn default() -> Self {
        Self {
            exact_match_weight: 5.0,
            substring_weight: 2.0,
            head_match_weight: 3.0,
            distance_penalty: 0.1,
        }
    }
}

impl HeuristicCorefProvider {
    /// Create a new heuristic provider with default weights.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set exact match weight.
    pub fn with_exact_match_weight(mut self, weight: f64) -> Self {
        self.exact_match_weight = weight;
        self
    }

    /// Set distance penalty.
    pub fn with_distance_penalty(mut self, penalty: f64) -> Self {
        self.distance_penalty = penalty;
        self
    }
}

impl CorefScoreProvider for HeuristicCorefProvider {
    fn antecedent_scores(
        &self,
        mention: &JointMention,
        candidates: &[&JointMention],
        _text: &str,
    ) -> Vec<(AntecedentValue, f64)> {
        let mention_text_lower = mention.text.to_lowercase();
        let mention_head_lower = mention.head.to_lowercase();

        let mut scores: Vec<(AntecedentValue, f64)> = candidates
            .iter()
            .enumerate()
            .map(|(i, cand)| {
                let cand_text_lower = cand.text.to_lowercase();
                let cand_head_lower = cand.head.to_lowercase();

                let mut score = 0.0;

                // Exact match bonus
                if mention_text_lower == cand_text_lower {
                    score += self.exact_match_weight;
                }

                // Substring match
                if mention_text_lower.contains(&cand_text_lower)
                    || cand_text_lower.contains(&mention_text_lower)
                {
                    score += self.substring_weight;
                }

                // Head word match
                if mention_head_lower == cand_head_lower {
                    score += self.head_match_weight;
                }

                // Distance penalty
                let distance = candidates.len() - i; // More recent = higher score
                score -= self.distance_penalty * distance as f64;

                (AntecedentValue::Mention(cand.idx), score)
            })
            .collect();

        // Add NEW_CLUSTER option
        // New cluster is preferred for proper nouns that don't match anything
        let new_cluster_score = if mention.mention_kind.is_proper_name() {
            1.0 // Proper nouns more likely to start new clusters
        } else {
            -1.0 // Pronouns/nominals more likely to be anaphoric
        };
        scores.push((AntecedentValue::NewCluster, new_cluster_score));

        scores
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heuristic_coref_provider() {
        let provider = HeuristicCorefProvider::default();

        let mention = JointMention {
            idx: 2,
            text: "he".to_string(),
            head: "he".to_string(),
            start: 20,
            end: 22,
            mention_kind: super::super::MentionKind::Pronominal,
            entity: None,
            entity_type: None,
        };

        let cand1 = JointMention {
            idx: 0,
            text: "John Smith".to_string(),
            head: "Smith".to_string(),
            start: 0,
            end: 10,
            mention_kind: super::super::MentionKind::Proper,
            entity: None,
            entity_type: None,
        };

        let cand2 = JointMention {
            idx: 1,
            text: "the CEO".to_string(),
            head: "CEO".to_string(),
            start: 12,
            end: 19,
            mention_kind: super::super::MentionKind::Nominal,
            entity: None,
            entity_type: None,
        };

        let candidates: Vec<&JointMention> = vec![&cand1, &cand2];
        let scores = provider.antecedent_scores(&mention, &candidates, "");

        // Should have 3 options: 2 candidates + NEW_CLUSTER
        assert_eq!(scores.len(), 3);

        // NEW_CLUSTER should have negative score for pronouns
        let new_cluster_score = scores
            .iter()
            .find(|(v, _)| matches!(v, AntecedentValue::NewCluster))
            .map(|(_, s)| *s)
            .unwrap();
        assert!(new_cluster_score < 0.0);
    }

    #[test]
    fn test_heuristic_coref_exact_match() {
        let provider = HeuristicCorefProvider::default();

        let mention = JointMention {
            idx: 1,
            text: "John Smith".to_string(),
            head: "Smith".to_string(),
            start: 50,
            end: 60,
            mention_kind: super::super::MentionKind::Proper,
            entity: None,
            entity_type: None,
        };

        let cand = JointMention {
            idx: 0,
            text: "John Smith".to_string(),
            head: "Smith".to_string(),
            start: 0,
            end: 10,
            mention_kind: super::super::MentionKind::Proper,
            entity: None,
            entity_type: None,
        };

        let scores = provider.antecedent_scores(&mention, &[&cand], "");

        // Exact match should have high score
        let mention_score = scores
            .iter()
            .find(|(v, _)| matches!(v, AntecedentValue::Mention(0)))
            .map(|(_, s)| *s)
            .unwrap();

        // Should include exact match + head match bonuses
        assert!(mention_score > 7.0); // 5.0 + 3.0 - small distance penalty
    }
}
