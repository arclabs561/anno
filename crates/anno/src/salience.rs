//! Entity salience and importance ranking.
//!
//! This module provides algorithms for ranking **already-extracted entities**
//! by their importance or salience in a document.
//!
//! # Architecture: Graph Centrality on Knowledge Graphs
//!
//! Entity salience is fundamentally **graph centrality** (PageRank) applied to
//! entity graphs. The key question is: what edges do we use?
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │  GRAPH CONSTRUCTION OPTIONS                                              │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                          │
//! │  STRONG: Actual relations (requires relation extraction)                 │
//! │  ────────────────────────────────────────────────────────                │
//! │  "Obama" ─[PRESIDENT_OF]→ "USA"                                         │
//! │  "Obama" ─[BORN_IN]→ "Hawaii"                                           │
//! │  → Use GraphDocument::from_extraction(entities, relations)              │
//! │  → PageRank reveals structurally central entities                       │
//! │                                                                          │
//! │  WEAK: Co-occurrence proximity (fallback when no relations)             │
//! │  ───────────────────────────────────────────────────────                │
//! │  "Obama" ─[NEAR]→ "USA"  (appeared within 50 chars)                     │
//! │  → Use GraphDocument::from_entities_cooccurrence(entities, window)      │
//! │  → PageRank on proximity is a noisy signal                              │
//! │                                                                          │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! **If you have relation extraction, use `GraphRankSalience` with a
//! `GraphDocument`. If you only have entities, `TextRankSalience` falls
//! back to co-occurrence.**
//!
//! # Relationship to community detection
//!
//! Both this module and any community-detection layer operate on the same `GraphDocument`:
//!
//! | Module | Algorithm | Output | Question Answered |
//! |--------|-----------|--------|-------------------|
//! | `salience` | PageRank | Node scores | "Which entities are important?" |
//! | community detection | Leiden/Louvain | Communities | "How do entities cluster?" |
//!
//! # Multilingual Support
//!
//! **These algorithms work across languages** because they operate on entities
//! extracted by the ML backends, not on raw text.
//!
//! # Algorithms
//!
//! | Algorithm | Graph Source | Description |
//! |-----------|--------------|-------------|
//! | `GraphRankSalience` | `GraphDocument` | PageRank on actual relations (PREFERRED) |
//! | `TextRankSalience` | Entities only | PageRank on co-occurrence (fallback) |
//! | `PositionSalience` | Entities only | Heuristic: early position + frequency |
//!
//! # Example
//!
//! ```rust,ignore
//! use anno::salience::{EntityRanker, TextRankSalience};
//! use anno::Entity;
//!
//! let text = "Barack Obama met Angela Merkel in Berlin. Obama discussed policy.";
//! let entities = vec![
//!     Entity::new("Barack Obama", EntityType::Person, 0, 12, 0.9),
//!     Entity::new("Angela Merkel", EntityType::Person, 17, 30, 0.9),
//!     Entity::new("Berlin", EntityType::Location, 34, 40, 0.9),
//!     Entity::new("Obama", EntityType::Person, 42, 47, 0.9),
//! ];
//!
//! let ranker = TextRankSalience::default();
//! let ranked = ranker.rank(text, &entities);
//!
//! // Obama mentioned twice, likely most salient
//! assert_eq!(ranked[0].0.text, "Barack Obama");
//! ```
//!
//! # Relationship to Other Modules
//!
//! ```text
//! anno (extract)      → entities with spans
//!       ↓
//! anno::salience      → entities ranked by importance  ← THIS MODULE
//!       ↓
//! anno::coalesce      → entity resolution (same entity?)
//!       ↓
//! community detection → community structure
//! ```
//!
//! # References
//!
//! - Mihalcea & Tarau (2004): TextRank: Bringing Order into Text
//! - Campos et al. (2018): YAKE! Collection-independent keyword extraction

use crate::features::{ChainFeatures, EntityFeatureExtractor, ExtractorConfig, MentionType};
use crate::pagerank::{pagerank, PageRankConfig};
use crate::Entity;
use std::collections::HashMap;

/// Trait for entity ranking algorithms.
pub trait EntityRanker: Send + Sync {
    /// Rank entities by salience/importance.
    ///
    /// Returns entities paired with their salience scores, sorted descending.
    fn rank(&self, text: &str, entities: &[Entity]) -> Vec<(Entity, f64)>;

    /// Get top-k most salient entities.
    fn top_k(&self, text: &str, entities: &[Entity], k: usize) -> Vec<(Entity, f64)> {
        let mut ranked = self.rank(text, entities);
        ranked.truncate(k);
        ranked
    }

    /// Filter to entities above a salience threshold.
    fn filter_by_threshold(
        &self,
        text: &str,
        entities: &[Entity],
        threshold: f64,
    ) -> Vec<(Entity, f64)> {
        self.rank(text, entities)
            .into_iter()
            .filter(|(_, score)| *score >= threshold)
            .collect()
    }
}

// =============================================================================
// TextRank Implementation
// =============================================================================

/// TextRank-based entity salience using co-occurrence graph.
///
/// Builds a graph where entities are nodes and edges connect entities
/// that co-occur within a window. Runs PageRank to find central entities.
///
/// # Algorithm
///
/// 1. Build co-occurrence graph (entities within window are connected)
/// 2. Run PageRank iterations until convergence
/// 3. Return entities sorted by PageRank score
///
/// # Parameters
///
/// - `window_size`: Co-occurrence window (default: 50 characters)
/// - `damping`: PageRank damping factor (default: 0.85)
/// - `iterations`: Max PageRank iterations (default: 30)
#[derive(Debug, Clone)]
pub struct TextRankSalience {
    /// Co-occurrence window size (characters)
    pub window_size: usize,
    /// PageRank damping factor (0-1)
    pub damping: f64,
    /// Maximum iterations for PageRank
    pub iterations: usize,
    /// Convergence threshold
    pub epsilon: f64,
}

impl Default for TextRankSalience {
    fn default() -> Self {
        Self {
            window_size: 50,
            damping: 0.85,
            iterations: 30,
            epsilon: 1e-6,
        }
    }
}

impl TextRankSalience {
    /// Create with custom window size.
    pub fn with_window(mut self, window: usize) -> Self {
        self.window_size = window;
        self
    }

    /// Create with custom damping factor.
    pub fn with_damping(mut self, damping: f64) -> Self {
        self.damping = damping.clamp(0.0, 1.0);
        self
    }

    /// Build co-occurrence graph from entities.
    fn build_graph(&self, entities: &[Entity]) -> Vec<Vec<f64>> {
        let n = entities.len();
        if n == 0 {
            return vec![];
        }

        let mut adjacency = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in (i + 1)..n {
                // Check if entities co-occur within window
                let e1 = &entities[i];
                let e2 = &entities[j];

                // Distance between entity spans
                let dist = if e1.end <= e2.start {
                    e2.start - e1.end
                } else {
                    e1.start.saturating_sub(e2.end)
                };

                if dist <= self.window_size {
                    // Weight by inverse distance (closer = stronger)
                    let weight = 1.0 / (1.0 + dist as f64);
                    adjacency[i][j] = weight;
                    adjacency[j][i] = weight;
                }
            }
        }

        adjacency
    }

    /// Run PageRank on adjacency matrix using shared implementation.
    fn run_pagerank(&self, adjacency: &[Vec<f64>]) -> Vec<f64> {
        let config = PageRankConfig {
            damping: self.damping,
            max_iterations: self.iterations,
            epsilon: self.epsilon,
        };
        pagerank(adjacency, &config)
    }
}

impl EntityRanker for TextRankSalience {
    fn rank(&self, _text: &str, entities: &[Entity]) -> Vec<(Entity, f64)> {
        if entities.is_empty() {
            return vec![];
        }

        // Build co-occurrence graph
        let adjacency = self.build_graph(entities);

        // Run PageRank
        let scores = self.run_pagerank(&adjacency);

        // Combine entities with scores and sort
        let mut ranked: Vec<(Entity, f64)> = entities
            .iter()
            .zip(scores.iter())
            .map(|(e, s)| (e.clone(), *s))
            .collect();

        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        ranked
    }
}

// =============================================================================
// YAKE Implementation
// =============================================================================

/// YAKE-style statistical salience (Yet Another Keyword Extractor).
///
/// Uses statistical features without requiring a corpus:
/// - Term frequency
/// - Position (earlier = more salient)
/// - Spread (mentions across document = more salient)
/// - Capitalization (proper nouns = more salient)
///
/// # Reference
///
/// Campos et al. (2018): "YAKE! Collection-Independent Automatic Keyword
/// Extractor"
#[derive(Debug, Clone)]
pub struct YakeSalience {
    /// Weight for term frequency
    pub tf_weight: f64,
    /// Weight for position (earlier = higher)
    pub position_weight: f64,
    /// Weight for spread across document
    pub spread_weight: f64,
    /// Weight for capitalization
    pub case_weight: f64,
}

impl Default for YakeSalience {
    fn default() -> Self {
        Self {
            tf_weight: 1.0,
            position_weight: 0.5,
            spread_weight: 0.3,
            case_weight: 0.2,
        }
    }
}

impl YakeSalience {
    /// Compute features for an entity.
    fn compute_features(&self, entity: &Entity, text_len: usize, mentions: &[&Entity]) -> f64 {
        let mut score = 0.0;

        // Term frequency (log scale)
        let tf = (mentions.len() as f64).ln_1p();
        score += self.tf_weight * tf;

        // Position: earlier mentions are more salient
        // Normalize to [0, 1] and invert (earlier = higher)
        let first_pos = mentions.iter().map(|e| e.start).min().unwrap_or(0);
        let position_score = 1.0 - (first_pos as f64 / text_len.max(1) as f64);
        score += self.position_weight * position_score;

        // Spread: mentions across document indicate importance
        if mentions.len() > 1 {
            let first = mentions.iter().map(|e| e.start).min().unwrap_or(0);
            let last = mentions.iter().map(|e| e.end).max().unwrap_or(0);
            let spread = (last - first) as f64 / text_len.max(1) as f64;
            score += self.spread_weight * spread;
        }

        // Capitalization: proper nouns (entities with capitals) are often salient
        let has_capital = entity
            .text
            .chars()
            .next()
            .map(|c| c.is_uppercase())
            .unwrap_or(false);
        if has_capital {
            score += self.case_weight;
        }

        score
    }
}

impl EntityRanker for YakeSalience {
    fn rank(&self, text: &str, entities: &[Entity]) -> Vec<(Entity, f64)> {
        if entities.is_empty() {
            return vec![];
        }

        let text_len = text.len();

        // Group entities by normalized text (case-insensitive)
        let mut groups: HashMap<String, Vec<&Entity>> = HashMap::new();
        for entity in entities {
            let key = entity.text.to_lowercase();
            groups.entry(key).or_default().push(entity);
        }

        // Score each unique entity
        let mut scored: HashMap<String, (Entity, f64)> = HashMap::new();
        for (key, mentions) in &groups {
            // Use first mention as representative
            let representative = mentions[0];
            let score = self.compute_features(representative, text_len, mentions);
            scored.insert(key.clone(), (representative.clone(), score));
        }

        // Convert to sorted vec
        let mut ranked: Vec<(Entity, f64)> = scored.into_values().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        ranked
    }
}

// =============================================================================
// TF-IDF Implementation (simplified, single-document)
// =============================================================================

/// Simple TF-IDF-style salience (term frequency only for single doc).
///
/// For single-document ranking, this reduces to term frequency
/// with optional length normalization.
#[derive(Debug, Clone)]
pub struct TfIdfSalience {
    /// Use log scaling for term frequency
    pub log_tf: bool,
    /// Normalize by entity length (longer entities = lower score per char)
    pub length_normalize: bool,
}

impl Default for TfIdfSalience {
    fn default() -> Self {
        Self {
            log_tf: true,
            length_normalize: false,
        }
    }
}

impl EntityRanker for TfIdfSalience {
    fn rank(&self, _text: &str, entities: &[Entity]) -> Vec<(Entity, f64)> {
        if entities.is_empty() {
            return vec![];
        }

        // Count entity frequencies
        let mut freq: HashMap<String, (Entity, usize)> = HashMap::new();
        for entity in entities {
            let key = entity.text.to_lowercase();
            freq.entry(key)
                .and_modify(|(_, count)| *count += 1)
                .or_insert((entity.clone(), 1));
        }

        // Compute scores
        let mut ranked: Vec<(Entity, f64)> = freq
            .into_values()
            .map(|(entity, count)| {
                let mut score = if self.log_tf {
                    (count as f64).ln_1p()
                } else {
                    count as f64
                };

                if self.length_normalize {
                    score /= entity.text.len().max(1) as f64;
                }

                (entity, score)
            })
            .collect();

        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        ranked
    }
}

// =============================================================================
// Position-based Salience (simple heuristic)
// =============================================================================

/// Simple position-based salience: earlier + more frequent = more salient.
///
/// A fast heuristic baseline that doesn't require graph construction.
#[derive(Debug, Clone, Default)]
pub struct PositionSalience {
    /// Weight for frequency
    pub freq_weight: f64,
    /// Weight for position (earlier = higher)
    pub position_weight: f64,
    /// Boost for entities in first 10% of document
    pub lead_boost: f64,
}

impl PositionSalience {
    /// Create with default weights.
    pub fn new() -> Self {
        Self {
            freq_weight: 1.0,
            position_weight: 0.5,
            lead_boost: 0.3,
        }
    }
}

impl EntityRanker for PositionSalience {
    fn rank(&self, text: &str, entities: &[Entity]) -> Vec<(Entity, f64)> {
        if entities.is_empty() {
            return vec![];
        }

        let text_len = text.len();
        let lead_cutoff = text_len / 10;

        // Group by entity text
        let mut groups: HashMap<String, Vec<&Entity>> = HashMap::new();
        for entity in entities {
            let key = entity.text.to_lowercase();
            groups.entry(key).or_default().push(entity);
        }

        let mut ranked: Vec<(Entity, f64)> = groups
            .into_values()
            .map(|mentions| {
                let entity = mentions[0].clone();
                let mut score = 0.0;

                // Frequency
                score += self.freq_weight * (mentions.len() as f64).ln_1p();

                // Position (normalized, inverted)
                let first_pos = mentions.iter().map(|e| e.start).min().unwrap_or(0);
                score += self.position_weight * (1.0 - first_pos as f64 / text_len.max(1) as f64);

                // Lead boost
                if first_pos < lead_cutoff {
                    score += self.lead_boost;
                }

                (entity, score)
            })
            .collect();

        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        ranked
    }
}

// =============================================================================
// Composite Ranker
// =============================================================================

/// Combine multiple rankers with weighted averaging.
#[derive(Debug)]
pub struct CompositeRanker {
    rankers: Vec<(Box<dyn EntityRanker>, f64)>,
}

impl CompositeRanker {
    /// Create a new composite ranker.
    pub fn new() -> Self {
        Self {
            rankers: Vec::new(),
        }
    }

    /// Add a ranker with weight.
    pub fn add<R: EntityRanker + 'static>(mut self, ranker: R, weight: f64) -> Self {
        self.rankers.push((Box::new(ranker), weight));
        self
    }
}

impl Default for CompositeRanker {
    fn default() -> Self {
        Self::new()
    }
}

impl EntityRanker for CompositeRanker {
    fn rank(&self, text: &str, entities: &[Entity]) -> Vec<(Entity, f64)> {
        if entities.is_empty() || self.rankers.is_empty() {
            return vec![];
        }

        // Get rankings from each ranker
        let rankings: Vec<Vec<(Entity, f64)>> = self
            .rankers
            .iter()
            .map(|(ranker, _)| ranker.rank(text, entities))
            .collect();

        // Normalize scores within each ranking to [0, 1]
        let normalized: Vec<HashMap<String, f64>> = rankings
            .iter()
            .map(|ranking| {
                let max_score = ranking
                    .iter()
                    .map(|(_, s)| *s)
                    .fold(f64::NEG_INFINITY, f64::max);
                let min_score = ranking
                    .iter()
                    .map(|(_, s)| *s)
                    .fold(f64::INFINITY, f64::min);
                let range = (max_score - min_score).max(1e-10);

                ranking
                    .iter()
                    .map(|(e, s)| (e.text.to_lowercase(), (s - min_score) / range))
                    .collect()
            })
            .collect();

        // Weighted average
        let mut combined: HashMap<String, (Entity, f64)> = HashMap::new();
        let total_weight: f64 = self.rankers.iter().map(|(_, w)| w).sum();

        for entity in entities {
            let key = entity.text.to_lowercase();
            let mut weighted_sum = 0.0;

            for (i, (_, weight)) in self.rankers.iter().enumerate() {
                if let Some(score) = normalized[i].get(&key) {
                    weighted_sum += score * weight;
                }
            }

            let final_score = weighted_sum / total_weight;
            combined.entry(key).or_insert((entity.clone(), final_score));
        }

        let mut ranked: Vec<(Entity, f64)> = combined.into_values().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        ranked
    }
}

// Make EntityRanker object-safe
impl std::fmt::Debug for dyn EntityRanker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "EntityRanker")
    }
}

// =============================================================================
// Chain-Feature Based Salience
// =============================================================================

/// Chain-feature based salience ranking.
///
/// Uses aggregate chain features (frequency, spread, mention type distribution)
/// to compute entity salience. This integrates with the `features` module
/// for comprehensive entity analysis.
///
/// # Features Used
///
/// | Feature | Weight | Rationale |
/// |---------|--------|-----------|
/// | Chain length | High | More mentions = more salient |
/// | Spread | Medium | Mentions across document = important |
/// | Named ratio | Medium | Named mentions indicate main entities |
/// | First position | Low | Earlier = slightly more salient |
///
/// # Example
///
/// ```rust,ignore
/// use anno::salience::{EntityRanker, ChainFeatureSalience};
///
/// let text = "Barack Obama met Angela Merkel. Obama discussed policy with her.";
/// let entities = ner.extract_entities(text, None)?;
///
/// let ranker = ChainFeatureSalience::default();
/// let ranked = ranker.rank(text, &entities);
/// // Obama has longer chain, ranked higher
/// ```
#[derive(Debug, Clone)]
pub struct ChainFeatureSalience {
    /// Weight for chain length (frequency).
    pub length_weight: f64,
    /// Weight for mention spread.
    pub spread_weight: f64,
    /// Weight for named mention ratio.
    pub named_ratio_weight: f64,
    /// Weight for first position (earlier = higher).
    pub position_weight: f64,
    /// Weight for confidence.
    pub confidence_weight: f64,
    /// Feature extractor configuration.
    extractor_config: ExtractorConfig,
}

impl Default for ChainFeatureSalience {
    fn default() -> Self {
        Self {
            length_weight: 1.0,
            spread_weight: 0.5,
            named_ratio_weight: 0.3,
            position_weight: 0.2,
            confidence_weight: 0.1,
            extractor_config: ExtractorConfig::default(),
        }
    }
}

impl ChainFeatureSalience {
    /// Create with custom extractor config.
    pub fn with_config(mut self, config: ExtractorConfig) -> Self {
        self.extractor_config = config;
        self
    }

    /// Compute salience from chain features.
    fn chain_salience(&self, features: &ChainFeatures, text_len: usize) -> f64 {
        let mut score = 0.0;

        // Chain length (log scale to avoid dominance)
        let length_score = (features.chain_length as f64).ln_1p();
        score += self.length_weight * length_score;

        // Spread: mentions across document indicate importance
        score += self.spread_weight * features.relative_spread;

        // Named mention ratio: named > nominal > pronominal
        let named_ratio = if features.chain_length > 0 {
            features.named_count as f64 / features.chain_length as f64
        } else {
            0.0
        };
        score += self.named_ratio_weight * named_ratio;

        // Position: earlier mentions slightly more salient
        let position_score = if text_len > 0 {
            1.0 - (features.first_mention_position as f64 / text_len as f64)
        } else {
            0.0
        };
        score += self.position_weight * position_score;

        // Confidence
        score += self.confidence_weight * features.mean_confidence;

        score
    }
}

impl EntityRanker for ChainFeatureSalience {
    fn rank(&self, text: &str, entities: &[Entity]) -> Vec<(Entity, f64)> {
        if entities.is_empty() {
            return vec![];
        }

        let text_len = text.chars().count();
        let extractor = EntityFeatureExtractor::new(self.extractor_config.clone());
        let chain_features = extractor.extract_chains(text, entities);

        // Compute salience for each chain
        let mut scores: HashMap<String, (Entity, f64)> = HashMap::new();

        for (key, features) in &chain_features {
            let salience = self.chain_salience(features, text_len);

            // Find representative entity (prefer named, longest)
            let representative = entities
                .iter()
                .filter(|e| e.text.to_lowercase() == *key)
                .max_by_key(|e| {
                    let is_named = MentionType::classify(&e.text) == MentionType::Proper;
                    (is_named as usize, e.text.len())
                })
                .cloned()
                .unwrap_or_else(|| entities[0].clone());

            scores.insert(key.clone(), (representative, salience));
        }

        let mut ranked: Vec<(Entity, f64)> = scores.into_values().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        ranked
    }
}

// =============================================================================
// Helper: Convert features to salience scores for coref integration
// =============================================================================

/// Convert entity features to salience scores for coreference integration.
///
/// This function extracts chain features and converts them to a HashMap
/// suitable for use with `MentionRankingCoref::with_salience()`.
///
/// # Example
///
/// ```rust,ignore
/// use anno::salience::features_to_salience_scores;
/// use anno::backends::mention_ranking::MentionRankingCoref;
///
/// let entities = ner.extract_entities(text, None)?;
/// let salience_scores = features_to_salience_scores(text, &entities);
///
/// let coref = MentionRankingCoref::new()
///     .with_salience(salience_scores);
/// ```
pub fn features_to_salience_scores(text: &str, entities: &[Entity]) -> HashMap<String, f64> {
    let ranker = ChainFeatureSalience::default();
    let ranked = ranker.rank(text, entities);

    // Normalize scores to [0, 1]
    let max_score = ranked
        .iter()
        .map(|(_, s)| *s)
        .fold(f64::NEG_INFINITY, f64::max);

    if max_score <= 0.0 {
        return HashMap::new();
    }

    ranked
        .into_iter()
        .map(|(e, score)| (e.text.to_lowercase(), score / max_score))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EntityType;

    fn sample_entities() -> Vec<Entity> {
        vec![
            Entity::new("Barack Obama", EntityType::Person, 0, 12, 0.9),
            Entity::new("Angela Merkel", EntityType::Person, 17, 30, 0.9),
            Entity::new("Berlin", EntityType::Location, 34, 40, 0.9),
            Entity::new("Obama", EntityType::Person, 50, 55, 0.9),
        ]
    }

    #[test]
    fn test_textrank_ranking() {
        let text = "Barack Obama met Angela Merkel in Berlin. Obama discussed policy.";
        let entities = sample_entities();

        let ranker = TextRankSalience::default();
        let ranked = ranker.rank(text, &entities);

        assert_eq!(ranked.len(), 4);
        // All should have positive scores
        for (_, score) in &ranked {
            assert!(*score > 0.0);
        }
    }

    #[test]
    fn test_yake_ranking() {
        let text = "Barack Obama met Angela Merkel in Berlin. Obama discussed policy.";
        let entities = sample_entities();

        let ranker = YakeSalience::default();
        let ranked = ranker.rank(text, &entities);

        // Obama mentioned twice, should rank high
        assert!(!ranked.is_empty());

        // Check that we get unique entities (by text)
        let unique_count = ranked.len();
        assert!(unique_count <= 4); // May be less due to deduplication
    }

    #[test]
    fn test_tfidf_ranking() {
        let text = "Obama Obama Obama Merkel Berlin";
        let entities = vec![
            Entity::new("Obama", EntityType::Person, 0, 5, 0.9),
            Entity::new("Obama", EntityType::Person, 6, 11, 0.9),
            Entity::new("Obama", EntityType::Person, 12, 17, 0.9),
            Entity::new("Merkel", EntityType::Person, 18, 24, 0.9),
            Entity::new("Berlin", EntityType::Location, 25, 31, 0.9),
        ];

        let ranker = TfIdfSalience::default();
        let ranked = ranker.rank(text, &entities);

        // Obama mentioned 3x should rank highest
        assert_eq!(ranked[0].0.text.to_lowercase(), "obama");
    }

    #[test]
    fn test_position_ranking() {
        let text = "First Entity appears here. Later Entity appears here.";
        let entities = vec![
            Entity::new(
                "First Entity",
                EntityType::Other("test".to_string()),
                0,
                12,
                0.9,
            ),
            Entity::new(
                "Later Entity",
                EntityType::Other("test".to_string()),
                27,
                39,
                0.9,
            ),
        ];

        let ranker = PositionSalience::new();
        let ranked = ranker.rank(text, &entities);

        // First entity should rank higher due to position
        assert_eq!(ranked[0].0.text, "First Entity");
    }

    #[test]
    fn test_composite_ranker() {
        let text = "Barack Obama met Angela Merkel in Berlin.";
        let entities = sample_entities();

        let ranker = CompositeRanker::new()
            .add(TextRankSalience::default(), 1.0)
            .add(YakeSalience::default(), 0.5);

        let ranked = ranker.rank(text, &entities);
        assert!(!ranked.is_empty());
    }

    #[test]
    fn test_top_k() {
        let text = "A B C D E";
        let entities = vec![
            Entity::new("A", EntityType::Person, 0, 1, 0.9),
            Entity::new("B", EntityType::Person, 2, 3, 0.9),
            Entity::new("C", EntityType::Person, 4, 5, 0.9),
            Entity::new("D", EntityType::Person, 6, 7, 0.9),
            Entity::new("E", EntityType::Person, 8, 9, 0.9),
        ];

        let ranker = PositionSalience::new();
        let top3 = ranker.top_k(text, &entities, 3);

        assert_eq!(top3.len(), 3);
    }

    #[test]
    fn test_empty_entities() {
        let ranker = TextRankSalience::default();
        let ranked = ranker.rank("some text", &[]);
        assert!(ranked.is_empty());
    }

    // =========================================================================
    // Edge Case and Integration Tests
    // =========================================================================

    /// Test: Single entity should still work
    #[test]
    fn test_single_entity() {
        let text = "Barack Obama spoke.";
        let entities = vec![Entity::new("Barack Obama", EntityType::Person, 0, 12, 0.9)];

        let ranker = TextRankSalience::default();
        let ranked = ranker.rank(text, &entities);

        assert_eq!(ranked.len(), 1);
        assert!(
            ranked[0].1 > 0.0,
            "Single entity should have positive score"
        );
    }

    /// Test: All entities with equal mentions
    #[test]
    fn test_equal_frequency_entities() {
        let text = "Alice met Bob. Carol met Dave.";
        let entities = vec![
            Entity::new("Alice", EntityType::Person, 0, 5, 0.9),
            Entity::new("Bob", EntityType::Person, 10, 13, 0.9),
            Entity::new("Carol", EntityType::Person, 15, 20, 0.9),
            Entity::new("Dave", EntityType::Person, 25, 29, 0.9),
        ];

        let ranker = TfIdfSalience::default();
        let ranked = ranker.rank(text, &entities);

        assert_eq!(ranked.len(), 4);
        // All should have similar (possibly equal) scores
        let scores: Vec<_> = ranked.iter().map(|(_, s)| *s).collect();
        let max_diff = scores
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
            - scores
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();
        // Equal frequency should give relatively similar scores
        assert!(
            max_diff < 0.5,
            "Equal frequency entities should have similar scores"
        );
    }

    /// Test: Repeated entity should rank higher
    #[test]
    fn test_repeated_entity_ranks_higher() {
        let text = "Obama Obama Obama Obama. Merkel.";
        let entities = vec![
            Entity::new("Obama", EntityType::Person, 0, 5, 0.9),
            Entity::new("Obama", EntityType::Person, 6, 11, 0.9),
            Entity::new("Obama", EntityType::Person, 12, 17, 0.9),
            Entity::new("Obama", EntityType::Person, 18, 23, 0.9),
            Entity::new("Merkel", EntityType::Person, 25, 31, 0.9),
        ];

        let ranker = TfIdfSalience::default();
        let ranked = ranker.rank(text, &entities);

        // Obama should rank higher than Merkel
        assert_eq!(
            ranked[0].0.text.to_lowercase(),
            "obama",
            "Most frequent entity should rank first"
        );
    }

    /// Test: Entity at beginning ranks higher (position salience)
    #[test]
    fn test_position_salience_first_sentence() {
        let text = "Alice is important. Later we mention Bob. Finally Carol.";
        let entities = vec![
            Entity::new("Alice", EntityType::Person, 0, 5, 0.9),
            Entity::new("Bob", EntityType::Person, 37, 40, 0.9),
            Entity::new("Carol", EntityType::Person, 50, 55, 0.9),
        ];

        let ranker = PositionSalience::new();
        let ranked = ranker.rank(text, &entities);

        assert_eq!(
            ranked[0].0.text, "Alice",
            "First entity should rank highest"
        );
    }

    /// Test: Multilingual entities
    #[test]
    fn test_multilingual_salience() {
        let text = "習近平在北京會見普京。習近平發表講話。";
        let entities = vec![
            Entity::new("習近平", EntityType::Person, 0, 9, 0.9),
            Entity::new("北京", EntityType::Location, 12, 18, 0.9),
            Entity::new("普京", EntityType::Person, 21, 27, 0.9),
            Entity::new("習近平", EntityType::Person, 30, 39, 0.9),
        ];

        let ranker = TfIdfSalience::default();
        let ranked = ranker.rank(text, &entities);

        // 習近平 mentioned twice, should rank higher
        assert_eq!(ranked[0].0.text, "習近平");
    }

    /// Test: Unicode normalization in salience
    #[test]
    fn test_unicode_entities_salience() {
        let text = "Müller met Müller again."; // same name, different normalization possible
        let entities = vec![
            Entity::new("Müller", EntityType::Person, 0, 6, 0.9),
            Entity::new("Müller", EntityType::Person, 11, 17, 0.9),
        ];

        let ranker = TextRankSalience::default();
        let ranked = ranker.rank(text, &entities);

        // Should handle unicode gracefully
        assert!(!ranked.is_empty());
    }

    /// Test: Very long text with many entities
    #[test]
    fn test_many_entities_performance() {
        let mut text = String::new();
        let mut entities = Vec::new();

        for i in 0..50 {
            let name = format!("Entity{}", i);
            let start = text.len();
            text.push_str(&name);
            text.push(' ');
            entities.push(Entity::new(name, EntityType::Person, start, start + 7, 0.9));
        }

        let ranker = TextRankSalience::default();
        let ranked = ranker.rank(&text, &entities);

        assert_eq!(ranked.len(), 50);
    }

    /// Test: CompositeRanker with different weights
    #[test]
    fn test_composite_ranker_weights() {
        let text = "First Entity. Second Entity Second Entity.";
        let entities = vec![
            Entity::new("First Entity", EntityType::Person, 0, 12, 0.9),
            Entity::new("Second Entity", EntityType::Person, 14, 27, 0.9),
            Entity::new("Second Entity", EntityType::Person, 28, 41, 0.9),
        ];

        // Position-heavy weighting
        let pos_heavy = CompositeRanker::new()
            .add(PositionSalience::new(), 10.0)
            .add(TfIdfSalience::default(), 1.0);
        let ranked_pos = pos_heavy.rank(text, &entities);

        // Frequency-heavy weighting
        let freq_heavy = CompositeRanker::new()
            .add(PositionSalience::new(), 1.0)
            .add(TfIdfSalience::default(), 10.0);
        let ranked_freq = freq_heavy.rank(text, &entities);

        // Position-heavy should favor "First Entity"
        // Frequency-heavy should favor "Second Entity"
        // (Both are valid rankings depending on use case)
        assert!(!ranked_pos.is_empty());
        assert!(!ranked_freq.is_empty());
    }

    /// Test: top_k with k > entities.len()
    #[test]
    fn test_top_k_exceeds_count() {
        let text = "A B";
        let entities = vec![
            Entity::new("A", EntityType::Person, 0, 1, 0.9),
            Entity::new("B", EntityType::Person, 2, 3, 0.9),
        ];

        let ranker = PositionSalience::new();
        let top10 = ranker.top_k(text, &entities, 10);

        assert_eq!(top10.len(), 2, "Should return all entities if k > count");
    }

    /// Test: top_k with k = 0
    #[test]
    fn test_top_k_zero() {
        let text = "A B C";
        let entities = vec![
            Entity::new("A", EntityType::Person, 0, 1, 0.9),
            Entity::new("B", EntityType::Person, 2, 3, 0.9),
            Entity::new("C", EntityType::Person, 4, 5, 0.9),
        ];

        let ranker = PositionSalience::new();
        let top0 = ranker.top_k(text, &entities, 0);

        assert!(top0.is_empty(), "top_k(0) should return empty");
    }

    /// Test: Entities with overlapping spans
    #[test]
    fn test_overlapping_entity_spans() {
        // "New York City" contains "York"
        let text = "New York City is nice.";
        let entities = vec![
            Entity::new("New York City", EntityType::Location, 0, 13, 0.9),
            Entity::new("York", EntityType::Location, 4, 8, 0.8),
        ];

        let ranker = TextRankSalience::default();
        let ranked = ranker.rank(text, &entities);

        // Should handle overlapping spans
        assert_eq!(ranked.len(), 2);
    }

    /// Test: Very short text
    #[test]
    fn test_very_short_text() {
        let text = "Hi";
        let entities = vec![Entity::new(
            "Hi",
            EntityType::Other("greeting".to_string()),
            0,
            2,
            0.9,
        )];

        let ranker = TextRankSalience::default();
        let ranked = ranker.rank(text, &entities);

        assert_eq!(ranked.len(), 1);
    }

    /// Test: Entity types preserved in ranking
    #[test]
    fn test_entity_types_preserved() {
        let text = "Obama in Berlin";
        let entities = vec![
            Entity::new("Obama", EntityType::Person, 0, 5, 0.9),
            Entity::new("Berlin", EntityType::Location, 9, 15, 0.9),
        ];

        let ranker = PositionSalience::new();
        let ranked = ranker.rank(text, &entities);

        // Types should be preserved
        assert!(matches!(ranked[0].0.entity_type, EntityType::Person));
        assert!(matches!(ranked[1].0.entity_type, EntityType::Location));
    }
}
