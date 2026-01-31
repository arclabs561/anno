//! Entity feature extraction for downstream ML and analysis.
//!
//! This module provides comprehensive feature extraction for entities at multiple
//! levels of granularity:
//!
//! - **Mention-level**: Context windows, position, syntactic role
//! - **Chain/Track-level**: Aggregate statistics across coreference chains
//! - **Co-occurrence**: Which entities appear together
//! - **Document-level**: Cross-entity patterns
//!
//! # Use Cases
//!
//! - Training coreference models (pairwise features)
//! - Entity classification/linking (mention context)
//! - Knowledge graph construction (co-occurrence patterns)
//! - Entity salience prediction (aggregate features)
//!
//! # Example
//!
//! ```rust,ignore
//! use anno::features::{EntityFeatureExtractor, ExtractorConfig};
//! use anno::{Model, StackedNER};
//!
//! let text = "Barack Obama met Angela Merkel in Berlin. He discussed policy with her.";
//! let ner = StackedNER::default();
//! let entities = ner.extract_entities(text, None)?;
//!
//! let extractor = EntityFeatureExtractor::new(ExtractorConfig::default());
//! let features = extractor.extract_all(text, &entities);
//!
//! // Get co-occurring entities for "Obama"
//! let obama_cooc = features.cooccurrence.get("barack obama").unwrap();
//! assert!(obama_cooc.cooccurring_entities.contains(&"angela merkel".to_string()));
//! ```

use crate::Entity;
use std::collections::{HashMap, HashSet};

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for entity feature extraction.
#[derive(Debug, Clone)]
pub struct ExtractorConfig {
    /// Window size (in characters) for context extraction around mentions.
    pub context_window: usize,
    /// Window size (in characters) for co-occurrence detection.
    pub cooccurrence_window: usize,
    /// Whether to normalize text (lowercase) for grouping.
    pub normalize_text: bool,
    /// Minimum frequency for an entity to be included in co-occurrence.
    pub min_cooccurrence_freq: usize,
}

impl Default for ExtractorConfig {
    fn default() -> Self {
        Self {
            context_window: 100,
            cooccurrence_window: 150,
            normalize_text: true,
            min_cooccurrence_freq: 1,
        }
    }
}

impl ExtractorConfig {
    /// Create config with custom context window.
    pub fn with_context_window(mut self, window: usize) -> Self {
        self.context_window = window;
        self
    }

    /// Create config with custom co-occurrence window.
    pub fn with_cooccurrence_window(mut self, window: usize) -> Self {
        self.cooccurrence_window = window;
        self
    }
}

// =============================================================================
// Mention-Level Features
// =============================================================================

/// Context and features for a single entity mention.
#[derive(Debug, Clone)]
pub struct MentionContext {
    /// The entity mention itself.
    pub entity: Entity,
    /// Text before the mention (up to context_window chars).
    pub left_context: String,
    /// Text after the mention (up to context_window chars).
    pub right_context: String,
    /// Position as fraction of document (0.0 = start, 1.0 = end).
    pub relative_position: f64,
    /// Character offset from document start.
    pub absolute_position: usize,
    /// Sentence index (if sentence boundaries detected).
    pub sentence_index: Option<usize>,
    /// Is this likely in subject position? (heuristic: near sentence start).
    pub likely_subject: bool,
    /// Is this in a heading or title? (heuristic: short line, capitalized).
    pub likely_heading: bool,
    /// Word count of the mention.
    pub word_count: usize,
    /// Character count of the mention.
    pub char_count: usize,
    /// Does the mention start with a capital letter?
    pub is_capitalized: bool,
    /// Is this mention all uppercase?
    pub is_all_caps: bool,
    /// Contains digits?
    pub contains_digits: bool,
}

impl MentionContext {
    /// Extract mention context from text and entity.
    pub fn extract(text: &str, entity: &Entity, config: &ExtractorConfig) -> Self {
        let text_chars: Vec<char> = text.chars().collect();
        let text_len = text_chars.len();

        // Safe bounds for context extraction
        let left_start = entity.start.saturating_sub(config.context_window);
        let left_end = entity.start;
        let right_start = entity.end.min(text_len);
        let right_end = (entity.end + config.context_window).min(text_len);

        let left_context: String = text_chars[left_start..left_end].iter().collect();
        let right_context: String = text_chars[right_start..right_end].iter().collect();

        let relative_position = if text_len > 0 {
            entity.start as f64 / text_len as f64
        } else {
            0.0
        };

        // Heuristic: likely subject if within first 50 chars of a sentence
        // (simplified: check if near a period/newline in left context)
        let likely_subject = {
            let trimmed = left_context.trim_end();
            trimmed.is_empty()
                || trimmed.ends_with('.')
                || trimmed.ends_with('!')
                || trimmed.ends_with('?')
                || trimmed.ends_with('\n')
                || trimmed.len() < 50
        };

        // Heuristic: likely heading if the line is short and mostly capitalized
        let likely_heading = {
            let line_start = left_context.rfind('\n').map(|i| i + 1).unwrap_or(0);
            let line_end = right_context.find('\n').unwrap_or(right_context.len());
            let line_len = (left_context.len() - line_start) + entity.text.len() + line_end;
            line_len < 100
                && entity
                    .text
                    .chars()
                    .next()
                    .map(|c| c.is_uppercase())
                    .unwrap_or(false)
        };

        let first_char = entity.text.chars().next();
        let is_capitalized = first_char.map(|c| c.is_uppercase()).unwrap_or(false);
        let is_all_caps = entity
            .text
            .chars()
            .all(|c| !c.is_alphabetic() || c.is_uppercase());
        let contains_digits = entity.text.chars().any(|c| c.is_ascii_digit());

        Self {
            entity: entity.clone(),
            left_context,
            right_context,
            relative_position,
            absolute_position: entity.start,
            sentence_index: None, // Would need sentence segmentation
            likely_subject,
            likely_heading,
            word_count: entity.text.split_whitespace().count(),
            char_count: entity.text.chars().count(),
            is_capitalized,
            is_all_caps,
            contains_digits,
        }
    }

    /// Get the full context string (left + entity + right).
    pub fn full_context(&self) -> String {
        format!(
            "{}[{}]{}",
            self.left_context, self.entity.text, self.right_context
        )
    }
}

// =============================================================================
// Chain/Track-Level Features
// =============================================================================

// Re-export the canonical MentionType from anno_core.
// This unifies the type system across the anno ecosystem.
//
// Note: The canonical type uses `Proper` instead of `Named`. For compatibility,
// use `MentionType::NAMED` constant or `MentionType::is_named()` method.
pub use anno_core::MentionType;

/// Aggregate features for a coreference chain (group of mentions referring to same entity).
#[derive(Debug, Clone)]
pub struct ChainFeatures {
    /// Canonical/representative surface form.
    pub canonical_form: String,
    /// All surface form variations observed.
    pub variations: Vec<String>,
    /// Number of mentions in the chain.
    pub chain_length: usize,
    /// Entity type (from first/canonical mention).
    pub entity_type: Option<String>,

    // Positional features
    /// Position of first mention (character offset).
    pub first_mention_position: usize,
    /// Position of last mention (character offset).
    pub last_mention_position: usize,
    /// Spread: distance from first to last mention.
    pub mention_spread: usize,
    /// Spread as fraction of document length.
    pub relative_spread: f64,

    // Type distribution
    /// Count of named mentions.
    pub named_count: usize,
    /// Count of nominal mentions.
    pub nominal_count: usize,
    /// Count of pronominal mentions.
    pub pronominal_count: usize,
    /// Fraction of mentions that are pronominal.
    pub pronoun_ratio: f64,

    // Statistical features
    /// Mean position of mentions.
    pub mean_position: f64,
    /// Positional entropy (how spread out are mentions?).
    pub positional_entropy: f64,
    /// Mean confidence across mentions.
    pub mean_confidence: f64,
    /// Min confidence.
    pub min_confidence: f64,
    /// Max confidence.
    pub max_confidence: f64,

    // Aggregate embedding (if available)
    /// Centroid embedding (mean of mention embeddings, if available).
    pub centroid_embedding: Option<Vec<f32>>,
}

impl ChainFeatures {
    /// Compute chain features from a group of related entity mentions.
    pub fn from_mentions(mentions: &[&Entity], text_len: usize) -> Self {
        if mentions.is_empty() {
            return Self::empty();
        }

        // Collect variations
        let mut variations_set: HashSet<String> = HashSet::new();
        for m in mentions {
            variations_set.insert(m.text.clone());
        }
        let variations: Vec<String> = variations_set.into_iter().collect();

        // Find canonical form (longest named mention, or first)
        let canonical_form = mentions
            .iter()
            .filter(|m| MentionType::classify(&m.text) == MentionType::Proper)
            .max_by_key(|m| m.text.len())
            .map(|m| m.text.clone())
            .unwrap_or_else(|| mentions[0].text.clone());

        // Positions
        let first_pos = mentions.iter().map(|m| m.start).min().unwrap_or(0);
        let last_pos = mentions.iter().map(|m| m.end).max().unwrap_or(0);
        let spread = last_pos.saturating_sub(first_pos);
        let relative_spread = if text_len > 0 {
            spread as f64 / text_len as f64
        } else {
            0.0
        };

        // Type distribution
        let mut named_count = 0;
        let mut nominal_count = 0;
        let mut pronominal_count = 0;
        for m in mentions {
            match MentionType::classify(&m.text) {
                MentionType::Proper => named_count += 1,
                MentionType::Nominal => nominal_count += 1,
                MentionType::Pronominal => pronominal_count += 1,
                MentionType::Zero => pronominal_count += 1, // Treat zeros like pronouns
                MentionType::Unknown => nominal_count += 1, // Conservative default
            }
        }
        let total = mentions.len();
        let pronoun_ratio = pronominal_count as f64 / total as f64;

        // Statistical features
        let positions: Vec<f64> = mentions.iter().map(|m| m.start as f64).collect();
        let mean_position = positions.iter().sum::<f64>() / total as f64;

        // Positional entropy: how spread out?
        let positional_entropy = if text_len > 0 && total > 1 {
            let n_bins = 10;
            let bin_size = text_len / n_bins;
            let mut bins = vec![0usize; n_bins];
            for m in mentions {
                let bin = (m.start / bin_size.max(1)).min(n_bins - 1);
                bins[bin] += 1;
            }
            let total_f = total as f64;
            bins.iter()
                .filter(|&&c| c > 0)
                .map(|&c| {
                    let p = c as f64 / total_f;
                    -p * p.ln()
                })
                .sum()
        } else {
            0.0
        };

        // Confidence stats
        let confidences: Vec<f64> = mentions.iter().map(|m| m.confidence).collect();
        let mean_confidence = confidences.iter().sum::<f64>() / total as f64;
        let min_confidence = confidences.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_confidence = confidences
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        // Entity type from first named mention or first mention
        let entity_type = mentions
            .iter()
            .find(|m| MentionType::classify(&m.text) == MentionType::Proper)
            .or_else(|| mentions.first())
            .map(|m| m.entity_type.as_label().to_string());

        Self {
            canonical_form,
            variations,
            chain_length: total,
            entity_type,
            first_mention_position: first_pos,
            last_mention_position: last_pos,
            mention_spread: spread,
            relative_spread,
            named_count,
            nominal_count,
            pronominal_count,
            pronoun_ratio,
            mean_position,
            positional_entropy,
            mean_confidence,
            min_confidence,
            max_confidence,
            centroid_embedding: None,
        }
    }

    /// Create empty chain features.
    fn empty() -> Self {
        Self {
            canonical_form: String::new(),
            variations: Vec::new(),
            chain_length: 0,
            entity_type: None,
            first_mention_position: 0,
            last_mention_position: 0,
            mention_spread: 0,
            relative_spread: 0.0,
            named_count: 0,
            nominal_count: 0,
            pronominal_count: 0,
            pronoun_ratio: 0.0,
            mean_position: 0.0,
            positional_entropy: 0.0,
            mean_confidence: 0.0,
            min_confidence: 0.0,
            max_confidence: 0.0,
            centroid_embedding: None,
        }
    }

    /// Set the centroid embedding.
    pub fn with_centroid(mut self, embedding: Vec<f32>) -> Self {
        self.centroid_embedding = Some(embedding);
        self
    }

    /// Is this a singleton chain (single mention)?
    pub fn is_singleton(&self) -> bool {
        self.chain_length == 1
    }

    /// Is this chain mostly pronominal?
    pub fn is_mostly_pronominal(&self) -> bool {
        self.pronoun_ratio > 0.5
    }

    /// Number of unique surface form variations.
    #[must_use]
    pub fn variation_count(&self) -> usize {
        self.variations.len()
    }
}

// =============================================================================
// Co-occurrence Features
// =============================================================================

/// Co-occurrence features for an entity.
#[derive(Debug, Clone)]
pub struct CooccurrenceFeatures {
    /// The entity (normalized key).
    pub entity_key: String,
    /// Entities that co-occur within the window.
    pub cooccurring_entities: Vec<String>,
    /// Co-occurrence counts per entity.
    pub cooccurrence_counts: HashMap<String, usize>,
    /// Total co-occurrence count (sum of all).
    pub total_cooccurrences: usize,
    /// Unique co-occurring entity count.
    pub unique_cooccurrences: usize,
    /// Entity types of co-occurring entities.
    pub cooccurring_types: HashMap<String, Vec<String>>,
}

impl CooccurrenceFeatures {
    /// Create new co-occurrence features for an entity.
    pub fn new(entity_key: String) -> Self {
        Self {
            entity_key,
            cooccurring_entities: Vec::new(),
            cooccurrence_counts: HashMap::new(),
            total_cooccurrences: 0,
            unique_cooccurrences: 0,
            cooccurring_types: HashMap::new(),
        }
    }

    /// Add a co-occurring entity.
    pub fn add_cooccurrence(&mut self, other_key: &str, other_type: Option<&str>) {
        *self
            .cooccurrence_counts
            .entry(other_key.to_string())
            .or_insert(0) += 1;
        self.total_cooccurrences += 1;

        if let Some(t) = other_type {
            self.cooccurring_types
                .entry(other_key.to_string())
                .or_default()
                .push(t.to_string());
        }
    }

    /// Finalize the features (dedupe, sort, count).
    pub fn finalize(&mut self) {
        self.cooccurring_entities = self.cooccurrence_counts.keys().cloned().collect();
        self.cooccurring_entities.sort_by(|a, b| {
            self.cooccurrence_counts
                .get(b)
                .cmp(&self.cooccurrence_counts.get(a))
        });
        self.unique_cooccurrences = self.cooccurring_entities.len();
    }

    /// Get top-k co-occurring entities by count.
    pub fn top_k(&self, k: usize) -> Vec<(&str, usize)> {
        self.cooccurring_entities
            .iter()
            .take(k)
            .filter_map(|e| self.cooccurrence_counts.get(e).map(|&c| (e.as_str(), c)))
            .collect()
    }
}

// =============================================================================
// Document-Level Feature Collection
// =============================================================================

/// Complete feature extraction results for a document.
#[derive(Debug, Clone)]
pub struct DocumentFeatures {
    /// Mention-level features for each entity occurrence.
    pub mention_contexts: Vec<MentionContext>,
    /// Chain features grouped by normalized entity key.
    pub chain_features: HashMap<String, ChainFeatures>,
    /// Co-occurrence features per entity.
    pub cooccurrence: HashMap<String, CooccurrenceFeatures>,
    /// Document-level statistics.
    pub document_stats: DocumentStats,
}

/// Document-level statistics.
#[derive(Debug, Clone)]
pub struct DocumentStats {
    /// Document length in characters.
    pub char_count: usize,
    /// Document length in words.
    pub word_count: usize,
    /// Total entity mention count.
    pub mention_count: usize,
    /// Unique entity count (by normalized text).
    pub unique_entity_count: usize,
    /// Entity density (mentions per 1000 chars).
    pub entity_density: f64,
    /// Entity type distribution.
    pub type_distribution: HashMap<String, usize>,
}

// =============================================================================
// Main Extractor
// =============================================================================

/// Entity feature extractor.
///
/// Extracts comprehensive features from entities at multiple levels:
/// - Mention context (surrounding text, position)
/// - Chain aggregates (for coreference chains)
/// - Co-occurrence patterns (which entities appear together)
#[derive(Debug, Clone)]
pub struct EntityFeatureExtractor {
    config: ExtractorConfig,
}

impl Default for EntityFeatureExtractor {
    fn default() -> Self {
        Self::new(ExtractorConfig::default())
    }
}

impl EntityFeatureExtractor {
    /// Create a new extractor with the given configuration.
    pub fn new(config: ExtractorConfig) -> Self {
        Self { config }
    }

    /// Extract all features from text and entities.
    pub fn extract_all(&self, text: &str, entities: &[Entity]) -> DocumentFeatures {
        let text_len = text.chars().count();

        // 1. Extract mention contexts
        let mention_contexts: Vec<MentionContext> = entities
            .iter()
            .map(|e| MentionContext::extract(text, e, &self.config))
            .collect();

        // 2. Group entities by normalized key
        let groups = self.group_entities(entities);

        // 3. Compute chain features
        let chain_features: HashMap<String, ChainFeatures> = groups
            .iter()
            .map(|(key, mentions)| {
                let refs: Vec<&Entity> = mentions.to_vec();
                (key.clone(), ChainFeatures::from_mentions(&refs, text_len))
            })
            .collect();

        // 4. Compute co-occurrence features
        let cooccurrence = self.extract_cooccurrence(entities);

        // 5. Document stats
        let word_count = text.split_whitespace().count();
        let unique_entity_count = groups.len();
        let entity_density = if text_len > 0 {
            (entities.len() as f64 / text_len as f64) * 1000.0
        } else {
            0.0
        };

        let mut type_distribution: HashMap<String, usize> = HashMap::new();
        for e in entities {
            *type_distribution
                .entry(e.entity_type.as_label().to_string())
                .or_insert(0) += 1;
        }

        let document_stats = DocumentStats {
            char_count: text_len,
            word_count,
            mention_count: entities.len(),
            unique_entity_count,
            entity_density,
            type_distribution,
        };

        DocumentFeatures {
            mention_contexts,
            chain_features,
            cooccurrence,
            document_stats,
        }
    }

    /// Extract only mention contexts (lightweight).
    pub fn extract_mentions(&self, text: &str, entities: &[Entity]) -> Vec<MentionContext> {
        entities
            .iter()
            .map(|e| MentionContext::extract(text, e, &self.config))
            .collect()
    }

    /// Extract only chain features (requires grouping).
    pub fn extract_chains(
        &self,
        text: &str,
        entities: &[Entity],
    ) -> HashMap<String, ChainFeatures> {
        let text_len = text.chars().count();
        let groups = self.group_entities(entities);

        groups
            .iter()
            .map(|(key, mentions)| {
                let refs: Vec<&Entity> = mentions.to_vec();
                (key.clone(), ChainFeatures::from_mentions(&refs, text_len))
            })
            .collect()
    }

    /// Extract only co-occurrence features.
    pub fn extract_cooccurrence(
        &self,
        entities: &[Entity],
    ) -> HashMap<String, CooccurrenceFeatures> {
        let mut result: HashMap<String, CooccurrenceFeatures> = HashMap::new();

        // Initialize features for each unique entity
        for e in entities {
            let key = self.normalize_key(&e.text);
            result
                .entry(key.clone())
                .or_insert_with(|| CooccurrenceFeatures::new(key));
        }

        // Find co-occurrences
        for (i, e1) in entities.iter().enumerate() {
            let key1 = self.normalize_key(&e1.text);

            for e2 in entities.iter().skip(i + 1) {
                let key2 = self.normalize_key(&e2.text);

                // Skip self-cooccurrence
                if key1 == key2 {
                    continue;
                }

                // Check if within cooccurrence window
                let distance = if e1.end <= e2.start {
                    e2.start - e1.end
                } else if e2.end <= e1.start {
                    e1.start.saturating_sub(e2.end)
                } else {
                    0 // overlapping
                };

                if distance <= self.config.cooccurrence_window {
                    if let Some(f) = result.get_mut(&key1) {
                        f.add_cooccurrence(&key2, Some(e2.entity_type.as_label()));
                    }
                    if let Some(f) = result.get_mut(&key2) {
                        f.add_cooccurrence(&key1, Some(e1.entity_type.as_label()));
                    }
                }
            }
        }

        // Finalize all
        for f in result.values_mut() {
            f.finalize();
        }

        result
    }

    /// Group entities by normalized key.
    fn group_entities<'a>(&self, entities: &'a [Entity]) -> HashMap<String, Vec<&'a Entity>> {
        let mut groups: HashMap<String, Vec<&'a Entity>> = HashMap::new();
        for e in entities {
            let key = self.normalize_key(&e.text);
            groups.entry(key).or_default().push(e);
        }
        groups
    }

    /// Normalize entity text to a key for grouping.
    fn normalize_key(&self, text: &str) -> String {
        if self.config.normalize_text {
            text.to_lowercase().trim().to_string()
        } else {
            text.trim().to_string()
        }
    }
}

// =============================================================================
// Pairwise Features (for coreference training)
// =============================================================================

/// Pairwise features between two mentions (for coreference scoring).
#[derive(Debug, Clone)]
pub struct PairwiseFeatures {
    /// Distance in characters between mentions.
    pub char_distance: usize,
    /// Distance in mentions (number of mentions between).
    pub mention_distance: usize,
    /// Do the surface forms match exactly?
    pub exact_match: bool,
    /// Do the surface forms match after lowercasing?
    pub case_insensitive_match: bool,
    /// String similarity (Jaccard on words).
    pub string_similarity: f64,
    /// Do the entity types match?
    pub type_match: bool,
    /// Mention type of first mention.
    pub mention_type_a: MentionType,
    /// Mention type of second mention.
    pub mention_type_b: MentionType,
    /// Is the second mention a pronoun referring back?
    pub is_pronominal_anaphora: bool,
}

impl PairwiseFeatures {
    /// Compute pairwise features between two mentions.
    pub fn compute(a: &Entity, b: &Entity, mention_distance: usize) -> Self {
        let char_distance = if a.end <= b.start {
            b.start - a.end
        } else if b.end <= a.start {
            a.start.saturating_sub(b.end)
        } else {
            0
        };

        let exact_match = a.text == b.text;
        let case_insensitive_match = a.text.to_lowercase() == b.text.to_lowercase();

        // Jaccard similarity on words
        let words_a: HashSet<&str> = a.text.split_whitespace().collect();
        let words_b: HashSet<&str> = b.text.split_whitespace().collect();
        let intersection = words_a.intersection(&words_b).count();
        let union = words_a.union(&words_b).count();
        let string_similarity = if union > 0 {
            intersection as f64 / union as f64
        } else {
            0.0
        };

        let type_match = a.entity_type == b.entity_type;

        let mention_type_a = MentionType::classify(&a.text);
        let mention_type_b = MentionType::classify(&b.text);

        // Pronominal anaphora: second mention is pronoun, first is not
        let is_pronominal_anaphora = mention_type_b == MentionType::Pronominal
            && mention_type_a != MentionType::Pronominal
            && b.start > a.start;

        Self {
            char_distance,
            mention_distance,
            exact_match,
            case_insensitive_match,
            string_similarity,
            type_match,
            mention_type_a,
            mention_type_b,
            is_pronominal_anaphora,
        }
    }

    /// Compute features for all mention pairs in a document.
    pub fn compute_all_pairs(entities: &[Entity]) -> Vec<(usize, usize, PairwiseFeatures)> {
        let mut pairs = Vec::new();
        for (i, a) in entities.iter().enumerate() {
            for (j, b) in entities.iter().enumerate().skip(i + 1) {
                let mention_distance = j - i;
                let features = Self::compute(a, b, mention_distance);
                pairs.push((i, j, features));
            }
        }
        pairs
    }
}

// =============================================================================
// Embedding Aggregation Utilities
// =============================================================================

/// Aggregate embeddings from multiple mentions.
pub fn aggregate_embeddings(
    embeddings: &[Vec<f32>],
    method: AggregationMethod,
) -> Option<Vec<f32>> {
    if embeddings.is_empty() {
        return None;
    }

    let dim = embeddings[0].len();
    if dim == 0 {
        return None;
    }

    // Verify all same dimension
    if !embeddings.iter().all(|e| e.len() == dim) {
        return None;
    }

    match method {
        AggregationMethod::Mean => {
            let mut result = vec![0.0f32; dim];
            for emb in embeddings {
                for (i, &v) in emb.iter().enumerate() {
                    result[i] += v;
                }
            }
            let n = embeddings.len() as f32;
            for v in &mut result {
                *v /= n;
            }
            Some(result)
        }
        AggregationMethod::Max => {
            let mut result = vec![f32::NEG_INFINITY; dim];
            for emb in embeddings {
                for (i, &v) in emb.iter().enumerate() {
                    result[i] = result[i].max(v);
                }
            }
            Some(result)
        }
        AggregationMethod::First => embeddings.first().cloned(),
        AggregationMethod::WeightedMean { ref weights } => {
            if weights.len() != embeddings.len() {
                return None;
            }
            let total_weight: f32 = weights.iter().sum();
            if total_weight == 0.0 {
                return None;
            }
            let mut result = vec![0.0f32; dim];
            for (emb, &w) in embeddings.iter().zip(weights.iter()) {
                for (i, &v) in emb.iter().enumerate() {
                    result[i] += v * w;
                }
            }
            for v in &mut result {
                *v /= total_weight;
            }
            Some(result)
        }
    }
}

/// Method for aggregating multiple embeddings into one.
#[derive(Debug, Clone, Default)]
pub enum AggregationMethod {
    /// Mean of all embeddings.
    #[default]
    Mean,
    /// Element-wise max.
    Max,
    /// Just use the first embedding.
    First,
    /// Weighted mean with custom weights per embedding dimension.
    WeightedMean {
        /// Weights for each embedding dimension.
        weights: Vec<f32>,
    },
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EntityType;

    fn sample_entities() -> Vec<Entity> {
        vec![
            Entity::new("Barack Obama", EntityType::Person, 0, 12, 0.95),
            Entity::new("Angela Merkel", EntityType::Person, 17, 30, 0.92),
            Entity::new("Berlin", EntityType::Location, 34, 40, 0.88),
            Entity::new("He", EntityType::Person, 42, 44, 0.85),
            Entity::new("Obama", EntityType::Person, 60, 65, 0.90),
        ]
    }

    #[test]
    fn test_mention_type_classification() {
        assert_eq!(MentionType::classify("he"), MentionType::Pronominal);
        assert_eq!(MentionType::classify("She"), MentionType::Pronominal);
        assert_eq!(MentionType::classify("Barack Obama"), MentionType::Proper);
        assert_eq!(MentionType::classify("the president"), MentionType::Nominal);
        assert_eq!(MentionType::classify("Apple Inc."), MentionType::Proper);
    }

    #[test]
    fn test_mention_context_extraction() {
        let text = "In Paris, Barack Obama met Angela Merkel. He discussed policy.";
        let entity = Entity::new("Barack Obama", EntityType::Person, 10, 22, 0.95);

        let ctx = MentionContext::extract(text, &entity, &ExtractorConfig::default());

        assert_eq!(ctx.entity.text, "Barack Obama");
        assert!(ctx.left_context.contains("Paris"));
        assert!(ctx.right_context.contains("met"));
        assert!(ctx.relative_position < 0.5); // Early in document
        assert!(ctx.is_capitalized);
    }

    #[test]
    fn test_chain_features() {
        let entities = sample_entities();
        let text_len = 100;

        // Group Obama mentions
        let obama_mentions: Vec<&Entity> = entities
            .iter()
            .filter(|e| e.text.to_lowercase().contains("obama") || e.text.to_lowercase() == "he")
            .collect();

        let features = ChainFeatures::from_mentions(&obama_mentions, text_len);

        assert_eq!(features.chain_length, 3); // Barack Obama, He, Obama
        assert!(features.variations.contains(&"Barack Obama".to_string()));
        assert!(features.pronominal_count >= 1); // "He"
        assert!(features.named_count >= 1); // "Barack Obama"

        // Test variation_count()
        // Should have: "Barack Obama", "He", "Obama"
        assert_eq!(features.variation_count(), 3);
        assert!(!features.is_singleton()); // Multiple mentions
    }

    #[test]
    fn test_cooccurrence_extraction() {
        let _text = "Barack Obama met Angela Merkel in Berlin. He discussed policy.";
        let entities = sample_entities();

        let extractor = EntityFeatureExtractor::default();
        let cooc = extractor.extract_cooccurrence(&entities);

        let obama_cooc = cooc.get("barack obama").unwrap();
        assert!(obama_cooc
            .cooccurring_entities
            .contains(&"angela merkel".to_string()));
        assert!(obama_cooc
            .cooccurring_entities
            .contains(&"berlin".to_string()));
    }

    #[test]
    fn test_pairwise_features() {
        let a = Entity::new("Barack Obama", EntityType::Person, 0, 12, 0.95);
        let b = Entity::new("Obama", EntityType::Person, 50, 55, 0.90);
        let c = Entity::new("He", EntityType::Person, 60, 62, 0.85);

        let ab = PairwiseFeatures::compute(&a, &b, 1);
        assert!(ab.case_insensitive_match || ab.string_similarity > 0.0);
        assert!(ab.type_match);

        let ac = PairwiseFeatures::compute(&a, &c, 2);
        assert!(ac.is_pronominal_anaphora);
    }

    #[test]
    fn test_full_extraction() {
        let text = "Barack Obama met Angela Merkel in Berlin. He discussed policy with her.";
        let entities = sample_entities();

        let extractor = EntityFeatureExtractor::default();
        let features = extractor.extract_all(text, &entities);

        assert_eq!(features.mention_contexts.len(), entities.len());
        assert!(!features.chain_features.is_empty());
        assert!(!features.cooccurrence.is_empty());
        assert!(features.document_stats.mention_count == entities.len());
    }

    #[test]
    fn test_aggregate_embeddings() {
        let emb1 = vec![1.0, 2.0, 3.0];
        let emb2 = vec![2.0, 4.0, 6.0];
        let embeddings = vec![emb1, emb2];

        let mean = aggregate_embeddings(&embeddings, AggregationMethod::Mean).unwrap();
        assert_eq!(mean, vec![1.5, 3.0, 4.5]);

        let max = aggregate_embeddings(&embeddings, AggregationMethod::Max).unwrap();
        assert_eq!(max, vec![2.0, 4.0, 6.0]);
    }
}
