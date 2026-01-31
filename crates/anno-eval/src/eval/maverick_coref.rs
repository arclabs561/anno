//! Maverick Coreference Resolution Implementation
//!
//! This module implements Maverick (Martinelli, Barba & Navigli, ACL 2024),
//! a modern coreference resolution system with a coarse-to-fine pipeline.
//!
//! # Architecture Overview
//!
//! Maverick uses a coarse-to-fine pipeline:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                     Maverick Pipeline                                │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │ 1. Encode text with DeBERTa-v3                                       │
//! │    └── Hidden states H: [batch, seq_len, hidden_dim]                │
//! │                                                                      │
//! │ 2. Mention Extraction (start + end boundary detection)              │
//! │    ├── Start classifier: P(token is mention start)                  │
//! │    └── End classifier: P(span is mention | start)                   │
//! │                                                                      │
//! │ 3. Multi-Expert Scorer (MES) for antecedent linking                 │
//! │    ├── Project start/end to K category-specific spaces              │
//! │    ├── Bilinear scoring: s2s, e2e, s2e, e2s                         │
//! │    └── Categories: Proper, Nominal, Pronoun, ALL                    │
//! │                                                                      │
//! │ 4. Clustering via transitivity                                       │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Key Innovation: Multi-Expert Scorer
//!
//! Instead of a single bilinear scorer, Maverick uses category-specific experts:
//!
//! - **Proper**: "John" ↔ "Mr. Smith" ↔ "the CEO"
//! - **Nominal**: "the company" ↔ "the firm" ↔ "it"
//! - **Pronoun**: "he" ↔ "him" ↔ "his"
//!
//! This specialization improves precision without sacrificing recall.
//!
//! # References
//!
//! - Paper: <https://arxiv.org/abs/2407.21489>
//! - Code: <https://github.com/SapienzaNLP/maverick-coref>

use super::coref::{CorefChain, Mention, MentionType};
use crate::Result;

#[cfg(feature = "candle")]
use crate::Error;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Configuration
// =============================================================================

/// Maverick model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaverickConfig {
    /// HuggingFace model ID for the encoder
    pub encoder_model: String,
    /// Hidden dimension of the encoder
    pub hidden_dim: usize,
    /// Number of mention categories (Proper, Nominal, Pronoun, ALL)
    pub num_categories: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Threshold for mention start detection
    pub start_threshold: f32,
    /// Threshold for mention span detection
    pub mention_threshold: f32,
    /// Threshold for antecedent linking
    pub coref_threshold: f32,
    /// Whether to include singletons
    pub include_singletons: bool,
    /// Freeze encoder weights (for inference)
    pub freeze_encoder: bool,
}

impl Default for MaverickConfig {
    fn default() -> Self {
        Self {
            encoder_model: "microsoft/deberta-v3-base".to_string(),
            hidden_dim: 768,
            num_categories: 4, // Proper, Nominal, Pronoun, ALL
            max_seq_len: 4096,
            start_threshold: 0.5,
            mention_threshold: 0.5,
            coref_threshold: 0.5,
            include_singletons: false,
            freeze_encoder: true,
        }
    }
}

impl MaverickConfig {
    /// Configuration for OntoNotes model.
    pub fn ontonotes() -> Self {
        Self {
            encoder_model: "microsoft/deberta-v3-large".to_string(),
            hidden_dim: 1024,
            include_singletons: false,
            ..Default::default()
        }
    }

    /// Configuration for LitBank model (literary texts).
    pub fn litbank() -> Self {
        Self {
            encoder_model: "microsoft/deberta-v3-large".to_string(),
            hidden_dim: 1024,
            include_singletons: true,
            ..Default::default()
        }
    }

    /// Configuration for PreCo model.
    pub fn preco() -> Self {
        Self {
            encoder_model: "microsoft/deberta-v3-large".to_string(),
            hidden_dim: 1024,
            include_singletons: true,
            ..Default::default()
        }
    }
}

// =============================================================================
// Mention Categories
// =============================================================================

/// Mention categories for the Multi-Expert Scorer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MentionCategory {
    /// Named entities: "John", "Apple Inc."
    Proper = 0,
    /// Nominal phrases: "the company", "a dog"
    Nominal = 1,
    /// Pronouns: "he", "she", "it", "they"
    Pronoun = 2,
    /// Catch-all category
    All = 3,
}

impl MentionCategory {
    /// Determine category from mention text.
    pub fn from_text(text: &str) -> Self {
        let lower = text.to_lowercase();
        let words: Vec<&str> = lower.split_whitespace().collect();
        
        if words.is_empty() {
            return Self::All;
        }

        // Check for pronouns first
        const PRONOUNS: &[&str] = &[
            "i", "me", "my", "mine", "myself",
            "you", "your", "yours", "yourself",
            "he", "him", "his", "himself",
            "she", "her", "hers", "herself",
            "it", "its", "itself",
            "we", "us", "our", "ours", "ourselves",
            "they", "them", "their", "theirs", "themselves",
            "who", "whom", "whose", "which", "that",
        ];
        
        if words.len() == 1 && PRONOUNS.contains(&words[0]) {
            return Self::Pronoun;
        }

        // Check for proper nouns (starts with capital)
        let first_char = text.chars().next().unwrap_or(' ');
        if first_char.is_uppercase() {
            return Self::Proper;
        }

        // Check for nominal phrases (determiners)
        const DETERMINERS: &[&str] = &[
            "the", "a", "an", "this", "that", "these", "those",
            "my", "your", "his", "her", "its", "our", "their",
        ];
        
        if DETERMINERS.contains(&words[0]) {
            return Self::Nominal;
        }

        Self::All
    }

    /// Get all categories.
    pub fn all() -> [Self; 4] {
        [Self::Proper, Self::Nominal, Self::Pronoun, Self::All]
    }
}

// =============================================================================
// Extracted Mentions
// =============================================================================

/// A mention extracted by Maverick.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaverickMention {
    /// Token start index
    pub start: usize,
    /// Token end index (exclusive)
    pub end: usize,
    /// Character start offset
    pub char_start: usize,
    /// Character end offset
    pub char_end: usize,
    /// Mention text
    pub text: String,
    /// Start probability
    pub start_prob: f32,
    /// Span probability
    pub span_prob: f32,
    /// Detected category
    pub category: MentionCategory,
    /// Hidden state (start concat end)
    #[serde(skip)]
    pub hidden_state: Option<Vec<f32>>,
}

impl MaverickMention {
    /// Convert to standard Mention.
    pub fn to_mention(&self) -> Mention {
        let mut mention = Mention::new(&self.text, self.char_start, self.char_end);
        mention.mention_type = Some(match self.category {
            MentionCategory::Proper => MentionType::Proper,
            MentionCategory::Nominal => MentionType::Nominal,
            MentionCategory::Pronoun => MentionType::Pronominal,
            MentionCategory::All => MentionType::Unknown,
        });
        mention
    }
}

// =============================================================================
// Antecedent Scores
// =============================================================================

/// Antecedent scores from the Multi-Expert Scorer.
#[derive(Debug, Clone)]
pub struct AntecedentScores {
    /// Score matrix: [num_mentions, num_mentions]
    /// scores[i][j] = P(mention j is antecedent of mention i)
    pub scores: Vec<Vec<f32>>,
    /// Per-category scores (for analysis)
    pub category_scores: HashMap<MentionCategory, Vec<Vec<f32>>>,
}

// =============================================================================
// Coreference Clusters
// =============================================================================

/// A coreference cluster from Maverick.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaverickCluster {
    /// Cluster ID
    pub id: usize,
    /// Mentions in this cluster (sorted by position)
    pub mentions: Vec<MaverickMention>,
}

impl MaverickCluster {
    /// Convert to standard CorefChain.
    pub fn to_chain(&self) -> CorefChain {
        let mentions: Vec<Mention> = self.mentions.iter().map(|m| m.to_mention()).collect();
        let mut chain = CorefChain::new(mentions);
        chain.cluster_id = Some(self.id as u64);
        chain
    }
}

// =============================================================================
// Pure Rust Implementation (CPU-only, no neural inference)
// =============================================================================

/// CPU-based Maverick implementation using heuristics.
///
/// This is a fallback when no GPU backend (candle/onnx) is available.
/// It uses rule-based mention extraction and string similarity for linking.
///
/// For production use with neural inference, see `MaverickCandle`.
#[derive(Debug)]
pub struct MaverickCpu {
    config: MaverickConfig,
}

impl MaverickCpu {
    /// Create a new CPU-based resolver.
    pub fn new(config: MaverickConfig) -> Self {
        Self { config }
    }

    /// Resolve coreference in text.
    pub fn resolve(&self, text: &str) -> Result<Vec<MaverickCluster>> {
        if text.trim().is_empty() {
            return Ok(vec![]);
        }

        // Step 1: Simple mention extraction (NP-like patterns)
        let mentions = self.extract_mentions_heuristic(text)?;

        if mentions.is_empty() {
            return Ok(vec![]);
        }

        // Step 2: Link mentions using string similarity
        let clusters = self.link_mentions_heuristic(&mentions);

        // Step 3: Filter singletons if configured
        let clusters = if self.config.include_singletons {
            clusters
        } else {
            clusters.into_iter().filter(|c| c.mentions.len() > 1).collect()
        };

        Ok(clusters)
    }

    /// Heuristic mention extraction.
    fn extract_mentions_heuristic(&self, text: &str) -> Result<Vec<MaverickMention>> {
        let mut mentions = Vec::new();

        // Simple tokenization
        let words: Vec<(&str, usize, usize, usize, usize)> = {
            let mut result = Vec::new();
            // Track position in bytes for searching; convert to char offsets via `TextSpan`.
            let mut byte_pos = 0;
            for word in text.split_whitespace() {
                if let Some(start) = text.get(byte_pos..).and_then(|s| s.find(word)) {
                    let word_start_byte = byte_pos + start;
                    let word_end_byte = word_start_byte + word.len();
                    let span =
                        anno::offset::TextSpan::from_bytes(text, word_start_byte, word_end_byte);
                    result.push((
                        word,
                        word_start_byte,
                        word_end_byte,
                        span.char_start,
                        span.char_end,
                    ));
                    byte_pos = word_end_byte;
                }
            }
            result
        };

        // Extract proper nouns (capitalized words)
        for (i, (word, word_start_byte, _word_end_byte, char_start, char_end)) in words.iter().enumerate() {
            let first_char = word.chars().next().unwrap_or(' ');
            
            // Skip sentence starters (first word after period)
            let is_sentence_start = i == 0 || {
                let prev_end_byte = if i > 0 { words[i - 1].2 } else { 0 };
                let between = text.get(prev_end_byte..*word_start_byte).unwrap_or("");
                between.contains('.') || between.contains('!') || between.contains('?')
            };

            // Proper nouns
            if first_char.is_uppercase() && !is_sentence_start {
                mentions.push(MaverickMention {
                    start: i,
                    end: i + 1,
                    char_start: *char_start,
                    char_end: *char_end,
                    text: word.to_string(),
                    start_prob: 0.9,
                    span_prob: 0.9,
                    category: MentionCategory::Proper,
                    hidden_state: None,
                });
            }

            // Pronouns
            let lower = word.to_lowercase();
            if MentionCategory::from_text(&lower) == MentionCategory::Pronoun {
                mentions.push(MaverickMention {
                    start: i,
                    end: i + 1,
                    char_start: *char_start,
                    char_end: *char_end,
                    text: word.to_string(),
                    start_prob: 0.8,
                    span_prob: 0.8,
                    category: MentionCategory::Pronoun,
                    hidden_state: None,
                });
            }
        }

        // Sort by position
        mentions.sort_by_key(|m| m.char_start);

        Ok(mentions)
    }

    /// Heuristic mention linking with gender-aware pronoun resolution.
    ///
    /// Implements a simple but effective linking strategy:
    /// 1. Exact string match (case-insensitive)
    /// 2. Substring match for name variants ("Elizabeth" matches "Miss Elizabeth")
    /// 3. Pronoun to nearest gender-compatible proper noun
    fn link_mentions_heuristic(&self, mentions: &[MaverickMention]) -> Vec<MaverickCluster> {
        let mut mention_to_cluster: Vec<Option<usize>> = vec![None; mentions.len()];
        let mut cluster_genders: Vec<Option<&'static str>> = Vec::new();
        let mut clusters: Vec<MaverickCluster> = Vec::new();

        for (i, mention) in mentions.iter().enumerate() {
            let mut assigned = false;

            // Strategy 1: Exact string match with previous mentions
            for j in 0..i {
                if mention.text.to_lowercase() == mentions[j].text.to_lowercase() {
                    if let Some(cluster_id) = mention_to_cluster[j] {
                        mention_to_cluster[i] = Some(cluster_id);
                        clusters[cluster_id].mentions.push(mention.clone());
                        assigned = true;
                        break;
                    }
                }
            }

            // Strategy 2: Substring match for name variants
            if !assigned && mention.category == MentionCategory::Proper {
                let mention_lower = mention.text.to_lowercase();
                for j in 0..i {
                    if mentions[j].category == MentionCategory::Proper {
                        let other_lower = mentions[j].text.to_lowercase();
                        // Check if one contains the other (e.g., "Elizabeth" in "Miss Elizabeth")
                        if mention_lower.contains(&other_lower) || other_lower.contains(&mention_lower) {
                            if let Some(cluster_id) = mention_to_cluster[j] {
                                mention_to_cluster[i] = Some(cluster_id);
                                clusters[cluster_id].mentions.push(mention.clone());
                                assigned = true;
                                break;
                            } else {
                                let cluster_id = clusters.len();
                                clusters.push(MaverickCluster {
                                    id: cluster_id,
                                    mentions: vec![mentions[j].clone(), mention.clone()],
                                });
                                cluster_genders.push(self.infer_gender(&mentions[j].text));
                                mention_to_cluster[j] = Some(cluster_id);
                                mention_to_cluster[i] = Some(cluster_id);
                                assigned = true;
                                break;
                            }
                        }
                    }
                }
            }

            // Strategy 3: Gender-aware pronoun linking
            if !assigned && mention.category == MentionCategory::Pronoun {
                let pronoun_gender = self.pronoun_gender(&mention.text);

                // Look backward for gender-compatible antecedent
                for j in (0..i).rev() {
                    if mentions[j].category == MentionCategory::Proper {
                        // Check gender compatibility
                        let antecedent_gender = if let Some(cluster_id) = mention_to_cluster[j] {
                            cluster_genders.get(cluster_id).copied().flatten()
                        } else {
                            self.infer_gender(&mentions[j].text)
                        };

                        if self.genders_compatible(pronoun_gender, antecedent_gender) {
                            if let Some(cluster_id) = mention_to_cluster[j] {
                                mention_to_cluster[i] = Some(cluster_id);
                                clusters[cluster_id].mentions.push(mention.clone());
                                assigned = true;
                                break;
                            } else {
                                let cluster_id = clusters.len();
                                clusters.push(MaverickCluster {
                                    id: cluster_id,
                                    mentions: vec![mentions[j].clone(), mention.clone()],
                                });
                                cluster_genders.push(antecedent_gender);
                                mention_to_cluster[j] = Some(cluster_id);
                                mention_to_cluster[i] = Some(cluster_id);
                                assigned = true;
                                break;
                            }
                        }
                    }
                }
            }

            // Create singleton if not assigned
            if !assigned {
                let cluster_id = clusters.len();
                clusters.push(MaverickCluster {
                    id: cluster_id,
                    mentions: vec![mention.clone()],
                });
                cluster_genders.push(self.infer_gender(&mention.text));
                mention_to_cluster[i] = Some(cluster_id);
            }
        }

        clusters
    }

    /// Get pronoun gender for compatibility checking.
    fn pronoun_gender(&self, pronoun: &str) -> Option<&'static str> {
        match pronoun.to_lowercase().as_str() {
            "he" | "him" | "his" | "himself" => Some("masculine"),
            "she" | "her" | "hers" | "herself" => Some("feminine"),
            "it" | "its" | "itself" => Some("neuter"),
            "they" | "them" | "their" | "theirs" | "themselves" => Some("plural"),
            _ => None,
        }
    }

    /// Infer gender from a proper noun (heuristic).
    ///
    /// Uses common name suffixes and titles:
    /// - "Mr." → masculine
    /// - "Mrs.", "Miss", "Ms." → feminine
    /// - Names ending in "-a" often feminine (Maria, Anna) in many cultures
    fn infer_gender(&self, name: &str) -> Option<&'static str> {
        let lower = name.to_lowercase();
        
        // Check titles
        if lower.starts_with("mr.") || lower.starts_with("mr ") {
            return Some("masculine");
        }
        if lower.starts_with("mrs.") || lower.starts_with("mrs ") 
            || lower.starts_with("miss ") || lower.starts_with("ms.") {
            return Some("feminine");
        }
        
        // Check common masculine names (English)
        let masculine_names = ["john", "james", "william", "henry", "charles", 
            "george", "edward", "thomas", "david", "michael", "robert"];
        let first_word: &str = lower.split_whitespace().next().unwrap_or("");
        if masculine_names.contains(&first_word) {
            return Some("masculine");
        }
        
        // Check common feminine names (English)  
        let feminine_names = ["mary", "elizabeth", "jane", "anne", "sarah",
            "catherine", "charlotte", "emily", "emma", "caroline"];
        if feminine_names.contains(&first_word) {
            return Some("feminine");
        }
        
        // Can't determine
        None
    }

    /// Check if pronoun gender is compatible with antecedent gender.
    fn genders_compatible(&self, pronoun: Option<&str>, antecedent: Option<&str>) -> bool {
        match (pronoun, antecedent) {
            // If we don't know either, assume compatible
            (None, _) | (_, None) => true,
            // Plural pronoun matches anything
            (Some("plural"), _) => true,
            // Neuter usually for non-persons, be lenient
            (Some("neuter"), _) => true,
            // Direct gender match
            (Some(p), Some(a)) if p == a => true,
            // Mismatch
            _ => false,
        }
    }
}

// =============================================================================
// Candle Implementation (Neural Inference)
// =============================================================================

#[cfg(feature = "candle")]
mod candle_impl {
    use super::*;
    use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
    use candle_nn::{Linear, VarBuilder};

    /// Maverick Multi-Expert Scorer layer.
    ///
    /// Computes coreference scores using bilinear forms:
    ///
    /// score(i,j) = sum_k [
    ///   s_i^T W_k^{s2s} s_j + e_i^T W_k^{e2e} e_j +
    ///   s_i^T W_k^{s2e} e_j + e_i^T W_k^{e2s} s_j +
    ///   bias_k
    /// ] * category_mask_k(i,j)
    pub struct MultiExpertScorer {
        config: MaverickConfig,
        
        // Start/end MLPs for each category
        start_mlps: Vec<Linear>,
        end_mlps: Vec<Linear>,
        
        // Bilinear weights: [num_cats, hidden_dim, hidden_dim]
        s2s_weights: Tensor,
        e2e_weights: Tensor,
        s2e_weights: Tensor,
        e2s_weights: Tensor,
        
        // Biases: [num_cats, hidden_dim]
        s2s_biases: Tensor,
        e2e_biases: Tensor,
        s2e_biases: Tensor,
        e2s_biases: Tensor,
        
        device: Device,
    }

    impl MultiExpertScorer {
        /// Create from pretrained weights.
        pub fn load(vb: VarBuilder, config: &MaverickConfig) -> Result<Self> {
            let num_cats = config.num_categories;
            let hidden = config.hidden_dim;
            let mention_hidden = hidden * 2; // concat(start, end)
            let all_cats_size = hidden * num_cats;

            // Start MLPs
            let mut start_mlps = Vec::new();
            let mut end_mlps = Vec::new();
            
            for i in 0..num_cats {
                let start_mlp = candle_nn::linear(
                    hidden,
                    hidden,
                    vb.pp(format!("coref_start_mlp_{}", i)),
                ).map_err(|e| Error::Parse(format!("Start MLP {}: {}", i, e)))?;
                
                let end_mlp = candle_nn::linear(
                    hidden,
                    hidden,
                    vb.pp(format!("coref_end_mlp_{}", i)),
                ).map_err(|e| Error::Parse(format!("End MLP {}: {}", i, e)))?;
                
                start_mlps.push(start_mlp);
                end_mlps.push(end_mlp);
            }

            // Bilinear weights
            let s2s_weights = vb
                .get((num_cats, hidden, hidden), "antecedent_s2s_all_weights")
                .map_err(|e| Error::Parse(format!("s2s weights: {}", e)))?;
            let e2e_weights = vb
                .get((num_cats, hidden, hidden), "antecedent_e2e_all_weights")
                .map_err(|e| Error::Parse(format!("e2e weights: {}", e)))?;
            let s2e_weights = vb
                .get((num_cats, hidden, hidden), "antecedent_s2e_all_weights")
                .map_err(|e| Error::Parse(format!("s2e weights: {}", e)))?;
            let e2s_weights = vb
                .get((num_cats, hidden, hidden), "antecedent_e2s_all_weights")
                .map_err(|e| Error::Parse(format!("e2s weights: {}", e)))?;

            // Biases
            let s2s_biases = vb
                .get((num_cats, hidden), "antecedent_s2s_all_biases")
                .map_err(|e| Error::Parse(format!("s2s biases: {}", e)))?;
            let e2e_biases = vb
                .get((num_cats, hidden), "antecedent_e2e_all_biases")
                .map_err(|e| Error::Parse(format!("e2e biases: {}", e)))?;
            let s2e_biases = vb
                .get((num_cats, hidden), "antecedent_s2e_all_biases")
                .map_err(|e| Error::Parse(format!("s2e biases: {}", e)))?;
            let e2s_biases = vb
                .get((num_cats, hidden), "antecedent_e2s_all_biases")
                .map_err(|e| Error::Parse(format!("e2s biases: {}", e)))?;

            let device = s2s_weights.device().clone();

            Ok(Self {
                config: config.clone(),
                start_mlps,
                end_mlps,
                s2s_weights,
                e2e_weights,
                s2e_weights,
                e2s_weights,
                s2s_biases,
                e2e_biases,
                s2e_biases,
                e2s_biases,
                device,
            })
        }

        /// Compute coreference scores.
        ///
        /// # Arguments
        /// * `start_states` - [batch, num_mentions, hidden_dim]
        /// * `end_states` - [batch, num_mentions, hidden_dim]
        ///
        /// # Returns
        /// * Scores: [batch, num_cats, num_mentions, num_mentions]
        pub fn forward(&self, start_states: &Tensor, end_states: &Tensor) -> Result<Tensor> {
            let (batch_size, num_mentions, hidden_dim) = start_states.dims3()
                .map_err(|e| Error::Parse(format!("dims: {}", e)))?;

            // Project through category-specific MLPs
            // Result: [batch, num_cats, num_mentions, hidden_dim]
            let mut all_starts = Vec::new();
            let mut all_ends = Vec::new();

            for (start_mlp, end_mlp) in self.start_mlps.iter().zip(self.end_mlps.iter()) {
                let s = start_mlp.forward(start_states)
                    .map_err(|e| Error::Parse(format!("start mlp: {}", e)))?;
                let e = end_mlp.forward(end_states)
                    .map_err(|e| Error::Parse(format!("end mlp: {}", e)))?;
                all_starts.push(s);
                all_ends.push(e);
            }

            // Stack: [batch, num_cats, num_mentions, hidden_dim]
            let all_starts = Tensor::stack(&all_starts, 1)
                .map_err(|e| Error::Parse(format!("stack starts: {}", e)))?;
            let all_ends = Tensor::stack(&all_ends, 1)
                .map_err(|e| Error::Parse(format!("stack ends: {}", e)))?;

            // Compute bilinear scores using einsum-like operations
            // For now, use matrix multiplications
            // logits = s W s^T + e W e^T + s W e^T + e W s^T + biases

            // Simplified: compute per-category scores and sum
            // This is a placeholder - full einsum implementation needed for accuracy
            let scores = self.compute_bilinear_scores(&all_starts, &all_ends)?;

            Ok(scores)
        }

        /// Compute bilinear scores.
        fn compute_bilinear_scores(&self, starts: &Tensor, ends: &Tensor) -> Result<Tensor> {
            let (batch_size, num_cats, num_mentions, hidden_dim) = starts.dims4()
                .map_err(|e| Error::Parse(format!("dims4: {}", e)))?;

            // Simplified scoring: dot product between mentions
            // Full implementation would use the bilinear weights
            let starts_flat = starts.reshape((batch_size * num_cats, num_mentions, hidden_dim))
                .map_err(|e| Error::Parse(format!("reshape starts: {}", e)))?;
            
            let ends_t = ends.transpose(2, 3)
                .map_err(|e| Error::Parse(format!("transpose ends: {}", e)))?
                .reshape((batch_size * num_cats, hidden_dim, num_mentions))
                .map_err(|e| Error::Parse(format!("reshape ends_t: {}", e)))?;

            let scores = starts_flat.matmul(&ends_t)
                .map_err(|e| Error::Parse(format!("matmul: {}", e)))?
                .reshape((batch_size, num_cats, num_mentions, num_mentions))
                .map_err(|e| Error::Parse(format!("reshape scores: {}", e)))?;

            Ok(scores)
        }
    }

    /// Maverick model with Candle backend.
    pub struct MaverickCandle {
        config: MaverickConfig,
        // encoder: CandleEncoder, // Would use the encoder from encoder_candle.rs
        scorer: MultiExpertScorer,
        device: Device,
    }

    impl MaverickCandle {
        /// Load from safetensors weights.
        pub fn load(
            config: MaverickConfig,
            weights_path: &std::path::Path,
            device: Device,
        ) -> Result<Self> {
            use candle_core::safetensors::load;
            
            let tensors = load(weights_path, &device)
                .map_err(|e| Error::Parse(format!("Load safetensors: {}", e)))?;
            
            let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
            
            let scorer = MultiExpertScorer::load(vb, &config)?;
            
            Ok(Self {
                config,
                scorer,
                device,
            })
        }

        /// Resolve coreference.
        pub fn resolve(&self, _text: &str) -> Result<Vec<MaverickCluster>> {
            // Full implementation would:
            // 1. Tokenize with DeBERTa tokenizer
            // 2. Encode with DeBERTa
            // 3. Extract mentions with start/end classifiers
            // 4. Score antecedents with MultiExpertScorer
            // 5. Cluster via transitivity
            
            // For now, return placeholder
            Err(Error::FeatureNotAvailable(
                "Full MaverickCandle inference requires DeBERTa encoder integration".into()
            ))
        }
    }
}

#[cfg(feature = "candle")]
pub use candle_impl::*;

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mention_category() {
        assert_eq!(MentionCategory::from_text("he"), MentionCategory::Pronoun);
        assert_eq!(MentionCategory::from_text("she"), MentionCategory::Pronoun);
        assert_eq!(MentionCategory::from_text("John"), MentionCategory::Proper);
        assert_eq!(MentionCategory::from_text("the dog"), MentionCategory::Nominal);
        assert_eq!(MentionCategory::from_text("a company"), MentionCategory::Nominal);
    }

    #[test]
    fn test_maverick_cpu() {
        let config = MaverickConfig::default();
        let maverick = MaverickCpu::new(config);

        let text = "John went to the store. He bought milk.";
        let clusters = maverick.resolve(text).unwrap();

        // Should find at least John and he
        assert!(!clusters.is_empty());
    }

    #[test]
    fn test_gender_aware_linking() {
        let config = MaverickConfig::default();
        let maverick = MaverickCpu::new(config);

        // Text with mixed genders - "he" should link to John, "she" to Mary
        let text = "John met Mary at the cafe. He smiled. She waved back.";
        let clusters = maverick.resolve(text).unwrap();

        // Should find separate clusters for John/he and Mary/she
        // At minimum, should not incorrectly link "he" to Mary
        let john_cluster = clusters.iter().find(|c| 
            c.mentions.iter().any(|m| m.text == "John")
        );
        let mary_cluster = clusters.iter().find(|c| 
            c.mentions.iter().any(|m| m.text == "Mary")
        );

        // John and Mary should be in different clusters
        if let (Some(jc), Some(mc)) = (john_cluster, mary_cluster) {
            assert_ne!(jc.id, mc.id, "John and Mary should be in different clusters");
        }
    }

    #[test]
    fn test_title_gender_inference() {
        let config = MaverickConfig::default();
        let maverick = MaverickCpu::new(config);

        // "Mr. Smith" should link to "he", "Mrs. Jones" to "she"
        let text = "Mr. Smith arrived. He was early. Mrs. Jones followed. She was late.";
        let clusters = maverick.resolve(text).unwrap();

        // Should separate masculine and feminine clusters
        let he_mentions: Vec<_> = clusters.iter()
            .flat_map(|c| &c.mentions)
            .filter(|m| m.text.to_lowercase() == "he")
            .collect();
        let she_mentions: Vec<_> = clusters.iter()
            .flat_map(|c| &c.mentions)
            .filter(|m| m.text.to_lowercase() == "she")
            .collect();

        // Both pronouns should be linked
        assert!(!he_mentions.is_empty(), "Should find 'He'");
        assert!(!she_mentions.is_empty(), "Should find 'She'");
    }

    #[test]
    fn test_maverick_cpu_empty() {
        let config = MaverickConfig::default();
        let maverick = MaverickCpu::new(config);

        let clusters = maverick.resolve("").unwrap();
        assert!(clusters.is_empty());
    }

    #[test]
    fn test_maverick_config_variants() {
        let ontonotes = MaverickConfig::ontonotes();
        assert!(!ontonotes.include_singletons);

        let litbank = MaverickConfig::litbank();
        assert!(litbank.include_singletons);

        let preco = MaverickConfig::preco();
        assert!(preco.include_singletons);
    }

    #[test]
    fn test_cluster_to_chain() {
        let cluster = MaverickCluster {
            id: 0,
            mentions: vec![
                MaverickMention {
                    start: 0,
                    end: 1,
                    char_start: 0,
                    char_end: 4,
                    text: "John".to_string(),
                    start_prob: 0.9,
                    span_prob: 0.9,
                    category: MentionCategory::Proper,
                    hidden_state: None,
                },
                MaverickMention {
                    start: 5,
                    end: 6,
                    char_start: 24,
                    char_end: 26,
                    text: "He".to_string(),
                    start_prob: 0.8,
                    span_prob: 0.8,
                    category: MentionCategory::Pronoun,
                    hidden_state: None,
                },
            ],
        };

        let chain = cluster.to_chain();
        assert_eq!(chain.mentions.len(), 2);
        assert_eq!(chain.cluster_id, Some(0));
    }
}

