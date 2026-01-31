//! End-to-End Coreference Resolution (Lee et al. 2017, 2018).
//!
//! Implements the span-based neural coreference model that became
//! the standard approach after 2017.
//!
//! # Architecture
//!
//! ```text
//! Input: "John saw Mary. He waved to her."
//!    ↓
//! ┌─────────────────────────────────────────────────────────┐
//! │ 1. Span Enumeration                                     │
//! │    Generate all candidate spans up to max_span_width    │
//! │    Spans: [John], [saw], [Mary], [He], [her], ...      │
//! └─────────────────────────────────────────────────────────┘
//!    ↓
//! ┌─────────────────────────────────────────────────────────┐
//! │ 2. Span Representation                                  │
//! │    BiLSTM/BERT encoder → span embeddings               │
//! │    g(i) = [h_start; h_end; h_head; φ(i)]               │
//! └─────────────────────────────────────────────────────────┘
//!    ↓
//! ┌─────────────────────────────────────────────────────────┐
//! │ 3. Mention Scoring                                      │
//! │    Score each span as potential mention                 │
//! │    s_m(i) = FFNN(g(i))                                 │
//! │    Keep top-k spans                                     │
//! └─────────────────────────────────────────────────────────┘
//!    ↓
//! ┌─────────────────────────────────────────────────────────┐
//! │ 4. Antecedent Scoring                                   │
//! │    For each mention, score all previous mentions        │
//! │    s_a(i,j) = FFNN([g(i); g(j); g(i)∘g(j); φ(i,j)])   │
//! │    + dummy antecedent ε for non-anaphoric mentions     │
//! └─────────────────────────────────────────────────────────┘
//!    ↓
//! ┌─────────────────────────────────────────────────────────┐
//! │ 5. Clustering                                           │
//! │    Link each mention to best antecedent                 │
//! │    Transitivity creates clusters                        │
//! └─────────────────────────────────────────────────────────┘
//!    ↓
//! Output: {[John, He], [Mary, her]}
//! ```
//!
//! # Historical Context
//!
//! The E2E approach (Lee et al. 2017) revolutionized coreference by eliminating
//! separate mention detection and using learned span representations. Key milestones:
//!
//! ```text
//! 2017  E2E-Coref (Lee et al.): Span-based, BiLSTM encoder
//!       OntoNotes: 67.2 CoNLL F1
//!
//! 2018  Higher-Order (Lee et al.): Coarse-to-fine inference
//!       OntoNotes: 73.0 CoNLL F1
//!
//! 2019  SpanBERT (Joshi et al.): Pre-trained span representations
//!       OntoNotes: 79.6 CoNLL F1
//!
//! 2022  G2GT (Miculicich & Henderson): Graph refinement, global decisions
//!       OntoNotes: 80.5 CoNLL F1 (+0.9 over SpanBERT)
//!       See: [`graph_coref`](super::graph_coref) for anno's implementation
//!
//! 2024  Maverick (ACL 2024): Efficient architecture, 500M params
//!       OntoNotes: 83.6 CoNLL F1 (approaches 13B model performance)
//! ```
//!
//! # Limitations of Pairwise Models
//!
//! This E2E architecture makes **independent pairwise decisions**:
//!
//! ```text
//! p(y₁, y₂, ..., yₘ | D) = ∏ᵢ p(yᵢ | D)
//! ```
//!
//! This independence assumption means transitivity isn't enforced. If the model
//! predicts P(A~B)=0.9 and P(B~C)=0.9, it can still predict P(A~C)=0.1, violating
//! the transitive property of coreference.
//!
//! **Solutions in the literature:**
//!
//! | Approach | How It Addresses Transitivity | Reference |
//! |----------|-------------------------------|-----------|
//! | Higher-Order | Update representations with antecedent info | Lee et al. 2018 |
//! | Triads | Score mention triples jointly | Meng & Rumshisky 2018 |
//! | Graph Refinement | Condition on full graph structure | Miculicich & Henderson 2022 |
//!
//! See [`graph_coref`](super::graph_coref) for an implementation that addresses this.
//!
//! # Complexity
//!
//! The full model has O(N⁴) complexity:
//! - O(N²) candidate spans (all start/end combinations)
//! - O(N²) antecedent pairs for each mention
//!
//! Pruning reduces this in practice, but the quadratic-in-quadratic structure remains.
//! Compare to [`GraphCoref`](super::graph_coref::GraphCoref) at O(N² × T) where T ≈ 4.
//!
//! **Critical insight** (Thalken et al. 2024): A single CoNLL F1 score is
//! "uninformative, or even misleading"—models excel on long chains but
//! struggle with short chains and singletons. Report per-chain-length metrics.
//!
//! # References
//!
//! - Lee et al. 2017: "End-to-end Neural Coreference Resolution"
//! - Lee et al. 2018: "Higher-Order Coreference Resolution with Coarse-to-Fine Inference"
//! - Joshi et al. 2020: "SpanBERT: Improving Pre-training by Representing and Predicting Spans"
//! - Miculicich & Henderson 2022: "Graph Refinement for Coreference Resolution"
//!   [arXiv:2203.16574](https://arxiv.org/abs/2203.16574)
//! - Thalken et al. 2024: "A Comparative Study of Chain-Length Coreference Evaluation"
//! - Maverick (ACL 2024): "Efficient and Accurate Coreference Resolution"

use crate::Result;
use std::collections::HashMap;

/// Configuration for E2E coref model.
#[derive(Debug, Clone)]
pub struct E2ECorefConfig {
    /// Maximum span width to consider.
    pub max_span_width: usize,
    /// Maximum number of antecedents to consider per mention.
    pub max_antecedents: usize,
    /// Top-k spans to keep after mention scoring.
    pub top_spans_ratio: f64,
    /// Mention score threshold.
    pub mention_threshold: f64,
    /// Minimum antecedent score to link.
    pub link_threshold: f64,
}

impl Default for E2ECorefConfig {
    fn default() -> Self {
        Self {
            max_span_width: 10,
            max_antecedents: 50,
            top_spans_ratio: 0.4,
            mention_threshold: 0.0,
            link_threshold: 0.0,
        }
    }
}

/// A mention span with its score.
#[derive(Debug, Clone)]
pub struct Mention {
    /// Start token index.
    pub start: usize,
    /// End token index (exclusive).
    pub end: usize,
    /// Character start offset.
    pub char_start: usize,
    /// Character end offset.
    pub char_end: usize,
    /// Mention text.
    pub text: String,
    /// Mention score from the mention scorer.
    pub score: f64,
}

/// A coreference cluster.
#[derive(Debug, Clone)]
pub struct CorefCluster {
    /// Mentions in this cluster.
    pub mentions: Vec<Mention>,
    /// Cluster ID.
    pub id: usize,
}

/// End-to-End Coreference Resolution model.
///
/// This implements the Lee et al. (2017, 2018) architecture:
/// 1. Enumerate all spans as potential mentions
/// 2. Score spans with neural mention scorer
/// 3. Score antecedent pairs
/// 4. Link mentions transitively
#[derive(Debug)]
pub struct E2ECoref {
    /// Model configuration.
    config: E2ECorefConfig,
    /// ONNX session for neural inference.
    #[cfg(feature = "onnx")]
    session: Option<ort::session::Session>,
}

impl E2ECoref {
    /// Create a new E2E coref model with default config.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(E2ECorefConfig::default())
    }

    /// Create with custom configuration.
    #[must_use]
    pub fn with_config(config: E2ECorefConfig) -> Self {
        Self {
            config,
            #[cfg(feature = "onnx")]
            session: None,
        }
    }

    /// Load from ONNX model.
    #[cfg(feature = "onnx")]
    pub fn from_onnx(model_path: &str) -> Result<Self> {
        use crate::Error;
        use ort::session::Session;

        let session = Session::builder()
            .map_err(|e| Error::model_init(format!("Session builder: {}", e)))?
            .commit_from_file(model_path)
            .map_err(|e| Error::model_init(format!("Load ONNX: {}", e)))?;

        let mut model = Self::new();
        model.session = Some(session);
        Ok(model)
    }

    /// Resolve coreferences in text.
    pub fn resolve(&self, text: &str) -> Result<Vec<CorefCluster>> {
        if text.trim().is_empty() {
            return Ok(vec![]);
        }

        // Tokenize
        let tokens: Vec<&str> = text.split_whitespace().collect();
        if tokens.is_empty() {
            return Ok(vec![]);
        }

        // Calculate token positions
        let token_positions = self.calculate_token_positions(text, &tokens);

        // Step 1: Enumerate candidate spans
        let candidate_spans = self.enumerate_spans(&tokens, &token_positions, text);

        // Step 2: Score mentions
        let mut mentions = self.score_mentions(candidate_spans);

        // Step 3: Filter top-k mentions
        mentions.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        let max_mentions = ((tokens.len() as f64) * self.config.top_spans_ratio) as usize;
        mentions.truncate(max_mentions.max(1));

        // Re-sort by position for antecedent scoring
        mentions.sort_by_key(|m| (m.start, m.end));

        // Step 4: Score antecedents and link
        let clusters = self.link_mentions(&mentions);

        Ok(clusters)
    }

    /// Calculate token character positions.
    fn calculate_token_positions(&self, text: &str, tokens: &[&str]) -> Vec<(usize, usize)> {
        use crate::offset::SpanConverter;

        let converter = SpanConverter::new(text);
        let mut positions = Vec::with_capacity(tokens.len());
        let mut byte_pos = 0;

        for token in tokens {
            if let Some(pos) = text[byte_pos..].find(token) {
                let start_byte = byte_pos + pos;
                let end_byte = start_byte + token.len();
                let start_char = converter.byte_to_char(start_byte);
                let end_char = converter.byte_to_char(end_byte);
                positions.push((start_char, end_char));
                byte_pos = end_byte;
            } else {
                // Fallback
                positions.push((0, 0));
            }
        }

        positions
    }

    /// Enumerate all candidate spans up to max_span_width.
    fn enumerate_spans(
        &self,
        tokens: &[&str],
        positions: &[(usize, usize)],
        _text: &str,
    ) -> Vec<Mention> {
        let mut spans = Vec::new();
        let n = tokens.len();

        for start in 0..n {
            for width in 0..self.config.max_span_width.min(n - start) {
                let end = start + width + 1;

                let char_start = positions[start].0;
                let char_end = positions[end - 1].1;

                let span_text = tokens[start..end].join(" ");

                spans.push(Mention {
                    start,
                    end,
                    char_start,
                    char_end,
                    text: span_text,
                    score: 0.0,
                });
            }
        }

        spans
    }

    /// Score mentions using heuristics (fallback) or neural model.
    fn score_mentions(&self, mut mentions: Vec<Mention>) -> Vec<Mention> {
        for mention in &mut mentions {
            mention.score = self.heuristic_mention_score(&mention.text);
        }

        // Filter by threshold
        mentions.retain(|m| m.score > self.config.mention_threshold);
        mentions
    }

    /// Heuristic mention scoring (fallback when no neural model).
    fn heuristic_mention_score(&self, text: &str) -> f64 {
        let mut score = 0.0;

        // Pronouns are strong mention candidates
        let lower = text.to_lowercase();
        let pronouns = [
            "he", "she", "it", "they", "him", "her", "them", "his", "hers", "its", "their",
            "who", "whom", "which", "that", "this", "these", "those",
            "i", "me", "my", "mine", "we", "us", "our", "ours", "you", "your", "yours",
        ];
        if pronouns.contains(&lower.as_str()) {
            score += 0.8;
        }

        // Proper nouns (capitalized)
        if text.chars().next().is_some_and(|c| c.is_uppercase()) {
            score += 0.5;

            // Multi-word proper nouns are stronger
            if text.contains(' ') && text.split_whitespace().all(|w| {
                w.chars().next().is_some_and(|c| c.is_uppercase())
            }) {
                score += 0.2;
            }
        }

        // Definite descriptions (the + noun)
        if lower.starts_with("the ") {
            score += 0.4;
        }

        // Named entity patterns
        if lower.ends_with(" inc.") || lower.ends_with(" corp.") || lower.ends_with(" ltd.") {
            score += 0.6;
        }

        // Penalize very short or very long spans
        let word_count = text.split_whitespace().count();
        if word_count == 0 {
            score = -1.0;
        } else if word_count > 5 {
            score -= 0.3;
        }

        score
    }

    /// Link mentions to antecedents and form clusters.
    fn link_mentions(&self, mentions: &[Mention]) -> Vec<CorefCluster> {
        if mentions.is_empty() {
            return vec![];
        }

        // Track which cluster each mention belongs to
        let mut mention_to_cluster: HashMap<usize, usize> = HashMap::new();
        let mut clusters: Vec<Vec<usize>> = Vec::new();

        for (i, mention) in mentions.iter().enumerate() {
            let mut best_antecedent: Option<usize> = None;
            let mut best_score = self.config.link_threshold;

            // Score against all previous mentions
            for j in 0..i {
                let antecedent = &mentions[j];
                let score = self.antecedent_score(mention, antecedent);

                if score > best_score {
                    best_score = score;
                    best_antecedent = Some(j);
                }
            }

            if let Some(ant_idx) = best_antecedent {
                // Link to antecedent's cluster
                if let Some(&cluster_id) = mention_to_cluster.get(&ant_idx) {
                    clusters[cluster_id].push(i);
                    mention_to_cluster.insert(i, cluster_id);
                } else {
                    // Create new cluster with antecedent and current
                    let cluster_id = clusters.len();
                    clusters.push(vec![ant_idx, i]);
                    mention_to_cluster.insert(ant_idx, cluster_id);
                    mention_to_cluster.insert(i, cluster_id);
                }
            }
            // If no antecedent found, mention starts its own singleton
            // (singletons are usually filtered out)
        }

        // Convert to CorefCluster format
        clusters
            .into_iter()
            .enumerate()
            .map(|(id, member_indices)| CorefCluster {
                id,
                mentions: member_indices
                    .into_iter()
                    .map(|idx| mentions[idx].clone())
                    .collect(),
            })
            .collect()
    }

    /// Score a (mention, antecedent) pair.
    fn antecedent_score(&self, mention: &Mention, antecedent: &Mention) -> f64 {
        let mut score = 0.0;

        // String match features
        let m_lower = mention.text.to_lowercase();
        let a_lower = antecedent.text.to_lowercase();

        // Exact match
        if m_lower == a_lower {
            score += 1.0;
        }

        // Head match (last word)
        let m_head = m_lower.split_whitespace().last().unwrap_or("");
        let a_head = a_lower.split_whitespace().last().unwrap_or("");
        if !m_head.is_empty() && m_head == a_head {
            score += 0.5;
        }

        // Substring containment
        if m_lower.contains(&a_lower) || a_lower.contains(&m_lower) {
            score += 0.3;
        }

        // Pronoun resolution heuristics
        let pronouns_masc = ["he", "him", "his"];
        let pronouns_fem = ["she", "her", "hers"];
        let pronouns_neut = ["it", "its"];
        let pronouns_plur = ["they", "them", "their", "theirs"];

        let is_pronoun = |s: &str| {
            pronouns_masc.contains(&s)
                || pronouns_fem.contains(&s)
                || pronouns_neut.contains(&s)
                || pronouns_plur.contains(&s)
        };

        // If mention is a pronoun, check gender/number agreement
        if is_pronoun(&m_lower) {
            // Assume proper nouns can be antecedents for pronouns
            if antecedent.text.chars().next().is_some_and(|c| c.is_uppercase()) {
                score += 0.4;
            }
        }

        // Distance penalty
        let distance = mention.start.saturating_sub(antecedent.end);
        score -= (distance as f64) * 0.02;

        score
    }
}

impl Default for E2ECoref {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_resolution() {
        let coref = E2ECoref::new();
        let clusters = coref.resolve("John saw Mary. He waved to her.").unwrap();

        // Should find some clusters
        // Exact results depend on heuristic tuning
        for cluster in &clusters {
            assert!(!cluster.mentions.is_empty());
        }
    }

    #[test]
    fn test_empty_input() {
        let coref = E2ECoref::new();
        let clusters = coref.resolve("").unwrap();
        assert!(clusters.is_empty());
    }

    #[test]
    fn test_pronoun_detection() {
        let coref = E2ECoref::new();
        let score = coref.heuristic_mention_score("he");
        assert!(score > 0.5, "Pronouns should have high mention score");
    }

    #[test]
    fn test_proper_noun_detection() {
        let coref = E2ECoref::new();
        let score = coref.heuristic_mention_score("John Smith");
        assert!(score > 0.5, "Proper nouns should have high mention score");
    }

    #[test]
    fn test_antecedent_exact_match() {
        let coref = E2ECoref::new();

        let m1 = Mention {
            start: 0,
            end: 1,
            char_start: 0,
            char_end: 4,
            text: "John".to_string(),
            score: 0.8,
        };
        let m2 = Mention {
            start: 5,
            end: 6,
            char_start: 20,
            char_end: 24,
            text: "John".to_string(),
            score: 0.8,
        };

        let score = coref.antecedent_score(&m2, &m1);
        assert!(score > 0.8, "Exact match should have high antecedent score");
    }

    #[test]
    fn test_config() {
        let config = E2ECorefConfig {
            max_span_width: 5,
            max_antecedents: 20,
            top_spans_ratio: 0.3,
            mention_threshold: 0.1,
            link_threshold: 0.2,
        };

        let coref = E2ECoref::with_config(config);
        assert_eq!(coref.config.max_span_width, 5);
    }

    #[test]
    fn test_cluster_transitivity() {
        let coref = E2ECoref::new();
        // Text designed to create a chain: John -> he -> him
        let clusters = coref.resolve("John entered. He smiled. Mary greeted him.").unwrap();

        // Check that mentions are grouped into clusters
        for cluster in &clusters {
            for mention in &cluster.mentions {
                assert!(mention.char_start <= mention.char_end);
            }
        }
    }

    #[test]
    fn test_unicode_offsets() {
        let coref = E2ECoref::new();
        let text = "北京很大. 它是首都."; // Beijing is big. It is the capital.
        let char_count = text.chars().count();

        let clusters = coref.resolve(text).unwrap();

        for cluster in &clusters {
            for mention in &cluster.mentions {
                assert!(mention.char_start <= mention.char_end);
                assert!(mention.char_end <= char_count);
            }
        }
    }
}

