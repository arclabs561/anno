//! Extractive summarization.
//!
//! This module provides algorithms for selecting important sentences from text.
//! Extractive summarization **selects** existing sentences rather than generating new text.
//!
//! # Conceptual Framework
//!
//! Extractive summarization follows the same graph-based pattern as other modules:
//!
//! ```text
//! Text → Sentences → Scoring → Ranking → Summary
//! ```
//!
//! Scoring can use:
//! - **Keyword overlap**: Sentences with important keywords score higher
//! - **Entity salience**: Sentences with salient entities score higher
//! - **Graph centrality**: Sentences similar to many others score higher (LexRank)
//! - **Position**: Earlier sentences often summarize
//!
//! # Relationship to Other Modules
//!
//! ```text
//! anno::keywords ────► Important terms
//!                             │
//!                             ▼
//! anno::summarize ◄───► Sentence scoring ◄──── anno::salience
//!                             │                      │
//!                             ▼                      ▼
//!                       Important sentences    Important entities
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use anno::summarize::{Summarizer, LexRankSummarizer};
//!
//! let text = "Long article text...";
//! let summarizer = LexRankSummarizer::default();
//! let summary = summarizer.summarize(text, 3); // Top 3 sentences
//!
//! for sentence in summary {
//!     println!("- {}", sentence);
//! }
//! ```
//!
//! # References
//!
//! - Erkan & Radev (2004): LexRank: Graph-based Lexical Centrality
//! - Mihalcea & Tarau (2004): TextRank for Summarization
//!
//! # Status
//!
//! Current implementation status:
//! - [x] Position-based baseline
//! - [x] Keyword-based scoring (TF-IDF)
//! - [x] LexRank (graph-based PageRank on sentence similarity)
//! - [ ] Entity-guided summarization (integrate with `salience` module)
//! - [ ] Multi-document summarization

use crate::keywords::{KeywordExtractor, TfIdfExtractor};
use crate::pagerank::{pagerank, PageRankConfig};
use std::collections::HashSet;

/// Trait for extractive summarizers.
pub trait Summarizer: Send + Sync {
    /// Extract the most important sentences.
    ///
    /// Returns sentences in their original document order.
    fn summarize(&self, text: &str, num_sentences: usize) -> Vec<String>;

    /// Summarize to approximately the given ratio of original length.
    fn summarize_ratio(&self, text: &str, ratio: f64) -> Vec<String> {
        let sentences = split_sentences(text);
        let target = (sentences.len() as f64 * ratio.clamp(0.0, 1.0)).ceil() as usize;
        self.summarize(text, target.max(1))
    }
}

/// Split text into sentences.
///
/// # Language Support
///
/// This implementation handles common sentence-ending punctuation for:
/// - Latin scripts (. ! ?)
/// - CJK (。！？)
/// - Arabic (؟)
///
/// For more robust sentence segmentation, consider using:
/// - `unicode-segmentation` crate
/// - Language-specific tools (spaCy, StanfordNLP)
///
/// # Limitations
///
/// - Does not handle abbreviations well ("Dr. Smith" → two sentences)
/// - May miss sentences in some scripts (Thai, etc.)
pub fn split_sentences(text: &str) -> Vec<String> {
    // Regex-free approach: split on common sentence terminators
    // Including Unicode variants for CJK and Arabic
    let terminators = [
        '.', '!', '?', // Latin
        '。', '！', '？', // CJK
        '؟', '۔', // Arabic/Urdu
        '।', // Devanagari
    ];

    let mut sentences = Vec::new();
    let mut current = String::new();

    for c in text.chars() {
        current.push(c);

        if terminators.contains(&c) {
            let trimmed = current.trim().to_string();
            // Only include if it has substantial content
            if !trimmed.is_empty() && trimmed.chars().count() > 5 {
                sentences.push(trimmed);
            }
            current.clear();
        }
    }

    // Don't forget any trailing text
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() && trimmed.chars().count() > 5 {
        sentences.push(trimmed);
    }

    sentences
}

/// Split text into sentences with language hint.
///
/// # Arguments
///
/// * `text` - The text to split
/// * `lang` - ISO 639-1 language code (e.g., "en", "zh", "ja", "ar")
///
/// Currently this just calls `split_sentences`, but provides an extension
/// point for language-specific implementations.
pub fn split_sentences_lang(text: &str, _lang: &str) -> Vec<String> {
    // Language-specific sentence splitting would require:
    // 1. CJK: Sentence boundaries are different (。 instead of .)
    // 2. Arabic: Right-to-left with different punctuation
    // 3. Thai: No spaces between words, special tokenization
    //
    // For full implementation, consider integrating with:
    // - unicode-segmentation for UAX#29 word boundaries
    // - language-specific rules from ICU4X
    //
    // For now, the generic split_sentences handles Latin-script languages well.
    split_sentences(text)
}

// =============================================================================
// Position-Based Summarizer (Baseline)
// =============================================================================

/// Simple position-based summarizer.
///
/// Selects first N sentences, based on the "inverted pyramid" structure
/// common in news articles where important information comes first.
#[derive(Debug, Clone, Default)]
pub struct PositionSummarizer;

impl Summarizer for PositionSummarizer {
    fn summarize(&self, text: &str, num_sentences: usize) -> Vec<String> {
        split_sentences(text)
            .into_iter()
            .take(num_sentences)
            .collect()
    }
}

// =============================================================================
// Keyword-Based Summarizer
// =============================================================================

/// Summarizer that scores sentences by keyword overlap.
///
/// Uses TF-IDF to find important keywords, then ranks sentences
/// by how many keywords they contain.
#[derive(Debug, Clone)]
pub struct KeywordSummarizer {
    num_keywords: usize,
}

impl Default for KeywordSummarizer {
    fn default() -> Self {
        Self { num_keywords: 20 }
    }
}

impl KeywordSummarizer {
    /// Create with custom number of keywords to use.
    pub fn with_num_keywords(mut self, n: usize) -> Self {
        self.num_keywords = n;
        self
    }
}

impl Summarizer for KeywordSummarizer {
    fn summarize(&self, text: &str, num_sentences: usize) -> Vec<String> {
        let sentences = split_sentences(text);
        if sentences.is_empty() {
            return vec![];
        }

        // Extract keywords
        let extractor = TfIdfExtractor::new();
        let keywords: HashSet<String> = extractor
            .extract(text, self.num_keywords)
            .into_iter()
            .map(|(k, _)| k.to_lowercase())
            .collect();

        // Score each sentence by keyword overlap
        let mut scored: Vec<(usize, f64, &String)> = sentences
            .iter()
            .enumerate()
            .map(|(i, sent)| {
                let words: HashSet<String> = sent
                    .split(|c: char| !c.is_alphanumeric())
                    .map(|w| w.to_lowercase())
                    .collect();

                let overlap = words.intersection(&keywords).count() as f64;
                let score = overlap / (sent.split_whitespace().count() as f64).sqrt();

                (i, score, sent)
            })
            .collect();

        // Sort by score descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top N and restore original order
        let mut selected: Vec<(usize, String)> = scored
            .into_iter()
            .take(num_sentences)
            .map(|(i, _, s)| (i, s.clone()))
            .collect();

        selected.sort_by_key(|(i, _)| *i);
        selected.into_iter().map(|(_, s)| s).collect()
    }
}

// =============================================================================
// LexRank Summarizer (Graph-Based)
// =============================================================================

/// LexRank summarizer using sentence similarity graph.
///
/// Builds a graph where sentences are nodes and edges are weighted by
/// cosine similarity. Runs PageRank to find central sentences.
///
/// # Algorithm
///
/// 1. Represent each sentence as TF-IDF vector
/// 2. Build similarity graph (edge if similarity > threshold)
/// 3. Run PageRank to rank sentences
/// 4. Select top-ranked sentences
///
/// # Reference
///
/// Erkan & Radev (2004): "LexRank: Graph-based Lexical Centrality
/// as Salience in Text Summarization"
#[derive(Debug, Clone)]
pub struct LexRankSummarizer {
    /// Similarity threshold for creating edges
    pub threshold: f64,
    /// PageRank damping factor
    pub damping: f64,
    /// Max iterations
    pub iterations: usize,
}

impl Default for LexRankSummarizer {
    fn default() -> Self {
        Self {
            threshold: 0.1,
            damping: 0.85,
            iterations: 30,
        }
    }
}

impl LexRankSummarizer {
    /// Compute TF vector for a sentence.
    fn sentence_tf(&self, sentence: &str) -> std::collections::HashMap<String, f64> {
        let mut tf = std::collections::HashMap::new();
        for word in sentence.split(|c: char| !c.is_alphanumeric()) {
            let lower = word.to_lowercase();
            if lower.len() > 2 {
                *tf.entry(lower).or_insert(0.0) += 1.0;
            }
        }
        tf
    }

    /// Compute cosine similarity between two TF vectors.
    fn cosine_similarity(
        &self,
        a: &std::collections::HashMap<String, f64>,
        b: &std::collections::HashMap<String, f64>,
    ) -> f64 {
        let dot: f64 = a
            .iter()
            .filter_map(|(k, v)| b.get(k).map(|bv| v * bv))
            .sum();

        let norm_a: f64 = a.values().map(|v| v * v).sum::<f64>().sqrt();
        let norm_b: f64 = b.values().map(|v| v * v).sum::<f64>().sqrt();

        if norm_a > 0.0 && norm_b > 0.0 {
            dot / (norm_a * norm_b)
        } else {
            0.0
        }
    }
}

impl Summarizer for LexRankSummarizer {
    fn summarize(&self, text: &str, num_sentences: usize) -> Vec<String> {
        let sentences = split_sentences(text);
        let n = sentences.len();

        if n == 0 {
            return vec![];
        }

        // Compute TF vectors for each sentence
        let tf_vectors: Vec<_> = sentences.iter().map(|s| self.sentence_tf(s)).collect();

        // Build similarity matrix
        let mut similarity = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let sim = self.cosine_similarity(&tf_vectors[i], &tf_vectors[j]);
                if sim > self.threshold {
                    similarity[i][j] = sim;
                    similarity[j][i] = sim;
                }
            }
        }

        // Run PageRank using shared implementation
        let config = PageRankConfig {
            damping: self.damping,
            max_iterations: self.iterations,
            epsilon: 1e-6,
        };
        let scores = pagerank(&similarity, &config);

        // Select top sentences and restore order
        let mut indexed: Vec<_> = scores.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut selected: Vec<(usize, String)> = indexed
            .into_iter()
            .take(num_sentences)
            .map(|(i, _)| (i, sentences[i].clone()))
            .collect();

        selected.sort_by_key(|(i, _)| *i);
        selected.into_iter().map(|(_, s)| s).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_TEXT: &str = "Machine learning is transforming technology. \
        Deep learning uses neural networks to learn patterns. \
        Neural networks are inspired by the human brain. \
        AI applications include image recognition and NLP. \
        The future of AI looks promising.";

    const LONGER_TEXT: &str = "The stock market showed significant gains today. \
        Technology companies led the rally with Apple and Microsoft at the forefront. \
        Apple announced record quarterly earnings driven by iPhone sales. \
        Microsoft's cloud division Azure continued its strong growth trajectory. \
        Investors remain optimistic about the tech sector's future. \
        Analysts predict continued growth in the semiconductor industry. \
        The Federal Reserve signaled it may adjust interest rates. \
        This news caused brief volatility in bond markets. \
        However, overall market sentiment remained positive. \
        Global supply chain issues are gradually improving.";

    #[test]
    fn test_position_summarizer() {
        let summarizer = PositionSummarizer;
        let summary = summarizer.summarize(TEST_TEXT, 2);

        assert_eq!(summary.len(), 2);
        assert!(summary[0].contains("Machine learning"));
    }

    #[test]
    fn test_keyword_summarizer() {
        let summarizer = KeywordSummarizer::default();
        let summary = summarizer.summarize(TEST_TEXT, 2);

        assert_eq!(summary.len(), 2);
        // Should select sentences with keywords
    }

    #[test]
    fn test_lexrank_summarizer() {
        let summarizer = LexRankSummarizer::default();
        let summary = summarizer.summarize(TEST_TEXT, 2);

        assert_eq!(summary.len(), 2);
        // Should select central sentences
    }

    #[test]
    fn test_empty_text() {
        let summarizer = KeywordSummarizer::default();
        let summary = summarizer.summarize("", 5);
        assert!(summary.is_empty());
    }

    #[test]
    fn test_summarize_ratio() {
        let summarizer = PositionSummarizer;
        let summary = summarizer.summarize_ratio(TEST_TEXT, 0.5);

        // 5 sentences, 50% = ~2-3 sentences
        assert!(summary.len() >= 2);
    }

    // =========================================================================
    // Sentence Splitting Tests
    // =========================================================================

    #[test]
    fn test_split_sentences_basic() {
        let text = "First sentence. Second sentence. Third sentence.";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 3);
    }

    #[test]
    fn test_split_sentences_cjk() {
        let text = "这是第一句。这是第二句。这是第三句。";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 3);
        assert!(sentences[0].contains("第一句"));
    }

    #[test]
    fn test_split_sentences_mixed_punctuation() {
        let text = "Hello! How are you? I am fine.";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 3);
    }

    #[test]
    fn test_split_sentences_japanese() {
        let text = "東京は日本の首都です。京都は古い都市です。大阪は大きい街です。";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 3);
    }

    #[test]
    fn test_split_sentences_arabic() {
        let text = "مرحبا بالعالم؟ كيف حالك؟ أنا بخير۔";
        let sentences = split_sentences(text);
        // Should split on Arabic question mark and Urdu full stop
        assert!(!sentences.is_empty());
    }

    // =========================================================================
    // Comprehensive Summarizer Tests
    // =========================================================================

    #[test]
    fn test_lexrank_on_longer_text() {
        let summarizer = LexRankSummarizer::default();
        let summary = summarizer.summarize(LONGER_TEXT, 3);

        assert_eq!(summary.len(), 3);
        // The sentences should be in original order
        // and should contain key topics (tech companies, market)
    }

    #[test]
    fn test_keyword_summarizer_selects_relevant() {
        let summarizer = KeywordSummarizer::default().with_num_keywords(10);
        let summary = summarizer.summarize(LONGER_TEXT, 3);

        assert_eq!(summary.len(), 3);
        // Should select sentences with keywords like "market", "technology", "Apple", etc.
        let combined = summary.join(" ");
        // At least some tech/market keywords should be present
        assert!(
            combined.contains("market")
                || combined.contains("tech")
                || combined.contains("Apple")
                || combined.contains("Microsoft")
        );
    }

    #[test]
    fn test_summarizers_preserve_order() {
        let lexrank = LexRankSummarizer::default();
        let summary = lexrank.summarize(LONGER_TEXT, 5);

        // Verify sentences are in original document order
        let all_sentences = split_sentences(LONGER_TEXT);
        let mut last_idx = 0;
        for s in &summary {
            if let Some(idx) = all_sentences.iter().position(|orig| orig == s) {
                assert!(idx >= last_idx, "Sentences should be in original order");
                last_idx = idx;
            }
        }
    }

    #[test]
    fn test_lexrank_threshold_effect() {
        // Higher threshold = fewer edges = different results
        let low_threshold = LexRankSummarizer {
            threshold: 0.05,
            ..Default::default()
        };
        let high_threshold = LexRankSummarizer {
            threshold: 0.3,
            ..Default::default()
        };

        let summary_low = low_threshold.summarize(LONGER_TEXT, 2);
        let summary_high = high_threshold.summarize(LONGER_TEXT, 2);

        // Both should return 2 sentences
        assert_eq!(summary_low.len(), 2);
        assert_eq!(summary_high.len(), 2);
        // Results may differ due to different graph structures
    }

    #[test]
    fn test_lexrank_damping_effect() {
        let low_damping = LexRankSummarizer {
            damping: 0.5,
            ..Default::default()
        };
        let high_damping = LexRankSummarizer {
            damping: 0.95,
            ..Default::default()
        };

        let summary_low = low_damping.summarize(LONGER_TEXT, 2);
        let summary_high = high_damping.summarize(LONGER_TEXT, 2);

        // Both should return 2 sentences
        assert_eq!(summary_low.len(), 2);
        assert_eq!(summary_high.len(), 2);
    }

    #[test]
    fn test_single_sentence_text() {
        let text = "This is a single sentence with enough words to pass the filter.";
        let summarizer = LexRankSummarizer::default();
        let summary = summarizer.summarize(text, 3);

        assert_eq!(summary.len(), 1);
        assert!(summary[0].contains("single sentence"));
    }

    #[test]
    fn test_very_short_sentences_filtered() {
        let text = "Hi. OK. Yes. This is a longer sentence that should be included.";
        let sentences = split_sentences(text);
        // Very short sentences (< 5 chars) should be filtered
        assert!(sentences.len() <= 2); // Only longer sentences
    }

    #[test]
    fn test_summarize_ratio_bounds() {
        let summarizer = PositionSummarizer;

        // 0% should still return at least 1
        let zero = summarizer.summarize_ratio(LONGER_TEXT, 0.0);
        assert!(!zero.is_empty());

        // 100% should return all
        let full = summarizer.summarize_ratio(LONGER_TEXT, 1.0);
        let all = split_sentences(LONGER_TEXT);
        assert_eq!(full.len(), all.len());

        // Over 100% should be clamped
        let over = summarizer.summarize_ratio(LONGER_TEXT, 2.0);
        assert_eq!(over.len(), all.len());
    }

    // =========================================================================
    // Edge Cases
    // =========================================================================

    #[test]
    fn test_whitespace_only() {
        let summarizer = KeywordSummarizer::default();
        let summary = summarizer.summarize("   \n\t   ", 5);
        assert!(summary.is_empty());
    }

    #[test]
    fn test_no_sentence_terminators() {
        let text = "This text has no sentence terminators but is long enough to be included";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 1);
        assert!(sentences[0].contains("terminators"));
    }

    #[test]
    fn test_repeated_sentences() {
        let text = "Hello world. Hello world. Hello world. Something different.";
        let summarizer = LexRankSummarizer::default();
        let summary = summarizer.summarize(text, 2);
        assert_eq!(summary.len(), 2);
    }
}
