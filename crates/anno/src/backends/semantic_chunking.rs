//! Chunking helpers for long text.
//!
//! **Status**:
//! - By default, this module provides a lightweight **rule-based** chunker (paragraph boundaries
//!   + size limits + overlap).
//! - With the `semantic-chunking` feature enabled, this module additionally provides a
//!   sentence-level similarity chunker (token-based Jaccard similarity; no embedding model
//!   required).
//!
//! This keeps chunking behavior explicit without implying that embeddings are in use.
//!
//! # Example
//!
//! ```rust,ignore
//! use anno::backends::semantic_chunking::{SemanticChunker, SemanticChunkConfig};
//!
//! let config = SemanticChunkConfig::default();
//! let chunker = anno::backends::semantic_chunking::create_semantic_chunker(config)?;
//! let chunks = chunker.chunk(long_text, Some("en"))?;
//!
//! for chunk in chunks {
//!     println!("Chunk: {} ({} chars)", chunk.text, chunk.text.len());
//! }
//! ```

use crate::Result;
use std::collections::BTreeSet;

/// Configuration for semantic chunking.
#[derive(Debug, Clone)]
pub struct SemanticChunkConfig {
    /// Target chunk size in characters (soft limit)
    pub target_size: usize,
    /// Minimum chunk size in characters (hard limit)
    pub min_size: usize,
    /// Maximum chunk size in characters (hard limit)
    pub max_size: usize,
    /// Similarity threshold for chunk boundaries (0.0-1.0)
    /// Lower = more chunks, Higher = fewer chunks
    pub similarity_threshold: f32,
    /// Overlap between chunks in characters
    pub overlap: usize,
    /// Use sentence boundaries as fallback when similarity is ambiguous
    pub fallback_to_sentences: bool,
}

impl Default for SemanticChunkConfig {
    fn default() -> Self {
        Self {
            target_size: 10_000,
            min_size: 1_000,
            max_size: 20_000,
            similarity_threshold: 0.7,
            overlap: 200,
            fallback_to_sentences: true,
        }
    }
}

impl SemanticChunkConfig {
    /// Create config optimized for long documents.
    pub fn long_document() -> Self {
        Self {
            target_size: 50_000,
            min_size: 5_000,
            max_size: 100_000,
            similarity_threshold: 0.75,
            overlap: 500,
            fallback_to_sentences: true,
        }
    }

    /// Create config for coreference resolution (smaller chunks, higher similarity).
    pub fn coreference() -> Self {
        Self {
            target_size: 5_000,
            min_size: 500,
            max_size: 10_000,
            similarity_threshold: 0.8, // Higher = keep related mentions together
            overlap: 300,
            fallback_to_sentences: true,
        }
    }
}

/// A semantically coherent chunk of text.
#[derive(Debug, Clone)]
pub struct SemanticChunk {
    /// The text content of this chunk
    pub text: String,
    /// Starting character offset in original text
    pub start: usize,
    /// Ending character offset in original text
    pub end: usize,
    /// Optional topic label (if available)
    pub topic: Option<String>,
    /// Semantic similarity score with previous chunk (if available)
    pub similarity_to_prev: Option<f32>,
}

/// Trait for semantic chunking strategies.
pub trait SemanticChunker: Send + Sync {
    /// Chunk text based on semantic similarity.
    ///
    /// Returns chunks sorted by position in the original text.
    fn chunk(&self, text: &str, language: Option<&str>) -> Result<Vec<SemanticChunk>>;
}

/// Simple rule-based semantic chunker (fallback when embeddings unavailable).
///
/// Uses paragraph boundaries and sentence clustering as a lightweight alternative
/// to embedding-based chunking.
#[derive(Debug)]
pub struct RuleBasedSemanticChunker {
    config: SemanticChunkConfig,
}

impl RuleBasedSemanticChunker {
    /// Create a new rule-based semantic chunker.
    pub fn new(config: SemanticChunkConfig) -> Self {
        Self { config }
    }
}

impl SemanticChunker for RuleBasedSemanticChunker {
    fn chunk(&self, text: &str, language: Option<&str>) -> Result<Vec<SemanticChunk>> {
        let _ = language; // Acknowledge parameter for future use

        let mut chunks = Vec::new();
        let mut current_start = 0;
        let mut current_text = String::new();

        // Split by paragraphs (double newlines)
        let paragraphs: Vec<&str> = text.split("\n\n").collect();

        for paragraph in paragraphs {
            let paragraph_len = paragraph.chars().count();

            // If adding this paragraph would exceed max_size, start a new chunk
            if !current_text.is_empty()
                && (current_text.chars().count() + paragraph_len) > self.config.max_size
            {
                // Save current chunk
                let chunk_end = current_start + current_text.chars().count();
                chunks.push(SemanticChunk {
                    text: current_text.clone(),
                    start: current_start,
                    end: chunk_end,
                    topic: None,
                    similarity_to_prev: None,
                });

                // Start new chunk with overlap
                let overlap_start = chunk_end.saturating_sub(self.config.overlap);
                let overlap_text: String = text
                    .chars()
                    .skip(overlap_start)
                    .take(self.config.overlap)
                    .collect();
                current_text = overlap_text;
                current_start = overlap_start;
            }

            // Add paragraph to current chunk
            if !current_text.is_empty() {
                current_text.push_str("\n\n");
            }
            current_text.push_str(paragraph);
        }

        // Add final chunk
        if !current_text.is_empty() {
            let chunk_end = current_start + current_text.chars().count();
            chunks.push(SemanticChunk {
                text: current_text,
                start: current_start,
                end: chunk_end,
                topic: None,
                similarity_to_prev: None,
            });
        }

        // Ensure chunks meet min_size requirement (merge small chunks)
        let mut merged_chunks: Vec<SemanticChunk> = Vec::new();
        for chunk in chunks {
            if chunk.text.chars().count() < self.config.min_size && !merged_chunks.is_empty() {
                // Merge with previous chunk
                let last = merged_chunks.last_mut().unwrap();
                last.text.push_str("\n\n");
                last.text.push_str(&chunk.text);
                last.end = chunk.end;
            } else {
                merged_chunks.push(chunk);
            }
        }

        Ok(merged_chunks)
    }
}

/// Sentence-similarity chunker (feature = `semantic-chunking`).
///
/// Uses sentence-level similarity to identify coarse boundaries.
///
/// Despite the name, the current implementation does **not** use embeddings: it uses a
/// sentence-level token Jaccard similarity to decide boundaries. This keeps the feature gate and
/// config surface stable while avoiding heavyweight dependencies.
#[cfg(feature = "semantic-chunking")]
#[derive(Debug)]
pub struct EmbeddingSemanticChunker {
    config: SemanticChunkConfig,
    // TODO: Add embedding model when available
    // embedding_model: Box<dyn EmbeddingModel>,
}

#[cfg(feature = "semantic-chunking")]
impl EmbeddingSemanticChunker {
    /// Create a new embedding-based semantic chunker.
    pub fn new(config: SemanticChunkConfig) -> Result<Self> {
        Ok(Self { config })
    }

    fn tokenize_for_similarity(s: &str) -> BTreeSet<String> {
        // Keep this intentionally simple and dependency-light.
        //
        // - lowercase (ASCII)
        // - scrub non-alphanumeric to spaces
        // - split on whitespace
        // - drop very short tokens (noise)
        let mut t = String::with_capacity(s.len());
        for c in s.chars() {
            if c.is_alphanumeric() {
                t.push(c.to_ascii_lowercase());
            } else {
                t.push(' ');
            }
        }
        t.split_whitespace()
            .filter(|w| w.chars().count() > 2)
            .map(|w| w.to_string())
            .collect()
    }

    fn jaccard(a: &BTreeSet<String>, b: &BTreeSet<String>) -> f32 {
        if a.is_empty() && b.is_empty() {
            return 1.0;
        }
        if a.is_empty() || b.is_empty() {
            return 0.0;
        }
        let inter = a.intersection(b).count() as f32;
        let uni = a.union(b).count() as f32;
        if uni <= 0.0 {
            0.0
        } else {
            inter / uni
        }
    }

    fn char_to_byte_map(text: &str) -> Vec<usize> {
        // Index i => byte offset of the i'th char (and one extra sentinel at the end).
        let mut map = Vec::with_capacity(text.chars().count() + 1);
        for (b, _) in text.char_indices() {
            map.push(b);
        }
        map.push(text.len());
        map
    }

    fn byte_at_char(map: &[usize], char_idx: usize) -> usize {
        *map.get(char_idx).unwrap_or(&map[map.len() - 1])
    }

    fn split_sentences_spans(text: &str) -> Vec<(usize, usize)> {
        // Return (start_char, end_char) spans for coarse sentence segments.
        let terminators = [
            '.', '!', '?', // Latin
            '。', '！', '？', // CJK
            '؟', '۔', // Arabic/Urdu
            '।', // Devanagari
        ];
        let mut out = Vec::new();
        let mut start = 0usize;
        let mut i = 0usize;
        for c in text.chars() {
            i += 1;
            if terminators.contains(&c) {
                if i > start {
                    out.push((start, i));
                }
                start = i;
            }
        }
        if i > start {
            out.push((start, i));
        }
        out
    }
}

#[cfg(feature = "semantic-chunking")]
impl SemanticChunker for EmbeddingSemanticChunker {
    fn chunk(&self, text: &str, language: Option<&str>) -> Result<Vec<SemanticChunk>> {
        let _ = language;
        let t = text.trim();
        if t.is_empty() {
            return Ok(vec![]);
        }

        let spans = Self::split_sentences_spans(text);
        if spans.is_empty() {
            let fallback = RuleBasedSemanticChunker::new(self.config.clone());
            return fallback.chunk(text, None);
        }

        let char_to_byte = Self::char_to_byte_map(text);

        let mut chunks: Vec<SemanticChunk> = Vec::new();
        let mut chunk_start_char = spans[0].0;
        let mut chunk_end_char = spans[0].1;
        let mut prev_sentence_tokens: Option<BTreeSet<String>> = None;
        let mut prev_chunk_similarity: Option<f32> = None;

        for (idx, (s0, s1)) in spans.iter().copied().enumerate() {
            let sent_start = s0;
            let sent_end = s1;

            let sent_bytes_start = Self::byte_at_char(&char_to_byte, sent_start);
            let sent_bytes_end = Self::byte_at_char(&char_to_byte, sent_end);
            let sent_text = text
                .get(sent_bytes_start..sent_bytes_end)
                .unwrap_or("")
                .trim();

            if sent_text.is_empty() {
                continue;
            }

            let tokens = Self::tokenize_for_similarity(sent_text);
            let sim_to_prev_sentence = prev_sentence_tokens
                .as_ref()
                .map(|p| Self::jaccard(p, &tokens));

            // Decide whether to cut before this sentence.
            if idx > 0 {
                let cur_len = chunk_end_char.saturating_sub(chunk_start_char);
                let would_len = sent_end.saturating_sub(chunk_start_char);
                let similarity_break = sim_to_prev_sentence
                    .map(|s| s < self.config.similarity_threshold)
                    .unwrap_or(false);
                let would_exceed =
                    would_len > self.config.max_size && cur_len >= self.config.min_size;

                if (similarity_break && cur_len >= self.config.min_size) || would_exceed {
                    let start_b = Self::byte_at_char(&char_to_byte, chunk_start_char);
                    let end_b = Self::byte_at_char(&char_to_byte, chunk_end_char);
                    let chunk_text = text.get(start_b..end_b).unwrap_or("").trim().to_string();
                    if !chunk_text.is_empty() {
                        chunks.push(SemanticChunk {
                            text: chunk_text,
                            start: chunk_start_char,
                            end: chunk_end_char,
                            topic: None,
                            similarity_to_prev: prev_chunk_similarity,
                        });
                    }

                    // Start new chunk, with optional overlap.
                    let overlap_start_char = chunk_end_char
                        .saturating_sub(self.config.overlap)
                        .min(sent_start);
                    chunk_start_char = overlap_start_char;
                    prev_chunk_similarity = sim_to_prev_sentence;
                }
            }

            // Extend chunk end to cover this sentence.
            chunk_end_char = sent_end;
            prev_sentence_tokens = Some(tokens);
        }

        // Final chunk.
        if chunk_end_char > chunk_start_char {
            let start_b = Self::byte_at_char(&char_to_byte, chunk_start_char);
            let end_b = Self::byte_at_char(&char_to_byte, chunk_end_char);
            let chunk_text = text.get(start_b..end_b).unwrap_or("").trim().to_string();
            if !chunk_text.is_empty() {
                chunks.push(SemanticChunk {
                    text: chunk_text,
                    start: chunk_start_char,
                    end: chunk_end_char,
                    topic: None,
                    similarity_to_prev: prev_chunk_similarity,
                });
            }
        }

        if chunks.is_empty() {
            let fallback = RuleBasedSemanticChunker::new(self.config.clone());
            return fallback.chunk(text, None);
        }

        Ok(chunks)
    }
}

/// Factory function to create appropriate chunker based on available features.
pub fn create_semantic_chunker(config: SemanticChunkConfig) -> Result<Box<dyn SemanticChunker>> {
    #[cfg(feature = "semantic-chunking")]
    {
        // Try embedding-based chunker first
        match EmbeddingSemanticChunker::new(config.clone()) {
            Ok(chunker) => Ok(Box::new(chunker)),
            Err(_) => Ok(Box::new(RuleBasedSemanticChunker::new(config))),
        }
    }

    #[cfg(not(feature = "semantic-chunking"))]
    {
        // Fall back to rule-based
        Ok(Box::new(RuleBasedSemanticChunker::new(config)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rule_based_chunker() {
        let config = SemanticChunkConfig {
            target_size: 100,
            min_size: 50,
            max_size: 200,
            similarity_threshold: 0.7,
            overlap: 20,
            fallback_to_sentences: true,
        };

        let chunker = RuleBasedSemanticChunker::new(config);
        let text = "Paragraph one.\n\nParagraph two.\n\nParagraph three.";
        let chunks = chunker.chunk(text, None).unwrap();

        assert!(!chunks.is_empty());
        assert_eq!(chunks[0].start, 0);
    }

    #[test]
    fn test_chunker_respects_min_size() {
        let config = SemanticChunkConfig {
            target_size: 1000,
            min_size: 100,
            max_size: 2000,
            similarity_threshold: 0.7,
            overlap: 50,
            fallback_to_sentences: true,
        };

        let chunker = RuleBasedSemanticChunker::new(config);
        let text = "Short.\n\nAlso short.";
        let chunks = chunker.chunk(text, None).unwrap();

        // Small chunks should be merged
        assert!(chunks.len() <= 1 || chunks[0].text.chars().count() >= 100);
    }
}
