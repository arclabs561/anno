//! Chunking helpers for long text.
//!
//! **Status**:
//! - By default, this module provides a lightweight **rule-based** chunker (paragraph boundaries
//!   + size limits + overlap).
//! - An embedding-based chunker is **not implemented** yet; the `semantic-chunking` feature gate
//!   currently falls back to the rule-based behavior.
//!
//! This keeps chunking behavior explicit without implying that semantic embeddings are in use.
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

/// Embedding-based semantic chunker (requires embedding model).
///
/// Uses sentence embeddings and similarity clustering to identify semantic boundaries.
/// This is a placeholder - full implementation would require:
/// - Embedding model integration (sentence-transformers, BERT, etc.)
/// - Similarity computation (cosine similarity)
/// - Clustering algorithm (hierarchical, DBSCAN, etc.)
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
        // TODO: Load embedding model
        // For now, fall back to rule-based
        Ok(Self { config })
    }
}

#[cfg(feature = "semantic-chunking")]
impl SemanticChunker for EmbeddingSemanticChunker {
    fn chunk(&self, text: &str, language: Option<&str>) -> Result<Vec<SemanticChunk>> {
        // TODO: Implement embedding-based chunking
        // 1. Split into sentences
        // 2. Compute sentence embeddings
        // 3. Cluster by similarity
        // 4. Merge clusters into chunks (respecting size limits)
        // 5. Add overlap

        // For now, fall back to rule-based
        let fallback = RuleBasedSemanticChunker::new(self.config.clone());
        fallback.chunk(text, language)
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
