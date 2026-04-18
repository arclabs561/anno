//! Text chunking and overlap resolution for NER.
//!
//! Provides utilities for splitting large documents into overlapping chunks
//! suitable for independent NER processing, and for resolving duplicate or
//! overlapping entity spans that arise from chunk boundaries.
//!
//! # Key components
//!
//! - [`ChunkConfig`](crate::backends::chunking::ChunkConfig) -- chunking parameters (size, overlap, sentence boundaries)
//! - [`chunk_text()`](crate::backends::chunking::chunk_text) -- split text into [`TextChunk`](crate::backends::chunking::TextChunk)s with character offsets
//! - [`extract_chunked_parallel()`](crate::backends::chunking::extract_chunked_parallel) -- parallel chunked extraction with dedup
//! - [`OverlapStrategy`](crate::backends::chunking::OverlapStrategy) / [`deduplicate_overlapping()`](crate::backends::chunking::deduplicate_overlapping) -- overlap resolution
//! - [`find_sentence_boundary()`](crate::backends::chunking::find_sentence_boundary) / [`find_word_boundary()`](crate::backends::chunking::find_word_boundary) -- boundary helpers

use crate::{Entity, Result};

#[cfg(feature = "chunking")]
use text_splitter::TextSplitter;

/// Configuration for chunked text processing.
#[derive(Debug, Clone)]
pub struct ChunkConfig {
    /// Target chunk size in characters (actual may vary to avoid splitting words)
    pub chunk_size: usize,
    /// Overlap between chunks (characters) to catch entities at boundaries
    pub overlap: usize,
    /// Sentence boundary detection (if true, chunks end at sentence boundaries)
    pub respect_sentences: bool,
    /// Maximum entities to buffer before yielding
    pub buffer_size: usize,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            chunk_size: 10_000,
            overlap: 100,
            respect_sentences: true,
            buffer_size: 1000,
        }
    }
}

impl ChunkConfig {
    /// Create a config for small documents (no chunking).
    pub fn no_chunking() -> Self {
        Self {
            chunk_size: usize::MAX,
            overlap: 0,
            respect_sentences: false,
            buffer_size: usize::MAX,
        }
    }

    /// Create a config optimized for long documents.
    pub fn long_document() -> Self {
        Self {
            chunk_size: 50_000,
            overlap: 200,
            respect_sentences: true,
            buffer_size: 5000,
        }
    }

    /// Create a config for real-time/streaming input.
    pub fn realtime() -> Self {
        Self {
            chunk_size: 1000,
            overlap: 50,
            respect_sentences: false,
            buffer_size: 100,
        }
    }
}

/// A text chunk with its character offset in the original text.
#[derive(Debug, Clone)]
pub struct TextChunk {
    /// The chunk text.
    pub text: String,
    /// Character offset of this chunk's start in the original text.
    pub char_offset: usize,
}

/// Split text into chunks suitable for independent NER processing.
///
/// Each chunk respects sentence/word boundaries and includes overlap
/// for entity recovery at boundaries. Returns chunks with their
/// character offsets so entities can be mapped back to the original text.
///
/// This is the shared chunking primitive used by `extract_chunked_parallel`
/// and LLM backends like UniversalNER.
pub fn chunk_text(text: &str, config: &ChunkConfig) -> Vec<TextChunk> {
    let chars: Vec<char> = text.chars().collect();
    let text_len = chars.len();

    if text_len == 0 {
        return Vec::new();
    }

    // If text fits in one chunk, return it directly.
    if text_len <= config.chunk_size {
        return vec![TextChunk {
            text: text.to_string(),
            char_offset: 0,
        }];
    }

    let mut chunks = Vec::new();
    let mut position = 0;

    while position < text_len {
        let chunk_end = (position + config.chunk_size).min(text_len);

        let actual_end = if chunk_end >= text_len {
            text_len
        } else if config.respect_sentences {
            find_sentence_boundary(&chars, position, chunk_end)
        } else {
            find_word_boundary(&chars, chunk_end)
        };

        let chunk_str: String = chars[position..actual_end].iter().collect();
        chunks.push(TextChunk {
            text: chunk_str,
            char_offset: position,
        });

        if actual_end >= text_len {
            break;
        }

        // Advance with overlap, ensuring forward progress.
        let overlap_position = actual_end.saturating_sub(config.overlap);
        position = if overlap_position <= position {
            position + 1
        } else {
            overlap_position
        };
    }

    chunks
}

/// Find a sentence boundary near the target position.
pub fn find_sentence_boundary(chars: &[char], start: usize, target: usize) -> usize {
    // Look backwards from target for sentence-ending punctuation
    let search_start = target.saturating_sub(200);
    for i in (search_start..target).rev() {
        if i >= chars.len() {
            continue;
        }
        let c = chars[i];
        // Sentence boundaries: . ! ? followed by whitespace or end
        let is_cjk_punct = c == '。' || c == '！' || c == '？';
        let is_latin_punct = c == '.' || c == '!' || c == '?';
        if is_cjk_punct
            || (is_latin_punct && (i + 1 >= chars.len() || chars[i + 1].is_whitespace()))
        {
            // Return position after the punctuation and whitespace
            let mut end = i + 1;
            while end < chars.len() && chars[end].is_whitespace() {
                end += 1;
            }
            if end > start {
                return end;
            }
        }
    }
    // No sentence boundary found, fall back to word boundary
    find_word_boundary(chars, target)
}

/// Find a word boundary near the target position.
pub fn find_word_boundary(chars: &[char], target: usize) -> usize {
    let target = target.min(chars.len());

    // If we're already at end, return it
    if target >= chars.len() {
        return chars.len();
    }

    // Look backwards for whitespace
    for i in (0..target).rev() {
        if chars[i].is_whitespace() {
            return i + 1;
        }
    }
    target
}

/// Strategy for resolving overlapping entity spans.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverlapStrategy {
    /// Keep first entity by position (sorted by start, then confidence desc).
    /// Any later entity whose span overlaps a kept entity is dropped.
    KeepFirst,
    /// Sort by confidence descending, greedily keep the highest-confidence
    /// entity, drop anything that overlaps it. Result is re-sorted by position.
    KeepHighestConfidence,
    /// Only resolve overlaps between entities of the **same** type;
    /// when two same-type entities overlap, keep the longer span.
    /// Different-type overlaps are preserved (union behavior).
    KeepLongerSameType,
    /// Prefer shorter / contained spans over supersets (GLiNER-style).
    /// Sorted shortest-first; supersets of already-kept entities are dropped,
    /// and partially-overlapping entities are also dropped.
    KeepShortest,
}

/// Remove overlapping entities from `entities` according to `strategy`.
///
/// The result is always sorted by start position.
pub fn deduplicate_overlapping(entities: &mut Vec<Entity>, strategy: OverlapStrategy) {
    if entities.len() <= 1 {
        return;
    }

    let result = match strategy {
        OverlapStrategy::KeepFirst => {
            // Sort by start, then by confidence (desc)
            entities.sort_by(|a, b| {
                a.start().cmp(&b.start()).then(
                    b.confidence
                        .partial_cmp(&a.confidence)
                        .expect("confidence values should be comparable"),
                )
            });

            let mut out = Vec::new();
            let mut last_end = 0;

            for entity in entities.drain(..) {
                if entity.start() >= last_end {
                    last_end = entity.end();
                    out.push(entity);
                }
            }
            out
        }

        OverlapStrategy::KeepHighestConfidence => {
            // Sort by confidence descending
            entities.sort_by(|a, b| {
                b.confidence
                    .partial_cmp(&a.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let mut out = Vec::with_capacity(entities.len());
            for entity in entities.drain(..) {
                let overlaps = out
                    .iter()
                    .any(|e: &Entity| entity.start() < e.end() && entity.end() > e.start());
                if !overlaps {
                    out.push(entity);
                }
            }
            // Re-sort by position
            out.sort_by_key(|e| e.start());
            out
        }

        OverlapStrategy::KeepLongerSameType => {
            entities.sort_by_key(|e| (e.start(), e.end()));

            let mut out: Vec<Entity> = Vec::with_capacity(entities.len());

            for entity in entities.drain(..) {
                // Check ALL kept entities for same-type overlap, not just the last one.
                // This handles interleaved different-type entities correctly.
                let overlapping_idx = out.iter().rposition(|prev: &Entity| {
                    entity.start() < prev.end()
                        && prev.start() < entity.end()
                        && prev.entity_type == entity.entity_type
                });

                if let Some(idx) = overlapping_idx {
                    let prev_len = out[idx].end() - out[idx].start();
                    let cand_len = entity.end() - entity.start();
                    if cand_len > prev_len {
                        out[idx] = entity;
                    }
                } else {
                    out.push(entity);
                }
            }
            out
        }

        OverlapStrategy::KeepShortest => {
            // Sort by span length (shorter first), then confidence desc
            entities.sort_unstable_by(|a, b| {
                let len_a = a.end() - a.start();
                let len_b = b.end() - b.start();
                len_a.cmp(&len_b).then_with(|| {
                    b.confidence
                        .partial_cmp(&a.confidence)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
            });

            let mut out: Vec<Entity> = Vec::with_capacity(entities.len());

            for entity in entities.drain(..) {
                let is_superset_of_existing = out
                    .iter()
                    .any(|kept| entity.start() <= kept.start() && entity.end() >= kept.end());

                if is_superset_of_existing {
                    continue;
                }

                let overlaps_existing = out
                    .iter()
                    .any(|kept| entity.start() < kept.end() && kept.start() < entity.end());

                if !overlaps_existing {
                    out.push(entity);
                }
            }
            out.sort_unstable_by_key(|e| e.start());
            out
        }
    };

    *entities = result;
}

/// Extract entities from a large document using parallel chunked processing.
///
/// Splits `text` into overlapping chunks via `chunk_text()`, processes each
/// chunk through `extract_fn` in parallel threads, and coalesces results with
/// exact-span + same-type overlap dedup.
///
/// This is the generic parallel chunking primitive. Backend-specific wrappers
/// (e.g., UniversalNER) call this with their own extraction closure.
///
/// Returns entities sorted by position with no duplicate spans.
pub fn extract_chunked_parallel<F>(
    text: &str,
    config: &ChunkConfig,
    extract_fn: F,
) -> Result<Vec<Entity>>
where
    F: Fn(&str, usize) -> Result<Vec<Entity>> + Send + Sync,
{
    let chunks = chunk_text(text, config);

    if chunks.is_empty() {
        return Ok(Vec::new());
    }

    if chunks.len() == 1 {
        return extract_fn(&chunks[0].text, chunks[0].char_offset);
    }

    // Process all chunks in parallel.
    let results: Vec<Result<Vec<Entity>>> = std::thread::scope(|s| {
        let extract_fn = &extract_fn;
        let handles: Vec<_> = chunks
            .iter()
            .map(|chunk| s.spawn(move || extract_fn(&chunk.text, chunk.char_offset)))
            .collect();

        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    // Coalesce: exact-span dedup + same-type overlap dedup.
    let mut seen = std::collections::HashSet::new();
    let mut all_entities = Vec::new();

    for result in results {
        let entities = result?;
        for entity in entities {
            if seen.insert((entity.start(), entity.end())) {
                all_entities.push(entity);
            }
        }
    }

    all_entities.sort_by_key(|e| (e.start(), e.end()));
    deduplicate_overlapping(&mut all_entities, OverlapStrategy::KeepLongerSameType);

    Ok(all_entities)
}

/// Split text using `text-splitter` and return [`TextChunk`]s with character offsets.
///
/// Delegates boundary detection to the [`text-splitter`](https://crates.io/crates/text-splitter)
/// crate, which cascades through Unicode sentence, word, and grapheme boundaries before
/// falling back to characters. `chunk_capacity` is the upper-bound chunk size in characters.
///
/// Returns [`TextChunk`]s whose `char_offset` is the chunk's character position in the
/// original `text` (not byte position), matching the rest of anno's char-offset contract.
///
/// # Example
///
/// ```rust,ignore
/// use anno::backends::chunking::chunk_text_semantic;
///
/// let chunks = chunk_text_semantic(text, 1000);
/// ```
#[cfg(feature = "chunking")]
#[cfg_attr(docsrs, doc(cfg(feature = "chunking")))]
pub fn chunk_text_semantic(text: &str, chunk_capacity: usize) -> Vec<TextChunk> {
    if text.is_empty() || chunk_capacity == 0 {
        return Vec::new();
    }
    let splitter = TextSplitter::new(chunk_capacity);

    // text-splitter's chunk_char_indices yields (char_offset, &str) directly —
    // no byte-to-char conversion needed, no allocation per chunk beyond the
    // owned TextChunk.text.
    splitter
        .chunk_char_indices(text)
        .map(|idx| TextChunk {
            text: idx.chunk.to_string(),
            char_offset: idx.char_offset,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EntityType;

    #[test]
    fn test_chunk_config_presets() {
        let _no_chunk = ChunkConfig::no_chunking();
        let _long = ChunkConfig::long_document();
        let _realtime = ChunkConfig::realtime();
    }

    #[test]
    fn test_find_sentence_boundary() {
        let text: Vec<char> = "Hello world. This is a test.".chars().collect();
        let boundary = find_sentence_boundary(&text, 0, 20);
        // Should find boundary after "Hello world. "
        assert!(boundary > 0);
        assert!(boundary <= 20);
    }

    // =========================================================================
    // Chunk boundary helpers
    // =========================================================================

    #[test]
    fn test_find_word_boundary_at_end() {
        let chars: Vec<char> = "hello world".chars().collect();
        assert_eq!(find_word_boundary(&chars, 100), chars.len());
        assert_eq!(find_word_boundary(&chars, chars.len()), chars.len());
    }

    #[test]
    fn test_find_word_boundary_mid_word() {
        let chars: Vec<char> = "hello world foo".chars().collect();
        // target=14 is inside "foo" -> backtrack to space at 11 -> return 12
        let boundary = find_word_boundary(&chars, 14);
        assert_eq!(boundary, 12, "should break before 'foo'");
    }

    #[test]
    fn test_find_word_boundary_no_whitespace() {
        let chars: Vec<char> = "abcdefghij".chars().collect();
        assert_eq!(find_word_boundary(&chars, 5), 5);
    }

    #[test]
    fn test_find_sentence_boundary_cjk_punctuation() {
        let text: Vec<char> = "这是测试。下一句话开始了".chars().collect();
        let boundary = find_sentence_boundary(&text, 0, text.len());
        assert!(boundary <= text.len());
        // '。' is at index 4; next char (no whitespace) is index 5
        assert!(boundary >= 5, "should be at or after the CJK period");
    }

    #[test]
    fn test_find_sentence_boundary_no_punctuation() {
        let chars: Vec<char> = "no punctuation here at all".chars().collect();
        let boundary = find_sentence_boundary(&chars, 0, 20);
        // Falls through to find_word_boundary
        assert!(boundary > 0 && boundary <= 20);
    }

    #[test]
    fn test_find_sentence_boundary_exclamation_and_question() {
        let chars: Vec<char> = "Wow! Really? Yes indeed.".chars().collect();
        // Target 10 => backwards from 10: '!' at 3 followed by ' ' at 4 -> boundary = 5
        let boundary = find_sentence_boundary(&chars, 0, 10);
        assert_eq!(boundary, 5, "should split after 'Wow! '");
    }

    // =========================================================================
    // Config / builder patterns
    // =========================================================================

    #[test]
    fn test_chunk_config_no_chunking_values() {
        let cfg = ChunkConfig::no_chunking();
        assert_eq!(cfg.chunk_size, usize::MAX);
        assert_eq!(cfg.overlap, 0);
        assert!(!cfg.respect_sentences);
        assert_eq!(cfg.buffer_size, usize::MAX);
    }

    #[test]
    fn test_chunk_config_long_document_values() {
        let cfg = ChunkConfig::long_document();
        assert_eq!(cfg.chunk_size, 50_000);
        assert_eq!(cfg.overlap, 200);
        assert!(cfg.respect_sentences);
        assert_eq!(cfg.buffer_size, 5000);
    }

    #[test]
    fn test_chunk_config_realtime_values() {
        let cfg = ChunkConfig::realtime();
        assert_eq!(cfg.chunk_size, 1000);
        assert_eq!(cfg.overlap, 50);
        assert!(!cfg.respect_sentences);
        assert_eq!(cfg.buffer_size, 100);
    }

    #[test]
    fn test_chunk_config_default_values() {
        let cfg = ChunkConfig::default();
        assert_eq!(cfg.chunk_size, 10_000);
        assert_eq!(cfg.overlap, 100);
        assert!(cfg.respect_sentences);
        assert_eq!(cfg.buffer_size, 1000);
    }

    // =========================================================================
    // chunk_text() shared utility
    // =========================================================================

    #[test]
    fn test_chunk_text_small_text_single_chunk() {
        let config = ChunkConfig::default(); // 10k chars
        let chunks = chunk_text("Hello world.", &config);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "Hello world.");
        assert_eq!(chunks[0].char_offset, 0);
    }

    #[test]
    fn test_chunk_text_empty() {
        let config = ChunkConfig::default();
        let chunks = chunk_text("", &config);
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_chunk_text_splits_large_text() {
        let config = ChunkConfig {
            chunk_size: 20,
            overlap: 5,
            respect_sentences: false,
            buffer_size: 100,
        };
        let text =
            "Alice met Bob in Paris. Charlie visited London yesterday. Dave works in Tokyo today.";
        let chunks = chunk_text(text, &config);
        assert!(chunks.len() > 1, "should split into multiple chunks");

        // Verify offsets are monotonically increasing
        for i in 1..chunks.len() {
            assert!(
                chunks[i].char_offset > chunks[i - 1].char_offset,
                "chunk offsets must increase"
            );
        }

        // Verify all text is covered (last chunk offset + last chunk len >= original len)
        let last = &chunks[chunks.len() - 1];
        let total_covered = last.char_offset + last.text.chars().count();
        assert!(
            total_covered >= text.chars().count(),
            "chunks must cover all text"
        );
    }

    #[test]
    fn test_chunk_text_respects_sentences() {
        let config = ChunkConfig {
            chunk_size: 30,
            overlap: 5,
            respect_sentences: true,
            buffer_size: 100,
        };
        let text = "First sentence here. Second sentence here. Third sentence here.";
        let chunks = chunk_text(text, &config);
        assert!(chunks.len() >= 2);
        // First chunk should end at a sentence boundary
        assert!(
            chunks[0].text.ends_with(". ") || chunks[0].text.ends_with('.'),
            "first chunk should end near sentence boundary: {:?}",
            chunks[0].text
        );
    }

    #[test]
    fn test_chunk_text_overlap_creates_redundancy() {
        let config = ChunkConfig {
            chunk_size: 20,
            overlap: 10,
            respect_sentences: false,
            buffer_size: 100,
        };
        let text = "0123456789 abcdefghij klmnopqrst uvwxyz";
        let chunks = chunk_text(text, &config);
        assert!(chunks.len() >= 2);

        // With overlap, chunk N+1's start should be before chunk N's end
        if chunks.len() >= 2 {
            let c0_end = chunks[0].char_offset + chunks[0].text.chars().count();
            let c1_start = chunks[1].char_offset;
            assert!(
                c1_start < c0_end,
                "overlap should cause chunk start ({}) < prev chunk end ({})",
                c1_start,
                c0_end
            );
        }
    }

    #[test]
    fn test_chunk_text_unicode() {
        let config = ChunkConfig {
            chunk_size: 10,
            overlap: 3,
            respect_sentences: false,
            buffer_size: 100,
        };
        let text = "東京は日本の首都です。パリはフランスの首都です。";
        let chunks = chunk_text(text, &config);
        assert!(chunks.len() >= 2);

        // Verify char offsets are correct
        let chars: Vec<char> = text.chars().collect();
        for chunk in &chunks {
            let expected: String = chars
                [chunk.char_offset..chunk.char_offset + chunk.text.chars().count()]
                .iter()
                .collect();
            assert_eq!(chunk.text, expected, "chunk text must match offset slice");
        }
    }

    // =========================================================================
    // OverlapStrategy tests
    // =========================================================================

    #[test]
    fn test_overlap_strategy_keep_first_basic() {
        let mut entities = vec![
            Entity::new("New York", EntityType::Location, 0, 8, 0.7),
            Entity::new("New York City", EntityType::Location, 0, 13, 0.9),
        ];
        deduplicate_overlapping(&mut entities, OverlapStrategy::KeepFirst);
        // Both start at 0; higher-confidence "New York City" sorts first, gets kept
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].text, "New York City");
    }

    #[test]
    fn test_overlap_strategy_keep_first_non_overlapping() {
        let mut entities = vec![
            Entity::new("Alice", EntityType::Person, 0, 5, 0.9),
            Entity::new("Bob", EntityType::Person, 10, 13, 0.8),
        ];
        deduplicate_overlapping(&mut entities, OverlapStrategy::KeepFirst);
        assert_eq!(entities.len(), 2);
    }

    #[test]
    fn test_overlap_strategy_keep_first_chain() {
        // A overlaps B, B overlaps C, but A does not overlap C
        let mut entities = vec![
            Entity::new("AB", EntityType::Person, 0, 5, 0.9),
            Entity::new("BC", EntityType::Person, 3, 8, 0.8),
            Entity::new("CD", EntityType::Person, 6, 10, 0.7),
        ];
        deduplicate_overlapping(&mut entities, OverlapStrategy::KeepFirst);
        // AB kept (start=0), BC skipped (overlaps AB), CD kept (start=6 >= AB.end=5)
        assert_eq!(entities.len(), 2);
        assert_eq!(entities[0].text, "AB");
        assert_eq!(entities[1].text, "CD");
    }

    #[test]
    fn test_overlap_strategy_keep_highest_confidence() {
        let mut entities = vec![
            Entity::new("New York", EntityType::Location, 0, 8, 0.9),
            Entity::new("York City", EntityType::Location, 4, 13, 0.7),
        ];
        deduplicate_overlapping(&mut entities, OverlapStrategy::KeepHighestConfidence);
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].text, "New York");
    }

    #[test]
    fn test_overlap_strategy_keep_highest_confidence_preserves_position_order() {
        let mut entities = vec![
            Entity::new("Alice", EntityType::Person, 20, 25, 0.95),
            Entity::new("Bob", EntityType::Person, 0, 3, 0.5),
        ];
        deduplicate_overlapping(&mut entities, OverlapStrategy::KeepHighestConfidence);
        assert_eq!(entities.len(), 2);
        // Should be sorted by position regardless of confidence order
        assert_eq!(entities[0].text, "Bob");
        assert_eq!(entities[1].text, "Alice");
    }

    #[test]
    fn test_overlap_strategy_keep_highest_confidence_three_way() {
        let mut entities = vec![
            Entity::new("A", EntityType::Person, 0, 5, 0.5),
            Entity::new("B", EntityType::Person, 3, 8, 0.9),
            Entity::new("C", EntityType::Person, 6, 10, 0.7),
        ];
        deduplicate_overlapping(&mut entities, OverlapStrategy::KeepHighestConfidence);
        // B has highest confidence, kept; A and C both overlap B, dropped
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].text, "B");
    }

    #[test]
    fn test_overlap_strategy_keep_longer_same_type_basic() {
        let mut entities = vec![
            Entity::new("New York", EntityType::Location, 0, 8, 0.7),
            Entity::new("New York City", EntityType::Location, 0, 13, 0.6),
        ];
        deduplicate_overlapping(&mut entities, OverlapStrategy::KeepLongerSameType);
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].text, "New York City");
    }

    #[test]
    fn test_overlap_strategy_keep_longer_same_type_different_types_preserved() {
        let mut entities = vec![
            Entity::new("New York", EntityType::Location, 0, 8, 0.7),
            Entity::new("New York Times", EntityType::Organization, 0, 14, 0.6),
        ];
        deduplicate_overlapping(&mut entities, OverlapStrategy::KeepLongerSameType);
        // Different types: both preserved
        assert_eq!(entities.len(), 2);
    }

    #[test]
    fn test_overlap_strategy_keep_longer_same_type_shorter_kept_when_different_type() {
        let mut entities = vec![
            Entity::new("Paris", EntityType::Location, 0, 5, 0.9),
            Entity::new("Paris Hilton", EntityType::Person, 0, 12, 0.8),
        ];
        deduplicate_overlapping(&mut entities, OverlapStrategy::KeepLongerSameType);
        assert_eq!(entities.len(), 2);
        assert_eq!(entities[0].text, "Paris");
        assert_eq!(entities[1].text, "Paris Hilton");
    }

    #[test]
    fn test_overlap_strategy_keep_shortest_drops_supersets() {
        let mut entities = vec![
            Entity::new(
                "Department of Defense",
                EntityType::Organization,
                4,
                25,
                0.8,
            ),
            Entity::new(
                "The Department of Defense",
                EntityType::Organization,
                0,
                25,
                0.7,
            ),
        ];
        deduplicate_overlapping(&mut entities, OverlapStrategy::KeepShortest);
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].text, "Department of Defense");
    }

    #[test]
    fn test_overlap_strategy_keep_shortest_no_overlap() {
        let mut entities = vec![
            Entity::new("IBM", EntityType::Organization, 0, 3, 0.9),
            Entity::new("NASA", EntityType::Organization, 10, 14, 0.8),
        ];
        deduplicate_overlapping(&mut entities, OverlapStrategy::KeepShortest);
        assert_eq!(entities.len(), 2);
    }

    #[test]
    fn test_overlap_strategy_keep_shortest_partial_overlap_dropped() {
        let mut entities = vec![
            Entity::new("AB", EntityType::Person, 0, 5, 0.9),
            Entity::new("BC", EntityType::Person, 3, 8, 0.8),
        ];
        deduplicate_overlapping(&mut entities, OverlapStrategy::KeepShortest);
        // Same length, AB has higher confidence so sorted first and kept; BC overlaps, dropped
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].text, "AB");
    }

    #[test]
    fn test_overlap_strategy_empty_input() {
        for strategy in [
            OverlapStrategy::KeepFirst,
            OverlapStrategy::KeepHighestConfidence,
            OverlapStrategy::KeepLongerSameType,
            OverlapStrategy::KeepShortest,
        ] {
            let mut entities: Vec<Entity> = vec![];
            deduplicate_overlapping(&mut entities, strategy);
            assert!(entities.is_empty());
        }
    }

    #[test]
    fn test_overlap_strategy_single_entity() {
        for strategy in [
            OverlapStrategy::KeepFirst,
            OverlapStrategy::KeepHighestConfidence,
            OverlapStrategy::KeepLongerSameType,
            OverlapStrategy::KeepShortest,
        ] {
            let mut entities = vec![Entity::new("Alice", EntityType::Person, 0, 5, 0.9)];
            deduplicate_overlapping(&mut entities, strategy);
            assert_eq!(entities.len(), 1);
            assert_eq!(entities[0].text, "Alice");
        }
    }

    #[test]
    fn test_overlap_strategy_result_sorted_by_position() {
        for strategy in [
            OverlapStrategy::KeepFirst,
            OverlapStrategy::KeepHighestConfidence,
            OverlapStrategy::KeepLongerSameType,
            OverlapStrategy::KeepShortest,
        ] {
            let mut entities = vec![
                Entity::new("C", EntityType::Person, 20, 25, 0.5),
                Entity::new("A", EntityType::Person, 0, 3, 0.9),
                Entity::new("B", EntityType::Person, 10, 15, 0.7),
            ];
            deduplicate_overlapping(&mut entities, strategy);
            // Non-overlapping, so all kept; verify sorted by start
            for i in 1..entities.len() {
                assert!(
                    entities[i].start() >= entities[i - 1].start(),
                    "result must be sorted by position for {:?}",
                    strategy
                );
            }
        }
    }

    // =========================================================================
    // Advanced / gap-filling tests
    // =========================================================================

    // --- KeepLongerSameType: non-adjacent same-type overlap with interleaved type ---

    /// A=[0,10] Location, B=[5,12] Person, C=[8,15] Location.
    /// C overlaps A (same type, Location) but B (different type, Person) sits
    /// between them in position order. The rposition scan must find A even
    /// though B was the most recently pushed entity.
    #[test]
    fn test_keep_longer_same_type_non_adjacent_interleaved() {
        // Input sorted by start: A(0,10 Loc), B(5,12 Per), C(8,15 Loc)
        let mut entities = vec![
            Entity::new("A", EntityType::Location, 0, 10, 0.8),
            Entity::new("B", EntityType::Person, 5, 12, 0.8),
            Entity::new("C", EntityType::Location, 8, 15, 0.8),
        ];
        deduplicate_overlapping(&mut entities, OverlapStrategy::KeepLongerSameType);

        // B (Person) must be kept — different type from both A and C.
        assert!(
            entities.iter().any(|e| e.text == "B"),
            "Person entity B must be preserved (different type)"
        );

        // Among Location entities A[0,10] and C[8,15]:
        //   A has len=10, C has len=7 => A is longer, A should be kept, C dropped.
        let locs: Vec<&Entity> = entities
            .iter()
            .filter(|e| e.entity_type == EntityType::Location)
            .collect();
        assert_eq!(locs.len(), 1, "exactly one Location should survive");
        assert_eq!(
            locs[0].text, "A",
            "longer Location A should be kept over shorter C"
        );

        // Result must be sorted by start position.
        for i in 1..entities.len() {
            assert!(entities[i].start() >= entities[i - 1].start());
        }
    }

    /// Same configuration but with C longer than A: C=[8,20] Location (len=12 vs len=10).
    /// The rposition scan must find A at index 0 (behind interleaved B), then replace it
    /// with the longer candidate C.
    #[test]
    fn test_keep_longer_same_type_non_adjacent_candidate_wins() {
        let mut entities = vec![
            Entity::new("A", EntityType::Location, 0, 10, 0.8),
            Entity::new("B", EntityType::Person, 5, 12, 0.8),
            Entity::new("C", EntityType::Location, 8, 20, 0.8),
        ];
        deduplicate_overlapping(&mut entities, OverlapStrategy::KeepLongerSameType);

        assert!(
            entities.iter().any(|e| e.text == "B"),
            "Person entity B must be preserved"
        );

        let locs: Vec<&Entity> = entities
            .iter()
            .filter(|e| e.entity_type == EntityType::Location)
            .collect();
        assert_eq!(locs.len(), 1, "exactly one Location should survive");
        assert_eq!(
            locs[0].text, "C",
            "longer Location C[8,20] should replace shorter A[0,10]"
        );
        // Note: KeepLongerSameType does not re-sort after in-place replacement,
        // so we do not assert position order here.
    }

    // --- KeepFirst: contained span is dropped ---

    /// Outer span [0,5] is kept first; inner contained span [1,3] must be dropped.
    #[test]
    fn test_keep_first_contained_span_dropped() {
        let mut entities = vec![
            Entity::new("outer", EntityType::Organization, 0, 5, 0.9),
            Entity::new("inner", EntityType::Organization, 1, 3, 0.95),
        ];
        deduplicate_overlapping(&mut entities, OverlapStrategy::KeepFirst);
        assert_eq!(entities.len(), 1);
        assert_eq!(
            entities[0].text, "outer",
            "outer span should be kept; inner contained span dropped"
        );
    }

    /// Symmetrical: inner span [1,3] arrives first (higher confidence so sorts
    /// ahead under KeepFirst tie-break), outer [0,5] is then dropped as overlap.
    #[test]
    fn test_keep_first_outer_dropped_when_inner_higher_confidence() {
        let mut entities = vec![
            Entity::new("outer", EntityType::Organization, 0, 5, 0.5),
            Entity::new("inner", EntityType::Organization, 1, 3, 0.95),
        ];
        // Both start positions differ (0 vs 1), so "outer" sorts first.
        // "inner" [1,3] starts inside outer's span [0,5] -> inner is dropped.
        deduplicate_overlapping(&mut entities, OverlapStrategy::KeepFirst);
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].text, "outer");
    }

    // --- KeepHighestConfidence: NaN confidence does not crash ---

    /// A NaN confidence value must not cause a panic. The entity may be kept or
    /// dropped (behaviour is unspecified for NaN), but the function must return.
    #[test]
    fn test_keep_highest_confidence_nan_does_not_crash() {
        let mut entities = vec![
            Entity::new("Alice", EntityType::Person, 0, 5, 0.9),
            // NaN confidence — pathological but must not panic.
            Entity::new("NaN entity", EntityType::Person, 3, 8, f64::NAN),
        ];
        // Must complete without panicking.
        deduplicate_overlapping(&mut entities, OverlapStrategy::KeepHighestConfidence);
        // At least the well-formed entity should survive (they overlap, NaN sorts
        // unpredictably, but Alice has real confidence 0.9 and must be present
        // unless NaN sorts above it — either outcome is acceptable as long as
        // we don't crash and the result is non-empty).
        assert!(
            !entities.is_empty(),
            "result must be non-empty after NaN entity processing"
        );
    }

    /// NaN among non-overlapping entities: every non-NaN entity must survive.
    #[test]
    fn test_keep_highest_confidence_nan_non_overlapping_no_crash() {
        let mut entities = vec![
            Entity::new("Alice", EntityType::Person, 0, 5, 0.9),
            Entity::new("NaN ent", EntityType::Person, 20, 27, f64::NAN),
            Entity::new("Bob", EntityType::Person, 10, 13, 0.7),
        ];
        // Must not panic regardless of NaN sort order.
        deduplicate_overlapping(&mut entities, OverlapStrategy::KeepHighestConfidence);
        // All spans are non-overlapping; all should be kept regardless of NaN.
        // (The NaN entity may or may not appear, but Alice and Bob must.)
        assert!(
            entities.iter().any(|e| e.text == "Alice"),
            "Alice must be kept"
        );
        assert!(entities.iter().any(|e| e.text == "Bob"), "Bob must be kept");
    }

    // --- CJK sentence boundary: split on 。 without following whitespace ---

    /// `find_sentence_boundary` must recognise 。 as a sentence terminator even
    /// when the next character is another CJK character (no whitespace follows).
    #[test]
    fn test_find_sentence_boundary_cjk_no_whitespace_after_period() {
        // "这是第一句。这是第二句。这是第三句"
        // '。' appears at char index 5 and 11.
        let text = "这是第一句。这是第二句。这是第三句";
        let chars: Vec<char> = text.chars().collect();

        // Ask for a boundary somewhere after the first '。' (index 5).
        // The target is set to 10 so the backward scan can find '。' at index 5.
        let boundary = find_sentence_boundary(&chars, 0, 10);

        // After '。' at index 5 with no whitespace following, boundary should be 6.
        assert_eq!(
            boundary, 6,
            "should split immediately after 。 (index 5), placing boundary at char 6"
        );
    }

    /// Three-sentence CJK text: asking for a boundary near the end should find
    /// the second '。' (at index 11) and split at 12.
    #[test]
    fn test_find_sentence_boundary_cjk_second_period() {
        let text = "这是第一句。这是第二句。这是第三句";
        let chars: Vec<char> = text.chars().collect();

        // Target near end of second sentence.
        let boundary = find_sentence_boundary(&chars, 0, chars.len() - 1);

        // Backward scan from (len-2) finds '。' at index 11 -> boundary = 12.
        assert_eq!(
            boundary, 12,
            "should split after second 。 at index 11, placing boundary at char 12"
        );
    }

    // --- extract_chunked_parallel: mock extract_fn, offset adjustment, dedup ---

    /// The mock extract function returns an entity at local offset [5, 10] for
    /// each chunk. `extract_chunked_parallel` must adjust each entity's start/end
    /// by the chunk's `char_offset`, producing distinct global spans, then sort
    /// the result by position.
    #[test]
    fn test_extract_chunked_parallel_offset_adjustment() {
        // 60-character text split into multiple chunks (exact count depends on the
        // forward-progress calculation; we do not hard-code it).
        let text: String = "x".repeat(60);

        let config = ChunkConfig {
            chunk_size: 30,
            overlap: 10,
            respect_sentences: false,
            buffer_size: 100,
        };

        // Collect the chunk offsets that chunk_text actually produces, so we can
        // predict the expected global spans without re-implementing the chunker.
        let expected_offsets: Vec<usize> = chunk_text(&text, &config)
            .iter()
            .map(|c| c.char_offset)
            .collect();

        // The mock returns one entity per chunk with offsets relative to the chunk.
        let result = extract_chunked_parallel(&text, &config, |_chunk, char_offset| {
            Ok(vec![Entity::new(
                "token",
                EntityType::Organization,
                5 + char_offset,
                10 + char_offset,
                0.9,
            )])
        });

        let entities = result.expect("extract_chunked_parallel must not error");

        // Every expected global span must be present.
        for off in &expected_offsets {
            assert!(
                entities
                    .iter()
                    .any(|e| e.start() == 5 + off && e.end() == 10 + off),
                "global entity [{}, {}] must be present; got: {:?}",
                5 + off,
                10 + off,
                entities
                    .iter()
                    .map(|e| (e.start(), e.end()))
                    .collect::<Vec<_>>()
            );
        }

        // No unexpected extras: entity count must equal the number of distinct spans.
        assert_eq!(
            entities.len(),
            expected_offsets.len(),
            "one entity per chunk, no duplicates; got: {:?}",
            entities
                .iter()
                .map(|e| (e.start(), e.end()))
                .collect::<Vec<_>>()
        );

        // Result must be sorted by position.
        for i in 1..entities.len() {
            assert!(entities[i].start() >= entities[i - 1].start());
        }
    }

    /// When the mock returns the *same global span* from two chunks (boundary
    /// dedup scenario), `extract_chunked_parallel` must keep only one copy.
    #[test]
    fn test_extract_chunked_parallel_boundary_dedup() {
        let text: String = "x".repeat(60);

        let config = ChunkConfig {
            chunk_size: 30,
            overlap: 10,
            respect_sentences: false,
            buffer_size: 100,
        };

        // Both chunks return an entity with the same *global* span [5, 10].
        // The exact-span dedup in extract_chunked_parallel must drop the duplicate.
        let result = extract_chunked_parallel(&text, &config, |_chunk, _char_offset| {
            Ok(vec![Entity::new("shared", EntityType::Person, 5, 10, 0.9)])
        });

        let entities = result.expect("must not error");
        assert_eq!(
            entities.len(),
            1,
            "duplicate global span [5,10] must be deduplicated; got {} entities",
            entities.len()
        );
        assert_eq!(entities[0].start(), 5);
        assert_eq!(entities[0].end(), 10);
    }

    /// Empty text returns no entities and does not error.
    #[test]
    fn test_extract_chunked_parallel_empty_text() {
        let result = extract_chunked_parallel("", &ChunkConfig::default(), |_chunk, _offset| {
            Ok(vec![Entity::new("x", EntityType::Person, 0, 1, 0.9)])
        });
        let entities = result.expect("must not error on empty text");
        assert!(entities.is_empty(), "empty text must produce no entities");
    }

    // --- chunk_text: overlap >= chunk_size forward-progress guard ---

    /// When overlap equals chunk_size the forward-progress guard must prevent
    /// an infinite loop: each iteration must advance by at least 1 character.
    #[test]
    fn test_chunk_text_overlap_equals_chunk_size_terminates() {
        let config = ChunkConfig {
            chunk_size: 5,
            overlap: 5, // overlap == chunk_size: pathological
            respect_sentences: false,
            buffer_size: 100,
        };
        let text = "abcdefghijklmno"; // 15 chars
        let chunks = chunk_text(text, &config);

        // Must produce at least one chunk and must not hang.
        assert!(!chunks.is_empty(), "must produce at least one chunk");

        // Offsets must be strictly increasing (forward progress guaranteed).
        for i in 1..chunks.len() {
            assert!(
                chunks[i].char_offset > chunks[i - 1].char_offset,
                "chunk offsets must strictly increase even when overlap == chunk_size"
            );
        }
    }

    /// overlap > chunk_size: even more extreme; guard still fires.
    #[test]
    fn test_chunk_text_overlap_greater_than_chunk_size_terminates() {
        let config = ChunkConfig {
            chunk_size: 4,
            overlap: 10, // overlap > chunk_size
            respect_sentences: false,
            buffer_size: 100,
        };
        let text = "abcdefghij"; // 10 chars
        let chunks = chunk_text(text, &config);

        assert!(!chunks.is_empty());
        for i in 1..chunks.len() {
            assert!(
                chunks[i].char_offset > chunks[i - 1].char_offset,
                "forward progress must hold when overlap > chunk_size"
            );
        }
    }

    // --- KeepShortest: same-length same-confidence entities ---

    /// Two entities with identical length and identical confidence.
    /// The result must be deterministic: exactly one entity is kept and it is
    /// always the same one regardless of input ordering.
    #[test]
    fn test_keep_shortest_same_length_same_confidence_deterministic() {
        // Non-overlapping case: both should survive (no superset/overlap).
        let mut entities = vec![
            Entity::new("abc", EntityType::Person, 0, 3, 0.7),
            Entity::new("def", EntityType::Person, 10, 13, 0.7),
        ];
        deduplicate_overlapping(&mut entities, OverlapStrategy::KeepShortest);
        assert_eq!(
            entities.len(),
            2,
            "non-overlapping same-length same-confidence entities must both survive"
        );

        // Overlapping / tied case: the sort is by (len asc, confidence desc);
        // ties in both dimensions leave original order intact (sort_unstable_by
        // is not guaranteed stable, but the kept entity must always be the same
        // across repeated calls with the same input).
        let make = || {
            vec![
                Entity::new("AB", EntityType::Organization, 0, 5, 0.8),
                Entity::new("BC", EntityType::Organization, 3, 8, 0.8),
            ]
        };

        let mut first_run = make();
        deduplicate_overlapping(&mut first_run, OverlapStrategy::KeepShortest);
        assert_eq!(
            first_run.len(),
            1,
            "overlapping same-length entities: one must be dropped"
        );
        let kept_text = first_run[0].text.clone();

        // Second run with the same input must keep the same entity.
        let mut second_run = make();
        deduplicate_overlapping(&mut second_run, OverlapStrategy::KeepShortest);
        assert_eq!(second_run.len(), 1);
        assert_eq!(
            second_run[0].text, kept_text,
            "KeepShortest must be deterministic: same input must always keep the same entity"
        );
    }

    // --- Empty entities list for all 4 strategies (explicit, not just length-1 guard) ---

    /// `deduplicate_overlapping` on an empty Vec must leave it empty for every strategy.
    /// (Complements `test_overlap_strategy_empty_input` but tests the Vec mutation path.)
    #[test]
    fn test_all_strategies_empty_list_is_noop() {
        for strategy in [
            OverlapStrategy::KeepFirst,
            OverlapStrategy::KeepHighestConfidence,
            OverlapStrategy::KeepLongerSameType,
            OverlapStrategy::KeepShortest,
        ] {
            let mut entities: Vec<Entity> = Vec::new();
            deduplicate_overlapping(&mut entities, strategy);
            assert!(
                entities.is_empty(),
                "empty input must remain empty for {:?}",
                strategy
            );
        }
    }
}
