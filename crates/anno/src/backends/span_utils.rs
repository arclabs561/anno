//! Shared span tensor utilities for GLiNER-family models.
//!
//! GLiNER and NuNER both use span-based architectures where the model scores
//! every possible span (up to `MAX_SPAN_WIDTH` words) against each entity type.
//! This module provides common utilities for generating span tensors and
//! decoding span-based outputs.
//!
//! # Architecture Overview
//!
//! ```text
//! Input Text: "Steve Jobs founded Apple"
//!             [0]    [1]     [2]    [3]
//!
//! Span Grid (MAX_SPAN_WIDTH=3):
//!
//!   Start=0: (0,0) (0,1) (0,2)  → "Steve", "Steve Jobs", "Steve Jobs founded"
//!   Start=1: (1,1) (1,2) (1,3)  → "Jobs", "Jobs founded", "Jobs founded Apple"
//!   Start=2: (2,2) (2,3)        → "founded", "founded Apple"
//!   Start=3: (3,3)              → "Apple"
//!
//! Total spans: sum(min(MAX_WIDTH, remaining_words)) for each start position
//! ```
//!
//! # Span Tensor Format
//!
//! - `span_idx`: `[num_spans, 2]` - (start_word, end_word) indices for each span
//! - `span_mask`: `[num_spans]` - boolean mask indicating valid spans
//!
//! # Output Decoding
//!
//! Model output shape: `[batch, num_words, max_width, num_entity_types]`
//!
//! Each cell `output[0][start][width][type_idx]` contains the score for:
//! - Span starting at word `start`
//! - With width `width + 1` words (i.e., ends at word `start + width`)
//! - Being an entity of type `type_idx`
//!
//! # References
//!
//! - GLiNER paper: "GLiNER: Generalist and Lightweight Model for Named Entity Recognition"
//! - NuNER: Token-based variant using same span representation
//! - Community GLiNER implementations (for span layout conventions)

use crate::{Entity, EntityType, Error, Result};

/// Default maximum span width (in words) for GLiNER-family models.
///
/// This matches the training configuration of most GLiNER/NuNER models.
/// Spans longer than this are not considered by the model.
pub const DEFAULT_MAX_SPAN_WIDTH: usize = 12;

/// Configuration for span-based NER decoding.
#[derive(Debug, Clone)]
pub struct SpanConfig {
    /// Maximum span width in words.
    pub max_span_width: usize,
    /// Confidence threshold for entity extraction.
    pub threshold: f32,
}

impl Default for SpanConfig {
    fn default() -> Self {
        Self {
            max_span_width: DEFAULT_MAX_SPAN_WIDTH,
            threshold: 0.5,
        }
    }
}

/// Generate span tensors for ONNX model input.
///
/// Creates the `span_idx` and `span_mask` tensors required by GLiNER-family models.
///
/// # Arguments
///
/// * `num_words` - Number of words in the input text
/// * `max_width` - Maximum span width to consider
///
/// # Returns
///
/// A tuple of:
/// - `span_idx`: Flattened `[num_spans * 2]` array of (start, end) pairs
/// - `span_mask`: `[num_spans]` boolean mask of valid spans
///
/// # Example
///
/// ```rust
/// use anno::backends::span_utils::make_span_tensors;
///
/// let (span_idx, span_mask) = make_span_tensors(4, 3);
///
/// // First span: (0, 0) -> "word 0"
/// assert_eq!(span_idx[0], 0); // start
/// assert_eq!(span_idx[1], 0); // end (exclusive would be 1, but GLiNER uses inclusive)
/// assert!(span_mask[0]);
/// ```
pub fn make_span_tensors(num_words: usize, max_width: usize) -> (Vec<i64>, Vec<bool>) {
    // Calculate total number of spans with overflow protection
    let num_spans = match num_words.checked_mul(max_width) {
        Some(v) => v,
        None => {
            log::warn!(
                "[span_utils] Span count overflow: {} words * {} max_width, returning empty",
                num_words,
                max_width
            );
            return (Vec::new(), Vec::new());
        }
    };

    let span_idx_len = match num_spans.checked_mul(2) {
        Some(v) => v,
        None => {
            log::warn!(
                "[span_utils] Span idx length overflow: {} * 2, returning empty",
                num_spans
            );
            return (Vec::new(), Vec::new());
        }
    };

    let mut span_idx: Vec<i64> = vec![0; span_idx_len];
    let mut span_mask: Vec<bool> = vec![false; num_spans];

    for start in 0..num_words {
        let remaining_width = num_words - start;
        let actual_max_width = max_width.min(remaining_width);

        for width in 0..actual_max_width {
            // Calculate linear index with overflow protection
            let dim = match start.checked_mul(max_width) {
                Some(v) => match v.checked_add(width) {
                    Some(d) => d,
                    None => continue,
                },
                None => continue,
            };

            // Bounds check before array access
            if let Some(dim2) = dim.checked_mul(2) {
                if dim2 + 1 < span_idx_len && dim < num_spans {
                    span_idx[dim2] = start as i64;
                    // End offset: start + width gives the last word index (inclusive)
                    span_idx[dim2 + 1] = (start + width) as i64;
                    span_mask[dim] = true;
                }
            }
        }
    }

    (span_idx, span_mask)
}

/// Calculate word positions (byte offsets) in the original text.
///
/// Maps word indices to their (start, end) byte positions in the source text.
///
/// # Arguments
///
/// * `text` - The original text
/// * `words` - Whitespace-split words
///
/// # Returns
///
/// A vector of (start_byte, end_byte) positions for each word.
///
/// # Errors
///
/// Returns an error if any word cannot be found at the expected position.
pub fn calculate_word_positions(text: &str, words: &[&str]) -> Result<Vec<(usize, usize)>> {
    let mut positions = Vec::with_capacity(words.len());
    let mut pos = 0;

    for (idx, word) in words.iter().enumerate() {
        // Find word starting from current position
        if let Some(rel_start) = text[pos..].find(word) {
            let abs_start = pos + rel_start;
            let abs_end = abs_start + word.len();

            // Validate: words should be in order
            if !positions.is_empty() {
                let (_prev_start, prev_end) = positions[positions.len() - 1];
                if abs_start < prev_end {
                    log::warn!(
                        "[span_utils] Word '{}' at {} overlaps with previous word ending at {}",
                        word,
                        abs_start,
                        prev_end
                    );
                }
            }

            positions.push((abs_start, abs_end));
            pos = abs_end;
        } else {
            return Err(Error::Parse(format!(
                "Word '{}' (index {}) not found in text starting at position {}",
                word, idx, pos
            )));
        }
    }

    Ok(positions)
}

/// Extract entity span from text given word positions.
///
/// # Arguments
///
/// * `text` - The original text
/// * `word_positions` - Byte positions of each word
/// * `start_word` - Starting word index (inclusive)
/// * `end_word` - Ending word index (inclusive)
///
/// # Returns
///
/// The text span and its byte range, or None if indices are invalid.
pub fn extract_span<'a>(
    text: &'a str,
    word_positions: &[(usize, usize)],
    start_word: usize,
    end_word: usize,
) -> Option<(&'a str, usize, usize)> {
    let start_pos = word_positions.get(start_word)?.0;
    let end_pos = word_positions.get(end_word)?.1;

    if start_pos > end_pos || end_pos > text.len() {
        return None;
    }

    Some((&text[start_pos..end_pos], start_pos, end_pos))
}

/// Decode span-based model output into entities.
///
/// This is the core decoding function for GLiNER-family models.
///
/// # Arguments
///
/// * `output_data` - Flattened model output tensor
/// * `shape` - Output shape `[batch, num_words, max_width, num_classes]`
/// * `text` - Original input text
/// * `text_words` - Whitespace-split words
/// * `entity_types` - Entity type labels
/// * `config` - Decoding configuration
///
/// # Returns
///
/// A vector of extracted entities.
///
/// # Output Format
///
/// The model output has shape `[batch, num_words, max_width, num_classes]`:
/// - `batch`: Always 1 for single-text inference
/// - `num_words`: Number of words in the input
/// - `max_width`: Maximum span width (DEFAULT_MAX_SPAN_WIDTH)
/// - `num_classes`: Number of entity types
///
/// Each cell contains the score for a (start, width, type) triple.
pub fn decode_span_output(
    output_data: &[f32],
    shape: &[i64],
    text: &str,
    text_words: &[&str],
    entity_types: &[&str],
    config: &SpanConfig,
) -> Result<Vec<Entity>> {
    // Validate shape
    if shape.len() < 3 {
        return Err(Error::Parse(format!(
            "Expected at least 3D output, got shape {:?}",
            shape
        )));
    }

    // Parse shape dimensions
    let (out_num_words, out_max_width, num_classes) = if shape.len() == 4 {
        // Standard GLiNER format: [batch, num_words, max_width, num_classes]
        (shape[1] as usize, shape[2] as usize, shape[3] as usize)
    } else if shape.len() == 3 {
        // Squeezed batch dimension: [num_words, max_width, num_classes]
        (shape[0] as usize, shape[1] as usize, shape[2] as usize)
    } else {
        return Err(Error::Parse(format!(
            "Unexpected output shape: {:?}",
            shape
        )));
    };

    log::debug!(
        "[span_utils] Decoding: words={}, max_width={}, classes={}, data_len={}",
        out_num_words,
        out_max_width,
        num_classes,
        output_data.len()
    );

    // Calculate word positions
    let word_positions = calculate_word_positions(text, text_words)?;
    // `calculate_word_positions` (and most tokenizer/regex style tooling) yields byte offsets.
    // `Entity` offsets are defined as character offsets, so convert at construction time.
    let span_converter = crate::offset::SpanConverter::new(text);

    // Validate dimensions match
    let num_text_words = text_words.len();
    if out_num_words < num_text_words {
        log::warn!(
            "[span_utils] Output has fewer words ({}) than input ({})",
            out_num_words,
            num_text_words
        );
    }

    let mut entities = Vec::with_capacity(32);

    // Iterate over all valid spans
    for start in 0..num_text_words.min(out_num_words) {
        for width in 0..config.max_span_width.min(out_max_width) {
            let end = start + width;
            if end >= num_text_words {
                break;
            }

            // Find best entity type for this span
            let base_idx = (start * out_max_width * num_classes) + (width * num_classes);

            let mut best_score = config.threshold;
            let mut best_type_idx = None;

            for type_idx in 0..num_classes.min(entity_types.len()) {
                let score = output_data.get(base_idx + type_idx).copied().unwrap_or(0.0);

                if score > best_score {
                    best_score = score;
                    best_type_idx = Some(type_idx);
                }
            }

            // Create entity if score exceeds threshold
            if let Some(type_idx) = best_type_idx {
                if let Some((span_text, start_byte, end_byte)) =
                    extract_span(text, &word_positions, start, end)
                {
                    let entity_type = map_label_to_entity_type(entity_types[type_idx]);
                    let mut entity = Entity::new(
                        span_text,
                        entity_type,
                        span_converter.byte_to_char(start_byte),
                        span_converter.byte_to_char(end_byte),
                        best_score as f64,
                    );
                    entity.provenance =
                        Some(crate::Provenance::ml("span-decoder", best_score as f64));
                    entities.push(entity);
                }
            }
        }
    }

    // Sort by position and remove overlaps (keep highest confidence)
    entities.sort_by(|a, b| {
        a.start
            .cmp(&b.start)
            // Use total ordering to avoid `partial_cmp` returning None on NaN.
            .then_with(|| b.confidence.total_cmp(&a.confidence))
    });

    // Remove overlapping entities (keep first = highest confidence due to sort)
    let mut filtered = Vec::with_capacity(entities.len());
    for entity in entities {
        let overlaps = filtered
            .iter()
            .any(|e: &Entity| ranges_overlap(e.start, e.end, entity.start, entity.end));
        if !overlaps {
            filtered.push(entity);
        }
    }

    Ok(filtered)
}

/// Check if two ranges overlap.
#[inline]
fn ranges_overlap(start1: usize, end1: usize, start2: usize, end2: usize) -> bool {
    start1 < end2 && start2 < end1
}

/// Map entity type label string to EntityType enum.
///
/// Handles common label variations (case-insensitive).
pub fn map_label_to_entity_type(label: &str) -> EntityType {
    match label.to_lowercase().as_str() {
        "person" | "per" => EntityType::Person,
        "organization" | "org" | "company" | "corp" => EntityType::Organization,
        "location" | "loc" | "place" | "gpe" => EntityType::Location,
        "date" => EntityType::Date,
        "datetime" => EntityType::Date,
        "time" => EntityType::Time,
        "money" | "currency" => EntityType::Money,
        "monetary" => EntityType::Money,
        "percent" | "percentage" => EntityType::Percent,
        "email" => EntityType::Email,
        "phone" => EntityType::Phone,
        "url" => EntityType::Url,
        "quantity" => EntityType::Quantity,
        "measure" => EntityType::Quantity,
        "cardinal" => EntityType::Cardinal,
        "number" | "num" => EntityType::Cardinal,
        "ordinal" => EntityType::Ordinal,
        "event" => EntityType::Other("EVENT".to_string()),
        "product" | "prod" => EntityType::Other("PRODUCT".to_string()),
        "work_of_art" | "work" => EntityType::Other("WORK_OF_ART".to_string()),
        "law" | "legal" => EntityType::Other("LAW".to_string()),
        "language" | "lang" => EntityType::Other("LANGUAGE".to_string()),
        "norp" => EntityType::Other("NORP".to_string()), // Nationalities, religions, political groups
        "fac" | "facility" => EntityType::Other("FACILITY".to_string()),
        // Fine-grained / CNER-inspired labels
        "animal" => EntityType::Other("ANIMAL".to_string()),
        "biology" => EntityType::Other("BIOLOGY".to_string()),
        "celestial" => EntityType::Other("CELESTIAL".to_string()),
        "culture" => EntityType::Other("CULTURE".to_string()),
        "discipline" => EntityType::Other("DISCIPLINE".to_string()),
        "disease" => EntityType::Other("DISEASE".to_string()),
        "feeling" => EntityType::Other("FEELING".to_string()),
        "food" => EntityType::Other("FOOD".to_string()),
        "group" => EntityType::Other("GROUP".to_string()),
        "instrument" => EntityType::Other("INSTRUMENT".to_string()),
        "media" => EntityType::Other("MEDIA".to_string()),
        "asset" => EntityType::Other("ASSET".to_string()),
        "artifact" => EntityType::Other("ARTIFACT".to_string()),
        "part" => EntityType::Other("PART".to_string()),
        "physical_phenomenon" | "physical" => EntityType::Other("PHYSICAL_PHENOMENON".to_string()),
        "plant" => EntityType::Other("PLANT".to_string()),
        "property" => EntityType::Other("PROPERTY".to_string()),
        "psych" => EntityType::Other("PSYCH".to_string()),
        "relation" => EntityType::Other("RELATION".to_string()),
        "struct" => EntityType::Other("STRUCT".to_string()),
        "substance" => EntityType::Other("SUBSTANCE".to_string()),
        "super" | "supernatural" => EntityType::Other("SUPER".to_string()),
        "vehicle" | "vehi" => EntityType::Other("VEHICLE".to_string()),
        _ => EntityType::Other(label.to_uppercase()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_make_span_tensors_basic() {
        let (span_idx, span_mask) = make_span_tensors(3, 2);

        // 3 words * 2 max_width = 6 spans
        assert_eq!(span_mask.len(), 6);
        assert_eq!(span_idx.len(), 12);

        // First span: word 0, width 0 → (0, 0)
        assert!(span_mask[0]);
        assert_eq!(span_idx[0], 0);
        assert_eq!(span_idx[1], 0);

        // Second span: word 0, width 1 → (0, 1)
        assert!(span_mask[1]);
        assert_eq!(span_idx[2], 0);
        assert_eq!(span_idx[3], 1);
    }

    #[test]
    fn test_make_span_tensors_overflow_protection() {
        // Very large input shouldn't panic
        let (span_idx, span_mask) = make_span_tensors(usize::MAX / 2, DEFAULT_MAX_SPAN_WIDTH);
        // Should return empty due to overflow
        assert!(span_idx.is_empty());
        assert!(span_mask.is_empty());
    }

    #[test]
    fn test_calculate_word_positions() {
        let text = "Steve Jobs founded Apple";
        let words: Vec<&str> = text.split_whitespace().collect();

        let positions = calculate_word_positions(text, &words).unwrap();

        assert_eq!(positions.len(), 4);
        assert_eq!(positions[0], (0, 5)); // "Steve"
        assert_eq!(positions[1], (6, 10)); // "Jobs"
        assert_eq!(positions[2], (11, 18)); // "founded"
        assert_eq!(positions[3], (19, 24)); // "Apple"
    }

    #[test]
    fn test_extract_span() {
        let text = "Steve Jobs founded Apple";
        let positions = vec![(0, 5), (6, 10), (11, 18), (19, 24)];

        // Single word span
        let (span, start, end) = extract_span(text, &positions, 0, 0).unwrap();
        assert_eq!(span, "Steve");
        assert_eq!((start, end), (0, 5));

        // Two-word span
        let (span, start, end) = extract_span(text, &positions, 0, 1).unwrap();
        assert_eq!(span, "Steve Jobs");
        assert_eq!((start, end), (0, 10));

        // Three-word span
        let (span, start, end) = extract_span(text, &positions, 1, 3).unwrap();
        assert_eq!(span, "Jobs founded Apple");
        assert_eq!((start, end), (6, 24));
    }

    #[test]
    fn test_map_label_to_entity_type() {
        assert_eq!(map_label_to_entity_type("person"), EntityType::Person);
        assert_eq!(map_label_to_entity_type("PER"), EntityType::Person);
        assert_eq!(
            map_label_to_entity_type("organization"),
            EntityType::Organization
        );
        assert_eq!(map_label_to_entity_type("ORG"), EntityType::Organization);
        assert_eq!(map_label_to_entity_type("location"), EntityType::Location);
        assert_eq!(map_label_to_entity_type("GPE"), EntityType::Location);
        assert_eq!(
            map_label_to_entity_type("custom_type"),
            EntityType::Other("CUSTOM_TYPE".to_string())
        );
    }

    #[test]
    fn test_ranges_overlap() {
        assert!(ranges_overlap(0, 10, 5, 15)); // Partial overlap
        assert!(ranges_overlap(0, 10, 0, 5)); // Contained
        assert!(ranges_overlap(5, 15, 0, 10)); // Partial overlap (reversed)
        assert!(!ranges_overlap(0, 5, 10, 15)); // No overlap
        assert!(!ranges_overlap(0, 5, 5, 10)); // Adjacent (not overlapping)
    }
}
