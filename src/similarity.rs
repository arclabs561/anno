//! Text similarity utilities for entity matching and coreference resolution.
//!
//! Provides unified similarity computation functions used across
//! cross-document coreference, entity linking, and clustering operations.

use std::collections::HashSet;

/// Compute string similarity using multiple strategies.
///
/// Returns a value in [0.0, 1.0] where:
/// - 1.0 = identical strings
/// - 0.8 = substring match (one contains the other)
/// - 0.0-0.8 = Jaccard similarity on word sets
///
/// # Algorithm
///
/// 1. **Exact match** (after lowercasing): Returns 1.0
/// 2. **Substring match**: Returns 0.8 if one string contains the other
/// 3. **Jaccard similarity**: Word-level Jaccard coefficient
///
/// # Examples
///
/// ```
/// use anno::similarity::string_similarity;
///
/// assert!((string_similarity("Apple", "Apple") - 1.0).abs() < 0.001);
/// assert!(string_similarity("Apple Inc", "Apple") > 0.5);
/// assert!(string_similarity("Apple", "Microsoft") < 0.5);
/// ```
#[must_use]
pub fn string_similarity(a: &str, b: &str) -> f64 {
    let a_lower = a.to_lowercase();
    let b_lower = b.to_lowercase();

    // Exact match
    if a_lower == b_lower {
        return 1.0;
    }

    // Substring match
    if a_lower.contains(&b_lower) || b_lower.contains(&a_lower) {
        return 0.8;
    }

    // Jaccard similarity on words
    jaccard_word_similarity(&a_lower, &b_lower)
}

/// Compute Jaccard similarity on word sets.
///
/// Splits strings by whitespace and computes the Jaccard coefficient
/// of the resulting word sets.
///
/// # Examples
///
/// ```
/// use anno::similarity::jaccard_word_similarity;
///
/// // "Apple Inc" and "Apple" share 1 word, union has 2 words → 0.5
/// let sim = jaccard_word_similarity("apple inc", "apple");
/// assert!((sim - 0.5).abs() < 0.001);
/// ```
#[must_use]
pub fn jaccard_word_similarity(a: &str, b: &str) -> f64 {
    let words_a: HashSet<&str> = a.split_whitespace().collect();
    let words_b: HashSet<&str> = b.split_whitespace().collect();

    let intersection = words_a.intersection(&words_b).count();
    let union = words_a.union(&words_b).count();

    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

/// Compute Jaccard similarity on word sets (f32 version).
///
/// Same as `jaccard_word_similarity` but returns f32 for compatibility
/// with existing code that uses f32.
#[must_use]
pub fn jaccard_word_similarity_f32(a: &str, b: &str) -> f32 {
    jaccard_word_similarity(a, b) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_similarity_identical() {
        assert!((string_similarity("Apple", "Apple") - 1.0).abs() < 0.001);
        assert!((string_similarity("Apple", "apple") - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_string_similarity_substring() {
        let sim = string_similarity("Apple Inc", "Apple");
        assert!((sim - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_string_similarity_jaccard() {
        // "Apple Inc" and "Apple Corporation" share "Apple", union has 3 words
        let sim = string_similarity("Apple Inc", "Apple Corporation");
        assert!(sim > 0.3 && sim < 0.8); // Should be Jaccard, not substring
    }

    #[test]
    fn test_string_similarity_empty() {
        assert_eq!(string_similarity("", ""), 1.0); // Exact match
                                                    // Empty string is substring of any string, so returns 0.8
        assert!((string_similarity("Apple", "") - 0.8).abs() < 0.001);
        assert!((string_similarity("", "Apple") - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_jaccard_word_similarity() {
        // "apple inc" and "apple" → intersection=1, union=2 → 0.5
        let sim = jaccard_word_similarity("apple inc", "apple");
        assert!((sim - 0.5).abs() < 0.001);

        // "apple inc" and "microsoft" → intersection=0, union=3 → 0.0
        let sim = jaccard_word_similarity("apple inc", "microsoft");
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_jaccard_word_similarity_f32() {
        let sim = jaccard_word_similarity_f32("apple inc", "apple");
        assert!((sim - 0.5).abs() < 0.001);
    }
}
