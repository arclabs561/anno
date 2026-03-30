//! Text similarity for entity matching and coreference resolution.

/// Compute string similarity using multiple strategies.
///
/// Returns a value in [0.0, 1.0] where:
/// - 1.0 = identical strings (case-insensitive)
/// - 0.8 = substring match (one contains the other)
/// - 0.0-0.8 = Jaccard similarity on word sets
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
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }

    let a_lower = a.to_lowercase();
    let b_lower = b.to_lowercase();

    if a_lower == b_lower {
        return 1.0;
    }

    if a_lower.contains(&b_lower) || b_lower.contains(&a_lower) {
        return 0.8;
    }

    textprep::similarity::word_jaccard(&a_lower, &b_lower)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical() {
        assert!((string_similarity("Apple", "Apple") - 1.0).abs() < 0.001);
        assert!((string_similarity("Apple", "apple") - 1.0).abs() < 0.001);
    }

    #[test]
    fn substring() {
        let sim = string_similarity("Apple Inc", "Apple");
        assert!((sim - 0.8).abs() < 0.001);
    }

    #[test]
    fn jaccard() {
        let sim = string_similarity("Apple Inc", "Apple Corporation");
        assert!(sim > 0.3 && sim < 0.8);
    }

    #[test]
    fn empty() {
        assert_eq!(string_similarity("", ""), 1.0);
        assert_eq!(string_similarity("Apple", ""), 0.0);
        assert_eq!(string_similarity("", "Apple"), 0.0);
    }
}
