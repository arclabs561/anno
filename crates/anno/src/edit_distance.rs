//! Edit distance utilities for fuzzy string matching.
//!
//! Provides edit distance algorithms for entity linking, fuzzy matching,
//! and handling damaged/OCR'd historical text.
//!
//! # Research Context
//!
//! Edit distance is fundamental to computational philology and ancient language
//! processing. The implementations here draw from:
//!
//! - **Levenshtein (1966)**: Classic edit distance for typo correction
//! - **Li & Liu (2007)**: Normalized edit distance with metric properties
//! - **Tamburini (2025)**: Edit distance with wildcards for damaged inscriptions
//!
//! The wildcard variant is particularly useful for ancient texts where:
//! - Characters may be illegible due to damage
//! - OCR/HTR may produce uncertain readings
//! - Scribal variations exist within the same text
//!
//! # Unicode Handling
//!
//! All functions operate on Unicode **characters**, not bytes. This is critical
//! for multilingual support:
//!
//! ```rust
//! use anno::edit_distance::levenshtein;
//!
//! // CJK: Each character is one unit
//! assert_eq!(levenshtein("北京", "北平"), 1);
//!
//! // Arabic: Character count, not byte count
//! assert_eq!(levenshtein("محمد", "أحمد"), 1);
//! ```
//!
//! # Example
//!
//! ```rust
//! use anno::edit_distance::{levenshtein, normalized_edit_distance, edit_distance_wildcards};
//!
//! // Basic typo detection
//! assert_eq!(levenshtein("Einstein", "Einstien"), 2);
//!
//! // Normalized for comparing strings of different lengths
//! let sim = 1.0 - normalized_edit_distance("Einstein", "Einstien");
//! assert!(sim > 0.7);
//!
//! // Wildcards for damaged text: "?" = 1 char, "*" = 0+ chars
//! assert_eq!(edit_distance_wildcards("R?ma", "Roma"), 0);
//! assert_eq!(edit_distance_wildcards("Ein*", "Einstein"), 0);
//! ```

use std::cmp::min;

// =============================================================================
// Basic Levenshtein Distance
// =============================================================================

/// Compute Levenshtein edit distance between two strings.
///
/// Returns the minimum number of single-character edits (insertions,
/// deletions, substitutions) required to transform `a` into `b`.
///
/// # Unicode
///
/// Operates on Unicode characters, not bytes. This is critical for
/// multilingual text where characters may be multi-byte.
///
/// # Complexity
///
/// - Time: O(|a| × |b|)
/// - Space: O(min(|a|, |b|)) using the optimized single-row algorithm
///
/// # Examples
///
/// ```rust
/// use anno::edit_distance::levenshtein;
///
/// assert_eq!(levenshtein("kitten", "sitting"), 3);
/// assert_eq!(levenshtein("", "abc"), 3);
/// assert_eq!(levenshtein("abc", "abc"), 0);
///
/// // CJK characters
/// assert_eq!(levenshtein("東京", "東京都"), 1);
/// ```
#[must_use]
pub fn levenshtein(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();

    levenshtein_chars(&a_chars, &b_chars)
}

/// Levenshtein distance on character slices (internal, reusable).
fn levenshtein_chars(a: &[char], b: &[char]) -> usize {
    let m = a.len();
    let n = b.len();

    // Early termination for empty strings
    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    // Use smaller string as the "column" to minimize space
    let (a, b, m, n) = if m > n { (b, a, n, m) } else { (a, b, m, n) };

    // Single-row optimization: only keep current and previous row
    let mut prev_row: Vec<usize> = (0..=m).collect();
    let mut curr_row: Vec<usize> = vec![0; m + 1];

    for j in 1..=n {
        curr_row[0] = j;

        for i in 1..=m {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };

            curr_row[i] = min(
                min(
                    prev_row[i] + 1,     // deletion
                    curr_row[i - 1] + 1, // insertion
                ),
                prev_row[i - 1] + cost, // substitution
            );
        }

        std::mem::swap(&mut prev_row, &mut curr_row);
    }

    prev_row[m]
}

// =============================================================================
// Weighted Edit Distance
// =============================================================================

/// Edit distance operation weights.
///
/// Allows customizing the cost of different edit operations.
/// Default weights are all 1.0 (standard Levenshtein).
#[derive(Debug, Clone, Copy)]
pub struct EditWeights {
    /// Cost of inserting a character
    pub insert: f64,
    /// Cost of deleting a character
    pub delete: f64,
    /// Cost of substituting a character
    pub substitute: f64,
}

impl Default for EditWeights {
    fn default() -> Self {
        Self {
            insert: 1.0,
            delete: 1.0,
            substitute: 1.0,
        }
    }
}

/// Compute weighted Levenshtein distance.
///
/// Allows different costs for insert/delete/substitute operations.
#[must_use]
pub fn weighted_levenshtein(a: &str, b: &str, weights: EditWeights) -> f64 {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();

    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 {
        return n as f64 * weights.insert;
    }
    if n == 0 {
        return m as f64 * weights.delete;
    }

    let mut prev_row: Vec<f64> = (0..=m).map(|i| i as f64 * weights.delete).collect();
    let mut curr_row: Vec<f64> = vec![0.0; m + 1];

    for j in 1..=n {
        curr_row[0] = j as f64 * weights.insert;

        for i in 1..=m {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0.0
            } else {
                weights.substitute
            };

            curr_row[i] = f64::min(
                f64::min(
                    prev_row[i] + weights.insert,     // Insert char from b
                    curr_row[i - 1] + weights.delete, // Delete char from a
                ),
                prev_row[i - 1] + cost,
            );
        }

        std::mem::swap(&mut prev_row, &mut curr_row);
    }

    prev_row[m]
}

// =============================================================================
// Normalized Edit Distance
// =============================================================================

/// Compute normalized edit distance (Li & Liu 2007).
///
/// Returns a value in [0.0, 1.0] where:
/// - 0.0 = identical strings
/// - 1.0 = maximally different
///
/// Formula: `2 * ED(a,b) / (|a| + |b| + ED(a,b))`
///
/// # Properties
///
/// This normalization satisfies the metric properties:
/// 1. Identity: d(x, x) = 0
/// 2. Symmetry: d(x, y) = d(y, x)
/// 3. Triangle inequality: d(x, z) ≤ d(x, y) + d(y, z)
///
/// These properties are important for clustering and indexing operations.
///
/// # Research Reference
///
/// Li, Y., & Liu, B. (2007). "A normalized Levenshtein distance metric."
/// IEEE Transactions on Pattern Analysis and Machine Intelligence.
///
/// # Examples
///
/// ```rust
/// use anno::edit_distance::normalized_edit_distance;
///
/// // Identical strings
/// assert!((normalized_edit_distance("hello", "hello") - 0.0).abs() < 0.001);
///
/// // Small difference
/// let d = normalized_edit_distance("hello", "hallo");
/// assert!(d > 0.0 && d < 0.3);
///
/// // Convert to similarity: sim = 1.0 - distance
/// let similarity = 1.0 - normalized_edit_distance("Einstein", "Einstien");
/// assert!(similarity > 0.7);
/// ```
#[must_use]
pub fn normalized_edit_distance(a: &str, b: &str) -> f64 {
    let a_len = a.chars().count();
    let b_len = b.chars().count();

    // Handle empty strings
    if a_len == 0 && b_len == 0 {
        return 0.0; // Both empty = identical
    }

    let ed = levenshtein(a, b);
    let denominator = a_len + b_len + ed;

    if denominator == 0 {
        0.0
    } else {
        (2 * ed) as f64 / denominator as f64
    }
}

/// Convert normalized edit distance to similarity score.
///
/// Returns 1.0 - normalized_edit_distance, so:
/// - 1.0 = identical
/// - 0.0 = maximally different
#[must_use]
#[inline]
pub fn edit_similarity(a: &str, b: &str) -> f64 {
    1.0 - normalized_edit_distance(a, b)
}

// =============================================================================
// Edit Distance with Wildcards
// =============================================================================

/// Compute edit distance with wildcards for damaged/uncertain text.
///
/// Supports two wildcard characters in the **first** string only:
/// - `?` matches exactly one character (any character)
/// - `*` matches zero or more characters
///
/// This is asymmetric: wildcards in `b` are treated as literal characters.
///
/// # Use Cases
///
/// 1. **OCR uncertainty**: When OCR produces uncertain readings, mark them
///    as wildcards rather than guessing.
///
/// 2. **Damaged inscriptions**: Ancient texts often have illegible portions.
///    Using `?` for single illegible characters and `*` for longer gaps
///    allows matching against known vocabulary.
///
/// 3. **Scribal variations**: Prefix/suffix matching with `*` handles
///    morphological variations.
///
/// # Research Reference
///
/// Tamburini, F. (2025). "On automatic decipherment of lost ancient scripts
/// relying on combinatorial optimisation and coupled simulated annealing."
/// Frontiers in Artificial Intelligence.
///
/// # Examples
///
/// ```rust
/// use anno::edit_distance::edit_distance_wildcards;
///
/// // Single character wildcard
/// assert_eq!(edit_distance_wildcards("R?ma", "Roma"), 0);
/// assert_eq!(edit_distance_wildcards("R?ma", "Rama"), 0);
///
/// // Multi-character wildcard
/// assert_eq!(edit_distance_wildcards("Ein*", "Einstein"), 0);
/// assert_eq!(edit_distance_wildcards("*stein", "Einstein"), 0);
/// assert_eq!(edit_distance_wildcards("*", "anything"), 0);
///
/// // Combined
/// assert_eq!(edit_distance_wildcards("M?r?e C*", "Marie Curie"), 0);
///
/// // Not matching
/// assert!(edit_distance_wildcards("R?ma", "Paris") > 0);
/// ```
#[must_use]
pub fn edit_distance_wildcards(pattern: &str, text: &str) -> usize {
    let p_chars: Vec<char> = pattern.chars().collect();
    let t_chars: Vec<char> = text.chars().collect();

    edit_distance_wildcards_chars(&p_chars, &t_chars)
}

/// Wildcard edit distance on character slices.
fn edit_distance_wildcards_chars(pattern: &[char], text: &[char]) -> usize {
    let m = pattern.len();
    let n = text.len();

    // dp[i][j] = edit distance between pattern[0..i] and text[0..j]
    let mut dp = vec![vec![usize::MAX / 2; n + 1]; m + 1];

    // Base cases
    dp[0][0] = 0;

    // Empty pattern vs non-empty text: need n deletions
    for (j, cell) in dp[0].iter_mut().enumerate().skip(1).take(n) {
        *cell = j;
    }

    // Pattern vs empty text
    for i in 1..=m {
        let c = pattern[i - 1];
        if c == '*' {
            // '*' can match zero characters
            dp[i][0] = dp[i - 1][0];
        } else {
            // Need to delete pattern character
            dp[i][0] = dp[i - 1][0] + 1;
        }
    }

    for i in 1..=m {
        let p_char = pattern[i - 1];

        for j in 1..=n {
            let t_char = text[j - 1];

            match p_char {
                '?' => {
                    // '?' matches exactly one character with cost 0
                    dp[i][j] = dp[i - 1][j - 1];
                }
                '*' => {
                    // '*' can:
                    // 1. Match zero characters: dp[i-1][j]
                    // 2. Match one+ characters: dp[i][j-1] (keep '*' active)
                    dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]);
                }
                _ => {
                    // Regular character
                    let cost = if p_char == t_char { 0 } else { 1 };
                    dp[i][j] = min(
                        min(
                            dp[i - 1][j] + 1, // delete from pattern
                            dp[i][j - 1] + 1, // insert into pattern
                        ),
                        dp[i - 1][j - 1] + cost, // match/substitute
                    );
                }
            }
        }
    }

    dp[m][n]
}

/// Normalized edit distance with wildcards.
///
/// Same semantics as `normalized_edit_distance` but supports wildcards.
#[must_use]
pub fn normalized_edit_distance_wildcards(pattern: &str, text: &str) -> f64 {
    let p_len = pattern.chars().count();
    let t_len = text.chars().count();

    if p_len == 0 && t_len == 0 {
        return 0.0;
    }

    let ed = edit_distance_wildcards(pattern, text);
    let denominator = p_len + t_len + ed;

    if denominator == 0 {
        0.0
    } else {
        (2 * ed) as f64 / denominator as f64
    }
}

/// Convert wildcard edit distance to similarity.
#[must_use]
#[inline]
pub fn edit_similarity_wildcards(pattern: &str, text: &str) -> f64 {
    1.0 - normalized_edit_distance_wildcards(pattern, text)
}

// =============================================================================
// Damerau-Levenshtein (Transpositions)
// =============================================================================

/// Compute Damerau-Levenshtein distance.
///
/// Like Levenshtein but also counts **adjacent transpositions** (swapping
/// two adjacent characters) as a single edit.
///
/// This better models common typos where users swap adjacent keys.
///
/// # Examples
///
/// ```rust
/// use anno::edit_distance::{levenshtein, damerau_levenshtein};
///
/// // "ab" -> "ba" is one transposition in Damerau-Levenshtein
/// assert_eq!(damerau_levenshtein("ab", "ba"), 1);
///
/// // But two substitutions in standard Levenshtein
/// assert_eq!(levenshtein("ab", "ba"), 2);
/// ```
#[must_use]
pub fn damerau_levenshtein(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();

    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    // Full DP matrix needed for transpositions
    let mut dp = vec![vec![0usize; n + 1]; m + 1];

    for (i, row) in dp.iter_mut().enumerate().take(m + 1) {
        row[0] = i;
    }
    for (j, cell) in dp[0].iter_mut().enumerate().take(n + 1) {
        *cell = j;
    }

    for i in 1..=m {
        for j in 1..=n {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };

            dp[i][j] = min(
                min(
                    dp[i - 1][j] + 1, // deletion
                    dp[i][j - 1] + 1, // insertion
                ),
                dp[i - 1][j - 1] + cost, // substitution
            );

            // Transposition: swap adjacent characters
            if i > 1
                && j > 1
                && a_chars[i - 1] == b_chars[j - 2]
                && a_chars[i - 2] == b_chars[j - 1]
            {
                dp[i][j] = min(dp[i][j], dp[i - 2][j - 2] + cost);
            }
        }
    }

    dp[m][n]
}

// =============================================================================
// Batch Operations
// =============================================================================

/// Find the best match for a query string in a list of candidates.
///
/// Returns (index, distance) of the closest match, or None if candidates is empty.
///
/// # Examples
///
/// ```rust
/// use anno::edit_distance::find_closest;
///
/// let candidates = vec!["Einstein", "Newton", "Curie", "Darwin"];
/// let (idx, dist) = find_closest("Einstien", &candidates).unwrap();
/// assert_eq!(idx, 0);  // Einstein is closest
/// assert_eq!(dist, 2); // 2 edits
/// ```
#[must_use]
pub fn find_closest(query: &str, candidates: &[&str]) -> Option<(usize, usize)> {
    if candidates.is_empty() {
        return None;
    }

    let mut best_idx = 0;
    let mut best_dist = levenshtein(query, candidates[0]);

    for (i, candidate) in candidates.iter().enumerate().skip(1) {
        let dist = levenshtein(query, candidate);
        if dist < best_dist {
            best_dist = dist;
            best_idx = i;
        }
    }

    Some((best_idx, best_dist))
}

/// Find all matches within a given edit distance threshold.
///
/// # Examples
///
/// ```rust
/// use anno::edit_distance::find_within_distance;
///
/// let candidates = vec!["cat", "car", "cart", "dog", "rat"];
/// let matches = find_within_distance("cat", &candidates, 1);
/// // Returns: [(0, 0), (1, 1), (4, 1)] for "cat", "car", "rat"
/// ```
#[must_use]
pub fn find_within_distance(
    query: &str,
    candidates: &[&str],
    max_distance: usize,
) -> Vec<(usize, usize)> {
    candidates
        .iter()
        .enumerate()
        .filter_map(|(i, c)| {
            let dist = levenshtein(query, c);
            if dist <= max_distance {
                Some((i, dist))
            } else {
                None
            }
        })
        .collect()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Basic Levenshtein Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_levenshtein_identical() {
        assert_eq!(levenshtein("hello", "hello"), 0);
        assert_eq!(levenshtein("", ""), 0);
    }

    #[test]
    fn test_levenshtein_empty() {
        assert_eq!(levenshtein("", "abc"), 3);
        assert_eq!(levenshtein("abc", ""), 3);
    }

    #[test]
    fn test_levenshtein_classic() {
        assert_eq!(levenshtein("kitten", "sitting"), 3);
        assert_eq!(levenshtein("saturday", "sunday"), 3);
    }

    #[test]
    fn test_levenshtein_single_char() {
        assert_eq!(levenshtein("a", "b"), 1);
        assert_eq!(levenshtein("a", "a"), 0);
        assert_eq!(levenshtein("a", "ab"), 1);
    }

    // -------------------------------------------------------------------------
    // Multilingual Tests (per workspace guidelines)
    // -------------------------------------------------------------------------

    #[test]
    fn test_levenshtein_cjk() {
        // Chinese: 北京 (Beijing) vs 北平 (old name Beiping)
        assert_eq!(levenshtein("北京", "北平"), 1);

        // Japanese: 東京 vs 東京都
        assert_eq!(levenshtein("東京", "東京都"), 1);

        // Korean: 서울 vs 서울시
        assert_eq!(levenshtein("서울", "서울시"), 1);
    }

    #[test]
    fn test_levenshtein_arabic() {
        // محمد (Muhammad) vs أحمد (Ahmad) - differ by first character
        assert_eq!(levenshtein("محمد", "أحمد"), 1);

        // الرياض (Riyadh) - same
        assert_eq!(levenshtein("الرياض", "الرياض"), 0);
    }

    #[test]
    fn test_levenshtein_cyrillic() {
        // Москва (Moscow) vs Москве (Moscow locative)
        assert_eq!(levenshtein("Москва", "Москве"), 1);

        // Путин vs Путин
        assert_eq!(levenshtein("Путин", "Путин"), 0);
    }

    #[test]
    fn test_levenshtein_diacritics() {
        // François vs Francois
        assert_eq!(levenshtein("François", "Francois"), 1);

        // José vs Jose
        assert_eq!(levenshtein("José", "Jose"), 1);

        // München vs Munchen
        assert_eq!(levenshtein("München", "Munchen"), 1);
    }

    #[test]
    fn test_levenshtein_devanagari() {
        // Hindi/Sanskrit: नमस्ते vs नमस्कार (different greetings)
        assert!(levenshtein("नमस्ते", "नमस्कार") > 0);

        // Same word
        assert_eq!(levenshtein("नमस्ते", "नमस्ते"), 0);
    }

    #[test]
    fn test_levenshtein_classical_greek() {
        // Ancient Greek with polytonic: ἐπιστήμη vs επιστημη
        // Removing diacritics changes multiple characters
        let dist = levenshtein("ἐπιστήμη", "επιστημη");
        assert!(dist > 0 && dist <= 4);
    }

    #[test]
    fn test_levenshtein_mixed_script() {
        // Code-switching: "Dr. 田中" vs "Dr. Tanaka"
        let dist = levenshtein("Dr. 田中", "Dr. Tanaka");
        assert!(dist > 0); // Different but comparable
    }

    // -------------------------------------------------------------------------
    // Normalized Edit Distance Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_normalized_identical() {
        assert!((normalized_edit_distance("hello", "hello") - 0.0).abs() < 0.001);
        assert!((normalized_edit_distance("", "") - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_normalized_bounds() {
        // Should always be in [0, 1]
        let pairs = [
            ("a", "b"),
            ("hello", "world"),
            ("abc", "xyz"),
            ("", "test"),
            ("北京", "東京"),
        ];

        for (a, b) in pairs {
            let d = normalized_edit_distance(a, b);
            assert!(
                (0.0..=1.0).contains(&d),
                "Distance {} out of bounds for ({}, {})",
                d,
                a,
                b
            );
        }
    }

    #[test]
    fn test_normalized_symmetry() {
        let pairs = [
            ("Einstein", "Einstien"),
            ("hello", "hallo"),
            ("北京", "北平"),
        ];

        for (a, b) in pairs {
            let d1 = normalized_edit_distance(a, b);
            let d2 = normalized_edit_distance(b, a);
            assert!(
                (d1 - d2).abs() < 0.001,
                "Asymmetric: {} vs {} for ({}, {})",
                d1,
                d2,
                a,
                b
            );
        }
    }

    #[test]
    fn test_edit_similarity() {
        // Identical = 1.0 similarity
        assert!((edit_similarity("hello", "hello") - 1.0).abs() < 0.001);

        // Similar strings have high similarity
        assert!(edit_similarity("Einstein", "Einstien") > 0.7);

        // Very different strings have low similarity
        assert!(edit_similarity("abc", "xyz") < 0.5);
    }

    // -------------------------------------------------------------------------
    // Wildcard Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_wildcard_question_mark() {
        // '?' matches exactly one character
        assert_eq!(edit_distance_wildcards("R?ma", "Roma"), 0);
        assert_eq!(edit_distance_wildcards("R?ma", "Rama"), 0);
        assert_eq!(edit_distance_wildcards("R?ma", "Rima"), 0);

        // Should not match if character count differs
        assert!(edit_distance_wildcards("R?ma", "Rooma") > 0);
    }

    #[test]
    fn test_wildcard_star() {
        // '*' matches zero or more characters
        assert_eq!(edit_distance_wildcards("Ein*", "Einstein"), 0);
        assert_eq!(edit_distance_wildcards("*stein", "Einstein"), 0);
        assert_eq!(edit_distance_wildcards("*", "anything"), 0);
        assert_eq!(edit_distance_wildcards("*", ""), 0);

        // Star in middle
        assert_eq!(edit_distance_wildcards("Ein*ein", "Einstein"), 0);
        assert_eq!(edit_distance_wildcards("M*e", "Marie"), 0);
    }

    #[test]
    fn test_wildcard_combined() {
        // Combination of ? and *
        assert_eq!(edit_distance_wildcards("M?r?e C*", "Marie Curie"), 0);
        assert_eq!(edit_distance_wildcards("?lbert *", "Albert Einstein"), 0);
    }

    #[test]
    fn test_wildcard_no_match() {
        // Wildcards shouldn't match everything incorrectly
        assert!(edit_distance_wildcards("R?ma", "Paris") > 0);
        assert!(edit_distance_wildcards("Ein*", "Newton") > 0);
    }

    #[test]
    fn test_wildcard_exact_no_wildcards() {
        // Without wildcards, should behave like regular edit distance
        assert_eq!(edit_distance_wildcards("hello", "hello"), 0);
        assert_eq!(edit_distance_wildcards("hello", "hallo"), 1);
    }

    #[test]
    fn test_wildcard_cjk() {
        // Wildcards with CJK
        assert_eq!(edit_distance_wildcards("北?", "北京"), 0);
        assert_eq!(edit_distance_wildcards("*京", "北京"), 0);
        assert_eq!(edit_distance_wildcards("東京*", "東京都"), 0);
    }

    #[test]
    fn test_wildcard_damaged_inscription() {
        // Simulating damaged ancient inscription
        // "???TOR" could match "CASTOR" or "NESTOR"
        assert_eq!(edit_distance_wildcards("???TOR", "CASTOR"), 0);
        assert_eq!(edit_distance_wildcards("???TOR", "NESTOR"), 0);

        // Partially readable with gap: "CA*R"
        assert_eq!(edit_distance_wildcards("CA*R", "CASTOR"), 0);
        assert_eq!(edit_distance_wildcards("CA*R", "CAR"), 0);
    }

    #[test]
    fn test_normalized_wildcard() {
        // Should be 0.0 for perfect match
        assert!((normalized_edit_distance_wildcards("R?ma", "Roma") - 0.0).abs() < 0.001);

        // Should be > 0 for imperfect match
        assert!(normalized_edit_distance_wildcards("R?ma", "Paris") > 0.0);
    }

    // -------------------------------------------------------------------------
    // Damerau-Levenshtein Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_damerau_transposition() {
        // Adjacent swap is 1 edit in Damerau, 2 in standard
        assert_eq!(damerau_levenshtein("ab", "ba"), 1);
        assert_eq!(levenshtein("ab", "ba"), 2);

        assert_eq!(damerau_levenshtein("abc", "bac"), 1);
    }

    #[test]
    fn test_damerau_vs_levenshtein() {
        // When no transpositions, should be same
        assert_eq!(
            damerau_levenshtein("kitten", "sitting"),
            levenshtein("kitten", "sitting")
        );
    }

    #[test]
    fn test_damerau_common_typos() {
        // Common keyboard typos: teh -> the
        assert_eq!(damerau_levenshtein("teh", "the"), 1);

        // recieve -> receive
        assert_eq!(damerau_levenshtein("recieve", "receive"), 1);
    }

    // -------------------------------------------------------------------------
    // Batch Operation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_find_closest() {
        let candidates = vec!["Einstein", "Newton", "Curie", "Darwin"];

        let (idx, dist) = find_closest("Einstien", &candidates).unwrap();
        assert_eq!(idx, 0);
        assert_eq!(dist, 2);

        let (idx, _) = find_closest("Neuton", &candidates).unwrap();
        assert_eq!(idx, 1); // Newton
    }

    #[test]
    fn test_find_closest_empty() {
        let candidates: Vec<&str> = vec![];
        assert!(find_closest("query", &candidates).is_none());
    }

    #[test]
    fn test_find_within_distance() {
        let candidates = vec!["cat", "car", "cart", "dog", "rat"];

        let matches = find_within_distance("cat", &candidates, 1);
        assert!(matches.contains(&(0, 0))); // cat
        assert!(matches.contains(&(1, 1))); // car
        assert!(matches.contains(&(4, 1))); // rat
        assert!(!matches.iter().any(|(i, _)| *i == 3)); // dog is too far
    }

    // -------------------------------------------------------------------------
    // Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_unicode_normalization_awareness() {
        // These tests document current behavior - we compare raw chars
        // Users should normalize before comparison if needed

        // Composed vs decomposed é (may differ on byte level)
        // This is a documentation test, not a correctness assertion
        let _composed = "café";
        let _decomposed = "cafe\u{0301}"; // e + combining acute

        // The distance may vary depending on Unicode form
        // Users should normalize to NFC or NFD before comparison
    }

    #[test]
    fn test_emoji() {
        // Emoji are valid Unicode characters
        assert_eq!(levenshtein("🎉", "🎉"), 0);
        assert_eq!(levenshtein("🎉🎊", "🎉"), 1);

        // Multi-codepoint emoji (family) - counts as multiple chars
        // This documents current behavior
        let _family = "👨‍👩‍👧"; // Actually multiple codepoints
    }

    #[test]
    fn test_very_long_strings() {
        // Should handle long strings without stack overflow
        let a: String = "a".repeat(1000);
        let b: String = "b".repeat(1000);

        let dist = levenshtein(&a, &b);
        assert_eq!(dist, 1000); // All substitutions
    }

    // -------------------------------------------------------------------------
    // Weighted Edit Distance Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_weighted_default() {
        // Default weights should match standard Levenshtein
        let weights = EditWeights::default();
        let d1 = levenshtein("kitten", "sitting");
        let d2 = weighted_levenshtein("kitten", "sitting", weights);
        assert!((d1 as f64 - d2).abs() < 0.001);
    }

    #[test]
    fn test_weighted_asymmetric() {
        // Make insertions more expensive than deletions
        let weights = EditWeights {
            insert: 2.0,
            delete: 0.5,
            substitute: 1.0,
        };

        let d1 = weighted_levenshtein("abc", "abcd", weights); // insert 'd'
        let d2 = weighted_levenshtein("abcd", "abc", weights); // delete 'd'

        assert!(d1 > d2, "Insertion should cost more");
    }
}

// =============================================================================
// Property Tests
// =============================================================================

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    // Strategy for generating test strings with diverse Unicode
    fn arb_unicode_string() -> impl Strategy<Value = String> {
        prop::string::string_regex("[a-zA-Z0-9 \\p{Han}\\p{Arabic}\\p{Cyrillic}\\p{Greek}]*")
            .unwrap()
            .prop_filter("non-empty or short", |s| s.len() < 200)
    }

    // Strategy for shorter strings (for expensive O(n²) operations)
    fn arb_short_string() -> impl Strategy<Value = String> {
        prop::string::string_regex("[a-zA-Z0-9]*")
            .unwrap()
            .prop_filter("reasonable length", |s| s.len() < 50)
    }

    // -------------------------------------------------------------------------
    // Levenshtein Properties
    // -------------------------------------------------------------------------

    proptest! {
        /// Identity: distance to self is always 0
        #[test]
        fn prop_levenshtein_identity(s in arb_unicode_string()) {
            prop_assert_eq!(levenshtein(&s, &s), 0);
        }

        /// Symmetry: d(a, b) = d(b, a)
        #[test]
        fn prop_levenshtein_symmetry(a in arb_short_string(), b in arb_short_string()) {
            prop_assert_eq!(levenshtein(&a, &b), levenshtein(&b, &a));
        }

        /// Upper bound: d(a, b) <= max(|a|, |b|)
        #[test]
        fn prop_levenshtein_upper_bound(a in arb_short_string(), b in arb_short_string()) {
            let dist = levenshtein(&a, &b);
            let max_len = a.chars().count().max(b.chars().count());
            prop_assert!(dist <= max_len, "dist {} > max_len {}", dist, max_len);
        }

        /// Lower bound: d(a, b) >= ||a| - |b||
        #[test]
        fn prop_levenshtein_lower_bound(a in arb_short_string(), b in arb_short_string()) {
            let dist = levenshtein(&a, &b);
            let len_a = a.chars().count() as i64;
            let len_b = b.chars().count() as i64;
            let min_dist = (len_a - len_b).unsigned_abs() as usize;
            prop_assert!(dist >= min_dist, "dist {} < min_dist {}", dist, min_dist);
        }

        /// Triangle inequality: d(a, c) <= d(a, b) + d(b, c)
        #[test]
        fn prop_levenshtein_triangle_inequality(
            a in arb_short_string(),
            b in arb_short_string(),
            c in arb_short_string()
        ) {
            let d_ac = levenshtein(&a, &c);
            let d_ab = levenshtein(&a, &b);
            let d_bc = levenshtein(&b, &c);
            prop_assert!(
                d_ac <= d_ab + d_bc,
                "Triangle inequality violated: d({},{})={} > d({},{})={} + d({},{})={}",
                a, c, d_ac, a, b, d_ab, b, c, d_bc
            );
        }
    }

    // -------------------------------------------------------------------------
    // Normalized Edit Distance Properties
    // -------------------------------------------------------------------------

    proptest! {
        /// Normalized distance is always in [0, 1]
        #[test]
        fn prop_normalized_bounds(a in arb_short_string(), b in arb_short_string()) {
            let d = normalized_edit_distance(&a, &b);
            prop_assert!(
                (0.0..=1.0).contains(&d),
                "Normalized distance {} out of bounds",
                d
            );
        }

        /// Normalized distance to self is 0
        #[test]
        fn prop_normalized_identity(s in arb_unicode_string()) {
            let d = normalized_edit_distance(&s, &s);
            prop_assert!((d - 0.0).abs() < 1e-10, "Self-distance should be 0, got {}", d);
        }

        /// Normalized distance is symmetric
        #[test]
        fn prop_normalized_symmetry(a in arb_short_string(), b in arb_short_string()) {
            let d1 = normalized_edit_distance(&a, &b);
            let d2 = normalized_edit_distance(&b, &a);
            prop_assert!(
                (d1 - d2).abs() < 1e-10,
                "Asymmetric: d({},{})={} != d({},{})={}", a, b, d1, b, a, d2
            );
        }

        /// Edit similarity = 1 - normalized distance
        #[test]
        fn prop_similarity_distance_complement(a in arb_short_string(), b in arb_short_string()) {
            let d = normalized_edit_distance(&a, &b);
            let s = edit_similarity(&a, &b);
            prop_assert!(
                (s - (1.0 - d)).abs() < 1e-10,
                "similarity {} != 1 - distance {}", s, d
            );
        }
    }

    // -------------------------------------------------------------------------
    // Damerau-Levenshtein Properties
    // -------------------------------------------------------------------------

    proptest! {
        /// Damerau-Levenshtein has same properties as Levenshtein
        #[test]
        fn prop_damerau_identity(s in arb_unicode_string()) {
            prop_assert_eq!(damerau_levenshtein(&s, &s), 0);
        }

        #[test]
        fn prop_damerau_symmetry(a in arb_short_string(), b in arb_short_string()) {
            prop_assert_eq!(damerau_levenshtein(&a, &b), damerau_levenshtein(&b, &a));
        }

        /// Damerau <= Levenshtein (transpositions are cheaper)
        #[test]
        fn prop_damerau_le_levenshtein(a in arb_short_string(), b in arb_short_string()) {
            let damerau = damerau_levenshtein(&a, &b);
            let lev = levenshtein(&a, &b);
            prop_assert!(
                damerau <= lev,
                "Damerau {} > Levenshtein {} for ({}, {})", damerau, lev, a, b
            );
        }
    }

    // -------------------------------------------------------------------------
    // Wildcard Edit Distance Properties
    // -------------------------------------------------------------------------

    proptest! {
        /// Wildcards never make distance worse than regular edit distance
        /// (wildcards can only help match)
        #[test]
        fn prop_wildcard_le_regular(
            pattern in arb_short_string(),
            text in arb_short_string()
        ) {
            // Only test patterns without actual wildcards to compare
            if !pattern.contains('?') && !pattern.contains('*') {
                let with_wildcards = edit_distance_wildcards(&pattern, &text);
                let without = levenshtein(&pattern, &text);
                prop_assert!(
                    with_wildcards <= without,
                    "Wildcard distance {} > regular {} for ({}, {})",
                    with_wildcards, without, pattern, text
                );
            }
        }

        /// '?' always matches exactly one character
        #[test]
        fn prop_question_mark_matches_one(c in "[a-zA-Z]") {
            let pattern = "?";
            let text = c.to_string();
            prop_assert_eq!(
                edit_distance_wildcards(pattern, &text), 0,
                "? should match single char '{}'", c
            );
        }

        /// '*' matches empty string
        #[test]
        fn prop_star_matches_empty(_unused in Just(())) {
            prop_assert_eq!(edit_distance_wildcards("*", ""), 0);
        }

        /// '*' alone matches any string
        #[test]
        fn prop_star_matches_any(s in arb_short_string()) {
            prop_assert_eq!(
                edit_distance_wildcards("*", &s), 0,
                "* should match any string '{}'", s
            );
        }
    }

    // -------------------------------------------------------------------------
    // Weighted Edit Distance Properties
    // -------------------------------------------------------------------------

    proptest! {
        /// With default weights, weighted == unweighted
        #[test]
        fn prop_weighted_default_equals_regular(a in arb_short_string(), b in arb_short_string()) {
            let weighted = weighted_levenshtein(&a, &b, EditWeights::default());
            let regular = levenshtein(&a, &b) as f64;
            prop_assert!(
                (weighted - regular).abs() < 1e-10,
                "Weighted {} != regular {} for ({}, {})", weighted, regular, a, b
            );
        }

        /// Weighted distance is always non-negative
        #[test]
        fn prop_weighted_non_negative(
            a in arb_short_string(),
            b in arb_short_string(),
            insert in 0.1f64..5.0,
            delete in 0.1f64..5.0,
            substitute in 0.1f64..5.0
        ) {
            let weights = EditWeights { insert, delete, substitute };
            let d = weighted_levenshtein(&a, &b, weights);
            prop_assert!(d >= 0.0, "Weighted distance {} < 0", d);
        }
    }

    // -------------------------------------------------------------------------
    // Batch Operation Properties
    // -------------------------------------------------------------------------

    proptest! {
        /// find_closest returns Some iff candidates is non-empty
        #[test]
        fn prop_find_closest_returns_iff_nonempty(
            query in arb_short_string(),
            candidates in prop::collection::vec(arb_short_string(), 0..10)
        ) {
            let candidate_refs: Vec<&str> = candidates.iter().map(|s| s.as_str()).collect();
            let result = find_closest(&query, &candidate_refs);
            if candidates.is_empty() {
                prop_assert!(result.is_none());
            } else {
                prop_assert!(result.is_some());
            }
        }

        /// find_closest index is in bounds
        #[test]
        fn prop_find_closest_in_bounds(
            query in arb_short_string(),
            candidates in prop::collection::vec(arb_short_string(), 1..10)
        ) {
            let candidate_refs: Vec<&str> = candidates.iter().map(|s| s.as_str()).collect();
            if let Some((idx, _)) = find_closest(&query, &candidate_refs) {
                prop_assert!(idx < candidates.len());
            }
        }

        /// find_within_distance returns subset of candidates
        #[test]
        fn prop_find_within_distance_subset(
            query in arb_short_string(),
            candidates in prop::collection::vec(arb_short_string(), 0..10),
            max_dist in 0usize..5
        ) {
            let candidate_refs: Vec<&str> = candidates.iter().map(|s| s.as_str()).collect();
            let results = find_within_distance(&query, &candidate_refs, max_dist);
            for (idx, dist) in results {
                prop_assert!(idx < candidates.len(), "Index {} out of bounds", idx);
                prop_assert!(dist <= max_dist, "Distance {} > max {}", dist, max_dist);
            }
        }
    }
}
