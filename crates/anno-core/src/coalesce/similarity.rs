//! Multilingual string similarity for entity resolution.
//!
//! This module provides a proper multilingual string similarity pipeline:
//!
//! 1. **Preprocessing**: Unicode normalization, case folding
//! 2. **Script detection**: Route to appropriate algorithm
//! 3. **Multiple strategies**: Word-based, n-gram, edit distance, Jaro-Winkler
//! 4. **Future**: Byte-level learned embeddings
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    StringSimilarity                         │
//! ├─────────────────────────────────────────────────────────────┤
//! │  preprocess(s) -> normalized string                         │
//! │  detect_script(s) -> Script enum                            │
//! │  similarity(a, b) -> f32                                    │
//! └─────────────────────────────────────────────────────────────┘
//!                              │
//!            ┌─────────────────┼─────────────────┐
//!            ▼                 ▼                 ▼
//!     ┌───────────┐     ┌───────────┐     ┌───────────┐
//!     │ WordBased │     │  NGram    │     │  EditDist │
//!     │ (English) │     │  (CJK)    │     │ (fallback)│
//!     └───────────┘     └───────────┘     └───────────┘
//! ```
//!
//! # Algorithms
//!
//! | Algorithm | Best For | Speed | CJK Support |
//! |-----------|----------|-------|-------------|
//! | Word Jaccard | Space-separated text | Fast | Poor |
//! | N-gram Jaccard | CJK, no word boundaries | Fast | Good |
//! | Levenshtein | Typo detection | Medium | Good |
//! | Jaro-Winkler | Short strings, names | Fast | Good |
//!
//! # Future: Byte-Level Embeddings
//!
//! The architecture is designed to eventually support learned embeddings:
//!
//! - **CANINE**: Tokenizer-free character-level transformer
//! - **ByT5**: Byte-level T5, robust across scripts
//! - **CharFormer**: Parameter-efficient character encoder
//!
//! Training approach:
//! 1. Collect entity pairs (same entity, different mentions)
//! 2. Contrastive loss (InfoNCE, triplet)
//! 3. Bi-encoder for fast inference
//! 4. Quantization for deployment
//!
//! # Example
//!
//! ```rust
//! use anno_core::coalesce::similarity::{multilingual_similarity, Script, Similarity};
//!
//! let sim = Similarity::new();
//!
//! // English (word-based)
//! let score = sim.compute("Lynn Conway", "Conway");
//! assert!(score > 0.0);
//!
//! // CJK (n-gram based)
//! let score = sim.compute("中华人民共和国", "中华民国");
//! assert!(score > 0.0);
//!
//! // Script detection
//! assert_eq!(Script::detect("北京"), Script::Cjk);
//! assert_eq!(Script::detect("Москва"), Script::Cyrillic);
//!
//! // Direct multilingual similarity function
//! let sim = multilingual_similarity("東京", "东京");
//! ```

use serde::{Deserialize, Serialize};
use std::borrow::Cow;

// Re-export Script from separate module to avoid compilation issues
pub use super::script::Script;

// =============================================================================
// Preprocessing
// =============================================================================

/// Strip BOM and bidi controls that can pollute comparisons.
///
/// This is **comparison-only** normalization (not offset-preserving). It is safe to remove these
/// characters here because similarity is computed over short strings (entity mentions), not over
/// source-text spans.
fn strip_bom_and_bidi_controls(s: &str) -> Cow<'_, str> {
    fn should_strip(c: char) -> bool {
        // BOM
        c == '\u{FEFF}'
            // Arabic Letter Mark, LRM/RLM
            || matches!(c, '\u{061C}' | '\u{200E}' | '\u{200F}')
            // Bidi embedding/override/isolates
            || matches!(c, '\u{202A}'..='\u{202E}' | '\u{2066}'..='\u{2069}')
    }

    if !s.chars().any(should_strip) {
        return Cow::Borrowed(s);
    }

    Cow::Owned(s.chars().filter(|&c| !should_strip(c)).collect())
}

/// Normalize a string for comparison.
///
/// Applies:
/// 1. Case folding (lowercase)
/// 2. Whitespace normalization
/// 3. Basic Unicode normalization
///
/// # Note
///
/// For production use with NFKC normalization, consider an explicit NFKC step
/// (e.g., via `unicode_normalization::UnicodeNormalization`).
pub fn normalize(s: &str) -> String {
    let s = strip_bom_and_bidi_controls(s);
    // Keep this policy explicit and reusable.
    //
    // Notes:
    // - We lowercase here because several downstream routines (including cross-lingual
    //   known-pair matching) assume a canonical case.
    // - We do collapse all whitespace runs to single ASCII spaces (matching prior behavior).
    let cfg = textprep::ScrubConfig {
        collapse_whitespace: true,
        normalization: textprep::ScrubNormalization::Nfc,
        case: textprep::ScrubCase::Lower,
        strip_diacritics: false,
        ..textprep::ScrubConfig::default()
    };
    textprep::scrub_with(s.as_ref(), &cfg)
}

// =============================================================================
// Similarity Strategies
// =============================================================================

/// Configuration for multilingual string similarity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityConfig {
    /// Whether to normalize strings before comparison.
    pub normalize: bool,
    /// Minimum string length for n-gram computation.
    pub min_ngram_length: usize,
    /// N-gram size (2 = bigrams, 3 = trigrams).
    pub ngram_size: usize,
}

impl Default for SimilarityConfig {
    fn default() -> Self {
        Self {
            normalize: true,
            min_ngram_length: 2,
            ngram_size: 2, // Bigrams work better for CJK
        }
    }
}

/// Multilingual string similarity calculator.
#[derive(Debug, Clone, Default)]
pub struct Similarity {
    config: SimilarityConfig,
}

impl Similarity {
    /// Create a new similarity calculator with default config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with custom config.
    pub fn with_config(config: SimilarityConfig) -> Self {
        Self { config }
    }

    /// Compute similarity between two strings.
    ///
    /// Automatically selects the best algorithm based on script detection.
    pub fn compute(&self, a: &str, b: &str) -> f32 {
        // Handle trivial cases
        if a == b {
            return 1.0;
        }
        if a.is_empty() || b.is_empty() {
            return if a.is_empty() && b.is_empty() {
                1.0
            } else {
                0.0
            };
        }

        // Normalize if configured
        let (a_norm, b_norm) = if self.config.normalize {
            (normalize(a), normalize(b))
        } else {
            (a.to_string(), b.to_string())
        };

        // Check again after normalization
        if a_norm == b_norm {
            return 1.0;
        }

        // Detect scripts
        let script_a = Script::detect(&a_norm);
        let script_b = Script::detect(&b_norm);

        // Route to appropriate algorithm
        if script_a.has_word_boundaries() && script_b.has_word_boundaries() {
            // Both have word boundaries: use word-based Jaccard
            self.word_jaccard(&a_norm, &b_norm)
        } else {
            // At least one lacks word boundaries: use n-gram
            self.ngram_jaccard(&a_norm, &b_norm)
        }
    }

    /// Word-based Jaccard similarity.
    fn word_jaccard(&self, a: &str, b: &str) -> f32 {
        textprep::similarity::word_jaccard(a, b) as f32
    }

    /// Character n-gram Jaccard similarity.
    fn ngram_jaccard(&self, a: &str, b: &str) -> f32 {
        textprep::similarity::char_ngram_jaccard(a, b, self.config.ngram_size) as f32
    }
}

// =============================================================================
// Edit Distance
// =============================================================================

/// Levenshtein edit distance.
///
/// Computes the minimum number of single-character edits (insertions,
/// deletions, substitutions) needed to transform one string into another.
///
/// # Complexity
///
/// Time: O(m * n), Space: O(n) where m, n are string lengths.
pub fn levenshtein_distance(a: &str, b: &str) -> usize {
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

    // Use two-row optimization for space efficiency
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr = vec![0; n + 1];

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };
            curr[j] = (prev[j] + 1) // deletion
                .min(curr[j - 1] + 1) // insertion
                .min(prev[j - 1] + cost); // substitution
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[n]
}

/// Normalized Levenshtein similarity (0.0 to 1.0).
///
/// Returns 1.0 for identical strings, 0.0 for completely different strings.
pub fn levenshtein_similarity(a: &str, b: &str) -> f32 {
    if a == b {
        return 1.0;
    }
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }

    let max_len = a.chars().count().max(b.chars().count());
    if max_len == 0 {
        return 1.0;
    }

    let distance = levenshtein_distance(a, b);
    1.0 - (distance as f32 / max_len as f32)
}

// =============================================================================
// Jaro-Winkler
// =============================================================================

/// Jaro similarity between two strings.
///
/// Good for short strings like names. Returns 0.0 to 1.0.
pub fn jaro_similarity(a: &str, b: &str) -> f32 {
    if a == b {
        return 1.0;
    }

    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let a_len = a_chars.len();
    let b_len = b_chars.len();

    if a_len == 0 || b_len == 0 {
        return 0.0;
    }

    // Match window
    let match_distance = (a_len.max(b_len) / 2).saturating_sub(1);

    let mut a_matches = vec![false; a_len];
    let mut b_matches = vec![false; b_len];

    let mut matches = 0;
    let mut transpositions = 0;

    // Find matches
    for i in 0..a_len {
        let start = i.saturating_sub(match_distance);
        let end = (i + match_distance + 1).min(b_len);

        for j in start..end {
            if b_matches[j] || a_chars[i] != b_chars[j] {
                continue;
            }
            a_matches[i] = true;
            b_matches[j] = true;
            matches += 1;
            break;
        }
    }

    if matches == 0 {
        return 0.0;
    }

    // Count transpositions
    let mut k = 0;
    for i in 0..a_len {
        if !a_matches[i] {
            continue;
        }
        while !b_matches[k] {
            k += 1;
        }
        if a_chars[i] != b_chars[k] {
            transpositions += 1;
        }
        k += 1;
    }

    let m = matches as f32;
    let t = (transpositions / 2) as f32;

    (m / a_len as f32 + m / b_len as f32 + (m - t) / m) / 3.0
}

/// Jaro-Winkler similarity (boosts common prefix).
///
/// Extends Jaro with a prefix bonus. Good for names that start similarly.
pub fn jaro_winkler_similarity(a: &str, b: &str) -> f32 {
    let jaro = jaro_similarity(a, b);

    // Calculate common prefix length (up to 4 chars)
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let prefix_len = a_chars
        .iter()
        .zip(b_chars.iter())
        .take(4)
        .take_while(|(a, b)| a == b)
        .count();

    // Winkler modification: boost for common prefix
    let p = 0.1; // Standard prefix scale
    jaro + (prefix_len as f32 * p * (1.0 - jaro))
}

// =============================================================================
// Public API
// =============================================================================

/// Compute multilingual string similarity using a hybrid approach.
///
/// Automatically selects algorithm based on script:
/// - Space-separated text (English, etc.): Word-based Jaccard
/// - CJK and other scripts without word boundaries: Character n-gram Jaccard
///
/// # Example
///
/// ```rust
/// use anno_core::coalesce::similarity::multilingual_similarity;
///
/// // English
/// let sim = multilingual_similarity("Marie Curie", "Curie");
/// assert!(sim > 0.0);
///
/// // CJK
/// let sim = multilingual_similarity("中华人民共和国", "中华民国");
/// assert!(sim > 0.0);
/// ```
pub fn multilingual_similarity(a: &str, b: &str) -> f32 {
    Similarity::new().compute(a, b)
}

/// Cross-lingual entity matching for transliteration variants.
///
/// Matches entities across different scripts/transliterations:
/// - "Moscow" ↔ "Москва" (English ↔ Russian)
/// - "Tokyo" ↔ "東京" (English ↔ Japanese)
/// - "Beijing" ↔ "北京" (English ↔ Chinese)
///
/// Uses a combination of:
/// 1. Script detection (different scripts = potential transliteration)
/// 2. Known transliteration pairs (common city/person names)
/// 3. Phonetic similarity (future: add phonetic algorithms)
/// 4. Context clues (parentheses, slashes indicating variants)
///
/// Returns similarity score in [0, 1]. Higher scores indicate
/// more likely cross-lingual match.
pub fn cross_lingual_similarity(a: &str, b: &str) -> f32 {
    let script_a = Script::detect(a);
    let script_b = Script::detect(b);

    // If same script (and not Mixed), use regular multilingual similarity
    // Mixed script strings (like "東京 (Tokyo)") should go through cross-lingual logic
    if script_a == script_b && script_a != Script::Mixed {
        return multilingual_similarity(a, b);
    }

    // Known transliteration pairs (common entities)
    // In production, this would be a larger database or learned model
    let known_pairs: &[(&str, &str)] = &[
        // Cities: English ↔ Russian
        ("moscow", "москва"),
        ("saint petersburg", "санкт-петербург"),
        ("kiev", "киев"),
        // Cities: English ↔ Chinese
        ("beijing", "北京"),
        ("shanghai", "上海"),
        ("guangzhou", "广州"),
        ("shenzhen", "深圳"),
        // Cities: English ↔ Japanese
        ("tokyo", "東京"),
        ("東京", "tokyo"), // Bidirectional
        ("osaka", "大阪"),
        ("kyoto", "京都"),
        // Cities: English ↔ Arabic
        ("cairo", "القاهرة"),
        ("riyadh", "الرياض"),
        ("dubai", "دبي"),
        // People: Common transliterations
        ("putin", "путин"),
        ("xi jinping", "习近平"),
        ("abe", "安倍"),
    ];

    let a_norm = normalize(a);
    let b_norm = normalize(b);

    // Extract base text (without parentheses) for better matching
    let base_a = a
        .split('(')
        .next()
        .and_then(|s| s.split('（').next())
        .map(|s| s.trim());
    let base_b = b
        .split('(')
        .next()
        .and_then(|s| s.split('（').next())
        .map(|s| s.trim());

    // Quick check: if base_a matches b (or vice versa) via known pairs, return early
    // This handles "東京 (Tokyo)" vs "Tokyo" case efficiently
    // For "東京 (Tokyo)" vs "Tokyo":
    // - base_a = "東京", b = "Tokyo", b_norm = "tokyo"
    // - For pair ("tokyo", "東京"): ba="東京" == pair_b="東京", b_norm="tokyo" == pair_a="tokyo"
    // This should match: ba == "東京" (pair_b) && b_norm == "tokyo" (pair_a)
    if let Some(ba) = base_a {
        for (pair_a, pair_b) in known_pairs {
            // Exact match: base_a == pair_b AND b_norm == pair_a (or vice versa)
            // Key case: ba="東京" == pair_b="東京" && b_norm="tokyo" == pair_a="tokyo"
            if (ba == *pair_b && b_norm == *pair_a) || (ba == *pair_a && b_norm == *pair_b) {
                return 0.85;
            }
        }
    }
    if let Some(bb) = base_b {
        let bb_norm = normalize(bb);
        for (pair_a, pair_b) in known_pairs {
            let a_matches_a =
                a_norm == *pair_a || a == *pair_a || a_norm.contains(pair_a) || a.contains(pair_a);
            let a_matches_b =
                a_norm == *pair_b || a == *pair_b || a_norm.contains(pair_b) || a.contains(pair_b);
            let bb_matches_a = bb_norm == *pair_a
                || bb == *pair_a
                || bb_norm.contains(pair_a)
                || bb.contains(pair_a);
            let bb_matches_b = bb_norm == *pair_b
                || bb == *pair_b
                || bb_norm.contains(pair_b)
                || bb.contains(pair_b);

            if (a_matches_a && bb_matches_b) || (a_matches_b && bb_matches_a) {
                return 0.85;
            }
        }
    }

    // Check known pairs (bidirectional) - full string check

    for (pair_a, pair_b) in known_pairs {
        // Check if a contains pair_a and b contains pair_b (or vice versa)
        // Check full string, normalized string, and base text
        let mut a_has_a = a_norm.contains(pair_a) || a.contains(pair_a);
        let mut a_has_b = a_norm.contains(pair_b) || a.contains(pair_b);
        let mut b_has_a = b_norm.contains(pair_a) || b.contains(pair_a);
        let mut b_has_b = b_norm.contains(pair_b) || b.contains(pair_b);

        // Also check base text
        if let Some(ba) = base_a {
            let ba_norm = normalize(ba);
            a_has_a = a_has_a || ba_norm.contains(pair_a) || ba.contains(pair_a);
            a_has_b = a_has_b || ba_norm.contains(pair_b) || ba.contains(pair_b);
        }
        if let Some(bb) = base_b {
            let bb_norm = normalize(bb);
            b_has_a = b_has_a || bb_norm.contains(pair_a) || bb.contains(pair_a);
            b_has_b = b_has_b || bb_norm.contains(pair_b) || bb.contains(pair_b);
        }

        // Match if (a has pair_a AND b has pair_b) OR (a has pair_b AND b has pair_a)
        if (a_has_a && b_has_b) || (a_has_b && b_has_a) {
            return 0.85; // High confidence for known transliterations
        }

        // Exact matches (base text)
        if let (Some(ba), Some(bb)) = (base_a, base_b) {
            let ba_norm = normalize(ba);
            let bb_norm = normalize(bb);
            if (ba_norm == *pair_a && bb_norm == *pair_b)
                || (ba_norm == *pair_b && bb_norm == *pair_a)
            {
                return 0.9;
            }
            if (ba == *pair_a && bb == *pair_b) || (ba == *pair_b && bb == *pair_a) {
                return 0.9;
            }
        }
    }

    // Check for transliteration indicators in text
    // Pattern: "Moscow (Москва)" or "東京 (Tokyo)"
    // Also handle case where one string has parentheses and the other doesn't
    // (e.g., "東京 (Tokyo)" vs "Tokyo")
    let has_translit_pattern = |text: &str| -> bool {
        text.contains('(') && text.contains(')')
            || text.contains('/')
            || text.contains('（') && text.contains('）') // Chinese parentheses
    };

    // If either string has transliteration pattern, extract variants from both
    // This handles "東京 (Tokyo)" vs "Tokyo" case
    if has_translit_pattern(a) || has_translit_pattern(b) {
        // Extract text in parentheses as potential transliteration
        let extract_variants = |text: &str| -> Vec<String> {
            let mut variants = Vec::new();
            // Always include the full text as a variant (handles "Tokyo" case)
            variants.push(text.to_string());
            // Also include the base text (without parentheses)
            let base_text = text
                .split('(')
                .next()
                .and_then(|s| s.split('（').next())
                .and_then(|s| s.split('/').next())
                .map(|s| s.trim().to_string());
            if let Some(base) = base_text {
                if !base.is_empty() && base != text {
                    variants.push(base);
                }
            }
            // Extract content in parentheses (English and Chinese)
            // Simple byte-based extraction (safe for ASCII parentheses)
            if let Some(start) = text.find('(') {
                // Find the matching ')' after the '('
                let after_start = &text[start..];
                if let Some(end_offset) = after_start.find(')') {
                    // Extract content between ( and )
                    // start is byte index of '(', end_offset is byte offset from start
                    let content = &text[start + 1..start + end_offset];
                    let content = content.trim();
                    if !content.is_empty() {
                        variants.push(content.to_string());
                    }
                }
            }
            if let Some(start) = text.find('（') {
                if let Some(end) = text[start..].find('）') {
                    let content = text[start + 1..start + end].trim();
                    if !content.is_empty() {
                        variants.push(content.to_string());
                    }
                }
            }
            // Extract content after slash
            if let Some(slash_pos) = text.find('/') {
                let after_slash = text[slash_pos + 1..].trim();
                if !after_slash.is_empty() {
                    variants.push(after_slash.to_string());
                }
            }
            variants
        };

        let variants_a = extract_variants(a);
        let variants_b = extract_variants(b);

        // Check if any variant from a matches b directly (or vice versa)
        // This handles "東京 (Tokyo)" vs "Tokyo" case - most common scenario
        let b_norm = normalize(b);
        let a_norm = normalize(a);

        // First check: variant from a matches b (exact or normalized)
        // This is the critical path for "東京 (Tokyo)" vs "Tokyo"
        // variants_a should contain ["東京 (Tokyo)", "東京", "Tokyo"] for "東京 (Tokyo)"
        // variants_b should contain ["Tokyo"] for "Tokyo"
        // The key: "Tokyo" from variants_a should match "Tokyo" from b
        for va in &variants_a {
            // Exact match (handles "Tokyo" == "Tokyo" - this should work!)
            if va == b {
                return 0.85;
            }
            // Normalized match
            let va_norm = normalize(va);
            if va_norm == b_norm {
                return 0.85;
            }
            // Case-insensitive for ASCII (handles "Tokyo" vs "tokyo")
            if va.eq_ignore_ascii_case(b) {
                return 0.85;
            }
        }
        // Second check: variant from b matches a
        for vb in &variants_b {
            if vb == a {
                return 0.85;
            }
            let vb_norm = normalize(vb);
            if vb_norm == a_norm || vb.eq_ignore_ascii_case(a) {
                return 0.85;
            }
        }
        // Third check: variant from a matches variant from b
        for va in &variants_a {
            let va_norm = normalize(va);
            for vb in &variants_b {
                if va == vb {
                    return 0.9;
                }
                let vb_norm = normalize(vb);
                if va_norm == vb_norm || va.eq_ignore_ascii_case(vb) {
                    return 0.9;
                }
            }
        }

        // Check variant combinations (simplified - avoid expensive similarity calls)
        for va in &variants_a {
            for vb in &variants_b {
                let va_norm = normalize(va);
                let vb_norm = normalize(vb);
                if va_norm == vb_norm || va == vb {
                    return 0.9;
                }
            }
        }
    }

    // Fallback: use base similarity (may catch some cases)
    // Cross-script similarity is typically lower, so we use a threshold
    let base_sim = multilingual_similarity(a, b);
    if base_sim > 0.3 {
        // Boost cross-script matches slightly if they have some similarity
        base_sim * 1.2
    } else {
        base_sim
    }
    .min(1.0)
}

// =============================================================================
// Acronym Detection
// =============================================================================

/// Check if one string is an acronym of another.
///
/// This is a language-agnostic algorithm that checks if the "short" form
/// consists of the first letters of words in the "long" form.
///
/// # Algorithm
///
/// 1. Identify which string is shorter (candidate acronym)
/// 2. Verify the short form looks like an acronym (mostly uppercase, 2-10 chars)
/// 3. Extract initials from the long form by splitting on whitespace/hyphens
/// 4. Compare initials to the short form (case-insensitive)
///
/// # Examples
///
/// ```rust
/// use anno_core::coalesce::similarity::is_acronym_match;
///
/// assert!(is_acronym_match("WHO", "World Health Organization"));
/// assert!(is_acronym_match("MRSA", "Methicillin-resistant Staphylococcus aureus"));
/// assert!(is_acronym_match("IBM", "International Business Machines"));
///
/// // Also works with reversed argument order
/// assert!(is_acronym_match("World Health Organization", "WHO"));
///
/// // Negative cases
/// assert!(!is_acronym_match("IBM", "Apple"));
/// assert!(!is_acronym_match("USA", "Canada"));
/// ```
///
/// # Language Agnosticism
///
/// This works for any language that uses spaces or hyphens to separate words:
/// - English: "WHO" ↔ "World Health Organization"
/// - German: "DDR" ↔ "Deutsche Demokratische Republik"
/// - Spanish: "ONU" ↔ "Organización de las Naciones Unidas"
///
/// For languages without word boundaries (CJK), this will not produce matches,
/// which is the correct behavior since CJK acronyms work differently.
pub fn is_acronym_match(a: &str, b: &str) -> bool {
    // Determine which is the potential acronym (shorter one)
    let (short, long) = if a.chars().count() < b.chars().count() {
        (a, b)
    } else {
        (b, a)
    };

    // Acronym should be reasonably short (2-10 chars)
    let short_len = short.chars().count();
    if !(2..=10).contains(&short_len) {
        return false;
    }

    // Check if short form looks like an acronym (mostly uppercase letters)
    let upper_count = short.chars().filter(|c| c.is_uppercase()).count();
    let alpha_count = short.chars().filter(|c| c.is_alphabetic()).count();

    // Need at least half uppercase, and mostly alphabetic
    if upper_count < short_len / 2 || alpha_count < short_len / 2 {
        return false;
    }

    // Extract initials from long form
    let initials: String = long
        .split(|c: char| c.is_whitespace() || c == '-')
        .filter(|w| !w.is_empty())
        .filter_map(|w| w.chars().next())
        .filter(|c| c.is_alphabetic())
        .collect();

    // Compare case-insensitively
    initials.eq_ignore_ascii_case(short)
}

// =============================================================================
// Synonym Infrastructure
// =============================================================================

/// Source for synonym relationships.
///
/// Implement this trait to provide synonym lookups from various sources:
/// - UMLS MRCONSO table (medical)
/// - WordNet (general English)
/// - Wikidata aliases (multilingual)
/// - Custom domain-specific tables
///
/// The trait is designed to be composable: you can chain multiple sources.
///
/// # Example
///
/// ```rust,ignore
/// use anno_core::coalesce::similarity::{SynonymMatch, SynonymSource};
///
/// struct UmlsSynonyms { /* UMLS connection */ }
///
/// impl SynonymSource for UmlsSynonyms {
///     fn lookup(&self, term: &str) -> Option<SynonymMatch> {
///         // Query UMLS MRCONSO for the term
///         // Return canonical CUI and confidence
///         None
///     }
/// }
/// ```
pub trait SynonymSource: Send + Sync {
    /// Look up a term and return synonym information if found.
    ///
    /// Returns `None` if the term is not in this source.
    fn lookup(&self, term: &str) -> Option<SynonymMatch>;

    /// Check if two terms are synonyms according to this source.
    ///
    /// Default implementation looks up both terms and checks if they
    /// share a canonical ID.
    fn are_synonyms(&self, a: &str, b: &str) -> Option<SynonymMatch> {
        let match_a = self.lookup(a)?;
        let match_b = self.lookup(b)?;

        // If they share a canonical ID, they're synonyms
        if match_a.canonical_id == match_b.canonical_id {
            Some(SynonymMatch {
                canonical_id: match_a.canonical_id,
                confidence: (match_a.confidence + match_b.confidence) / 2.0,
                source: match_a.source,
            })
        } else {
            None
        }
    }

    /// Name of this synonym source for provenance tracking.
    fn source_name(&self) -> &str;
}

/// Result of a synonym lookup.
#[derive(Debug, Clone)]
pub struct SynonymMatch {
    /// Canonical identifier (e.g., UMLS CUI, WordNet synset ID)
    pub canonical_id: String,
    /// Confidence in the match [0, 1]
    pub confidence: f32,
    /// Name of the source that produced this match
    pub source: String,
}

/// Empty synonym source that never matches.
///
/// Use this as the default when no synonym sources are configured.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoSynonyms;

impl SynonymSource for NoSynonyms {
    fn lookup(&self, _term: &str) -> Option<SynonymMatch> {
        None
    }

    fn source_name(&self) -> &str {
        "none"
    }
}

/// Chained synonym sources that try each source in order.
#[derive(Default)]
pub struct ChainedSynonyms {
    sources: Vec<Box<dyn SynonymSource>>,
}

impl ChainedSynonyms {
    /// Create a new empty chain.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a synonym source to the chain.
    pub fn with_source<S: SynonymSource + 'static>(mut self, source: S) -> Self {
        self.sources.push(Box::new(source));
        self
    }
}

impl SynonymSource for ChainedSynonyms {
    fn lookup(&self, term: &str) -> Option<SynonymMatch> {
        for source in &self.sources {
            if let Some(m) = source.lookup(term) {
                return Some(m);
            }
        }
        None
    }

    fn source_name(&self) -> &str {
        "chained"
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Script detection tests moved to script.rs module

    #[test]
    fn test_normalize() {
        assert_eq!(normalize("  Hello   World  "), "hello world");
        assert_eq!(normalize("UPPERCASE"), "uppercase");
    }

    #[test]
    fn test_similarity_identical() {
        let sim = Similarity::new();
        assert_eq!(sim.compute("test", "test"), 1.0);
        assert_eq!(sim.compute("北京", "北京"), 1.0);
    }

    #[test]
    fn test_similarity_cjk_partial() {
        let sim = Similarity::new();
        let score = sim.compute("中华人民共和国", "中华民国");
        assert!(score > 0.0 && score < 1.0, "CJK partial: {}", score);
    }

    #[test]
    fn test_similarity_english_words() {
        let sim = Similarity::new();
        let score = sim.compute("Marie Curie", "Curie");
        assert!(score > 0.0 && score < 1.0, "English partial: {}", score);
    }

    #[test]
    fn test_levenshtein() {
        assert_eq!(levenshtein_distance("kitten", "sitting"), 3);
        assert_eq!(levenshtein_distance("", "test"), 4);
        assert_eq!(levenshtein_distance("same", "same"), 0);
    }

    #[test]
    fn test_levenshtein_similarity() {
        assert_eq!(levenshtein_similarity("same", "same"), 1.0);
        let sim = levenshtein_similarity("kitten", "sitting");
        assert!(sim > 0.5 && sim < 1.0);
    }

    #[test]
    fn test_jaro_winkler() {
        let sim = jaro_winkler_similarity("MARTHA", "MARHTA");
        assert!(sim > 0.9, "Jaro-Winkler for similar strings: {}", sim);

        let sim = jaro_winkler_similarity("DWAYNE", "DUANE");
        assert!(sim > 0.8, "Jaro-Winkler: {}", sim);
    }

    #[test]
    fn test_multilingual_api() {
        // Verify the public API works
        let sim = multilingual_similarity("test", "test");
        assert_eq!(sim, 1.0);
    }
}

// =============================================================================
// Property Tests
// =============================================================================

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Similarity is symmetric
        #[test]
        fn similarity_symmetric(a in "\\PC{1,30}", b in "\\PC{1,30}") {
            let sim = Similarity::new();
            let ab = sim.compute(&a, &b);
            let ba = sim.compute(&b, &a);
            prop_assert!((ab - ba).abs() < 0.001,
                "Symmetry: {} vs {}", ab, ba);
        }

        /// Similarity is bounded [0, 1]
        #[test]
        fn similarity_bounded(a in "\\PC{0,50}", b in "\\PC{0,50}") {
            let sim = Similarity::new();
            let score = sim.compute(&a, &b);
            prop_assert!((0.0..=1.0).contains(&score),
                "Bounds: {}", score);
        }

        /// Identical strings have similarity 1.0
        #[test]
        fn similarity_identity(s in "\\PC{0,50}") {
            let sim = Similarity::new();
            let score = sim.compute(&s, &s);
            prop_assert!((score - 1.0).abs() < 0.001,
                "Identity: {}", score);
        }

        /// Levenshtein distance is non-negative
        #[test]
        fn levenshtein_non_negative(a in "\\PC{0,30}", b in "\\PC{0,30}") {
            let dist = levenshtein_distance(&a, &b);
            prop_assert!(dist <= a.chars().count() + b.chars().count());
        }

        /// Jaro-Winkler is bounded [0, 1]
        #[test]
        fn jaro_winkler_bounded(a in "\\PC{1,30}", b in "\\PC{1,30}") {
            let sim = jaro_winkler_similarity(&a, &b);
            prop_assert!((0.0..=1.0).contains(&sim),
                "Jaro-Winkler bounds: {}", sim);
        }
    }

    // =========================================================================
    // Acronym matching tests
    // =========================================================================

    #[test]
    fn test_acronym_who() {
        assert!(is_acronym_match("WHO", "World Health Organization"));
        assert!(is_acronym_match("World Health Organization", "WHO"));
    }

    #[test]
    fn test_acronym_mrsa() {
        assert!(is_acronym_match(
            "MRSA",
            "Methicillin-resistant Staphylococcus aureus"
        ));
    }

    #[test]
    fn test_acronym_ibm() {
        assert!(is_acronym_match("IBM", "International Business Machines"));
    }

    #[test]
    fn test_acronym_german() {
        // German acronyms work too (DDR = Deutsche Demokratische Republik)
        assert!(is_acronym_match("DDR", "Deutsche Demokratische Republik"));
        // EU works in German too
        assert!(is_acronym_match("EU", "Europäische Union"));
    }

    #[test]
    fn test_acronym_negative() {
        assert!(!is_acronym_match("IBM", "Apple"));
        assert!(!is_acronym_match("WHO", "United Nations"));
        assert!(!is_acronym_match("USA", "Canada"));
    }

    #[test]
    fn test_acronym_too_short() {
        // Single letter is not an acronym
        assert!(!is_acronym_match("A", "Apple"));
    }

    #[test]
    fn test_acronym_not_mostly_uppercase() {
        // Lowercase doesn't look like an acronym
        assert!(!is_acronym_match("who", "World Health Organization"));
    }

    // =========================================================================
    // Synonym infrastructure tests
    // =========================================================================

    #[test]
    fn test_no_synonyms_returns_none() {
        let source = NoSynonyms;
        assert!(source.lookup("test").is_none());
        assert!(source.are_synonyms("a", "b").is_none());
    }

    #[test]
    fn test_chained_synonyms_empty() {
        let chain = ChainedSynonyms::new();
        assert!(chain.lookup("test").is_none());
    }

    // =========================================================================
    // Cross-lingual transliteration tests
    // =========================================================================

    #[test]
    fn test_cross_lingual_same_script() {
        // Same script should use regular similarity
        let sim = cross_lingual_similarity("Moscow", "Moskva");
        assert!((0.0..=1.0).contains(&sim), "Similarity should be in [0, 1]");

        let sim_cjk = cross_lingual_similarity("北京", "北京");
        assert!(
            (sim_cjk - 1.0).abs() < 0.01,
            "Identical strings should have similarity 1.0"
        );
    }

    #[test]
    fn test_cross_lingual_known_pairs() {
        // Known transliteration pairs should have high similarity
        let sim = cross_lingual_similarity("Moscow", "Москва");
        assert!(sim > 0.8, "Moscow ↔ Москва should match");

        let sim = cross_lingual_similarity("Tokyo", "東京");
        assert!(sim > 0.8, "Tokyo ↔ 東京 should match");

        let sim = cross_lingual_similarity("Beijing", "北京");
        assert!(sim > 0.8, "Beijing ↔ 北京 should match");
    }

    #[test]
    fn test_cross_lingual_with_parentheses() {
        // Text with transliteration in parentheses
        let sim = cross_lingual_similarity("Moscow (Москва)", "Москва");
        assert!(
            sim > 0.6,
            "Should extract variant from parentheses, got {}",
            sim
        );

        // For CJK with Latin: "東京 (Tokyo)" vs "Tokyo"
        // Known pair: ("tokyo", "東京")
        // base_a = "東京" (from "東京 (Tokyo)")
        // b_norm = "tokyo" (from normalize("Tokyo"))
        // Should match: ba == "東京" (pair_b) && b_norm == "tokyo" (pair_a)
        // OR variant extraction: "Tokyo" from parentheses matches "Tokyo"
        let sim = cross_lingual_similarity("東京 (Tokyo)", "Tokyo");
        // Should match either via known pair (東京↔Tokyo) or via extracted variant
        assert!(
            sim > 0.5,
            "Should handle CJK with Latin transliteration, got {}",
            sim
        );

        // Test the reverse direction
        let sim = cross_lingual_similarity("Tokyo", "東京 (Tokyo)");
        assert!(sim > 0.5, "Should work in reverse direction, got {}", sim);
    }

    #[test]
    fn test_cross_lingual_different_scripts() {
        // Different scripts without known pairs should have lower similarity
        let sim = cross_lingual_similarity("Paris", "パリ"); // Paris in Katakana
                                                             // May or may not match depending on known pairs, but should handle gracefully
        assert!((0.0..=1.0).contains(&sim));
    }
}
