//! Unified byte/character/token offset handling.
//!
//! # The Three Coordinate Systems
//!
//! When working with text, different tools use different ways to count positions.
//! This causes bugs when tools disagree on where an entity starts and ends.
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────────┐
//! │                    THE OFFSET ALIGNMENT PROBLEM                          │
//! ├──────────────────────────────────────────────────────────────────────────┤
//! │                                                                          │
//! │  Text: "The café costs €50"                                              │
//! │                                                                          │
//! │  ┌─────────────────────────────────────────────────────────────────────┐ │
//! │  │ BYTE INDEX (what regex/file I/O returns)                            │ │
//! │  │                                                                     │ │
//! │  │   T   h   e       c   a   f   [  é  ]       c   o   s   t   s       │ │
//! │  │   0   1   2   3   4   5   6   7-8   9  10  11  12  13  14  15  16   │ │
//! │  │                               └─2 bytes─┘                           │ │
//! │  │                                                                     │ │
//! │  │   [     €     ]   5   0                                             │ │
//! │  │   17-18-19   20  21  22                                             │ │
//! │  │   └─3 bytes──┘                                                      │ │
//! │  └─────────────────────────────────────────────────────────────────────┘ │
//! │                                                                          │
//! │  ┌─────────────────────────────────────────────────────────────────────┐ │
//! │  │ CHAR INDEX (what humans count, what eval tools expect)              │ │
//! │  │                                                                     │ │
//! │  │   T   h   e       c   a   f   é       c   o   s   t   s       €   5 │ │
//! │  │   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16 │ │
//! │  │                               └─1 char─┘              └─1 char─┘    │ │
//! │  └─────────────────────────────────────────────────────────────────────┘ │
//! │                                                                          │
//! │  ┌─────────────────────────────────────────────────────────────────────┐ │
//! │  │ TOKEN INDEX (what BERT/transformers return)                         │ │
//! │  │                                                                     │ │
//! │  │   [CLS]  The  café  costs   €    50   [SEP]                         │ │
//! │  │     0     1    2      3     4     5     6                           │ │
//! │  │                                                                     │ │
//! │  │   But wait! "café" might be split:                                  │ │
//! │  │   [CLS]  The  ca  ##fe  costs   €    50   [SEP]                     │ │
//! │  │     0     1    2    3     4     5     6     7                       │ │
//! │  └─────────────────────────────────────────────────────────────────────┘ │
//! │                                                                          │
//! │  THE PROBLEM:                                                            │
//! │  • Regex finds "€50" at byte positions (17, 22)                          │
//! │  • Evaluation tool expects char positions (15, 18)                       │
//! │  • BERT returns token positions (5, 6)                                   │
//! │                                                                          │
//! │  Without conversion, your F1 score will be WRONG.                        │
//! └──────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # The Subword Problem
//!
//! Transformer models split words into subword tokens. This breaks NER labels:
//!
//! ```text
//! Text:      "playing"
//!
//! Tokenizer: WordPiece splits unknown words
//!            "playing" → ["play", "##ing"]
//!
//! Problem:   Which token gets the NER label?
//!
//! ┌────────────────────────────────────────────────────┐
//! │                 OPTION 1: First-only               │
//! │                                                    │
//! │   Tokens:  ["play", "##ing"]                       │
//! │   Labels:  [B-PER,    O    ]  ← "##ing" ignored!   │
//! │                                                    │
//! │   Problem: Model never learns "##ing" is part of  │
//! │            the entity. Loses signal.              │
//! ├────────────────────────────────────────────────────┤
//! │                 OPTION 2: All tokens               │
//! │                                                    │
//! │   Tokens:  ["play", "##ing"]                       │
//! │   Labels:  [B-PER,  I-PER ]  ← Continuation!       │
//! │                                                    │
//! │   Better, but requires propagating labels during  │
//! │   both training AND inference.                    │
//! └────────────────────────────────────────────────────┘
//! ```
//!
//! # Solution: Dual Representations
//!
//! ```text
//! ┌────────────────────────────────────────────────────┐
//! │  Use TextSpan at boundaries, TokenSpan for models  │
//! ├────────────────────────────────────────────────────┤
//! │                                                    │
//! │  Entity: "John" in "Hello John!"                   │
//! │                                                    │
//! │  TextSpan {                                        │
//! │      byte_start: 6,   byte_end: 10,                │
//! │      char_start: 6,   char_end: 10,  // ASCII: same│
//! │  }                                                 │
//! │                                                    │
//! │  TokenSpan {                                       │
//! │      token_start: 2,  // [CLS] Hello John [SEP]    │
//! │      token_end: 3,    //   0     1     2     3     │
//! │  }                                                 │
//! │                                                    │
//! │  Store BOTH. Convert at boundaries.                │
//! └────────────────────────────────────────────────────┘
//! ```
//!
//! This module provides:
//! - [`TextSpan`]: Stores both byte and char offsets together
//! - [`TokenSpan`]: Stores subword token indices
//! - [`OffsetMapping`]: Maps between token ↔ character positions
//! - [`CharOffset`]: Newtype wrapper for character offsets (type safety)
//! - [`ByteOffset`]: Newtype wrapper for byte offsets (type safety)
//!
//! # API Boundary Conventions
//!
//! Anno uses **character offsets** as the canonical representation at API boundaries:
//!
//! | Type | Offset Convention | Notes |
//! |------|-------------------|-------|
//! | `Entity.start/end` | Character | Public API, evaluation, serialization |
//! | `Signal` with `Location::Text` | Character | Grounded document model |
//! | `Span::Text` | Character | Entity span representation |
//! | Backend internals | Often byte | Regex, JSON parsing, byte slicing |
//! | Token indices | Token | BERT/transformer models |
//!
//! **Rule of thumb**: Convert to character offsets as early as possible (at the
//! backend boundary), and use the newtype wrappers (`CharOffset`, `ByteOffset`)
//! when you need to be explicit about which you're working with.
//!
//! # Type Safety with Newtypes
//!
//! The most common source of Unicode bugs is accidentally mixing byte and character
//! offsets. Use the newtype wrappers to make this impossible at compile time:
//!
//! ```rust
//! use anno::offset::{CharOffset, ByteOffset};
//!
//! fn process_span(start: CharOffset, end: CharOffset) {
//!     // Can only receive CharOffset, not ByteOffset
//! }
//!
//! let char_pos = CharOffset::new(5);
//! let byte_pos = ByteOffset::new(10);
//!
//! process_span(char_pos, CharOffset::new(10));  // OK
//! // process_span(byte_pos, CharOffset::new(10));  // Compile error!
//! ```

use serde::{Deserialize, Serialize};
use std::ops::Range;

// =============================================================================
// Newtype Wrappers for Type Safety
// =============================================================================

// Re-export the canonical offset newtypes from anno-core.
// These have private inner fields, enforcing construction via `new()` / `From`
// and access via `get()` / `Into<usize>`.
pub use anno_core::{ByteOffset, CharOffset};

/// A character range (start and end as character offsets).
///
/// This is a convenience type for APIs that need to express spans
/// in character coordinates with compile-time type safety.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CharRange {
    /// Start offset (inclusive)
    pub start: CharOffset,
    /// End offset (exclusive)
    pub end: CharOffset,
}

impl CharRange {
    /// Create a new character range.
    #[must_use]
    pub const fn new(start: CharOffset, end: CharOffset) -> Self {
        Self { start, end }
    }

    /// Create from raw usize values.
    #[must_use]
    pub const fn from_raw(start: usize, end: usize) -> Self {
        Self {
            start: CharOffset::new(start),
            end: CharOffset::new(end),
        }
    }

    /// Length in characters.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.end.get().saturating_sub(self.start.get())
    }

    /// Check if empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.start.get() >= self.end.get()
    }

    /// Convert to a standard Range.
    #[must_use]
    pub const fn as_range(&self) -> Range<usize> {
        self.start.get()..self.end.get()
    }
}

impl From<(usize, usize)> for CharRange {
    fn from((start, end): (usize, usize)) -> Self {
        Self::from_raw(start, end)
    }
}

/// A byte range (start and end as byte offsets).
///
/// Use for APIs that work with raw byte positions (regex, file I/O).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ByteRange {
    /// Start offset (inclusive)
    pub start: ByteOffset,
    /// End offset (exclusive)
    pub end: ByteOffset,
}

impl ByteRange {
    /// Create a new byte range.
    #[must_use]
    pub const fn new(start: ByteOffset, end: ByteOffset) -> Self {
        Self { start, end }
    }

    /// Create from raw usize values.
    #[must_use]
    pub const fn from_raw(start: usize, end: usize) -> Self {
        Self {
            start: ByteOffset::new(start),
            end: ByteOffset::new(end),
        }
    }

    /// Length in bytes.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.end.get().saturating_sub(self.start.get())
    }

    /// Check if empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.start.get() >= self.end.get()
    }

    /// Convert to a standard Range.
    #[must_use]
    pub const fn as_range(&self) -> Range<usize> {
        self.start.get()..self.end.get()
    }
}

impl From<(usize, usize)> for ByteRange {
    fn from((start, end): (usize, usize)) -> Self {
        Self::from_raw(start, end)
    }
}

/// A text span with both byte and character offsets.
///
/// This is the canonical representation for entity positions.
/// Store both to avoid repeated conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TextSpan {
    /// Byte offset (start, inclusive)
    pub byte_start: usize,
    /// Byte offset (end, exclusive)
    pub byte_end: usize,
    /// Character offset (start, inclusive)
    pub char_start: usize,
    /// Character offset (end, exclusive)
    pub char_end: usize,
}

impl TextSpan {
    /// Create a span from byte offsets, computing char offsets from text.
    ///
    /// # Arguments
    /// * `text` - The full text (needed to compute char offsets)
    /// * `byte_start` - Byte offset start (inclusive)
    /// * `byte_end` - Byte offset end (exclusive)
    ///
    /// # Example
    /// ```
    /// use anno::offset::TextSpan;
    ///
    /// let text = "Price €50";
    /// // "Price " = 6 bytes, € = 3 bytes, "50" = 2 bytes = 11 total
    /// let span = TextSpan::from_bytes(text, 6, 11); // "€50"
    /// assert_eq!(span.char_start, 6);
    /// assert_eq!(span.char_end, 9); // € is 1 char but 3 bytes
    /// ```
    #[must_use]
    pub fn from_bytes(text: &str, byte_start: usize, byte_end: usize) -> Self {
        let (char_start, char_end) = bytes_to_chars(text, byte_start, byte_end);
        Self {
            byte_start,
            byte_end,
            char_start,
            char_end,
        }
    }

    /// Create a span from character offsets, computing byte offsets from text.
    ///
    /// # Arguments
    /// * `text` - The full text (needed to compute byte offsets)
    /// * `char_start` - Character offset start (inclusive)
    /// * `char_end` - Character offset end (exclusive)
    #[must_use]
    pub fn from_chars(text: &str, char_start: usize, char_end: usize) -> Self {
        let (byte_start, byte_end) = chars_to_bytes(text, char_start, char_end);
        Self {
            byte_start,
            byte_end,
            char_start,
            char_end,
        }
    }

    /// Create a span for ASCII text where byte == char offsets.
    ///
    /// This is a fast path for ASCII-only text.
    #[must_use]
    pub const fn ascii(start: usize, end: usize) -> Self {
        Self {
            byte_start: start,
            byte_end: end,
            char_start: start,
            char_end: end,
        }
    }

    /// Get byte range.
    #[must_use]
    pub const fn byte_range(&self) -> Range<usize> {
        self.byte_start..self.byte_end
    }

    /// Get character range.
    #[must_use]
    pub const fn char_range(&self) -> Range<usize> {
        self.char_start..self.char_end
    }

    /// Byte length.
    #[must_use]
    pub const fn byte_len(&self) -> usize {
        self.byte_end.saturating_sub(self.byte_start)
    }

    /// Character length.
    #[must_use]
    pub const fn char_len(&self) -> usize {
        self.char_end.saturating_sub(self.char_start)
    }

    /// Check if this span is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.byte_start >= self.byte_end
    }

    /// Check if this is ASCII (byte == char offsets).
    #[must_use]
    pub const fn is_ascii(&self) -> bool {
        self.byte_start == self.char_start && self.byte_end == self.char_end
    }

    /// Extract the text for this span.
    #[must_use]
    pub fn extract<'a>(&self, text: &'a str) -> &'a str {
        text.get(self.byte_start..self.byte_end).unwrap_or("")
    }
}

impl From<Range<usize>> for TextSpan {
    /// Create from byte range (assumes ASCII).
    fn from(range: Range<usize>) -> Self {
        Self::ascii(range.start, range.end)
    }
}

// =============================================================================
// Token Span (Subword-Level)
// =============================================================================

/// Span in subword token space.
///
/// # Research Context (BERT for NER, NAACL 2019)
///
/// Transformer models operate on subword tokens, not characters.
/// Entity boundaries often split mid-token:
///
/// ```text
/// Text:       "New York City"
/// Tokens:     ["New", "York", "City"]      <- clean split
/// Token IDs:  [2739, 1816, 2103]
/// TokenSpan:  (0, 3) for "New York City"
///
/// Text:       "playing"
/// Tokens:     ["play", "##ing"]            <- mid-word split
/// Token IDs:  [2377, 2075]
/// TokenSpan:  (0, 2) for "playing"
/// ```
///
/// Key insight: When propagating BIO labels to continuation tokens (##),
/// use I- prefix to avoid treating them as separate entities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TokenSpan {
    /// Token index (start, inclusive)
    pub start: usize,
    /// Token index (end, exclusive)
    pub end: usize,
    /// Original text span (for reconstruction)
    pub text_span: TextSpan,
}

impl TokenSpan {
    /// Create a token span with its corresponding text span.
    #[must_use]
    pub const fn new(start: usize, end: usize, text_span: TextSpan) -> Self {
        Self {
            start,
            end,
            text_span,
        }
    }

    /// Number of tokens in this span.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    /// Check if empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.start >= self.end
    }

    /// Token range.
    #[must_use]
    pub const fn token_range(&self) -> Range<usize> {
        self.start..self.end
    }
}

/// Offset mapping from tokenizer.
///
/// Maps each token to its character span in the original text.
/// Used to convert between token indices and character positions.
///
/// # Research Note (HuggingFace Tokenizers)
///
/// The `offset_mapping` from HuggingFace tokenizers is a list of
/// `(char_start, char_end)` for each token. Special tokens like
/// `[CLS]` and `[SEP]` have offset `(0, 0)`.
#[derive(Debug, Clone)]
pub struct OffsetMapping {
    /// Character spans for each token: `[(char_start, char_end), ...]`
    offsets: Vec<(usize, usize)>,
}

impl OffsetMapping {
    /// Create from tokenizer output.
    ///
    /// # Arguments
    /// * `offsets` - List of (char_start, char_end) for each token
    #[must_use]
    pub fn new(offsets: Vec<(usize, usize)>) -> Self {
        Self { offsets }
    }

    /// Get character span for a token.
    #[must_use]
    pub fn get(&self, token_idx: usize) -> Option<(usize, usize)> {
        self.offsets.get(token_idx).copied()
    }

    /// Find tokens that overlap with a character span.
    ///
    /// Returns `(first_token, last_token_exclusive)`.
    ///
    /// # Note on Label Propagation
    ///
    /// For entity "playing" tokenized as `["play", "##ing"]`:
    /// - Assign B-PER to "play" (first token)
    /// - Assign I-PER to "##ing" (continuation)
    #[must_use]
    pub fn char_span_to_tokens(
        &self,
        char_start: usize,
        char_end: usize,
    ) -> Option<(usize, usize)> {
        let mut first_token = None;
        let mut last_token = 0;

        for (idx, &(tok_start, tok_end)) in self.offsets.iter().enumerate() {
            // Skip special tokens (offset 0, 0)
            if tok_start == 0 && tok_end == 0 && idx != 0 {
                continue;
            }

            // Check overlap
            if tok_end > char_start && tok_start < char_end {
                if first_token.is_none() {
                    first_token = Some(idx);
                }
                last_token = idx + 1;
            }
        }

        first_token.map(|first| (first, last_token))
    }

    /// Convert token span to character span.
    #[must_use]
    pub fn tokens_to_char_span(
        &self,
        token_start: usize,
        token_end: usize,
    ) -> Option<(usize, usize)> {
        if token_start >= token_end || token_end > self.offsets.len() {
            return None;
        }

        // Find first non-special token's start
        // Make logic consistent - always skip special tokens (0, 0)
        let char_start = (token_start..token_end)
            .filter_map(|idx| {
                let (s, e) = self.offsets.get(idx)?;
                // Skip special tokens (0, 0)
                if *s == 0 && *e == 0 {
                    None
                } else {
                    Some(*s)
                }
            })
            .next()
            .or_else(|| {
                // If all tokens are special, return the start of the first token's position
                // (which is 0 for special tokens, but we need a fallback)
                self.offsets.get(token_start).map(|(s, _)| *s)
            })?;

        // Find last non-special token's end
        let char_end = (token_start..token_end)
            .rev()
            .filter_map(|idx| {
                let (s, e) = self.offsets.get(idx)?;
                // Skip special tokens
                if *s == 0 && *e == 0 {
                    None
                } else {
                    Some(*e)
                }
            })
            .next()?;

        Some((char_start, char_end))
    }

    /// Number of tokens.
    #[must_use]
    pub fn len(&self) -> usize {
        self.offsets.len()
    }

    /// Check if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.offsets.is_empty()
    }
}

// =============================================================================
// Conversion Functions
// =============================================================================

/// Convert byte offsets to character offsets.
///
/// Handles cases where byte offsets fall in the middle of multi-byte UTF-8 characters
/// by mapping them to the containing character's start position.
///
/// # Arguments
///
/// * `text` - The source text
/// * `byte_start` - Byte offset for the start (may be in middle of a character)
/// * `byte_end` - Byte offset for the end (may be in middle of a character)
///
/// # Returns
///
/// A tuple `(char_start, char_end)` where:
/// - `char_start` is the character index containing `byte_start`
/// - `char_end` is the character index after the character containing `byte_end` (exclusive)
///
/// # Behavior
///
/// - If `byte_start` falls in the middle of a multi-byte character, it maps to that character's start
/// - If `byte_end` falls in the middle of a multi-byte character, it maps to the next character (exclusive end)
/// - If `byte_start` or `byte_end` are beyond the text length, they map to the end character index
///
/// Uses standard library's `char_indices()` for iteration.
#[must_use]
pub fn bytes_to_chars(text: &str, byte_start: usize, byte_end: usize) -> (usize, usize) {
    if text.is_empty() {
        return (0, 0);
    }

    let mut char_start = 0;
    let mut found_start = false;
    let mut last_char_idx = 0;
    let mut last_byte_idx = 0;

    for (char_idx, (byte_idx, ch)) in text.char_indices().enumerate() {
        last_char_idx = char_idx;
        last_byte_idx = byte_idx;

        // Check if byte_start falls within this character's byte range
        let char_byte_end = byte_idx + ch.len_utf8();
        if !found_start {
            if byte_idx == byte_start {
                // Exact match at character start
                char_start = char_idx;
                found_start = true;
            } else if byte_idx < byte_start && byte_start < char_byte_end {
                // byte_start is in the middle of this character - map to character start
                char_start = char_idx;
                found_start = true;
            }
        }

        // Check if byte_end falls within this character's byte range
        if byte_idx == byte_end {
            // Exact match at character start - char_end is exclusive, so return this char index
            // (meaning range is [char_start, char_idx), which includes chars up to but not including char_idx)
            return (char_start, char_idx);
        } else if byte_idx < byte_end && byte_end < char_byte_end {
            // byte_end is in the middle of this character - map to next character (exclusive)
            return (char_start, char_idx + 1);
        } else if byte_idx > byte_end {
            // We've passed byte_end - use current character (exclusive end)
            return (char_start, char_idx);
        }
    }

    // Handle end of string
    let char_count = last_char_idx + 1;
    if !found_start {
        // byte_start was beyond all characters or in the last character
        if byte_start >= last_byte_idx {
            // Check if byte_start is in the last character's range
            if let Some(last_ch) = text.chars().last() {
                let last_char_byte_end = last_byte_idx + last_ch.len_utf8();
                if byte_start < last_char_byte_end {
                    char_start = last_char_idx;
                } else {
                    char_start = char_count;
                }
            } else {
                char_start = char_count;
            }
        } else {
            // Shouldn't happen, but fallback
            char_start = char_count;
        }
    }

    (char_start, char_count)
}

/// Convert character offsets to byte offsets.
#[must_use]
pub fn chars_to_bytes(text: &str, char_start: usize, char_end: usize) -> (usize, usize) {
    let mut byte_start = 0;
    let mut byte_end = text.len();
    let mut found_start = false;

    for (char_idx, (byte_idx, _ch)) in text.char_indices().enumerate() {
        if char_idx == char_start {
            byte_start = byte_idx;
            found_start = true;
        }
        if char_idx == char_end {
            byte_end = byte_idx;
            return (byte_start, byte_end);
        }
    }

    if !found_start {
        byte_start = text.len();
    }

    (byte_start, byte_end)
}

/// Build an offset mapping table for efficient repeated conversions.
///
/// Returns a vec where `mapping[byte_idx]` gives the character index.
/// Useful when converting many spans from the same text.
#[must_use]
pub fn build_byte_to_char_map(text: &str) -> Vec<usize> {
    let mut map = vec![0usize; text.len() + 1];

    for (char_idx, (byte_idx, ch)) in text.char_indices().enumerate() {
        // Fill all bytes of this character with the same char index
        let ch_len = ch.len_utf8();
        for i in 0..ch_len {
            if byte_idx + i < map.len() {
                map[byte_idx + i] = char_idx;
            }
        }
    }

    // Set the final position
    if !map.is_empty() {
        map[text.len()] = text.chars().count();
    }

    map
}

/// Build an offset mapping table from char to byte.
///
/// Returns a vec where `mapping[char_idx]` gives the byte index.
#[must_use]
pub fn build_char_to_byte_map(text: &str) -> Vec<usize> {
    let char_count = text.chars().count();
    let mut map = vec![0usize; char_count + 1];

    for (char_idx, (byte_idx, _ch)) in text.char_indices().enumerate() {
        map[char_idx] = byte_idx;
    }

    // Set the final position
    if !map.is_empty() {
        map[char_count] = text.len();
    }

    map
}

/// Fast check if text is ASCII-only.
#[must_use]
pub fn is_ascii(text: &str) -> bool {
    text.is_ascii()
}

// =============================================================================
// Span Converter (batch operations)
// =============================================================================

/// Converter for efficiently handling many spans from the same text.
///
/// Pre-computes mapping tables so each conversion is O(1).
pub struct SpanConverter {
    byte_to_char: Vec<usize>,
    char_to_byte: Vec<usize>,
    is_ascii: bool,
}

impl SpanConverter {
    /// Create a converter for the given text.
    #[must_use]
    pub fn new(text: &str) -> Self {
        let is_ascii = is_ascii(text);
        if is_ascii {
            // For ASCII, mappings are identity
            Self {
                byte_to_char: Vec::new(),
                char_to_byte: Vec::new(),
                is_ascii: true,
            }
        } else {
            Self {
                byte_to_char: build_byte_to_char_map(text),
                char_to_byte: build_char_to_byte_map(text),
                is_ascii: false,
            }
        }
    }

    /// Convert byte offset to char offset.
    ///
    /// # Arguments
    ///
    /// * `byte_idx` - Byte offset in the text
    ///
    /// # Returns
    ///
    /// Character offset corresponding to the byte offset. If `byte_idx` is out of bounds,
    /// returns the last valid character index (or 0 if the map is empty).
    ///
    /// # Panics
    ///
    /// In debug mode, panics if `byte_idx` exceeds the text length by more than 1
    /// (allowing for the exclusive end position).
    #[must_use]
    pub fn byte_to_char(&self, byte_idx: usize) -> usize {
        if self.is_ascii {
            byte_idx
        } else {
            self.byte_to_char.get(byte_idx).copied().unwrap_or_else(|| {
                // Bounds check: byte_idx should be <= text.len() (inclusive end position)
                // The map has length text.len() + 1 to include the end position
                #[cfg(debug_assertions)]
                {
                    let max_valid = self.byte_to_char.len().saturating_sub(1);
                    if byte_idx > max_valid {
                        debug_assert!(
                            byte_idx <= max_valid + 1,
                            "byte_idx {} out of bounds (max valid: {}, map len: {})",
                            byte_idx,
                            max_valid,
                            self.byte_to_char.len()
                        );
                    }
                }
                self.byte_to_char.last().copied().unwrap_or(0)
            })
        }
    }

    /// Like [`byte_to_char`](Self::byte_to_char) but rounds up when `byte_idx`
    /// falls inside a multi-byte character.  Use this for **exclusive end
    /// offsets** so the span includes the full final character rather than
    /// silently truncating it.
    #[must_use]
    pub fn byte_to_char_ceil(&self, byte_idx: usize) -> usize {
        if self.is_ascii {
            return byte_idx;
        }
        let floor = self.byte_to_char(byte_idx);
        // If byte_idx is exactly on a character boundary, floor is correct.
        // Otherwise, the next character boundary is floor + 1 (unless we're
        // already at the end of the string).
        if byte_idx < self.byte_to_char.len() {
            // Check if this byte is the first byte of its character.
            // If map[byte_idx] == map[byte_idx-1] (and byte_idx > 0), we're
            // mid-character and should round up.
            if byte_idx > 0
                && self
                    .byte_to_char
                    .get(byte_idx.wrapping_sub(1))
                    .copied()
                    .unwrap_or(0)
                    == floor
                && (self.byte_to_char.get(byte_idx).copied() == Some(floor))
            {
                // Mid-character: check if the previous byte starts the same char.
                // We need to confirm we're not at the start of a char.
                // A byte is at the start of a char if it's the first byte with
                // this char_idx value.
                let at_char_start = byte_idx == 0
                    || self.byte_to_char.get(byte_idx - 1).copied().unwrap_or(0) != floor;
                if !at_char_start {
                    return floor + 1;
                }
            }
        }
        floor
    }

    /// Convert char offset to byte offset.
    ///
    /// # Arguments
    ///
    /// * `char_idx` - Character offset in the text
    ///
    /// # Returns
    ///
    /// Byte offset corresponding to the character offset. If `char_idx` is out of bounds,
    /// returns the last valid byte index (or 0 if the map is empty).
    ///
    /// # Panics
    ///
    /// In debug mode, panics if `char_idx` exceeds the character count by more than 1
    /// (allowing for the exclusive end position).
    #[must_use]
    pub fn char_to_byte(&self, char_idx: usize) -> usize {
        if self.is_ascii {
            char_idx
        } else {
            self.char_to_byte.get(char_idx).copied().unwrap_or_else(|| {
                // Bounds check: char_idx should be <= char_count (inclusive end position)
                // The map has length char_count + 1 to include the end position
                #[cfg(debug_assertions)]
                {
                    let max_valid = self.char_to_byte.len().saturating_sub(1);
                    if char_idx > max_valid {
                        debug_assert!(
                            char_idx <= max_valid + 1,
                            "char_idx {} out of bounds (max valid: {}, map len: {})",
                            char_idx,
                            max_valid,
                            self.char_to_byte.len()
                        );
                    }
                }
                self.char_to_byte.last().copied().unwrap_or(0)
            })
        }
    }

    /// Convert byte span to TextSpan.
    #[must_use]
    pub fn from_bytes(&self, byte_start: usize, byte_end: usize) -> TextSpan {
        TextSpan {
            byte_start,
            byte_end,
            char_start: self.byte_to_char(byte_start),
            char_end: self.byte_to_char(byte_end),
        }
    }

    /// Convert char span to TextSpan.
    #[must_use]
    pub fn from_chars(&self, char_start: usize, char_end: usize) -> TextSpan {
        TextSpan {
            byte_start: self.char_to_byte(char_start),
            byte_end: self.char_to_byte(char_end),
            char_start,
            char_end,
        }
    }

    /// Check if this text is ASCII.
    #[must_use]
    pub const fn is_ascii(&self) -> bool {
        self.is_ascii
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ascii_text() {
        let text = "Hello World";
        let span = TextSpan::from_bytes(text, 0, 5);

        assert_eq!(span.byte_start, 0);
        assert_eq!(span.byte_end, 5);
        assert_eq!(span.char_start, 0);
        assert_eq!(span.char_end, 5);
        assert!(span.is_ascii());
        assert_eq!(span.extract(text), "Hello");
    }

    #[test]
    fn test_euro_symbol() {
        let text = "Price €50";
        // "Price " = 6 bytes, 6 chars
        // € = 3 bytes (E2 82 AC), 1 char
        // "50" = 2 bytes, 2 chars
        // Total: 11 bytes, 9 chars
        //
        // "€50" starts at byte 6, ends at byte 11
        // "€50" starts at char 6, ends at char 9

        let span = TextSpan::from_bytes(text, 6, 11);

        assert_eq!(span.byte_start, 6);
        assert_eq!(span.byte_end, 11);
        assert_eq!(span.char_start, 6);
        assert_eq!(span.char_end, 9);
        assert!(!span.is_ascii());
        assert_eq!(span.extract(text), "€50");
    }

    #[test]
    fn test_pound_symbol() {
        let text = "Fee: £25";
        // "Fee: " = 5 bytes, 5 chars
        // £ = 2 bytes (C2 A3), 1 char
        // "25" = 2 bytes, 2 chars
        // Total: 9 bytes, 8 chars
        //
        // "£25" starts at byte 5, ends at byte 9
        // "£25" starts at char 5, ends at char 8

        let span = TextSpan::from_bytes(text, 5, 9);

        assert_eq!(span.byte_start, 5);
        assert_eq!(span.byte_end, 9);
        assert_eq!(span.char_start, 5);
        assert_eq!(span.char_end, 8);
        assert_eq!(span.extract(text), "£25");
    }

    #[test]
    fn test_emoji() {
        let text = "Hello 👋 World";
        // "Hello " = 6 bytes, 6 chars
        // 👋 = 4 bytes, 1 char
        // " World" = 6 bytes, 6 chars
        // Total: 16 bytes, 13 chars
        //
        // "World" starts at byte 11, ends at byte 16
        // "World" starts at char 8, ends at char 13

        let span = TextSpan::from_bytes(text, 11, 16);

        assert_eq!(span.char_start, 8);
        assert_eq!(span.char_end, 13);
        assert_eq!(span.extract(text), "World");
    }

    #[test]
    fn test_cjk() {
        let text = "日本語 test";
        // 日 = 3 bytes, 1 char
        // 本 = 3 bytes, 1 char
        // 語 = 3 bytes, 1 char
        // " " = 1 byte, 1 char
        // "test" = 4 bytes, 4 chars
        // Total: 14 bytes, 8 chars
        //
        // "test" starts at byte 10, ends at byte 14
        // "test" starts at char 4, ends at char 8

        let span = TextSpan::from_bytes(text, 10, 14);

        assert_eq!(span.char_start, 4);
        assert_eq!(span.char_end, 8);
        assert_eq!(span.extract(text), "test");
    }

    #[test]
    fn test_from_chars() {
        let text = "Price €50";
        // "€50" is chars 6..9

        let span = TextSpan::from_chars(text, 6, 9);

        assert_eq!(span.char_start, 6);
        assert_eq!(span.char_end, 9);
        assert_eq!(span.byte_start, 6);
        assert_eq!(span.byte_end, 11);
        assert_eq!(span.extract(text), "€50");
    }

    #[test]
    fn test_converter_ascii() {
        let text = "Hello World";
        let conv = SpanConverter::new(text);

        assert!(conv.is_ascii());
        assert_eq!(conv.byte_to_char(5), 5);
        assert_eq!(conv.char_to_byte(5), 5);
    }

    #[test]
    fn test_converter_unicode() {
        let text = "Price €50";
        let conv = SpanConverter::new(text);

        assert!(!conv.is_ascii());

        // Byte 6 -> Char 6 (start of €)
        assert_eq!(conv.byte_to_char(6), 6);
        // Byte 9 -> Char 7 (end of €, which spans bytes 6-8)
        assert_eq!(conv.byte_to_char(9), 7);
        // Byte 11 -> Char 9 (end of string)
        assert_eq!(conv.byte_to_char(11), 9);

        // Char 6 -> Byte 6
        assert_eq!(conv.char_to_byte(6), 6);
        // Char 9 -> Byte 11
        assert_eq!(conv.char_to_byte(9), 11);
    }

    #[test]
    fn test_empty_span() {
        let text = "test";
        let span = TextSpan::from_bytes(text, 2, 2);

        assert!(span.is_empty());
        assert_eq!(span.byte_len(), 0);
        assert_eq!(span.char_len(), 0);
    }

    #[test]
    fn test_full_text_span() {
        let text = "日本語";
        let span = TextSpan::from_bytes(text, 0, text.len());

        assert_eq!(span.char_start, 0);
        assert_eq!(span.char_end, 3);
        assert_eq!(span.byte_len(), 9);
        assert_eq!(span.char_len(), 3);
    }

    // =========================================================================
    // Newtype wrapper tests
    // =========================================================================

    #[test]
    fn test_char_offset_newtype() {
        let offset = CharOffset::new(5);
        assert_eq!(offset.get(), 5);

        let from_usize: CharOffset = 10.into();
        assert_eq!(from_usize.get(), 10);

        let back_to_usize: usize = CharOffset::new(15).into();
        assert_eq!(back_to_usize, 15);
    }

    #[test]
    fn test_byte_offset_newtype() {
        let offset = ByteOffset::new(5);
        assert_eq!(offset.get(), 5);

        let from_usize: ByteOffset = 10.into();
        assert_eq!(from_usize.get(), 10);

        let back_to_usize: usize = ByteOffset::new(15).into();
        assert_eq!(back_to_usize, 15);
    }

    #[test]
    fn test_char_range() {
        let range = CharRange::new(CharOffset::new(5), CharOffset::new(10));
        assert_eq!(range.len(), 5);
        assert!(!range.is_empty());
        assert_eq!(range.as_range(), 5..10);

        let from_raw = CharRange::from_raw(0, 5);
        assert_eq!(from_raw.start.get(), 0);
        assert_eq!(from_raw.end.get(), 5);

        let from_tuple: CharRange = (2, 7).into();
        assert_eq!(from_tuple.len(), 5);
    }

    #[test]
    fn test_byte_range() {
        let range = ByteRange::new(ByteOffset::new(5), ByteOffset::new(10));
        assert_eq!(range.len(), 5);
        assert!(!range.is_empty());
        assert_eq!(range.as_range(), 5..10);

        let empty_range = ByteRange::from_raw(5, 5);
        assert!(empty_range.is_empty());
    }

    #[test]
    fn test_char_offset_ordering() {
        let a = CharOffset::new(5);
        let b = CharOffset::new(10);
        let c = CharOffset::new(5);

        assert!(a < b);
        assert!(b > a);
        assert_eq!(a, c);
    }

    #[test]
    fn test_byte_offset_ordering() {
        let a = ByteOffset::new(5);
        let b = ByteOffset::new(10);
        let c = ByteOffset::new(5);

        assert!(a < b);
        assert!(b > a);
        assert_eq!(a, c);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Round-trip: bytes -> chars -> bytes should preserve byte offsets.
        #[test]
        fn roundtrip_bytes_chars_bytes(text in ".{0,100}") {
            if text.is_empty() {
                return Ok(());
            }

            let byte_end = text.len();
            let (char_start, char_end) = bytes_to_chars(&text, 0, byte_end);
            let (byte_start2, byte_end2) = chars_to_bytes(&text, char_start, char_end);

            prop_assert_eq!(byte_start2, 0);
            prop_assert_eq!(byte_end2, byte_end);
        }

        /// TextSpan extraction should always succeed for valid spans.
        #[test]
        fn textspan_extract_valid(text in ".{1,50}") {
            let span = TextSpan::from_bytes(&text, 0, text.len());
            let extracted = span.extract(&text);
            prop_assert_eq!(extracted, &text);
        }

        /// Converter should match direct conversion.
        #[test]
        fn converter_matches_direct(text in ".{1,50}") {
            let conv = SpanConverter::new(&text);

            let span_direct = TextSpan::from_bytes(&text, 0, text.len());
            let span_conv = conv.from_bytes(0, text.len());

            prop_assert_eq!(span_direct.char_start, span_conv.char_start);
            prop_assert_eq!(span_direct.char_end, span_conv.char_end);
        }

        /// ASCII detection should be correct.
        #[test]
        fn ascii_detection(text in "[a-zA-Z0-9 ]{0,50}") {
            prop_assert!(is_ascii(&text));
        }

        /// CharOffset preserves value through conversions.
        #[test]
        fn char_offset_roundtrip(val in 0usize..1_000_000) {
            let offset = CharOffset::new(val);
            prop_assert_eq!(offset.get(), val);

            let from_into: usize = CharOffset::from(val).into();
            prop_assert_eq!(from_into, val);
        }

        /// ByteOffset preserves value through conversions.
        #[test]
        fn byte_offset_roundtrip(val in 0usize..1_000_000) {
            let offset = ByteOffset::new(val);
            prop_assert_eq!(offset.get(), val);

            let from_into: usize = ByteOffset::from(val).into();
            prop_assert_eq!(from_into, val);
        }

        /// CharRange length is always end - start.
        #[test]
        fn char_range_length(start in 0usize..1000, len in 0usize..1000) {
            let end = start + len;
            let range = CharRange::from_raw(start, end);
            prop_assert_eq!(range.len(), len);
            prop_assert_eq!(range.is_empty(), len == 0);
        }

        /// ByteRange length is always end - start.
        #[test]
        fn byte_range_length(start in 0usize..1000, len in 0usize..1000) {
            let end = start + len;
            let range = ByteRange::from_raw(start, end);
            prop_assert_eq!(range.len(), len);
            prop_assert_eq!(range.is_empty(), len == 0);
        }
    }
}
