//! Property-based tests for offset conversion functions.
//!
//! These tests verify that byte/char/token conversions are correct and
//! roundtrip properly, even with complex Unicode input.

use anno::offset::{
    build_byte_to_char_map, build_char_to_byte_map, bytes_to_chars, chars_to_bytes, SpanConverter,
};
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    /// Byte-to-char-to-byte roundtrip should preserve byte offsets (when aligned to char boundaries).
    #[test]
    fn offset_conversion_roundtrip(
        text in ".{0,1000}",  // Random Unicode text
        byte_start in 0usize..1000,
        byte_end in 0usize..1000,
    ) {
        if byte_start <= byte_end && byte_end <= text.len() {
            // Check if byte offsets are aligned to character boundaries
            let is_char_boundary = |idx: usize| -> bool {
                if idx >= text.len() {
                    return true; // End of string is always a boundary
                }
                text.is_char_boundary(idx)
            };

            if is_char_boundary(byte_start) && is_char_boundary(byte_end) {
                let (char_start, char_end) = bytes_to_chars(&text, byte_start, byte_end);
                let (byte_start2, byte_end2) = chars_to_bytes(&text, char_start, char_end);

                // Roundtrip should preserve byte offsets when aligned to char boundaries
                prop_assert_eq!(byte_start, byte_start2,
                    "Byte start mismatch: {} -> {} -> {}", byte_start, char_start, byte_start2);
                prop_assert_eq!(byte_end, byte_end2,
                    "Byte end mismatch: {} -> {} -> {}", byte_end, char_end, byte_end2);
            }
            // If not aligned, the conversion is still valid but may not roundtrip exactly
            // (this is expected behavior - byte_end in middle of char maps to char boundary)
        }
    }

    /// Char-to-byte-to-char roundtrip should preserve char offsets.
    #[test]
    fn offset_conversion_roundtrip_chars(
        text in ".{0,1000}",
        char_start in 0usize..1000,
        char_end in 0usize..1000,
    ) {
        let char_count = text.chars().count();
        if char_start <= char_end && char_end <= char_count {
            let (byte_start, byte_end) = chars_to_bytes(&text, char_start, char_end);
            let (char_start2, char_end2) = bytes_to_chars(&text, byte_start, byte_end);

            // Roundtrip should preserve char offsets
            prop_assert_eq!(char_start, char_start2,
                "Char start mismatch: {} -> {} -> {}", char_start, byte_start, char_start2);
            prop_assert_eq!(char_end, char_end2,
                "Char end mismatch: {} -> {} -> {}", char_end, byte_end, char_end2);
        }
    }

    /// SpanConverter should be consistent with direct conversion functions.
    #[test]
    fn span_converter_consistency(
        text in ".{0,500}",
        byte_idx in 0usize..500,
    ) {
        let converter = SpanConverter::new(&text);
        let char_idx = converter.byte_to_char(byte_idx);
        let byte_idx2 = converter.char_to_byte(char_idx);

        // For ASCII, should be exact
        if text.is_ascii() && byte_idx < text.len() {
            prop_assert_eq!(byte_idx, byte_idx2,
                "ASCII conversion mismatch: {} -> {} -> {}", byte_idx, char_idx, byte_idx2);
        } else {
            // For Unicode, should be close (within one char)
            prop_assert!(byte_idx2 <= byte_idx + 4, // Max 4 bytes per UTF-8 char
                "Byte index too far: {} -> {} -> {}", byte_idx, char_idx, byte_idx2);
        }
    }

    /// Offset bounds should always be valid.
    #[test]
    fn offset_bounds_always_valid(
        text in ".{0,1000}",
        char_start in 0usize..1000,
        char_end in 0usize..1000,
    ) {
        let char_count = text.chars().count();
        if char_start <= char_end && char_end <= char_count {
            let (byte_start, byte_end) = chars_to_bytes(&text, char_start, char_end);
            prop_assert!(byte_start <= byte_end,
                "Invalid byte span: start={}, end={}", byte_start, byte_end);
            prop_assert!(byte_end <= text.len(),
                "Byte end exceeds text length: {} > {}", byte_end, text.len());
        }
    }

    /// Byte-to-char mapping should be monotonic.
    #[test]
    fn byte_to_char_map_monotonic(
        text in ".{0,500}",
    ) {
        let map = build_byte_to_char_map(&text);

        // Map should be monotonic (byte index increases -> char index increases or stays same)
        for i in 1..map.len() {
            prop_assert!(map[i] >= map[i-1],
                "Map not monotonic at {}: {} -> {}", i, map[i-1], map[i]);
        }
    }

    /// Char-to-byte mapping should be strictly increasing.
    #[test]
    fn char_to_byte_map_strictly_increasing(
        text in ".{0,500}",
    ) {
        let map = build_char_to_byte_map(&text);

        // Map should be strictly increasing
        for i in 1..map.len() {
            prop_assert!(map[i] > map[i-1],
                "Map not strictly increasing at {}: {} -> {}", i, map[i-1], map[i]);
        }
    }

    /// SpanConverter should handle boundary conditions correctly.
    #[test]
    fn span_converter_boundaries(
        text in ".{1,500}",  // Non-empty
    ) {
        let converter = SpanConverter::new(&text);
        let char_count = text.chars().count();
        let byte_len = text.len();

        // Start of text
        prop_assert_eq!(converter.byte_to_char(0), 0);
        prop_assert_eq!(converter.char_to_byte(0), 0);

        // End of text
        let last_char = converter.byte_to_char(byte_len);
        prop_assert_eq!(last_char, char_count);

        let last_byte = converter.char_to_byte(char_count);
        prop_assert_eq!(last_byte, byte_len);
    }

    /// TextSpan should preserve all coordinate systems.
    #[test]
    fn text_span_coordinate_consistency(
        text in ".{0,500}",
        char_start in 0usize..500,
        char_end in 0usize..500,
    ) {
        let char_count = text.chars().count();
        if char_start <= char_end && char_end <= char_count {
            let converter = SpanConverter::new(&text);
            let span = converter.from_chars(char_start, char_end);

            // All coordinates should be consistent
            prop_assert_eq!(span.char_start, char_start);
            prop_assert_eq!(span.char_end, char_end);
            prop_assert!(span.byte_start <= span.byte_end);
            prop_assert!(span.byte_end <= text.len());

            // Roundtrip should work
            let span2 = converter.from_bytes(span.byte_start, span.byte_end);
            prop_assert_eq!(span2.char_start, char_start);
            prop_assert_eq!(span2.char_end, char_end);
        }
    }

    /// ASCII text should have byte == char offsets.
    #[test]
    fn ascii_byte_char_equality(
        text in "[ -~]{0,500}",  // ASCII only
    ) {
        for (char_idx, (byte_idx, _)) in text.char_indices().enumerate() {
            prop_assert_eq!(char_idx, byte_idx,
                "ASCII mismatch at char {}: byte {}", char_idx, byte_idx);
        }
    }

    /// Multi-byte UTF-8 characters should be handled correctly.
    #[test]
    fn utf8_multi_byte_handling(
        text in ".*",  // Any Unicode including emoji, CJK, etc.
    ) {
        let converter = SpanConverter::new(&text);

        for (char_idx, (byte_idx, ch)) in text.char_indices().enumerate() {
            let char_from_byte = converter.byte_to_char(byte_idx);
            prop_assert_eq!(char_from_byte, char_idx,
                "Char index mismatch: expected {}, got {} at byte {}", char_idx, char_from_byte, byte_idx);

            let byte_from_char = converter.char_to_byte(char_idx);
            prop_assert_eq!(byte_from_char, byte_idx,
                "Byte index mismatch: expected {}, got {} at char {}", byte_idx, byte_from_char, char_idx);

            // Multi-byte chars should span multiple bytes
            let ch_len = ch.len_utf8();
            if ch_len > 1 {
                // All bytes of this char should map to the same char index
                for offset in 0..ch_len {
                    if byte_idx + offset < text.len() {
                        let mapped_char = converter.byte_to_char(byte_idx + offset);
                        prop_assert_eq!(mapped_char, char_idx,
                            "Multi-byte char mapping inconsistent at byte offset {}", offset);
                    }
                }
            }
        }
    }
}
