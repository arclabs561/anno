//! Tests for offset conversion bugs and edge cases.
//!
//! These tests specifically target the bugs identified in BUGS_FOUND.md:
//! - bytes_to_chars() handling of middle-of-character positions
//! - DiscontinuousSpan offset system consistency
//! - Entity::total_len() with discontinuous spans

use anno::offset::{bytes_to_chars, chars_to_bytes};
use anno::{DiscontinuousSpan, Entity, EntityType};

#[test]
fn test_bytes_to_chars_middle_of_multibyte_char() {
    // Test case: byte_start in middle of multi-byte character
    let text = "café"; // "é" is 2 bytes: [0xC3, 0xA9] at bytes 3-4
                       // c=0, a=1, f=2, é=3-4

    // byte_start = 4 (second byte of "é") should map to character 3 (the "é")
    let (char_start, char_end) = bytes_to_chars(text, 4, 4);
    assert_eq!(
        char_start, 3,
        "byte_start=4 (middle of 'é') should map to char 3"
    );
    assert_eq!(char_end, 4, "byte_end=4 should map to exclusive end char 4");

    // byte_start = 3 (first byte of "é") should map to character 3
    let (char_start, char_end) = bytes_to_chars(text, 3, 4);
    assert_eq!(char_start, 3);
    assert_eq!(char_end, 4);

    // byte_start = 0, byte_end = 4 (middle of "é") should map to [0, 4)
    let (char_start, char_end) = bytes_to_chars(text, 0, 4);
    assert_eq!(char_start, 0);
    assert_eq!(char_end, 4);
}

#[test]
fn test_bytes_to_chars_emoji_middle() {
    // Test with emoji (4 bytes)
    let text = "Hello 👋 World";
    // "Hello " = 6 bytes, 6 chars
    // 👋 = 4 bytes (bytes 6-9), 1 char (char 6)
    // " World" = 6 bytes, 6 chars

    // byte_start = 7 (middle of emoji) should map to char 6
    let (char_start, char_end) = bytes_to_chars(text, 7, 7);
    assert_eq!(
        char_start, 6,
        "byte_start=7 (middle of emoji) should map to char 6"
    );
    assert_eq!(char_end, 7, "byte_end=7 should map to exclusive end char 7");

    // byte_start = 6, byte_end = 8 (spanning emoji)
    let (char_start, char_end) = bytes_to_chars(text, 6, 8);
    assert_eq!(char_start, 6);
    assert_eq!(char_end, 7);
}

#[test]
fn test_bytes_to_chars_cjk_middle() {
    // Test with CJK characters (3 bytes each)
    let text = "日本語";
    // 日 = 3 bytes (0-2), 1 char (0)
    // 本 = 3 bytes (3-5), 1 char (1)
    // 語 = 3 bytes (6-8), 1 char (2)

    // byte_start = 1 (middle of "日") should map to char 0
    let (char_start, char_end) = bytes_to_chars(text, 1, 1);
    assert_eq!(char_start, 0);
    assert_eq!(char_end, 1);

    // byte_start = 4 (middle of "本") should map to char 1
    let (char_start, char_end) = bytes_to_chars(text, 4, 4);
    assert_eq!(char_start, 1);
    assert_eq!(char_end, 2);
}

#[test]
fn test_bytes_to_chars_beyond_text() {
    let text = "Hello";
    let text_len = text.len();

    // byte_start beyond text should map to end
    let (char_start, char_end) = bytes_to_chars(text, text_len + 10, text_len + 20);
    assert_eq!(char_start, 5); // 5 chars in "Hello"
    assert_eq!(char_end, 5);
}

#[test]
fn test_bytes_to_chars_empty_text() {
    let text = "";
    let (char_start, char_end) = bytes_to_chars(text, 0, 0);
    assert_eq!(char_start, 0);
    assert_eq!(char_end, 0);
}

#[test]
fn test_bytes_to_chars_byte_end_at_char_boundary() {
    // Test when byte_end is exactly at a character boundary
    let text = "Hello World";
    // "Hello" = 5 bytes, 5 chars
    // Space = 1 byte at position 5, 1 char at position 5
    // "World" = 5 bytes starting at position 6, 5 chars starting at position 6

    // byte_end = 5 is at the start of the space character (char 5)
    // Since byte_end is exclusive, we want [0, 5) which is "Hello"
    // So char_end should be 5 (exclusive), meaning chars [0, 5)
    let (char_start, char_end) = bytes_to_chars(text, 0, 5);
    assert_eq!(char_start, 0);
    assert_eq!(
        char_end, 5,
        "byte_end=5 at char boundary should map to char_end=5 (exclusive)"
    );

    // Verify the extracted text
    let extracted: String = text.chars().take(char_end).skip(char_start).collect();
    assert_eq!(extracted, "Hello");
}

#[test]
fn test_discontinuous_span_total_len_byte_length() {
    // Verify that DiscontinuousSpan::total_len() returns byte length, not character length
    let text = "café"; // 4 chars, 5 bytes
    let span = DiscontinuousSpan::new(vec![0..2, 3..5]); // "ca" (2 bytes) + "é" (2 bytes) = 4 bytes total

    // total_len should be 4 bytes, not 3 characters
    assert_eq!(
        span.total_len(),
        4,
        "total_len should return byte length, not character length"
    );

    // Verify extraction works correctly
    let extracted = span.extract_text(text, " ");
    assert_eq!(extracted, "ca é");
}

#[test]
fn test_discontinuous_span_unicode_byte_vs_char() {
    // Test that DiscontinuousSpan correctly handles Unicode byte offsets
    let text = "Hello 世界"; // "Hello " = 6 bytes, "世界" = 6 bytes (3 bytes each)
                             // Total: 12 bytes, 8 chars

    // Create span with byte offsets
    let span = DiscontinuousSpan::new(vec![0..5, 6..12]); // "Hello" (5 bytes) + "世界" (6 bytes) = 11 bytes

    assert_eq!(span.total_len(), 11, "Should sum byte lengths: 5 + 6 = 11");

    let extracted = span.extract_text(text, " ");
    assert_eq!(extracted, "Hello 世界");
}

#[test]
fn test_entity_total_len_discontinuous_byte_length() {
    // Verify Entity::total_len() with discontinuous span returns byte length
    let mut entity = Entity::new(
        "severe pain",
        EntityType::Other("MISC".to_string()),
        0,
        11,
        0.9,
    );

    // Create discontinuous span with byte offsets
    let disc_span = DiscontinuousSpan::new(vec![0..6, 12..16]); // "severe" (6 bytes) + "pain" (4 bytes) = 10 bytes

    entity.set_discontinuous_span(disc_span);

    // total_len should return byte length (10), not character length
    // Note: This is documented behavior - Entity uses char offsets but
    // discontinuous spans use byte offsets
    assert_eq!(
        entity.total_len(),
        10,
        "Should return byte length for discontinuous spans"
    );
}

#[test]
fn test_entity_total_len_contiguous_char_length() {
    // Verify Entity::total_len() with contiguous span returns character length
    let entity = Entity::new("Hello", EntityType::Person, 0, 5, 0.9);
    assert_eq!(
        entity.total_len(),
        5,
        "Should return character length for contiguous entities"
    );
}

#[test]
fn test_discontinuous_span_contains_byte_offset() {
    // Verify contains() uses byte offsets
    let span = DiscontinuousSpan::new(vec![0..5, 10..15]);

    // Should return true for byte offsets within segments
    assert!(span.contains(2), "Byte offset 2 should be in first segment");
    assert!(
        span.contains(12),
        "Byte offset 12 should be in second segment"
    );
    assert!(
        !span.contains(7),
        "Byte offset 7 should not be in any segment"
    );
    assert!(
        !span.contains(20),
        "Byte offset 20 should not be in any segment"
    );
}

#[test]
fn test_bytes_to_chars_roundtrip_middle_of_char() {
    // Test that bytes_to_chars handles middle-of-character positions correctly
    // and that the result can be converted back (with some loss of precision)
    let text = "café test";
    // "café" = 5 bytes (c=0, a=1, f=2, é=3-4), 4 chars
    // " test" = 5 bytes, 5 chars

    // Test byte_start in middle of "é"
    let (char_start, char_end) = bytes_to_chars(text, 4, 5);
    assert_eq!(char_start, 3, "Middle of 'é' should map to char 3");
    assert_eq!(char_end, 4, "End should map to char 4");

    // Convert back to bytes
    let (byte_start2, byte_end2) = chars_to_bytes(text, char_start, char_end);
    // Should map back to start of "é" (byte 3) and start of next char (byte 5)
    assert_eq!(byte_start2, 3, "Should map back to start of character");
    assert_eq!(byte_end2, 5, "Should map to start of next character");
}

#[test]
fn test_bytes_to_chars_exact_boundaries() {
    // Test exact character boundaries (should work perfectly)
    let text = "Hello World";
    let (char_start, char_end) = bytes_to_chars(text, 0, 5);
    assert_eq!(char_start, 0);
    assert_eq!(char_end, 5);

    // Roundtrip should be exact
    let (byte_start2, byte_end2) = chars_to_bytes(text, char_start, char_end);
    assert_eq!(byte_start2, 0);
    assert_eq!(byte_end2, 5);
}

#[test]
fn test_bytes_to_chars_single_char() {
    // Test with single character
    let text = "A";
    let (char_start, char_end) = bytes_to_chars(text, 0, 1);
    assert_eq!(char_start, 0);
    assert_eq!(char_end, 1);
}

#[test]
fn test_bytes_to_chars_zero_length() {
    // Test zero-length range
    let text = "Hello";
    let (char_start, char_end) = bytes_to_chars(text, 2, 2);
    assert_eq!(char_start, 2);
    assert_eq!(char_end, 2); // Exclusive end, so same as start means empty range
}
