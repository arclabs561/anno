//! Tests for edge cases in offset conversion functions.
//!
//! These tests specifically target the bugs fixed:
//! - bytes_to_chars() handling byte_start in middle of multi-byte characters
//! - Edge cases with out-of-bounds byte offsets
//! - Unicode text with multi-byte characters

use anno::offset::bytes_to_chars;

#[test]
fn test_bytes_to_chars_middle_of_multibyte_char() {
    // "café" where "é" is 2 bytes: [0xC3, 0xA9] at bytes 3-4
    let text = "café";
    // Bytes: c=0, a=1, f=2, é=[3,4]
    // Chars: c=0, a=1, f=2, é=3

    // Test byte_start at start of "é" (byte 3)
    let (char_start, char_end) = bytes_to_chars(text, 3, 5);
    assert_eq!(char_start, 3, "Byte 3 (start of é) should map to char 3");
    assert_eq!(
        char_end, 4,
        "Byte 5 (end of text) should map to char 4 (exclusive)"
    );

    // Test byte_start in middle of "é" (byte 4 - second byte of é)
    // Should map to the character containing that byte (char 3)
    let (char_start, char_end) = bytes_to_chars(text, 4, 5);
    assert_eq!(char_start, 3, "Byte 4 (middle of é) should map to char 3");
    assert_eq!(
        char_end, 4,
        "Byte 5 (end of text) should map to char 4 (exclusive)"
    );

    // Test range that includes middle of é
    let (char_start, char_end) = bytes_to_chars(text, 2, 4);
    assert_eq!(char_start, 2, "Byte 2 (f) should map to char 2");
    assert_eq!(char_end, 4, "Byte 4 (end of é) should map to char 4");
}

#[test]
fn test_bytes_to_chars_emoji_middle() {
    // "Hello 🌍" - emoji is 4 bytes
    let text = "Hello 🌍";
    // Bytes: H=0, e=1, l=2, l=3, o=4, space=5, 🌍=[6,7,8,9]
    // Chars: H=0, e=1, l=2, l=3, o=4, space=5, 🌍=6

    // Test byte_start in middle of emoji (byte 7, 8, or 9)
    let (char_start, char_end) = bytes_to_chars(text, 7, 10);
    assert_eq!(char_start, 6, "Byte 7 (middle of 🌍) should map to char 6");
    assert_eq!(
        char_end, 7,
        "Byte 10 (beyond text) should map to char 7 (exclusive)"
    );

    // Test byte_start at byte 8 (middle of emoji)
    let (char_start, char_end) = bytes_to_chars(text, 8, 10);
    assert_eq!(char_start, 6, "Byte 8 (middle of 🌍) should map to char 6");
    assert_eq!(
        char_end, 7,
        "Byte 10 (beyond text) should map to char 7 (exclusive)"
    );
}

#[test]
fn test_bytes_to_chars_cjk_middle() {
    // "北京" - each character is 3 bytes
    let text = "北京";
    // Bytes: 北=[0,1,2], 京=[3,4,5]
    // Chars: 北=0, 京=1

    // Test byte_start in middle of first character (byte 1)
    let (char_start, char_end) = bytes_to_chars(text, 1, 4);
    assert_eq!(char_start, 0, "Byte 1 (middle of 北) should map to char 0");
    assert_eq!(
        char_end, 2,
        "Byte 4 (start of 京) should map to char 2 (exclusive)"
    );

    // Test byte_start in middle of second character (byte 4)
    let (char_start, char_end) = bytes_to_chars(text, 4, 6);
    assert_eq!(char_start, 1, "Byte 4 (middle of 京) should map to char 1");
    assert_eq!(
        char_end, 2,
        "Byte 6 (end of text) should map to char 2 (exclusive)"
    );
}

#[test]
fn test_bytes_to_chars_out_of_bounds() {
    let text = "Hello";
    // Bytes: 0-4, length=5
    // Chars: 0-4, length=5

    // Test byte_start beyond text length
    let (char_start, char_end) = bytes_to_chars(text, 10, 15);
    assert_eq!(
        char_start, 5,
        "Byte 10 (beyond text) should map to char 5 (end)"
    );
    assert_eq!(
        char_end, 5,
        "Byte 15 (beyond text) should map to char 5 (end)"
    );

    // Test byte_end beyond text length but byte_start valid
    let (char_start, char_end) = bytes_to_chars(text, 2, 10);
    assert_eq!(char_start, 2, "Byte 2 should map to char 2");
    assert_eq!(
        char_end, 5,
        "Byte 10 (beyond text) should map to char 5 (end)"
    );
}

#[test]
fn test_bytes_to_chars_exact_boundaries() {
    let text = "café";
    // Test exact character boundaries
    // "café" = bytes 0,1,2,[3,4] = chars 0,1,2,3
    let (char_start, char_end) = bytes_to_chars(text, 0, 3);
    assert_eq!(char_start, 0, "Byte 0 should map to char 0");
    assert_eq!(
        char_end, 3,
        "Byte 3 (start of é) should map to char 3 (exclusive, so range is [0,3))"
    );

    let (char_start, char_end) = bytes_to_chars(text, 3, 5);
    assert_eq!(char_start, 3, "Byte 3 should map to char 3");
    assert_eq!(
        char_end, 4,
        "Byte 5 (beyond text) should map to char 4 (exclusive end)"
    );
}

#[test]
fn test_bytes_to_chars_empty_text() {
    let text = "";

    // Any byte offset in empty text should map to char 0
    let (char_start, char_end) = bytes_to_chars(text, 0, 0);
    assert_eq!(char_start, 0);
    assert_eq!(char_end, 0);

    let (char_start, char_end) = bytes_to_chars(text, 5, 10);
    assert_eq!(char_start, 0);
    assert_eq!(char_end, 0);
}

#[test]
fn test_bytes_to_chars_ascii_text() {
    // ASCII text - byte offsets == character offsets
    let text = "Hello World";

    for i in 0..text.len() {
        let (char_start, char_end) = bytes_to_chars(text, i, i + 1);
        assert_eq!(char_start, i, "ASCII byte {} should map to char {}", i, i);
        assert_eq!(
            char_end,
            i + 1,
            "ASCII byte {} end should map to char {} (exclusive)",
            i + 1,
            i + 1
        );
    }
}

#[test]
fn test_bytes_to_chars_mixed_unicode() {
    // Mix of ASCII, 2-byte, 3-byte, and 4-byte characters
    let text = "Aé中🌍";
    // Bytes: A=0, é=[1,2], 中=[3,4,5], 🌍=[6,7,8,9]
    // Chars: A=0, é=1, 中=2, 🌍=3

    // Test various byte positions
    // "Aé中🌍" = A(0), é(1-2), 中(3-5), 🌍(6-9)
    // Chars: A=0, é=1, 中=2, 🌍=3
    let (char_start, char_end) = bytes_to_chars(text, 0, 10);
    assert_eq!(char_start, 0);
    assert_eq!(
        char_end, 4,
        "Byte 10 (beyond text) should map to char 4 (exclusive)"
    );

    // Test byte in middle of 中 (byte 4)
    let (char_start, char_end) = bytes_to_chars(text, 4, 6);
    assert_eq!(char_start, 2, "Byte 4 (middle of 中) should map to char 2");
    assert_eq!(
        char_end, 3,
        "Byte 6 (start of 🌍) should map to char 3 (exclusive)"
    );

    // Test byte in middle of 🌍 (byte 7)
    let (char_start, char_end) = bytes_to_chars(text, 7, 10);
    assert_eq!(char_start, 3, "Byte 7 (middle of 🌍) should map to char 3");
    assert_eq!(
        char_end, 4,
        "Byte 10 (beyond text) should map to char 4 (exclusive)"
    );
}

#[test]
fn test_bytes_to_chars_roundtrip_with_middle_bytes() {
    // Test that even when byte_start is in middle of char, we can roundtrip
    let text = "café";

    // Start with character offsets
    let (char_start, char_end) = (2, 4); // "f" to end

    // Convert to bytes (should give us byte boundaries)
    use anno::offset::chars_to_bytes;
    let (byte_start, byte_end) = chars_to_bytes(text, char_start, char_end);

    // Convert back to chars - should get same char offsets
    let (char_start2, char_end2) = bytes_to_chars(text, byte_start, byte_end);
    assert_eq!(char_start, char_start2);
    assert_eq!(char_end, char_end2);
}
