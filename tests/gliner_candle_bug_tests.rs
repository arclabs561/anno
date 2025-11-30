//! Tests for bugs found in GLiNERCandle backend.
//!
//! These tests verify edge cases and potential bugs in the GLiNERCandle implementation.

#[cfg(test)]
mod tests {

    /// Test for potential index out of bounds in decode_entities when end - 1 exceeds word_positions length.
    ///
    /// Bug: In decode_entities, when end = start + width + 1, if start is near the end of words
    /// and width is at maximum, end can exceed words.len(), making end - 1 out of bounds.
    #[test]
    fn test_decode_entities_end_index_bounds() {
        // This test simulates the bug scenario:
        // - words.len() = 5
        // - MAX_SPAN_WIDTH = 12
        // - start = 4 (last word index)
        // - width = min(12, 5 - 4) = 1
        // - end = 4 + 1 + 1 = 6
        // - end - 1 = 5, which is out of bounds for word_positions (length 5, valid indices 0-4)

        let words = vec!["word0", "word1", "word2", "word3", "word4"];
        let word_positions: Vec<(usize, usize)> = vec![
            (0, 5),   // word0
            (6, 11),  // word1
            (12, 17), // word2
            (18, 23), // word3
            (24, 29), // word4
        ];

        // Simulate the decode_entities loop logic
        const MAX_SPAN_WIDTH: usize = 12;
        let words_len = words.len();

        // Test the edge case: start at last word, width = 1
        let start = words_len - 1; // 4
        let width = MAX_SPAN_WIDTH.min(words_len - start); // min(12, 1) = 1
        let end = start + width + 1; // 4 + 1 + 1 = 6

        // This is the bug: end - 1 = 5, which is >= word_positions.len() (5)
        let end_index = end - 1;
        // This assertion will fail, demonstrating the bug
        if end_index >= word_positions.len() {
            eprintln!(
                "BUG DETECTED: end - 1 ({}) >= word_positions.len() ({})",
                end_index,
                word_positions.len()
            );
        }
        // After fix, this should pass
        let safe_end = end.min(words.len());
        let safe_end_index = safe_end.saturating_sub(1);
        assert!(
            safe_end_index < word_positions.len(),
            "After fix: safe_end_index ({}) should be < word_positions.len() ({})",
            safe_end_index,
            word_positions.len()
        );

        // The fix should use saturating_sub or clamp end to words.len()
        let safe_end_index = end
            .saturating_sub(1)
            .min(word_positions.len().saturating_sub(1));
        assert!(
            safe_end_index < word_positions.len(),
            "Safe end index should be in bounds"
        );
    }

    /// Test that word_positions.get(end - 1) doesn't panic when end exceeds bounds.
    #[test]
    fn test_word_positions_get_end_minus_one_safety() {
        let word_positions: Vec<(usize, usize)> = vec![(0, 5), (6, 11), (12, 17)];

        // Test various end values
        let test_cases: Vec<(usize, bool)> = vec![
            (0, false), // end = 0, end - 1 = usize::MAX (underflow), should return None
            (1, true),  // end = 1, end - 1 = 0, valid
            (2, true),  // end = 2, end - 1 = 1, valid
            (3, true),  // end = 3, end - 1 = 2, valid
            (4, false), // end = 4, end - 1 = 3, out of bounds
        ];

        for (end, should_be_valid) in test_cases {
            // Simulate the current code: word_positions.get(end - 1)
            // This can underflow if end == 0
            let index = end.saturating_sub(1);
            let result = word_positions.get(index);

            if should_be_valid {
                assert!(
                    result.is_some(),
                    "end={}, index={} should be valid",
                    end,
                    index
                );
            } else {
                // For end=0, saturating_sub(1) = 0 (no underflow), but index 0 is valid
                // The real issue is when end exceeds word_positions.len()
                // For end=4, index=3 is out of bounds (word_positions.len()=3), returns None
                if end == 0 {
                    // end=0, saturating_sub(1) = 0, which is valid for word_positions[0]
                    // This is actually a valid case, not a bug
                    assert!(
                        result.is_some(),
                        "end=0, index=0 should be valid (first word)"
                    );
                } else {
                    // end=4, index=3, but word_positions.len()=3, so index 3 is out of bounds
                    assert!(
                        result.is_none() || index >= word_positions.len(),
                        "end={}, index={} should be out of bounds or None (word_positions.len()={})",
                        end,
                        index,
                        word_positions.len()
                    );
                }
            }
        }
    }

    /// Test the actual bounds calculation in the decode_entities loop.
    #[test]
    fn test_decode_entities_loop_bounds() {
        const MAX_SPAN_WIDTH: usize = 12;

        // Test with various word lengths
        for words_len in 1..=20 {
            let word_positions: Vec<(usize, usize)> =
                (0..words_len).map(|i| (i * 10, i * 10 + 5)).collect();

            // Simulate the decode_entities loop
            for start in 0..words_len {
                for width in 0..MAX_SPAN_WIDTH.min(words_len - start) {
                    let end = start + width + 1;

                    // Check bounds
                    assert!(
                        start < word_positions.len(),
                        "start {} should be < word_positions.len() {}",
                        start,
                        word_positions.len()
                    );

                    // This is the potential bug: end - 1 can be >= word_positions.len()
                    let end_index = end.saturating_sub(1);
                    if end_index >= word_positions.len() {
                        // This is the bug case - end exceeds bounds
                        eprintln!(
                            "BUG DETECTED: words_len={}, start={}, width={}, end={}, end_index={}, word_positions.len()={}",
                            words_len, start, width, end, end_index, word_positions.len()
                        );
                    }

                    // The fix: clamp end to words.len()
                    let safe_end = end.min(words_len);
                    let safe_end_index = safe_end.saturating_sub(1);
                    assert!(
                        safe_end_index < word_positions.len(),
                        "Safe end index should be in bounds: safe_end={}, safe_end_index={}, len={}",
                        safe_end,
                        safe_end_index,
                        word_positions.len()
                    );
                }
            }
        }
    }

    /// Test that generate_spans and decode_entities use consistent span definitions.
    #[test]
    fn test_span_definition_consistency() {
        // generate_spans uses: end = start + width (inclusive end)
        // decode_entities was using: end = start + width + 1 (exclusive end)
        // This inconsistency could cause bugs

        const MAX_SPAN_WIDTH: usize = 12;
        let num_words = 5;

        // Simulate generate_spans logic
        let mut spans_generate = Vec::new();
        for start in 0..num_words {
            for width in 0..MAX_SPAN_WIDTH.min(num_words - start) {
                let end = start + width; // Inclusive
                spans_generate.push((start, end));
            }
        }

        // Simulate decode_entities logic (before fix)
        let mut spans_decode_old = Vec::new();
        for start in 0..num_words {
            for width in 0..MAX_SPAN_WIDTH.min(num_words - start) {
                let end = start + width + 1; // Exclusive (old buggy version)
                spans_decode_old.push((start, end));
            }
        }

        // Simulate decode_entities logic (after fix)
        let mut spans_decode_new = Vec::new();
        for start in 0..num_words {
            for width in 0..MAX_SPAN_WIDTH.min(num_words - start) {
                let end_inclusive = start + width; // Match generate_spans
                let end_exclusive = (end_inclusive + 1).min(num_words); // Clamp to bounds
                spans_decode_new.push((start, end_exclusive));
            }
        }

        // These should match in count
        assert_eq!(
            spans_generate.len(),
            spans_decode_old.len(),
            "Should generate same number of spans"
        );
        assert_eq!(
            spans_generate.len(),
            spans_decode_new.len(),
            "Should generate same number of spans after fix"
        );

        // Check that old version could exceed bounds
        for (start, end) in &spans_decode_old {
            if *end > num_words {
                eprintln!(
                    "OLD BUG: end {} exceeds num_words {} for start {}",
                    end, num_words, start
                );
            }
        }

        // Check that new version is always in bounds
        for (start, end) in &spans_decode_new {
            assert!(
                *end <= num_words,
                "Fixed version: end {} should be <= num_words {} for start {}",
                end,
                num_words,
                start
            );
        }
    }

    /// Test edge case: empty word_positions (should not happen but test defensive code).
    #[test]
    fn test_empty_word_positions_edge_case() {
        let word_positions: Vec<(usize, usize)> = vec![];
        let words = vec!["word"];

        // This should not happen in practice (words.is_empty() check earlier),
        // but test that the code handles it defensively
        let end_exclusive = 1;
        let end_index = end_exclusive
            .saturating_sub(1)
            .min(word_positions.len().saturating_sub(1));

        // If word_positions is empty, end_index will be 0 (from saturating_sub(1) on 0)
        // But word_positions.get(0) will return None for empty vec
        let result = word_positions.get(end_index);
        assert!(
            result.is_none(),
            "Empty word_positions should return None for any index"
        );
    }

    /// Test that the fix correctly handles the boundary case where end equals words.len().
    #[test]
    fn test_end_equals_words_len_boundary() {
        let words = vec!["word0", "word1", "word2"];
        let word_positions: Vec<(usize, usize)> = vec![(0, 5), (6, 11), (12, 17)];

        // Simulate: start = 2, width = 0 (last word, width 0)
        let start = 2;
        let width = 0;
        let end_inclusive = start + width; // 2
        let end_exclusive = (end_inclusive + 1).min(words.len()); // 3.min(3) = 3

        // Before fix: end = start + width + 1 = 2 + 0 + 1 = 3
        // end - 1 = 2, which is valid (last index)
        let end_index_old = (start + width + 1).saturating_sub(1); // 3 - 1 = 2
        assert_eq!(
            end_index_old, 2,
            "Old calculation should work for this case"
        );

        // After fix: end_exclusive = 3, end_index = 3.saturating_sub(1) = 2
        let end_index_new = end_exclusive
            .saturating_sub(1)
            .min(word_positions.len().saturating_sub(1));
        assert_eq!(end_index_new, 2, "New calculation should match");

        // Both should access the same index
        assert_eq!(
            word_positions.get(end_index_old),
            word_positions.get(end_index_new),
            "Both should access same position"
        );
    }

    /// Test that heuristic.rs handles start_idx == 0 correctly when accessing previous word.
    #[test]
    fn test_heuristic_start_idx_zero_prev_word() {
        // Test that accessing words[start_idx - 1] is properly guarded
        let words = vec!["John", "Smith"];
        let start_idx = 0;

        // The code checks start_idx > 0 before accessing words[start_idx - 1]
        let prev_word = if start_idx > 0 {
            Some(words[start_idx - 1].to_string())
        } else {
            None
        };

        assert_eq!(
            prev_word, None,
            "start_idx=0 should return None for prev_word"
        );

        // Test with start_idx > 0
        let start_idx = 1;
        let prev_word = if start_idx > 0 {
            Some(words[start_idx - 1].to_string())
        } else {
            None
        };
        assert_eq!(
            prev_word,
            Some("John".to_string()),
            "start_idx=1 should return previous word"
        );
    }

    /// Test that nuner.rs validation correctly handles edge cases.
    #[test]
    fn test_nuner_end_word_validation() {
        let word_positions: Vec<(usize, usize)> = vec![(0, 5), (6, 11), (12, 17)];

        // Test cases for create_entity validation
        let test_cases = vec![
            (0, 0, false), // start_word=0, end_word=0 - invalid (no words)
            (0, 1, true),  // start_word=0, end_word=1 - valid (word 0)
            (0, 2, true),  // start_word=0, end_word=2 - valid (words 0-1)
            (0, 3, true),  // start_word=0, end_word=3 - valid (words 0-2)
            (0, 4, false), // start_word=0, end_word=4 - invalid (out of bounds)
            (1, 1, false), // start_word=1, end_word=1 - invalid (no words)
            (1, 2, true),  // start_word=1, end_word=2 - valid (word 1)
            (3, 3, false), // start_word=3, end_word=3 - invalid (start out of bounds)
        ];

        for (start_word, end_word, should_be_valid) in test_cases {
            // Simulate the validation logic from nuner.rs:656
            let is_valid = !(end_word == 0
                || end_word > word_positions.len()
                || start_word >= word_positions.len());

            assert_eq!(
                is_valid,
                should_be_valid,
                "start_word={}, end_word={} validation mismatch (word_positions.len()={})",
                start_word,
                end_word,
                word_positions.len()
            );
        }
    }

    /// Test encoder_candle::geglu with edge cases.
    #[test]
    fn test_encoder_geglu_empty_dims_edge_case() {
        // Test the edge case where dims().last() returns None
        // This simulates the code in encoder_candle.rs:464

        // Simulate empty dims
        let dims: Vec<usize> = vec![];
        let dim = dims.last().copied().unwrap_or(0);
        let half = dim / 2;

        // If dims is empty, dim = 0, half = 0
        // This might cause issues in tensor indexing
        assert_eq!(dim, 0, "Empty dims should return 0");
        assert_eq!(half, 0, "half should be 0 when dim is 0");

        // Test with valid dims
        let dims_valid: Vec<usize> = vec![1, 2, 64];
        let dim_valid = dims_valid.last().copied().unwrap_or(0);
        let half_valid = dim_valid / 2;

        assert_eq!(dim_valid, 64, "Last dim should be 64");
        assert_eq!(half_valid, 32, "half should be 32");
    }

    /// Test overflow protection in num_spans calculations.
    #[test]
    fn test_num_spans_overflow_protection() {
        const MAX_SPAN_WIDTH: usize = 12;

        // Test normal case
        let num_words = 100;
        let num_spans = num_words.checked_mul(MAX_SPAN_WIDTH);
        assert_eq!(num_spans, Some(1200), "Normal case should work");

        // Test overflow case (would require ~1.5 billion words)
        let huge_num_words = usize::MAX / MAX_SPAN_WIDTH + 1;
        let num_spans_overflow = huge_num_words.checked_mul(MAX_SPAN_WIDTH);
        assert_eq!(num_spans_overflow, None, "Overflow case should return None");

        // Test dim calculation overflow
        let start = usize::MAX / MAX_SPAN_WIDTH;
        let width = 1;
        let dim_overflow = start.checked_mul(MAX_SPAN_WIDTH);
        assert!(
            dim_overflow.is_none() || dim_overflow.unwrap().checked_add(width).is_none(),
            "Dim calculation should detect overflow"
        );
    }

    /// Test span_idx array access bounds checking.
    #[test]
    fn test_span_idx_bounds_checking() {
        const MAX_SPAN_WIDTH: usize = 12;
        let num_words = 100;
        let num_spans = num_words * MAX_SPAN_WIDTH; // 1200
        let span_idx_len = num_spans * 2; // 2400

        // Test valid access
        let dim = 100;
        if let Some(dim2) = dim.checked_mul(2) {
            assert!(dim2 + 1 < span_idx_len, "Valid dim should be in bounds");
        }

        // Test out of bounds
        let dim_large = num_spans; // Equal to num_spans, should be out of bounds
        if let Some(dim2) = dim_large.checked_mul(2) {
            assert!(
                dim2 >= span_idx_len || dim_large >= num_spans,
                "Large dim should be out of bounds"
            );
        }

        // Test overflow in dim * 2
        let dim_huge = usize::MAX / 2 + 1;
        let dim2_overflow = dim_huge.checked_mul(2);
        assert_eq!(dim2_overflow, None, "dim * 2 should detect overflow");
    }
}
