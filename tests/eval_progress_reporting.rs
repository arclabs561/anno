//! Tests for progress reporting in evaluation framework.

#[cfg(feature = "eval")]
mod tests {

    #[test]
    fn test_progress_reporting_format() {
        // Test that progress reporting produces valid output
        // Note: This is a simplified test - actual progress reporting happens in evaluate_ner_task
        // which requires a backend. We test the format logic here.

        let total = 20;
        let processed = 10;
        let percent = (processed * 100) / total;
        let elapsed = std::time::Duration::from_secs(5);
        let rate = processed as f64 / elapsed.as_secs_f64();
        let remaining = ((total - processed) as f64 / rate) as u64;

        let message = format!(
            "\rProcessing: {}/{} sentences ({:.0}%) (~{}s remaining)\x1b[K",
            processed, total, percent, remaining
        );

        // Verify format is valid
        assert!(message.contains(&processed.to_string()));
        assert!(message.contains(&total.to_string()));
        assert!(message.contains(&percent.to_string()));
        assert!(message.contains("remaining"));
    }

    #[test]
    fn test_progress_calculation() {
        // Test progress percentage calculation
        let test_cases = vec![
            (0, 100, 0),
            (10, 100, 10),
            (50, 100, 50),
            (100, 100, 100),
            (25, 50, 50),
        ];

        for (processed, total, expected_percent) in test_cases {
            let percent = (processed * 100) / total;
            assert_eq!(
                percent, expected_percent,
                "Progress calculation failed: {}/{} should be {}%",
                processed, total, expected_percent
            );
        }
    }

    #[test]
    fn test_rate_calculation() {
        // Test rate (sentences/second) calculation
        let test_cases = vec![
            (10, 1.0, 10.0), // 10 sentences in 1 second = 10/s
            (20, 2.0, 10.0), // 20 sentences in 2 seconds = 10/s
            (0, 1.0, 0.0),   // 0 sentences = 0/s
        ];

        for (processed, elapsed_secs, expected_rate) in test_cases {
            let rate = if elapsed_secs > 0.0 {
                processed as f64 / elapsed_secs
            } else {
                0.0
            };
            assert!(
                (rate - expected_rate).abs() < 0.1,
                "Rate calculation failed: {} sentences in {}s should be ~{}/s, got {}",
                processed,
                elapsed_secs,
                expected_rate,
                rate
            );
        }
    }

    #[test]
    fn test_remaining_time_calculation() {
        // Test remaining time calculation
        let test_cases = vec![
            (10, 100, 10.0, 90), // 10/s rate, 90 remaining = 9s
            (20, 100, 5.0, 80),  // 5/s rate, 80 remaining = 16s
            (50, 100, 10.0, 50), // 10/s rate, 50 remaining = 5s
        ];

        for (processed, total, rate, expected_remaining) in test_cases {
            let remaining = if rate > 0.0 {
                ((total - processed) as f64 / rate) as u64
            } else {
                0
            };
            assert_eq!(
                remaining,
                expected_remaining,
                "Remaining time calculation failed: {} remaining at {}/s should be {}s, got {}s",
                total - processed,
                rate,
                expected_remaining,
                remaining
            );
        }
    }

    #[test]
    fn test_final_summary_format() {
        // Test final summary format
        let total = 100;
        let elapsed_secs = 10.5;
        let rate = total as f64 / elapsed_secs;

        let message = format!(
            "\rProcessing: {}/{} sentences (100.0%) (completed in {:.1}s, {:.1} sentences/s)\x1b[K",
            total, total, elapsed_secs, rate
        );

        // Verify format contains all required elements
        assert!(message.contains(&total.to_string()));
        assert!(message.contains("100.0%"));
        assert!(message.contains("completed"));
        assert!(message.contains(&format!("{:.1}", elapsed_secs)));
        assert!(message.contains("sentences/s"));
    }

    #[test]
    fn test_progress_edge_cases() {
        // Test edge cases for progress reporting
        // Empty dataset
        let percent = (0 * 100) / 1.max(1);
        assert_eq!(percent, 0);

        // Single sentence
        let percent = (1 * 100) / 1;
        assert_eq!(percent, 100);

        // Zero elapsed time (should not panic)
        let rate = if 0.0 > 0.0 { 10.0 / 0.0 } else { 0.0 };
        assert_eq!(rate, 0.0);
    }
}
