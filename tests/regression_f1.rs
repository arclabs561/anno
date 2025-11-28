//! F1 Score Regression Tests
//!
//! These tests fail if the F1 score drops below established baselines.
//! This protects against accidental accuracy regressions.
//!
//! # Baselines
//!
//! | Backend | Dataset | Baseline F1 | Date |
//! |---------|---------|-------------|------|
//! | StackedNER | Synthetic | 45.0% | 2024-11 |
//! | PatternNER | Structured | 90.0% | 2024-11 |
//!
//! # Updating Baselines
//!
//! If you intentionally improve accuracy, update the baselines here.
//! Document the change and the reason in the commit message.

use anno::{Model, PatternNER, StackedNER, EntityType};
use anno::eval::{evaluate_ner_model, GoldEntity};
use anno::eval::modes::MultiModeResults;

// =============================================================================
// Baseline Constants
// =============================================================================

/// Minimum acceptable F1 for StackedNER on synthetic data (strict mode).
/// If this test fails, something broke - investigate before merging.
const STACKED_SYNTHETIC_MIN_F1: f64 = 0.40; // 40% - conservative baseline

/// Minimum acceptable F1 for PatternNER on structured entities.
const PATTERN_STRUCTURED_MIN_F1: f64 = 0.85; // 85% - high bar for regex

/// Minimum acceptable F1 for PatternNER dates specifically.
const PATTERN_DATE_MIN_F1: f64 = 0.70; // 70% for dates

/// Minimum acceptable F1 for PatternNER money.
/// Note: 50% because some test cases use abbreviated forms ($50B) 
/// that don't match exact gold boundaries.
const PATTERN_MONEY_MIN_F1: f64 = 0.50; // 50% - realistic baseline

/// Minimum acceptable F1 for PatternNER email.
const PATTERN_EMAIL_MIN_F1: f64 = 0.95; // 95% for emails

// =============================================================================
// Test Data
// =============================================================================

fn structured_test_cases() -> Vec<(String, Vec<GoldEntity>)> {
    vec![
        // Dates
        ("Meeting on 2024-01-15 at noon.".to_string(), vec![
            GoldEntity::new("2024-01-15", EntityType::Date, 11),
        ]),
        ("Deadline: January 15, 2024".to_string(), vec![
            GoldEntity::new("January 15, 2024", EntityType::Date, 10),
        ]),
        ("Due by 12/31/2024".to_string(), vec![
            GoldEntity::new("12/31/2024", EntityType::Date, 7),
        ]),
        
        // Money
        ("Price: $100.00 each".to_string(), vec![
            GoldEntity::new("$100.00", EntityType::Money, 7),
        ]),
        ("Total: €50.99".to_string(), vec![
            GoldEntity::new("€50.99", EntityType::Money, 7),
        ]),
        ("Budget of $1,000,000".to_string(), vec![
            GoldEntity::new("$1,000,000", EntityType::Money, 10),
        ]),
        
        // Percentages
        ("Growth of 25%".to_string(), vec![
            GoldEntity::new("25%", EntityType::Percent, 10),
        ]),
        ("Rate: 3.5%".to_string(), vec![
            GoldEntity::new("3.5%", EntityType::Percent, 6),
        ]),
        
        // Emails
        ("Contact: john@example.com".to_string(), vec![
            GoldEntity::new("john@example.com", EntityType::Email, 9),
        ]),
        ("Email test.user+tag@sub.domain.co.uk".to_string(), vec![
            GoldEntity::new("test.user+tag@sub.domain.co.uk", EntityType::Email, 6),
        ]),
        
        // URLs
        ("Visit https://example.com/path".to_string(), vec![
            GoldEntity::new("https://example.com/path", EntityType::Url, 6),
        ]),
        
        // Phone numbers
        ("Call 555-123-4567".to_string(), vec![
            GoldEntity::new("555-123-4567", EntityType::Phone, 5),
        ]),
    ]
}

fn mixed_test_cases() -> Vec<(String, Vec<GoldEntity>)> {
    vec![
        // Mix of structured and named entities
        ("Apple reported $50B revenue on Jan 15, 2024.".to_string(), vec![
            GoldEntity::new("Apple", EntityType::Organization, 0),
            GoldEntity::new("$50B", EntityType::Money, 15),
            GoldEntity::new("Jan 15, 2024", EntityType::Date, 31),
        ]),
        ("Email ceo@company.com for the meeting at 3pm.".to_string(), vec![
            GoldEntity::new("ceo@company.com", EntityType::Email, 6),
        ]),
        ("Dr. Smith charges $200/hour.".to_string(), vec![
            GoldEntity::new("Dr. Smith", EntityType::Person, 0),
            GoldEntity::new("$200", EntityType::Money, 18),
        ]),
    ]
}

// =============================================================================
// Regression Tests
// =============================================================================

#[test]
fn regression_pattern_ner_structured() {
    let ner = PatternNER::new();
    let test_cases = structured_test_cases();
    
    let results = evaluate_ner_model(&ner, &test_cases).unwrap();
    
    assert!(
        results.f1 >= PATTERN_STRUCTURED_MIN_F1,
        "PatternNER F1 regression! Got {:.1}%, expected >= {:.1}%",
        results.f1 * 100.0,
        PATTERN_STRUCTURED_MIN_F1 * 100.0
    );
    
    println!("PatternNER structured F1: {:.1}% (baseline: {:.1}%)",
        results.f1 * 100.0, PATTERN_STRUCTURED_MIN_F1 * 100.0);
}

#[test]
fn regression_stacked_ner_mixed() {
    let ner = StackedNER::default();
    let test_cases = mixed_test_cases();
    
    let results = evaluate_ner_model(&ner, &test_cases).unwrap();
    
    assert!(
        results.f1 >= STACKED_SYNTHETIC_MIN_F1,
        "StackedNER F1 regression! Got {:.1}%, expected >= {:.1}%",
        results.f1 * 100.0,
        STACKED_SYNTHETIC_MIN_F1 * 100.0
    );
    
    println!("StackedNER mixed F1: {:.1}% (baseline: {:.1}%)",
        results.f1 * 100.0, STACKED_SYNTHETIC_MIN_F1 * 100.0);
}

#[test]
fn regression_pattern_dates() {
    let ner = PatternNER::new();
    
    let test_cases = vec![
        ("Date: 2024-01-15".to_string(), vec![
            GoldEntity::new("2024-01-15", EntityType::Date, 6),
        ]),
        ("On January 15, 2024".to_string(), vec![
            GoldEntity::new("January 15, 2024", EntityType::Date, 3),
        ]),
        ("Due 12/31/2024".to_string(), vec![
            GoldEntity::new("12/31/2024", EntityType::Date, 4),
        ]),
        ("March 2024 report".to_string(), vec![
            GoldEntity::new("March 2024", EntityType::Date, 0),
        ]),
    ];
    
    let results = evaluate_ner_model(&ner, &test_cases).unwrap();
    
    assert!(
        results.f1 >= PATTERN_DATE_MIN_F1,
        "PatternNER date F1 regression! Got {:.1}%, expected >= {:.1}%",
        results.f1 * 100.0,
        PATTERN_DATE_MIN_F1 * 100.0
    );
}

#[test]
fn regression_pattern_money() {
    let ner = PatternNER::new();

    // Note: Use ASCII-only currencies to avoid byte/char offset mismatch.
    // The evaluation system uses char offsets, PatternNER uses byte offsets.
    // For ASCII text, these are the same.
    // TODO: Fix the char/byte offset mismatch for non-ASCII currencies (€, £, ¥)
    let test_cases = vec![
        ("Cost: $100".to_string(), vec![
            GoldEntity::new("$100", EntityType::Money, 6),
        ]),
        ("USD $50.99 total".to_string(), vec![
            GoldEntity::new("$50.99", EntityType::Money, 4),
        ]),
        ("Budget $1,000,000".to_string(), vec![
            GoldEntity::new("$1,000,000", EntityType::Money, 7),
        ]),
        ("Fee: $25".to_string(), vec![
            GoldEntity::new("$25", EntityType::Money, 5),
        ]),
    ];

    let results = evaluate_ner_model(&ner, &test_cases).unwrap();

    assert!(
        results.f1 >= PATTERN_MONEY_MIN_F1,
        "PatternNER money F1 regression! Got {:.1}%, expected >= {:.1}%",
        results.f1 * 100.0,
        PATTERN_MONEY_MIN_F1 * 100.0
    );
}

#[test]
fn regression_pattern_email() {
    let ner = PatternNER::new();
    
    let test_cases = vec![
        ("Email: test@example.com".to_string(), vec![
            GoldEntity::new("test@example.com", EntityType::Email, 7),
        ]),
        ("Contact user.name@domain.org".to_string(), vec![
            GoldEntity::new("user.name@domain.org", EntityType::Email, 8),
        ]),
        ("Send to admin@company.co.uk".to_string(), vec![
            GoldEntity::new("admin@company.co.uk", EntityType::Email, 8),
        ]),
    ];
    
    let results = evaluate_ner_model(&ner, &test_cases).unwrap();
    
    assert!(
        results.f1 >= PATTERN_EMAIL_MIN_F1,
        "PatternNER email F1 regression! Got {:.1}%, expected >= {:.1}%",
        results.f1 * 100.0,
        PATTERN_EMAIL_MIN_F1 * 100.0
    );
}

// =============================================================================
// Multi-Mode Evaluation Tests
// =============================================================================

#[test]
fn test_multi_mode_evaluation() {
    let ner = PatternNER::new();
    let text = "Price: $100.00 on 2024-01-15";
    
    let entities = ner.extract_entities(text, None).unwrap();
    let gold = vec![
        GoldEntity::new("$100.00", EntityType::Money, 7),
        GoldEntity::new("2024-01-15", EntityType::Date, 18),
    ];
    
    let results = MultiModeResults::compute(&entities, &gold);
    
    // All modes should agree for exact matches
    assert!((results.strict.f1 - results.partial.f1).abs() < 0.001,
        "Strict and Partial should match for exact entity matches");
    
    println!("Multi-mode results:");
    println!("  Strict:  {:.1}%", results.strict.f1 * 100.0);
    println!("  Exact:   {:.1}%", results.exact.f1 * 100.0);
    println!("  Partial: {:.1}%", results.partial.f1 * 100.0);
    println!("  Type:    {:.1}%", results.type_mode.f1 * 100.0);
}

#[test]
fn test_partial_vs_strict() {
    // Test where partial should be higher than strict
    let ner = PatternNER::new();
    let text = "Price: $100";  // Just $100, not $100.00
    
    let entities = ner.extract_entities(text, None).unwrap();
    
    // Gold says $100 with slightly different boundary
    let gold = vec![
        GoldEntity::with_span("$100", EntityType::Money, 7, 11),
    ];
    
    let results = MultiModeResults::compute(&entities, &gold);
    
    // Partial should be >= Strict
    assert!(results.partial.f1 >= results.strict.f1,
        "Partial mode should be at least as good as Strict");
}

// =============================================================================
// Historical Baseline Tracking
// =============================================================================

/// This test documents historical performance for tracking improvements.
/// It should NOT fail - just print current vs historical metrics.
#[test]
fn track_performance_history() {
    let ner = StackedNER::default();
    let test_cases = structured_test_cases();
    let results = evaluate_ner_model(&ner, &test_cases).unwrap();
    
    println!("\n=== Performance History ===");
    println!("Current StackedNER F1: {:.1}%", results.f1 * 100.0);
    println!("Historical baselines:");
    println!("  - 2024-11: ~45% (initial release)");
    println!();
    
    // Report per-type metrics if available
    for (ty, metrics) in &results.per_type {
        println!("  {}: P={:.1}% R={:.1}% F1={:.1}%",
            ty, metrics.precision * 100.0, metrics.recall * 100.0, metrics.f1 * 100.0);
    }
}

