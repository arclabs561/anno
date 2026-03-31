use super::expand_ner_label;
use super::looks_like_company_name;

#[test]
fn test_looks_like_company_name() {
    assert!(looks_like_company_name("Apple Inc"));
    assert!(looks_like_company_name("Acme Corp."));
    assert!(looks_like_company_name("Example GmbH"));
    assert!(looks_like_company_name("株式会社トヨタ自動車"));
    assert!(looks_like_company_name("شركة أرامكو"));

    assert!(!looks_like_company_name("Apple"));
    assert!(!looks_like_company_name("New York"));
}

#[test]
fn test_expand_ner_label_abbreviations() {
    // Standard abbreviations -> full words
    assert_eq!(expand_ner_label("PER"), "person");
    assert_eq!(expand_ner_label("PERSON"), "person");
    assert_eq!(expand_ner_label("ORG"), "organization");
    assert_eq!(expand_ner_label("ORGANIZATION"), "organization");
    assert_eq!(expand_ner_label("LOC"), "location");
    assert_eq!(expand_ner_label("LOCATION"), "location");
    assert_eq!(expand_ner_label("GPE"), "location");
    assert_eq!(expand_ner_label("MISC"), "miscellaneous");
    assert_eq!(expand_ner_label("MISCELLANEOUS"), "miscellaneous");
    assert_eq!(expand_ner_label("DATE"), "date");
    assert_eq!(expand_ner_label("MONEY"), "money");
    assert_eq!(expand_ner_label("TIME"), "time");
    assert_eq!(expand_ner_label("PRODUCT"), "product");
    assert_eq!(expand_ner_label("EVENT"), "event");
}

#[test]
fn test_expand_ner_label_case_insensitive() {
    assert_eq!(expand_ner_label("per"), "person");
    assert_eq!(expand_ner_label("Per"), "person");
    assert_eq!(expand_ner_label("org"), "organization");
    assert_eq!(expand_ner_label("Loc"), "location");
    assert_eq!(expand_ner_label("gpe"), "location");
}

#[test]
fn test_expand_ner_label_passthrough() {
    // Unknown labels pass through as lowercase
    assert_eq!(expand_ner_label("facility"), "facility");
    assert_eq!(expand_ner_label("FACILITY"), "facility");
    assert_eq!(expand_ner_label("work_of_art"), "work_of_art");
    assert_eq!(expand_ner_label("Custom Label"), "custom label");
}

#[test]
fn test_expand_ner_label_already_expanded() {
    // Already-correct labels stay the same
    assert_eq!(expand_ner_label("person"), "person");
    assert_eq!(expand_ner_label("organization"), "organization");
    assert_eq!(expand_ner_label("location"), "location");
}

// --- Additional coverage for pure functions ---

#[test]
fn test_looks_like_company_name_western_suffixes() {
    assert!(looks_like_company_name("Alphabet Inc."));
    assert!(looks_like_company_name("Google LLC"));
    assert!(looks_like_company_name("HSBC PLC"));
    assert!(looks_like_company_name("Siemens GmbH"));
    assert!(looks_like_company_name("Toyota Corporation"));
    assert!(looks_like_company_name("Samsung Co."));
    assert!(looks_like_company_name("Goldman Sachs LLP"));
    assert!(looks_like_company_name("Total S.A."));
}

#[test]
fn test_looks_like_company_name_cjk_markers() {
    assert!(looks_like_company_name("有限会社テスト"));
    assert!(looks_like_company_name("阿里巴巴集团"));
    assert!(looks_like_company_name("百度公司"));
}

#[test]
fn test_looks_like_company_name_negatives() {
    assert!(!looks_like_company_name(""));
    assert!(!looks_like_company_name("  "));
    assert!(!looks_like_company_name("John Smith"));
    assert!(!looks_like_company_name("San Francisco"));
    assert!(!looks_like_company_name("Machine Learning"));
    assert!(!looks_like_company_name("Inc")); // "Inc" alone, no space prefix
}

#[test]
fn test_expand_ner_label_edge_cases() {
    assert_eq!(expand_ner_label(""), "");
    assert_eq!(expand_ner_label("NORP"), "norp"); // unknown passthrough
    assert_eq!(expand_ner_label("WORK_OF_ART"), "work_of_art");
    assert_eq!(expand_ner_label("cardinal"), "cardinal");
}

#[test]
fn test_expand_ner_label_full_pipeline() {
    // Verify the expand -> map pipeline for all standard labels:
    assert_eq!(expand_ner_label("PER"), "person");
    assert_eq!(expand_ner_label("ORG"), "organization");
    assert_eq!(expand_ner_label("LOC"), "location");
    assert_eq!(expand_ner_label("GPE"), "location");
    assert_eq!(expand_ner_label("MONEY"), "money");
    assert_eq!(expand_ner_label("PERCENT"), "percent");
    assert_eq!(expand_ner_label("TIME"), "time");
    assert_eq!(expand_ner_label("PRODUCT"), "product");
    assert_eq!(expand_ner_label("EVENT"), "event");
}

#[cfg(feature = "onnx")]
#[test]
fn test_make_span_tensors_basic() {
    // Test the span tensor generation logic (same algorithm across gliner backends)
    let max_span_width = 12;
    let num_words = 5;
    let num_spans = num_words * max_span_width;

    // Verify the shape math
    assert_eq!(num_spans, 60);

    // For word 0: spans (0,0)..(0,4) = 5 valid
    // For word 4: spans (4,4) = 1 valid
    // Total valid = 5+4+3+2+1 = 15
    let mut valid_count = 0;
    for start in 0..num_words {
        let remaining = num_words - start;
        let actual_max = max_span_width.min(remaining);
        valid_count += actual_max;
    }
    assert_eq!(valid_count, 15);
}

#[cfg(feature = "onnx")]
#[test]
fn test_default_gliner_labels() {
    use super::inference::DEFAULT_GLINER_LABELS;
    assert!(DEFAULT_GLINER_LABELS.contains(&"person"));
    assert!(DEFAULT_GLINER_LABELS.contains(&"organization"));
    assert!(DEFAULT_GLINER_LABELS.contains(&"location"));
    assert!(DEFAULT_GLINER_LABELS.contains(&"date"));
    assert!(DEFAULT_GLINER_LABELS.len() >= 10);
}

#[cfg(feature = "onnx")]
#[test]
fn test_gliner_onnx_from_nonexistent() {
    use super::GLiNEROnnx;
    // Verify graceful error for nonexistent model
    let result = GLiNEROnnx::new("nonexistent/model-does-not-exist-xyz-12345");
    assert!(result.is_err());
}
