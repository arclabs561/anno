use super::*;

#[test]
fn test_nuner_creation() {
    let ner = NuNER::new();
    assert_eq!(ner.model_id(), "numind/NuNER_Zero");
    assert!((ner.threshold() - 0.5).abs() < f64::EPSILON);
}

#[test]
fn test_nuner_with_custom_model() {
    let ner = NuNER::with_model("custom/model")
        .with_threshold(0.7)
        .with_labels(vec!["technology".to_string()]);

    assert_eq!(ner.model_id(), "custom/model");
    assert!((ner.threshold() - 0.7).abs() < f64::EPSILON);
    assert_eq!(ner.default_labels.len(), 1);
    assert_eq!(ner.max_input_chars(), super::MAX_INPUT_CHARS_512);
}

#[test]
fn test_nuner_4k_creation() {
    let ner = NuNER::new_4k();
    assert_eq!(ner.model_id(), "numind/NuNER_Zero-4k");
    assert_eq!(ner.max_input_chars(), super::MAX_INPUT_CHARS_4K);
}

#[test]
fn test_nuner_with_model_auto_detects_4k() {
    let ner = NuNER::with_model("numind/NuNER_Zero-4k");
    assert_eq!(ner.max_input_chars(), super::MAX_INPUT_CHARS_4K);

    let ner = NuNER::with_model("some-user/custom-nuner-4k-onnx");
    assert_eq!(ner.max_input_chars(), super::MAX_INPUT_CHARS_4K);

    let ner = NuNER::with_model("numind/NuNER_Zero");
    assert_eq!(ner.max_input_chars(), super::MAX_INPUT_CHARS_512);
}

#[test]
fn test_label_mapping() {
    assert_eq!(
        NuNER::map_label_to_entity_type("person"),
        EntityType::Person
    );
    assert_eq!(NuNER::map_label_to_entity_type("PER"), EntityType::Person);
    assert_eq!(
        NuNER::map_label_to_entity_type("organization"),
        EntityType::Organization
    );
    assert_eq!(
        NuNER::map_label_to_entity_type("custom"),
        EntityType::custom("custom", EntityCategory::Misc)
    );
}

#[test]
fn test_supported_types() {
    let ner = NuNER::new();
    let types = ner.supported_types();
    assert!(types.contains(&EntityType::Person));
    assert!(types.contains(&EntityType::Organization));
    assert!(types.contains(&EntityType::Location));
}

#[test]
fn test_empty_input() {
    let ner = NuNER::new();
    let entities = ner.extract_entities("", None).unwrap();
    assert!(entities.is_empty());
}

#[test]
fn test_not_available_without_model() {
    let ner = NuNER::new();
    assert!(!ner.is_available());
}

#[test]
#[cfg(feature = "onnx")]
fn test_create_entity_converts_byte_offsets_to_char_offsets() {
    let ner = NuNER::new();
    let text = "北京 Beijing";
    let word_positions = vec![(0usize, 6usize), (7usize, 14usize)]; // byte offsets
    let entity_types = ["loc"];
    let span_converter = crate::offset::SpanConverter::new(text);

    // Select the second word ("Beijing"): start_word=1, end_word=2 (exclusive)
    let e = ner
        .create_entity(
            text,
            &span_converter,
            &word_positions,
            1,
            2,
            0,
            0.9,
            &entity_types,
        )
        .expect("expected entity");

    assert_eq!(e.text, "Beijing");
    assert_eq!(
        (e.start(), e.end()),
        (3, 10),
        "expected char offsets for Beijing"
    );
}

#[test]
fn test_nuner_threshold_clamping() {
    let ner = NuNER::new().with_threshold(1.5);
    assert!(
        (ner.threshold() - 1.0).abs() < f64::EPSILON,
        "threshold should be clamped to 1.0"
    );

    let ner = NuNER::new().with_threshold(-0.5);
    assert!(
        ner.threshold().abs() < f64::EPSILON,
        "threshold should be clamped to 0.0"
    );
}

#[test]
fn test_nuner_with_labels_replaces() {
    let ner = NuNER::new().with_labels(vec!["custom_type".to_string()]);
    assert_eq!(ner.default_labels.len(), 1);
    assert_eq!(ner.default_labels[0], "custom_type");
}

#[test]
fn test_label_mapping_aliases() {
    // Test all documented aliases
    assert_eq!(NuNER::map_label_to_entity_type("per"), EntityType::Person);
    assert_eq!(
        NuNER::map_label_to_entity_type("PERSON"),
        EntityType::Person
    );
    assert_eq!(
        NuNER::map_label_to_entity_type("org"),
        EntityType::Organization
    );
    assert_eq!(
        NuNER::map_label_to_entity_type("company"),
        EntityType::Organization
    );
    assert_eq!(NuNER::map_label_to_entity_type("loc"), EntityType::Location);
    assert_eq!(
        NuNER::map_label_to_entity_type("place"),
        EntityType::Location
    );
    assert_eq!(NuNER::map_label_to_entity_type("gpe"), EntityType::Location);
    assert_eq!(NuNER::map_label_to_entity_type("date"), EntityType::Date);
    assert_eq!(NuNER::map_label_to_entity_type("money"), EntityType::Money);
    assert_eq!(
        NuNER::map_label_to_entity_type("currency"),
        EntityType::Money
    );
    assert_eq!(
        NuNER::map_label_to_entity_type("percent"),
        EntityType::Percent
    );
    assert_eq!(NuNER::map_label_to_entity_type("time"), EntityType::Time);
}

#[test]
fn test_label_mapping_unknown() {
    let et = NuNER::map_label_to_entity_type("vehicle");
    assert!(matches!(et, EntityType::Custom { .. }));
}

#[test]
fn test_nuner_model_metadata() {
    let ner = NuNER::new();
    assert_eq!(ner.name(), "nuner");
    assert!(!ner.description().is_empty());
    assert!(!ner.version().is_empty());
}

#[test]
fn test_nuner_capabilities() {
    let ner = NuNER::new();
    let caps = ner.capabilities();
    assert!(caps.zero_shot, "NuNER should be zero-shot capable");
}

#[test]
fn test_nuner_4k_auto_detection_case_insensitive() {
    let ner = NuNER::with_model("SomeUser/NuNER-4K-Custom");
    assert_eq!(ner.max_input_chars(), super::MAX_INPUT_CHARS_4K);
}

#[test]
#[cfg(feature = "onnx")]
fn test_create_entity_first_word() {
    let ner = NuNER::new();
    let text = "Apple Inc is a company";
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut word_positions = Vec::new();
    let mut pos = 0;
    for word in &words {
        let start = text[pos..].find(word).unwrap() + pos;
        word_positions.push((start, start + word.len()));
        pos = start + word.len();
    }
    let entity_types = ["organization"];
    let span_converter = crate::offset::SpanConverter::new(text);

    let e = ner
        .create_entity(
            text,
            &span_converter,
            &word_positions,
            0,
            2,
            0,
            0.9,
            &entity_types,
        )
        .expect("expected entity");
    assert_eq!(e.text, "Apple Inc");
    assert_eq!((e.start(), e.end()), (0, 9));
}

#[test]
#[cfg(feature = "onnx")]
fn test_create_entity_out_of_bounds() {
    let ner = NuNER::new();
    let text = "hello";
    let word_positions = vec![(0, 5)];
    let entity_types = ["misc"];
    let span_converter = crate::offset::SpanConverter::new(text);

    // start_word beyond word_positions should return None
    let e = ner.create_entity(
        text,
        &span_converter,
        &word_positions,
        5,
        6,
        0,
        0.9,
        &entity_types,
    );
    assert!(e.is_none());
}
