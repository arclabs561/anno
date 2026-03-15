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
