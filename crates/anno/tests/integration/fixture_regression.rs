//! Validates CLI fixture documents against expected.json.
//!
//! These fixtures were created for QA but had no automated consumer.
//! This test ensures the extraction pipeline finds expected entities
//! in each domain-specific document.

use anno::{Entity, EntityType, Model};
use std::collections::HashMap;

const FIXTURES_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../anno-cli/tests/fixtures");

#[derive(serde::Deserialize)]
struct Expected {
    documents: HashMap<String, DocExpected>,
}

#[derive(serde::Deserialize)]
#[allow(non_snake_case)]
struct DocExpected {
    min_entities: usize,
    #[serde(default)]
    must_find_PER: Vec<String>,
    #[serde(default)]
    must_find_LOC: Vec<String>,
    #[serde(default)]
    must_find_ORG: Vec<String>,
    #[serde(default)]
    must_not_contain: Vec<String>,
}

fn load_expected() -> Expected {
    let path = format!("{FIXTURES_DIR}/expected.json");
    let data = std::fs::read_to_string(&path).expect("expected.json must exist");
    serde_json::from_str(&data).expect("expected.json must be valid")
}

fn read_fixture(name: &str) -> String {
    let path = format!("{FIXTURES_DIR}/{name}.txt");
    std::fs::read_to_string(&path).unwrap_or_else(|_| {
        let html_path = format!("{FIXTURES_DIR}/{name}.html");
        std::fs::read_to_string(&html_path)
            .unwrap_or_else(|_| panic!("fixture {name}.txt or {name}.html must exist"))
    })
}

fn extract_from_fixture(name: &str) -> Vec<Entity> {
    let text = read_fixture(name);
    let ner = anno::StackedNER::default();
    ner.extract_entities(&text, None)
        .unwrap_or_else(|e| panic!("extraction failed for {name}: {e}"))
}

fn entity_texts_of_type(entities: &[Entity], ty: EntityType) -> Vec<String> {
    entities
        .iter()
        .filter(|e| e.entity_type == ty)
        .map(|e| e.text.clone())
        .collect()
}

/// Non-ONNX stacked backend: check minimum entity counts, must-not-contain
/// (HTML stripping), and that at least some entities of each expected type
/// are found. Full PER/ORG recall requires ONNX (see ignored test below).
#[test]
fn fixture_documents_structure_and_min_counts() {
    let expected = load_expected();

    for (doc_name, doc_expected) in &expected.documents {
        let entities = extract_from_fixture(doc_name);

        // Check minimum entity count
        assert!(
            entities.len() >= doc_expected.min_entities,
            "{doc_name}: expected >= {} entities, got {}",
            doc_expected.min_entities,
            entities.len()
        );

        // Check must-not-contain (HTML stripping regression)
        let all_texts: Vec<&str> = entities.iter().map(|e| e.text.as_str()).collect();
        for forbidden in &doc_expected.must_not_contain {
            assert!(
                !all_texts.contains(&forbidden.as_str()),
                "{doc_name}: found forbidden entity '{forbidden}'"
            );
        }
    }
}

/// Full PER/ORG/LOC recall requires ONNX backends.
/// Run with: cargo nextest run -p anno-lib --features onnx --test fixture_regression -- --ignored
#[cfg(feature = "onnx")]
#[test]
#[ignore]
fn fixture_documents_full_recall_with_onnx() {
    let expected = load_expected();

    for (doc_name, doc_expected) in &expected.documents {
        let text = read_fixture(doc_name);
        let ner = anno::StackedNER::default();
        let entities = ner
            .extract_entities(&text, None)
            .unwrap_or_else(|e| panic!("extraction failed for {doc_name}: {e}"));

        let per_texts = entity_texts_of_type(&entities, EntityType::Person);
        for must in &doc_expected.must_find_PER {
            assert!(
                per_texts.iter().any(|t| t.contains(must.as_str())),
                "{doc_name}: missing PER '{must}'. Found PER: {per_texts:?}"
            );
        }

        let loc_texts = entity_texts_of_type(&entities, EntityType::Location);
        for must in &doc_expected.must_find_LOC {
            assert!(
                loc_texts.iter().any(|t| t.contains(must.as_str())),
                "{doc_name}: missing LOC '{must}'. Found LOC: {loc_texts:?}"
            );
        }

        let org_texts = entity_texts_of_type(&entities, EntityType::Organization);
        for must in &doc_expected.must_find_ORG {
            assert!(
                org_texts.iter().any(|t| t.contains(must.as_str())),
                "{doc_name}: missing ORG '{must}'. Found ORG: {org_texts:?}"
            );
        }
    }
}
