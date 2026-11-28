use anno::{EntityCategory, EntityType, HashMapLexicon, LexiconNER, Model};
use wasm_bindgen::prelude::*;

fn mentions(text: &str, reader: &str, vessels: bool) -> Result<Vec<anno::Entity>, JsValue> {
    if text.len() > 32_000 { return Err(JsValue::from_str("Log exceeds 32 KB")); }
    let model: Box<dyn Model> = match reader {
        "registry" => {
            let mut lexicon = HashMapLexicon::new("Saltglass port registry");
            for (name, kind) in [
                ("Iona Vale", EntityType::Person), ("Vale", EntityType::Person),
                ("Saltglass Reach", EntityType::Location), ("Low Bell", EntityType::Location),
                ("Brackwater Packet Company", EntityType::Organization),
            ] { lexicon.insert(&name.to_lowercase(), kind, 1.0); }
            if vessels {
                lexicon.insert("kestrel", EntityType::custom("VESSEL", EntityCategory::Misc), 1.0);
            }
            Box::new(LexiconNER::new(lexicon))
        }
        "patterns" => Box::new(anno::RegexNER::new()),
        "heuristic" => Box::new(anno::backends::HeuristicNER::new()),
        _ => return Err(JsValue::from_str("Unknown reader")),
    };
    model.extract_entities(text, None).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[wasm_bindgen]
pub fn extract(text: &str, reader: &str, vessels: bool) -> Result<String, JsValue> {
    let entities = mentions(text, reader, vessels)?;
    let rows: Vec<_> = entities.into_iter().map(|e| serde_json::json!({
        "text": e.text, "label": e.entity_type.as_label(), "start": e.start(), "end": e.end()
    })).collect();
    Ok(serde_json::Value::Array(rows).to_string())
}

#[wasm_bindgen]
pub fn inspect(text: &str, reader: &str, vessels: bool, task: &str) -> Result<String, JsValue> {
    use anno::backends::coref::simple::SimpleCorefResolver;
    use anno::backends::inference::extract_relation_triples_simple;
    use serde_json::json;
    use std::collections::BTreeMap;

    if text.len() > 32_000 {
        return Err(JsValue::from_str("Log exceeds 32 KB"));
    }
    if task == "redact" {
        return Ok(anno::pii::redact_patterns(text));
    }
    let entities = mentions(text, reader, vessels)?;
    if entities.len() > 96 {
        return Err(JsValue::from_str("Choose a shorter passage for pair analysis (up to 96 mentions)."));
    }
    match task {
        "links" => {
            let mut groups = BTreeMap::new();
            for entity in SimpleCorefResolver::default().resolve_entities(&entities) {
                if let Some(id) = entity.canonical_id {
                    groups.entry(id).or_insert_with(Vec::new).push(json!({
                        "text":entity.text, "start":entity.start(), "end":entity.end()
                    }));
                }
            }
            Ok(json!(groups.into_values().filter(|g| g.len() > 1).collect::<Vec<_>>()).to_string())
        }
        "relations" => {
            let triples = extract_relation_triples_simple(&entities, text,
                &["FOUNDED","WORKS_FOR","LOCATED_IN"], &Default::default());
            Ok(json!(triples.iter().map(|r| json!({
                "head":entities[r.head_idx].text, "predicate":r.relation_type,
                "tail":entities[r.tail_idx].text
            })).collect::<Vec<_>>()).to_string())
        }
        _ => Err(JsValue::from_str("Unknown experiment")),
    }
}
