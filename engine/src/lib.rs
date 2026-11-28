use anno::{EntityCategory, EntityType, HashMapLexicon, LexiconNER, Model};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn extract(text: &str, reader: &str, vessels: bool) -> Result<String, JsValue> {
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
    let entities = model.extract_entities(text, None).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows: Vec<_> = entities.into_iter().map(|e| serde_json::json!({
        "text": e.text, "label": e.entity_type.as_label(), "start": e.start(), "end": e.end()
    })).collect();
    Ok(serde_json::Value::Array(rows).to_string())
}
