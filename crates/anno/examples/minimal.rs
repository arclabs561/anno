use anno::{Model, StackedNER};

fn main() {
    let ner = StackedNER::default();
    let text = "John met Mary in Paris.";

    // The default build has ONNX enabled; if you want a no-ML-deps build, compile with
    // `--no-default-features` and this will still work (it will choose the best available
    // non-ML backend internally).
    let entities = ner
        .extract_entities(text, None)
        .expect("NER should succeed");

    for e in entities {
        println!(
            "{} [{}] ({},{}) {:.2}",
            e.text,
            e.entity_type,
            e.start(),
            e.end(),
            e.confidence
        );
    }
}
