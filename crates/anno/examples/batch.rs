/// Extract entities from multiple documents at once.
///
/// `anno::extract_batch` processes documents in parallel (when `rayon`
/// is available) and returns one result per document.
///
/// ```sh
/// cargo run --example batch
/// ```
///
/// Example output:
///
/// ```text
/// Doc 1: 2 entities
///   Marie Curie [PER]
///   Paris [LOC]
/// Doc 2: 2 entities
///   Alan Turing [PER]
///   Bletchley Park [ORG]
/// Doc 3: 2 entities
///   Grace Hopper [PER]
///   COBOL [misc]
/// ```
fn main() {
    let docs = [
        "Marie Curie moved to Paris.",
        "Alan Turing worked at Bletchley Park.",
        "Grace Hopper helped develop COBOL.",
    ];

    let results = anno::extract_batch(&docs);

    for (i, result) in results.iter().enumerate() {
        match result {
            Ok(entities) => {
                println!("Doc {}: {} entities", i + 1, entities.len());
                for e in entities {
                    println!("  {} [{}]", e.text, e.entity_type);
                }
            }
            Err(e) => println!("Doc {}: error: {}", i + 1, e),
        }
    }
}
