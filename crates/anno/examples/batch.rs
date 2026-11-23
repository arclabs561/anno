/// Extract entities from multiple documents at once.
///
/// `anno::extract_batch` processes documents sequentially using a single
/// `StackedNER` instance and returns one result per document. For parallel
/// extraction across cores, enable the `parallel` feature and call
/// `Model::par_extract_batch` on a shared model reference.
///
/// ```sh
/// cargo run -p anno --no-default-features --example batch
/// cargo run -p anno --no-default-features --example batch --features parallel   # uses rayon
/// ```
///
/// Output with either offline command:
///
/// ```text
/// Doc 1: 2 entities
///   Marie Curie [PER]
///   Paris [LOC]
/// Doc 2: 2 entities
///   Alan Turing [PER]
///   Bletchley Park [LOC]
/// Doc 3: 2 entities
///   Grace Hopper [PER]
///   COBOL [ORG]
/// ```
fn main() {
    let docs = [
        "Marie Curie moved to Paris.",
        "Alan Turing worked at Bletchley Park.",
        "Grace Hopper helped develop COBOL.",
    ];

    #[cfg(not(feature = "parallel"))]
    let results = anno::extract_batch(&docs);
    #[cfg(feature = "parallel")]
    let results = {
        use anno::{Model, StackedNER};
        StackedNER::default().par_extract_batch(&docs, None)
    };

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
