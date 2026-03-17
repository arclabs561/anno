/// Quickstart: extract entities in one line.
///
/// Uses `anno::extract()` -- the simplest possible API.
///
/// ```sh
/// cargo run --example quickstart
/// ```
///
/// Example output:
///
/// ```text
/// Sophie Wilson [PER] (0,13) 0.95
/// ARM [ORG] (27,30) 0.90
/// ```
fn main() -> anno::Result<()> {
    // One-liner extraction
    let entities = anno::extract("Sophie Wilson designed the ARM processor.")?;

    for e in &entities {
        println!(
            "{} [{}] ({},{}) {:.2}",
            e.text,
            e.entity_type,
            e.start(),
            e.end(),
            e.confidence
        );
    }

    // Filter with EntitySliceExt
    use anno::prelude::*;
    let people: Vec<_> = entities.of_type(&EntityType::Person).collect();
    println!("\nPeople: {}", people.len());

    let confident: Vec<_> = entities.above_confidence(0.8).collect();
    println!("High-confidence: {}", confident.len());

    Ok(())
}
