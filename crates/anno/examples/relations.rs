/// Extract entities and relations using TPLinker.
///
/// TPLinker uses a handshaking matrix to detect entity-pair relations.
/// No model downloads required (heuristic mode).
///
/// ```sh
/// cargo run --example relations --no-default-features
/// ```
///
/// Example output (heuristic mode, no ONNX):
///
/// ```text
/// Entities: 2
///   Acme Corp [ORG] (15,24)
///   Portland [LOC] (28,36)
/// Relations: 0
/// ```
fn main() -> anno::Result<()> {
    use anno::backends::inference::RelationExtractor;
    use anno::TPLinker;

    let text = "Alice works at Acme Corp in Portland.";
    let tp = TPLinker::new()?;
    let (entities, relations) = tp.extract_relations_default(text)?;

    println!("Entities: {}", entities.len());
    for e in &entities {
        println!(
            "  {} [{}] ({},{})",
            e.text,
            e.entity_type,
            e.start(),
            e.end()
        );
    }

    println!("Relations: {}", relations.len());
    for r in &relations {
        println!(
            "  {} --[{}]--> {} ({:.2})",
            r.head.text, r.relation_type, r.tail.text, r.confidence
        );
    }

    Ok(())
}
