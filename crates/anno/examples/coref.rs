/// Group entity mentions into coreference chains.
///
/// The rule-based resolver links mentions that refer to the same entity
/// across sentences. Requires the `analysis` feature.
///
/// ```sh
/// cargo run -p anno --no-default-features --example coref --features analysis
/// ```
///
/// Example output:
///
/// ```text
/// Entities: Marie Curie [PER], Paris [LOC], Marie Curie [PER]
/// Chain: "Marie Curie" = "Marie Curie"
/// ```
#[cfg(feature = "analysis")]
fn main() -> anno::Result<()> {
    use anno::backends::coref::simple::SimpleCorefResolver;
    use anno::{Model, StackedNER};

    let text = "Marie Curie moved to Paris. Marie Curie later returned.";

    let ner = StackedNER::default();
    let entities = ner.extract_entities(text, None)?;

    print!("Entities:");
    for (i, e) in entities.iter().enumerate() {
        if i > 0 {
            print!(",");
        }
        print!(" {} [{}]", e.text, e.entity_type);
    }
    println!();

    let resolver = SimpleCorefResolver::default();
    let chains = resolver.resolve_to_chains(&entities);

    for chain in &chains {
        if chain.mentions.len() > 1 {
            let quoted: Vec<String> = chain
                .mentions
                .iter()
                .map(|m| format!("\"{}\"", m.text))
                .collect();
            println!("Chain: {}", quoted.join(" = "));
        }
    }

    Ok(())
}

#[cfg(not(feature = "analysis"))]
fn main() {
    eprintln!(
        "This example requires the `analysis` feature: cargo run -p anno --no-default-features --example coref --features analysis"
    );
}
