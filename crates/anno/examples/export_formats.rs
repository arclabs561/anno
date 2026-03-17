/// Export entity extraction results to various formats.
///
/// Demonstrates brat standoff, CoNLL BIO tags, JSONL, and graph CSV.
///
/// ```sh
/// cargo run --example export_formats
/// ```
///
/// Example output:
///
/// ```text
/// === brat ===
/// T1  PER 0 12    Marie Curie
/// T2  ORG 44 49   Nobel
///
/// === CoNLL ===
/// Marie   B-PER
/// Curie   I-PER
/// won     O
/// ...
///
/// === JSONL ===
/// {"end":12,"source":"example","start":0,"text":"Marie Curie","type":"PER"}
/// ...
/// ```
fn main() -> anno::Result<()> {
    use anno::{export, Model, StackedNER};

    let text = "Marie Curie won the Nobel Prize in Physics.";
    let m = StackedNER::default();
    let entities = m.extract_entities(text, None)?;

    println!("=== brat ===");
    println!("{}", export::to_brat(text, &entities, false));

    println!("\n=== CoNLL ===");
    println!("{}", export::to_conll(text, &entities));

    println!("\n=== JSONL ===");
    println!("{}", export::to_jsonl(&entities, "example", false));

    println!("\n=== Graph CSV (nodes) ===");
    let (nodes, edges) = export::to_graph_csv(&entities, &[], "example", false);
    print!("{}", nodes);
    if edges.lines().count() > 1 {
        println!("\n=== Graph CSV (edges) ===");
        print!("{}", edges);
    }

    Ok(())
}
