/// Export entity extraction results to various formats.
///
/// Demonstrates brat standoff, CoNLL BIO tags, JSONL, and graph CSV.
///
/// ```sh
/// cargo run -p anno --no-default-features --example export_formats
/// ```
///
/// Example output with tab-separated fields shown as columns.
///
/// ```text
/// === brat ===
/// T1    PER 0 11    Lynn Conway
/// T2    ORG 22 25    IBM
/// ...
///
/// === CoNLL ===
/// Lynn    B-PER
/// Conway    I-PER
/// worked    O
/// ...
///
/// === JSONL ===
/// {"end":11,"source":"example","start":0,"text":"Lynn Conway","type":"PER"}
/// ...
/// ```
fn main() -> anno::Result<()> {
    use anno::{export, Model, StackedNER};

    let text = "Lynn Conway worked at IBM and Xerox PARC in California.";
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
