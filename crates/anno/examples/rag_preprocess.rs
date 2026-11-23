/// Preprocess text for RAG: chunk and rewrite pronouns.
///
/// Chunk boundaries and pronoun rewrites depend on the input and extractor.
/// The offline model in this example returns named entities and leaves the
/// pronouns unchanged.
///
/// ```sh
/// cargo run -p anno --no-default-features --example rag_preprocess
/// ```
///
/// Example output:
///
/// ```text
/// --- Chunk 1 (chars 0..100) ---
/// Original: Marie Curie moved to Paris. She joined the Sorbonne. Her work on radioactivity earned a Nobel Prize.
/// Resolved: Marie Curie moved to Paris. She joined the Sorbonne. Her work on radioactivity earned a Nobel Prize.
///   0 rewrite(s), 4 entities
/// ```
fn main() -> anno::Result<()> {
    use anno::{rag, StackedNER};

    let text = "Marie Curie moved to Paris. She joined the Sorbonne. \
                Her work on radioactivity earned a Nobel Prize.";

    let m = StackedNER::default();
    let chunks = rag::preprocess(text, &m, None)?;

    for (i, chunk) in chunks.iter().enumerate() {
        println!(
            "--- Chunk {} (chars {}..{}) ---",
            i + 1,
            chunk.char_start,
            chunk.char_end
        );
        println!("Original: {}", chunk.original_text);
        println!("Resolved: {}", chunk.text);
        println!(
            "  {} rewrite(s), {} entities\n",
            chunk.rewrites,
            chunk.entities.len()
        );
    }

    Ok(())
}
