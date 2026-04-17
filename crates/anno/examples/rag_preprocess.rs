/// Preprocess text for RAG: chunk and rewrite pronouns.
///
/// Splits a document into chunks and resolves pronouns so each chunk
/// is self-contained for embedding and retrieval.
///
/// ```sh
/// cargo run --example rag_preprocess
/// ```
///
/// Example output:
///
/// ```text
/// --- Chunk 1 (chars 0..56) ---
/// Original: Marie Curie moved to Paris. She joined the Sorbonne.
/// Resolved: Marie Curie moved to Paris. Marie Curie joined the Sorbonne.
///   1 rewrite(s), 2 entities
///
/// --- Chunk 2 (chars 56..109) ---
/// Original: Her work on radioactivity earned a Nobel Prize.
/// Resolved: Marie Curie's work on radioactivity earned a Nobel Prize.
///   1 rewrite(s), 1 entities
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
