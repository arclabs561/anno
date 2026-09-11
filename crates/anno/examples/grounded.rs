//! Extract one identified document into anno's canonical grounded representation.
//!
//! ```sh
//! cargo run --example grounded
//! ```

use anno::{annotate_grounded_with, StackedNER};

fn main() -> anno::Result<()> {
    let text = "Grace Hopper developed COBOL.";
    let model = StackedNER::builder()
        .layer(anno::RegexNER::new())
        .layer(anno::HeuristicNER::new())
        .build();
    let document = annotate_grounded_with(&model, "hopper-bio", text, None)?;

    println!("{}: {} signals", document.id(), document.signals().len());
    for signal in document.signals() {
        let (start, end) = signal.text_offsets().unwrap_or((0, 0));
        println!("  {} [{}] ({start},{end})", signal.surface, signal.label());
    }
    Ok(())
}
