//! Info command - Show model and version info

use super::super::output::{color, type_color};
use anno::{available_backends, Model, StackedNER};

/// Execute the info command.
pub fn run() -> Result<(), String> {
    println!();
    println!("{}", color("1;36", "anno"));
    println!("  Information Extraction: NER + Coreference");
    println!();
    println!("{}:", color("1;33", "Version"));
    println!("  {}", env!("CARGO_PKG_VERSION"));
    println!();
    println!("{}:", color("1;33", "Available Models (this build)"));

    // Use the actual available_backends() function to show real availability.
    // Backends can be: (1) feature not compiled in, (2) feature compiled but
    // model not yet downloaded, or (3) fully ready.
    let backends = available_backends();
    let onnx_model_backends = ["W2NER", "BertNEROnnx", "NuNER", "GLiNEROnnx", "CandleNER"];
    for (name, available) in backends {
        let (status, note) = if !available {
            (color("90", "✗"), " (requires feature flag)")
        } else if onnx_model_backends.contains(&name) {
            // Feature is enabled but these backends need a downloaded model.
            // We can't cheaply check if the model exists, so note the requirement.
            (
                color("33", "~"),
                " (needs model download -- run: anno models download)",
            )
        } else {
            (color("32", "✓"), "")
        };
        println!("  {} {} {}", status, name, note);
    }
    println!();

    let model: Box<dyn Model> = Box::new(StackedNER::default());
    println!("{}:", color("1;33", "Supported Entity Types (stacked)"));
    for t in model.supported_types() {
        let color_code = type_color(t.as_label());
        println!("  {} {}", color(color_code, "*"), t.as_label());
    }
    println!();

    println!("{}:", color("1;33", "Enabled Features"));
    #[allow(clippy::vec_init_then_push)] // Feature-gated pushes can't use vec![]
    let features: Vec<&str> = {
        #[allow(unused_mut)] // Some builds compile with none of these cfg-gated pushes.
        let mut v = Vec::with_capacity(4);
        #[cfg(feature = "onnx")]
        v.push("onnx");
        #[cfg(feature = "candle")]
        v.push("candle");
        #[cfg(feature = "eval")]
        v.push("eval");
        #[cfg(feature = "eval-bias")]
        v.push("eval-bias");
        v
    };
    if features.is_empty() {
        println!("  (default features only)");
    } else {
        println!("  {}", features.join(", "));
    }
    println!();

    Ok(())
}
