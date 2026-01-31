//! Info command - Show model and version info

use super::super::output::{color, type_color};
use anno::{available_backends, Model, StackedNER};

/// Execute the info command.
pub fn run() -> Result<(), String> {
    println!();
    println!("{}", color("1;36", "anno"));
    println!("  Information Extraction: NER + Coreference + Relations + Entity Linking");
    println!();
    println!("{}:", color("1;33", "Version"));
    println!("  {}", env!("CARGO_PKG_VERSION"));
    println!();
    println!("{}:", color("1;33", "Available Models (this build)"));

    // Use the actual available_backends() function to show real availability
    let backends = available_backends();
    for (name, available) in backends {
        let status = if available {
            color("32", "✓")
        } else {
            color("90", "✗")
        };
        let note = if available {
            ""
        } else {
            " (requires feature flag)"
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
        let mut v = Vec::with_capacity(4);
        #[cfg(feature = "onnx")]
        v.push("onnx");
        #[cfg(feature = "candle")]
        v.push("candle");
        #[cfg(feature = "eval")]
        v.push("eval");
        #[cfg(feature = "eval-bias")]
        v.push("eval-bias");
        #[cfg(feature = "eval-advanced")]
        v.push("eval-advanced");
        #[cfg(feature = "discourse")]
        v.push("discourse");
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
