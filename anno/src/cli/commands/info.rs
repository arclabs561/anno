//! Info command - Show model and version info

use super::super::output::{color, type_color};
use crate::{available_backends, Model, StackedNER};

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
    let mut features: Vec<&str> = Vec::new();
    #[cfg(feature = "onnx")]
    features.push("onnx");
    #[cfg(feature = "candle")]
    features.push("candle");
    #[cfg(feature = "eval")]
    features.push("eval");
    #[cfg(feature = "eval-bias")]
    features.push("eval-bias");
    #[cfg(feature = "eval-advanced")]
    features.push("eval-advanced");
    #[cfg(feature = "discourse")]
    features.push("discourse");
    if features.is_empty() {
        println!("  (default features only)");
    } else {
        println!("  {}", features.join(", "));
    }
    println!();

    Ok(())
}
