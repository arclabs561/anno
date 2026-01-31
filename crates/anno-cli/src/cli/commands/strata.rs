//! Tier command - Hierarchical clustering: reveal tier of abstraction

use super::super::parser::OutputFormat;

/// Hierarchical clustering: reveal tier of abstraction
#[derive(clap::Parser, Debug)]
pub struct TierArgs {
    /// Input file containing GraphDocument (JSON format)
    #[arg(short, long, value_name = "FILE")]
    pub input: Option<String>,

    /// Read GraphDocument from stdin (JSON format)
    #[arg(long)]
    pub stdin: bool,

    /// Clustering method to use
    #[arg(short, long, default_value = "leiden")]
    pub method: String,

    /// Resolution parameter for clustering (higher = more, smaller communities)
    #[arg(short, long, default_value = "1.0")]
    pub resolution: f32,

    /// Number of hierarchical levels to compute
    #[arg(short, long, default_value = "3")]
    pub levels: usize,

    /// Output format
    #[arg(short, long, default_value = "json")]
    pub format: OutputFormat,

    /// Output file path (if not specified, prints to stdout)
    #[arg(short = 'o', long)]
    pub output: Option<String>,

    /// Show progress and detailed cluster information
    #[arg(short, long)]
    pub verbose: bool,
}

#[cfg(feature = "eval-advanced")]
/// Execute the tier command.
pub fn run(args: TierArgs) -> Result<(), String> {
    let _ = args;
    Err(
        "The `tier` command is not available in the publishable `anno` crate.\n\
         Hierarchical clustering has been archived out of the main workspace."
            .to_string(),
    )
}

/// Execute the tier command (stub when eval-advanced is disabled).
#[cfg(not(feature = "eval-advanced"))]
pub fn run(_args: TierArgs) -> Result<(), String> {
    Err("Hierarchical clustering requires 'eval-advanced' feature. Build with: cargo build --features eval-advanced".to_string())
}

// Intentionally empty: this module is a stub in the publishable crate.
