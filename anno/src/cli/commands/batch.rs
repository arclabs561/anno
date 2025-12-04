//! Batch command - Batch processing

use super::super::parser::{ModelBackend, OutputFormat};
use clap::Parser;

/// Batch processing
#[derive(Parser, Debug)]
pub struct BatchArgs {
    /// Process directory of files
    #[arg(short, long, value_name = "DIR")]
    pub dir: Option<String>,

    /// Read from stdin (JSONL format)
    #[arg(long)]
    pub stdin: bool,

    /// Model backend to use
    #[arg(short, long, default_value = "stacked")]
    pub model: ModelBackend,

    /// Run coreference resolution
    #[arg(long)]
    pub coref: bool,

    /// Link tracks to KB identities
    #[arg(long)]
    pub link_kb: bool,

    /// Number of parallel workers
    #[arg(short, long, default_value = "1")]
    pub parallel: usize,

    /// Show progress bar
    #[arg(long)]
    pub progress: bool,

    /// Enable caching
    #[arg(long)]
    pub cache: bool,

    /// Output directory for results
    #[arg(short, long, value_name = "DIR")]
    pub output: Option<String>,

    /// Output format
    #[arg(long, default_value = "grounded")]
    pub format: OutputFormat,

    /// Suppress status messages
    #[arg(short, long)]
    pub quiet: bool,
}

pub fn run(_args: BatchArgs) -> Result<(), String> {
    // Similar to pipeline but optimized for batch processing
    // Implementation would be similar to pipeline but with better parallelization
    // For now, delegate to pipeline command
    // TODO: Update to use pipeline::run once pipeline is extracted
    // For now, batch is extracted but pipeline is not yet, so we return an error
    // This will be fixed when pipeline is extracted
    Err("Batch command requires pipeline to be extracted first. Please use 'anno pipeline' instead for now.".to_string())
}
