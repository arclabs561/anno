//! Classify directed relations between supplied entity clusters in a document.
use crate::cli::CliError;
use clap::Args;
use std::{path::PathBuf, process::Command};

#[derive(Args)]
pub struct GlidreArgs {
    /// Pinned checkout of the authors' glidre repository.
    #[arg(long)]
    upstream: PathBuf,
    /// Local GLiDRE v2 checkpoint directory.
    #[arg(long)]
    model_dir: PathBuf,
    /// JSON text document or batch with entity occurrences. Also accepts DocRED arrays.
    #[arg(long)]
    input: PathBuf,
    /// JSON object mapping relation IDs to unique prompt labels.
    #[arg(long)]
    schema: PathBuf,
    /// Multi-label decision threshold for the released v2 checkpoints.
    #[arg(long, default_value_t = 0.3)]
    threshold: f64,
    /// PyTorch device.
    #[arg(long, default_value = "cpu")]
    device: String,
    /// Python interpreter with GLiDRE dependencies.
    #[arg(long, default_value = "python3")]
    python: PathBuf,
}

pub fn run(args: GlidreArgs) -> Result<(), CliError> {
    let status = Command::new(args.python)
        .args(["-u", "-c", include_str!("glidre.py")])
        .arg(args.upstream)
        .arg(args.model_dir)
        .arg(args.input)
        .arg(args.schema)
        .arg(args.threshold.to_string())
        .arg(args.device)
        .status()
        .map_err(|e| format!("starting GLiDRE worker: {e}"))?;
    if !status.success() {
        return Err(format!("GLiDRE worker exited with {status}").into());
    }
    Ok(())
}
