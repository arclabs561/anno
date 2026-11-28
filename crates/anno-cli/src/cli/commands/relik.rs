//! Entity linking against a supplied catalog using the local ReLiK reader.
use crate::cli::CliError;
use clap::Args;
use std::{
    io::Write,
    path::PathBuf,
    process::{Command, Stdio},
};

#[derive(Args)]
pub struct RelikArgs {
    /// Directory written by scripts/prepare_relik.py.
    #[arg(long)]
    checkpoint: PathBuf,
    /// JSON array of candidates, each with a unique string id and title.
    #[arg(long)]
    candidates: PathBuf,
    /// One source document. Omit to read UTF-8 text from stdin.
    #[arg(long)]
    text: Option<String>,
    /// Python environment containing the ReLiK reader dependencies.
    #[arg(long, default_value = "python3")]
    python: PathBuf,
}

pub fn run(args: RelikArgs) -> Result<(), CliError> {
    let mut command = Command::new(args.python);
    command
        .args(["-u", "-c", include_str!("relik.py")])
        .arg(args.checkpoint)
        .arg(args.candidates)
        .env("HF_HUB_OFFLINE", "1")
        .env("TRANSFORMERS_OFFLINE", "1");
    if args.text.is_some() {
        command.stdin(Stdio::piped());
    }
    let mut child = command
        .spawn()
        .map_err(|e| format!("starting ReLiK reader: {e}"))?;
    let write_result = args.text.map(|text| {
        child
            .stdin
            .take()
            .expect("piped stdin")
            .write_all(text.as_bytes())
    });
    let status = child
        .wait()
        .map_err(|e| format!("waiting for ReLiK reader: {e}"))?;
    if !status.success() {
        return Err(format!("ReLiK reader exited with {status}").into());
    }
    write_result
        .transpose()
        .map_err(|e| format!("writing ReLiK document: {e}"))?;
    Ok(())
}
