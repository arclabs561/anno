//! Multilingual raw-text coreference with the upstream CorPipe 26 one-stage predictor.
use crate::cli::CliError;
use clap::{Args, ValueEnum};
use std::{
    io::Write,
    path::PathBuf,
    process::{Command, Stdio},
};

#[derive(Clone, Copy, ValueEnum)]
enum Output {
    Json,
    Conllu,
}

#[derive(Args)]
pub struct CorpipeArgs {
    /// Local corpipe26_onestage.py from ufal/crac2026-corpipe.
    #[arg(long)]
    predictor: PathBuf,
    /// Local pretrained CorPipe 26 one-stage checkpoint directory.
    #[arg(long)]
    checkpoint: PathBuf,
    /// Local UDPipe 1 tokenizer/parser model for the document language.
    #[arg(long)]
    tokenizer_model: PathBuf,
    /// One raw document. Omit to read the full UTF-8 document from stdin.
    #[arg(long)]
    text: Option<String>,
    /// Emit source-grounded clusters or the complete predicted CorefUD document.
    #[arg(long, value_enum, default_value = "json")]
    format: Output,
    /// Python executable with CorPipe dependencies and ufal.udpipe installed.
    #[arg(long, default_value = "python3")]
    python: PathBuf,
}

pub fn run(args: CorpipeArgs) -> Result<(), CliError> {
    let mut worker = Command::new(args.python);
    worker
        .args(["-u", "-c", include_str!("corpipe.py")])
        .arg(args.predictor)
        .arg(args.checkpoint)
        .arg(args.tokenizer_model)
        .arg(match args.format {
            Output::Json => "json",
            Output::Conllu => "conllu",
        });
    if args.text.is_some() {
        worker.stdin(Stdio::piped());
    }
    let mut child = worker
        .spawn()
        .map_err(|e| format!("starting CorPipe: {e}"))?;
    let input = args.text.map(|text| {
        child
            .stdin
            .take()
            .expect("piped stdin")
            .write_all(text.as_bytes())
    });
    let status = child
        .wait()
        .map_err(|e| format!("waiting for CorPipe: {e}"))?;
    if !status.success() {
        return Err(format!("CorPipe worker exited with {status}").into());
    }
    input
        .transpose()
        .map_err(|e| format!("writing CorPipe input: {e}"))?;
    Ok(())
}
