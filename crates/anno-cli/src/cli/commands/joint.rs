//! Typed GLiNER2.5 extraction using the upstream JointIE decoder.
use crate::cli::CliError;
use clap::Args;
use std::{
    io::Write,
    path::PathBuf,
    process::{Command, Stdio},
};

#[derive(Args)]
pub struct JointArgs {
    /// Local Fastino boundary PyTorch checkpoint directory.
    #[arg(long)]
    checkpoint: PathBuf,
    /// Upstream JointSchema JSON with entity types, relation endpoints and constraints.
    #[arg(long)]
    schema: PathBuf,
    /// Analyze one document. Otherwise read {doc_id,text} JSONL documents from stdin.
    #[arg(long)]
    text: Option<String>,
    /// Python executable with gliner2[local] installed.
    #[arg(long, default_value = "python3")]
    python: PathBuf,
}

pub fn run(args: JointArgs) -> Result<(), CliError> {
    let mut worker = Command::new(args.python);
    worker
        .args(["-u", "-c", include_str!("joint.py")])
        .arg(args.checkpoint)
        .arg(args.schema);
    if args.text.is_some() {
        worker.stdin(Stdio::piped());
    }
    let mut child = worker
        .spawn()
        .map_err(|e| format!("starting JointIE: {e}"))?;
    let input_result = if let Some(text) = args.text {
        let mut stdin = child.stdin.take().expect("piped stdin");
        writeln!(stdin, "{}", serde_json::json!({"text":text}))
    } else {
        Ok(())
    };
    let status = child
        .wait()
        .map_err(|e| format!("waiting for JointIE: {e}"))?;
    input_result.map_err(|e| format!("writing JointIE input: {e}"))?;
    if !status.success() {
        return Err(format!("JointIE worker exited with {status}").into());
    }
    Ok(())
}
