//! Source-aligned Wikidata linking with the historical MHEL-LLaMo pipeline.
use std::path::PathBuf;
use std::process::Command;

use clap::{Args, Subcommand};

use crate::cli::CliError;

#[derive(Args)]
pub struct MhelArgs {
    /// Python for the selected stage. Retrieval and reranking use different environments.
    #[arg(long, global = true, default_value = "python3")]
    python: PathBuf,
    #[command(subcommand)]
    stage: Stage,
}

#[derive(Args)]
#[group(required = true, multiple = false)]
struct LinkInput {
    /// JSON array of source documents with language, text and typed mention spans.
    #[arg(long)]
    documents: Option<PathBuf>,
    /// Source directory in the authors' CSV format.
    #[arg(long)]
    dataset: Option<PathBuf>,
}

impl LinkInput {
    fn append(self, cmd: &mut Command) {
        if let Some(path) = self.documents {
            cmd.arg(path).arg("--documents");
        } else if let Some(path) = self.dataset {
            cmd.arg(path);
        }
    }
}

#[derive(Subcommand)]
enum Stage {
    /// Retrieve WikiBELA candidates for supplied mentions. Writes the candidate array to stdout.
    Retrieve {
        #[arg(long)]
        upstream: PathBuf,
        #[arg(long)]
        model_dir: PathBuf,
        #[command(flatten)]
        input: LinkInput,
        #[arg(long, default_value_t = 50)]
        top_k: usize,
        #[arg(long, default_value = "cpu")]
        device: String,
    },
    /// Run the authors' two-step historical entity reranker. Writes source-aligned JSONL.
    Rerank {
        #[arg(long)]
        upstream: PathBuf,
        /// Local Transformers chat checkpoint directory.
        #[arg(long)]
        model_dir: PathBuf,
        #[command(flatten)]
        input: LinkInput,
        #[arg(long)]
        candidates: PathBuf,
        #[arg(long, default_value_t = 20)]
        top_k: usize,
        /// Accept the top retrieval result above this score. Zero prompts on every mention.
        #[arg(long, default_value_t = 0.)]
        threshold: f64,
    },
    /// Import a published or previously generated output.csv. Writes source-aligned JSONL.
    Import {
        #[arg(long)]
        dataset: PathBuf,
        #[arg(long)]
        predictions: PathBuf,
    },
}

pub fn run(args: MhelArgs) -> Result<(), CliError> {
    let mut cmd = Command::new(args.python);
    cmd.args(["-c", include_str!("mhel.py")]);
    match args.stage {
        Stage::Retrieve {
            upstream,
            model_dir,
            input,
            top_k,
            device,
        } => {
            cmd.arg("retrieve").arg(upstream).arg(model_dir);
            input.append(&mut cmd);
            cmd.arg(top_k.to_string()).arg(device);
        }
        Stage::Rerank {
            upstream,
            model_dir,
            input,
            candidates,
            top_k,
            threshold,
        } => {
            cmd.arg("rerank").arg(upstream).arg(model_dir);
            input.append(&mut cmd);
            cmd.arg(candidates)
                .arg(top_k.to_string())
                .arg(threshold.to_string());
        }
        Stage::Import {
            dataset,
            predictions,
        } => {
            cmd.arg("import").arg(dataset).arg(predictions);
        }
    }
    let status = cmd
        .status()
        .map_err(|e| format!("starting MHEL Python: {e}"))?;
    if !status.success() {
        return Err(format!("MHEL worker exited with {status}").into());
    }
    Ok(())
}
