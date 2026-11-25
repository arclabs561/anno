// SPDX-License-Identifier: MIT OR Apache-2.0
use super::memberships::read_edition;
use anno::coalesce::{lookup_predecessors, PredecessorOptions};
use anno::{TrackId, TrackRef};
use clap::Args;
use std::io;
use std::path::{Path, PathBuf};

#[derive(Args, Debug)]
#[command(
    after_help = "To widen the search to up to five positive name matches, use --limit 5 --minimum-score 0."
)]
pub struct PredecessorArgs {
    /// Complete membership edition containing the query track. Use - for stdin.
    current: PathBuf,
    /// Complete earlier membership edition of the same stream. One edition may use stdin.
    #[arg(long, value_name = "PRIOR")]
    prior: PathBuf,
    /// Source document ID of the query track in CURRENT.
    #[arg(long)]
    doc_id: String,
    /// Submitted numeric track ID in the query document.
    #[arg(long)]
    track_id: u64,
    /// Maximum candidates to show (1–128). A tied group that would exceed the limit is withheld.
    #[arg(long, default_value_t = PredecessorOptions::default().limit)]
    limit: usize,
    /// Inclusive word-Jaccard threshold (0–1). Only positive scores qualify.
    #[arg(long, default_value_t = PredecessorOptions::default().minimum_score)]
    minimum_score: f32,
}

pub(super) fn run(args: PredecessorArgs) -> Result<(), String> {
    if args.current == Path::new("-") && args.prior == Path::new("-") {
        return Err("only one membership edition may read stdin".into());
    }
    let current = read_edition(&args.current)?;
    let prior = read_edition(&args.prior)?;
    let query = TrackRef {
        doc_id: args.doc_id,
        track_id: TrackId::new(args.track_id),
    };
    let options = PredecessorOptions {
        limit: args.limit,
        minimum_score: args.minimum_score,
    };
    let lookup =
        lookup_predecessors(&current, &prior, &query, options).map_err(|e| e.to_string())?;
    serde_json::to_writer(io::stdout().lock(), &lookup).map_err(|e| e.to_string())
}
