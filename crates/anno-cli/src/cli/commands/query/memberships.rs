// SPDX-License-Identifier: MIT OR Apache-2.0
use anno::coalesce::memberships::MembershipEdition;
use anno::publication::IndexSnapshot;
use anno::TrackRef;
use clap::Args;
use serde_json::json;
use std::fs::File;
use std::io::{self, BufReader};
use std::path::{Path, PathBuf};

#[derive(Args, Debug)]
pub struct MembershipArgs {
    /// Complete edition with stream, sequence, threshold, require_type_match and documents.
    /// Each documents entry supplies document and optional source_revision. Use - for stdin.
    current: PathBuf,
    /// Emit removals and upserts relative to this complete earlier edition.
    #[arg(long, value_name = "PREVIOUS")]
    since: Option<PathBuf>,
    /// Read the current component for a source document and submitted numeric track ID.
    #[arg(long, num_args = 2, value_names = ["DOCUMENT", "TRACK"], conflicts_with = "since")]
    neighbors: Vec<String>,
    /// Explain one native direct comparison between submitted tracks in this edition.
    #[arg(long, num_args = 4, value_names = ["LEFT_DOC", "LEFT_TRACK", "RIGHT_DOC", "RIGHT_TRACK"], conflicts_with_all = ["neighbors", "since"])]
    explain_pair: Vec<String>,
}

pub(super) fn read_edition(path: &Path) -> Result<MembershipEdition, String> {
    if path == Path::new("-") {
        serde_json::from_reader(io::stdin().lock())
    } else {
        let f = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        serde_json::from_reader(BufReader::new(f))
    }
    .map_err(|e| format!("{}: {e}", path.display()))
}

fn read(path: &Path) -> Result<IndexSnapshot, String> {
    read_edition(path)?.snapshot().map_err(|e| e.to_string())
}

pub(super) fn run(args: MembershipArgs) -> Result<(), String> {
    if args.current == Path::new("-") && args.since.as_deref() == Some(Path::new("-")) {
        return Err("only one membership edition may read stdin".into());
    }
    if let [left_doc, left_track, right_doc, right_track] = args.explain_pair.as_slice() {
        let track = |document: &str, id: &str| -> Result<TrackRef, String> {
            Ok(TrackRef {
                doc_id: document.to_owned(),
                track_id: id
                    .parse::<u64>()
                    .map_err(|_| "track must be an unsigned integer")?
                    .into(),
            })
        };
        let answer = read_edition(&args.current)?
            .explain_pair(
                &track(left_doc, left_track)?,
                &track(right_doc, right_track)?,
            )
            .map_err(|e| e.to_string())?;
        return serde_json::to_writer(io::stdout().lock(), &answer).map_err(|e| e.to_string());
    }
    let current = read(&args.current)?;
    let output = if let [document, track] = args.neighbors.as_slice() {
        let track = track
            .parse::<u64>()
            .map_err(|_| "track must be an unsigned integer")?
            .to_string();
        let query = current
            .rows()
            .find(|(key, _)| key.first() == Some(document) && key.last() == Some(&track));
        let mut neighbors = Vec::new();
        let membership =
            query.and_then(|(key, row)| row["component"].as_str().map(|c| (key, row, c)));
        if let Some((key, row, component)) = membership {
            // Both the publisher's direct edges and snapshot rows use source-key order.
            let mut edges = row["direct"]
                .as_array()
                .expect("membership schema")
                .iter()
                .peekable();
            for (other, posting) in current
                .rows()
                .filter(|(id, p)| *id != key && p["component"].as_str() == Some(component))
            {
                let direct = edges.next_if(|edge| edge["track"] == json!(other));
                neighbors.push(json!({
                    "key": other,
                    "surface": posting["surface"],
                    "catalog_record": posting["catalog_record"],
                    "direct_similarity": direct.map(|e| &e["similarity"]),
                    "direct_basis": direct.map(|e| &e["basis"])
                }));
            }
        }
        json!({"checkpoint":current.checkpoint(),"track":query.map(|(key,_)|key),"neighbors":neighbors})
    } else if let Some(path) = args.since {
        let previous = read(&path)?;
        let delta = current
            .changes_since(&previous)
            .map_err(|e| e.to_string())?;
        json!({"base":delta.base(),"target":delta.target(),"removed":delta.removed(),"upserted":delta.upserted()})
    } else {
        json!({"checkpoint":current.checkpoint(),"rows":current.rows().collect::<Vec<_>>()})
    };
    serde_json::to_writer(io::stdout().lock(), &output).map_err(|e| e.to_string())
}
