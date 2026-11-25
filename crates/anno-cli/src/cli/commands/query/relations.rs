// SPDX-License-Identifier: MIT OR Apache-2.0
use anno::core::grounded::html_source::source_sha256;
use anno::GroundedDocument;
use clap::Args;
use serde_json::json;
use std::fs::File;
use std::io::{self, BufReader};
use std::path::{Path, PathBuf};

#[derive(Args, Debug)]
pub struct RelationsArgs {
    /// GroundedDocument JSON containing saved relation evidence. Use - for stdin.
    input: PathBuf,
}

pub(super) fn run(args: RelationsArgs) -> Result<(), String> {
    let path = args.input;
    let doc: GroundedDocument = if path == Path::new("-") {
        serde_json::from_reader(io::stdin().lock())
    } else {
        let file = File::open(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        serde_json::from_reader(BufReader::new(file))
    }
    .map_err(|e| format!("{}: {e}", path.display()))?;
    let mut mentions = Vec::new();
    for (stored, view) in doc
        .recorded_relations()
        .iter()
        .zip(doc.project_recorded_relations())
    {
        let view = view.map_err(|error| error.to_string())?;
        mentions.push(json!({
            "evidence": stored,
            "head_track_id": view.head.track.map(|track| track.id),
            "tail_track_id": view.tail.track.map(|track| track.id),
        }));
    }
    let mut concepts = Vec::new();
    for view in doc.project_recorded_concept_relations() {
        let view = view?;
        concepts.push(json!({
            "evidence": view.relation,
            "head_track_id": view.head_track.map(|track| track.id),
            "tail_track_id": view.tail_track.map(|track| track.id),
        }));
    }
    let output = json!({
        "document_id": doc.id(),
        "text_sha256": source_sha256(doc.text()),
        "mentions": mentions,
        "concepts": concepts,
    });
    serde_json::to_writer(io::stdout().lock(), &output).map_err(|e| e.to_string())
}
