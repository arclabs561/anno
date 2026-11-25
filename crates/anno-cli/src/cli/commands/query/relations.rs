// SPDX-License-Identifier: MIT OR Apache-2.0
use anno::core::grounded::html_source::source_sha256;
use clap::Args;
use serde_json::json;
use std::io;
use std::path::PathBuf;

#[derive(Args, Debug)]
pub struct RelationsArgs {
    /// GroundedDocument JSON containing saved relation evidence. Use - for stdin.
    input: PathBuf,
}

pub(super) fn run(args: RelationsArgs) -> Result<(), String> {
    let doc = super::read_grounded(&args.input)?;
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
