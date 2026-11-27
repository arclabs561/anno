// SPDX-License-Identifier: MIT OR Apache-2.0
use anno::coalesce::memberships::{MembershipEdition, MembershipPair};
use anno::publication::IndexSnapshot;
use anno::TrackRef;
use clap::{ArgAction, Args};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{self, BufReader, Read, Write};
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
    #[arg(long, num_args = 2, action = ArgAction::Set, allow_hyphen_values = true, value_names = ["DOCUMENT", "TRACK"], conflicts_with = "since")]
    neighbors: Option<Vec<String>>,
    /// Explain one native direct comparison between submitted tracks in this edition.
    #[arg(long, num_args = 4, action = ArgAction::Set, allow_hyphen_values = true, value_names = ["LEFT_DOC", "LEFT_TRACK", "RIGHT_DOC", "RIGHT_TRACK"], conflicts_with_all = ["neighbors", "since"])]
    explain_pair: Option<Vec<String>>,
    /// Verify a saved explanation for this pair against the exact edition bytes and current resolver.
    #[arg(long, value_name = "RECEIPT", requires = "explain_pair")]
    verify_explanation: Option<PathBuf>,
}

fn input(path: &Path) -> Result<Box<dyn Read>, String> {
    if path == Path::new("-") {
        Ok(Box::new(io::stdin().lock()))
    } else {
        let f = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(Box::new(f))
    }
}

fn read_json<T: DeserializeOwned>(path: &Path, source: impl Read) -> Result<T, String> {
    serde_json::from_reader(BufReader::new(source)).map_err(|e| format!("{}: {e}", path.display()))
}

pub(super) fn read_edition(path: &Path) -> Result<MembershipEdition, String> {
    read_json(path, input(path)?)
}

struct EditionInput {
    source: Box<dyn Read>,
    digest: Sha256,
}

impl Read for EditionInput {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.source.read(buf)?;
        self.digest.update(&buf[..n]);
        Ok(n)
    }
}

const EXPLANATION_FORMAT: &str = "anno.memberships.pair-explanation.v1";

#[derive(Deserialize, Serialize)]
struct RecordedExplanation {
    format: String,
    edition_sha256: String,
    #[serde(flatten)]
    answer: MembershipPair,
}

fn read_explanation(path: &Path) -> Result<Value, String> {
    let mut bytes = Vec::new();
    input(path)?
        .read_to_end(&mut bytes)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let saved: Value = read_json(path, bytes.as_slice())?;
    if saved["format"] != EXPLANATION_FORMAT {
        return Err("saved explanation lacks a supported recording format".into());
    }
    // Typed decoding rejects repeated answer fields. Keep the JSON value to compare
    // every field, including any unrecognized additions, with the current wire answer.
    let _: RecordedExplanation = read_json(path, bytes.as_slice())?;
    Ok(saved)
}

fn explain(
    path: &Path,
    left: &TrackRef,
    right: &TrackRef,
    receipt: Option<&Path>,
) -> Result<(), String> {
    if path == Path::new("-") && receipt == Some(Path::new("-")) {
        return Err("only one of the edition and explanation may read stdin".into());
    }
    let saved = receipt.map(read_explanation).transpose()?;
    // Hash and decode one byte stream. Reopening the path could pick up a replacement edition.
    let mut source = EditionInput {
        source: input(path)?,
        digest: Sha256::new(),
    };
    let edition: MembershipEdition = read_json(path, &mut source)?;
    let edition_sha256: String = source
        .digest
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    if saved
        .as_ref()
        .is_some_and(|r| r["edition_sha256"].as_str() != Some(&edition_sha256))
    {
        return Err("saved explanation does not match the SHA-256 of these edition bytes".into());
    }
    let recorded = RecordedExplanation {
        format: EXPLANATION_FORMAT.to_owned(),
        edition_sha256,
        answer: edition
            .explain_pair(left, right)
            .map_err(|e| e.to_string())?,
    };
    let bytes = serde_json::to_vec(&recorded).map_err(|e| e.to_string())?;
    if let Some(saved) = saved {
        let observed: Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        if saved != observed {
            return Err("saved explanation differs from the replayed pair answer".into());
        }
    }
    io::stdout()
        .lock()
        .write_all(&bytes)
        .map_err(|e| e.to_string())
}

fn read(path: &Path) -> Result<IndexSnapshot, String> {
    read_edition(path)?.snapshot().map_err(|e| e.to_string())
}

pub(super) fn run(args: MembershipArgs) -> Result<(), String> {
    if args.current == Path::new("-") && args.since.as_deref() == Some(Path::new("-")) {
        return Err("only one membership edition may read stdin".into());
    }
    if let Some(values) = args.explain_pair {
        let [left_doc, left_track, right_doc, right_track] = values.as_slice() else {
            return Err("--explain-pair requires two document and track addresses".into());
        };
        let track = |document: &str, id: &str| -> Result<TrackRef, String> {
            Ok(TrackRef {
                doc_id: document.to_owned(),
                track_id: id
                    .parse::<u64>()
                    .map_err(|_| "track must be an unsigned integer")?
                    .into(),
            })
        };
        return explain(
            &args.current,
            &track(left_doc, left_track)?,
            &track(right_doc, right_track)?,
            args.verify_explanation.as_deref(),
        );
    }
    let current = read(&args.current)?;
    let output = if let Some(values) = args.neighbors {
        let [document, track] = values.as_slice() else {
            return Err("--neighbors requires one document and track address".into());
        };
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
