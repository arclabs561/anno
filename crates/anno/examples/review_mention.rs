// SPDX-License-Identifier: MIT OR Apache-2.0
//! A caller's review of a catalog correction in an original synthetic quay log.
use anno::core::grounded::{
    html_source::source_sha256, Corpus, Identity, IdentitySource, KbAssignment,
    KbAssignmentReceipt, SignalId,
};
use anno::{GroundedDocument, Location, Signal};
use serde::{Deserialize, Serialize};
use std::{
    error::Error,
    io::{self, BufRead},
    process::ExitCode,
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const HELP: &str = "Review the second Ora Venn in an original synthetic quay log.
  review_mention preview
  review_mention review   (enter apply or defer, then a reason; cancel to exit)
  review_mention show < saved-decision.json";

#[derive(Serialize, Deserialize)]
struct Evidence {
    doc_id: String,
    source_sha256: String,
    signal_id: SignalId,
    start: usize,
    end: usize,
    offset_unit: String,
    quote: String,
    identity_source: Option<IdentitySource>,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase", deny_unknown_fields)]
enum Decision {
    Applied { receipt: KbAssignmentReceipt },
    Deferred { proposal: KbAssignment },
}

#[derive(Serialize, Deserialize)]
struct Record {
    document: GroundedDocument,
    evidence: Evidence,
    reason: String,
    decision: Decision,
}

fn describe(text: &str, evidence: &Evidence, assignment: &KbAssignment) -> Result<String> {
    let prior = assignment
        .previous
        .as_ref()
        .and_then(|p| p.kb_name.as_deref().zip(p.kb_id.as_deref()));
    let prior = prior.map_or_else(
        || "(no qualified catalog target)".to_owned(),
        |(c, k)| format!("{c}/{k}"),
    );
    let origin = serde_json::to_string(&evidence.identity_source)?;
    let scope = serde_json::to_string(&assignment.affected_tracks)?;
    Ok(format!(
        "Source {} (SHA-256 {})\nText {text}\nSignal {} at Unicode scalars [{}, {}) {:?}\nPrior identity source {origin}\nPrior target {prior}\nProposed target {}/{}\nAffected tracks {scope}",
        evidence.doc_id, evidence.source_sha256, evidence.signal_id, evidence.start, evidence.end, evidence.quote,
        assignment.kb_name, assignment.kb_id
    ))
}

fn review(preview_only: bool) -> Result<()> {
    let mut doc = GroundedDocument::new(
        "quay-log",
        "Écho quay. Ora Venn tuned the horn. Ora Venn signed the cargo log.",
    );
    let mut add_mention = |start, end| -> Result<_> {
        let signal = doc.add_signal_validated(Signal::new(
            0,
            Location::text(start, end),
            "Ora Venn",
            "PER",
            1.0,
        ))?;
        let track = doc
            .create_track_from_signals("Ora Venn", &[signal])
            .ok_or("failed to create fixture track")?;
        Ok((signal, doc.track_ref(track).ok_or("missing fixture track")?))
    };
    let (_, first) = add_mention(11, 19)?;
    let (selected, trigger) = add_mention(36, 44)?;
    let mut corpus = Corpus::new();
    corpus.add_document(doc);
    for track in [&first, &trigger] {
        corpus.link_track_to_kb(track, "quay-catalog", "musician", "Ora Venn")?;
    }

    let doc = corpus
        .get_document(&trigger.doc_id)
        .ok_or("missing fixture document")?;
    let signal = doc.get_signal(selected).ok_or("missing fixture signal")?;
    let (start, end) = signal
        .text_offsets()
        .ok_or("expected a contiguous mention")?;
    let evidence = Evidence {
        doc_id: doc.id().into(),
        source_sha256: source_sha256(doc.text()),
        signal_id: selected,
        start,
        end,
        offset_unit: "unicode_scalar".into(),
        quote: signal.surface().into(),
        identity_source: doc
            .identity_for_signal(selected)
            .and_then(|id| id.source())
            .cloned(),
    };
    let preview = corpus.preview_kb_assignment(&trigger, "quay-catalog", "clerk", "Ora Venn")?;
    if preview_only {
        let output = serde_json::json!({"source_text": doc.text(), "evidence": evidence, "proposal": preview.assignment()});
        println!("{}", serde_json::to_string_pretty(&output)?);
        return Ok(());
    }
    eprintln!("{}", describe(doc.text(), &evidence, preview.assignment())?);
    eprintln!("Enter apply, defer, or cancel.");
    let mut input = io::stdin().lock().lines();
    let choice = input
        .next()
        .transpose()?
        .ok_or("review ended before a decision")?;
    let choice = choice.trim();
    if choice == "cancel" {
        return Ok(());
    }
    if !matches!(choice, "apply" | "defer") {
        return Err("expected apply, defer, or cancel".into());
    }
    eprintln!("Enter your reason.");
    let reason = input
        .next()
        .transpose()?
        .filter(|s| !s.trim().is_empty())
        .ok_or("a reader reason is required")?;
    let decision = match choice {
        "apply" => Decision::Applied {
            receipt: corpus.apply_kb_assignment(preview)?,
        },
        _ => Decision::Deferred {
            proposal: preview.assignment().clone(),
        },
    };
    let document = corpus
        .get_document(&trigger.doc_id)
        .ok_or("missing reviewed document")?
        .clone();
    let record = Record {
        document,
        evidence,
        reason,
        decision,
    };
    println!("{}", serde_json::to_string_pretty(&record)?);
    Ok(())
}

fn source_catalog_key(source: Option<&IdentitySource>) -> Option<(&str, &str)> {
    match source {
        Some(
            IdentitySource::KnowledgeBase { kb_name, kb_id }
            | IdentitySource::Hybrid { kb_name, kb_id, .. },
        ) => Some((kb_name.as_str(), kb_id.as_str())),
        _ => None,
    }
}

fn show(record: Record) -> Result<()> {
    let Record {
        document,
        evidence,
        reason,
        decision,
    } = record;
    let (outcome, assignment, applied_id) = match &decision {
        Decision::Applied { receipt } => {
            ("applied", &receipt.assignment, Some(receipt.identity_id))
        }
        Decision::Deferred { proposal } => ("deferred", proposal, None),
    };
    if reason.trim().is_empty() || !document.validate_invariants().is_empty() {
        return Err("saved decision has a blank reason or inconsistent document".into());
    }
    let signal = document
        .get_signal(evidence.signal_id)
        .ok_or("saved signal is missing")?;
    if evidence.doc_id != document.id()
        || evidence.source_sha256 != source_sha256(document.text())
        || evidence.offset_unit != "unicode_scalar"
        || signal.location != Location::text(evidence.start, evidence.end)
        || signal.surface() != evidence.quote
        || signal.validate_against(document.text()).is_some()
    {
        return Err("saved source and signal disagree with the recorded anchor".into());
    }
    let track = document
        .track_for_signal(signal.id)
        .ok_or("saved signal has no track")?;
    // Both namesakes in this fixture have independent corpus records.
    if document.track_ref(track.id).as_ref() != Some(&assignment.trigger)
        || assignment.affected_tracks.as_slice() != std::slice::from_ref(&assignment.trigger)
    {
        return Err("saved signal's track disagrees with the reviewed scope".into());
    }
    let current = document
        .identity_for_signal(signal.id)
        .ok_or("missing saved identity")?;
    let previous = assignment
        .previous
        .as_ref()
        .ok_or("missing original binding")?;
    if !assignment.updates_shared_identity || previous.identity_id != current.id {
        return Err("the original corpus identity should be updated in place".into());
    }
    let prior_key = previous.kb_name.as_deref().zip(previous.kb_id.as_deref());
    if source_catalog_key(evidence.identity_source.as_ref()) != prior_key
        || (applied_id.is_none() && current.source() != evidence.identity_source.as_ref())
    {
        return Err("recorded identity source disagrees with the prior binding".into());
    }
    let proposed = Identity::from_kb(
        current.id,
        &assignment.canonical_name,
        &assignment.kb_name,
        &assignment.kb_id,
    );
    let proposed_key = proposed
        .qualified_kb_ref()
        .ok_or("saved proposal needs a qualified catalog target")?;
    let expected = if let Some(id) = applied_id {
        if source_catalog_key(current.source()) != Some(proposed_key) {
            return Err(
                "applied identity source disagrees with the reviewed catalog target".into(),
            );
        }
        (
            id,
            Some(proposed_key.0),
            Some(proposed_key.1),
            proposed.canonical_name(),
        )
    } else {
        (
            previous.identity_id,
            previous.kb_name.as_deref(),
            previous.kb_id.as_deref(),
            previous.canonical_name.as_str(),
        )
    };
    let actual = (
        current.id,
        current.kb_name(),
        current.kb_id(),
        current.canonical_name(),
    );
    if actual != expected {
        return Err("saved document binding disagrees with the recorded decision".into());
    }

    println!("Decision {outcome}");
    println!("{}", describe(document.text(), &evidence, assignment)?);
    let target = match current.qualified_kb_ref() {
        Some((catalog, key)) => format!("{catalog}/{key}"),
        None => "(no qualified catalog target)".to_owned(),
    };
    println!("Target in saved document {target}");
    println!("Reader reason {reason}");
    Ok(())
}

fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [action] if action == "--help" || action == "-h" => {
            println!("{HELP}");
            Ok(())
        }
        [action] if action == "preview" => review(true),
        [action] if action == "review" => review(false),
        [action] if action == "show" => show(serde_json::from_reader(io::stdin().lock())?),
        _ => Err(format!("expected a review action and its arguments\n{HELP}").into()),
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}
