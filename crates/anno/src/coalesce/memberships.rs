// SPDX-License-Identifier: MIT OR Apache-2.0
//! Publish source-addressed track memberships from a complete corpus edition.
//! An accepted edge records a comparison in this run. Equal component digests
//! denote current coreference membership, including transitive links. The catalog
//! column is independently copied from each track's qualified assignment.

use super::{resolver::PairMeasure, Resolver};
use crate::core::grounded::html_source::source_sha256;
use crate::publication::{IndexSnapshot, PublicationError};
use crate::{Corpus, GroundedDocument, Identity, TrackId, TrackRef};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::BTreeMap;
use thiserror::Error;

/// A source document and its optional externally assigned edition.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MembershipDocument {
    /// Distinguishes separately issued editions with byte-identical text.
    pub source_revision: Option<String>,
    /// One current source document with its original local track handles.
    pub document: GroundedDocument,
}

/// All documents in the publisher's corpus at one sequence number.
/// Omitting a previous document withdraws its tracks from this edition.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MembershipEdition {
    /// Publisher-defined collection, audience and extraction policy.
    pub stream: String,
    /// Monotone publication revision.
    pub sequence: u64,
    /// Inclusive native resolver threshold. Keep it fixed when comparing snapshots.
    pub threshold: f32,
    /// Whether comparisons require matching track types.
    pub require_type_match: bool,
    /// Exactly one current edition per document ID.
    pub documents: Vec<MembershipDocument>,
}

/// Invalid corpus edition or publication sequence.
#[derive(Debug, Error)]
pub enum MembershipError {
    /// Input requires a named stream, valid threshold and unambiguous source keys.
    #[error("invalid membership edition: {0}")]
    Invalid(String),
    /// Index comparison or serialization failed.
    #[error(transparent)]
    Publication(#[from] PublicationError),
}

type LocalTrack = (String, TrackId);

fn local(reference: &TrackRef) -> LocalTrack {
    (reference.doc_id.clone(), reference.track_id)
}

impl MembershipEdition {
    /// Resolve once from supplied documents and capture one row per submitted track.
    /// Keys contain document, source-text SHA-256, optional revision and track ID.
    /// Component digests identify exact member sets within this publication stream.
    /// A merge, split or member-edition change invalidates every affected posting.
    /// Complete capture uses exhaustive native comparison. Applying a subsequent
    /// [delta](IndexSnapshot::changes_since) touches only removed or changed rows.
    pub fn snapshot(mut self) -> Result<IndexSnapshot, MembershipError> {
        let mut corpus = Corpus::new();
        let mut revisions = BTreeMap::new();
        self.documents
            .sort_by(|a, b| a.document.id().cmp(b.document.id()));
        for edition in self.documents {
            let doc = edition.document;
            if doc.id().trim().is_empty()
                || corpus.get_document(doc.id()).is_some()
                || edition
                    .source_revision
                    .as_ref()
                    .is_some_and(|rev| rev.trim().is_empty())
            {
                return Err(MembershipError::Invalid(format!(
                    "empty or duplicate document ID or empty revision for {:?}",
                    doc.id()
                )));
            }
            if let Some(revision) = edition.source_revision {
                revisions.insert(doc.id().to_owned(), revision);
            }
            corpus.add_document(doc);
        }
        publish_memberships(
            &mut corpus,
            &self.stream,
            self.sequence,
            self.threshold,
            self.require_type_match,
            &revisions,
        )
    }
}

pub(super) fn validate_source_tracks(doc: &GroundedDocument) -> Result<(), MembershipError> {
    if let Some(error) = doc.validate().first() {
        return Err(MembershipError::Invalid(format!(
            "invalid source occurrence in {}: {error}",
            doc.id()
        )));
    }
    if let Some(error) = doc.validate_invariants().first() {
        return Err(MembershipError::Invalid(format!(
            "invalid tracks in {}: {error}",
            doc.id()
        )));
    }
    Ok(())
}

/// Resolve an existing corpus and publish membership with the same keys as a fresh edition.
/// Unlike importing independent documents, this retains corpus-owned catalog and hybrid
/// assignments. The caller scopes document IDs and the stream. Supply the original source
/// revision for each document that has one; a missing revision selects a digest-only key.
pub fn publish_memberships(
    corpus: &mut Corpus,
    stream: &str,
    sequence: u64,
    threshold: f32,
    require_type_match: bool,
    revisions: &BTreeMap<String, String>,
) -> Result<IndexSnapshot, MembershipError> {
    if stream.trim().is_empty() || !(0.0..=1.0).contains(&threshold) {
        return Err(MembershipError::Invalid(
            "supply a stream and a finite threshold in 0..=1".into(),
        ));
    }
    if revisions
        .iter()
        .any(|(id, rev)| rev.trim().is_empty() || corpus.get_document(id).is_none())
    {
        return Err(MembershipError::Invalid(
            "source revisions must name a current document and nonempty edition".into(),
        ));
    }
    let stream = json!([
        "anno.query.memberships.v1",
        stream,
        threshold,
        require_type_match
    ])
    .to_string();
    let mut keys = BTreeMap::new();
    for doc in corpus.documents() {
        if doc.id().trim().is_empty() {
            return Err(MembershipError::Invalid("empty document ID".into()));
        }
        validate_source_tracks(doc)?;
        let mut prefix = vec![doc.id().to_owned(), source_sha256(doc.text())];
        prefix.extend(revisions.get(doc.id()).cloned());
        let mut submitted = std::collections::BTreeSet::new();
        for track in doc.tracks() {
            let source_id = corpus.submitted_track_id(doc.id(), track.id);
            if !submitted.insert(source_id) {
                return Err(MembershipError::Invalid(format!(
                    "ambiguous submitted track in {}",
                    doc.id()
                )));
            }
            let mut key = prefix.clone();
            key.push(source_id.to_string());
            keys.insert((doc.id().to_owned(), track.id), key);
        }
    }
    let mut edges = BTreeMap::<LocalTrack, BTreeMap<Vec<String>, (f32, PairMeasure)>>::new();
    Resolver::new().resolve_with_edges(
        corpus,
        Some(threshold),
        Some(require_type_match),
        |a, b, score, measure| {
            let a = local(a);
            let b = local(b);
            edges
                .entry(a.clone())
                .or_default()
                .insert(keys[&b].clone(), (score, measure));
            edges
                .entry(b)
                .or_default()
                .insert(keys[&a].clone(), (score, measure));
        },
    );
    let mut components = BTreeMap::new();
    for (_, members) in corpus.current_coref_clusters() {
        let mut addresses: Vec<_> = members.iter().map(|m| &keys[&local(m)]).collect();
        addresses.sort_unstable();
        // Structured JSON prevents boundary collisions among document, revision and track.
        let encoded = serde_json::to_string(&addresses).map_err(PublicationError::from)?;
        let digest = source_sha256(&encoded);
        for member in &members {
            components.insert(local(member), digest.clone());
        }
    }
    let mut rows = BTreeMap::new();
    for doc in corpus.documents() {
        for track in doc.tracks() {
            let address = (doc.id().to_owned(), track.id);
            let direct: Vec<_> = edges
                .remove(&address)
                .unwrap_or_default()
                .into_iter()
                .map(|(track, (similarity, basis))| json!({"track": track, "similarity": similarity, "basis": basis}))
                .collect();
            let mut signals = Vec::with_capacity(track.signals().len());
            for member in track.signals() {
                let signal = doc
                    .get_signal(member.signal_id)
                    .expect("validated track member");
                signals.push(json!({
                    "signal_id": signal.id,
                    "surface": signal.surface(),
                    "location": signal.location,
                    "label": signal.label(),
                    "position": member.position
                }));
            }
            let catalog = doc
                .identity_for_track(track.id)
                .and_then(Identity::qualified_kb_ref)
                .map(|(namespace, accession)| json!({"catalog":namespace,"accession":accession}));
            let key = keys.remove(&address).expect("submitted track");
            rows.insert(
                key,
                json!({
                    "surface": track.canonical_surface,
                    "signals": signals,
                    "component": components.remove(&address),
                    "direct": direct,
                    "catalog_record": catalog
                }),
            );
        }
    }
    Ok(IndexSnapshot::new(stream, sequence, rows)?)
}
