// SPDX-License-Identifier: MIT OR Apache-2.0
//! Read-only predecessor suggestions from an explicitly supplied earlier edition.
use super::memberships::{
    validate_source_tracks, MembershipDocument, MembershipEdition, MembershipError,
};
use super::resolver::string_similarity;
use crate::core::grounded::html_source::source_sha256;
use crate::{Track, TrackId, TrackRef};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Display policy for a predecessor lookup.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PredecessorOptions {
    /// Maximum displayed candidates (1 through 128). A score group is always kept whole.
    pub limit: usize,
    /// Inclusive threshold from zero through one. A zero overlap never proposes a track.
    pub minimum_score: f32,
}

impl Default for PredecessorOptions {
    fn default() -> Self {
        Self {
            limit: 1,
            minimum_score: 0.5,
        }
    }
}

/// The method used to compare the two retained surfaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum PredecessorScorer {
    /// Native binary32 Jaccard similarity of normalized word sets, as in [`string_similarity`].
    WordJaccardV1,
}

/// Submitted track handle and the source evidence used when scoring it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PredecessorTrack {
    /// Document ID in the supplied edition.
    pub doc_id: String,
    /// Submitted track ID within that document.
    pub track_id: TrackId,
    /// Publisher-assigned revision, when supplied.
    pub source_revision: Option<String>,
    /// SHA-256 of the complete source text, UTF-8 encoded.
    pub text_sha256: String,
    /// Canonical track surface at the time of lookup.
    pub surface: String,
}

impl PredecessorTrack {
    fn capture(doc: &MembershipDocument, track: &Track, text_sha256: String) -> Self {
        Self {
            doc_id: doc.document.id().to_owned(),
            track_id: track.id,
            source_revision: doc.source_revision.clone(),
            text_sha256,
            surface: track.canonical_surface.clone(),
        }
    }
}

/// One candidate for inspection in the prior edition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PredecessorCandidate {
    /// The earlier track and its compared surface.
    pub track: PredecessorTrack,
    /// Word-set similarity of the earlier and current surfaces.
    pub score: f32,
}

/// An entire score group withheld because only part of it would fit.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WithheldTie {
    /// Score shared by every track in this group.
    pub score: f32,
    /// Number of tracks tied at that score.
    pub count: usize,
}

/// Evidence for a read-only lookup at two publisher-issued sequences.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PredecessorLookup {
    /// Publisher-defined stream shared by both editions.
    pub stream: String,
    /// Sequence containing the query track.
    pub current_sequence: u64,
    /// Sequence searched for earlier tracks.
    pub prior_sequence: u64,
    /// Method used to score the canonical surfaces.
    pub scorer: PredecessorScorer,
    /// Maximum requested candidate count.
    pub limit: usize,
    /// Requested inclusive threshold.
    pub minimum_score: f32,
    /// The later track and its compared surface.
    pub query: PredecessorTrack,
    /// Qualified earlier tracks in descending score order. Ties sort by document and track ID.
    pub candidates: Vec<PredecessorCandidate>,
    /// True when the display limit omitted at least one qualifying earlier track.
    pub truncated: bool,
    /// Populated when a tie could fill some remaining places only by splitting the group.
    pub withheld_tie: Option<WithheldTie>,
}

fn source_documents(
    edition: &MembershipEdition,
) -> Result<BTreeMap<&str, &MembershipDocument>, MembershipError> {
    if !(0.0..=1.0).contains(&edition.threshold) {
        return Err(MembershipError::Invalid(
            "invalid membership edition threshold".into(),
        ));
    }
    let mut docs = BTreeMap::new();
    for source in &edition.documents {
        let id = source.document.id();
        if id.trim().is_empty()
            || source
                .source_revision
                .as_deref()
                .is_some_and(|r| r.trim().is_empty())
            || docs.insert(id, source).is_some()
        {
            return Err(MembershipError::Invalid(format!(
                "empty or duplicate document ID or empty revision for {id:?}"
            )));
        }
        validate_source_tracks(&source.document)?;
    }
    Ok(docs)
}

/// Rank prior tracks by lexical similarity to one submitted track in the current edition.
///
/// Both editions must belong to the same stream, and the prior sequence must be strictly
/// earlier. Their membership resolution policies do not affect this search. It considers
/// every prior track with a nonblank canonical surface, across entity types and catalog
/// assignments. Blank query surfaces produce an empty selection. Output retains the actual
/// inputs to each reported score. This operation cannot create an edge or catalog assignment.
///
/// ```
/// use anno::coalesce::memberships::MembershipEdition;
/// use anno::coalesce::{lookup_predecessors, PredecessorLookup, PredecessorOptions};
/// use anno::TrackRef;
///
/// # fn inspect(current: &MembershipEdition, prior: &MembershipEdition, track: &TrackRef)
/// # -> Result<PredecessorLookup, anno::coalesce::memberships::MembershipError> {
/// let result = lookup_predecessors(current, prior, track, PredecessorOptions::default())?;
/// if let Some(candidate) = result.candidates.first() {
///     println!("{} compared with {} ({})",
///         result.query.surface, candidate.track.surface, candidate.score);
/// } else if let Some(tie) = result.withheld_tie {
///     println!("{} tracks tied at {}", tie.count, tie.score);
/// }
/// # Ok(result)
/// # }
/// ```
pub fn lookup_predecessors(
    current: &MembershipEdition,
    prior: &MembershipEdition,
    query: &TrackRef,
    options: PredecessorOptions,
) -> Result<PredecessorLookup, MembershipError> {
    if current.stream.trim().is_empty()
        || current.stream != prior.stream
        || prior.sequence >= current.sequence
    {
        return Err(MembershipError::Invalid(
            "predecessor lookup requires the same nonempty stream and a strictly earlier prior sequence".into(),
        ));
    }
    if !(1..=128).contains(&options.limit) || !(0.0..=1.0).contains(&options.minimum_score) {
        return Err(MembershipError::Invalid(
            "predecessor limit must be in 1..=128 and minimum score must be finite and in 0..=1"
                .into(),
        ));
    }
    let current_docs = source_documents(current)?;
    let prior_docs = source_documents(prior)?;
    let query_doc = current_docs.get(query.doc_id.as_str()).ok_or_else(|| {
        MembershipError::Invalid(format!("unknown query document {:?}", query.doc_id))
    })?;
    let query_track = query_doc
        .document
        .get_track(query.track_id)
        .ok_or_else(|| {
            MembershipError::Invalid(format!(
                "unknown query track {} in {:?}",
                query.track_id, query.doc_id
            ))
        })?;
    let query = PredecessorTrack::capture(
        query_doc,
        query_track,
        source_sha256(query_doc.document.text()),
    );

    struct Scored<'a> {
        doc: &'a MembershipDocument,
        track: &'a Track,
        score: f32,
    }
    let mut ranked = Vec::new();
    if !query.surface.trim().is_empty() {
        for doc in prior_docs.into_values() {
            for track in doc
                .document
                .tracks()
                .filter(|track| !track.canonical_surface.trim().is_empty())
            {
                let score = string_similarity(&query.surface, &track.canonical_surface);
                if score > 0.0 && score >= options.minimum_score {
                    ranked.push(Scored { doc, track, score });
                }
            }
        }
    }
    ranked.sort_unstable_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.doc.document.id().cmp(b.doc.document.id()))
            .then_with(|| a.track.id.cmp(&b.track.id))
    });

    let mut candidates = Vec::new();
    let mut digests = BTreeMap::new();
    let mut withheld_tie = None;
    for group in ranked.chunk_by(|a, b| a.score == b.score) {
        let remaining = options.limit - candidates.len();
        if group.len() > remaining {
            if remaining > 0 {
                withheld_tie = Some(WithheldTie {
                    score: group[0].score,
                    count: group.len(),
                });
            }
            break;
        }
        for item in group {
            let digest = digests
                .entry(item.doc.document.id())
                .or_insert_with(|| source_sha256(item.doc.document.text()));
            candidates.push(PredecessorCandidate {
                track: PredecessorTrack::capture(item.doc, item.track, digest.clone()),
                score: item.score,
            });
        }
    }
    Ok(PredecessorLookup {
        stream: current.stream.clone(),
        current_sequence: current.sequence,
        prior_sequence: prior.sequence,
        scorer: PredecessorScorer::WordJaccardV1,
        limit: options.limit,
        minimum_score: options.minimum_score,
        query,
        truncated: candidates.len() < ranked.len(),
        candidates,
        withheld_tie,
    })
}
