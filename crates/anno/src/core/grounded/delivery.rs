//! Session-scoped, per-document delivery receipts. Callers authenticate issuers
//! and authorize restores before presenting a delivery to the corpus.

use super::{Corpus, GroundedDocument, Identity, IdentityId, TrackId, TrackRef};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

/// How incoming ownership assertions enter this corpus.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum DeliveryOwnership {
    /// Allocate a fresh owner for each identity in this document, including a fork.
    Independent,
    /// Restore an issuer's shared records. The issuer, origin and original
    /// numeric identity ID together identify a record. The caller vouches for this origin.
    Restore {
        /// Stable identifier of the source corpus/authority snapshot.
        origin: String,
    },
}

/// One document in a source batch. Give each member a distinct delivery ID.
/// Source revisions increase strictly per (issuer, embedded document ID).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DocumentDelivery {
    /// Authenticated upstream namespace, as established by the caller.
    pub issuer: String,
    /// Idempotency token unique within that issuer.
    pub delivery_id: String,
    /// Source document's monotonically increasing revision.
    pub source_revision: u64,
    /// Explicit import or restoration intent.
    pub ownership: DeliveryOwnership,
}

impl DocumentDelivery {
    /// Unambiguous live corpus key for an issuer-local document ID.
    #[must_use]
    pub fn corpus_document_id(&self, local: &str) -> String {
        format!("{}:{}/{}", self.issuer.len(), self.issuer, local)
    }
}

/// Evidence of the original acceptance. Later retries return these exact fields.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeliveryReceipt {
    /// Accepted envelope.
    pub request: DocumentDelivery,
    /// Document ID printed in the original grounded payload.
    pub source_document_id: String,
    /// Qualified key at which to look up the live document.
    pub document_id: String,
    /// SHA-256 of the exact incoming bytes before parsing or remapping.
    pub payload_sha256: String,
    /// One-based accepted-delivery sequence in this session.
    pub applied_revision: u64,
}

/// Result of a successful retry-safe delivery.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeliveryOutcome {
    /// First acceptance of this document delivery.
    Applied(DeliveryReceipt),
    /// An accepted retry. No corpus mutation occurred.
    PreviouslyApplied(DeliveryReceipt),
}

impl DeliveryOutcome {
    /// The original receipt for either successful outcome.
    #[must_use]
    pub fn receipt(&self) -> &DeliveryReceipt {
        match self {
            Self::Applied(r) | Self::PreviouslyApplied(r) => r,
        }
    }
}

/// Catalog fields visible on a reviewed track or its incoming candidate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeliveryCatalogBinding {
    /// Namespace, if supplied.
    pub kb_name: Option<String>,
    /// Accession, if supplied.
    pub kb_id: Option<String>,
    /// Recorded label.
    pub canonical_name: String,
}

impl From<&Identity> for DeliveryCatalogBinding {
    fn from(identity: &Identity) -> Self {
        Self {
            kb_name: identity.kb_name.clone(),
            kb_id: identity.kb_id.clone(),
            canonical_name: identity.canonical_name.clone(),
        }
    }
}

/// A previously reviewed track whose catalog fields or ownership would be in dispute.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeliveryBindingConflict {
    /// Current live handle for this edition.
    pub track: TrackRef,
    /// Submitted track ID (prior to replacement rekeying).
    pub source_track_id: TrackId,
    /// Current reviewed assignment.
    pub current: Option<DeliveryCatalogBinding>,
    /// Fields asserted by the source at the same submitted track ID.
    pub incoming: Option<DeliveryCatalogBinding>,
    /// Effective binding after approval. Restored members use their live shared owner.
    pub after: Option<DeliveryCatalogBinding>,
    /// Tracks that currently share this assignment, including the trigger.
    pub current_members: Vec<TrackRef>,
    /// Expected members after approval; tracks in the incoming document use submitted IDs.
    pub proposed_members: Vec<TrackRef>,
}

/// Serializable description of the proposed source revision and affected reviewed assignments.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeliveryReviewSummary {
    /// Incoming envelope.
    pub request: DocumentDelivery,
    /// Qualified corpus document ID.
    pub document_id: String,
    /// Current accepted source revision.
    pub previous_source_revision: u64,
    /// Original incoming payload digest.
    pub payload_sha256: String,
    /// Disputed bindings in submitted track ID order.
    pub affected_tracks: Vec<DeliveryBindingConflict>,
}

/// Pending replacement bound to one corpus revision. Drop to keep the current edition.
#[derive(Debug)]
pub struct DeliveryReview {
    summary: DeliveryReviewSummary,
    token: Arc<()>,
    document: GroundedDocument,
}

impl DeliveryReview {
    /// Inspect or persist the proposal without granting permission to apply it.
    pub fn summary(&self) -> &DeliveryReviewSummary {
        &self.summary
    }
}

/// Typed rejection of a delivery; these paths leave the corpus and outstanding previews intact.
#[derive(Debug, thiserror::Error)]
pub enum DeliveryError {
    /// Required identifying strings were empty.
    #[error("issuer, delivery ID, source document ID and restore origin must be nonempty")]
    InvalidKey,
    /// The embedded, original GroundedDocument could not be read or validated.
    #[error("invalid grounded delivery payload: {0}")]
    InvalidPayload(serde_json::Error),
    /// Accepted issuer token reused with different bytes or intent.
    #[error("delivery ID was already accepted with different bytes or intent")]
    TokenConflict,
    /// A new delivery must advance the source head.
    #[error("source revision must increase for a new delivery ID")]
    SourceRevisionConflict,
    /// A legacy caller already owns the computed destination key.
    #[error("qualified document key contains a document outside delivery management")]
    UnmanagedDocument,
    /// Issuer supplied differing originals under a shared restore key.
    #[error("restored identity conflicts with the recorded origin")]
    OriginConflict,
    /// The source seeks to change a reviewed catalog assignment or its membership.
    #[error("source revision requires review of existing catalog assignments")]
    ReviewRequired(Box<DeliveryReview>),
    /// Replacement approval no longer belongs to the current corpus state.
    #[error("delivery review is stale or belongs to another corpus")]
    StaleReview,
    /// No space for the new receipt or local identity/track IDs.
    #[error("delivery would exhaust the ID space")]
    Exhausted,
}

#[derive(Default, Clone, Debug)]
pub(super) struct DeliveryState {
    receipts: HashMap<(String, String), DeliveryReceipt>,
    pub(super) heads: HashMap<String, SourceHead>,
    // (issuer, restore origin, submitted identity ID) -> original payload and live owner.
    origins: HashMap<(String, String, IdentityId), (Identity, IdentityId)>,
    reviewed_owners: HashSet<IdentityId>,
    sequence: u64,
}

#[derive(Clone, Debug)]
pub(super) struct SourceHead {
    revision: u64,
    // Submitted ID -> live handle for assignments explicitly reviewed in this corpus.
    reviewed: BTreeMap<TrackId, TrackId>,
}

impl Corpus {
    /// Import exactly one grounded JSON document, using its original bytes for retry
    /// detection. Receipts are retained for the entire lifetime of this in-memory
    /// corpus. After restart, recover a durable journal before accepting late retries.
    /// A raw add_document replacement takes its key out of delivery management.
    pub fn deliver_document(
        &mut self,
        request: &DocumentDelivery,
        payload: &[u8],
    ) -> Result<DeliveryOutcome, DeliveryError> {
        if request.issuer.trim().is_empty()
            || request.delivery_id.trim().is_empty()
            || matches!(&request.ownership, DeliveryOwnership::Restore { origin } if origin.trim().is_empty())
        {
            return Err(DeliveryError::InvalidKey);
        }
        let digest = super::html_source::bytes_sha256(payload);
        let ledger = &self.state().delivery;
        let token = (request.issuer.clone(), request.delivery_id.clone());
        if let Some(receipt) = ledger.receipts.get(&token) {
            if receipt.request != *request || receipt.payload_sha256 != digest {
                return Err(DeliveryError::TokenConflict);
            }
            return Ok(DeliveryOutcome::PreviouslyApplied(receipt.clone()));
        }
        let doc: GroundedDocument =
            serde_json::from_slice(payload).map_err(DeliveryError::InvalidPayload)?;
        if doc.id().trim().is_empty() {
            return Err(DeliveryError::InvalidKey);
        }
        let key = request.corpus_document_id(doc.id());
        if let Some(head) = ledger.heads.get(&key) {
            if request.source_revision <= head.revision {
                return Err(DeliveryError::SourceRevisionConflict);
            }
            for original in doc.identities.values() {
                if let Some((prior, _)) = self.restored(request, original.id) {
                    if !prior.matches_copy(Some(original)) {
                        return Err(DeliveryError::OriginConflict);
                    }
                }
            }
            let affected_tracks = self.delivery_conflicts(request, &doc, &key, head);
            if !affected_tracks.is_empty() {
                return Err(DeliveryError::ReviewRequired(Box::new(DeliveryReview {
                    summary: DeliveryReviewSummary {
                        request: request.clone(),
                        document_id: key,
                        previous_source_revision: head.revision,
                        payload_sha256: digest,
                        affected_tracks,
                    },
                    token: self.review_token(),
                    document: doc,
                })));
            }
        } else if self.get_document(&key).is_some() {
            return Err(DeliveryError::UnmanagedDocument);
        }
        self.commit_delivery(request, doc, digest)
    }

    /// Accept a staged replacement after inspecting its summary. Independent
    /// replacement uses incoming records. Shared restoration follows the current
    /// canonical record at the authorized origin. Other documents retain their bindings.
    pub fn accept_document_replacement(
        &mut self,
        review: DeliveryReview,
    ) -> Result<DeliveryOutcome, DeliveryError> {
        if !self.accepts_review(&review.token) {
            return Err(DeliveryError::StaleReview);
        }
        self.commit_delivery(
            &review.summary.request,
            review.document,
            review.summary.payload_sha256,
        )
    }

    pub(super) fn protect_reviewed_bindings(&mut self, tracks: &[TrackRef]) {
        let state = self.state_mut();
        for track in tracks {
            if let Some(head) = state.delivery.heads.get_mut(&track.doc_id) {
                let submitted = state
                    .submitted_track_ids
                    .get(&track.doc_id)
                    .and_then(|map| map.get(&track.track_id))
                    .copied()
                    .unwrap_or(track.track_id);
                head.reviewed.insert(submitted, track.track_id);
                if let Some(owner) =
                    state.documents[&track.doc_id].identity_for_track(track.track_id)
                {
                    state.delivery.reviewed_owners.insert(owner.id);
                }
            }
        }
    }

    fn restored(
        &self,
        request: &DocumentDelivery,
        id: IdentityId,
    ) -> Option<&(Identity, IdentityId)> {
        let DeliveryOwnership::Restore { origin } = &request.ownership else {
            return None;
        };
        self.state()
            .delivery
            .origins
            .get(&(request.issuer.clone(), origin.clone(), id))
    }

    fn delivery_members(&self, owner: Option<&Identity>) -> Vec<TrackRef> {
        let Some(owner) = owner.filter(|i| self.is_shared_identity(i)) else {
            return Vec::new();
        };
        let mut members = Vec::new();
        for doc in self
            .documents()
            .filter(|doc| owner.matches_copy(doc.get_identity(owner.id)))
        {
            members.extend(
                doc.tracks()
                    .filter(|t| t.identity_id == Some(owner.id))
                    .map(|t| TrackRef {
                        doc_id: doc.id.clone(),
                        track_id: t.id,
                    }),
            );
        }
        members.sort_by(|a, b| (&a.doc_id, a.track_id).cmp(&(&b.doc_id, b.track_id)));
        members
    }

    fn delivery_conflicts(
        &self,
        request: &DocumentDelivery,
        incoming: &GroundedDocument,
        key: &str,
        head: &SourceHead,
    ) -> Vec<DeliveryBindingConflict> {
        let current = &self.state().documents[key];
        let mut conflicts = Vec::new();
        for (&source_id, &live_id) in &head.reviewed {
            let current_owner = current.identity_for_track(live_id);
            let proposed = incoming.identity_for_track(source_id);
            let continuing = proposed
                .and_then(|i| self.restored(request, i.id))
                .and_then(|(_, id)| self.get_identity(*id));
            let current_binding = current_owner.map(DeliveryCatalogBinding::from);
            let incoming_binding = proposed.map(DeliveryCatalogBinding::from);
            let after = continuing.or(proposed).map(DeliveryCatalogBinding::from);
            let track = TrackRef {
                doc_id: key.to_owned(),
                track_id: live_id,
            };
            let mut members = self.delivery_members(current_owner);
            if members.is_empty() && current_owner.is_some() {
                members.push(track.clone());
            }
            let mut proposed_members = self.delivery_members(continuing);
            proposed_members.retain(|m| m.doc_id != key);
            if let Some(p) = proposed {
                proposed_members.extend(
                    incoming
                        .tracks()
                        .filter(|t| t.identity_id == Some(p.id))
                        .map(|t| TrackRef {
                            doc_id: key.to_owned(),
                            track_id: t.id,
                        }),
                );
            }
            let mut previous = members.clone();
            for member in &mut previous {
                if member.doc_id == key {
                    if let Some(submitted) = self
                        .state()
                        .submitted_track_ids
                        .get(key)
                        .and_then(|map| map.get(&member.track_id))
                    {
                        member.track_id = *submitted;
                    }
                }
            }
            let compare =
                |a: &TrackRef, b: &TrackRef| (&a.doc_id, a.track_id).cmp(&(&b.doc_id, b.track_id));
            previous.sort_by(compare);
            proposed_members.sort_by(compare);
            if current_binding != incoming_binding
                || current_binding != after
                || previous != proposed_members
                || current_owner.is_none()
            {
                conflicts.push(DeliveryBindingConflict {
                    track,
                    source_track_id: source_id,
                    current: current_binding,
                    incoming: incoming_binding,
                    after,
                    current_members: members,
                    proposed_members,
                });
            }
        }
        conflicts
    }

    // A partial restore temporarily loses the weakest-pair score. Restore it only
    // when all original document-qualified handles survive with exactly that membership.
    fn restore_complete_evidence(
        &mut self,
        request: &DocumentDelivery,
        origin: &str,
        assigned: &HashMap<IdentityId, IdentityId>,
    ) {
        for (&original_id, &live_id) in assigned {
            let original = &self.state().delivery.origins
                [&(request.issuer.clone(), origin.to_owned(), original_id)]
                .0;
            let Some(minimum) = original.minimum_pair_similarity else {
                continue;
            };
            let Some(current) = self.get_identity(live_id) else {
                continue;
            };
            let Some(current_refs) = current
                .source
                .as_ref()
                .map(super::IdentitySource::track_refs)
            else {
                continue;
            };
            let Some(original_refs) = original
                .source
                .as_ref()
                .map(super::IdentitySource::track_refs)
            else {
                continue;
            };
            if current.minimum_pair_similarity.is_some()
                || current_refs.is_empty()
                || current_refs.len() != original_refs.len()
            {
                continue;
            }
            let mut expected: Vec<_> = original_refs
                .iter()
                .map(|r| TrackRef {
                    doc_id: request.corpus_document_id(&r.doc_id),
                    track_id: r.track_id,
                })
                .collect();
            expected.sort_by(|a, b| (&a.doc_id, a.track_id).cmp(&(&b.doc_id, b.track_id)));
            if expected == current_refs {
                let mut restored = current.clone();
                restored.minimum_pair_similarity = Some(minimum);
                restored.confidence = original.confidence;
                self.update_shared_identity(restored);
            }
        }
    }

    fn commit_delivery(
        &mut self,
        request: &DocumentDelivery,
        mut doc: GroundedDocument,
        digest: String,
    ) -> Result<DeliveryOutcome, DeliveryError> {
        let source_id = doc.id.clone();
        let key = request.corpus_document_id(&source_id);
        let applied_revision = self
            .state()
            .delivery
            .sequence
            .checked_add(1)
            .ok_or(DeliveryError::Exhausted)?;
        let previous_reviews = self
            .state()
            .delivery
            .heads
            .get(&key)
            .map(|h| h.reviewed.clone())
            .unwrap_or_default();
        // add_document rekeys incoming tracks that could alias retired handles.
        // Check allocation before changing the canonical identity registry.
        if let Some(previous) = self.get_document(&key) {
            doc.replacement_track_ids(previous.next_track_id)
                .ok_or(DeliveryError::Exhausted)?;
        }
        // Prepare all allocations and consistency checks before publishing any changes.
        let mut assigned = HashMap::new();
        let mut new_records = Vec::new();
        let mut new_origins = Vec::new();
        let mut next = self.state().next_identity_id;
        let originals: BTreeMap<_, _> = std::mem::take(&mut doc.identities).into_iter().collect();
        for (original_id, original) in originals {
            let existing = self.restored(request, original_id);
            if existing.is_some_and(|(prior, _)| !prior.matches_copy(Some(&original))) {
                return Err(DeliveryError::OriginConflict);
            }
            let canonical = if let Some(current) =
                existing.and_then(|(_, id)| self.get_identity(*id))
            {
                current.clone()
            } else {
                while self.get_identity(next).is_some()
                    || self.documents().any(|d| d.get_identity(next).is_some())
                {
                    next =
                        IdentityId::new(next.get().checked_add(1).ok_or(DeliveryError::Exhausted)?);
                }
                let id = next;
                next = IdentityId::new(id.get().checked_add(1).ok_or(DeliveryError::Exhausted)?);
                let mut canonical = original.clone();
                canonical.id = id;
                if let Some(refs) = canonical
                    .source
                    .as_mut()
                    .and_then(super::IdentitySource::track_refs_mut)
                {
                    for member in refs {
                        member.doc_id = request.corpus_document_id(&member.doc_id);
                    }
                }
                if let DeliveryOwnership::Restore { origin } = &request.ownership {
                    new_origins.push((
                        (request.issuer.clone(), origin.clone(), original_id),
                        (original, id),
                    ));
                }
                new_records.push(canonical.clone());
                canonical
            };
            assigned.insert(original_id, canonical.id);
            doc.identities.insert(canonical.id, canonical);
        }
        for track in doc.tracks.values_mut() {
            if let Some(id) = track.identity_id {
                track.identity_id = Some(assigned[&id]);
            }
        }
        doc.id = key.clone();
        if let Some(evidence) = &mut doc.relation_evidence {
            evidence.document_id.clone_from(&key);
        }
        doc.rebuild_indexes();
        let state = self.state_mut();
        state.next_identity_id = next;
        state
            .identities
            .extend(new_records.into_iter().map(|r| (r.id, r)));
        state.delivery.origins.extend(new_origins);
        self.insert_document(doc, true);
        if let DeliveryOwnership::Restore { origin } = &request.ownership {
            self.restore_complete_evidence(request, origin, &assigned);
        }
        let state = self.state_mut();
        let doc = &state.documents[&key];
        let submitted = state.submitted_track_ids.get(&key);
        let mut reviewed = BTreeMap::new();
        for track in doc.tracks() {
            let source = submitted
                .and_then(|m| m.get(&track.id))
                .copied()
                .unwrap_or(track.id);
            if track
                .identity_id
                .is_some_and(|id| state.delivery.reviewed_owners.contains(&id))
                || (track.identity_id.is_some() && previous_reviews.contains_key(&source))
            {
                reviewed.insert(source, track.id);
            }
        }
        let receipt = DeliveryReceipt {
            request: request.clone(),
            source_document_id: source_id,
            document_id: key.clone(),
            payload_sha256: digest,
            applied_revision,
        };
        state.delivery.sequence = applied_revision;
        state.delivery.heads.insert(
            key,
            SourceHead {
                revision: request.source_revision,
                reviewed,
            },
        );
        state.delivery.receipts.insert(
            (request.issuer.clone(), request.delivery_id.clone()),
            receipt.clone(),
        );
        Ok(DeliveryOutcome::Applied(receipt))
    }
}
