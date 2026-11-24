use super::{Corpus, Identity, IdentityId, Track, TrackRef};
use crate::core::error::Error;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Catalog fields visible on the triggering track before a reviewed assignment.
/// The original strings are retained even for imports with a missing namespace.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KbAssignmentPrior {
    /// Identity previously visible in the trigger document.
    pub identity_id: IdentityId,
    /// Previous catalog name, when recorded.
    pub kb_name: Option<String>,
    /// Previous accession, when recorded.
    pub kb_id: Option<String>,
    /// Previous display label.
    pub canonical_name: String,
}

/// The requested catalog assignment and the live track bindings within its scope.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KbAssignment {
    /// Track through which the caller requested the assignment.
    pub trigger: TrackRef,
    /// Trigger's previous identity and catalog binding, if present.
    pub previous: Option<KbAssignmentPrior>,
    /// True when the write updates the existing corpus identity in place.
    /// Otherwise it creates a corpus identity for the trigger alone.
    pub updates_shared_identity: bool,
    /// Live track bindings covered by the assignment, ordered by document and track.
    /// Matching identity copies in documents without a bound track can also be refreshed.
    pub affected_tracks: Vec<TrackRef>,
    /// New catalog name.
    pub kb_name: String,
    /// New catalog accession.
    pub kb_id: String,
    /// New display label.
    pub canonical_name: String,
}

/// An immutable proposal tied to one in-memory corpus revision. Dropping it cancels.
#[derive(Debug)]
#[must_use]
pub struct KbAssignmentPreview {
    revision: Arc<()>,
    assignment: KbAssignment,
}

impl KbAssignmentPreview {
    /// Inspect the proposed change. A copied description can be displayed or serialized.
    pub fn assignment(&self) -> &KbAssignment {
        &self.assignment
    }
}

/// Result of a successful reviewed write. Callers can retain this for their own history.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KbAssignmentReceipt {
    /// Identity actually created or updated.
    pub identity_id: IdentityId,
    /// Reviewed scope, previous binding, and applied target.
    pub assignment: KbAssignment,
}

/// Failure to prepare or apply a reviewed catalog assignment.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum KbAssignmentError {
    /// The proposed external reference needs a catalog name and accession.
    #[error("expected a named catalog and a nonempty accession")]
    InvalidCatalogKey,
    /// An independent document record under the shared ID already carries the proposed
    /// catalog name, accession and label. Updating could make it appear corpus-owned.
    #[error(
        "an independent local identity under the shared ID already carries this target and label"
    )]
    AmbiguousLocalIdentity,
    /// The corpus changed or the preview belongs to a different corpus instance.
    #[error("this preview is stale for this corpus; prepare a fresh assignment")]
    StalePreview,
    /// The underlying corpus lookup or assignment failed.
    #[error(transparent)]
    Corpus(#[from] Error),
}

pub(super) fn qualified_key<'a>(name: &'a str, accession: &'a str) -> Option<(&'a str, &'a str)> {
    (!name.trim().is_empty()
        && !name.trim().eq_ignore_ascii_case("unknown")
        && !accession.trim().is_empty())
    .then_some((name, accession))
}

impl Corpus {
    pub(super) fn kb_link_track(
        &self,
        trigger: &TrackRef,
    ) -> Result<(&Track, Option<&Identity>), Error> {
        let document = self.get_document(&trigger.doc_id).ok_or_else(|| {
            Error::track_ref(format!("Document '{}' not found in corpus", trigger.doc_id))
        })?;
        let track = document.get_track(trigger.track_id).ok_or_else(|| {
            Error::track_ref(format!(
                "Track {} not found in document '{}'",
                trigger.track_id, trigger.doc_id
            ))
        })?;
        Ok((
            track,
            track.identity_id.and_then(|id| document.get_identity(id)),
        ))
    }

    pub(super) fn shared_kb_record(
        &self,
        track: &Track,
        local: Option<&Identity>,
    ) -> Option<&Identity> {
        let canonical = self.get_identity(track.identity_id?)?;
        canonical.matches_copy(local).then_some(canonical)
    }

    /// Prepare a catalog write for inspection without holding a borrow during review.
    ///
    /// The scope includes KB-only bindings and the live members of a coreference
    /// identity. Preparing and dropping a preview leaves those bindings untouched.
    /// Previews are local to this corpus instance. Corpus changes, including unrelated
    /// document writes and handing out mutable document access, may require a fresh
    /// preview. Read-only access and preparing another assignment preserve it.
    ///
    /// ```
    /// use anno::core::grounded::{Corpus, GroundedDocument, Track};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut doc = GroundedDocument::new("edition", "North Quay");
    /// let id = doc.add_track(Track::new(0, "North Quay"));
    /// let byline = doc.track_ref(id).unwrap();
    /// let mut corpus = Corpus::new();
    /// corpus.add_document(doc);
    ///
    /// let preview = corpus.preview_kb_assignment(&byline, "library", "42", "North Quay")?;
    /// assert_eq!(preview.assignment().affected_tracks, [byline]);
    /// let receipt = corpus.apply_kb_assignment(preview)?;
    /// assert_eq!(corpus.get_identity(receipt.identity_id).unwrap().kb_id(), Some("42"));
    /// # Ok(())
    /// # }
    /// ```
    pub fn preview_kb_assignment(
        &self,
        trigger: &TrackRef,
        kb_name: impl Into<String>,
        kb_id: impl Into<String>,
        canonical_name: impl Into<String>,
    ) -> Result<KbAssignmentPreview, KbAssignmentError> {
        let (kb_name, kb_id) = (kb_name.into(), kb_id.into());
        let canonical_name = canonical_name.into();
        qualified_key(&kb_name, &kb_id).ok_or(KbAssignmentError::InvalidCatalogKey)?;
        let (track, local) = self.kb_link_track(trigger)?;
        let shared = self.shared_kb_record(track, local);
        let affected_tracks = if let Some(identity) = shared {
            let mut members = Vec::new();
            for (doc_id, document) in &self.state().documents {
                let Some(copy) = document.get_identity(identity.id) else {
                    continue;
                };
                if identity.matches_copy(Some(copy)) {
                    members.extend(
                        document
                            .tracks
                            .iter()
                            .filter(|(_, t)| t.identity_id == Some(identity.id))
                            .map(|(&track_id, _)| TrackRef {
                                doc_id: doc_id.clone(),
                                track_id,
                            }),
                    );
                } else if copy.kb_name() == Some(&kb_name)
                    && copy.kb_id() == Some(&kb_id)
                    && copy.canonical_name() == canonical_name
                {
                    // Shared ownership is recognized by record equality. A write also
                    // refreshes membership and its diagnostics, so a currently distinct
                    // record with the proposed binding could become a matching copy.
                    return Err(KbAssignmentError::AmbiguousLocalIdentity);
                }
            }
            members.sort_by(|a, b| (&a.doc_id, a.track_id).cmp(&(&b.doc_id, b.track_id)));
            members
        } else {
            vec![trigger.clone()]
        };
        Ok(KbAssignmentPreview {
            revision: self.review_token(),
            assignment: KbAssignment {
                trigger: trigger.clone(),
                previous: local.map(|identity| KbAssignmentPrior {
                    identity_id: identity.id,
                    kb_name: identity.kb_name.clone(),
                    kb_id: identity.kb_id.clone(),
                    canonical_name: identity.canonical_name.clone(),
                }),
                updates_shared_identity: shared.is_some(),
                affected_tracks,
                kb_name,
                kb_id,
                canonical_name,
            },
        })
    }

    /// Apply a reviewed assignment under an exclusive borrow and return its receipt.
    /// A stale or foreign preview fails before writing. A successful apply consumes
    /// the preview and invalidates other outstanding previews for this corpus.
    pub fn apply_kb_assignment(
        &mut self,
        preview: KbAssignmentPreview,
    ) -> Result<KbAssignmentReceipt, KbAssignmentError> {
        if !self.accepts_review(&preview.revision) {
            return Err(KbAssignmentError::StalePreview);
        }
        let assignment = preview.assignment;
        let identity_id = self.link_track_to_kb(
            &assignment.trigger,
            &assignment.kb_name,
            &assignment.kb_id,
            &assignment.canonical_name,
        )?;
        self.protect_reviewed_bindings(&assignment.affected_tracks);
        // Accepting a proposal retires competing reviews even for an unchanged target.
        self.state_mut();
        Ok(KbAssignmentReceipt {
            identity_id,
            assignment,
        })
    }
}
