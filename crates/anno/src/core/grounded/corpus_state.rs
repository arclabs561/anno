//! Own the corpus maps and their review revision together. Writers in sibling
//! modules must go through `state_mut` to change a binding or hand out mutable access.

use super::{GroundedDocument, Identity, IdentityId, TrackId};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock};

/// A corpus of grounded documents for cross-document operations.
///
/// Enables inter-document coreference resolution and entity linking
/// across multiple documents.
#[derive(Debug, Default)]
pub struct Corpus {
    data: CorpusData,
    review_revision: OnceLock<Arc<()>>,
}

#[derive(Clone, Debug, Default)]
pub(super) struct CorpusData {
    pub(super) documents: HashMap<String, GroundedDocument>,
    pub(super) identities: HashMap<IdentityId, Identity>,
    pub(super) next_identity_id: IdentityId,
    pub(super) resolver_identity_ids: HashSet<IdentityId>,
    // Live handle -> submitted local ID, only for tracks rekeyed in the current edition.
    pub(super) submitted_track_ids: HashMap<String, HashMap<TrackId, TrackId>>,
}

impl Clone for Corpus {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            // An independent corpus must never accept the original's previews.
            review_revision: OnceLock::new(),
        }
    }
}

impl Corpus {
    pub(super) fn state(&self) -> &CorpusData {
        &self.data
    }

    pub(super) fn state_mut(&mut self) -> &mut CorpusData {
        self.review_revision.take();
        &mut self.data
    }

    // A preview keeps this allocation alive, so its identity cannot be reused.
    pub(super) fn review_token(&self) -> Arc<()> {
        Arc::clone(self.review_revision.get_or_init(Arc::default))
    }

    pub(super) fn accepts_review(&self, token: &Arc<()>) -> bool {
        self.review_revision
            .get()
            .is_some_and(|revision| Arc::ptr_eq(revision, token))
    }
}
