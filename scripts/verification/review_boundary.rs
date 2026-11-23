//! Compile-time contracts exercised by check_review_boundary.py.
#![allow(dead_code)]

use anno::core::grounded::{Corpus, GroundedDocument, Identity, IdentityId, KbAssignmentPreview};

// Compile the real storage module. Inside this fixture, `sibling` has exactly
// the access granted to a sibling module under `core::grounded`.
#[path = "../../crates/anno/src/core/grounded/corpus_state.rs"]
mod corpus_state;

mod sibling {
    use super::corpus_state::Corpus;

    fn permitted(corpus: &mut Corpus) {
        let _ = corpus.state().identities.len();
        corpus.state_mut().identities.clear();
    }

    #[cfg(review_case = "direct_data")]
    fn bypass_gate(corpus: &mut Corpus) {
        corpus.data.identities.clear();
    }

    #[cfg(review_case = "mutable_read")]
    fn write_through_read(corpus: &mut Corpus) {
        corpus.state().identities.clear();
    }

    #[cfg(review_case = "mutable_deref")]
    fn require_mutable_deref() {
        fn require<T: std::ops::DerefMut>() {}
        require::<Corpus>();
    }

    #[cfg(review_case = "restore_revision")]
    fn restore_approval(corpus: &mut Corpus) {
        let token = corpus.review_token();
        corpus.state_mut().identities.clear();
        let _ = corpus.review_revision.set(token);
    }
}

fn permitted(corpus: &mut Corpus, preview: KbAssignmentPreview) {
    let mut display = preview.assignment().clone();
    display.kb_id = String::from("display only");
    let _ = corpus.apply_kb_assignment(preview);
}

#[cfg(review_case = "edited_copy")]
fn submit_copy(corpus: &mut Corpus, preview: KbAssignmentPreview) {
    let mut display = preview.assignment().clone();
    display.kb_id = String::from("different record");
    let _ = corpus.apply_kb_assignment(display);
}

#[cfg(review_case = "change_target")]
fn edit_proposal(preview: KbAssignmentPreview) {
    preview.assignment().kb_id = String::from("different record");
}

#[cfg(review_case = "direct_proposal")]
fn edit_private_proposal(mut preview: KbAssignmentPreview) {
    preview.assignment.kb_id = String::from("different record");
}

#[cfg(review_case = "read_token")]
fn read_token(preview: KbAssignmentPreview) {
    let _ = preview.revision;
}

#[cfg(review_case = "double_apply")]
fn use_twice(corpus: &mut Corpus, preview: KbAssignmentPreview) {
    let _ = corpus.apply_kb_assignment(preview);
    let _ = corpus.apply_kb_assignment(preview);
}
