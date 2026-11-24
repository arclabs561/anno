// SPDX-License-Identifier: MIT OR Apache-2.0
//! Downstream reader checks for edition-bound source quotes and editorial histories.

use anno::citation::{
    CitationError, Correspondence, EditorialReview, Successor, SuccessorDecision, TextCitation,
    TextEdition,
};
use anno::offset::CharRange;
use anno::GroundedDocument;

const EARLIER: &str = "Desk: é.\nLoft: é.\n";
const TWIN: &str = "Desk: é.\nLoft: é.\nLoft: é.\n";

fn review<'a>(
    decisions: &'a [SuccessorDecision],
    authority: &'a str,
    revision: u64,
) -> EditorialReview<'a> {
    EditorialReview {
        authority,
        through_revision: revision,
        decisions,
    }
}

fn decision(
    original: &TextCitation,
    edition: &TextEdition<'_>,
    authority: &str,
    revision: u64,
    correspondence: Correspondence,
) -> SuccessorDecision {
    SuccessorDecision {
        original: original.clone(),
        target: edition.locator().clone(),
        authority: authority.into(),
        revision,
        correspondence,
    }
}

fn candidates(
    target: &TextEdition<'_>,
    context_chars: usize,
    spans: &[(usize, usize)],
) -> Successor<'static> {
    Successor::Candidates {
        target: target.locator().clone(),
        context_chars,
        ranges: spans.iter().copied().map(CharRange::from).collect(),
    }
}

#[test]
fn historical_reads_require_all_edition_fields_and_accessible_evidence() {
    let doc = GroundedDocument::new("log", EARLIER);
    let source = TextEdition::new("publisher", "1", &doc).unwrap();
    let citation = source.cite((15, 16).into()).unwrap();
    let saved = serde_json::to_string(&citation).unwrap();
    let restored: TextCitation = serde_json::from_str(&saved).unwrap();
    assert_eq!(restored.open(Some(&source)), Ok("é"));
    assert_eq!(restored.open(None), Err(CitationError::EvidenceUnavailable));
    let different_doc = GroundedDocument::new("another", EARLIER);
    let edited = GroundedDocument::new("log", "Desk: ë.\nLoft: é.\n");
    for other in [
        TextEdition::new("other", "1", &doc).unwrap(),
        TextEdition::new("publisher", "2", &doc).unwrap(),
        TextEdition::new("publisher", "1", &different_doc).unwrap(),
        TextEdition::new("publisher", "1", &edited).unwrap(),
    ] {
        assert_eq!(
            restored.open(Some(&other)),
            Err(CitationError::EvidenceUnavailable)
        );
    }
    for bad in [(5, 5), (5, 4), (15, usize::MAX)] {
        assert_eq!(source.cite(bad.into()), Err(CitationError::InvalidRange));
        assert_eq!(
            TextCitation {
                range: bad.into(),
                ..restored.clone()
            }
            .open(Some(&source)),
            Err(CitationError::EvidenceUnavailable)
        );
    }
}

#[test]
fn duplicate_before_and_after_need_separate_editorial_lineage() {
    let first = GroundedDocument::new("log", EARLIER);
    let last = GroundedDocument::new("log", TWIN);
    let original = TextEdition::new("press", "1", &first).unwrap();
    let next = TextEdition::new("press", "2", &last).unwrap();
    let citation = original.cite((15, 16).into()).unwrap();
    // Same edition bytes in both hypothetical histories. The two physical Loft rows
    // have identical context. Lineage alone identifies which one was retained.
    for expected in [(24, 25), (15, 16)] {
        let d = decision(
            &citation,
            &next,
            "editor",
            3,
            Correspondence::Confirmed(expected.into()),
        );
        let history = [d.clone()];
        assert_eq!(
            citation
                .successor(&original, &next, 6, review(&history, "editor", 2))
                .unwrap(),
            candidates(&next, 6, &[(15, 16), (24, 25)])
        );
        assert_eq!(
            citation
                .successor(&original, &next, 6, review(&history, "another", 3))
                .unwrap(),
            candidates(&next, 6, &[(15, 16), (24, 25)])
        );
        assert_eq!(
            citation
                .successor(&original, &next, 6, review(&history, "editor", 3))
                .unwrap(),
            Successor::Confirmed {
                citation: next.cite(expected.into()).unwrap(),
                decision: &d
            }
        );
        assert_eq!(citation.open(Some(&original)), Ok("é"));
    }
}

#[test]
fn identical_replacement_and_renamed_field_have_independent_as_of_decisions() {
    let doc = GroundedDocument::new("log", EARLIER);
    let old = TextEdition::new("press", "1", &doc).unwrap();
    let same = TextEdition::new("press", "2", &doc).unwrap();
    let citation = old.cite((15, 16).into()).unwrap();
    let continued = decision(
        &citation,
        &same,
        "a",
        1,
        Correspondence::Confirmed((15, 16).into()),
    );
    let withdrawn = decision(&citation, &same, "a", 4, Correspondence::Withdrawn);
    let history = [withdrawn.clone(), continued.clone(), continued.clone()];
    assert_eq!(
        citation
            .successor(&old, &same, usize::MAX, review(&history, "a", 0))
            .unwrap(),
        candidates(&same, usize::MAX, &[(15, 16)])
    );
    assert_eq!(
        citation
            .successor(&old, &same, 0, review(&history, "a", 2))
            .unwrap(),
        Successor::Confirmed {
            citation: same.cite((15, 16).into()).unwrap(),
            decision: &continued
        }
    );
    assert_eq!(
        citation
            .successor(&old, &same, 0, review(&history, "a", 5))
            .unwrap(),
        Successor::Withdrawn(&withdrawn)
    );
    let renamed = GroundedDocument::new("log", "🦊 moved\nLoft: ël.\n");
    let moved = TextEdition::new("press", "3", &renamed).unwrap();
    // Existing records for edition 2 cannot establish either a deletion or a mapping in edition 3.
    assert_eq!(
        citation
            .successor(&old, &moved, 0, review(&history, "a", 5))
            .unwrap(),
        candidates(&moved, 0, &[])
    );
    let mapping = [decision(
        &citation,
        &moved,
        "a",
        5,
        Correspondence::Confirmed((14, 16).into()),
    )];
    let Successor::Confirmed {
        citation: successor,
        ..
    } = citation
        .successor(&old, &moved, 0, review(&mapping, "a", 5))
        .unwrap()
    else {
        panic!("expected editor's correspondence")
    };
    assert_eq!(successor.open(Some(&moved)), Ok("ël"));
    assert_eq!(citation.open(Some(&old)), Ok("é"));
}

#[test]
fn wrong_scope_conflicting_reviews_and_invalid_ranges_fail_closed() {
    let doc = GroundedDocument::new("log", "éééé");
    let old = TextEdition::new("press", "1", &doc).unwrap();
    let new = TextEdition::new("press", "2", &doc).unwrap();
    let other = TextEdition::new("secondary", "2", &doc).unwrap();
    let source = old.cite((0, 2).into()).unwrap();
    assert_eq!(
        source
            .successor(&old, &new, 0, review(&[], "a", 1))
            .unwrap(),
        candidates(&new, 0, &[(0, 2), (1, 3), (2, 4)])
    );
    assert_eq!(
        source.successor(&old, &other, 0, review(&[], "a", 1)),
        Err(CitationError::WrongTarget)
    );
    assert_eq!(
        source.successor(&new, &old, 0, review(&[], "a", 1)),
        Err(CitationError::EvidenceUnavailable)
    );
    assert_eq!(
        source.successor(&old, &old, 0, review(&[], "a", 1)),
        Err(CitationError::WrongTarget)
    );
    let conflict = [
        decision(&source, &new, "a", 1, Correspondence::Withdrawn),
        decision(
            &source,
            &new,
            "a",
            1,
            Correspondence::Confirmed((0, 2).into()),
        ),
    ];
    assert_eq!(
        source.successor(&old, &new, 0, review(&conflict, "a", 1)),
        Err(CitationError::ConflictingDecisions)
    );
    let corrupt = [decision(
        &source,
        &new,
        "a",
        1,
        Correspondence::Confirmed((99, 101).into()),
    )];
    assert_eq!(
        source.successor(&old, &new, 0, review(&corrupt, "a", 1)),
        Err(CitationError::InvalidRange)
    );
}
