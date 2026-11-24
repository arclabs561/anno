// SPDX-License-Identifier: MIT OR Apache-2.0
//! Read a person-specific search result back against the source that supports it.
//!
//! `cargo run --locked -p anno --no-default-features --example rag_citation`
//!
//! This two-note fixture supplies the mention spans, tracks and catalog links.
//! The search rule is a literal sentence lookup. No extractor, ranker or language
//! model is evaluated here.

use anno::citation::{
    Correspondence, EditorialReview, Successor, SuccessorDecision, TextCitation, TextEdition,
};
use anno::offset::CharRange;
use anno::rag::resolve_for_rag;
use anno::{GroundedDocument, Identity, Signal, SignalId};
use std::collections::BTreeMap;

const STREAM: &str = "sound-archive";
const FIRST_EDITION: &str = "printed-1";
const NAME: &str = "Éva Lorn";
const QUERY: &str = "Éva Lorn recorded";

fn note(id: &str, text: &str, catalog: &str) -> (GroundedDocument, SignalId) {
    let mut doc = GroundedDocument::new(id, text);
    let name = doc.add_signal(Signal::from_text(text, NAME, "PER", 1.0).unwrap());
    let subject = doc.add_signal(Signal::from_text(text, "She", "PER", 1.0).unwrap());
    let track = doc
        .create_track_from_signals(NAME, &[name, subject])
        .unwrap();
    let identity = doc.add_identity(Identity::from_kb(0, NAME, catalog, "17"));
    doc.link_track_to_identity(track, identity);
    (doc, subject)
}

struct Hit {
    citation: TextCitation,
    subject: SignalId,
    pronoun_end: usize,
    indexed_sentence: String,
    flat_accession: String,
}

impl Hit {
    fn from_note(doc: &GroundedDocument, subject: SignalId, source_revision: &str) -> Self {
        let entities = doc.to_entities();
        let result = resolve_for_rag(doc.text(), &entities, None);
        let [rewrite] = result.rewrites.as_slice() else {
            panic!("each sample note has one pronoun rewrite")
        };
        let original = doc.get_signal(subject).unwrap();
        assert_eq!(original.text_offsets(), Some((rewrite.start, rewrite.end)));
        assert_eq!(rewrite.original, original.surface());
        assert_eq!(rewrite.replacement, NAME);
        let flat_subject = entities
            .iter()
            .find(|e| (e.start(), e.end()) == (rewrite.start, rewrite.end))
            .unwrap();

        // The observation is the final sentence in each of these two notes.
        let (_, indexed) = result.text.split_once(". ").unwrap();
        Hit {
            citation: TextEdition::new(STREAM, source_revision, doc)
                .unwrap()
                .cite(CharRange::from_raw(
                    rewrite.start,
                    doc.text().chars().count(),
                ))
                .unwrap(),
            subject,
            pronoun_end: rewrite.end,
            indexed_sentence: indexed.into(),
            flat_accession: flat_subject.kb_id.clone().unwrap(),
        }
    }

    fn source_quote<'a>(
        &self,
        doc: &'a GroundedDocument,
        source_revision: &str,
    ) -> Option<&'a str> {
        let edition = TextEdition::new(STREAM, source_revision, doc).ok()?;
        self.citation.open(Some(&edition)).ok()
    }

    fn quote_for<'a>(
        &self,
        doc: &'a GroundedDocument,
        source_revision: &str,
        catalog: &str,
        key: &str,
    ) -> Option<&'a str> {
        let quote = self.source_quote(doc, source_revision)?;
        let subject = doc.get_signal(self.subject)?;
        if subject.text_offsets() != Some((self.citation.range.start.get(), self.pronoun_end)) {
            return None;
        }
        let current = doc.identity_for_signal(self.subject)?.qualified_kb_ref()?;
        (current == (catalog, key)).then_some(quote)
    }
}

// A partially annotated note mixes an untracked detection with a supplied track.
// Readers can compare a proposed rewrite with the antecedent on that track.
fn inspect_partial_note() {
    let source = "Iris Moss checked the bell. Éva Lorn tuned the reeds. She recorded the chime.";
    let mut doc = GroundedDocument::new("reed-log", source);
    let [untracked, name, pronoun] = ["Iris Moss", NAME, "She"]
        .map(|word| doc.add_signal(Signal::from_text(source, word, "PER", 1.0).unwrap()));
    doc.create_track_from_signals(NAME, &[name, pronoun])
        .unwrap();
    assert!(doc.track_for_signal(untracked).is_none());
    let result = resolve_for_rag(source, &doc.to_entities(), None);
    let [rewrite] = result.rewrites.as_slice() else {
        panic!("the partial note supplies a track for its pronoun")
    };
    let antecedent = doc
        .track_for_signal(pronoun)
        .unwrap()
        .canonical_surface
        .as_str();
    let agrees = rewrite.replacement == antecedent;
    println!(
        "The supplied track names {antecedent:?}. The RAG rewrite uses {:?}. Agreement = {agrees}.",
        rewrite.replacement
    );
}

fn main() {
    let (music, music_subject) = note(
        "bell-log",
        "Éva Lorn tuned a bell. She recorded the chime.",
        "music",
    );
    let (tides, tides_subject) = note(
        "tide-log",
        "Éva Lorn set a gauge. She recorded the tide.",
        "tides",
    );
    let music_hit = Hit::from_note(&music, music_subject, FIRST_EDITION);
    let tides_hit = Hit::from_note(&tides, tides_subject, FIRST_EDITION);
    let rows = [(&music_hit, &music), (&tides_hit, &tides)];

    let original = rows
        .iter()
        .filter(|(hit, doc)| {
            hit.source_quote(doc, FIRST_EDITION)
                .unwrap()
                .contains(QUERY)
        })
        .count();
    let hits: Vec<_> = rows
        .iter()
        .filter(|(hit, _)| hit.indexed_sentence.contains(QUERY))
        .collect();
    let by_accession = hits
        .iter()
        .filter(|(hit, _)| hit.flat_accession == "17")
        .count();
    let answers: Vec<_> = hits
        .iter()
        .filter_map(|(hit, doc)| hit.quote_for(doc, FIRST_EDITION, "music", "17"))
        .collect();
    assert_eq!((original, hits.len(), by_accession), (0, 2, 2));
    assert_eq!(answers, ["She recorded the chime."]);
    assert!(!music.text().contains(&music_hit.indexed_sentence));
    println!(
        "Exact sentence lookup found {original}/2 observations before rewriting and {}/2 afterward.",
        hits.len()
    );
    println!(
        "Accession 17 returns {by_accession} sentences for music/17, including one from tides/17."
    );
    println!(
        "For music/17 the index says {:?}. The source says {:?}.",
        music_hit.indexed_sentence, answers[0]
    );

    let revised = GroundedDocument::new(music.id(), music.text().replace("bell", "gong"));
    assert!(revised.text().ends_with(answers[0]));
    let original_edition = TextEdition::new(STREAM, FIRST_EDITION, &music).unwrap();
    let new_edition = TextEdition::new(STREAM, "printed-2", &revised).unwrap();
    assert!(music_hit.citation.open(Some(&new_edition)).is_err());
    let review = |decisions| {
        Some(EditorialReview {
            authority: "archive-editor",
            through_revision: 4,
            decisions,
        })
    };
    let Successor::Candidates { ranges, .. } = music_hit
        .citation
        .successor(&original_edition, &new_edition, 0, None)
        .unwrap()
    else {
        panic!("no reviewed correspondence")
    };
    assert_eq!(ranges.len(), 1);
    assert_eq!(
        new_edition
            .cite(ranges[0])
            .unwrap()
            .open(Some(&new_edition))
            .unwrap(),
        answers[0]
    );
    // The editor's independent correspondence can be stored with the two edition locators.
    let decisions = [SuccessorDecision {
        original: music_hit.citation.clone(),
        target: new_edition.locator().clone(),
        authority: "archive-editor".into(),
        revision: 4,
        correspondence: Correspondence::Confirmed(music_hit.citation.range),
    }];
    assert!(matches!(
        music_hit
            .citation
            .successor(&original_edition, &new_edition, 0, review(&decisions)),
        Ok(Successor::Confirmed { .. })
    ));
    println!("The later edition contains one matching quote. The editor confirmed its correspondence at revision 4.");

    let mut corrected = music.clone();
    let track = corrected.track_for_signal(music_subject).unwrap().id;
    let new_record = corrected.add_identity(Identity::from_kb(0, NAME, "music", "23"));
    corrected.link_track_to_identity(track, new_record);
    assert_eq!(
        music_hit.source_quote(&corrected, FIRST_EDITION),
        Some(answers[0])
    );
    // A catalog reader selects an authorized annotation snapshot separately from the
    // cited print edition. Keys in this example are scoped to this note and print edition.
    let snapshots = BTreeMap::from([
        (("music-editor", 1u64), &music),
        (("music-editor", 3), &corrected),
        (("first-cataloguer", 3), &music),
    ]);
    let as_of = |authority, revision, accession: &str| {
        let (_, snapshot) = snapshots
            .range((authority, 0)..=(authority, revision))
            .next_back()?;
        music_hit.quote_for(snapshot, FIRST_EDITION, "music", accession)
    };
    assert_eq!(as_of("music-editor", 1, "17"), Some(answers[0]));
    assert_eq!(as_of("music-editor", 3, "23"), Some(answers[0]));
    assert_eq!(as_of("music-editor", 3, "17"), None);
    assert_eq!(as_of("first-cataloguer", 3, "17"), Some(answers[0]));
    assert_eq!(as_of("music-editor", 0, "17"), None);
    println!("The printed quote stays available. The music editor recorded accession 17 at revision 1 and 23 at revision 3.");
    inspect_partial_note();
}
