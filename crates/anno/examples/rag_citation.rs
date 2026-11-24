// SPDX-License-Identifier: MIT OR Apache-2.0
//! Read a person-specific search result back against the source that supports it.
//!
//! `cargo run --locked -p anno --no-default-features --example rag_citation`
//!
//! This two-note fixture supplies the mention spans, tracks and catalog links.
//! The search rule is a literal sentence lookup. No extractor, ranker or language
//! model is evaluated here.

use anno::core::grounded::html_source::source_sha256;
use anno::offset::chars_to_bytes;
use anno::rag::resolve_for_rag;
use anno::{GroundedDocument, Identity, Signal, SignalId};

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
    doc_id: String,
    source_digest: String,
    subject: SignalId,
    start: usize,
    pronoun_end: usize,
    sentence_end: usize,
    indexed_sentence: String,
    flat_accession: String,
}

impl Hit {
    fn from_note(doc: &GroundedDocument, subject: SignalId) -> Self {
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
            doc_id: doc.id().into(),
            source_digest: source_sha256(doc.text()),
            subject,
            start: rewrite.start,
            pronoun_end: rewrite.end,
            sentence_end: doc.text().chars().count(),
            indexed_sentence: indexed.into(),
            flat_accession: flat_subject.kb_id.clone().unwrap(),
        }
    }

    fn source_quote<'a>(&self, doc: &'a GroundedDocument) -> Option<&'a str> {
        if doc.id() != self.doc_id || source_sha256(doc.text()) != self.source_digest {
            return None;
        }
        let (start, end) = chars_to_bytes(doc.text(), self.start, self.sentence_end);
        doc.text().get(start..end)
    }

    fn quote_for<'a>(
        &self,
        doc: &'a GroundedDocument,
        catalog: &str,
        key: &str,
    ) -> Option<&'a str> {
        let quote = self.source_quote(doc)?;
        let subject = doc.get_signal(self.subject)?;
        if subject.text_offsets() != Some((self.start, self.pronoun_end)) {
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
    let music_hit = Hit::from_note(&music, music_subject);
    let tides_hit = Hit::from_note(&tides, tides_subject);
    let rows = [(&music_hit, &music), (&tides_hit, &tides)];

    let original = rows
        .iter()
        .filter(|(hit, doc)| hit.source_quote(doc).unwrap().contains(QUERY))
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
        .filter_map(|(hit, doc)| hit.quote_for(doc, "music", "17"))
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
    assert!(music_hit.source_quote(&revised).is_none());
    let (start, end) = chars_to_bytes(revised.text(), music_hit.start, music_hit.sentence_end);
    assert_eq!(revised.text().get(start..end), Some(answers[0]));
    println!("The quote still occupies the same range in the edited text. The citation selects the original edition.");

    let mut corrected = music.clone();
    let track = corrected.track_for_signal(music_subject).unwrap().id;
    let new_record = corrected.add_identity(Identity::from_kb(0, NAME, "music", "23"));
    corrected.link_track_to_identity(track, new_record);
    assert_eq!(music_hit.source_quote(&corrected), Some(answers[0]));
    assert!(music_hit.quote_for(&corrected, "music", "17").is_none());
    assert_eq!(
        music_hit.quote_for(&corrected, "music", "23"),
        Some(answers[0])
    );
    println!("The catalog correction preserves the source quote and moves the supported answer from music/17 to music/23.");
    inspect_partial_note();
}
