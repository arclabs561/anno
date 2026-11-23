//! Caller-supplied annotations for two people with the same canonical surface.
use anno::{GroundedDocument, Identity, Location, Signal, SignalValidationError};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    text: String,
    offset_unit: String,
    catalog: String,
    tracks: Vec<GoldTrack>,
    mentions: Vec<Mention>,
}

#[derive(Deserialize)]
struct GoldTrack {
    id: String,
    label: String,
    catalog_id: Option<String>,
}

#[derive(Deserialize)]
struct Mention {
    surface: String,
    start: usize,
    end: usize,
    track: Option<String>,
}

fn fixture() -> Fixture {
    serde_json::from_str(include_str!("../../../../testdata/grounded/namesakes.json")).unwrap()
}

fn document(gold: &Fixture) -> GroundedDocument {
    let mut doc = GroundedDocument::new("namesakes", &gold.text);
    for m in &gold.mentions {
        doc.add_signal_validated(Signal::new(
            0,
            Location::text(m.start, m.end),
            &m.surface,
            "PERSON",
            1.0,
        ))
        .unwrap();
    }
    for t in &gold.tracks {
        let members: Vec<_> = doc
            .signals()
            .iter()
            .zip(&gold.mentions)
            .filter(|(_, m)| m.track.as_ref() == Some(&t.id))
            .map(|(s, _)| s.id)
            .collect();
        let track = doc.create_track_from_signals(&t.label, &members).unwrap();
        if let Some(key) = &t.catalog_id {
            let identity = doc.add_identity(Identity::from_kb(0, &t.label, &gold.catalog, key));
            doc.link_track_to_identity(track, identity);
        }
    }
    doc
}

// Compare the other members' offsets, independent of allocated track IDs.
// An unassigned signal and a singleton both have no partners.
fn partners(doc: &GroundedDocument, index: usize) -> Vec<usize> {
    let signal = &doc.signals()[index];
    let Some(track) = doc.track_for_signal(signal.id) else {
        return vec![];
    };
    doc.signals()
        .iter()
        .filter(|s| {
            s.id != signal.id && doc.track_for_signal(s.id).is_some_and(|t| t.id == track.id)
        })
        .map(|s| s.location.text_offsets().unwrap().0)
        .collect()
}

fn check_mentions(doc: &GroundedDocument, gold: &Fixture) {
    assert_eq!(doc.text(), gold.text);
    assert_eq!(doc.signals().len(), gold.mentions.len());
    assert!(doc.validate_invariants().is_empty());
    assert!(doc.validate().is_empty());
    for (i, (s, m)) in doc.signals().iter().zip(&gold.mentions).enumerate() {
        assert_eq!(s.location.text_offsets(), Some((m.start, m.end)));
        assert_eq!(s.surface, m.surface);
        let expected: Vec<_> = gold
            .mentions
            .iter()
            .filter(|other| m.track.is_some() && other.track == m.track && other.start != m.start)
            .map(|other| other.start)
            .collect();
        assert_eq!(partners(doc, i), expected, "membership at {}", m.start);
        let target = gold
            .tracks
            .iter()
            .find(|t| Some(&t.id) == m.track.as_ref())
            .and_then(|t| t.catalog_id.as_deref());
        assert_eq!(
            doc.identity_for_signal(s.id)
                .and_then(|id| id.kb_id.as_deref()),
            target
        );
    }
}

#[test]
fn namesake_membership_and_catalog_link_survive_reload_and_checked_regrouping() {
    let gold = fixture();
    let doc = document(&gold);
    let reloaded = serde_json::from_slice(&serde_json::to_vec(&doc).unwrap()).unwrap();
    for d in [&doc, &reloaded] {
        check_mentions(d, &gold);
        assert_eq!(d.tracks().count(), 2);
        let identity = d.identities().next().unwrap();
        assert_eq!(d.identities().count(), 1);
        assert_eq!(identity.kb_name.as_deref(), Some(gold.catalog.as_str()));
        for (s, m) in d.signals().iter().zip(&gold.mentions) {
            let expected = gold.tracks.iter().find(|t| Some(&t.id) == m.track.as_ref());
            assert_eq!(
                d.track_for_signal(s.id).map(|t| &t.canonical_surface),
                expected.map(|t| &t.label)
            );
        }
    }
    let flat = reloaded.to_entities();
    assert!(flat.last().unwrap().canonical_id.is_none());
    let regrouped = GroundedDocument::try_from_entities("regrouped", &gold.text, &flat).unwrap();
    check_mentions(&regrouped, &gold);
    let ambiguous = regrouped.signals().last().unwrap();
    assert_eq!(
        regrouped
            .track_for_signal(ambiguous.id)
            .unwrap()
            .signals()
            .len(),
        1
    );
}

#[test]
fn utf16_coordinates_rejected_as_scalar_offsets() {
    let gold = fixture();
    assert_eq!(gold.offset_unit, "unicode_scalar");
    let first = &gold.mentions[0];
    let prefix: String = gold.text.chars().take(first.start).collect();
    let start = prefix.encode_utf16().count();
    let end = start + first.surface.encode_utf16().count();
    assert_eq!((start, end), (18, 26));
    let mut doc = GroundedDocument::new("wrong-unit", &gold.text);
    let result = doc.add_signal_validated(Signal::new(
        0,
        Location::text(start, end),
        &first.surface,
        "PERSON",
        1.0,
    ));
    assert!(matches!(result, Err(SignalValidationError::TextMismatch {
        start: 18, end: 26, expected, actual, ..
    }) if expected == first.surface && actual == "éa Noor "));
    assert!(doc.signals().is_empty());
}

#[test]
fn swapped_namesakes_change_membership_despite_valid_spans_and_equal_counts() {
    let mut gold = fixture();
    let correct = document(&gold);
    assert_eq!(gold.mentions[0].surface, gold.mentions[3].surface);
    let (first, rest) = gold.mentions.split_at_mut(3);
    std::mem::swap(&mut first[0].track, &mut rest[0].track);
    let swapped = document(&gold);
    assert!(swapped.validate_invariants().is_empty());
    assert!(swapped.validate().is_empty());
    for doc in [&correct, &swapped] {
        assert_eq!(doc.tracks().count(), 2);
        assert!(doc.tracks().all(|t| t.signals().len() == 3));
    }
    assert_eq!(partners(&correct, 0), [33, 48]); // alias and pronoun
    assert_eq!(partners(&swapped, 0), [84, 102]); // both pilot mentions
    assert_eq!(partners(&swapped, 3), partners(&correct, 0));
}
