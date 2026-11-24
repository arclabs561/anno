//! Exercise the full grounded JSON reader and its human-readable signal display.
use anno::{GroundedDocument, Identity, Location, Signal};
use std::process::Command;

#[test]
fn reading_a_nested_name_keeps_the_phrase_membership_and_catalog() {
    let source = "🪶 Captain Iva Reed arrived. She spoke.";
    let mut original = GroundedDocument::new("memo", source);
    let mut add = |start, end, surface, label| {
        original
            .add_signal_validated(Signal::new(
                0,
                Location::text(start, end),
                surface,
                label,
                1.0,
            ))
            .unwrap()
    };
    let phrase = add(2, 18, "Captain Iva Reed", "COREF_mention");
    let name = add(10, 18, "Iva Reed", "PER");
    let pronoun = add(28, 31, "She", "COREF_mention");
    let track = original
        .create_track_from_signals("Captain Iva Reed", &[phrase, pronoun])
        .unwrap();
    let identity = original.add_identity(Identity::from_kb(0, "Iva Reed", "fixture-catalog", "A"));
    original.link_track_to_identity(track, identity);

    let file = tempfile::NamedTempFile::new().unwrap();
    serde_json::to_writer(&file, &original).unwrap();
    let enhance = |format| {
        let output = Command::new(env!("CARGO_BIN_EXE_anno"))
            .arg("enhance")
            .arg(file.path())
            .args(["--format", format, "--quiet", "--color", "never"])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        output.stdout
    };

    let restored: GroundedDocument = serde_json::from_slice(&enhance("grounded")).unwrap();
    assert_eq!(restored.text(), source);
    for id in [phrase, name, pronoun] {
        let before = original.get_signal(id).unwrap();
        let after = restored.get_signal(id).unwrap();
        assert_eq!(after.text_offsets(), before.text_offsets());
        assert_eq!(after.surface(), before.surface());
    }
    assert_eq!(restored.track_for_signal(phrase).unwrap().id, track);
    assert_eq!(restored.track_for_signal(pronoun).unwrap().id, track);
    assert!(restored.track_for_signal(name).is_none());
    let linked = restored.identity_for_signal(phrase).unwrap();
    assert_eq!(linked.kb_name(), Some("fixture-catalog"));
    assert_eq!(linked.kb_id(), Some("A"));
    assert!(restored.identity_for_signal(name).is_none());

    let human = String::from_utf8(enhance("human")).unwrap();
    assert!(human.contains("\"Captain Iva Reed\""));
    assert!(human.contains("\"Iva Reed\""));
}
