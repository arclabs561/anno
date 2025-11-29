//! Generate an HTML analysis file for a grounded document.
//!
//! Run with: `cargo run --example grounded_analysis_output`
//!
//! This writes a file `grounded_analysis.html` to the current directory
//! that can be opened in a browser to visualize the Signal → Track → Identity
//! hierarchy.

use anno::grounded::{render_document_html, GroundedDocument, Identity, Location, Signal, Track};

/// Helper to find exact substring offsets in text
fn find_offset(text: &str, needle: &str, start_from: usize) -> Option<(usize, usize)> {
    text[start_from..].find(needle).map(|pos| {
        let start = start_from + pos;
        (start, start + needle.len())
    })
}

fn main() -> std::io::Result<()> {
    let text = "Marie Curie was a pioneering physicist who won the Nobel Prize in Physics in 1903. She later won the Nobel Prize in Chemistry in 1911. Her research on radioactivity laid the foundation for modern nuclear physics. Marie Curie remains an inspiration to scientists worldwide.";

    let mut doc = GroundedDocument::new("curie_analysis", text);

    // Find offsets dynamically to avoid manual counting errors
    let (s1_start, s1_end) = find_offset(text, "Marie Curie", 0).unwrap();
    let (s2_start, s2_end) = find_offset(text, "Nobel Prize", 0).unwrap();
    let (s3_start, s3_end) = find_offset(text, "Physics", 0).unwrap();
    let (s4_start, s4_end) = find_offset(text, "1903", 0).unwrap();
    let (s5_start, s5_end) = find_offset(text, "She", 0).unwrap();
    let (s6_start, s6_end) = find_offset(text, "Nobel Prize", s2_end).unwrap(); // second occurrence
    let (s7_start, s7_end) = find_offset(text, "Chemistry", 0).unwrap();
    let (s8_start, s8_end) = find_offset(text, "1911", 0).unwrap();
    let (s9_start, s9_end) = find_offset(text, "Her", s5_end).unwrap();
    let (s10_start, s10_end) = find_offset(text, "radioactivity", 0).unwrap();
    let (s11_start, s11_end) = find_offset(text, "nuclear physics", 0).unwrap();
    let (s12_start, s12_end) = find_offset(text, "Marie Curie", s1_end).unwrap(); // second occurrence

    // Add signals (Level 1: Raw detections)
    let s1 = doc.add_signal(Signal::new(
        0,
        Location::text(s1_start, s1_end),
        "Marie Curie",
        "PER",
        0.97,
    ));
    let s2 = doc.add_signal(Signal::new(
        0,
        Location::text(s2_start, s2_end),
        "Nobel Prize",
        "MISC",
        0.94,
    ));
    let s3 = doc.add_signal(Signal::new(
        0,
        Location::text(s3_start, s3_end),
        "Physics",
        "MISC",
        0.88,
    ));
    let _s4 = doc.add_signal(Signal::new(
        0,
        Location::text(s4_start, s4_end),
        "1903",
        "DATE",
        0.92,
    ));
    let s5 = doc.add_signal(Signal::new(
        0,
        Location::text(s5_start, s5_end),
        "She",
        "PER",
        0.85,
    ));
    let s6 = doc.add_signal(Signal::new(
        0,
        Location::text(s6_start, s6_end),
        "Nobel Prize",
        "MISC",
        0.93,
    ));
    let s7 = doc.add_signal(Signal::new(
        0,
        Location::text(s7_start, s7_end),
        "Chemistry",
        "MISC",
        0.91,
    ));
    let _s8 = doc.add_signal(Signal::new(
        0,
        Location::text(s8_start, s8_end),
        "1911",
        "DATE",
        0.90,
    ));
    let s9 = doc.add_signal(Signal::new(
        0,
        Location::text(s9_start, s9_end),
        "Her",
        "PER",
        0.82,
    ));
    let _s10 = doc.add_signal(Signal::new(
        0,
        Location::text(s10_start, s10_end),
        "radioactivity",
        "MISC",
        0.89,
    ));
    let s11 = doc.add_signal(Signal::new(
        0,
        Location::text(s11_start, s11_end),
        "nuclear physics",
        "MISC",
        0.87,
    ));
    let s12 = doc.add_signal(Signal::new(
        0,
        Location::text(s12_start, s12_end),
        "Marie Curie",
        "PER",
        0.96,
    ));

    // Form tracks (Level 2: Within-document coreference)
    let mut curie_track = Track::new(0, "Marie Curie").with_type("PER");
    curie_track.add_signal(s1, 0);
    curie_track.add_signal(s5, 1);
    curie_track.add_signal(s9, 2);
    curie_track.add_signal(s12, 3);
    let curie_track_id = doc.add_track(curie_track);

    let mut nobel_physics_track = Track::new(1, "Nobel Prize in Physics").with_type("MISC");
    nobel_physics_track.add_signal(s2, 0);
    nobel_physics_track.add_signal(s3, 1);
    doc.add_track(nobel_physics_track);

    let mut nobel_chem_track = Track::new(2, "Nobel Prize in Chemistry").with_type("MISC");
    nobel_chem_track.add_signal(s6, 0);
    nobel_chem_track.add_signal(s7, 1);
    doc.add_track(nobel_chem_track);

    let mut nucl_track = Track::new(3, "Nuclear Physics").with_type("MISC");
    nucl_track.add_signal(s11, 0);
    doc.add_track(nucl_track);

    // Add identities (Level 3: Knowledge base linking)
    let curie_identity = Identity::from_kb(0, "Marie Curie", "wikidata", "Q7186")
        .with_type("PER")
        .with_description("Polish-French physicist and chemist");
    let curie_id = doc.add_identity(curie_identity);
    doc.link_track_to_identity(curie_track_id, curie_id);

    // Generate HTML
    let html = render_document_html(&doc);

    // Write to file
    let output_path = "grounded_analysis.html";
    std::fs::write(output_path, &html)?;
    println!("wrote {output_path} ({} bytes)", html.len());

    // Print statistics
    let stats = doc.stats();
    println!("{stats}");

    Ok(())
}
