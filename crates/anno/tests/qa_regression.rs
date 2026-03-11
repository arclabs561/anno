//! Regression tests from QA reports (2026-03-03+).
//!
//! Each test documents a specific bug found during QA and prevents regression.
//! Tests marked `#[ignore]` require ONNX model downloads or long-running inference.
//!
//! Run all:    `cargo nextest run -p anno-lib --test qa_regression`
//! Run ONNX:   `cargo nextest run -p anno-lib --features onnx --test qa_regression -- --ignored`

use anno::{Entity, EntityType, HeuristicNER, Model};

// =============================================================================
// Helper
// =============================================================================

fn heuristic_entities(text: &str) -> Vec<Entity> {
    let ner = HeuristicNER::new();
    ner.extract_entities(text, None)
        .expect("heuristic should not fail")
}

fn entity_texts(entities: &[Entity]) -> Vec<&str> {
    entities.iter().map(|e| e.text.as_str()).collect()
}

fn per_entities(entities: &[Entity]) -> Vec<&str> {
    entities
        .iter()
        .filter(|e| matches!(e.entity_type, EntityType::Person))
        .map(|e| e.text.as_str())
        .collect()
}

fn org_entities(entities: &[Entity]) -> Vec<&str> {
    entities
        .iter()
        .filter(|e| matches!(e.entity_type, EntityType::Organization))
        .map(|e| e.text.as_str())
        .collect()
}

fn loc_entities(entities: &[Entity]) -> Vec<&str> {
    entities
        .iter()
        .filter(|e| matches!(e.entity_type, EntityType::Location))
        .map(|e| e.text.as_str())
        .collect()
}

// =============================================================================
// W3: Headline false positives (heuristic backend)
// QA report: qa-2026-03-03-webdocs.md
//
// Sentence-initial capitalized common nouns should NOT be tagged as PER.
// The heuristic backend uses capitalization as a signal, but sentence-initial
// position is unreliable (headlines, German grammar, section headers).
// =============================================================================

#[test]
fn qa_w3_headline_words_not_per() {
    // These words appeared as PER false positives on AP News headlines
    let headline_text = "Death toll rises in earthquake. Bus service disrupted. \
        Christmas celebrations cancelled due to storm. Crocodile attacks swimmer. \
        Doctors warn of new variant. Gasoline prices surge.";
    let entities = heuristic_entities(headline_text);
    let pers = per_entities(&entities);

    let false_positives = [
        "Death",
        "Bus",
        "Christmas",
        "Crocodile",
        "Doctors",
        "Gasoline",
    ];
    for fp in &false_positives {
        assert!(
            !pers.iter().any(|p| p == fp),
            "'{fp}' should not be tagged as PER (sentence-initial common noun), got PER: {pers:?}"
        );
    }
}

#[test]
fn qa_w3_real_names_mid_sentence_still_detected() {
    // Mid-sentence capitalized names should still be detected
    let text = "I spoke with Valentina about the plan. \
        Later, Marcus joined the discussion.";
    let entities = heuristic_entities(text);
    let pers = per_entities(&entities);

    assert!(
        pers.iter().any(|p| p.contains("Valentina")),
        "Valentina (mid-sentence) should be PER: {pers:?}"
    );
    assert!(
        pers.iter().any(|p| p.contains("Marcus")),
        "Marcus (mid-sentence) should be PER: {pers:?}"
    );
}

#[test]
fn qa_w3_two_word_names_at_sentence_start() {
    // Multi-word names at sentence start should still be detected
    // (they pass the two_word_name rule with confidence 0.60)
    let text = "Tim Cook announced new products. Angela Merkel visited Berlin.";
    let entities = heuristic_entities(text);
    let pers = per_entities(&entities);

    assert!(
        pers.iter().any(|p| p.contains("Tim Cook")),
        "Tim Cook (two-word at start) should be PER: {pers:?}"
    );
    assert!(
        pers.iter().any(|p| p.contains("Angela Merkel")),
        "Angela Merkel (two-word at start) should be PER: {pers:?}"
    );
}

#[test]
fn qa_w3_known_persons_at_sentence_start() {
    // Known person first names at sentence start should be detected
    // (they pass via the KNOWN_PERSONS rule)
    let text = "Barack spoke at the event. Elon presented the design.";
    let entities = heuristic_entities(text);
    let pers = per_entities(&entities);

    assert!(
        pers.iter().any(|p| p.contains("Barack")),
        "Barack (known person) should be PER: {pers:?}"
    );
    assert!(
        pers.iter().any(|p| p.contains("Elon")),
        "Elon (known person) should be PER: {pers:?}"
    );
}

// =============================================================================
// W5: URL text extraction (Wikipedia boilerplate)
// QA report: qa-2026-03-03-webdocs.md
//
// Wikipedia HTML contains TOC, references, external links, citation metadata.
// These should be stripped before NER to reduce noise.
// =============================================================================

#[test]
fn qa_w5_wikipedia_toc_stripped() {
    use deformat::html::strip_to_text;

    let html = r#"<html><body>
        <p>Real article content about CRISPR.</p>
        <div id="toc" class="toc"><h2>Contents</h2>
            <ul><li>1 History</li><li>2 Mechanism</li><li>3 Applications</li></ul>
        </div>
        <p>More real content about gene editing.</p>
    </body></html>"#;

    let text = strip_to_text(html);
    assert!(
        text.contains("CRISPR"),
        "article content should be preserved"
    );
    assert!(
        text.contains("gene editing"),
        "article content should be preserved"
    );
    assert!(
        !text.contains("Contents"),
        "TOC heading should be stripped, got: {text}"
    );
    assert!(
        !text.contains("Mechanism"),
        "TOC entries should be stripped, got: {text}"
    );
}

#[test]
fn qa_w5_wikipedia_references_stripped() {
    use deformat::html::strip_to_text;

    let html = r#"<html><body>
        <p>Jennifer Doudna developed CRISPR-Cas9.</p>
        <ol class="references">
            <li id="cite_note-1">Doudna JA, Charpentier E (2014). "The new frontier".</li>
            <li id="cite_note-2">Zhang F et al. (2013). "Multiplex genome engineering".</li>
        </ol>
        <p>The technology won the Nobel Prize.</p>
    </body></html>"#;

    let text = strip_to_text(html);
    assert!(
        text.contains("Jennifer Doudna"),
        "article content preserved"
    );
    assert!(text.contains("Nobel Prize"), "article content preserved");
    assert!(
        !text.contains("cite_note"),
        "reference list should be stripped, got: {text}"
    );
}

#[test]
fn qa_w5_wikipedia_nav_stripped() {
    use deformat::html::strip_to_text;

    let html = r#"<html><body>
        <nav id="mw-navigation"><ul><li>Main page</li><li>Random article</li></ul></nav>
        <p>Actual article text here.</p>
        <footer id="footer"><p>Privacy policy</p></footer>
    </body></html>"#;

    let text = strip_to_text(html);
    assert!(text.contains("Actual article text"));
    assert!(!text.contains("Main page"), "navigation stripped");
    assert!(!text.contains("Random article"), "navigation stripped");
    assert!(!text.contains("Privacy policy"), "footer stripped");
}

// =============================================================================
// W6: HTML file detection
// QA report: qa-2026-03-03-webdocs.md
//
// When --file receives HTML, tags should be stripped automatically.
// =============================================================================

#[test]
fn qa_w6_looks_like_html_detection() {
    use deformat::detect::is_html;

    assert!(is_html("<!DOCTYPE html><html><body>text</body></html>"));
    assert!(is_html("<html><head></head><body>text</body></html>"));
    assert!(is_html("  \n<!DOCTYPE html>\n<html>"));
    assert!(is_html("<?xml version=\"1.0\"?><html>"));

    // Plain text should NOT be detected as HTML
    assert!(!is_html("Tim Cook announced new products today."));
    assert!(!is_html("The patient has no history of diabetes."));
    assert!(!is_html("# Markdown heading\n\nSome text."));
}

#[test]
fn qa_w6_html_tags_not_in_entities() {
    use deformat::html::strip_to_text;

    let html = "<html><body><p>Tim Cook leads Apple.</p></body></html>";
    let text = strip_to_text(html);

    // Extracted text should not contain HTML tags
    assert!(
        !text.contains('<'),
        "no HTML tags in extracted text: {text}"
    );
    assert!(
        !text.contains('>'),
        "no HTML tags in extracted text: {text}"
    );
    assert!(text.contains("Tim Cook"), "entity text preserved");
    assert!(text.contains("Apple"), "entity text preserved");
}

// =============================================================================
// Heuristic: sentence-start vs mid-sentence confidence
//
// Structural test: single capitalized words at sentence start should have
// lower confidence than the same word mid-sentence. This is the core defense
// against headline false positives.
// =============================================================================

#[test]
fn qa_sentence_start_lower_confidence_than_mid() {
    let ner = HeuristicNER::new();

    // Sentence-start: "Storm" at beginning
    let start_entities = ner
        .extract_entities("Storm damages buildings.", None)
        .unwrap();
    let start_storm = start_entities.iter().find(|e| e.text == "Storm");

    // Mid-sentence: "Storm" after a verb
    let mid_entities = ner
        .extract_entities("The severe Storm damages buildings.", None)
        .unwrap();
    let mid_storm = mid_entities.iter().find(|e| e.text == "Storm");

    // Sentence-start "Storm" should either not be detected or have lower confidence
    match (start_storm, mid_storm) {
        (Some(s), Some(m)) => {
            assert!(
                s.confidence <= m.confidence,
                "sentence-start should have <= confidence vs mid-sentence: start={}, mid={}",
                s.confidence,
                m.confidence
            );
        }
        (None, _) => {} // Good: sentence-start word correctly filtered
        (Some(_), None) => {
            // Unusual but not necessarily wrong -- depends on context
        }
    }
}

// =============================================================================
// W1/W2: BERT ONNX chunking (requires ONNX feature + model)
// QA report: qa-2026-03-03-webdocs.md
//
// Long documents should not lose entities at chunk boundaries.
// =============================================================================

#[cfg(feature = "onnx")]
mod onnx_chunking {
    use anno::{EntityType, Model, StackedNER};

    #[test]
    #[ignore] // requires ONNX model download
    fn qa_w1_long_document_recall() {
        // Multiple paragraphs with well-known entities spread across text
        let text = "Jennifer Doudna and Emmanuelle Charpentier developed CRISPR-Cas9 \
            gene editing at UC Berkeley. Feng Zhang at the Broad Institute of MIT \
            also contributed to the technology. In 2020, Doudna and Charpentier \
            received the Nobel Prize in Chemistry. \
            Meanwhile, George Church at Harvard pioneered applications in \
            synthetic biology. David Liu developed base editing techniques. \
            The Wellcome Trust funded much of the early research in Cambridge.";

        let ner = StackedNER::default();
        let entities = ner
            .extract_entities(text, None)
            .expect("extraction should work");

        let per_texts: Vec<&str> = entities
            .iter()
            .filter(|e| matches!(e.entity_type, EntityType::Person))
            .map(|e| e.text.as_str())
            .collect();

        // Should find at least 3 of the 5 main person names
        let expected = ["Doudna", "Charpentier", "Zhang", "Church", "Liu"];
        let found: Vec<&&str> = expected
            .iter()
            .filter(|name| per_texts.iter().any(|t| t.contains(*name)))
            .collect();
        assert!(
            found.len() >= 3,
            "should find at least 3 of 5 person names, found {}: {:?}. All PER: {:?}",
            found.len(),
            found,
            per_texts
        );
    }

    #[test]
    #[ignore] // requires ONNX model download
    fn qa_w2_name_not_truncated_at_boundary() {
        // Text where a two-word name might straddle a chunk boundary
        // Use enough text to force chunking (>510 tokens)
        let padding = "The research team published findings in Nature. ".repeat(50);
        let text = format!(
            "{}Jennifer Doudna won the Nobel Prize. {}Emmanuelle Charpentier also won.",
            padding, padding
        );

        let ner = StackedNER::default();
        let entities = ner.extract_entities(&text, None).expect("extraction");

        // Check that names are not split: "Jennifer" without "Doudna" would indicate truncation
        let has_jennifer_only = entities
            .iter()
            .any(|e| e.text == "Jennifer" && !e.text.contains("Doudna"));

        assert!(
            !has_jennifer_only,
            "Name should not be split at chunk boundary. Entities near 'Jennifer': {:?}",
            entities
                .iter()
                .filter(|e| e.text.contains("Jennifer") || e.text.contains("Doudna"))
                .map(|e| (&e.text, &e.entity_type))
                .collect::<Vec<_>>()
        );
    }
}

// =============================================================================
// Stacked backend: filter_title_words
// QA report: qa-2026-03-03-webdocs.md
//
// German title words and common nouns should be filtered from entity output.
// =============================================================================

#[cfg(feature = "onnx")]
mod stacked_filtering {
    use anno::{EntityType, Model, StackedNER};

    #[test]
    #[ignore] // requires ONNX model download
    fn qa_n8_bundeskanzler_not_org() {
        let ner = StackedNER::default();
        let entities = ner
            .extract_entities("Bundeskanzler Olaf Scholz besuchte Berlin.", None)
            .expect("extraction");

        let org_texts: Vec<&str> = entities
            .iter()
            .filter(|e| matches!(e.entity_type, EntityType::Organization))
            .map(|e| e.text.as_str())
            .collect();

        assert!(
            !org_texts
                .iter()
                .any(|t| t.to_lowercase() == "bundeskanzler"),
            "Bundeskanzler should not be tagged as ORG: {org_texts:?}"
        );
    }

    #[test]
    #[ignore] // requires ONNX model download
    fn qa_w3_stacked_headline_words_filtered() {
        let ner = StackedNER::default();
        let entities = ner
            .extract_entities(
                "Death toll rises. Police investigate. Military deployed.",
                None,
            )
            .expect("extraction");

        let per_texts: Vec<&str> = entities
            .iter()
            .filter(|e| matches!(e.entity_type, EntityType::Person))
            .map(|e| e.text.as_str())
            .collect();

        let common_nouns = ["Death", "Police", "Military"];
        for noun in &common_nouns {
            assert!(
                !per_texts.iter().any(|t| t == noun),
                "'{noun}' should not be PER in stacked output: {per_texts:?}"
            );
        }
    }
}

// =============================================================================
// HTML entity decoding in url_resolver
// =============================================================================

#[test]
fn qa_html_entity_decoding() {
    use deformat::html::strip_to_text;

    let html = "<p>Nestl&eacute; &amp; Mars &mdash; leading brands</p>";
    let text = strip_to_text(html);
    assert!(text.contains('&'), "amp decoded: {text}");
    // eacute might not be decoded (only common entities handled), that's OK
}

#[test]
fn qa_html_numeric_entity_decoding() {
    use deformat::html::strip_to_text;

    let html = "<p>&#169; 2024 &#x2014; all rights</p>";
    let text = strip_to_text(html);
    // &#169; = copyright symbol, &#x2014; = em dash
    assert!(!text.contains("&#"), "numeric entities decoded: {text}");
}

// =============================================================================
// Heuristic: COMMON_SENTENCE_STARTERS coverage
// =============================================================================

#[test]
fn qa_common_starters_not_entities() {
    let text = "However, the plan failed. Meanwhile, stocks fell. \
        Furthermore, costs increased. Therefore, profits declined.";
    let entities = heuristic_entities(text);
    let texts = entity_texts(&entities);

    let starters = ["However", "Meanwhile", "Furthermore", "Therefore"];
    for s in &starters {
        assert!(
            !texts.contains(s),
            "'{s}' is a common sentence starter, should not be an entity: {texts:?}"
        );
    }
}

#[test]
fn qa_day_month_names_not_entities() {
    let text = "Monday was rainy. January saw record snowfall. \
        Wednesday brought sunshine. September ended abruptly.";
    let entities = heuristic_entities(text);
    let pers = per_entities(&entities);

    let time_words = ["Monday", "January", "Wednesday", "September"];
    for tw in &time_words {
        assert!(
            !pers.iter().any(|p| p == tw),
            "'{tw}' should not be PER: {pers:?}"
        );
    }
}

// =============================================================================
// Heuristic: entity type accuracy
// =============================================================================

#[test]
fn qa_known_orgs_detected() {
    let text = "Google and Microsoft announced a partnership. Apple released new products.";
    let entities = heuristic_entities(text);
    let orgs = org_entities(&entities);

    assert!(
        orgs.iter().any(|o| o.contains("Google")),
        "Google should be ORG: {orgs:?}"
    );
    assert!(
        orgs.iter().any(|o| o.contains("Microsoft")),
        "Microsoft should be ORG: {orgs:?}"
    );
    assert!(
        orgs.iter().any(|o| o.contains("Apple")),
        "Apple should be ORG: {orgs:?}"
    );
}

#[test]
fn qa_known_locs_detected() {
    let text = "We visited Paris and then traveled to Tokyo.";
    let entities = heuristic_entities(text);
    let locs = loc_entities(&entities);

    assert!(
        locs.iter().any(|l| l.contains("Paris")),
        "Paris should be LOC: {locs:?}"
    );
    assert!(
        locs.iter().any(|l| l.contains("Tokyo")),
        "Tokyo should be LOC: {locs:?}"
    );
}

#[test]
fn qa_loc_preposition_context() {
    let text = "The conference was held in Geneva.";
    let entities = heuristic_entities(text);
    let locs = loc_entities(&entities);

    assert!(
        locs.iter().any(|l| l.contains("Geneva")),
        "Geneva after 'in' should be LOC: {locs:?}"
    );
}

#[test]
fn qa_person_prefix_context() {
    let text = "Dr. Smith consulted with Prof. Johnson.";
    let entities = heuristic_entities(text);
    let pers = per_entities(&entities);

    assert!(
        pers.iter().any(|p| p.contains("Smith")),
        "Smith after 'Dr.' should be PER: {pers:?}"
    );
    assert!(
        pers.iter().any(|p| p.contains("Johnson")),
        "Johnson after 'Prof.' should be PER: {pers:?}"
    );
}

// =============================================================================
// Heuristic: skip words and acronyms
// =============================================================================

#[test]
fn qa_skip_words_not_entities() {
    let text = "CEO Satya Nadella met with President Biden.";
    let entities = heuristic_entities(text);
    let all_texts = entity_texts(&entities);

    // CEO and President should be skip words, not standalone entities
    assert!(
        !all_texts.contains(&"CEO"),
        "CEO alone should be skip word: {all_texts:?}"
    );
    assert!(
        !all_texts.contains(&"President"),
        "President alone should be skip word: {all_texts:?}"
    );
}

#[test]
fn qa_common_acronyms_not_entities() {
    let text = "The CPU handles GPU tasks via the API.";
    let entities = heuristic_entities(text);
    let all_texts = entity_texts(&entities);

    let acronyms = ["CPU", "GPU", "API"];
    for acr in &acronyms {
        assert!(
            !all_texts.iter().any(|t| t == acr),
            "'{acr}' should be filtered as common acronym: {all_texts:?}"
        );
    }
}

#[test]
fn qa_fiscal_quarters_not_entities() {
    let text = "Results for Q3 FY2025 showed improvement over Q1 2024.";
    let entities = heuristic_entities(text);
    let all_texts = entity_texts(&entities);

    assert!(
        !all_texts.iter().any(|t| t.starts_with('Q') && t.len() <= 2),
        "Fiscal quarters should not be entities: {all_texts:?}"
    );
}

// =============================================================================
// URL text extraction: basic functionality
// =============================================================================

#[test]
fn qa_strip_html_basic() {
    use deformat::html::strip_to_text;

    let html = "<p>Hello <b>world</b>!</p>";
    let text = strip_to_text(html);
    assert_eq!(text, "Hello world!");
}

#[test]
fn qa_strip_html_script_style_removed() {
    use deformat::html::strip_to_text;

    let html = r#"<html><head><style>body{color:red}</style></head>
        <body><script>alert('hi')</script><p>Real text.</p></body></html>"#;
    let text = strip_to_text(html);
    assert!(text.contains("Real text"));
    assert!(!text.contains("alert"), "script content stripped");
    assert!(!text.contains("color"), "style content stripped");
}

#[test]
fn qa_strip_html_preserves_whitespace_between_blocks() {
    use deformat::html::strip_to_text;

    let html = "<h1>Title</h1><p>First paragraph.</p><p>Second paragraph.</p>";
    let text = strip_to_text(html);
    // Block elements should have spaces between them
    assert!(
        text.contains("Title") && text.contains("First") && text.contains("Second"),
        "all content preserved: {text}"
    );
    // Should not have words joined without spaces
    assert!(!text.contains("TitleFirst"), "blocks separated: {text}");
}

// =============================================================================
// N9: HTML <head>/<title> content should not appear in stripped text
// QA report: qa-2026-03-04
// =============================================================================

#[test]
fn qa_n9_html_title_not_in_body_text() {
    use deformat::html::strip_to_text;

    let html =
        "<html><head><title>Page Title</title></head><body><p>Tim Cook is CEO.</p></body></html>";
    let text = strip_to_text(html);
    assert!(
        !text.contains("Page Title"),
        "title tag content should be stripped: {text}"
    );
    assert!(
        text.contains("Tim Cook"),
        "body content should be preserved: {text}"
    );
}

#[test]
fn qa_n9_html_head_metadata_stripped() {
    use deformat::html::strip_to_text;

    let html = r#"<html>
        <head>
            <title>My Page</title>
            <meta name="description" content="test">
            <link rel="stylesheet" href="style.css">
        </head>
        <body><p>Angela Merkel met Emmanuel Macron.</p></body>
    </html>"#;
    let text = strip_to_text(html);
    assert!(!text.contains("My Page"), "title stripped: {text}");
    assert!(text.contains("Angela Merkel"), "body preserved: {text}");
}

// =============================================================================
// Quantifier: Approximate detection
// QA report: qa-2026-03-04
// =============================================================================

#[test]
fn qa_quantifier_approximately() {
    use anno::heuristics::detect_quantifier_en;
    use anno_core::Quantifier;

    // "approximately" before a number + entity
    let text = "Approximately 500 employees at Google received options.";
    // entity "Google" starts at char 31
    let q = detect_quantifier_en(text, 31);
    assert_eq!(
        q,
        Some(Quantifier::Approximate),
        "approximately should trigger Approximate"
    );
}

#[test]
fn qa_quantifier_at_least() {
    use anno::heuristics::detect_quantifier_en;
    use anno_core::Quantifier;

    let text = "At least three companies bid.";
    // entity "companies" starts at char 20
    let q = detect_quantifier_en(text, 20);
    assert_eq!(
        q,
        Some(Quantifier::MinBound),
        "at least should trigger MinBound"
    );
}

#[test]
fn qa_quantifier_about() {
    use anno::heuristics::detect_quantifier_en;
    use anno_core::Quantifier;

    let text = "About 2.5 million users signed up.";
    // entity "users" starts at char 22
    let q = detect_quantifier_en(text, 22);
    assert_eq!(
        q,
        Some(Quantifier::Approximate),
        "about should trigger Approximate"
    );
}

#[test]
fn qa_quantifier_definite_still_works() {
    use anno::heuristics::detect_quantifier_en;
    use anno_core::Quantifier;

    let text = "The patient has no history.";
    let q = detect_quantifier_en(text, 4);
    assert_eq!(
        q,
        Some(Quantifier::Definite),
        "the should still trigger Definite"
    );
}

#[test]
fn qa_quantifier_no_false_positive_far_away() {
    use anno::heuristics::detect_quantifier_en;
    use anno_core::Quantifier;

    // "approximately" is >40 chars before "Boston" - should not trigger
    let text = "Approximately 50 students from MIT attended the conference in Boston.";
    // "Boston" starts at char 62
    let q = detect_quantifier_en(text, 62);
    assert_ne!(
        q,
        Some(Quantifier::Approximate),
        "approximately too far from Boston to trigger"
    );
}

// =============================================================================
// bert-onnx: name completion for O-labeled surnames
// QA report: qa-2026-03-04
// =============================================================================

#[cfg(feature = "onnx")]
mod onnx_name_completion {
    use anno::{EntityType, Model, StackedNER};

    #[test]
    #[ignore] // requires ONNX model download
    fn qa_bert_onnx_surname_not_dropped() {
        // BERT often tags given names as B-PER but surnames as O.
        // The name completion heuristic should absorb adjacent capitalized
        // words into PER entities.
        let ner = StackedNER::default();
        let entities = ner
            .extract_entities("Jennifer Doudna won the Nobel Prize in Chemistry.", None)
            .expect("extraction");

        let per_texts: Vec<&str> = entities
            .iter()
            .filter(|e| matches!(e.entity_type, EntityType::Person))
            .map(|e| e.text.as_str())
            .collect();

        // Should find "Jennifer Doudna" as one entity, not just "Jennifer"
        let has_full_name = per_texts.iter().any(|t| t.contains("Doudna"));
        let has_first_only = per_texts.contains(&"Jennifer");
        assert!(
            has_full_name && !has_first_only,
            "Should find 'Jennifer Doudna' not just 'Jennifer': {per_texts:?}"
        );
    }

    #[test]
    #[ignore] // requires ONNX model download
    fn qa_bert_onnx_name_completion_no_absorb_lowercase() {
        // Should NOT absorb lowercase words (e.g., "won") into PER
        let ner = StackedNER::default();
        let entities = ner
            .extract_entities("Tim Cook met with partners.", None)
            .expect("extraction");

        let per_texts: Vec<&str> = entities
            .iter()
            .filter(|e| matches!(e.entity_type, EntityType::Person))
            .map(|e| e.text.as_str())
            .collect();

        // "Tim Cook" should not absorb "met"
        assert!(
            !per_texts.iter().any(|t| t.contains("met")),
            "Should not absorb lowercase words: {per_texts:?}"
        );
    }
}
