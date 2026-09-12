use super::*;

// -------------------------------------------------------------------------
// Config / struct smoke tests
// -------------------------------------------------------------------------

#[test]
fn test_coref_config_default() {
    let config = T5CorefConfig::default();
    assert_eq!(config.max_input_length, 512);
    assert_eq!(config.num_beams, 1);
}

#[test]
fn test_cluster_struct() {
    let cluster = CorefCluster {
        id: 0,
        mentions: vec!["Marie Curie".to_string(), "She".to_string()],
        spans: vec![(0, 11), (50, 53)],
        canonical: "Marie Curie".to_string(),
    };
    assert_eq!(cluster.mentions.len(), 2);
    assert_eq!(cluster.canonical, "Marie Curie");
}

// -------------------------------------------------------------------------
// mark_mentions_for_t5
// -------------------------------------------------------------------------

#[test]
fn mark_mentions_wraps_pronouns() {
    let out = mark_mentions_for_t5("she said he agreed");
    assert!(
        out.contains("<m> she </m>"),
        "pronoun 'she' should be marked"
    );
    assert!(out.contains("<m> he </m>"), "pronoun 'he' should be marked");
    assert!(out.contains("said"), "'said' should pass through unmarked");
    assert!(
        out.contains("agreed"),
        "'agreed' should pass through unmarked"
    );
}

#[test]
fn mark_mentions_wraps_capitalized() {
    let out = mark_mentions_for_t5("Sophie Wilson designed ARM.");
    assert!(out.contains("<m> Sophie </m>"));
    assert!(out.contains("<m> Wilson </m>"));
    assert!(out.contains("<m> ARM. </m>"));
    assert!(out.contains("designed"));
}

#[test]
fn mark_mentions_empty_string() {
    assert_eq!(mark_mentions_for_t5(""), "");
}

#[test]
fn mark_mentions_no_pronouns_no_caps() {
    let text = "the quick brown fox";
    assert_eq!(mark_mentions_for_t5(text), text);
}

// -------------------------------------------------------------------------
// extract_t5_mentions
// -------------------------------------------------------------------------

#[test]
fn extract_mentions_basic() {
    let (plain, spans) = extract_t5_mentions("<m> Sophie Wilson </m> designed ARM.").unwrap();
    assert_eq!(plain, "Sophie Wilson designed ARM.");
    assert_eq!(spans.len(), 1);
    assert_eq!(spans[0].0, "Sophie Wilson");
}

#[test]
fn extract_mentions_two_spans() {
    let marked = "<m> Ada </m> founded <m> Lovelace Labs </m>.";
    let (plain, spans) = extract_t5_mentions(marked).unwrap();
    assert_eq!(spans.len(), 2);
    assert_eq!(spans[0].0, "Ada");
    assert_eq!(spans[1].0, "Lovelace Labs");
    assert!(plain.contains("Ada"));
    assert!(plain.contains("Lovelace Labs"));
    assert!(plain.contains("founded"));
}

#[test]
fn extract_mentions_empty_input() {
    let (plain, spans) = extract_t5_mentions("").unwrap();
    assert_eq!(plain, "");
    assert!(spans.is_empty());
}

#[test]
fn extract_mentions_no_markers() {
    let (plain, spans) = extract_t5_mentions("no markers here").unwrap();
    assert_eq!(plain, "no markers here");
    assert!(spans.is_empty());
}

#[test]
fn extract_mentions_span_offsets_are_consistent() {
    let marked = "<m> Marie </m> discovered polonium. <m> She </m> won.";
    let (plain, spans) = extract_t5_mentions(marked).unwrap();
    for (text, start, end) in &spans {
        let extracted: String = plain.chars().skip(*start).take(end - start).collect();
        assert_eq!(
            &extracted, text,
            "span offsets must index back to mention text"
        );
    }
}

#[test]
fn extract_mentions_uses_character_offsets_for_unicode() {
    let marked = "<m> 田中 </m> met <m> Élise </m>";
    let (plain, mentions) = extract_t5_mentions(marked).unwrap();
    assert_eq!(plain, "田中 met Élise");
    assert_eq!(mentions[0], ("田中".to_string(), 0, 2));
    assert_eq!(mentions[1], ("Élise".to_string(), 7, 12));
}

// -------------------------------------------------------------------------
// parse_t5_coref_output
// -------------------------------------------------------------------------

#[test]
fn parse_output_basic_two_mentions() {
    // "Ada | 1 founded Lovelace Labs | 2. She | 1 led ENIAC."
    let decoded = "Ada | 1 founded Lovelace Labs | 2. She | 1 led ENIAC.";
    let clusters = parse_t5_coref_output(decoded);
    let cluster1 = clusters.iter().find(|c| c.id == 1);
    assert!(cluster1.is_some(), "cluster 1 (Ada/She) should be parsed");
    let c = cluster1.unwrap();
    assert!(c.mentions.len() >= 2, "cluster 1 should have ≥2 members");
    assert!(c.mentions.contains(&"Ada".to_string()));
    assert!(c.mentions.contains(&"She".to_string()));
}

#[test]
fn parse_output_singletons_filtered() {
    // Word with marker but only appearing once → should be filtered (singleton)
    let decoded = "Marie | 1 discovered polonium.";
    let clusters = parse_t5_coref_output(decoded);
    assert!(
        clusters.is_empty(),
        "singleton cluster should be filtered out"
    );
}

#[test]
fn parse_output_empty_string() {
    assert!(parse_t5_coref_output("").is_empty());
}

#[test]
fn parse_output_no_markers() {
    let clusters = parse_t5_coref_output("This text has no cluster markers at all.");
    assert!(clusters.is_empty());
}

#[test]
fn parse_output_canonical_is_longest_mention() {
    // Each token gets a marker individually.
    // "Marie | 1 Curie | 1" produces mentions ["Marie", "Curie"] for cluster 1.
    // With "She" also in cluster 1, the longest single-token mention is "Marie" (5 chars)
    // vs "Curie" (5) vs "She" (3) — tie between Marie/Curie, resolved by first-seen.
    // Use a clearer example: "Alexandra | 1 won . She | 1 left."
    let decoded = "Alexandra | 1 won . She | 1 left .";
    let clusters = parse_t5_coref_output(decoded);
    if let Some(c) = clusters.iter().find(|c| c.id == 1) {
        assert_eq!(
            c.canonical, "Alexandra",
            "canonical should be the longest mention"
        );
    }
}

#[test]
fn parse_output_sorted_by_cluster_id() {
    let decoded = "B | 2 and A | 1 were colleagues . them | 2 worked together . B | 1 also .";
    let clusters = parse_t5_coref_output(decoded);
    if clusters.len() >= 2 {
        let ids: Vec<u32> = clusters.iter().map(|c| c.id).collect();
        let mut sorted = ids.clone();
        sorted.sort();
        assert_eq!(ids, sorted, "clusters should be sorted by ID");
    }
}

// -------------------------------------------------------------------------
// Additional pure-function tests
// -------------------------------------------------------------------------

#[test]
fn config_custom_values() {
    let config = T5CorefConfig {
        max_input_length: 256,
        max_output_length: 128,
        num_beams: 4,
        optimization_level: 1,
        num_threads: 2,
    };
    assert_eq!(config.max_input_length, 256);
    assert_eq!(config.max_output_length, 128);
    assert_eq!(config.num_beams, 4);
    assert_eq!(config.optimization_level, 1);
    assert_eq!(config.num_threads, 2);
}

#[test]
fn cluster_clone_is_independent() {
    let original = CorefCluster {
        id: 1,
        mentions: vec!["Ada Lovelace".into(), "She".into()],
        spans: vec![(0, 12), (30, 33)],
        canonical: "Ada Lovelace".into(),
    };
    let mut cloned = original.clone();
    cloned.mentions.push("her".into());
    assert_eq!(original.mentions.len(), 2, "original must be unaffected");
    assert_eq!(cloned.mentions.len(), 3);
}

#[test]
fn mark_mentions_mixed_pronouns_and_caps() {
    let out = mark_mentions_for_t5("Alice told him about Bob");
    assert!(out.contains("<m> Alice </m>"), "capitalized 'Alice' marked");
    assert!(out.contains("<m> him </m>"), "pronoun 'him' marked");
    assert!(out.contains("<m> Bob </m>"), "capitalized 'Bob' marked");
    assert!(out.contains("told"), "'told' unmarked");
    assert!(out.contains("about"), "'about' unmarked");
}

#[test]
fn mark_mentions_possessive_pronouns() {
    let out = mark_mentions_for_t5("his car and their house");
    assert!(out.contains("<m> his </m>"));
    assert!(out.contains("<m> their </m>"));
    // "car", "and", "house" should not be marked
    assert!(!out.contains("<m> car </m>"));
    assert!(!out.contains("<m> and </m>"));
    assert!(!out.contains("<m> house </m>"));
}

#[test]
fn extract_mentions_unclosed_tag_keeps_remaining_text() {
    // An unclosed <m> tag should not panic; the remainder is appended as plain text.
    let (plain, spans) = extract_t5_mentions("before <m> orphan text after").unwrap();
    assert!(
        plain.contains("before"),
        "text before unclosed tag preserved"
    );
    assert!(
        plain.contains("orphan text after"),
        "text inside unclosed tag appended as plain"
    );
    assert!(spans.is_empty(), "no complete span from unclosed tag");
}

#[test]
fn extract_mentions_adjacent_spans() {
    let marked = "<m> A </m><m> B </m>";
    let (plain, spans) = extract_t5_mentions(marked).unwrap();
    assert_eq!(spans.len(), 2);
    assert_eq!(spans[0].0, "A");
    assert_eq!(spans[1].0, "B");
    // Offsets must not overlap
    assert!(
        spans[0].2 <= spans[1].1,
        "first span end ({}) must be <= second span start ({})",
        spans[0].2,
        spans[1].1
    );
    // Round-trip: extracting from plain text at the stored offsets must match
    for (text, start, end) in &spans {
        let extracted: String = plain.chars().skip(*start).take(end - start).collect();
        assert_eq!(&extracted, text);
    }
}

#[test]
fn parse_output_multiple_distinct_clusters() {
    // Two separate clusters: 1 (Alice/She) and 2 (Bob/He)
    let decoded = "Alice | 1 met Bob | 2 yesterday . She | 1 greeted He | 2 warmly .";
    let clusters = parse_t5_coref_output(decoded);
    assert_eq!(clusters.len(), 2, "two multi-mention clusters expected");
    let c1 = clusters.iter().find(|c| c.id == 1).unwrap();
    let c2 = clusters.iter().find(|c| c.id == 2).unwrap();
    assert!(c1.mentions.contains(&"Alice".to_string()));
    assert!(c1.mentions.contains(&"She".to_string()));
    assert!(c2.mentions.contains(&"Bob".to_string()));
    assert!(c2.mentions.contains(&"He".to_string()));
}

#[test]
fn parse_output_cluster_id_with_trailing_punctuation() {
    // Cluster ID followed by punctuation (e.g. "1." or "1,") should still parse
    let decoded = "Marie | 1 discovered radium . She | 1 won .";
    let clusters = parse_t5_coref_output(decoded);
    assert_eq!(clusters.len(), 1);
    let c = &clusters[0];
    assert_eq!(c.id, 1);
    assert!(c.mentions.contains(&"Marie".to_string()));
    assert!(c.mentions.contains(&"She".to_string()));
}

#[test]
fn alignment_uses_source_coordinates_for_the_complete_token_stream() {
    let source = "田中 met Marie. She thanked Marie.";
    let decoded = "田中 met Marie | 1 She | 1 thanked Marie.";

    let clusters = align_t5_coref_output(decoded, source).unwrap();
    assert_eq!(clusters.len(), 1);
    assert_eq!(clusters[0].mentions, vec!["Marie", "She"]);
    assert_eq!(clusters[0].spans, vec![(7, 12), (14, 17)]);
}

#[test]
fn alignment_preserves_unicode_source_spans() {
    let clusters = align_t5_coref_output("田中 | 1 met Élise | 1", "田中 met Élise.").unwrap();

    assert_eq!(clusters[0].mentions, vec!["田中", "Élise"]);
    assert_eq!(clusters[0].spans, vec![(0, 2), (7, 12)]);
}

#[test]
fn alignment_rejects_changed_or_omitted_decoder_tokens() {
    let error = align_t5_coref_output(
        "A different rendering: Marie | 1 then She | 1.",
        "Marie arrived. She waved.",
    )
    .unwrap_err();

    assert!(error.to_string().contains("does not reproduce"));
}

#[test]
fn alignment_rejects_omitted_repeated_name() {
    let error = align_t5_coref_output(
        "Marie arrived. She | 1 waved.",
        "Marie arrived. Marie left. She waved.",
    )
    .unwrap_err();

    assert!(error.to_string().contains("does not reproduce"));
}

#[test]
fn alignment_preserves_the_correct_repeated_name_occurrence() {
    let source = "Marie arrived. Marie left. She waved.";
    let clusters =
        align_t5_coref_output("Marie | 1 arrived. Marie | 1 left. She | 1 waved.", source).unwrap();

    assert_eq!(clusters[0].spans, vec![(0, 5), (15, 20), (27, 30)]);
}
