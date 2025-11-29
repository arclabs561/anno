//! Full NER → Event → Discourse pipeline demonstration.
//!
//! Run: `cargo run --example discourse_pipeline`
//!
//! This example demonstrates the complete information extraction pipeline:
//! 1. NER: Extract named entities (people, organizations, locations)
//! 2. Event Extraction: Identify events and their triggers
//! 3. Discourse Resolution: Resolve abstract anaphors ("this", "that") to events

use anno::backends::PatternNER;
use anno::discourse::{DiscourseScope, EventExtractor};
use anno::eval::coref_resolver::{DiscourseAwareResolver, DiscourseCorefConfig};
use anno::{Entity, EntityType, Model};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║        NER → Event → Discourse Pipeline Demonstration            ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // Example documents with abstract anaphora
    let documents = vec![
        (
            "news_01",
            "Russia invaded Ukraine in February 2022. This caused a massive humanitarian crisis. \
             The United Nations condemned the action.",
        ),
        (
            "business_01",
            "Apple Inc. announced record quarterly earnings yesterday. This surprised analysts \
             who had predicted a downturn. Tim Cook attributed this to strong iPhone sales.",
        ),
        (
            "science_01",
            "The earthquake struck the coastal region at 3:42 AM. It measured 7.1 on the Richter scale. \
             This prompted immediate evacuation orders from local authorities.",
        ),
    ];

    for (doc_id, text) in documents {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Document: {}", doc_id);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("\nText: \"{}\"\n", text);

        process_document(text);
        println!();
    }

    // Demonstrate GLiNER integration note
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("NEURAL BACKEND (GLiNER)");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    let extractor = EventExtractor::new();
    if extractor.has_neural_backend() {
        println!("✓ GLiNER neural backend is available");
    } else {
        println!("ℹ Neural event extraction available with GLiNER:");
        println!("  cargo run --example discourse_pipeline --features candle");
        println!();
        println!("  Code to enable:");
        println!("    use anno::discourse::EventExtractorConfig;");
        println!("    let config = EventExtractorConfig::default()");
        println!("        .with_gliner(\"urchade/gliner_small-v2.1\");");
        println!("    let extractor = EventExtractor::with_config(config)?;");
    }
}

fn process_document(text: &str) {
    // =========================================================================
    // Step 1: Named Entity Recognition
    // =========================================================================
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Step 1: Named Entity Recognition                               │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    let ner = PatternNER::default();
    let entities = ner.extract_entities(text, None).unwrap_or_default();

    if entities.is_empty() {
        println!("  (No named entities found with pattern matcher)\n");
    } else {
        for entity in &entities {
            println!(
                "  • {:?}: \"{}\" [{}-{}]",
                entity.entity_type, entity.text, entity.start, entity.end
            );
        }
        println!();
    }

    // =========================================================================
    // Step 2: Event Extraction
    // =========================================================================
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Step 2: Event Extraction                                       │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    let event_extractor = EventExtractor::new();
    let events = event_extractor.extract_with_entities(text, &entities);

    if events.is_empty() {
        println!("  (No events detected)\n");
    } else {
        for event in &events {
            println!(
                "  • Event: \"{}\" ({})",
                event.trigger,
                event.trigger_type.as_deref().unwrap_or("unknown")
            );
            println!(
                "    Polarity: {:?}, Tense: {:?}",
                event.polarity, event.tense
            );
            if !event.arguments.is_empty() {
                println!("    Arguments:");
                for (role, value) in &event.arguments {
                    println!("      - {}: {}", role, value);
                }
            }
        }
        println!();
    }

    // =========================================================================
    // Step 3: Discourse Resolution
    // =========================================================================
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Step 3: Discourse & Abstract Anaphora Resolution               │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    let config = DiscourseCorefConfig::default();
    let resolver = DiscourseAwareResolver::new(config, text);

    // Find abstract anaphors in the text
    let abstract_anaphors = find_abstract_anaphors(text);
    let anaphor_count = abstract_anaphors.len();

    if abstract_anaphors.is_empty() {
        println!("  (No abstract anaphors found)\n");
    } else {
        for (anaphor_text, start, end) in abstract_anaphors {
            println!("  • Anaphor: \"{}\" at position {}", anaphor_text, start);

            // Create an entity for the anaphor
            let anaphor_entity = Entity::new(
                &anaphor_text,
                EntityType::Other("anaphor".to_string()),
                start,
                end,
                1.0,
            );

            // Try to resolve
            if let Some(referent) = resolver.find_discourse_antecedent(&anaphor_entity) {
                println!(
                    "    → Antecedent: \"{}\"",
                    referent.text.as_deref().unwrap_or("(span)")
                );
                println!("    → Type: {:?}", referent.referent_type);
                if let Some(event) = &referent.event {
                    println!("    → Event trigger: \"{}\"", event.trigger);
                }
            } else {
                println!("    → (Could not resolve)");
            }
        }
        println!();
    }

    // =========================================================================
    // Summary
    // =========================================================================
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Pipeline Summary                                               │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    let scope = DiscourseScope::analyze(text);
    println!("  Sentences: {}", scope.sentence_count());
    println!("  Named Entities: {}", entities.len());
    println!("  Events: {}", events.len());
    println!("  Abstract Anaphors: {}", anaphor_count);
}

/// Find abstract anaphors (This, That, It at sentence start) in text.
fn find_abstract_anaphors(text: &str) -> Vec<(String, usize, usize)> {
    let mut anaphors = Vec::new();
    let patterns = ["This ", "That ", "It "];

    for pattern in patterns {
        let mut search_start = 0;
        while let Some(pos) = text[search_start..].find(pattern) {
            let abs_pos = search_start + pos;

            // Check if it's at sentence start (after '. ' or at beginning)
            let is_sentence_start =
                abs_pos == 0 || (abs_pos >= 2 && &text[abs_pos - 2..abs_pos] == ". ");

            if is_sentence_start {
                let anaphor = pattern.trim();
                anaphors.push((anaphor.to_string(), abs_pos, abs_pos + anaphor.len()));
            }

            search_start = abs_pos + pattern.len();
        }
    }

    anaphors.sort_by_key(|(_, start, _)| *start);
    anaphors
}
