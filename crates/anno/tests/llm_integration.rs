//! Live integration tests for the UniversalNER LLM backend.
//!
//! These tests make real API calls via OpenRouter using `OPENROUTER_API_KEY`
//! from `dev/.env`. They validate entity extraction quality, caching,
//! prompt strategies, domain context, and multi-model support.
//!
//! Run with:
//!   cargo test -p anno-lib --features llm --test llm_integration -- --ignored --nocapture

#![cfg(feature = "llm")]

use anno::backends::llm_client::LlmConfig;
use anno::backends::universal_ner::{PromptStrategy, UniversalNER};
use anno::{Entity, EntityType, Model};

// =============================================================================
// Helpers
// =============================================================================

fn ensure_api_key() -> bool {
    anno::env::load_dotenv();
    anno::env::has_llm_api_key()
}

fn assert_entity_invariants(entities: &[Entity], text: &str) {
    let char_count = text.chars().count();
    for e in entities {
        assert!(
            e.start < e.end,
            "start < end: {:?} ({},{})",
            e.text,
            e.start,
            e.end
        );
        assert!(
            e.end <= char_count,
            "end <= char_count: {:?} end={} chars={}",
            e.text,
            e.end,
            char_count
        );
        assert!(
            (0.0..=1.0).contains(&e.confidence),
            "confidence in [0,1]: {}",
            e.confidence
        );
        assert!(!e.text.is_empty(), "entity text must be non-empty");

        // Extracted text should match entity text
        let extracted: String = text.chars().skip(e.start).take(e.end - e.start).collect();
        assert_eq!(
            extracted, e.text,
            "span [{},{}) = {:?} vs entity {:?}",
            e.start, e.end, extracted, e.text
        );
    }
}

// =============================================================================
// Basic extraction (default model: Gemini 2.5 Flash via OpenRouter)
// =============================================================================

#[test]
#[ignore]
fn llm_basic_extraction_default_model() {
    if !ensure_api_key() {
        eprintln!("SKIP: no API key available");
        return;
    }

    let model = UniversalNER::new().expect("create UniversalNER");
    assert!(
        model.is_available(),
        "model should be available with API key"
    );

    let text = "Marie Curie discovered radium in Paris in 1898.";
    let entities = model
        .extract_entities(text, None)
        .expect("extraction should succeed");

    eprintln!(
        "Default model (GPT-5 Nano) entities: {:?}",
        entities
            .iter()
            .map(|e| (&e.text, &e.entity_type))
            .collect::<Vec<_>>()
    );

    assert!(!entities.is_empty(), "should find at least one entity");
    assert_entity_invariants(&entities, text);

    // Should find Marie Curie as a person
    let has_curie = entities
        .iter()
        .any(|e| e.text.contains("Curie") && matches!(e.entity_type, EntityType::Person));
    assert!(
        has_curie,
        "should find Marie Curie as Person, got: {:?}",
        entities
            .iter()
            .map(|e| (&e.text, &e.entity_type))
            .collect::<Vec<_>>()
    );

    // Should find Paris as a location
    let has_paris = entities
        .iter()
        .any(|e| e.text.contains("Paris") && matches!(e.entity_type, EntityType::Location));
    assert!(has_paris, "should find Paris as Location");
}

#[test]
#[ignore]
fn llm_extraction_gemini_flash_lite() {
    if !ensure_api_key() {
        eprintln!("SKIP: no API key available");
        return;
    }

    let config = LlmConfig::gemini_flash_lite();
    let model = UniversalNER::with_config(config).expect("create with gemini flash lite config");

    let text = "Sundar Pichai leads Google from Mountain View, California.";
    let entities = model
        .extract_entities(text, None)
        .expect("gemini flash lite extraction");

    eprintln!(
        "Gemini Flash Lite entities: {:?}",
        entities
            .iter()
            .map(|e| (&e.text, &e.entity_type))
            .collect::<Vec<_>>()
    );

    assert!(
        !entities.is_empty(),
        "Gemini Flash Lite should find entities"
    );
    assert_entity_invariants(&entities, text);
}

// =============================================================================
// Claude Haiku 4.5 via OpenRouter
// =============================================================================

#[test]
#[ignore]
fn llm_extraction_haiku() {
    if !ensure_api_key() {
        eprintln!("SKIP: no API key available");
        return;
    }

    let config = LlmConfig::haiku();
    let model = UniversalNER::with_config(config).expect("create with haiku config");

    let text = "Tim Cook is the CEO of Apple Inc., headquartered in Cupertino.";
    let entities = model
        .extract_entities(text, None)
        .expect("haiku extraction");

    eprintln!(
        "Haiku entities: {:?}",
        entities
            .iter()
            .map(|e| (&e.text, &e.entity_type))
            .collect::<Vec<_>>()
    );

    assert!(!entities.is_empty(), "Haiku should find entities");
    assert_entity_invariants(&entities, text);

    let has_person = entities
        .iter()
        .any(|e| matches!(e.entity_type, EntityType::Person));
    let has_org = entities
        .iter()
        .any(|e| matches!(e.entity_type, EntityType::Organization));
    assert!(has_person, "should find Tim Cook as Person");
    assert!(has_org, "should find Apple Inc. as Organization");
}

// =============================================================================
// Llama 3.3 70B via OpenRouter
// =============================================================================

#[test]
#[ignore]
fn llm_extraction_llama3() {
    if !ensure_api_key() {
        eprintln!("SKIP: no API key available");
        return;
    }

    let config = LlmConfig::llama3();
    let model = UniversalNER::with_config(config).expect("create with llama3 config");

    let text = "Jensen Huang founded NVIDIA in Santa Clara, California.";
    let entities = model
        .extract_entities(text, None)
        .expect("llama3 extraction");

    eprintln!(
        "Llama 3.3 70B entities: {:?}",
        entities
            .iter()
            .map(|e| (&e.text, &e.entity_type))
            .collect::<Vec<_>>()
    );

    assert!(!entities.is_empty(), "Llama 3.3 should find entities");
    assert_entity_invariants(&entities, text);
}

// =============================================================================
// Llama 4 Scout via OpenRouter
// =============================================================================

#[test]
#[ignore]
fn llm_extraction_llama4() {
    if !ensure_api_key() {
        eprintln!("SKIP: no API key available");
        return;
    }

    let config = LlmConfig::llama4();
    let model = UniversalNER::with_config(config).expect("create with llama4 config");

    let text = "Satya Nadella leads Microsoft from Redmond, Washington.";
    let entities = model
        .extract_entities(text, None)
        .expect("llama4 extraction");

    eprintln!(
        "Llama 4 Scout entities: {:?}",
        entities
            .iter()
            .map(|e| (&e.text, &e.entity_type))
            .collect::<Vec<_>>()
    );

    assert!(!entities.is_empty(), "Llama 4 should find entities");
    assert_entity_invariants(&entities, text);
}

// =============================================================================
// DeepSeek V3 via OpenRouter
// =============================================================================

#[test]
#[ignore]
fn llm_extraction_deepseek() {
    if !ensure_api_key() {
        eprintln!("SKIP: no API key available");
        return;
    }

    let config = LlmConfig::deepseek();
    let model = UniversalNER::with_config(config).expect("create with deepseek config");

    let text = "Angela Merkel led Germany through the European debt crisis.";
    let entities = model
        .extract_entities(text, None)
        .expect("deepseek extraction");

    eprintln!(
        "DeepSeek entities: {:?}",
        entities
            .iter()
            .map(|e| (&e.text, &e.entity_type))
            .collect::<Vec<_>>()
    );

    assert!(!entities.is_empty(), "DeepSeek should find entities");
    assert_entity_invariants(&entities, text);
}

// =============================================================================
// Gemini 2.5 Flash (full) via OpenRouter
// =============================================================================

#[test]
#[ignore]
fn llm_extraction_gemini_flash() {
    if !ensure_api_key() {
        eprintln!("SKIP: no API key available");
        return;
    }

    let config = LlmConfig::gemini_flash();
    let model = UniversalNER::with_config(config).expect("create with gemini flash config");

    let text = "Jeff Bezos founded Amazon in Seattle, Washington.";
    let entities = model
        .extract_entities(text, None)
        .expect("gemini flash extraction");

    eprintln!(
        "Gemini Flash entities: {:?}",
        entities
            .iter()
            .map(|e| (&e.text, &e.entity_type))
            .collect::<Vec<_>>()
    );

    assert!(!entities.is_empty(), "Gemini Flash should find entities");
    assert_entity_invariants(&entities, text);
}

// =============================================================================
// Caching: second call should return cached result
// =============================================================================

#[test]
#[ignore]
fn llm_caching_avoids_duplicate_calls() {
    if !ensure_api_key() {
        eprintln!("SKIP: no API key available");
        return;
    }

    let model = UniversalNER::new().expect("create UniversalNER");

    let text = "Barack Obama served as the 44th President of the United States.";

    // First call: hits the API
    let t0 = std::time::Instant::now();
    let entities1 = model
        .extract_entities(text, None)
        .expect("first extraction");
    let elapsed1 = t0.elapsed();

    // Second call: should hit cache (much faster)
    let t1 = std::time::Instant::now();
    let entities2 = model
        .extract_entities(text, None)
        .expect("second extraction");
    let elapsed2 = t1.elapsed();

    eprintln!("First call: {:?} ({} entities)", elapsed1, entities1.len());
    eprintln!("Second call: {:?} ({} entities)", elapsed2, entities2.len());

    // Verify same results
    assert_eq!(
        entities1.len(),
        entities2.len(),
        "cache should return same count"
    );
    for (a, b) in entities1.iter().zip(entities2.iter()) {
        assert_eq!(a.text, b.text, "cached entity text should match");
        assert_eq!(a.start, b.start, "cached entity start should match");
        assert_eq!(a.end, b.end, "cached entity end should match");
    }

    // Cache hit should be at least 10x faster (API call typically >500ms, cache <1ms)
    if elapsed1.as_millis() > 100 {
        assert!(
            elapsed2 < elapsed1 / 5,
            "cached call ({:?}) should be much faster than API call ({:?})",
            elapsed2,
            elapsed1
        );
    }
}

// =============================================================================
// CodeNER prompt strategy
// =============================================================================

#[test]
#[ignore]
fn llm_codener_prompt_strategy() {
    if !ensure_api_key() {
        eprintln!("SKIP: no API key available");
        return;
    }

    let model = UniversalNER::new()
        .expect("create UniversalNER")
        .prompt_strategy(PromptStrategy::CodeNER {
            chain_of_thought: false,
        });

    let text = "Steve Jobs co-founded Apple with Steve Wozniak in Los Altos.";
    let entities = model
        .extract_entities(text, None)
        .expect("CodeNER extraction");

    eprintln!(
        "CodeNER entities: {:?}",
        entities
            .iter()
            .map(|e| (&e.text, &e.entity_type))
            .collect::<Vec<_>>()
    );

    // CodeNER returns BIO-tagged output; parse_llm_response should handle it.
    // Even if the CodeNER prompt produces different format, the fallback JSON parsing
    // should still work since we request JSON output in the prompt.
    assert_entity_invariants(&entities, text);
}

// =============================================================================
// Compact prompt strategy (token-efficient)
// =============================================================================

#[test]
#[ignore]
fn llm_compact_prompt_strategy() {
    if !ensure_api_key() {
        eprintln!("SKIP: no API key available");
        return;
    }

    let model = UniversalNER::new()
        .expect("create UniversalNER")
        .prompt_strategy(PromptStrategy::Compact);

    let text = "Marie Curie discovered radium in Paris in 1898.";
    let entities = model
        .extract_entities(text, None)
        .expect("compact extraction");

    eprintln!(
        "Compact entities: {:?}",
        entities
            .iter()
            .map(|e| (&e.text, &e.entity_type, e.start, e.end))
            .collect::<Vec<_>>()
    );

    assert!(
        !entities.is_empty(),
        "Compact strategy should find entities"
    );
    assert_entity_invariants(&entities, text);
}

// =============================================================================
// Domain context improves recall (NER4all finding)
// =============================================================================

#[test]
#[ignore]
fn llm_domain_context_injection() {
    if !ensure_api_key() {
        eprintln!("SKIP: no API key available");
        return;
    }

    // Use a domain-specific text where context helps
    let text = "BRCA1 interacts with p53 and RAD51 in DNA repair pathways.";

    // Without domain context
    let model_plain = UniversalNER::new().expect("create plain model");
    let entities_plain = model_plain
        .extract_entities(text, None)
        .expect("plain extraction");

    // With domain context
    let model_domain = UniversalNER::new()
        .expect("create domain model")
        .domain_context(
        "Biomedical research: gene and protein entity recognition in molecular biology literature.",
    );
    let entities_domain = model_domain
        .extract_entities(text, None)
        .expect("domain extraction");

    eprintln!(
        "Without domain context: {:?}",
        entities_plain
            .iter()
            .map(|e| (&e.text, &e.entity_type))
            .collect::<Vec<_>>()
    );
    eprintln!(
        "With domain context: {:?}",
        entities_domain
            .iter()
            .map(|e| (&e.text, &e.entity_type))
            .collect::<Vec<_>>()
    );

    assert_entity_invariants(&entities_domain, text);

    // Domain context should find at least some entities (gene/protein names)
    // This is a soft assertion -- we log both for comparison
    eprintln!(
        "Entity count: plain={}, domain={}",
        entities_plain.len(),
        entities_domain.len()
    );
}

// =============================================================================
// Custom entity types (zero-shot)
// =============================================================================

#[test]
#[ignore]
fn llm_zero_shot_custom_types() {
    if !ensure_api_key() {
        eprintln!("SKIP: no API key available");
        return;
    }

    use anno::backends::inference::ZeroShotNER;

    let model = UniversalNER::new().expect("create UniversalNER");

    let text = "The Tesla Model 3 costs $35,000 and has a range of 358 miles.";
    let entities = model
        .extract_with_types(text, &["vehicle", "money", "measurement"], 0.3)
        .expect("zero-shot custom types");

    eprintln!(
        "Custom type entities: {:?}",
        entities
            .iter()
            .map(|e| (&e.text, &e.entity_type))
            .collect::<Vec<_>>()
    );

    assert!(
        !entities.is_empty(),
        "should find at least one entity with custom types"
    );
    assert_entity_invariants(&entities, text);
}

// =============================================================================
// Multilingual (Unicode) text
// =============================================================================

#[test]
#[ignore]
fn llm_multilingual_extraction() {
    if !ensure_api_key() {
        eprintln!("SKIP: no API key available");
        return;
    }

    let model = UniversalNER::new().expect("create UniversalNER");

    // Mixed CJK + Latin + Arabic
    let text = "李明 met François Hollande in الرياض last Tuesday.";
    let entities = model
        .extract_entities(text, None)
        .expect("multilingual extraction");

    eprintln!(
        "Multilingual entities: {:?}",
        entities
            .iter()
            .map(|e| (&e.text, &e.entity_type, e.start, e.end))
            .collect::<Vec<_>>()
    );

    assert!(
        !entities.is_empty(),
        "should find entities in multilingual text"
    );
    assert_entity_invariants(&entities, text);
}

// =============================================================================
// Self-verification (GPT-NER paper)
// =============================================================================

#[test]
#[ignore]
fn llm_self_verification() {
    if !ensure_api_key() {
        eprintln!("SKIP: no API key available");
        return;
    }

    let model = UniversalNER::new()
        .expect("create UniversalNER")
        .self_verify(true);

    let text = "Albert Einstein developed the theory of relativity at the ETH Zurich.";
    let entities = model
        .extract_entities(text, None)
        .expect("self-verified extraction");

    eprintln!(
        "Self-verified entities: {:?}",
        entities
            .iter()
            .map(|e| (&e.text, &e.entity_type))
            .collect::<Vec<_>>()
    );

    assert!(
        !entities.is_empty(),
        "self-verification should still return valid entities"
    );
    assert_entity_invariants(&entities, text);

    // Einstein should survive verification
    let has_einstein = entities.iter().any(|e| e.text.contains("Einstein"));
    assert!(has_einstein, "Einstein should survive self-verification");
}

// =============================================================================
// Edge cases
// =============================================================================

#[test]
#[ignore]
fn llm_empty_and_short_inputs() {
    if !ensure_api_key() {
        eprintln!("SKIP: no API key available");
        return;
    }

    let model = UniversalNER::new().expect("create UniversalNER");

    // Very short text with no entities
    let text = "Hello.";
    let entities = model
        .extract_entities(text, None)
        .expect("short text extraction");
    assert_entity_invariants(&entities, text);
    eprintln!(
        "Short text entities: {:?}",
        entities.iter().map(|e| &e.text).collect::<Vec<_>>()
    );
}

// =============================================================================
// Provenance tracking
// =============================================================================

#[test]
#[ignore]
fn llm_entities_have_provenance() {
    if !ensure_api_key() {
        eprintln!("SKIP: no API key available");
        return;
    }

    let model = UniversalNER::new().expect("create UniversalNER");

    let text = "Satya Nadella leads Microsoft in Redmond.";
    let entities = model.extract_entities(text, None).expect("extraction");

    for e in &entities {
        assert!(
            e.provenance.is_some(),
            "entity {:?} should have provenance",
            e.text
        );
        let prov = e.provenance.as_ref().unwrap();
        assert_eq!(prov.source, "universal_ner");
    }
}

// =============================================================================
// Chunking: large document (parallel chunk processing + coalescing)
// =============================================================================

#[test]
#[ignore]
fn llm_chunking_large_document() {
    if !ensure_api_key() {
        eprintln!("SKIP: no API key available");
        return;
    }

    // Build a document that exceeds the chunk threshold by repeating
    // paragraphs with distinct entities.
    let paragraphs = [
        "Albert Einstein developed the theory of relativity in Berlin. ",
        "Marie Curie conducted groundbreaking research in Paris at the Sorbonne. ",
        "Isaac Newton formulated the laws of motion in Cambridge, England. ",
        "Nikola Tesla pioneered alternating current technology in New York City. ",
        "Ada Lovelace wrote the first algorithm at the University of London. ",
        "Charles Darwin published On the Origin of Species while living in Kent. ",
        "Rosalind Franklin contributed to understanding DNA structure at King's College London. ",
        "Alan Turing broke the Enigma code at Bletchley Park in Buckinghamshire. ",
    ];
    // Repeat enough times to exceed 4000 chars (each paragraph ~60-90 chars)
    let mut text = String::new();
    for _ in 0..10 {
        for p in &paragraphs {
            text.push_str(p);
        }
    }
    let char_count = text.chars().count();
    eprintln!("Document size: {} chars", char_count);
    assert!(
        char_count > 4000,
        "test document should exceed chunk threshold"
    );

    // Use small chunk size to force multiple chunks + parallelism.
    let model = UniversalNER::new()
        .expect("create UniversalNER")
        .max_chunk_chars(800);

    let start = std::time::Instant::now();
    let entities = model
        .extract_entities(&text, None)
        .expect("chunked extraction");
    let elapsed = start.elapsed();
    eprintln!(
        "Chunked extraction: {} entities in {:.2}s",
        entities.len(),
        elapsed.as_secs_f64()
    );

    // Verify invariants
    assert_entity_invariants(&entities, &text);

    // Should find entities across the document
    assert!(
        entities.len() >= 5,
        "should find multiple entities across chunks, got {}",
        entities.len()
    );

    // Verify no duplicate (start, end) pairs (overlap dedup works)
    let mut spans: Vec<(usize, usize)> = entities.iter().map(|e| (e.start, e.end)).collect();
    let before = spans.len();
    spans.sort();
    spans.dedup();
    assert_eq!(
        before,
        spans.len(),
        "no duplicate entity spans after coalescing"
    );

    // Verify entities are sorted by position
    for i in 1..entities.len() {
        assert!(
            entities[i].start >= entities[i - 1].start,
            "entities should be sorted by position"
        );
    }

    // Check that known entities appear somewhere
    let texts: Vec<&str> = entities.iter().map(|e| e.text.as_str()).collect();
    let has_person = entities
        .iter()
        .any(|e| matches!(e.entity_type, EntityType::Person));
    let has_location = entities.iter().any(|e| {
        matches!(
            e.entity_type,
            EntityType::Location | EntityType::Custom { .. }
        )
    });
    eprintln!("Entity texts sample: {:?}", &texts[..texts.len().min(10)]);
    assert!(has_person, "should find at least one person entity");
    assert!(has_location, "should find at least one location/org entity");
}

// =============================================================================
// Chunking: small document (no chunking, passes through directly)
// =============================================================================

#[test]
#[ignore]
fn llm_chunking_small_document_passthrough() {
    if !ensure_api_key() {
        eprintln!("SKIP: no API key available");
        return;
    }

    let text = "Steve Jobs founded Apple in Cupertino.";
    assert!(
        text.chars().count() < 4000,
        "test text should be below chunk threshold"
    );

    let model = UniversalNER::new().expect("create UniversalNER");

    let entities = model
        .extract_entities(text, None)
        .expect("small doc extraction");
    assert_entity_invariants(&entities, text);
    assert!(
        !entities.is_empty(),
        "should find entities in small document"
    );
    eprintln!(
        "Small doc entities: {:?}",
        entities.iter().map(|e| &e.text).collect::<Vec<_>>()
    );
}
