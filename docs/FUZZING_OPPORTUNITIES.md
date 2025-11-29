# Fuzzing Opportunities Analysis

This document identifies areas where additional property-based testing (fuzzing) would significantly improve robustness and catch edge cases.

## Current Fuzzing Coverage

### ✅ Well-Covered Areas
- **PatternNER/HeuristicNER**: Basic fuzzing with ASCII/UTF-8 strings (`fuzz_edge_cases.rs`)
- **Corpus operations**: Property tests for inter-doc coref (`corpus_proptest.rs`)
- **Inference module**: Comprehensive property tests (`inference_tests.rs`)
- **Entity construction**: Basic offset validation (`fuzz_edge_cases.rs`)

### ⚠️ Areas Needing More Fuzzing

## 1. **Offset Conversion** (`src/offset.rs`) - 🔴 CRITICAL

**Why Critical**: Offset conversion bugs are the #1 source of NER errors. Byte/char/token mismatches cause entities to be extracted at wrong positions.

**Current Coverage**: ❌ **ZERO fuzzing tests**

**Recommended Fuzzing**:
```rust
proptest! {
    #[test]
    fn offset_conversion_roundtrip(
        text in ".{0,1000}",  // Random Unicode text
        byte_start in 0usize..1000,
        byte_end in 0usize..1000,
    ) {
        if byte_start <= byte_end && byte_end <= text.len() {
            let (char_start, char_end) = bytes_to_chars(&text, byte_start, byte_end);
            let (byte_start2, byte_end2) = chars_to_bytes(&text, char_start, char_end);
            
            // Roundtrip should preserve byte offsets
            prop_assert_eq!(byte_start, byte_start2);
            prop_assert_eq!(byte_end, byte_end2);
        }
    }
    
    #[test]
    fn span_converter_consistency(
        text in ".{0,500}",
        byte_idx in 0usize..500,
    ) {
        let converter = SpanConverter::new(&text);
        let char_idx = converter.byte_to_char(byte_idx);
        let byte_idx2 = converter.char_to_byte(char_idx);
        
        // Should be consistent (may differ for multi-byte chars)
        if text.is_ascii() {
            prop_assert_eq!(byte_idx, byte_idx2);
        }
    }
    
    #[test]
    fn offset_bounds_always_valid(
        text in ".{0,1000}",
        char_start in 0usize..1000,
        char_end in 0usize..1000,
    ) {
        let char_count = text.chars().count();
        if char_start <= char_end && char_end <= char_count {
            let (byte_start, byte_end) = chars_to_bytes(&text, char_start, char_end);
            prop_assert!(byte_start <= byte_end);
            prop_assert!(byte_end <= text.len());
        }
    }
}
```

**Edge Cases to Fuzz**:
- Multi-byte UTF-8 characters (é, 日本, emoji)
- Surrogate pairs
- Combining characters
- Zero-width characters
- Boundary conditions (start=0, end=len, start=end)

## 2. **String Similarity** (`src/similarity.rs`) - 🟡 HIGH PRIORITY

**Why Important**: Used for coreference resolution and entity linking. Bugs cause incorrect clustering.

**Current Coverage**: ❌ **Only unit tests, no property-based tests**

**Recommended Fuzzing**:
```rust
proptest! {
    #[test]
    fn similarity_symmetric(
        a in ".{0,100}",
        b in ".{0,100}",
    ) {
        let sim_ab = string_similarity(&a, &b);
        let sim_ba = string_similarity(&b, &a);
        
        // Similarity should be symmetric
        prop_assert!((sim_ab - sim_ba).abs() < 0.001);
    }
    
    #[test]
    fn similarity_bounded(
        a in ".{0,100}",
        b in ".{0,100}",
    ) {
        let sim = string_similarity(&a, &b);
        // Should always be in [0.0, 1.0]
        prop_assert!(sim >= 0.0 && sim <= 1.0);
    }
    
    #[test]
    fn similarity_identical_is_one(
        text in ".{0,100}",
    ) {
        let sim = string_similarity(&text, &text);
        prop_assert!((sim - 1.0).abs() < 0.001);
    }
    
    #[test]
    fn jaccard_commutative(
        a in ".{0,100}",
        b in ".{0,100}",
    ) {
        let j1 = jaccard_word_similarity(&a, &b);
        let j2 = jaccard_word_similarity(&b, &a);
        prop_assert!((j1 - j2).abs() < 0.001);
    }
}
```

## 3. **Graph Export** (`src/graph.rs`) - 🟡 HIGH PRIORITY

**Why Important**: Graph export bugs cause data corruption in downstream systems (Neo4j, NetworkX).

**Current Coverage**: ❌ **Only unit tests, no fuzzing**

**Recommended Fuzzing**:
```rust
proptest! {
    #[test]
    fn graph_export_roundtrip(
        entities in prop::collection::vec(
            entity_strategy(), 0..20
        ),
        relations in prop::collection::vec(
            relation_strategy(), 0..10
        ),
    ) {
        let graph = GraphDocument::from_extraction(&entities, &relations, None);
        
        // Cypher export should be parseable (basic syntax check)
        let cypher = graph.to_cypher();
        prop_assert!(cypher.contains("CREATE"));
        
        // NetworkX JSON should be valid JSON
        let nx_json = graph.to_networkx_json();
        let parsed: serde_json::Value = serde_json::from_str(&nx_json)
            .expect("Should be valid JSON");
        prop_assert!(parsed.get("nodes").is_some());
        prop_assert!(parsed.get("links").is_some());
        
        // JSON-LD should be valid JSON
        let json_ld = graph.export(GraphExportFormat::JsonLd);
        let _parsed: serde_json::Value = serde_json::from_str(&json_ld)
            .expect("Should be valid JSON");
    }
    
    #[test]
    fn graph_node_count_matches_entities(
        entities in prop::collection::vec(
            entity_strategy(), 0..50
        ),
    ) {
        let graph = GraphDocument::from_extraction(&entities, &[], None);
        // Should deduplicate by canonical_id
        let unique_ids: HashSet<_> = entities
            .iter()
            .filter_map(|e| e.canonical_id)
            .collect();
        let expected_nodes = unique_ids.len().max(entities.len());
        prop_assert!(graph.node_count() <= expected_nodes);
    }
}
```

## 4. **Schema Mapping** (`src/schema.rs`) - 🟡 MEDIUM PRIORITY

**Why Important**: Schema mapping bugs cause incorrect entity type assignments.

**Current Coverage**: ❌ **Only unit tests, no fuzzing**

**Recommended Fuzzing**:
```rust
proptest! {
    #[test]
    fn schema_mapper_idempotent(
        label in "[A-Z_]{1,20}",  // Random label
    ) {
        let mapper = SchemaMapper::for_dataset(DatasetSchema::OntoNotes);
        let canonical1 = mapper.to_canonical(&label);
        let canonical2 = mapper.to_canonical(&label);
        
        // Should be idempotent
        prop_assert_eq!(canonical1, canonical2);
    }
    
    #[test]
    fn schema_mapper_handles_bio_prefixes(
        label in "[A-Z_]{1,15}",
        prefix in prop::sample::select(vec!["B-", "I-", "S-", "E-"]),
    ) {
        let prefixed = format!("{}{}", prefix, label);
        let mapper = SchemaMapper::for_dataset(DatasetSchema::CoNLL2003);
        let canonical = mapper.to_canonical(&prefixed);
        
        // Should strip prefix and map correctly
        prop_assert!(canonical != CanonicalType::Misc || label == "UNKNOWN");
    }
    
    #[test]
    fn coarse_type_from_canonical_always_valid(
        canonical in canonical_type_strategy(),
    ) {
        let coarse = CoarseType::from_canonical(canonical);
        // Should always produce a valid CoarseType
        match coarse {
            CoarseType::Person | CoarseType::Organization | 
            CoarseType::Location | CoarseType::DateTime |
            CoarseType::Numeric | CoarseType::Other => {}
        }
    }
}
```

## 5. **Language Detection** (`src/lang.rs`) - 🟢 LOW PRIORITY

**Why Important**: Language detection bugs cause incorrect backend selection.

**Current Coverage**: ❌ **Only unit tests, no fuzzing**

**Recommended Fuzzing**:
```rust
proptest! {
    #[test]
    fn language_detection_never_panics(
        text in ".{0,500}",
    ) {
        let lang = detect_language(&text);
        // Should always return a valid Language
        match lang {
            Language::English | Language::German | Language::French |
            Language::Spanish | Language::Chinese | Language::Japanese |
            Language::Korean | Language::Arabic | Language::Hebrew |
            Language::Russian | Language::Other => {}
        }
    }
    
    #[test]
    fn language_detection_empty_text(
        text in "\\s*",  // Whitespace only
    ) {
        let lang = detect_language(&text);
        // Empty/whitespace should default to English
        prop_assert_eq!(lang, Language::English);
    }
}
```

## 6. **Entity Serialization** - 🟡 MEDIUM PRIORITY

**Why Important**: Serialization bugs cause data loss or corruption in production.

**Current Coverage**: ⚠️ **Some roundtrip tests, but no fuzzing**

**Recommended Fuzzing**:
```rust
proptest! {
    #[test]
    fn entity_json_roundtrip(
        entity in entity_strategy(),
    ) {
        let json = serde_json::to_string(&entity).unwrap();
        let restored: Entity = serde_json::from_str(&json).unwrap();
        
        // Critical fields should match
        prop_assert_eq!(entity.text, restored.text);
        prop_assert_eq!(entity.entity_type, restored.entity_type);
        prop_assert_eq!(entity.start, restored.start);
        prop_assert_eq!(entity.end, restored.end);
        prop_assert!((entity.confidence - restored.confidence).abs() < 0.001);
    }
    
    #[test]
    fn entity_list_serialization_roundtrip(
        entities in prop::collection::vec(entity_strategy(), 0..100),
    ) {
        let json = serde_json::to_string(&entities).unwrap();
        let restored: Vec<Entity> = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(entities.len(), restored.len());
    }
}
```

## 7. **Entity Validation** - 🟡 MEDIUM PRIORITY

**Why Important**: Invalid entities cause downstream errors.

**Current Coverage**: ⚠️ **Some validation tests, but no fuzzing**

**Recommended Fuzzing**:
```rust
proptest! {
    #[test]
    fn entity_validation_catches_invalid_spans(
        text in ".{0,100}",
        start in 0usize..200,
        end in 0usize..200,
    ) {
        let char_count = text.chars().count();
        let entity = Entity::new("test", EntityType::Person, start, end, 0.9);
        let issues = entity.validate(&text);
        
        if start >= end {
            // Invalid span should be caught
            prop_assert!(issues.iter().any(|i| matches!(i, ValidationIssue::InvalidSpan {..})));
        } else if end > char_count {
            // Out of bounds should be caught
            prop_assert!(issues.iter().any(|i| matches!(i, ValidationIssue::SpanOutOfBounds {..})));
        }
    }
    
    #[test]
    fn entity_validation_confidence_bounds(
        confidence in -1.0f64..2.0f64,
    ) {
        let entity = Entity::new("test", EntityType::Person, 0, 4, confidence);
        // Entity::new clamps confidence, so validation should pass
        let issues = entity.validate("test");
        prop_assert!(!issues.iter().any(|i| matches!(i, ValidationIssue::InvalidConfidence {..})));
    }
}
```

## 8. **GroundedDocument Operations** - 🟢 LOW PRIORITY

**Why Important**: Signal/Track/Identity operations are complex and error-prone.

**Current Coverage**: ⚠️ **Some tests, but no fuzzing**

**Recommended Fuzzing**:
```rust
proptest! {
    #[test]
    fn grounded_document_signal_track_consistency(
        signals in prop::collection::vec(signal_strategy(), 1..20),
    ) {
        let mut doc = GroundedDocument::new("doc1", "Test text");
        let mut track_ids = Vec::new();
        
        for signal in signals {
            let signal_id = doc.add_signal(signal);
            let mut track = Track::new(0, "Test");
            track.add_signal(signal_id, 0);
            let track_id = doc.add_track(track);
            track_ids.push(track_id);
        }
        
        // All tracks should be retrievable
        for track_id in track_ids {
            prop_assert!(doc.track(track_id).is_some());
        }
    }
}
```

## 9. **Coreference Resolution** - 🟡 MEDIUM PRIORITY

**Why Important**: Coreference bugs cause incorrect entity clustering.

**Current Coverage**: ⚠️ **Some tests in inference_tests.rs, but could use more**

**Recommended Fuzzing**:
```rust
proptest! {
    #[test]
    fn coref_chains_well_formed(
        mentions in prop::collection::vec(mention_strategy(), 1..50),
    ) {
        // Create chains from mentions
        let chains = create_coref_chains(&mentions);
        
        for chain in &chains {
            // All mentions in chain should have valid spans
            for mention in &chain.mentions {
                prop_assert!(mention.start < mention.end);
            }
            
            // Mentions should be ordered by position
            for i in 1..chain.mentions.len() {
                prop_assert!(chain.mentions[i-1].start <= chain.mentions[i].start);
            }
        }
    }
}
```

## 10. **Entity Builder Fluent API** - 🟢 LOW PRIORITY

**Why Important**: Builder pattern bugs cause incorrect entity construction.

**Current Coverage**: ❌ **No fuzzing**

**Recommended Fuzzing**:
```rust
proptest! {
    #[test]
    fn entity_builder_fluent_chaining(
        text in ".{1,100}",
        entity_type in entity_type_strategy(),
        start in 0usize..100,
        end in 0usize..100,
        confidence in 0.0f64..1.0f64,
    ) {
        if start < end {
            let entity = Entity::builder(&text, entity_type)
                .span(start, end)
                .confidence(confidence)
                .build();
            
            prop_assert_eq!(entity.text, text);
            prop_assert_eq!(entity.start, start);
            prop_assert_eq!(entity.end, end);
            prop_assert!((entity.confidence - confidence.clamp(0.0, 1.0)).abs() < 0.001);
        }
    }
}
```

## Implementation Priority

1. **🔴 CRITICAL**: Offset conversion fuzzing (most likely to catch production bugs)
2. **🟡 HIGH**: String similarity, Graph export fuzzing
3. **🟡 MEDIUM**: Schema mapping, Entity serialization, Entity validation, Coreference
4. **🟢 LOW**: Language detection, GroundedDocument, Entity builder

## Test File Organization

Recommended new test files:
- `tests/offset_fuzz_tests.rs` - Offset conversion property tests
- `tests/similarity_fuzz_tests.rs` - Similarity function property tests
- `tests/graph_fuzz_tests.rs` - Graph export property tests
- `tests/schema_fuzz_tests.rs` - Schema mapping property tests
- `tests/serialization_fuzz_tests.rs` - Serialization roundtrip tests
- `tests/entity_validation_fuzz_tests.rs` - Entity validation property tests

## Strategy Generators Needed

```rust
// In a shared test utilities module
fn entity_strategy() -> impl Strategy<Value = Entity> {
    (
        ".{1,50}",  // text
        entity_type_strategy(),
        0usize..1000,  // start
        0usize..1000,  // end
        0.0f64..1.0f64,  // confidence
    ).prop_map(|(text, entity_type, start, end, confidence)| {
        Entity::new(text, entity_type, start.min(end), end.max(start), confidence)
    })
}

fn canonical_type_strategy() -> impl Strategy<Value = CanonicalType> {
    prop::sample::select(vec![
        CanonicalType::Person,
        CanonicalType::Organization,
        CanonicalType::Location,
        // ... all variants
    ])
}
```

## Benefits of Additional Fuzzing

1. **Catch edge cases** that unit tests miss
2. **Verify invariants** hold for all inputs
3. **Find bugs early** before they reach production
4. **Document expected behavior** through property tests
5. **Regression prevention** when code changes

## Running Fuzzing Tests

```bash
# Run all fuzzing tests
cargo test --test '*fuzz*'

# Run with more cases (default is 256)
PROPTEST_CASES=10000 cargo test --test offset_fuzz_tests

# Run with seed for reproducibility
PROPTEST_SEED=12345 cargo test --test offset_fuzz_tests
```

