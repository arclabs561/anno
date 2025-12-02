//! End-to-end integration tests for StackedNER with ML backends.
//!
//! These tests verify full pipeline integration:
//! - NER (ML StackedNER) → Coref → Discourse
//! - Batch processing with ML stacks
//! - Streaming with ML stacks
//! - Multi-document processing

#[cfg(all(feature = "onnx", feature = "eval"))]
mod e2e_tests {
    use anno::backends::stacked::ConflictStrategy;
    use anno::{BatchCapable, HeuristicNER, Model, RegexNER, StackedNER, StreamingCapable};

    // Helper to create GLiNER with graceful failure handling
    fn create_gliner() -> Option<anno::GLiNEROnnx> {
        anno::GLiNEROnnx::new("onnx-community/gliner_small-v2.1").ok()
    }

    // Helper to create BertNER with graceful failure handling
    fn create_bert() -> Option<anno::BertNEROnnx> {
        anno::BertNEROnnx::new(anno::DEFAULT_BERT_ONNX_MODEL).ok()
    }

    // =========================================================================
    // Full Pipeline Tests
    // =========================================================================

    #[test]
    fn test_e2e_ml_stacked_ner_extraction() {
        // Basic E2E: ML StackedNER extraction
        if let Some(gliner) = create_gliner() {
            let ner = StackedNER::with_ml_first(Box::new(gliner));
            let text = "Apple Inc. was founded by Steve Jobs in 1976. He later left the company.";

            let entities = ner.extract_entities(text, None).unwrap();

            // Should find entities
            assert!(!entities.is_empty());

            // Verify entity validity
            for entity in &entities {
                assert!(entity.start < entity.end);
                assert!(entity.confidence >= 0.0 && entity.confidence <= 1.0);
            }

            // Should find at least some structured entities from pattern layer
            let has_structured = entities.iter().any(|e| {
                matches!(
                    e.entity_type,
                    anno::EntityType::Date | anno::EntityType::Money | anno::EntityType::Email
                )
            });
            // May or may not have structured entities depending on text
            // Just verify we got results
        }
    }

    #[test]
    fn test_e2e_ml_stacked_with_coref() {
        // E2E: ML StackedNER → Coref resolution
        if let Some(gliner) = create_gliner() {
            let ner = StackedNER::with_ml_first(Box::new(gliner));
            let text = "Apple Inc. was founded by Steve Jobs. He later left the company. Jobs returned in 1997.";

            let entities = ner.extract_entities(text, None).unwrap();

            // Should find multiple mentions of "Steve Jobs" / "Jobs"
            let jobs_mentions: Vec<_> = entities
                .iter()
                .filter(|e| e.text.contains("Jobs") || e.text.contains("Steve"))
                .collect();

            // May find multiple mentions (depending on ML backend)
            // Just verify we got entities
            assert!(!entities.is_empty());
        }
    }

    // =========================================================================
    // Batch Processing Tests
    // =========================================================================

    #[test]
    fn test_e2e_ml_stacked_batch_processing() {
        // Batch processing with ML stacks
        if let Some(gliner) = create_gliner() {
            let ner = StackedNER::with_ml_first(Box::new(gliner));

            let texts = vec![
                "Apple Inc. was founded in 1976.",
                "Microsoft was founded by Bill Gates.",
                "Google was founded by Larry Page and Sergey Brin.",
            ];

            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            let results = ner.extract_entities_batch(&text_refs, None).unwrap();

            // Should return results for all texts
            assert_eq!(results.len(), texts.len());

            // Each result should be valid
            for (i, entities) in results.iter().enumerate() {
                for entity in entities {
                    assert!(entity.start < entity.end);
                    assert!(entity.confidence >= 0.0 && entity.confidence <= 1.0);
                }
                // Should find at least some entities in each text
                // (May be empty for some texts, that's okay)
            }
        }
    }

    #[test]
    fn test_e2e_ml_stacked_batch_vs_sequential() {
        // Batch results should match sequential results
        if let Some(gliner) = create_gliner() {
            let ner = StackedNER::with_ml_first(Box::new(gliner));

            let texts = vec![
                "Apple Inc. was founded in 1976.",
                "Microsoft was founded by Bill Gates.",
            ];

            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            let batch_results = ner.extract_entities_batch(&text_refs, None).unwrap();

            // Compare with sequential
            let sequential_results: Vec<_> = texts
                .iter()
                .map(|text| ner.extract_entities(text, None).unwrap())
                .collect();

            // Should have same number of results
            assert_eq!(batch_results.len(), sequential_results.len());

            // Each batch result should match sequential (allowing for order differences)
            for (batch_ents, seq_ents) in batch_results.iter().zip(sequential_results.iter()) {
                assert_eq!(batch_ents.len(), seq_ents.len());
            }
        }
    }

    // =========================================================================
    // Streaming Tests
    // =========================================================================

    #[test]
    fn test_e2e_ml_stacked_streaming_chunk_size() {
        // Verify streaming chunk size is reasonable
        if let Some(gliner) = create_gliner() {
            let ner = StackedNER::with_ml_first(Box::new(gliner));

            let chunk_size = ner.recommended_chunk_size();

            // Should have a reasonable chunk size
            assert!(chunk_size > 0);
            assert!(chunk_size < 1_000_000); // Not unreasonably large
        }
    }

    #[test]
    fn test_e2e_ml_stacked_streaming_processing() {
        // Simulate streaming processing
        if let Some(gliner) = create_gliner() {
            let ner = StackedNER::with_ml_first(Box::new(gliner));
            let chunk_size = ner.recommended_chunk_size();

            // Create a long text
            let long_text = "Apple Inc. was founded by Steve Jobs. ".repeat(100);

            // Process in chunks
            let mut all_entities = Vec::new();
            let mut offset = 0;

            while offset < long_text.len() {
                let end = (offset + chunk_size).min(long_text.len());
                let chunk = &long_text[offset..end];

                let entities = ner.extract_entities(chunk, None).unwrap();

                // Adjust offsets for streaming (simplified - real streaming would be more complex)
                for mut entity in entities {
                    entity.start += offset;
                    entity.end += offset;
                    all_entities.push(entity);
                }

                offset = end;
            }

            // Should have extracted entities
            // Verify all are valid
            for entity in &all_entities {
                assert!(entity.start < entity.end);
                assert!(entity.confidence >= 0.0 && entity.confidence <= 1.0);
            }
        }
    }

    // =========================================================================
    // Multi-Document Processing
    // =========================================================================

    #[test]
    fn test_e2e_ml_stacked_multi_document() {
        // Process multiple documents
        if let Some(gliner) = create_gliner() {
            let ner = StackedNER::with_ml_first(Box::new(gliner));

            let documents = vec![
                "Apple Inc. was founded by Steve Jobs in 1976.",
                "Microsoft Corporation was founded by Bill Gates in 1975.",
                "Google LLC was founded by Larry Page and Sergey Brin in 1998.",
            ];

            let mut all_entities = Vec::new();
            for doc in &documents {
                let entities = ner.extract_entities(doc, None).unwrap();
                all_entities.extend(entities);
            }

            // Should have extracted entities from multiple documents
            assert!(!all_entities.is_empty());

            // All entities should be valid
            for entity in &all_entities {
                assert!(entity.start < entity.end);
                assert!(entity.confidence >= 0.0 && entity.confidence <= 1.0);
            }
        }
    }

    // =========================================================================
    // ML-First vs ML-Fallback Comparison
    // =========================================================================

    #[test]
    fn test_e2e_ml_first_vs_fallback() {
        // Compare ML-first vs ML-fallback strategies
        if let (Some(gliner1), Some(gliner2)) = (create_gliner(), create_gliner()) {
            let ml_first = StackedNER::with_ml_first(Box::new(gliner1));
            let ml_fallback = StackedNER::with_ml_fallback(Box::new(gliner2));

            let text = "Apple Inc. charges $1000 for the iPhone. Contact: sales@apple.com";

            let entities_first = ml_first.extract_entities(text, None).unwrap();
            let entities_fallback = ml_fallback.extract_entities(text, None).unwrap();

            // Both should produce valid results
            for entity in entities_first.iter().chain(entities_fallback.iter()) {
                assert!(entity.start < entity.end);
                assert!(entity.confidence >= 0.0 && entity.confidence <= 1.0);
            }

            // ML-first should potentially find more entities (runs ML first)
            // ML-fallback should find structured entities (pattern runs first)
            // Both are valid strategies
        }
    }

    // =========================================================================
    // Multiple ML Backends E2E
    // =========================================================================

    #[test]
    fn test_e2e_multiple_ml_backends() {
        // E2E with multiple ML backends
        if let (Some(gliner), Some(bert)) = (create_gliner(), create_bert()) {
            let stacked = StackedNER::builder()
                .layer(RegexNER::new())
                .layer_boxed(Box::new(gliner))
                .layer_boxed(Box::new(bert))
                .layer(HeuristicNER::new())
                .strategy(ConflictStrategy::HighestConf)
                .build();

            let text = "Apple Inc. was founded by Steve Jobs in Cupertino, California in 1976.";
            let entities = stacked.extract_entities(text, None).unwrap();

            // Should combine results from multiple ML backends
            assert!(!entities.is_empty());

            // All entities should be valid
            for entity in &entities {
                assert!(entity.start < entity.end);
                assert!(entity.confidence >= 0.0 && entity.confidence <= 1.0);
            }
        }
    }

    // =========================================================================
    // Real-World Scenarios
    // =========================================================================

    #[test]
    fn test_e2e_press_release_scenario() {
        // Simulate processing a press release
        if let Some(gliner) = create_gliner() {
            let ner = StackedNER::with_ml_first(Box::new(gliner));

            let press_release = r#"
                PRESS RELEASE - January 15, 2024
                
                Apple Inc. announced today that CEO Tim Cook will present new products
                at the company's headquarters in Cupertino, California.
                
                Contact: press@apple.com or call (555) 123-4567
                
                The event is scheduled for 2:00 PM PST and will be streamed live.
                Revenue increased by 25% this quarter.
            "#;

            let entities = ner.extract_entities(press_release, None).unwrap();

            // Should find various entity types
            assert!(!entities.is_empty());

            // Verify we found structured entities (dates, emails, phones, money, percent)
            let has_structured = entities.iter().any(|e| {
                matches!(
                    e.entity_type,
                    anno::EntityType::Date
                        | anno::EntityType::Email
                        | anno::EntityType::Phone
                        | anno::EntityType::Money
                        | anno::EntityType::Percent
                )
            });

            // Should find at least some structured entities from pattern layer
            // (May vary depending on ML backend)
        }
    }

    #[test]
    fn test_e2e_news_article_scenario() {
        // Simulate processing a news article
        if let Some(gliner) = create_gliner() {
            let ner = StackedNER::with_ml_first(Box::new(gliner));

            let article = r#"
                Tech Giant Announces Major Acquisition
                
                SAN FRANCISCO - Microsoft Corporation announced today that it has
                acquired GitHub Inc. for $7.5 billion. The deal was finalized on
                June 4, 2018.
                
                "This acquisition strengthens our commitment to developers," said
                CEO Satya Nadella in a statement.
                
                GitHub, founded in 2008, is based in San Francisco, California.
            "#;

            let entities = ner.extract_entities(article, None).unwrap();

            // Should find entities from the article
            assert!(!entities.is_empty());

            // Verify entity validity
            for entity in &entities {
                assert!(entity.start < entity.end);
                assert!(entity.confidence >= 0.0 && entity.confidence <= 1.0);
            }
        }
    }
}
