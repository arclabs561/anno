//! Dataset command - Work with NER datasets

use clap::{Parser, Subcommand};
use std::time::Instant;

use super::super::output::color;
use super::super::parser::{EvalTask, ModelBackend};
use super::super::utils::types_match_flexible;

#[cfg(feature = "eval-advanced")]
use super::super::utils::create_entity_pair_relations;

#[cfg(feature = "eval")]
use crate::eval::loader::DatasetId;

/// Work with NER datasets
#[derive(Parser, Debug)]
pub struct DatasetArgs {
    /// Action to perform
    #[command(subcommand)]
    pub action: DatasetAction,
}

#[derive(Subcommand, Debug)]
pub enum DatasetAction {
    /// List available datasets
    #[command(visible_alias = "ls")]
    List,

    /// Show dataset statistics
    #[command(visible_alias = "i")]
    Info {
        /// Dataset name
        #[arg(short, long)]
        dataset: String,
    },

    /// Evaluate model on dataset
    #[command(visible_alias = "e")]
    Eval {
        /// Dataset name
        #[arg(short, long, default_value = "synthetic")]
        dataset: String,

        /// Model backend to use
        #[arg(short, long, default_value = "stacked")]
        model: ModelBackend,

        /// Task type: ner, coref, or relation
        #[arg(short, long, default_value = "ner")]
        task: EvalTask,
    },
}

pub fn run(args: DatasetArgs) -> Result<(), String> {
    match args.action {
        DatasetAction::List => {
            println!();
            println!("{}", color("1;36", "Available Datasets"));
            println!();

            #[cfg(feature = "eval-advanced")]
            {
                println!("  Downloadable (with --features eval-advanced):");
                println!("    - wikigold    : WikiGold NER corpus");
                println!("    - wnut17      : WNUT 2017 emerging entities");
                println!("    - conll2003   : CoNLL 2003 (requires manual download)");
            }

            println!();
            println!("  Synthetic (always available):");
            println!("    - synthetic   : Generated test cases");
            println!("    - robustness  : Adversarial perturbations");
            println!();
        }
        DatasetAction::Info { dataset } => {
            println!("Dataset: {}", dataset);
            // TODO: Show stats about the dataset
        }
        DatasetAction::Eval {
            dataset,
            model,
            task,
        } => {
            #[cfg(feature = "eval")]
            {
                let m = model.create_model()?;

                let (name, test_cases) = if dataset == "synthetic" {
                    (
                        "synthetic".to_string(),
                        vec![
                            (
                                "Marie Curie won the Nobel Prize.".to_string(),
                                vec![
                                    crate::eval::GoldEntity {
                                        text: "Marie Curie".to_string(),
                                        original_label: "PER".to_string(),
                                        entity_type: anno_core::EntityType::Person,
                                        start: 0,
                                        end: 11,
                                    },
                                    crate::eval::GoldEntity {
                                        text: "Nobel Prize".to_string(),
                                        original_label: "MISC".to_string(),
                                        entity_type: anno_core::EntityType::Other(
                                            "MISC".to_string(),
                                        ),
                                        start: 20,
                                        end: 31,
                                    },
                                ],
                            ),
                            (
                                "Apple Inc. is based in California.".to_string(),
                                vec![
                                    crate::eval::GoldEntity {
                                        text: "Apple Inc.".to_string(),
                                        original_label: "ORG".to_string(),
                                        entity_type: anno_core::EntityType::Organization,
                                        start: 0,
                                        end: 10,
                                    },
                                    crate::eval::GoldEntity {
                                        text: "California".to_string(),
                                        original_label: "LOC".to_string(),
                                        entity_type: anno_core::EntityType::Location,
                                        start: 24,
                                        end: 34,
                                    },
                                ],
                            ),
                            (
                                "Contact john@example.com for help.".to_string(),
                                vec![crate::eval::GoldEntity {
                                    text: "john@example.com".to_string(),
                                    original_label: "EMAIL".to_string(),
                                    entity_type: anno_core::EntityType::Other("EMAIL".to_string()),
                                    start: 8,
                                    end: 24,
                                }],
                            ),
                        ],
                    )
                } else {
                    // Parse dataset ID
                    let dataset_id: DatasetId = dataset
                        .parse::<DatasetId>()
                        .map_err(|e| format!("Invalid dataset '{}': {}", dataset, e))?;

                    #[cfg(not(feature = "eval-advanced"))]
                    {
                        let _ = dataset_id; // Suppress unused warning
                        return Err(
                            "Loading real datasets requires --features eval-advanced".to_string()
                        );
                    }

                    #[cfg(feature = "eval-advanced")]
                    {
                        use crate::eval::loader::DatasetLoader;

                        let loader = DatasetLoader::new()
                            .map_err(|e| format!("Failed to init dataset loader: {}", e))?;

                        println!(
                            "Loading {} (may download if not cached)...",
                            dataset_id.name()
                        );
                        let ds = loader
                            .load_or_download(dataset_id)
                            .map_err(|e| format!("Failed to load dataset: {}", e))?;

                        // Only warn if evaluating NER on non-NER dataset (not for coref/relation tasks)
                        if matches!(task, EvalTask::Ner)
                            && (dataset_id.is_coreference() || dataset_id.is_relation_extraction())
                        {
                            println!("{} Warning: Evaluating NER on non-NER dataset. Results may be empty.", color("33", "!"));
                        }

                        (ds.stats().name, ds.to_test_cases())
                    }
                };

                // Route to appropriate evaluation based on task
                match task {
                    EvalTask::Ner => {
                        #[cfg(feature = "eval-advanced")]
                        let type_mapper: Option<crate::TypeMapper> = if dataset != "synthetic" {
                            dataset
                                .parse::<DatasetId>()
                                .ok()
                                .and_then(|id| id.type_mapper())
                        } else {
                            None
                        };
                        #[cfg(not(feature = "eval-advanced"))]
                        let type_mapper: Option<crate::TypeMapper> = None;

                        println!();
                        println!("Evaluating {} on {} dataset (NER)...", model.name(), name);
                        if type_mapper.is_some() {
                            println!(
                                "  {} Using type mapping for domain-specific dataset",
                                color("33", "!")
                            );
                        }
                        println!("  Sentences: {}", test_cases.len());
                        println!();

                        let mut total_gold = 0;
                        let mut total_pred = 0;
                        let mut total_correct = 0;

                        let start_time = Instant::now();

                        // Validate gold annotations before evaluation (warn but continue)
                        #[cfg(feature = "eval-advanced")]
                        {
                            use crate::eval::validation::validate_ground_truth_entities;
                            for (text, gold) in &test_cases {
                                let validation = validate_ground_truth_entities(text, gold, false);
                                if !validation.is_valid {
                                    eprintln!(
                                        "{} Invalid gold annotations: {}",
                                        color("33", "warning:"),
                                        validation.errors.join("; ")
                                    );
                                }
                                // Note: Warnings are typically non-critical (e.g., overlapping entities)
                                // Only show first few warnings to avoid spam
                                if !validation.warnings.is_empty() && validation.warnings.len() <= 3
                                {
                                    for warning in validation.warnings.iter().take(3) {
                                        eprintln!("{} {}", color("33", "warning:"), warning);
                                    }
                                }
                            }
                        }

                        for (text, gold) in &test_cases {
                            let entities = m.extract_entities(text, None).unwrap_or_default();

                            total_gold += gold.len();
                            total_pred += entities.len();

                            // Track which predictions have been matched to prevent double-counting
                            let mut matched_pred = vec![false; entities.len()];

                            for gold_entity in gold {
                                // Apply type mapping if available
                                let gold_type = if let Some(ref mapper) = type_mapper {
                                    mapper.normalize(&gold_entity.original_label)
                                } else {
                                    anno_core::EntityType::from_label(&gold_entity.original_label)
                                };

                                // Match: exact span + type match (with flexible type matching)
                                // Find first unmatched prediction that matches
                                let matched = entities.iter().enumerate().any(|(i, e)| {
                                    if matched_pred[i] {
                                        return false; // Already matched
                                    }

                                    let span_match =
                                        e.start == gold_entity.start && e.end == gold_entity.end;
                                    if !span_match {
                                        return false;
                                    }

                                    // Type match with flexible matching
                                    let pred_type_str = e.entity_type.as_label();
                                    let gold_type_str = gold_type.as_label();

                                    // Exact match or flexible match
                                    let type_matches = pred_type_str == gold_type_str
                                        || types_match_flexible(pred_type_str, gold_type_str);

                                    if type_matches {
                                        matched_pred[i] = true; // Mark as matched
                                        return true;
                                    }

                                    false
                                });

                                if matched {
                                    total_correct += 1;
                                }
                            }
                        }

                        let elapsed = start_time.elapsed();

                        let p = if total_pred > 0 {
                            total_correct as f64 / total_pred as f64
                        } else {
                            0.0
                        };
                        let r = if total_gold > 0 {
                            total_correct as f64 / total_gold as f64
                        } else {
                            0.0
                        };
                        let f1 = if p + r > 0.0 {
                            2.0 * p * r / (p + r)
                        } else {
                            0.0
                        };

                        println!("Results:");
                        println!(
                            "  Gold: {}  Predicted: {}  Correct: {}",
                            total_gold, total_pred, total_correct
                        );
                        println!(
                            "  P: {:.1}%  R: {:.1}%  F1: {:.1}%",
                            p * 100.0,
                            r * 100.0,
                            f1 * 100.0
                        );
                        let ms_per_sent = if !test_cases.is_empty() {
                            elapsed.as_secs_f64() * 1000.0 / test_cases.len() as f64
                        } else {
                            0.0
                        };
                        println!(
                            "  Time: {:.1}s ({:.1}ms/sent)",
                            elapsed.as_secs_f64(),
                            ms_per_sent
                        );
                        println!();
                    }
                    EvalTask::Coref => {
                        #[cfg(not(feature = "eval-advanced"))]
                        {
                            return Err("Coreference evaluation requires --features eval-advanced"
                                .to_string());
                        }
                        #[cfg(feature = "eval-advanced")]
                        {
                            use crate::eval::coref_resolver::SimpleCorefResolver;
                            use crate::eval::loader::DatasetLoader;

                            if dataset == "synthetic" {
                                return Err("Coreference evaluation requires a real dataset (e.g., gap, preco, litbank)".to_string());
                            }

                            let dataset_id: DatasetId = dataset
                                .parse::<DatasetId>()
                                .map_err(|e| format!("Invalid dataset '{}': {}", dataset, e))?;

                            if !dataset_id.is_coreference() {
                                return Err(format!("Dataset '{}' is not a coreference dataset. Use: gap, preco, or litbank", dataset));
                            }

                            let loader = DatasetLoader::new()
                                .map_err(|e| format!("Failed to init dataset loader: {}", e))?;

                            println!();
                            println!(
                                "Evaluating coreference resolution on {} dataset...",
                                dataset_id.name()
                            );
                            println!("Loading dataset (may download if not cached)...");

                            let gold_docs =
                                loader.load_or_download_coref(dataset_id).map_err(|e| {
                                    format!("Failed to load coreference dataset: {}", e)
                                })?;

                            println!("  Documents: {}", gold_docs.len());
                            println!();

                            let resolver = SimpleCorefResolver::default();
                            let mut all_pred_chains: Vec<Vec<crate::eval::coref::CorefChain>> =
                                Vec::new();
                            let mut all_gold_chains: Vec<&[crate::eval::coref::CorefChain]> =
                                Vec::new();
                            let start_time = Instant::now();

                            for doc in &gold_docs {
                                let text = doc.text.as_str();
                                all_gold_chains.push(&doc.chains);

                                // Extract entities using NER
                                let entities = m.extract_entities(text, None).unwrap_or_default();

                                // Resolve coreference
                                let pred_chains = resolver.resolve_to_chains(&entities);
                                all_pred_chains.push(pred_chains);
                            }

                            let elapsed = start_time.elapsed();

                            // Build document pairs
                            let document_pairs: Vec<_> = all_pred_chains
                                .iter()
                                .zip(all_gold_chains.iter())
                                .map(|(pred, gold)| (pred.as_slice(), *gold))
                                .collect();

                            // Compute aggregate metrics
                            let results =
                                crate::eval::coref_metrics::AggregateCorefEvaluation::compute(
                                    &document_pairs,
                                );

                            println!("Results:");
                            println!("  CoNLL F1: {:.3}", results.mean.conll_f1);
                            println!(
                                "  MUC: P={:.3} R={:.3} F1={:.3}",
                                results.mean.muc.precision,
                                results.mean.muc.recall,
                                results.mean.muc.f1
                            );
                            println!(
                                "  B³: P={:.3} R={:.3} F1={:.3}",
                                results.mean.b_cubed.precision,
                                results.mean.b_cubed.recall,
                                results.mean.b_cubed.f1
                            );
                            println!(
                                "  CEAF-e: P={:.3} R={:.3} F1={:.3}",
                                results.mean.ceaf_e.precision,
                                results.mean.ceaf_e.recall,
                                results.mean.ceaf_e.f1
                            );
                            println!(
                                "  LEA: P={:.3} R={:.3} F1={:.3}",
                                results.mean.lea.precision,
                                results.mean.lea.recall,
                                results.mean.lea.f1
                            );
                            println!(
                                "  BLANC: P={:.3} R={:.3} F1={:.3}",
                                results.mean.blanc.precision,
                                results.mean.blanc.recall,
                                results.mean.blanc.f1
                            );
                            println!("  Documents: {}", results.num_documents);
                            println!("  Time: {:.1}s", elapsed.as_secs_f64());
                            println!();
                        }
                    }
                    EvalTask::Relation => {
                        #[cfg(not(feature = "eval-advanced"))]
                        {
                            return Err(
                                "Relation extraction evaluation requires --features eval-advanced"
                                    .to_string(),
                            );
                        }
                        #[cfg(feature = "eval-advanced")]
                        {
                            use crate::backends::inference::RelationExtractor;
                            use crate::eval::loader::DatasetLoader;
                            use crate::eval::relation::{
                                evaluate_relations, RelationEvalConfig, RelationPrediction,
                            };

                            if dataset == "synthetic" {
                                return Err("Relation extraction evaluation requires a real dataset (e.g., docred, retacred)".to_string());
                            }

                            let dataset_id: DatasetId = dataset
                                .parse::<DatasetId>()
                                .map_err(|e| format!("Invalid dataset '{}': {}", dataset, e))?;

                            if !dataset_id.is_relation_extraction() {
                                return Err(format!("Dataset '{}' is not a relation extraction dataset. Use: docred or retacred", dataset));
                            }

                            let loader = DatasetLoader::new()
                                .map_err(|e| format!("Failed to init dataset loader: {}", e))?;

                            println!();
                            println!(
                                "Evaluating relation extraction on {} dataset...",
                                dataset_id.name()
                            );
                            println!("Loading dataset (may download if not cached)...");

                            let gold_docs = loader
                                .load_or_download_relation(dataset_id)
                                .map_err(|e| format!("Failed to load relation dataset: {}", e))?;

                            println!("  Documents: {}", gold_docs.len());
                            println!();

                            // Try to use RelationExtractor if available (e.g., GLiNER2)
                            // Otherwise fall back to entity-pair heuristic

                            // Collect entity types and relation types from gold data
                            let mut entity_types = std::collections::HashSet::new();
                            let mut relation_types = std::collections::HashSet::new();
                            for doc in &gold_docs {
                                for rel in &doc.relations {
                                    entity_types.insert(rel.head_type.clone());
                                    entity_types.insert(rel.tail_type.clone());
                                    relation_types.insert(rel.relation_type.clone());
                                }
                            }

                            let entity_types_vec: Vec<&str> =
                                entity_types.iter().map(|s| s.as_str()).collect();
                            let relation_types_vec: Vec<&str> =
                                relation_types.iter().map(|s| s.as_str()).collect();

                            println!("  Entity types: {}", entity_types_vec.join(", "));
                            println!(
                                "  Relation types: {} ({} total)",
                                relation_types_vec.len(),
                                relation_types_vec
                                    .iter()
                                    .take(5)
                                    .cloned()
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            );
                            println!();

                            // Check if we can use GLiNER2 for relation extraction
                            let use_relation_extractor: Option<Box<dyn RelationExtractor>> = {
                                #[cfg(feature = "onnx")]
                                {
                                    // Try to create GLiNER2 multitask model for relation extraction
                                    if let Ok(gliner2) =
                                        crate::backends::gliner2::GLiNER2Onnx::from_pretrained(
                                            "onnx-community/gliner-multitask-large-v0.5",
                                        )
                                    {
                                        Some(Box::new(gliner2) as Box<dyn RelationExtractor>)
                                    } else {
                                        None
                                    }
                                }
                                #[cfg(not(feature = "onnx"))]
                                {
                                    None
                                }
                            };

                            let mut all_gold = Vec::new();
                            let mut all_pred = Vec::new();
                            let start_time = Instant::now();

                            if let Some(ref rel_extractor) = use_relation_extractor {
                                println!("{} Using GLiNER2 RelationExtractor (heuristic-based regex matching)", color("32", "✓"));
                                println!("  Note: This uses regex matching on text, not a neural relation model.",);
                                println!();

                                for doc in &gold_docs {
                                    let text = doc.text.as_str();
                                    all_gold.extend(doc.relations.clone());

                                    // Use RelationExtractor
                                    match rel_extractor.extract_with_relations(
                                        text,
                                        &entity_types_vec,
                                        &relation_types_vec,
                                        0.5,
                                    ) {
                                        Ok(result) => {
                                            // Convert RelationTriples to RelationPredictions
                                            for triple in &result.relations {
                                                if let Some(pred) =
                                                    RelationPrediction::from_triple_with_entities(
                                                        triple,
                                                        &result.entities,
                                                    )
                                                {
                                                    all_pred.push(pred);
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            eprintln!(
                                                "{} Relation extraction failed: {}",
                                                color("33", "warning:"),
                                                e
                                            );
                                            // Fall back to entity-pair heuristic for this document
                                            let entities =
                                                m.extract_entities(text, None).unwrap_or_default();
                                            all_pred.extend(create_entity_pair_relations(
                                                &entities,
                                                text,
                                                &relation_types_vec,
                                            ));
                                        }
                                    }
                                }
                            } else {
                                println!("{} Using entity-pair heuristic (GLiNER2 multitask not available)", color("33", "!"));
                                println!();

                                for doc in &gold_docs {
                                    let text = doc.text.as_str();
                                    all_gold.extend(doc.relations.clone());

                                    // Extract entities using NER
                                    let entities =
                                        m.extract_entities(text, None).unwrap_or_default();

                                    // Create relation predictions from entity pairs
                                    all_pred.extend(create_entity_pair_relations(
                                        &entities,
                                        text,
                                        &relation_types_vec,
                                    ));
                                }
                            }

                            let elapsed = start_time.elapsed();

                            // Evaluate relations
                            // Note: require_entity_type_match=false because entity types may differ
                            // (e.g., gold uses "person" but pred uses "Person", or "PER" vs "PERSON")
                            let config = RelationEvalConfig {
                                overlap_threshold: 0.5,
                                require_entity_type_match: false, // More lenient for evaluation
                                directed_relations: true,
                            };
                            let metrics = evaluate_relations(&all_gold, &all_pred, &config);

                            // Output results (human-readable by default)
                            println!();
                            println!("{}", color("1;36", "======================================================================="));
                            println!(
                                "  {}  model={}  time={:.1}s",
                                color("1;36", "RELATION EXTRACTION EVALUATION"),
                                model.name(),
                                elapsed.as_secs_f64()
                            );
                            println!("{}", color("1;36", "======================================================================="));
                            println!();
                            println!("{}", metrics.to_string_human(false)); // verbose=false for now
                            println!();
                        }
                    }
                }
            }
            #[cfg(not(feature = "eval"))]
            {
                let _ = (dataset, model, task);
                return Err("Dataset evaluation requires --features eval".to_string());
            }
        }
    }

    Ok(())
}
