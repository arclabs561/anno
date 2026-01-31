//! Dataset command - Work with NER datasets
//!
//! Provides CLI access to the dataset registry and loader.
//! - `list`: Browse available datasets from the registry
//! - `info`: Show metadata and statistics for a dataset
//! - `eval`: Evaluate a model on a dataset

use clap::{Parser, Subcommand};
#[cfg(feature = "eval-advanced")]
use itertools::Itertools;
#[cfg(feature = "eval")]
use std::fs;
#[cfg(feature = "eval")]
use std::path::{Path, PathBuf};
#[cfg(feature = "eval")]
use std::time::Instant;

use super::super::output::color;
use super::super::parser::{EvalTask, ModelBackend};

#[cfg(feature = "eval-advanced")]
use super::super::utils::create_entity_pair_relations;

#[cfg(feature = "eval")]
use super::super::utils::types_match_flexible;

#[cfg(feature = "eval")]
use crate::grounded::{
    render_eval_html_with_title, EvalComparison, EvalMatch, Location, Signal, SignalId,
};

#[cfg(all(feature = "eval", feature = "eval-advanced"))]
use crate::grounded::{render_document_html, GroundedDocument};

#[cfg(feature = "eval")]
use crate::eval::loader::DatasetId;
#[cfg(feature = "eval-advanced")]
use crate::eval::loader::LoadableDatasetId;

#[cfg(feature = "eval-advanced")]
use anno_core::CoreferenceResolver;

#[cfg(feature = "eval")]
#[derive(Debug)]
struct HtmlWorstCase {
    case_idx: usize,
    errors: usize,
    cmp: EvalComparison,
}

/// Work with NER datasets
#[derive(Parser, Debug)]
pub struct DatasetArgs {
    /// Action to perform
    #[command(subcommand)]
    pub action: DatasetAction,
}

/// Dataset subcommand actions.
#[derive(Subcommand, Debug)]
pub enum DatasetAction {
    /// List available datasets
    #[command(visible_alias = "ls")]
    List {
        /// Filter by task (ner, coref, re, el)
        #[arg(short, long)]
        task: Option<String>,

        /// Filter by domain (biomedical, news, social_media, etc.)
        #[arg(short, long)]
        domain: Option<String>,

        /// Show only loadable datasets (vs full registry)
        #[arg(long)]
        loadable: bool,

        /// Show detailed information
        #[arg(short, long)]
        verbose: bool,
    },

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

        /// Write an HTML error explorer (index + per-example eval pages).
        ///
        /// This is most useful for debugging datasets and evaluation assumptions:
        /// you can click into the worst examples and see gold vs predicted spans.
        #[arg(long)]
        html: bool,

        /// Output HTML path (required with `--html`).
        ///
        /// Per-example pages are written to `<output_stem>_files/`.
        #[arg(long, value_name = "PATH")]
        output: Option<String>,

        /// Max examples to include in the HTML explorer (worst by error count).
        #[arg(long, default_value_t = 50)]
        max_cases: usize,

        /// Minimum errors for an example to be included in the HTML explorer.
        ///
        /// Notes:
        /// - NER (`--task ner`): minimum number of errors (mismatches / FP / FN) in the example.
        /// - Coref (`--task coref`): minimum number of GOLD mentions in the document.
        /// - Relation (`--task relation`): minimum number of GOLD relations in the document.
        ///
        /// Prefer the task-specific flags (`--min-case-errors`, `--min-gold-mentions`,
        /// `--min-gold-relations`) to avoid ambiguity. If a task-specific flag is set, it
        /// takes precedence over this value for that task.
        #[arg(long, default_value_t = 1)]
        min_errors: usize,

        /// Minimum NER errors for an example to be included in the HTML explorer.
        ///
        /// Overrides `--min-errors` for `--task ner` if set.
        #[arg(long)]
        min_case_errors: Option<usize>,

        /// Minimum GOLD mentions for a doc to be included in the coref HTML explorer.
        ///
        /// Overrides `--min-errors` for `--task coref` if set.
        #[arg(long)]
        min_gold_mentions: Option<usize>,

        /// Minimum GOLD relations for a doc to be included in the relation HTML explorer.
        ///
        /// Overrides `--min-errors` for `--task relation` if set.
        #[arg(long)]
        min_gold_relations: Option<usize>,

        /// Coref only: use GOLD mentions as input to the resolver (oracle mention detection).
        ///
        /// This isolates clustering quality from mention detection quality.
        /// When set:
        /// - Coref evaluation clusters gold mentions instead of NER-extracted mentions.
        /// - Coref HTML explorer predicted view uses the same gold mention set.
        #[arg(long)]
        coref_oracle_mentions: bool,
    },

    /// Export annotations to brat/CoNLL/JSONL formats
    #[command(visible_alias = "ex")]
    Export(super::export::ExportArgs),

    /// Import annotations from brat/CoNLL/JSONL formats
    #[command(visible_alias = "im")]
    Import(super::import::ImportArgs),

    /// View entities with surrounding context for review
    #[command(visible_alias = "ctx")]
    Context(super::context::ContextArgs),

    /// Check dataset metadata consistency and issues
    #[command(visible_alias = "c")]
    Check {
        /// Show only issues (errors and warnings)
        #[arg(short, long)]
        issues_only: bool,

        /// Check specific dataset
        #[arg(short, long)]
        dataset: Option<String>,

        /// Fix issues automatically where possible
        #[arg(short, long)]
        fix: bool,
    },

    /// Check URL health for dataset download sources
    #[command(visible_alias = "ch")]
    CheckHealth {
        /// Check specific dataset (otherwise checks sample)
        #[arg(short, long)]
        dataset: Option<String>,

        /// Check all datasets with URLs (can take a while)
        #[arg(long)]
        all: bool,

        /// Do not fail the command for known-non-automatable datasets.
        ///
        /// This is useful for surveying the full registry (including gated / deprecated /
        /// registration-required sources) without turning the command into a failing wall of
        /// 404/401/403/TLS errors. In relaxed mode, only datasets that are expected to be
        /// automatable (`access_status.is_automatable()` + `is_automatable_download()`) will
        /// cause a non-zero exit.
        #[arg(long)]
        relaxed: bool,

        /// Show verbose breakdown of filtering (why datasets are included/excluded)
        #[arg(short, long)]
        verbose: bool,

        /// Number of parallel workers for URL checking
        #[arg(short, long, default_value = "5")]
        workers: usize,

        /// Request timeout in seconds
        #[arg(short, long, default_value = "10")]
        timeout: u64,
    },

    /// Download/cache a dataset (and validate it parses).
    ///
    /// This uses the same loader + cache directory as evaluation. It will download
    /// if missing (requires `--features eval-advanced` for network downloads).
    #[command(visible_alias = "dl")]
    Download {
        /// Dataset name (e.g., WikiGold, Wnut17, DocRED, CHisIEC)
        #[arg(short, long)]
        dataset: String,
    },

    /// Summarize dataset facets (language/domain/categories/tasks).
    ///
    /// Optionally compare against a matrix distribution report JSON (from `ANNO_MATRIX_DISTRIBUTION_REPORT`)
    /// to show which facets are being exercised ("touched") by the matrix.
    Facets {
        /// Path to a matrix distribution report JSON (optional).
        #[arg(long, value_name = "PATH")]
        touched_report: Option<String>,

        /// Print a gap report: facets that are under-touched vs their registry prevalence.
        #[arg(long)]
        gaps: bool,
    },
}

/// Execute the dataset command.
pub fn run(args: DatasetArgs) -> Result<(), String> {
    match args.action {
        DatasetAction::List {
            task,
            domain,
            loadable,
            verbose,
        } => {
            run_list(task, domain, loadable, verbose)?;
        }
        DatasetAction::Info { dataset } => {
            run_info(&dataset)?;
        }
        DatasetAction::Download { dataset } => {
            #[cfg(feature = "eval-advanced")]
            {
                run_download(&dataset)?;
            }
            #[cfg(not(feature = "eval-advanced"))]
            {
                let _ = dataset;
                return Err("Dataset download requires --features eval-advanced".to_string());
            }
        }
        DatasetAction::Facets {
            touched_report,
            gaps,
        } => {
            #[cfg(feature = "eval")]
            {
                run_facets(touched_report.as_deref(), gaps)?;
            }
            #[cfg(not(feature = "eval"))]
            {
                let _ = (touched_report, gaps);
                return Err("Dataset facets require --features eval".to_string());
            }
        }
        DatasetAction::Eval {
            dataset,
            model,
            task,
            html,
            output,
            max_cases,
            min_errors,
            min_case_errors,
            min_gold_mentions,
            min_gold_relations,
            coref_oracle_mentions,
        } => {
            #[cfg(feature = "eval")]
            {
                let m = model.create_model()?;
                let ner_min_errors = min_case_errors.unwrap_or(min_errors);
                #[cfg(feature = "eval-advanced")]
                let coref_min_mentions = min_gold_mentions.unwrap_or(min_errors);
                #[cfg(feature = "eval-advanced")]
                let rel_min_gold = min_gold_relations.unwrap_or(min_errors);

                // Avoid unused warnings in non-`eval-advanced` builds.
                #[cfg(not(feature = "eval-advanced"))]
                let _ = (min_gold_mentions, min_gold_relations, coref_oracle_mentions);

                let (name, test_cases) = if dataset == "synthetic" {
                    ("synthetic".to_string(), synthetic_ner_test_cases())
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
                        let loadable = LoadableDatasetId::try_from(dataset_id)
                            .map_err(|e| format!("Dataset is not loadable: {}", e))?;
                        let ds = loader
                            .load_or_download(loadable)
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

                // Parse dataset ID once for reuse (avoid duplicate parsing)
                // Store Result to preserve error message if parsing fails
                #[cfg(feature = "eval-advanced")]
                let parsed_dataset_result: Result<DatasetId, String> = if dataset != "synthetic" {
                    dataset
                        .parse::<DatasetId>()
                        .map_err(|e| format!("Invalid dataset '{}': {}", dataset, e))
                } else {
                    Err("synthetic dataset".to_string()) // Not a real error, just indicates synthetic
                };

                // Route to appropriate evaluation based on task
                match task {
                    EvalTask::Ner => {
                        println!();
                        println!("Evaluating {} on {} dataset (NER)...", model.name(), name);
                        println!("  Sentences: {}", test_cases.len());
                        println!();

                        // Per-entity-type tracking: (gold, pred, correct)
                        let mut per_type_stats: std::collections::HashMap<
                            String,
                            (usize, usize, usize),
                        > = std::collections::HashMap::new();
                        let mut total_gold = 0;
                        let mut total_pred = 0;
                        let mut total_correct = 0;

                        let html_output_path: Option<PathBuf> = if html {
                            Some(PathBuf::from(output.as_deref().ok_or_else(|| {
                                "--html requires --output PATH".to_string()
                            })?))
                        } else {
                            None
                        };
                        if html_output_path.is_some() && max_cases == 0 {
                            return Err("--max-cases must be > 0 when --html is set".to_string());
                        }

                        let mut worst_cases: Vec<HtmlWorstCase> = Vec::new();

                        let start_time = Instant::now();

                        // Validate gold annotations before evaluation (warn but continue)
                        #[cfg(feature = "eval-advanced")]
                        {
                            use crate::eval::validation::validate_ground_truth_entities;
                            let mut total_warnings = 0;
                            for (text, gold) in &test_cases {
                                let validation = validate_ground_truth_entities(text, gold, false);
                                if !validation.is_valid {
                                    eprintln!(
                                        "{} Invalid gold annotations: {}",
                                        color("33", "warning:"),
                                        validation.errors.join("; ")
                                    );
                                }
                                total_warnings += validation.warnings.len();
                            }
                            if total_warnings > 0 {
                                eprintln!(
                                    "{} {} validation warnings in gold annotations",
                                    color("33", "warning:"),
                                    total_warnings
                                );
                            }
                        }

                        for (case_idx, (text, gold)) in test_cases.iter().enumerate() {
                            let entities = m.extract_entities(text, None).unwrap_or_default();

                            total_gold += gold.len();
                            total_pred += entities.len();

                            if html_output_path.is_some() {
                                let gold_signals: Vec<Signal<Location>> = gold
                                    .iter()
                                    .enumerate()
                                    .map(|(i, g)| {
                                        Signal::new(
                                            SignalId::new(i as u64),
                                            Location::text(g.start, g.end),
                                            g.text.as_str(),
                                            g.original_label.as_str(),
                                            1.0,
                                        )
                                    })
                                    .collect();

                                let pred_signals: Vec<Signal<Location>> = entities
                                    .iter()
                                    .enumerate()
                                    .map(|(i, e)| {
                                        Signal::new(
                                            SignalId::new(i as u64),
                                            Location::text(e.start, e.end),
                                            e.text.as_str(),
                                            e.entity_type.as_label(),
                                            e.confidence as f32,
                                        )
                                    })
                                    .collect();

                                let cmp = EvalComparison::compare(text, gold_signals, pred_signals);
                                let errors = cmp.error_count();
                                if errors >= ner_min_errors {
                                    worst_cases.push(HtmlWorstCase {
                                        case_idx,
                                        errors,
                                        cmp,
                                    });
                                    // Keep memory bounded on large datasets (retain only the worst cases).
                                    if worst_cases.len() > max_cases.saturating_mul(5) {
                                        worst_cases.sort_by(|a, b| {
                                            b.errors
                                                .cmp(&a.errors)
                                                .then_with(|| a.case_idx.cmp(&b.case_idx))
                                        });
                                        worst_cases.truncate(max_cases);
                                    }
                                }
                            }

                            // Track which predictions have been matched to prevent double-counting
                            let mut matched_pred = vec![false; entities.len()];

                            for gold_entity in gold {
                                let gold_type =
                                    anno_core::EntityType::from_label(&gold_entity.original_label);

                                // Use canonical normalization for consistent grouping
                                let gold_type_key =
                                    crate::cli::utils::normalize_entity_type_canonical(
                                        gold_type.as_label(),
                                    );

                                // Track per-type gold counts using normalized type
                                per_type_stats
                                    .entry(gold_type_key.clone())
                                    .or_insert((0, 0, 0))
                                    .0 += 1;

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
                                    // Track per-type correct count
                                    per_type_stats.entry(gold_type_key).or_insert((0, 0, 0)).2 += 1;
                                }
                            }

                            // Track per-type pred counts
                            // Use canonical normalization so PER/PERSON etc. are grouped together
                            for e in &entities {
                                let raw_type = e.entity_type.as_label().to_uppercase();
                                // Normalize pred type to match gold type normalization
                                let type_key =
                                    crate::cli::utils::normalize_entity_type_canonical(&raw_type);
                                per_type_stats.entry(type_key).or_insert((0, 0, 0)).1 += 1;
                            }
                        }

                        let elapsed = start_time.elapsed();

                        // Handle edge case: if no gold entities and no predictions, that's perfect
                        let (p, r, f1) = if total_gold == 0 && total_pred == 0 {
                            (1.0, 1.0, 1.0)
                        } else {
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
                            (p, r, f1)
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

                        // Per-entity-type breakdown (sorted by gold count descending)
                        let mut type_entries: Vec<_> = per_type_stats.iter().collect();
                        type_entries.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));

                        if !type_entries.is_empty() {
                            println!();
                            println!("Per-type breakdown:");
                            for (type_name, (gold_count, pred_count, correct_count)) in type_entries
                            {
                                if *gold_count == 0 && *pred_count == 0 {
                                    continue;
                                }
                                let type_p = if *pred_count > 0 {
                                    *correct_count as f64 / *pred_count as f64
                                } else {
                                    0.0
                                };
                                let type_r = if *gold_count > 0 {
                                    *correct_count as f64 / *gold_count as f64
                                } else {
                                    0.0
                                };
                                let type_f1 = if type_p + type_r > 0.0 {
                                    2.0 * type_p * type_r / (type_p + type_r)
                                } else {
                                    0.0
                                };
                                println!(
                                    "  {:12} F1={:5.1}%  P={:5.1}%  R={:5.1}%  [gold={} pred={} correct={}]",
                                    type_name,
                                    type_f1 * 100.0,
                                    type_p * 100.0,
                                    type_r * 100.0,
                                    gold_count,
                                    pred_count,
                                    correct_count
                                );
                            }
                        }

                        let ms_per_sent = if !test_cases.is_empty() {
                            elapsed.as_secs_f64() * 1000.0 / test_cases.len() as f64
                        } else {
                            0.0
                        };
                        println!();
                        println!(
                            "  Time: {:.1}s ({:.1}ms/sent)",
                            elapsed.as_secs_f64(),
                            ms_per_sent
                        );

                        if let Some(output_path) = html_output_path {
                            // Final selection + stable ordering: worst errors first.
                            worst_cases.sort_by(|a, b| {
                                b.errors
                                    .cmp(&a.errors)
                                    .then_with(|| a.case_idx.cmp(&b.case_idx))
                            });
                            worst_cases.truncate(max_cases);

                            write_ner_error_explorer_html(
                                output_path.as_path(),
                                &name,
                                model.name(),
                                test_cases.len(),
                                total_gold,
                                total_pred,
                                total_correct,
                                &worst_cases,
                            )?;
                            println!(
                                "{} HTML written to: {}",
                                color("32", "ok:"),
                                output_path.display()
                            );
                        }
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

                            let html_output_path: Option<PathBuf> = if html {
                                Some(PathBuf::from(output.as_deref().ok_or_else(|| {
                                    "--html requires --output PATH".to_string()
                                })?))
                            } else {
                                None
                            };
                            if html_output_path.is_some() && max_cases == 0 {
                                return Err(
                                    "--max-cases must be > 0 when --html is set".to_string()
                                );
                            }

                            if dataset == "synthetic" {
                                return Err("Coreference evaluation requires a real dataset (e.g., gap, preco, litbank)".to_string());
                            }

                            // Reuse parsed result (preserves original error message)
                            let dataset_id: DatasetId = parsed_dataset_result.clone()?;

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

                                // Mentions source: model NER (default) or oracle gold mentions.
                                let entities = if coref_oracle_mentions {
                                    coref_doc_to_oracle_mentions(doc)
                                } else {
                                    m.extract_entities(text, None).unwrap_or_default()
                                };

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

                            if let Some(output_path) = html_output_path {
                                // Select worst docs (lowest CoNLL F1) with a minimum gold mention count.
                                let mut scored: Vec<(usize, f64)> = results
                                    .per_document
                                    .iter()
                                    .enumerate()
                                    .filter_map(|(i, ev)| {
                                        let doc = gold_docs.get(i)?;
                                        if doc.mention_count() < coref_min_mentions {
                                            return None;
                                        }
                                        Some((i, ev.conll_f1))
                                    })
                                    .collect();
                                scored.sort_by(|a, b| {
                                    a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                                });
                                let selected: Vec<usize> =
                                    scored.into_iter().take(max_cases).map(|(i, _)| i).collect();

                                write_coref_error_explorer_html(
                                    output_path.as_path(),
                                    &name,
                                    model.name(),
                                    resolver.name(),
                                    &gold_docs,
                                    &results.per_document,
                                    &selected,
                                    m.as_ref(),
                                    &resolver,
                                    coref_oracle_mentions,
                                )?;
                                println!(
                                    "{} HTML written to: {}",
                                    color("32", "ok:"),
                                    output_path.display()
                                );
                            }
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

                            let html_output_path: Option<PathBuf> = if html {
                                Some(PathBuf::from(output.as_deref().ok_or_else(|| {
                                    "--html requires --output PATH".to_string()
                                })?))
                            } else {
                                None
                            };
                            if html_output_path.is_some() && max_cases == 0 {
                                return Err(
                                    "--max-cases must be > 0 when --html is set".to_string()
                                );
                            }

                            if dataset == "synthetic" {
                                return Err("Relation extraction evaluation requires a real dataset (e.g., docred, retacred)".to_string());
                            }

                            // Reuse parsed result (preserves original error message)
                            let dataset_id: DatasetId = parsed_dataset_result.clone()?;

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
                                entity_types.iter().map(|s| s.as_str()).collect_vec();
                            let relation_types_vec: Vec<&str> =
                                relation_types.iter().map(|s| s.as_str()).collect_vec();

                            println!("  Entity types: {}", entity_types_vec.join(", "));
                            println!(
                                "  Relation types: {} ({} total)",
                                relation_types_vec.len(),
                                relation_types_vec.iter().take(5).join(", ")
                            );
                            println!();

                            // Use the *requested* backend for relation extraction if it supports it.
                            let use_relation_extractor: Option<(
                                String,
                                Box<dyn RelationExtractor>,
                            )> = match model {
                                #[cfg(feature = "onnx")]
                                ModelBackend::Gliner2 => {
                                    crate::backends::gliner2::GLiNER2Onnx::from_pretrained(
                                        "onnx-community/gliner-multitask-large-v0.5",
                                    )
                                    .ok()
                                    .map(|m| {
                                        (
                                            "gliner2".to_string(),
                                            Box::new(m) as Box<dyn RelationExtractor>,
                                        )
                                    })
                                }
                                ModelBackend::Tplinker => {
                                    crate::backends::tplinker::TPLinker::new().ok().map(|m| {
                                        (
                                            "tplinker".to_string(),
                                            Box::new(m) as Box<dyn RelationExtractor>,
                                        )
                                    })
                                }
                                _ => None,
                            };

                            let mut all_gold = Vec::new();
                            let mut all_pred = Vec::new();
                            let mut pred_by_doc: Vec<Vec<RelationPrediction>> =
                                Vec::with_capacity(gold_docs.len());
                            let start_time = Instant::now();

                            if let Some((ref extractor_name, ref rel_extractor)) =
                                use_relation_extractor
                            {
                                println!(
                                    "{} Using {} RelationExtractor",
                                    color("32", "[OK]"),
                                    extractor_name
                                );
                                println!(
                                    "  Note: This uses heuristics, not a neural relation model."
                                );
                                println!();

                                for (doc_idx, doc) in gold_docs.iter().enumerate() {
                                    let text = doc.text.as_str();
                                    all_gold.extend(doc.relations.clone());
                                    let mut pred_this: Vec<RelationPrediction> = Vec::new();

                                    // Use RelationExtractor
                                    match rel_extractor.extract_with_relations(
                                        text,
                                        &entity_types_vec,
                                        &relation_types_vec,
                                        0.5,
                                    ) {
                                        Ok(result) => {
                                            // If the model produced no entities (common for CHisIEC with an English model),
                                            // fall back to an “oracle entities” baseline: use gold entity spans/types, and
                                            // run lightweight relation heuristics over those entities.
                                            let allow_oracle_entities =
                                                std::env::var("ANNO_RELATION_ORACLE_ENTITIES")
                                                    .ok()
                                                    .map(|v| {
                                                        let v = v.trim().to_lowercase();
                                                        v == "1"
                                                            || v == "true"
                                                            || v == "yes"
                                                            || v == "y"
                                                    })
                                                    .unwrap_or(true);

                                            if dataset_id == DatasetId::CHisIEC
                                                && matches!(model, ModelBackend::Gliner2)
                                                && allow_oracle_entities
                                                && result.entities.is_empty()
                                                && !doc.relations.is_empty()
                                            {
                                                use crate::backends::inference::{
                                                    extract_relation_triples,
                                                    RelationExtractionConfig, SemanticRegistry,
                                                };
                                                use crate::{Entity as PredEntity, EntityType};
                                                use std::collections::BTreeMap;

                                                if doc_idx == 0 {
                                                    eprintln!(
                                                        "{} CHisIEC fallback: using gold entity spans as oracle candidates (NER produced 0 entities)",
                                                        color("33", "note:")
                                                    );
                                                }

                                                let mut by_key: BTreeMap<
                                                    (usize, usize, String, String),
                                                    PredEntity,
                                                > = BTreeMap::new();
                                                for r in &doc.relations {
                                                    for (ty, span, txt) in [
                                                        (&r.head_type, r.head_span, &r.head_text),
                                                        (&r.tail_type, r.tail_span, &r.tail_text),
                                                    ] {
                                                        let (start, end) = span;
                                                        let text_fallback: String =
                                                            if !txt.is_empty() {
                                                                txt.clone()
                                                            } else {
                                                                text.chars()
                                                                    .skip(start)
                                                                    .take(end.saturating_sub(start))
                                                                    .collect()
                                                            };
                                                        let ent = PredEntity::new(
                                                            text_fallback.clone(),
                                                            EntityType::from_label(ty),
                                                            start,
                                                            end,
                                                            1.0,
                                                        );
                                                        by_key
                                                            .entry((
                                                                start,
                                                                end,
                                                                ty.clone(),
                                                                text_fallback,
                                                            ))
                                                            .or_insert(ent);
                                                    }
                                                }
                                                let oracle_entities: Vec<PredEntity> =
                                                    by_key.into_values().collect();

                                                let mut builder = SemanticRegistry::builder();
                                                for rt in &relation_types_vec {
                                                    builder = builder.add_relation(rt, rt);
                                                }
                                                let registry = builder.build_placeholder(1);
                                                let rel_cfg = RelationExtractionConfig {
                                                    threshold: 0.5,
                                                    max_span_distance: 120,
                                                    extract_triggers: false,
                                                };
                                                let triples = extract_relation_triples(
                                                    &oracle_entities,
                                                    text,
                                                    &registry,
                                                    &rel_cfg,
                                                );
                                                for t in &triples {
                                                    if let (Some(head), Some(tail)) = (
                                                        oracle_entities.get(t.head_idx),
                                                        oracle_entities.get(t.tail_idx),
                                                    ) {
                                                        let pred = RelationPrediction {
                                                            head_span: (head.start, head.end),
                                                            head_type: head
                                                                .entity_type
                                                                .as_label()
                                                                .to_string(),
                                                            tail_span: (tail.start, tail.end),
                                                            tail_type: tail
                                                                .entity_type
                                                                .as_label()
                                                                .to_string(),
                                                            relation_type: t.relation_type.clone(),
                                                            confidence: t.confidence,
                                                        };
                                                        all_pred.push(pred.clone());
                                                        pred_this.push(pred);
                                                    }
                                                }
                                            } else {
                                                // Convert RelationTriples to RelationPredictions
                                                for triple in &result.relations {
                                                    if let Some(pred) =
                                                        RelationPrediction::from_triple_with_entities(
                                                            triple,
                                                            &result.entities,
                                                        )
                                                    {
                                                        all_pred.push(pred.clone());
                                                        pred_this.push(pred);
                                                    }
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
                                            let fallback = create_entity_pair_relations(
                                                &entities,
                                                text,
                                                &relation_types_vec,
                                            );
                                            all_pred.extend(fallback.iter().cloned());
                                            pred_this.extend(fallback);
                                        }
                                    }
                                    pred_by_doc.push(pred_this);
                                }
                            } else {
                                println!(
                                    "{} Using entity-pair heuristic (no RelationExtractor for this backend)",
                                    color("33", "!")
                                );
                                println!();

                                for doc in &gold_docs {
                                    let text = doc.text.as_str();
                                    all_gold.extend(doc.relations.clone());

                                    // Extract entities using NER
                                    let entities =
                                        m.extract_entities(text, None).unwrap_or_default();

                                    // Create relation predictions from entity pairs
                                    let pred_this = create_entity_pair_relations(
                                        &entities,
                                        text,
                                        &relation_types_vec,
                                    );
                                    all_pred.extend(pred_this.iter().cloned());
                                    pred_by_doc.push(pred_this);
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

                            if let Some(output_path) = html_output_path {
                                // Per-doc metrics + worst-doc selection
                                let mut per_doc_metrics: Vec<
                                    crate::eval::relation::RelationMetrics,
                                > = Vec::with_capacity(gold_docs.len());
                                for (i, doc) in gold_docs.iter().enumerate() {
                                    let pred = pred_by_doc.get(i).cloned().unwrap_or_default();
                                    per_doc_metrics.push(evaluate_relations(
                                        &doc.relations,
                                        &pred,
                                        &config,
                                    ));
                                }

                                // Filter tiny docs (few gold relations). Task-specific flag:
                                // `--min-gold-relations` (or legacy `--min-errors`).
                                let mut scored: Vec<(usize, f64, usize)> = per_doc_metrics
                                    .iter()
                                    .enumerate()
                                    .filter_map(|(i, m)| {
                                        let gold_n = gold_docs.get(i)?.relations.len();
                                        if gold_n < rel_min_gold {
                                            return None;
                                        }
                                        // Score by strict f1 ascending; tie-break by "strict errors" descending.
                                        let strict_errors = (m.num_gold - m.strict_matches)
                                            + (m.num_predicted - m.strict_matches);
                                        Some((i, m.strict_f1, strict_errors))
                                    })
                                    .collect();
                                scored.sort_by(|a, b| {
                                    a.1.partial_cmp(&b.1)
                                        .unwrap_or(std::cmp::Ordering::Equal)
                                        .then_with(|| b.2.cmp(&a.2))
                                });
                                let selected: Vec<usize> = scored
                                    .into_iter()
                                    .take(max_cases)
                                    .map(|(i, _, _)| i)
                                    .collect();

                                write_relation_error_explorer_html(
                                    output_path.as_path(),
                                    &name,
                                    model.name(),
                                    &gold_docs,
                                    &pred_by_doc,
                                    &per_doc_metrics,
                                    &selected,
                                )?;
                                println!(
                                    "{} HTML written to: {}",
                                    color("32", "ok:"),
                                    output_path.display()
                                );
                            }
                        }
                    }
                }
            }
            #[cfg(not(feature = "eval"))]
            {
                let _ = (
                    dataset,
                    model,
                    task,
                    html,
                    output,
                    max_cases,
                    min_errors,
                    min_case_errors,
                    min_gold_mentions,
                    min_gold_relations,
                    coref_oracle_mentions,
                );
                return Err("Dataset evaluation requires --features eval".to_string());
            }
        }
        DatasetAction::Export(args) => {
            super::export::run(args)?;
        }
        DatasetAction::Import(args) => {
            super::import::run(args)?;
        }
        DatasetAction::Context(args) => {
            super::context::run(args)?;
        }
        DatasetAction::Check {
            issues_only,
            dataset,
            fix,
        } => {
            run_check(issues_only, dataset.as_deref(), fix)?;
        }
        DatasetAction::CheckHealth {
            dataset,
            all,
            relaxed,
            verbose,
            workers,
            timeout,
        } => {
            run_check_health(dataset.as_deref(), all, relaxed, verbose, workers, timeout)?;
        }
    }

    Ok(())
}

#[cfg(feature = "eval-advanced")]
fn run_download(dataset: &str) -> Result<(), String> {
    use crate::eval::loader::{DatasetId, DatasetLoader, LoadableDatasetId};

    let dataset_id: DatasetId = dataset
        .parse::<DatasetId>()
        .map_err(|e| format!("Invalid dataset '{}': {}", dataset, e))?;

    let loader =
        DatasetLoader::new().map_err(|e| format!("Failed to init dataset loader: {}", e))?;

    if dataset_id.is_relation_extraction() {
        let docs = loader
            .load_or_download_relation(dataset_id)
            .map_err(|e| format!("Failed to load relation dataset: {}", e))?;
        println!(
            "{} cached relation dataset {} (documents={})",
            color("32", "ok:"),
            dataset_id.name(),
            docs.len()
        );
        return Ok(());
    }

    if dataset_id.is_coreference() {
        let docs = loader
            .load_or_download_coref(dataset_id)
            .map_err(|e| format!("Failed to load coref dataset: {}", e))?;
        println!(
            "{} cached coref dataset {} (documents={})",
            color("32", "ok:"),
            dataset_id.name(),
            docs.len()
        );
        return Ok(());
    }

    let loadable = LoadableDatasetId::try_from(dataset_id)
        .map_err(|e| format!("Dataset is not loadable: {}", e))?;
    let ds = loader
        .load_or_download(loadable)
        .map_err(|e| format!("Failed to load dataset: {}", e))?;
    println!(
        "{} cached dataset {} (sentences={})",
        color("32", "ok:"),
        dataset_id.name(),
        ds.sentences.len()
    );
    Ok(())
}

#[cfg(feature = "eval")]
fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

#[cfg(feature = "eval")]
fn preview_text(s: &str, max_chars: usize) -> String {
    let mut out: String = s.chars().take(max_chars).collect();
    out = out.replace(['\n', '\r', '\t'], " ");
    // Keep it simple: collapse repeated spaces in a tiny loop.
    while out.contains("  ") {
        out = out.replace("  ", " ");
    }
    out.trim().to_string()
}

#[cfg(feature = "eval")]
fn synthetic_ner_test_cases() -> Vec<(String, Vec<crate::eval::GoldEntity>)> {
    use anno_core::EntityType;

    fn find_char_span(text: &str, needle: &str) -> (usize, usize) {
        let start_byte = text.find(needle).unwrap_or_else(|| {
            panic!("synthetic_ner_test_cases bug: needle not found: {needle:?}")
        });
        let end_byte = start_byte + needle.len();
        let start = text[..start_byte].chars().count();
        let end = text[..end_byte].chars().count();
        (start, end)
    }

    fn entity_type_for_label(label: &str) -> EntityType {
        match label {
            "PER" | "PERSON" => EntityType::Person,
            "ORG" | "ORGANIZATION" => EntityType::Organization,
            "LOC" | "LOCATION" | "GPE" => EntityType::Location,
            other => EntityType::Other(other.to_string()),
        }
    }

    fn ge(text: &str, needle: &str, label: &str) -> crate::eval::GoldEntity {
        let (start, end) = find_char_span(text, needle);
        crate::eval::GoldEntity {
            text: needle.to_string(),
            original_label: label.to_string(),
            entity_type: entity_type_for_label(label),
            start,
            end,
        }
    }

    let mut cases: Vec<(String, Vec<crate::eval::GoldEntity>)> = Vec::new();

    // Latin
    let t = "Marie Curie won the Nobel Prize.";
    cases.push((
        t.to_string(),
        vec![ge(t, "Marie Curie", "PER"), ge(t, "Nobel Prize", "MISC")],
    ));

    // CJK (no spaces)
    let t = "習近平在北京會見了普京。";
    cases.push((
        t.to_string(),
        vec![
            ge(t, "習近平", "PER"),
            ge(t, "北京", "LOC"),
            ge(t, "普京", "PER"),
        ],
    ));

    // Arabic (RTL)
    let t = "التقى محمد بن سلمان بالرئيس في الرياض";
    cases.push((
        t.to_string(),
        vec![ge(t, "محمد بن سلمان", "PER"), ge(t, "الرياض", "LOC")],
    ));

    // Cyrillic
    let t = "Путин встретился с Си Цзиньпином в Москве.";
    cases.push((
        t.to_string(),
        vec![
            ge(t, "Путин", "PER"),
            ge(t, "Си Цзиньпином", "PER"),
            ge(t, "Москве", "LOC"),
        ],
    ));

    // Devanagari
    let t = "डॉ. शर्मा ने दिल्ली में सम्मेलन में भाषण दिया।";
    cases.push((
        t.to_string(),
        vec![ge(t, "शर्मा", "PER"), ge(t, "दिल्ली", "LOC")],
    ));

    // Mixed / code-switching
    let t = "Dr. 田中 presented at MIT in 東京.";
    cases.push((
        t.to_string(),
        vec![
            ge(t, "田中", "PER"),
            ge(t, "MIT", "ORG"),
            ge(t, "東京", "LOC"),
        ],
    ));

    // Diacritics
    let t = "François Müller and José García met in São Paulo.";
    cases.push((
        t.to_string(),
        vec![
            ge(t, "François Müller", "PER"),
            ge(t, "José García", "PER"),
            ge(t, "São Paulo", "LOC"),
        ],
    ));

    // Special characters
    let t = "Contact john@example.com for help.";
    cases.push((t.to_string(), vec![ge(t, "john@example.com", "EMAIL")]));

    cases
}

#[cfg(feature = "eval")]
#[allow(clippy::too_many_arguments)]
fn write_ner_error_explorer_html(
    output_path: &Path,
    dataset_name: &str,
    model_name: &str,
    total_sentences: usize,
    total_gold: usize,
    total_pred: usize,
    total_correct: usize,
    worst_cases: &[HtmlWorstCase],
) -> Result<(), String> {
    let parent = output_path.parent().unwrap_or_else(|| Path::new("."));
    let stem = output_path
        .file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("dataset_eval");
    let files_dir = parent.join(format!("{stem}_files"));
    fs::create_dir_all(&files_dir)
        .map_err(|e| format!("Failed to create {:?}: {}", files_dir, e))?;

    for case in worst_cases {
        let case_filename = format!("case_{:06}.html", case.case_idx);
        let case_path = files_dir.join(&case_filename);
        let title = format!("{model_name} — {dataset_name} — case {}", case.case_idx);
        let case_html = render_eval_html_with_title(&case.cmp, &title);
        fs::write(&case_path, case_html)
            .map_err(|e| format!("Failed to write {:?}: {}", case_path, e))?;
    }

    let mut index = String::new();
    index.push_str("<!DOCTYPE html><html><head><meta charset=\"UTF-8\"><meta name=\"color-scheme\" content=\"dark light\">");
    index.push_str(&format!(
        "<title>{}</title>",
        html_escape(&format!(
            "dataset eval explorer — {model_name} — {dataset_name}"
        ))
    ));
    index.push_str(
        r#"<style>
:root{color-scheme:light dark;--bg:#0a0a0a;--text:#b0b0b0;--text-strong:#fff;--muted:#777;--border:#222;--border-strong:#333;--hover:#111;--input-bg:#080808;--link:#9ad;--code:#bbb}
@media (prefers-color-scheme: light){:root{--bg:#fff;--text:#222;--text-strong:#000;--muted:#555;--border:#d6d6d6;--border-strong:#c6c6c6;--hover:#f0f0f0;--input-bg:#fff;--link:#06c;--code:#333}}
html[data-theme='dark']{--bg:#0a0a0a;--text:#b0b0b0;--text-strong:#fff;--muted:#777;--border:#222;--border-strong:#333;--hover:#111;--input-bg:#080808;--link:#9ad;--code:#bbb}
html[data-theme='light']{--bg:#fff;--text:#222;--text-strong:#000;--muted:#555;--border:#d6d6d6;--border-strong:#c6c6c6;--hover:#f0f0f0;--input-bg:#fff;--link:#06c;--code:#333}
*{box-sizing:border-box;margin:0;padding:0}
body{font:12px/1.4 monospace;background:var(--bg);color:var(--text);padding:12px}
h1{font-size:14px;color:var(--text-strong);font-weight:normal;border-bottom:1px solid var(--border-strong);padding:4px 0;margin:0 0 12px}
.meta{color:var(--muted);margin:0 0 12px}
.meta b{color:var(--text);font-weight:normal}
.row{display:flex;gap:12px;align-items:center;margin:0 0 10px}
input{flex:1;background:var(--input-bg);border:1px solid var(--border);color:var(--text);padding:6px 8px}
.count{color:var(--muted)}
table{width:100%;border-collapse:collapse;font-size:11px}
th,td{padding:4px 8px;text-align:left;border:1px solid var(--border);vertical-align:top}
th{background:var(--hover);color:var(--muted);font-weight:normal;text-transform:uppercase;font-size:10px}
tr:hover{background:var(--hover)}
a{color:var(--link);text-decoration:none}
a:hover{text-decoration:underline}
.num{text-align:right;font-variant-numeric:tabular-nums}
code{color:var(--code)}
.toggle{cursor:pointer;user-select:none;color:var(--muted);border:1px solid var(--border);background:var(--bg);padding:2px 6px;font-size:10px}
</style></head><body>"#,
    );
    index.push_str(&format!(
        "<div class=\"row\" style=\"justify-content:space-between\"><h1>dataset eval explorer</h1><span class=\"toggle\" id=\"theme-toggle\" title=\"toggle theme (auto → dark → light)\">theme: auto</span></div><div class=\"meta\"><b>model</b> {} &nbsp; <b>dataset</b> {} &nbsp; <b>sentences</b> {} &nbsp; <b>gold</b> {} &nbsp; <b>pred</b> {} &nbsp; <b>correct</b> {}</div>",
        html_escape(model_name),
        html_escape(dataset_name),
        total_sentences,
        total_gold,
        total_pred,
        total_correct
    ));
    index.push_str(
        r#"<div class="row">
  <input id="case-filter" placeholder="filter (case id, label, text…)" />
  <div id="case-count" class="count"></div>
</div>
<table id="case-table"><thead><tr>
  <th>case</th><th class="num">errors</th><th class="num">f1</th><th class="num">gold</th><th class="num">pred</th><th class="num">✓</th><th>text</th>
</tr></thead><tbody>"#,
    );

    for case in worst_cases {
        let first_error_mid = case
            .cmp
            .matches
            .iter()
            .position(|m| !matches!(m, EvalMatch::Correct { .. }))
            .unwrap_or(0);
        let rel = format!(
            "{stem}_files/case_{:06}.html#M{}",
            case.case_idx, first_error_mid
        );
        let preview = preview_text(&case.cmp.text, 180);
        index.push_str(&format!(
            "<tr data-hay=\"{hay}\"><td><a target=\"_blank\" rel=\"noopener\" href=\"{href}\">case {case_id}</a></td><td class=\"num\">{errors}</td><td class=\"num\">{f1:.1}%</td><td class=\"num\">{gold}</td><td class=\"num\">{pred}</td><td class=\"num\">{ok}</td><td><code>{text}</code></td></tr>",
            hay = html_escape(&format!(
                "case {} errors {} {} {}",
                case.case_idx, case.errors, preview, dataset_name
            ))
            .to_lowercase(),
            href = html_escape(&rel),
            case_id = case.case_idx,
            errors = case.errors,
            f1 = case.cmp.f1() * 100.0,
            gold = case.cmp.gold.len(),
            pred = case.cmp.predicted.len(),
            ok = case.cmp.correct_count(),
            text = html_escape(&preview),
        ));
    }

    index.push_str(
        r##"</tbody></table>
<script>
(() => {
  // Theme toggle: auto → dark → light (persisted).
  const themeBtn = document.getElementById("theme-toggle");
  const themeKey = "anno-theme";
  const applyTheme = (theme) => {
    const t = theme || "auto";
    if (t === "auto") {
      delete document.documentElement.dataset.theme;
    } else {
      document.documentElement.dataset.theme = t;
    }
    if (themeBtn) themeBtn.textContent = `theme: ${t}`;
  };
  const readTheme = () => {
    try { return localStorage.getItem(themeKey) || "auto"; } catch (_) { return "auto"; }
  };
  const writeTheme = (t) => {
    try { localStorage.setItem(themeKey, t); } catch (_) { /* ignore */ }
  };
  applyTheme(readTheme());
  if (themeBtn) {
    themeBtn.addEventListener("click", () => {
      const cur = readTheme();
      const next = cur === "auto" ? "dark" : (cur === "dark" ? "light" : "auto");
      writeTheme(next);
      applyTheme(next);
    });
  }

  const input = document.getElementById("case-filter");
  const rows = Array.from(document.querySelectorAll("#case-table tbody tr"));
  const count = document.getElementById("case-count");
  function update() {
    const q = (input.value || "").toLowerCase().trim();
    let shown = 0;
    for (const tr of rows) {
      const hay = (tr.dataset.hay || "");
      const show = !q || hay.includes(q);
      tr.style.display = show ? "" : "none";
      if (show) shown++;
    }
    count.textContent = `${shown} shown / ${rows.length} total`;
  }
  input.addEventListener("input", update);
  update();
})();
</script>
</body></html>"##,
    );

    fs::write(output_path, index)
        .map_err(|e| format!("Failed to write {:?}: {}", output_path, e))?;
    Ok(())
}

#[cfg(feature = "eval-advanced")]
fn coref_doc_to_gold_entities(doc: &crate::eval::coref::CorefDocument) -> Vec<anno_core::Entity> {
    use anno_core::{Entity, EntityType};

    let mut entities: Vec<Entity> = Vec::new();
    let mut next_cluster = anno_core::types::CanonicalId::ZERO;

    for chain in &doc.chains {
        let cid = chain.cluster_id.unwrap_or_else(|| {
            let id = next_cluster;
            next_cluster += 1;
            id
        });
        let label = chain
            .entity_type
            .as_deref()
            .or_else(|| {
                chain
                    .mentions
                    .first()
                    .and_then(|m| m.entity_type.as_deref())
            })
            .unwrap_or("COREF");

        for m in &chain.mentions {
            let mut e = Entity::new(
                m.text.clone(),
                EntityType::Other(label.to_string()),
                m.start,
                m.end,
                1.0,
            );
            e.canonical_id = Some(cid);
            entities.push(e);
        }
    }

    entities
}

#[cfg(feature = "eval-advanced")]
fn coref_doc_to_oracle_mentions(doc: &crate::eval::coref::CorefDocument) -> Vec<anno_core::Entity> {
    use anno_core::{Entity, EntityType};

    let mut entities: Vec<Entity> = Vec::new();

    for chain in &doc.chains {
        let label = chain
            .entity_type
            .as_deref()
            .or_else(|| {
                chain
                    .mentions
                    .first()
                    .and_then(|m| m.entity_type.as_deref())
            })
            .unwrap_or("COREF");

        for m in &chain.mentions {
            let e = Entity::new(
                m.text.clone(),
                EntityType::Other(label.to_string()),
                m.start,
                m.end,
                1.0,
            );
            entities.push(e);
        }
    }

    entities
}

#[cfg(feature = "eval-advanced")]
#[allow(clippy::too_many_arguments)]
fn write_coref_error_explorer_html(
    output_path: &Path,
    dataset_name: &str,
    model_name: &str,
    resolver_name: &str,
    gold_docs: &[crate::eval::coref::CorefDocument],
    per_doc: &[crate::eval::coref_metrics::CorefEvaluation],
    selected_doc_indices: &[usize],
    model: &dyn crate::Model,
    resolver: &crate::eval::coref_resolver::SimpleCorefResolver,
    coref_oracle_mentions: bool,
) -> Result<(), String> {
    let parent = output_path.parent().unwrap_or_else(|| Path::new("."));
    let stem = output_path
        .file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("coref_eval");
    let files_dir = parent.join(format!("{stem}_files"));
    fs::create_dir_all(&files_dir)
        .map_err(|e| format!("Failed to create {:?}: {}", files_dir, e))?;

    // Per-doc pages: gold inspector + predicted inspector + a small container page
    for &idx in selected_doc_indices {
        let doc = gold_docs
            .get(idx)
            .ok_or_else(|| format!("Invalid doc index {}", idx))?;
        let scores = per_doc
            .get(idx)
            .ok_or_else(|| format!("Missing per-doc eval for doc {}", idx))?;

        let doc_id = doc
            .doc_id
            .clone()
            .unwrap_or_else(|| format!("doc_{:06}", idx));

        // Gold inspector
        let gold_entities = coref_doc_to_gold_entities(doc);
        let gold_gdoc = GroundedDocument::from_entities(
            format!("{dataset_name}:{doc_id}:gold"),
            doc.text.clone(),
            &gold_entities,
        );
        let gold_html = render_document_html(&gold_gdoc);
        let gold_filename = format!("coref_{:06}_gold.html", idx);
        let gold_path = files_dir.join(&gold_filename);
        fs::write(&gold_path, gold_html)
            .map_err(|e| format!("Failed to write {:?}: {}", gold_path, e))?;

        // Predicted inspector (re-run NER + coref for selected docs only)
        let entities = if coref_oracle_mentions {
            coref_doc_to_oracle_mentions(doc)
        } else {
            model
                .extract_entities(doc.text.as_str(), None)
                .unwrap_or_default()
        };
        let resolved = resolver.resolve(&entities);
        let pred_gdoc = GroundedDocument::from_entities(
            format!("{dataset_name}:{doc_id}:pred"),
            doc.text.clone(),
            &resolved,
        );
        let pred_html = render_document_html(&pred_gdoc);
        let pred_filename = format!("coref_{:06}_pred.html", idx);
        let pred_path = files_dir.join(&pred_filename);
        fs::write(&pred_path, pred_html)
            .map_err(|e| format!("Failed to write {:?}: {}", pred_path, e))?;

        // Container page (side-by-side)
        let container_filename = format!("coref_{:06}.html", idx);
        let container_path = files_dir.join(&container_filename);
        let mut page = String::new();
        page.push_str("<!DOCTYPE html><html><head><meta charset=\"UTF-8\"><meta name=\"color-scheme\" content=\"dark light\">");
        page.push_str(&format!(
            "<title>{}</title>",
            html_escape(&format!(
                "coref explorer — {dataset_name} — {doc_id} — F1={:.3}",
                scores.conll_f1
            ))
        ));
        page.push_str(
            r#"<style>
:root{color-scheme:light dark;--bg:#0a0a0a;--text:#b0b0b0;--text-strong:#fff;--muted:#777;--border:#222;--border-strong:#333;--hover:#111;--panel-bg:#0d0d0d;--link:#9ad}
@media (prefers-color-scheme: light){:root{--bg:#fff;--text:#222;--text-strong:#000;--muted:#555;--border:#d6d6d6;--border-strong:#c6c6c6;--hover:#f0f0f0;--panel-bg:#f7f7f7;--link:#06c}}
html[data-theme='dark']{--bg:#0a0a0a;--text:#b0b0b0;--text-strong:#fff;--muted:#777;--border:#222;--border-strong:#333;--hover:#111;--panel-bg:#0d0d0d;--link:#9ad}
html[data-theme='light']{--bg:#fff;--text:#222;--text-strong:#000;--muted:#555;--border:#d6d6d6;--border-strong:#c6c6c6;--hover:#f0f0f0;--panel-bg:#f7f7f7;--link:#06c}
*{box-sizing:border-box;margin:0;padding:0}
body{font:12px/1.4 monospace;background:var(--bg);color:var(--text);padding:10px}
h1{font-size:14px;color:var(--text-strong);font-weight:normal;border-bottom:1px solid var(--border-strong);padding:4px 0;margin:0 0 10px}
.meta{color:var(--muted);margin:0 0 10px}
.meta b{color:var(--text);font-weight:normal}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.panel{border:1px solid var(--border);background:var(--panel-bg)}
.hdr{display:flex;justify-content:space-between;align-items:center;padding:6px 8px;border-bottom:1px solid var(--border)}
.hdr a{color:var(--link);text-decoration:none}
.hdr a:hover{text-decoration:underline}
iframe{width:100%;height:82vh;border:0;background:var(--bg)}
.toggle{cursor:pointer;user-select:none;color:var(--muted);border:1px solid var(--border);background:var(--bg);padding:2px 6px;font-size:10px}
</style></head><body>"#,
        );
        page.push_str(&format!(
            "<div style=\"display:flex;justify-content:space-between;align-items:center\"><h1>coref doc {idx:06} — {}</h1><span class=\"toggle\" id=\"theme-toggle\" title=\"toggle theme (auto → dark → light)\">theme: auto</span></div>",
            html_escape(&doc_id)
        ));
        page.push_str(&format!(
            "<div class=\"meta\"><b>dataset</b> {} &nbsp; <b>model</b> {} &nbsp; <b>resolver</b> {} &nbsp; <b>CoNLL F1</b> {:.3} &nbsp; <b>MUC</b> {:.3} &nbsp; <b>B³</b> {:.3} &nbsp; <b>CEAF-e</b> {:.3}</div>",
            html_escape(dataset_name),
            html_escape(model_name),
            html_escape(resolver_name),
            scores.conll_f1,
            scores.muc.f1,
            scores.b_cubed.f1,
            scores.ceaf_e.f1,
        ));
        page.push_str("<div class=\"grid\">");
        page.push_str(&format!(
            "<div class=\"panel\"><div class=\"hdr\"><div>gold</div><div><a target=\"_blank\" rel=\"noopener\" href=\"{}\">open</a></div></div><iframe id=\"gold-frame\" src=\"{}\"></iframe></div>",
            html_escape(&gold_filename),
            html_escape(&gold_filename),
        ));
        page.push_str(&format!(
            "<div class=\"panel\"><div class=\"hdr\"><div>predicted</div><div><a target=\"_blank\" rel=\"noopener\" href=\"{}\">open</a></div></div><iframe id=\"pred-frame\" src=\"{}\"></iframe></div>",
            html_escape(&pred_filename),
            html_escape(&pred_filename),
        ));
        page.push_str("</div>");
        page.push_str(
            r#"<script>
(() => {
  // Theme toggle: auto → dark → light (persisted).
  const themeBtn = document.getElementById('theme-toggle');
  const themeKey = 'anno-theme';
  const applyTheme = (theme) => {
    const t = theme || 'auto';
    if (t === 'auto') {
      delete document.documentElement.dataset.theme;
    } else {
      document.documentElement.dataset.theme = t;
    }
    if (themeBtn) themeBtn.textContent = `theme: ${t}`;
  };
  const readTheme = () => {
    try { return localStorage.getItem(themeKey) || 'auto'; } catch (_) { return 'auto'; }
  };
  const writeTheme = (t) => {
    try { localStorage.setItem(themeKey, t); } catch (_) { /* ignore */ }
  };
  applyTheme(readTheme());
  if (themeBtn) {
    themeBtn.addEventListener('click', () => {
      const cur = readTheme();
      const next = cur === 'auto' ? 'dark' : (cur === 'dark' ? 'light' : 'auto');
      writeTheme(next);
      applyTheme(next);
    });
  }

  const gold = document.getElementById('gold-frame');
  const pred = document.getElementById('pred-frame');
  if (!gold || !pred) return;
  window.addEventListener('message', (ev) => {
    const data = ev && ev.data ? ev.data : null;
    if (!data || data.type !== 'anno:activate-span') return;
    if (ev.source === gold.contentWindow) {
      pred.contentWindow && pred.contentWindow.postMessage(data, '*');
    } else if (ev.source === pred.contentWindow) {
      gold.contentWindow && gold.contentWindow.postMessage(data, '*');
    }
  });
})();
</script>"#,
        );
        page.push_str("</body></html>");
        fs::write(&container_path, page)
            .map_err(|e| format!("Failed to write {:?}: {}", container_path, e))?;
    }

    // Index page
    let mut index = String::new();
    index.push_str("<!DOCTYPE html><html><head><meta charset=\"UTF-8\"><meta name=\"color-scheme\" content=\"dark light\">");
    index.push_str(&format!(
        "<title>{}</title>",
        html_escape(&format!(
            "coref eval explorer — {model_name} — {dataset_name}"
        ))
    ));
    index.push_str(
        r##"<style>
:root{color-scheme:light dark;--bg:#0a0a0a;--text:#b0b0b0;--text-strong:#fff;--muted:#777;--border:#222;--border-strong:#333;--hover:#111;--input-bg:#080808;--link:#9ad;--code:#bbb}
@media (prefers-color-scheme: light){:root{--bg:#fff;--text:#222;--text-strong:#000;--muted:#555;--border:#d6d6d6;--border-strong:#c6c6c6;--hover:#f0f0f0;--input-bg:#fff;--link:#06c;--code:#333}}
html[data-theme='dark']{--bg:#0a0a0a;--text:#b0b0b0;--text-strong:#fff;--muted:#777;--border:#222;--border-strong:#333;--hover:#111;--input-bg:#080808;--link:#9ad;--code:#bbb}
html[data-theme='light']{--bg:#fff;--text:#222;--text-strong:#000;--muted:#555;--border:#d6d6d6;--border-strong:#c6c6c6;--hover:#f0f0f0;--input-bg:#fff;--link:#06c;--code:#333}
*{box-sizing:border-box;margin:0;padding:0}
body{font:12px/1.4 monospace;background:var(--bg);color:var(--text);padding:12px}
h1{font-size:14px;color:var(--text-strong);font-weight:normal;border-bottom:1px solid var(--border-strong);padding:4px 0;margin:0 0 12px}
.meta{color:var(--muted);margin:0 0 12px}
.meta b{color:var(--text);font-weight:normal}
.row{display:flex;gap:12px;align-items:center;margin:0 0 10px}
input{flex:1;background:var(--input-bg);border:1px solid var(--border);color:var(--text);padding:6px 8px}
.count{color:var(--muted)}
table{width:100%;border-collapse:collapse;font-size:11px}
th,td{padding:4px 8px;text-align:left;border:1px solid var(--border);vertical-align:top}
th{background:var(--hover);color:var(--muted);font-weight:normal;text-transform:uppercase;font-size:10px}
tr:hover{background:var(--hover)}
a{color:var(--link);text-decoration:none}
a:hover{text-decoration:underline}
.num{text-align:right;font-variant-numeric:tabular-nums}
code{color:var(--code)}
.toggle{cursor:pointer;user-select:none;color:var(--muted);border:1px solid var(--border);background:var(--bg);padding:2px 6px;font-size:10px}
</style></head><body>"##,
    );
    index.push_str("<div class=\"row\" style=\"justify-content:space-between\"><h1>coref eval explorer</h1><span class=\"toggle\" id=\"theme-toggle\" title=\"toggle theme (auto → dark → light)\">theme: auto</span></div>");
    index.push_str(&format!(
        "<div class=\"meta\"><b>model</b> {} &nbsp; <b>resolver</b> {} &nbsp; <b>dataset</b> {} &nbsp; <b>docs</b> {}</div>",
        html_escape(model_name),
        html_escape(resolver_name),
        html_escape(dataset_name),
        gold_docs.len()
    ));
    index.push_str(
        r##"<div class="row">
  <input id="doc-filter" placeholder="filter (doc id, script, text…)" />
  <div id="doc-count" class="count"></div>
</div>
<table id="doc-table"><thead><tr>
  <th>doc</th><th class="num">conll</th><th class="num">muc</th><th class="num">b3</th><th class="num">ceaf</th><th class="num">mentions</th><th class="num">chains</th><th>text</th>
</tr></thead><tbody>"##,
    );

    for &idx in selected_doc_indices {
        let doc = &gold_docs[idx];
        let scores = &per_doc[idx];
        let doc_id = doc
            .doc_id
            .clone()
            .unwrap_or_else(|| format!("doc_{:06}", idx));
        let mention_count = doc.mention_count();
        let chain_count = doc.chain_count();
        let preview = preview_text(&doc.text, 180);
        let href = format!("coref_{:06}.html", idx);
        index.push_str(&format!(
            "<tr data-hay=\"{hay}\"><td><a target=\"_blank\" rel=\"noopener\" href=\"{href}\">{doc_id}</a></td><td class=\"num\">{conll:.3}</td><td class=\"num\">{muc:.3}</td><td class=\"num\">{b3:.3}</td><td class=\"num\">{ceaf:.3}</td><td class=\"num\">{mentions}</td><td class=\"num\">{chains}</td><td><code>{text}</code></td></tr>",
            hay = html_escape(&format!("{} {}", doc_id, preview)).to_lowercase(),
            href = html_escape(&href),
            doc_id = html_escape(&doc_id),
            conll = scores.conll_f1,
            muc = scores.muc.f1,
            b3 = scores.b_cubed.f1,
            ceaf = scores.ceaf_e.f1,
            mentions = mention_count,
            chains = chain_count,
            text = html_escape(&preview),
        ));
    }

    index.push_str(
        r##"</tbody></table>
<script>
(() => {
  // Theme toggle: auto → dark → light (persisted).
  const themeBtn = document.getElementById("theme-toggle");
  const themeKey = "anno-theme";
  const applyTheme = (theme) => {
    const t = theme || "auto";
    if (t === "auto") {
      delete document.documentElement.dataset.theme;
    } else {
      document.documentElement.dataset.theme = t;
    }
    if (themeBtn) themeBtn.textContent = `theme: ${t}`;
  };
  const readTheme = () => {
    try { return localStorage.getItem(themeKey) || "auto"; } catch (_) { return "auto"; }
  };
  const writeTheme = (t) => {
    try { localStorage.setItem(themeKey, t); } catch (_) { /* ignore */ }
  };
  applyTheme(readTheme());
  if (themeBtn) {
    themeBtn.addEventListener("click", () => {
      const cur = readTheme();
      const next = cur === "auto" ? "dark" : (cur === "dark" ? "light" : "auto");
      writeTheme(next);
      applyTheme(next);
    });
  }

  const input = document.getElementById("doc-filter");
  const rows = Array.from(document.querySelectorAll("#doc-table tbody tr"));
  const count = document.getElementById("doc-count");
  function update() {
    const q = (input.value || "").toLowerCase().trim();
    let shown = 0;
    for (const tr of rows) {
      const hay = (tr.dataset.hay || "");
      const show = !q || hay.includes(q);
      tr.style.display = show ? "" : "none";
      if (show) shown++;
    }
    count.textContent = `${shown} shown / ${rows.length} total`;
  }
  input.addEventListener("input", update);
  update();
})();
</script>
</body></html>"##,
    );

    fs::write(output_path, index)
        .map_err(|e| format!("Failed to write {:?}: {}", output_path, e))?;
    Ok(())
}

#[cfg(feature = "eval-advanced")]
#[derive(Debug, Clone)]
struct RelHtmlSpan {
    start: usize,
    end: usize,
    label: String,
    id: String,
    class: &'static str,
}

#[cfg(feature = "eval-advanced")]
fn extract_span_text(text: &str, start: usize, end: usize) -> String {
    let char_count = text.chars().count();
    if start >= char_count || end > char_count || start >= end {
        return String::new();
    }
    text.chars().skip(start).take(end - start).collect()
}

#[cfg(feature = "eval-advanced")]
fn annotate_text_with_rel_spans(text: &str, spans: &[RelHtmlSpan]) -> String {
    let mut sorted = spans.to_vec();
    sorted.sort_by_key(|s| (s.start, s.end));

    let char_count = text.chars().count();
    let mut out = String::new();
    let mut last_end = 0usize;

    for s in sorted {
        let start = s.start;
        let end = s.end.min(char_count);
        if start < last_end || start >= char_count || start >= end {
            continue;
        }

        if start > last_end {
            let before: String = text.chars().skip(last_end).take(start - last_end).collect();
            out.push_str(&html_escape(&before));
        }

        let span_text: String = text.chars().skip(start).take(end - start).collect();
        let title = format!("[{}] {}..{}", s.label, start, end);
        out.push_str(&format!(
            "<span id=\"{id}\" class=\"e {class}\" data-label=\"{label}\" data-start=\"{start}\" data-end=\"{end}\" title=\"{title}\">{txt}</span>",
            id = html_escape(&s.id),
            class = s.class,
            label = html_escape(&s.label),
            start = start,
            end = end,
            title = html_escape(&title),
            txt = html_escape(&span_text),
        ));
        last_end = end;
    }

    if last_end < char_count {
        let after: String = text.chars().skip(last_end).collect();
        out.push_str(&html_escape(&after));
    }

    out
}

#[cfg(feature = "eval-advanced")]
fn build_rel_spans_from_gold(
    text: &str,
    gold: &[crate::eval::relation::RelationGold],
    prefix: &str,
    class: &'static str,
) -> (
    Vec<RelHtmlSpan>,
    std::collections::HashMap<(usize, usize), String>,
) {
    use std::collections::{HashMap, HashSet};
    let mut uniq: HashSet<(usize, usize)> = HashSet::new();
    for r in gold {
        uniq.insert(r.head_span);
        uniq.insert(r.tail_span);
    }
    let mut spans: Vec<(usize, usize)> = uniq.into_iter().collect();
    spans.sort_by_key(|(s, e)| (*s, *e));

    let mut out = Vec::new();
    let mut map: HashMap<(usize, usize), String> = HashMap::new();
    for (i, (s, e)) in spans.into_iter().enumerate() {
        let id = format!("{prefix}{i}");
        // Prefer exact mention text if we can find it in any gold relation; otherwise slice.
        let mut label = String::new();
        for r in gold {
            if r.head_span == (s, e) {
                label = r.head_type.clone();
                break;
            }
            if r.tail_span == (s, e) {
                label = r.tail_type.clone();
                break;
            }
        }
        if label.is_empty() {
            label = "ENT".to_string();
        }
        let _surface = extract_span_text(text, s, e);
        out.push(RelHtmlSpan {
            start: s,
            end: e,
            label,
            id: id.clone(),
            class,
        });
        map.insert((s, e), id);
    }
    (out, map)
}

#[cfg(feature = "eval-advanced")]
fn build_rel_spans_from_pred(
    _text: &str,
    pred: &[crate::eval::relation::RelationPrediction],
    prefix: &str,
    class: &'static str,
) -> (
    Vec<RelHtmlSpan>,
    std::collections::HashMap<(usize, usize), String>,
) {
    use std::collections::{HashMap, HashSet};
    let mut uniq: HashSet<(usize, usize)> = HashSet::new();
    for r in pred {
        uniq.insert(r.head_span);
        uniq.insert(r.tail_span);
    }
    let mut spans: Vec<(usize, usize)> = uniq.into_iter().collect();
    spans.sort_by_key(|(s, e)| (*s, *e));

    let mut out = Vec::new();
    let mut map: HashMap<(usize, usize), String> = HashMap::new();
    for (i, (s, e)) in spans.into_iter().enumerate() {
        let id = format!("{prefix}{i}");
        let mut label = String::new();
        for r in pred {
            if r.head_span == (s, e) {
                label = r.head_type.clone();
                break;
            }
            if r.tail_span == (s, e) {
                label = r.tail_type.clone();
                break;
            }
        }
        if label.is_empty() {
            label = "ENT".to_string();
        }
        out.push(RelHtmlSpan {
            start: s,
            end: e,
            label,
            id: id.clone(),
            class,
        });
        map.insert((s, e), id);
    }
    (out, map)
}

#[cfg(feature = "eval-advanced")]
fn render_relation_doc_html(
    dataset_name: &str,
    model_name: &str,
    doc_id: &str,
    text: &str,
    gold: &[crate::eval::relation::RelationGold],
    pred: &[crate::eval::relation::RelationPrediction],
    metrics: &crate::eval::relation::RelationMetrics,
) -> String {
    // Strict matching alignment (doc-local) to mark rows:
    // - matched: gold[i] ↔ pred[j]
    // - fn: gold[i] unmatched
    // - fp: pred[j] unmatched
    //
    // This mirrors the strict matching logic in `evaluate_relations` with:
    // - case-insensitive relation type
    // - directed relations
    // - entity type match NOT required (consistent with dataset eval config)
    let mut gold_taken = vec![false; gold.len()];
    let mut pred_taken = vec![false; pred.len()];
    let mut gold_to_pred: Vec<Option<usize>> = vec![None; gold.len()];
    let mut pred_to_gold: Vec<Option<usize>> = vec![None; pred.len()];

    for (pi, p) in pred.iter().enumerate() {
        if pred_taken[pi] {
            continue;
        }
        for (gi, g) in gold.iter().enumerate() {
            if gold_taken[gi] {
                continue;
            }
            if p.relation_type.to_lowercase() != g.relation_type.to_lowercase() {
                continue;
            }
            let forward = p.head_span == g.head_span && p.tail_span == g.tail_span;
            if forward {
                gold_taken[gi] = true;
                pred_taken[pi] = true;
                gold_to_pred[gi] = Some(pi);
                pred_to_gold[pi] = Some(gi);
                break;
            }
        }
    }

    let (gold_spans, gold_id) = build_rel_spans_from_gold(text, gold, "G", "e-gold");
    let (pred_spans, pred_id) = build_rel_spans_from_pred(text, pred, "P", "e-pred");

    let mut html = String::new();
    html.push_str("<!DOCTYPE html><html><head><meta charset=\"UTF-8\"><meta name=\"color-scheme\" content=\"dark light\">");
    html.push_str(&format!(
        "<title>{}</title>",
        html_escape(&format!("relation explorer — {dataset_name} — {doc_id}"))
    ));
    html.push_str(
        r##"<style>
:root{
  color-scheme: light dark;
  --bg:#0a0a0a;
  --panel-bg:#0d0d0d;
  --text:#b0b0b0;
  --text-strong:#fff;
  --muted:#777;
  --border:#222;
  --border-strong:#333;
  --hover:#111;
  --input-bg:#080808;
  --active:#ddd;
  --gold-bg:#1a2e1a; --gold-br:#4a8a4a; --gold-tx:#88cc88;
  --pred-bg:#1a1a2e; --pred-br:#4a4a8a; --pred-tx:#8888cc;
  --fn-bg:#2a1010;
  --fp-bg:#2a1c10;
  --head:#ffcc66;
  --tail:#66ccff;
}
@media (prefers-color-scheme: light){
  :root{
    --bg:#ffffff;
    --panel-bg:#f7f7f7;
    --text:#222;
    --text-strong:#000;
    --muted:#555;
    --border:#d6d6d6;
    --border-strong:#c6c6c6;
    --hover:#f0f0f0;
    --input-bg:#ffffff;
    --active:#000;
    --gold-bg:#e9f7e9; --gold-br:#2f8a2f; --gold-tx:#1f5a1f;
    --pred-bg:#e9e9ff; --pred-br:#6c6cff; --pred-tx:#2b2b7a;
    --fn-bg:#ffe9e9;
    --fp-bg:#fff2df;
    --head:#a05a00;
    --tail:#0a5a8a;
  }
}
html[data-theme='dark']{
  --bg:#0a0a0a; --panel-bg:#0d0d0d; --text:#b0b0b0; --text-strong:#fff;
  --muted:#777; --border:#222; --border-strong:#333; --hover:#111; --input-bg:#080808; --active:#ddd;
  --gold-bg:#1a2e1a; --gold-br:#4a8a4a; --gold-tx:#88cc88;
  --pred-bg:#1a1a2e; --pred-br:#4a4a8a; --pred-tx:#8888cc;
  --fn-bg:#2a1010; --fp-bg:#2a1c10; --head:#ffcc66; --tail:#66ccff;
}
html[data-theme='light']{
  --bg:#ffffff; --panel-bg:#f7f7f7; --text:#222; --text-strong:#000;
  --muted:#555; --border:#d6d6d6; --border-strong:#c6c6c6; --hover:#f0f0f0; --input-bg:#ffffff; --active:#000;
  --gold-bg:#e9f7e9; --gold-br:#2f8a2f; --gold-tx:#1f5a1f;
  --pred-bg:#e9e9ff; --pred-br:#6c6cff; --pred-tx:#2b2b7a;
  --fn-bg:#ffe9e9; --fp-bg:#fff2df; --head:#a05a00; --tail:#0a5a8a;
}

*{box-sizing:border-box;margin:0;padding:0}
body{font:12px/1.4 monospace;background:var(--bg);color:var(--text);padding:10px}
h1,h2{color:var(--text-strong);font-weight:normal;border-bottom:1px solid var(--border-strong);padding:4px 0;margin:12px 0 8px}
h1{font-size:14px}h2{font-size:12px}
.meta{color:var(--muted);margin:0 0 10px}
.meta b{color:var(--text);font-weight:normal}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.panel{border:1px solid var(--border);background:var(--panel-bg);padding:8px}
.text{background:var(--input-bg);border:1px solid var(--border);padding:8px;white-space:pre-wrap;word-break:break-word;line-height:1.6;min-height:110px}
table{width:100%;border-collapse:collapse;font-size:11px;margin:6px 0 0}
th,td{padding:4px 8px;text-align:left;border:1px solid var(--border);vertical-align:top}
th{background:var(--hover);color:var(--muted);font-weight:normal;text-transform:uppercase;font-size:10px}
tr:hover{background:var(--hover)}
.match-ok{opacity:0.55}
.match-fn{background:var(--fn-bg)}
.match-fp{background:var(--fp-bg)}
.e{padding:1px 2px;border-bottom:2px solid}
.seg{cursor:pointer}
.e-gold{background:var(--gold-bg);border-color:var(--gold-br);color:var(--gold-tx)}
.e-pred{background:var(--pred-bg);border-color:var(--pred-br);color:var(--pred-tx)}
.e-head{outline:2px solid var(--head);outline-offset:1px}
.e-tail{outline:2px solid var(--tail);outline-offset:1px}
.row-active{outline:1px solid var(--muted)}
.sel{color:var(--muted);margin:6px 0 12px}
.toggle{cursor:pointer;user-select:none;color:var(--muted);border:1px solid var(--border);background:var(--bg);padding:2px 6px;font-size:10px}
</style></head><body>"##,
    );

    html.push_str(&format!(
        "<div style=\"display:flex;justify-content:space-between;align-items:center\"><h1>relation doc {}</h1><span class=\"toggle\" id=\"theme-toggle\" title=\"toggle theme (auto → dark → light)\">theme: auto</span></div>",
        html_escape(doc_id)
    ));
    html.push_str(&format!(
        "<div class=\"meta\"><b>dataset</b> {} &nbsp; <b>model</b> {} &nbsp; <b>gold</b> {} &nbsp; <b>pred</b> {} &nbsp; <b>strict F1</b> {:.3} &nbsp; <b>boundary F1</b> {:.3}</div>",
        html_escape(dataset_name),
        html_escape(model_name),
        gold.len(),
        pred.len(),
        metrics.strict_f1,
        metrics.boundary_f1
    ));
    html.push_str("<div id=\"selection\" class=\"sel\">click a relation row to highlight head/tail spans</div>");
    html.push_str("<div class=\"sel\"><label><input type=\"checkbox\" id=\"only-errors\" /> only errors</label></div>");

    html.push_str("<div class=\"grid\">");

    // Gold side
    html.push_str("<div class=\"panel\"><h2>gold</h2>");
    html.push_str("<div class=\"text\">");
    html.push_str(&annotate_text_with_rel_spans(text, &gold_spans));
    html.push_str("</div>");
    html.push_str("<table><tr><th>rel</th><th>head</th><th>tail</th></tr>");
    for (i, r) in gold.iter().enumerate() {
        let hid = gold_id.get(&r.head_span).cloned().unwrap_or_default();
        let tid = gold_id.get(&r.tail_span).cloned().unwrap_or_default();
        let (row_class, peer_attr) = if let Some(pi) = gold_to_pred.get(i).and_then(|x| *x) {
            ("match-ok", format!(" data-peer=\"RP{}\"", pi))
        } else {
            ("match-fn", String::new())
        };
        html.push_str(&format!(
            "<tr id=\"RG{i}\" class=\"rel-row {row_class}\" data-side=\"gold\" data-hid=\"{hid}\" data-tid=\"{tid}\" data-rel=\"{rel}\"{peer}><td><a class=\"rel-link\" href=\"#RG{i}\">{rel}</a></td><td>[{ht}] {hx}</td><td>[{tt}] {tx}</td></tr>",
            i = i,
            hid = html_escape(&hid),
            tid = html_escape(&tid),
            rel = html_escape(&r.relation_type),
            ht = html_escape(&r.head_type),
            hx = html_escape(&r.head_text),
            tt = html_escape(&r.tail_type),
            tx = html_escape(&r.tail_text),
            row_class = row_class,
            peer = peer_attr,
        ));
    }
    html.push_str("</table></div>");

    // Pred side
    html.push_str("<div class=\"panel\"><h2>predicted</h2>");
    html.push_str("<div class=\"text\">");
    html.push_str(&annotate_text_with_rel_spans(text, &pred_spans));
    html.push_str("</div>");
    html.push_str("<table><tr><th>rel</th><th>head</th><th>tail</th><th>conf</th></tr>");
    for (i, r) in pred.iter().enumerate() {
        let hid = pred_id.get(&r.head_span).cloned().unwrap_or_default();
        let tid = pred_id.get(&r.tail_span).cloned().unwrap_or_default();
        let head_txt = extract_span_text(text, r.head_span.0, r.head_span.1);
        let tail_txt = extract_span_text(text, r.tail_span.0, r.tail_span.1);
        let (row_class, peer_attr) = if let Some(gi) = pred_to_gold.get(i).and_then(|x| *x) {
            ("match-ok", format!(" data-peer=\"RG{}\"", gi))
        } else {
            ("match-fp", String::new())
        };
        html.push_str(&format!(
            "<tr id=\"RP{i}\" class=\"rel-row {row_class}\" data-side=\"pred\" data-hid=\"{hid}\" data-tid=\"{tid}\" data-rel=\"{rel}\"{peer}><td><a class=\"rel-link\" href=\"#RP{i}\">{rel}</a></td><td>[{ht}] {hx}</td><td>[{tt}] {tx}</td><td>{conf:.2}</td></tr>",
            i = i,
            hid = html_escape(&hid),
            tid = html_escape(&tid),
            rel = html_escape(&r.relation_type),
            ht = html_escape(&r.head_type),
            hx = html_escape(&head_txt),
            tt = html_escape(&r.tail_type),
            tx = html_escape(&tail_txt),
            conf = r.confidence as f64,
            row_class = row_class,
            peer = peer_attr,
        ));
    }
    html.push_str("</table></div>");

    html.push_str("</div>"); // grid

    html.push_str(
        r##"<script>
(() => {
  // Theme toggle: auto → dark → light (persisted).
  const themeBtn = document.getElementById('theme-toggle');
  const themeKey = 'anno-theme';
  const applyTheme = (theme) => {
    const t = theme || 'auto';
    if (t === 'auto') {
      delete document.documentElement.dataset.theme;
    } else {
      document.documentElement.dataset.theme = t;
    }
    if (themeBtn) themeBtn.textContent = `theme: ${t}`;
  };
  const readTheme = () => {
    try { return localStorage.getItem(themeKey) || 'auto'; } catch (_) { return 'auto'; }
  };
  const writeTheme = (t) => {
    try { localStorage.setItem(themeKey, t); } catch (_) { /* ignore */ }
  };
  applyTheme(readTheme());
  if (themeBtn) {
    themeBtn.addEventListener('click', () => {
      const cur = readTheme();
      const next = cur === 'auto' ? 'dark' : (cur === 'dark' ? 'light' : 'auto');
      writeTheme(next);
      applyTheme(next);
    });
  }

  function clearActive() {
    document.querySelectorAll(".e-head").forEach((el) => el.classList.remove("e-head"));
    document.querySelectorAll(".e-tail").forEach((el) => el.classList.remove("e-tail"));
    document.querySelectorAll("tr.rel-row.row-active").forEach((el) => el.classList.remove("row-active"));
  }

  function activate(row) {
    clearActive();
    if (!row) return;
    row.classList.add("row-active");
    const hid = row.dataset.hid;
    const tid = row.dataset.tid;
    const rel = row.dataset.rel || "";
    const sel = document.getElementById("selection");

    const head = hid ? document.getElementById(hid) : null;
    const tail = tid ? document.getElementById(tid) : null;
    if (head) head.classList.add("e-head");
    if (tail) tail.classList.add("e-tail");

    // Also highlight matching spans (same start/end) in the opposite panel.
    const peerClass = (row.dataset.side === 'gold') ? 'e-pred' : 'e-gold';
    const highlightPeer = (el, cls) => {
      if (!el) return;
      const s = el.getAttribute('data-start');
      const e = el.getAttribute('data-end');
      if (s === null || e === null) return;
      document.querySelectorAll(`span.${peerClass}[data-start='${s}'][data-end='${e}']`).forEach((p) => p.classList.add(cls));
    };
    highlightPeer(head, "e-head");
    highlightPeer(tail, "e-tail");

    // Also highlight the matched peer row (if any).
    const peerId = row.dataset.peer;
    if (peerId) {
      const peerRow = document.getElementById(peerId);
      if (peerRow) peerRow.classList.add("row-active");
    }

    if (sel) {
      const parts = [];
      parts.push(`${row.dataset.side} ${row.id}`);
      if (rel) parts.push(`rel=${rel}`);
      if (head) parts.push(`head=${hid}${head.dataset.label ? ' [' + head.dataset.label + ']' : ''}`);
      if (tail) parts.push(`tail=${tid}${tail.dataset.label ? ' [' + tail.dataset.label + ']' : ''}`);
      sel.textContent = parts.join("  |  ");
    }

    if (row.id) history.replaceState(null, "", '#' + row.id);
    const target = head || tail || row;
    if (target) target.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  document.querySelectorAll("tr.rel-row").forEach((tr) => {
    tr.addEventListener("click", () => activate(tr));
  });
  document.querySelectorAll("a.rel-link").forEach((a) => {
    a.addEventListener("click", (ev) => {
      ev.preventDefault();
      const tr = a.closest("tr.rel-row");
      if (tr) activate(tr);
    });
  });

  const hash = (location.hash || '').slice(1);
  if (hash && (hash.startsWith('RG') || hash.startsWith('RP'))) {
    const tr = document.getElementById(hash);
    if (tr && tr.classList && tr.classList.contains('rel-row')) activate(tr);
  }

  // Toggle: show only errors (hide matched rows).
  const only = document.getElementById('only-errors');
  if (only) {
    const update = () => {
      const hideMatched = !!only.checked;
      document.querySelectorAll('tr.rel-row.match-ok').forEach((tr) => {
        tr.style.display = hideMatched ? 'none' : '';
      });
    };
    only.addEventListener('change', update);
    update();
  }
})();
</script>"##,
    );

    html.push_str("</body></html>");
    html
}

#[cfg(feature = "eval-advanced")]
fn write_relation_error_explorer_html(
    output_path: &Path,
    dataset_name: &str,
    model_name: &str,
    docs: &[crate::eval::loader::RelationDocument],
    pred_by_doc: &[Vec<crate::eval::relation::RelationPrediction>],
    per_doc: &[crate::eval::relation::RelationMetrics],
    selected_doc_indices: &[usize],
) -> Result<(), String> {
    let parent = output_path.parent().unwrap_or_else(|| Path::new("."));
    let stem = output_path
        .file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("relation_eval");
    let files_dir = parent.join(format!("{stem}_files"));
    fs::create_dir_all(&files_dir)
        .map_err(|e| format!("Failed to create {:?}: {}", files_dir, e))?;

    for &idx in selected_doc_indices {
        let doc = docs
            .get(idx)
            .ok_or_else(|| format!("Invalid doc index {}", idx))?;
        let pred = pred_by_doc
            .get(idx)
            .ok_or_else(|| format!("Missing pred for doc {}", idx))?;
        let m = per_doc
            .get(idx)
            .ok_or_else(|| format!("Missing metrics for doc {}", idx))?;
        let doc_id = format!("doc_{:06}", idx);

        let page = render_relation_doc_html(
            dataset_name,
            model_name,
            &doc_id,
            &doc.text,
            &doc.relations,
            pred,
            m,
        );
        let filename = format!("rel_{:06}.html", idx);
        let path = files_dir.join(&filename);
        fs::write(&path, page).map_err(|e| format!("Failed to write {:?}: {}", path, e))?;
    }

    let mut index = String::new();
    index.push_str("<!DOCTYPE html><html><head><meta charset=\"UTF-8\"><meta name=\"color-scheme\" content=\"dark light\">");
    index.push_str(&format!(
        "<title>{}</title>",
        html_escape(&format!(
            "relation eval explorer — {model_name} — {dataset_name}"
        ))
    ));
    index.push_str(
        r##"<style>
:root{color-scheme:light dark;--bg:#0a0a0a;--text:#b0b0b0;--text-strong:#fff;--muted:#777;--border:#222;--border-strong:#333;--hover:#111;--input-bg:#080808;--link:#9ad;--code:#bbb}
@media (prefers-color-scheme: light){:root{--bg:#fff;--text:#222;--text-strong:#000;--muted:#555;--border:#d6d6d6;--border-strong:#c6c6c6;--hover:#f0f0f0;--input-bg:#fff;--link:#06c;--code:#333}}
html[data-theme='dark']{--bg:#0a0a0a;--text:#b0b0b0;--text-strong:#fff;--muted:#777;--border:#222;--border-strong:#333;--hover:#111;--input-bg:#080808;--link:#9ad;--code:#bbb}
html[data-theme='light']{--bg:#fff;--text:#222;--text-strong:#000;--muted:#555;--border:#d6d6d6;--border-strong:#c6c6c6;--hover:#f0f0f0;--input-bg:#fff;--link:#06c;--code:#333}
*{box-sizing:border-box;margin:0;padding:0}
body{font:12px/1.4 monospace;background:var(--bg);color:var(--text);padding:12px}
h1{font-size:14px;color:var(--text-strong);font-weight:normal;border-bottom:1px solid var(--border-strong);padding:4px 0;margin:0 0 12px}
.meta{color:var(--muted);margin:0 0 12px}
.meta b{color:var(--text);font-weight:normal}
.row{display:flex;gap:12px;align-items:center;margin:0 0 10px}
input{flex:1;background:var(--input-bg);border:1px solid var(--border);color:var(--text);padding:6px 8px}
.count{color:var(--muted)}
table{width:100%;border-collapse:collapse;font-size:11px}
th,td{padding:4px 8px;text-align:left;border:1px solid var(--border);vertical-align:top}
th{background:var(--hover);color:var(--muted);font-weight:normal;text-transform:uppercase;font-size:10px}
tr:hover{background:var(--hover)}
a{color:var(--link);text-decoration:none}
a:hover{text-decoration:underline}
.num{text-align:right;font-variant-numeric:tabular-nums}
code{color:var(--code)}
.toggle{cursor:pointer;user-select:none;color:var(--muted);border:1px solid var(--border);background:var(--bg);padding:2px 6px;font-size:10px}
</style></head><body>"##,
    );
    index.push_str("<div class=\"row\" style=\"justify-content:space-between\"><h1>relation eval explorer</h1><span class=\"toggle\" id=\"theme-toggle\" title=\"toggle theme (auto → dark → light)\">theme: auto</span></div>");
    index.push_str(&format!(
        "<div class=\"meta\"><b>model</b> {} &nbsp; <b>dataset</b> {} &nbsp; <b>docs</b> {}</div>",
        html_escape(model_name),
        html_escape(dataset_name),
        docs.len()
    ));
    index.push_str(
        r##"<div class="row">
  <input id="doc-filter" placeholder="filter (doc id, relation type, text…)" />
  <div id="doc-count" class="count"></div>
</div>
<table id="doc-table"><thead><tr>
  <th>doc</th><th class="num">strict f1</th><th class="num">bound f1</th><th class="num">gold</th><th class="num">pred</th><th class="num">strict ✓</th><th>text</th>
</tr></thead><tbody>"##,
    );

    for &idx in selected_doc_indices {
        let doc_id = format!("doc_{:06}", idx);
        // Deep link to the first strict error row if possible.
        let mut href = format!("{stem}_files/rel_{:06}.html", idx);
        {
            let doc = &docs[idx];
            let pred = &pred_by_doc[idx];
            let mut gold_taken = vec![false; doc.relations.len()];
            let mut pred_taken = vec![false; pred.len()];
            let mut gold_to_pred: Vec<Option<usize>> = vec![None; doc.relations.len()];
            let mut pred_to_gold: Vec<Option<usize>> = vec![None; pred.len()];
            for (pi, p) in pred.iter().enumerate() {
                if pred_taken[pi] {
                    continue;
                }
                for (gi, g) in doc.relations.iter().enumerate() {
                    if gold_taken[gi] {
                        continue;
                    }
                    if p.relation_type.to_lowercase() != g.relation_type.to_lowercase() {
                        continue;
                    }
                    let forward = p.head_span == g.head_span && p.tail_span == g.tail_span;
                    if forward {
                        gold_taken[gi] = true;
                        pred_taken[pi] = true;
                        gold_to_pred[gi] = Some(pi);
                        pred_to_gold[pi] = Some(gi);
                        break;
                    }
                }
            }
            if let Some((gi, _)) = gold_to_pred.iter().enumerate().find(|(_, m)| m.is_none()) {
                href.push_str(&format!("#RG{}", gi));
            } else if let Some((pi, _)) = pred_to_gold.iter().enumerate().find(|(_, m)| m.is_none())
            {
                href.push_str(&format!("#RP{}", pi));
            }
        }
        let doc = &docs[idx];
        let m = &per_doc[idx];
        let preview = preview_text(&doc.text, 180);
        index.push_str(&format!(
            "<tr data-hay=\"{hay}\"><td><a target=\"_blank\" rel=\"noopener\" href=\"{href}\">{doc_id}</a></td><td class=\"num\">{sf1:.3}</td><td class=\"num\">{bf1:.3}</td><td class=\"num\">{g}</td><td class=\"num\">{p}</td><td class=\"num\">{ok}</td><td><code>{txt}</code></td></tr>",
            hay = html_escape(&format!("{} {} {}", doc_id, preview, dataset_name)).to_lowercase(),
            href = html_escape(&href),
            doc_id = html_escape(&doc_id),
            sf1 = m.strict_f1,
            bf1 = m.boundary_f1,
            g = doc.relations.len(),
            p = pred_by_doc[idx].len(),
            ok = m.strict_matches,
            txt = html_escape(&preview),
        ));
    }

    index.push_str(
        r##"</tbody></table>
<script>
(() => {
  // Theme toggle: auto → dark → light (persisted).
  const themeBtn = document.getElementById("theme-toggle");
  const themeKey = "anno-theme";
  const applyTheme = (theme) => {
    const t = theme || "auto";
    if (t === "auto") {
      delete document.documentElement.dataset.theme;
    } else {
      document.documentElement.dataset.theme = t;
    }
    if (themeBtn) themeBtn.textContent = `theme: ${t}`;
  };
  const readTheme = () => {
    try { return localStorage.getItem(themeKey) || "auto"; } catch (_) { return "auto"; }
  };
  const writeTheme = (t) => {
    try { localStorage.setItem(themeKey, t); } catch (_) { /* ignore */ }
  };
  applyTheme(readTheme());
  if (themeBtn) {
    themeBtn.addEventListener("click", () => {
      const cur = readTheme();
      const next = cur === "auto" ? "dark" : (cur === "dark" ? "light" : "auto");
      writeTheme(next);
      applyTheme(next);
    });
  }

  const input = document.getElementById("doc-filter");
  const rows = Array.from(document.querySelectorAll("#doc-table tbody tr"));
  const count = document.getElementById("doc-count");
  function update() {
    const q = (input.value || "").toLowerCase().trim();
    let shown = 0;
    for (const tr of rows) {
      const hay = (tr.dataset.hay || "");
      const show = !q || hay.includes(q);
      tr.style.display = show ? "" : "none";
      if (show) shown++;
    }
    count.textContent = `${shown} shown / ${rows.length} total`;
  }
  input.addEventListener("input", update);
  update();
})();
</script>
</body></html>"##,
    );

    fs::write(output_path, index)
        .map_err(|e| format!("Failed to write {:?}: {}", output_path, e))?;
    Ok(())
}

/// Run the info subcommand
fn run_info(dataset: &str) -> Result<(), String> {
    println!();

    #[cfg(feature = "eval")]
    {
        use crate::eval::dataset_registry::DatasetId as RegistryDatasetId;
        use crate::eval::loader::LoadableDatasetId;

        // Try to find in registry (by display name or variant name)
        let registry_match = RegistryDatasetId::all()
            .iter()
            .find(|d| {
                // Match by display name (case-insensitive)
                d.name().eq_ignore_ascii_case(dataset)
                    // Or by variant name (e.g., "BroadTwitterCorpus")
                    || format!("{:?}", d).eq_ignore_ascii_case(dataset)
            })
            .copied();

        // Consider a dataset "loadable" if it has a loader implementation.
        let loadable_match = registry_match.and_then(|rid| LoadableDatasetId::try_from(rid).ok());

        if let Some(registry_id) = registry_match {
            // Show registry metadata
            println!(
                "{}",
                color("1;36", &format!("Dataset: {}", registry_id.name()))
            );
            println!();

            // Basic info
            println!("  Description: {}", registry_id.description());
            println!("  Language:    {}", registry_id.language());
            println!("  Domain:      {}", registry_id.domain());

            // Optional metadata
            if let Some(year) = registry_id.year() {
                println!("  Year:        {}", year);
            }
            if let Some(citation) = registry_id.citation() {
                println!("  Citation:    {}", citation);
            }
            if let Some(license) = registry_id.license() {
                println!("  License:     {}", license);
            }
            if let Some(paper_url) = registry_id.paper_url() {
                println!("  Paper:       {}", paper_url);
            }
            if let Some(size_hint) = registry_id.size_hint() {
                println!("  Size:        {}", size_hint);
            }

            // Entity types
            let entity_types = registry_id.entity_types();
            if !entity_types.is_empty() {
                println!("  Entity types: {}", entity_types.join(", "));
            }

            // Task capabilities
            println!();
            println!("  Tasks:");
            if registry_id.is_ner() {
                println!("    - Named Entity Recognition");
            }
            if registry_id.is_coreference() {
                println!("    - Coreference Resolution");
            }
            if registry_id.is_biomedical() {
                println!("    (Biomedical domain)");
            }

            // Check if loadable
            println!();
            if loadable_match.is_some() {
                println!(
                    "  Status: {} (can be downloaded)",
                    color("1;32", "Loadable")
                );

                // Try to load and show stats if eval-advanced is enabled
                #[cfg(feature = "eval-advanced")]
                {
                    use crate::eval::loader::DatasetLoader;
                    if let Some(loadable_id) = loadable_match {
                        let loader = DatasetLoader::new()
                            .map_err(|e| format!("Failed to create loader: {}", e))?;

                        match loader.load(loadable_id) {
                            Ok(loaded) => {
                                let stats = loaded.stats();
                                println!();
                                println!("  Loaded Statistics:");
                                println!("    Sentences: {}", stats.sentences);
                                println!("    Tokens:    {}", stats.tokens);
                                println!("    Entities:  {}", stats.entities);
                                if stats.sentences > 0 {
                                    println!(
                                        "    Avg entities/sentence: {:.2}",
                                        stats.entities as f64 / stats.sentences as f64
                                    );
                                }
                            }
                            Err(e) => {
                                println!("  (Could not load: {})", e);
                            }
                        }
                    }
                }
            } else {
                let access_status = registry_id.access_status();
                println!(
                    "  Status: {} ({})",
                    color("1;33", "Not loadable"),
                    access_status.as_str()
                );
            }
        } else {
            return Err(format!(
                "Unknown dataset '{}'. Use 'anno dataset list' to see available datasets.",
                dataset
            ));
        }
    }

    #[cfg(not(feature = "eval"))]
    {
        println!("Dataset: {}", dataset);
        println!();
        println!("Note: Full dataset info requires --features eval");
    }

    println!();
    Ok(())
}

/// Run the list subcommand
fn run_list(
    task_filter: Option<String>,
    domain_filter: Option<String>,
    loadable_only: bool,
    verbose: bool,
) -> Result<(), String> {
    println!();
    println!("{}", color("1;36", "Available Datasets"));
    println!();

    #[cfg(feature = "eval")]
    {
        use crate::eval::dataset_registry::DatasetId as RegistryDatasetId;
        use crate::eval::loader::LoadableDatasetId;
        use crate::eval::task_mapping::Task;

        fn task_predicate(task_raw: &str) -> Result<Box<dyn Fn(&[Task]) -> bool>, String> {
            let t = task_raw.trim().to_lowercase();
            match t.as_str() {
                // NER family
                "ner" | "sequence_labeling" | "nested-ner" | "mner" | "pii_detection"
                | "slot_filling" => Ok(Box::new(|tasks| tasks.contains(&Task::NER))),

                // Coref family (both intra + inter + abstract anaphora)
                "coref" => Ok(Box::new(|tasks| tasks.iter().any(|x| x.is_coref_family()))),
                "intra-coref" | "intra_coref" | "intracoref" => {
                    Ok(Box::new(|tasks| tasks.contains(&Task::IntraDocCoref)))
                }
                "inter-coref" | "inter_coref" | "intercoref" | "cdcr" | "coalesce"
                | "event_coref" => Ok(Box::new(|tasks| tasks.contains(&Task::InterDocCoref))),

                // Relation extraction
                "re" | "rel" | "relation" | "relation_extraction" | "relation-extraction" => Ok(
                    Box::new(|tasks| tasks.contains(&Task::RelationExtraction)),
                ),

                // Entity linking / disambiguation
                "el" | "ned" | "entity_linking" | "entity-linking" => {
                    Ok(Box::new(|tasks| tasks.contains(&Task::NED)))
                }

                _ => Err(format!(
                    "Unknown --task '{}'. Expected one of: ner, coref, intra-coref, inter-coref, re, el",
                    task_raw
                )),
            }
        }

        if loadable_only {
            let task_pred = match task_filter.as_deref() {
                Some(t) => Some(task_predicate(t)?),
                None => None,
            };

            // Show only loadable datasets
            let loadable_datasets: Vec<RegistryDatasetId> = RegistryDatasetId::all()
                .iter()
                .copied()
                .filter(|id| LoadableDatasetId::try_from(*id).is_ok())
                .collect();
            println!(
                "  {} loadable datasets (can be downloaded and parsed):",
                loadable_datasets.len()
            );
            println!();

            for id in loadable_datasets {
                let name = id.name();
                if let Some(ref task_pred) = task_pred {
                    let tasks = id.tasks_typed();
                    if !task_pred(&tasks) {
                        continue;
                    }
                }

                if verbose {
                    let citation = id.citation().unwrap_or("N/A");
                    let license = id.license().unwrap_or("Unknown");
                    let year = id
                        .year()
                        .map(|y| y.to_string())
                        .unwrap_or_else(|| "N/A".to_string());
                    println!("    {:<20} [{:>4}] {} ({})", name, year, citation, license);
                } else {
                    println!("    {}", name);
                }
            }
        } else {
            // Show registry (full catalog)
            let all_datasets: Vec<_> = RegistryDatasetId::all().iter().collect();
            let loadable_count = RegistryDatasetId::all()
                .iter()
                .copied()
                .filter(|id| LoadableDatasetId::try_from(*id).is_ok())
                .count();
            let automatable_count = RegistryDatasetId::all()
                .iter()
                .copied()
                .filter(|id| id.is_automatable_download())
                .count();
            println!(
                "  {} datasets in registry ({} loadable):",
                all_datasets.len(),
                loadable_count
            );
            println!();

            // Group by domain if no filter
            if domain_filter.is_none() && task_filter.is_none() {
                // Just show counts by category
                let ner_count = all_datasets.iter().filter(|d| d.is_ner()).count();
                let coref_count = all_datasets.iter().filter(|d| d.is_coreference()).count();
                let bio_count = all_datasets.iter().filter(|d| d.is_biomedical()).count();

                println!("    NER datasets:           {}", ner_count);
                println!("    Coreference datasets:   {}", coref_count);
                println!("    Biomedical datasets:    {}", bio_count);
                println!("    Automatable downloads:  {}", automatable_count);
                println!();
                println!("  Use --loadable to see only datasets with loader implementations");
                println!("  Use --task ner/coref/re/el to filter by task");
                println!("  Use --verbose for more details");
            } else {
                let task_pred = match task_filter.as_deref() {
                    Some(t) => Some(task_predicate(t)?),
                    None => None,
                };

                // Filter and display
                for dataset in &all_datasets {
                    // Apply task filter
                    if let Some(ref task_pred) = task_pred {
                        let tasks = dataset.tasks_typed();
                        if !task_pred(&tasks) {
                            continue;
                        }
                    }

                    // Apply domain filter
                    if let Some(ref domain) = domain_filter {
                        let dataset_domain = dataset.domain().to_lowercase();
                        if !dataset_domain.contains(&domain.to_lowercase()) {
                            continue;
                        }
                    }

                    if verbose {
                        let citation = dataset.citation().unwrap_or("N/A");
                        let year = dataset
                            .year()
                            .map(|y| y.to_string())
                            .unwrap_or_else(|| "----".to_string());
                        println!("    {:<25} [{:>4}] {}", dataset.name(), year, citation);
                    } else {
                        println!("    {}", dataset.name());
                    }
                }
            }
        }
    }

    #[cfg(not(feature = "eval"))]
    {
        let _ = (task_filter, domain_filter, loadable_only, verbose);
        println!("  Synthetic (always available):");
        println!("    - synthetic   : Generated test cases");
        println!("    - robustness  : Adversarial perturbations");
        println!();
        println!("  Note: Full dataset catalog requires --features eval");
    }

    println!();
    Ok(())
}

#[cfg(feature = "eval")]
fn run_check(issues_only: bool, dataset: Option<&str>, _fix: bool) -> Result<(), String> {
    use crate::eval::dataset_registry::DatasetId as RegistryDatasetId;
    use crate::eval::loader::LoadableDatasetId;

    println!();
    println!("{}", color("1;36", "Dataset Metadata Check"));
    println!();

    let errors: Vec<String> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();
    let mut info: Vec<String> = Vec::new();

    // Check if specific dataset or all
    let datasets_to_check: Vec<RegistryDatasetId> = if let Some(ds_name) = dataset {
        // Try to find in registry
        RegistryDatasetId::all()
            .iter()
            .find(|d| {
                d.name().eq_ignore_ascii_case(ds_name)
                    || format!("{:?}", d).eq_ignore_ascii_case(ds_name)
            })
            .copied()
            .map(|d| vec![d])
            .ok_or_else(|| format!("Dataset '{}' not found in registry", ds_name))?
    } else {
        RegistryDatasetId::all().to_vec()
    };

    for registry_id in &datasets_to_check {
        let is_loadable = LoadableDatasetId::try_from(*registry_id).is_ok();

        // Check 1: Missing download URL
        let url = registry_id.download_url();
        if url.is_empty() {
            let access_status = registry_id.access_status();
            // Local datasets are allowed to have no URL (they live in-repo / on-disk).
            if access_status == crate::eval::dataset_registry::DatasetAccessibility::Local {
                // no-op
            } else if access_status.is_automatable() {
                warnings.push(format!(
                    "{}: Missing download URL but marked as automatable ({})",
                    registry_id.name(),
                    access_status.as_str()
                ));
            } else {
                info.push(format!(
                    "{}: No URL (requires {})",
                    registry_id.name(),
                    access_status.as_str()
                ));
            }
        }

        // Check 2: Generic entity types
        //
        // Only meaningful for NER-family datasets; many datasets in the registry are
        // coref/RE/EL/bias benchmarks where "entity types" are not a first-class concept.
        let entity_types = registry_id.entity_types();
        if registry_id.supports_ner() && entity_types.is_empty() {
            warnings.push(format!("{}: Missing entity_types", registry_id.name()));
        } else if registry_id.supports_ner() && entity_types == ["ENTITY"] {
            warnings.push(format!(
                "{}: Using generic entity type 'ENTITY' (should specify actual types)",
                registry_id.name()
            ));
        }

        // Check 3: Generic domain
        let domain = registry_id.domain();
        if domain == "general" && !registry_id.is_multilingual() {
            warnings.push(format!(
                "{}: Using generic domain 'general' (should specify actual domain)",
                registry_id.name()
            ));
        }

        // If something is automatable *in practice*, but we don't have a loader implementation,
        // that's actionable (it will never be sampled in evals without manual glue).
        // Only warn for datasets that have NER/coref/RE tasks (our loader scope).
        let has_loadable_task =
            registry_id.supports_ner() || registry_id.supports_coref() || registry_id.supports_re();
        if registry_id.access_status().is_automatable()
            && registry_id.is_automatable_download()
            && !is_loadable
            && has_loadable_task
        {
            warnings.push(format!(
                "{}: Automatable access_status ({}) but not loadable (missing loader impl)",
                registry_id.name(),
                registry_id.access_status().as_str()
            ));
        }
    }

    // Summary
    if !issues_only {
        println!("Checked {} datasets", datasets_to_check.len());
        println!();
    }

    if !errors.is_empty() {
        println!("{} {} Errors:", color("31", "✗"), errors.len());
        for err in &errors {
            println!("  {}", err);
        }
        println!();
    }

    if !warnings.is_empty() {
        println!("{} {} Warnings:", color("33", "!"), warnings.len());
        for warn in &warnings {
            println!("  {}", warn);
        }
        println!();
    }

    if !issues_only && !info.is_empty() {
        println!("{} {} Info:", color("36", "i"), info.len());
        for msg in &info {
            println!("  {}", msg);
        }
        println!();
    }

    // Statistics
    if !issues_only {
        let registry_count = RegistryDatasetId::all().len();
        let loadable_count = LoadableDatasetId::all().len();
        let with_urls = datasets_to_check
            .iter()
            .filter(|d| !d.download_url().is_empty())
            .count();
        let with_entity_types = datasets_to_check
            .iter()
            .filter(|d| {
                let types = d.entity_types();
                !types.is_empty() && types != ["ENTITY"]
            })
            .count();

        println!("Statistics:");
        println!("  Registry datasets:    {}", registry_count);
        println!("  Loadable datasets:   {}", loadable_count);
        println!("  With download URLs: {}", with_urls);
        println!("  With entity types:   {}", with_entity_types);
        println!();
    }

    if !errors.is_empty() {
        return Err(format!("Found {} errors", errors.len()));
    }

    if issues_only && warnings.is_empty() && errors.is_empty() {
        println!("{} No issues found", color("32", "✓"));
    } else if !issues_only && errors.is_empty() && warnings.is_empty() {
        println!("{} All checks passed", color("32", "✓"));
    }

    Ok(())
}

#[cfg(not(feature = "eval"))]
fn run_check(_issues_only: bool, _dataset: Option<&str>, _fix: bool) -> Result<(), String> {
    println!("Dataset checking requires --features eval");
    Ok(())
}

#[cfg(feature = "eval-advanced")]
fn run_check_health(
    dataset: Option<&str>,
    all: bool,
    relaxed: bool,
    verbose: bool,
    workers: usize,
    timeout: u64,
) -> Result<(), String> {
    use crate::eval::dataset_registry::DatasetId;
    use std::sync::mpsc;
    use std::thread;

    println!();
    println!("{}", color("1;36", "Dataset URL Health Check"));
    println!();

    crate::env::load_dotenv();
    let allow_manual = matches!(
        std::env::var("ANNO_DATASET_ALLOW_MANUAL").as_deref(),
        Ok("1") | Ok("true") | Ok("yes")
    );

    // Determine which datasets to check
    let datasets_to_check: Vec<DatasetId> = if let Some(ds_name) = dataset {
        // Try to parse as DatasetId
        ds_name
            .parse::<DatasetId>()
            .map(|id| vec![id])
            .map_err(|e| format!("Invalid dataset '{}': {}", ds_name, e))?
    } else {
        let total = DatasetId::all().len();
        let with_urls: Vec<DatasetId> = DatasetId::all()
            .iter()
            .copied()
            .filter(|d| !d.download_url().is_empty())
            .collect();

        if verbose {
            println!("Dataset filtering breakdown:");
            println!("  Total in registry: {}", total);
            println!("  With URLs: {}", with_urls.len());
        }

        let mut excluded_not_automatable = 0;
        let mut excluded_not_download = 0;
        let mut excluded_no_token = 0;

        let mut candidates: Vec<DatasetId> = with_urls
            .iter()
            .copied()
            .filter(|d| {
                if allow_manual {
                    return true;
                }
                if !d.access_status().is_automatable() {
                    excluded_not_automatable += 1;
                    return false;
                }
                if !d.is_automatable_download() {
                    excluded_not_download += 1;
                    return false;
                }
                if d.requires_hf_token() && !crate::env::has_hf_token() {
                    excluded_no_token += 1;
                    return false;
                }
                true
            })
            .collect();

        if verbose {
            println!("  After filtering: {}", candidates.len());
            if excluded_not_automatable > 0 {
                println!(
                    "    Excluded (not automatable): {}",
                    excluded_not_automatable
                );
            }
            if excluded_not_download > 0 {
                println!(
                    "    Excluded (not automatable download): {}",
                    excluded_not_download
                );
            }
            if excluded_no_token > 0 {
                println!("    Excluded (requires HF_TOKEN): {}", excluded_no_token);
            }
        }

        if !all {
            let before_truncate = candidates.len();
            candidates.truncate(50);
            if verbose && before_truncate > 50 {
                println!("  After truncation (--all not set): {}", candidates.len());
            }
        }

        candidates
    };

    if datasets_to_check.is_empty() {
        println!("No datasets to check (no URLs or dataset not found)");
        return Ok(());
    }

    if verbose {
        println!();
        println!(
            "Final selection: {} datasets to check",
            datasets_to_check.len()
        );
        println!();
    }

    println!("Checking {} dataset URLs...", datasets_to_check.len());
    println!();

    // Use a channel to collect results
    let (tx, rx) = mpsc::channel();
    let mut handles = Vec::new();

    // Spawn worker threads
    for dataset_id in datasets_to_check {
        let tx = tx.clone();
        let url = dataset_id.download_url().to_string();
        let timeout_secs = timeout;

        let handle = thread::spawn(move || {
            let result = check_single_url(dataset_id.name(), &url, timeout_secs);
            tx.send((dataset_id, result)).ok();
        });

        handles.push(handle);

        // Limit concurrent workers
        if handles.len() >= workers {
            // Wait for one to complete
            for handle in handles.drain(..1) {
                handle.join().ok();
            }
        }
    }

    // Wait for remaining threads
    for handle in handles {
        handle.join().ok();
    }

    drop(tx); // Close sender so receiver knows we're done

    // Collect results
    let mut results: Vec<(DatasetId, URLHealthResult)> = Vec::new();
    while let Ok((dataset_id, result)) = rx.recv() {
        results.push((dataset_id, result));
    }

    // Sort by name for consistent output
    results.sort_by_key(|(dataset_id, _)| dataset_id.name());

    // Count and display results
    let mut ok_count = 0;
    let mut strict_error_count = 0;
    let mut relaxed_error_count = 0;
    let mut redirect_count = 0;

    for (dataset_id, result) in &results {
        let name = dataset_id.name();
        match result.status.as_str() {
            "ok" => {
                ok_count += 1;
                if dataset.is_some() {
                    // Show details for single dataset check
                    println!(
                        "  {} {} ({})",
                        color("32", "OK"),
                        name,
                        result.code.unwrap_or(0)
                    );
                }
            }
            "redirect" => {
                redirect_count += 1;
                ok_count += 1; // Redirects usually work
                if dataset.is_some() {
                    println!(
                        "  {} {} ({}): {}",
                        color("33", "REDIRECT"),
                        name,
                        result.code.unwrap_or(0),
                        result.message
                    );
                }
            }
            "missing" => {
                if dataset.is_some() {
                    println!("  {} {}: No URL", color("33", "SKIP"), name);
                }
            }
            _ => {
                let expected_automatable = dataset_id.access_status().is_automatable()
                    && dataset_id.is_automatable_download()
                    && (!dataset_id.requires_hf_token() || crate::env::has_hf_token());

                if expected_automatable {
                    strict_error_count += 1;
                    println!("  {} {}: {}", color("31", "ERROR"), name, result.message);
                } else {
                    relaxed_error_count += 1;
                    println!("  {} {}: {}", color("33", "WARN"), name, result.message);
                    println!("      (non-automatable source per registry metadata)");
                }
                if let Some(code) = result.code {
                    println!("      HTTP {}", code);
                }
            }
        }
    }

    println!();
    if relaxed {
        println!(
            "Summary: {} OK ({} redirects), {} errors, {} warnings",
            ok_count, redirect_count, strict_error_count, relaxed_error_count
        );
        if strict_error_count > 0 {
            return Err(format!(
                "{} expected-automatable URLs failed health check",
                strict_error_count
            ));
        }
    } else {
        let total_errors = strict_error_count + relaxed_error_count;
        println!(
            "Summary: {} OK ({} redirects), {} errors",
            ok_count, redirect_count, total_errors
        );
        if total_errors > 0 {
            return Err(format!("{} URLs failed health check", total_errors));
        }
    }

    Ok(())
}

#[cfg(not(feature = "eval-advanced"))]
fn run_check_health(
    _dataset: Option<&str>,
    _all: bool,
    _relaxed: bool,
    _verbose: bool,
    _workers: usize,
    _timeout: u64,
) -> Result<(), String> {
    println!("URL health checking requires --features eval-advanced");
    Ok(())
}

#[cfg(feature = "eval-advanced")]
struct URLHealthResult {
    status: String,
    code: Option<u16>,
    message: String,
}

#[cfg(feature = "eval-advanced")]
fn check_single_url(_name: &str, url: &str, timeout_secs: u64) -> URLHealthResult {
    if url.is_empty() {
        return URLHealthResult {
            status: "missing".to_string(),
            code: None,
            message: "No URL defined".to_string(),
        };
    }

    // Skip non-HTTP URLs
    if !url.starts_with("http://") && !url.starts_with("https://") {
        return URLHealthResult {
            status: "skip".to_string(),
            code: None,
            message: "Non-HTTP URL".to_string(),
        };
    }

    // Use ureq for simple HTTP checking (already a dependency)
    match ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_secs(timeout_secs))
        .build()
        .head(url)
        .call()
    {
        Ok(response) => {
            let status = response.status();
            if status == 200 {
                URLHealthResult {
                    status: "ok".to_string(),
                    code: Some(status),
                    message: "OK".to_string(),
                }
            } else if (300..400).contains(&status) {
                URLHealthResult {
                    status: "redirect".to_string(),
                    code: Some(status),
                    message: format!(
                        "Redirects to {}",
                        response.header("Location").unwrap_or("unknown")
                    ),
                }
            } else if status == 405 {
                // HEAD not allowed, try GET
                match ureq::get(url)
                    .timeout(std::time::Duration::from_secs(timeout_secs))
                    .call()
                {
                    Ok(resp) => URLHealthResult {
                        status: "ok".to_string(),
                        code: Some(resp.status()),
                        message: "OK (HEAD not allowed)".to_string(),
                    },
                    Err(e) => URLHealthResult {
                        status: "error".to_string(),
                        code: None,
                        message: format!("GET failed: {}", e),
                    },
                }
            } else {
                URLHealthResult {
                    status: "error".to_string(),
                    code: Some(status),
                    message: format!("HTTP {}", status),
                }
            }
        }
        Err(ureq::Error::Status(code, _)) => URLHealthResult {
            status: "error".to_string(),
            code: Some(code),
            message: format!("HTTP {}", code),
        },
        Err(ureq::Error::Transport(e)) => URLHealthResult {
            status: "error".to_string(),
            code: None,
            message: format!("Connection error: {}", e),
        },
    }
}

// =============================================================================
// Facet report (registry vs touched)
// =============================================================================

#[cfg(feature = "eval")]
fn run_facets(touched_report: Option<&str>, gaps: bool) -> Result<(), String> {
    use std::collections::{BTreeMap, BTreeSet};
    use std::fs;
    use std::io::Write;

    use crate::eval::loader::DatasetId;

    fn bump(map: &mut BTreeMap<String, u64>, k: impl Into<String>) {
        *map.entry(k.into()).or_insert(0) += 1;
    }

    macro_rules! outln {
        ($out:expr, $($arg:tt)*) => {{
            if let Err(e) = writeln!($out, $($arg)*) {
                if e.kind() == std::io::ErrorKind::BrokenPipe {
                    return Ok(());
                }
                return Err(format!("stdout: {}", e));
            }
        }};
    }

    fn pct(n: u64, d: u64) -> f64 {
        if d == 0 {
            0.0
        } else {
            (n as f64) * 100.0 / (d as f64)
        }
    }

    fn touched_set_from_report(path: &str) -> Result<BTreeSet<String>, String> {
        let content = fs::read_to_string(path).map_err(|e| format!("read {}: {}", path, e))?;
        let v: serde_json::Value =
            serde_json::from_str(&content).map_err(|e| format!("parse {}: {}", path, e))?;
        let obj = v
            .get("chosen_dataset_counts")
            .and_then(|x| x.as_object())
            .ok_or_else(|| "distribution JSON missing chosen_dataset_counts".to_string())?;
        Ok(obj.keys().cloned().collect())
    }

    let all: Vec<DatasetId> = DatasetId::all().to_vec();
    let total = all.len() as u64;

    let touched: Option<BTreeSet<String>> = match touched_report {
        None => None,
        Some(p) if p.trim().is_empty() => None,
        Some(p) => Some(touched_set_from_report(p.trim())?),
    };
    let touched_total = touched.as_ref().map(|set| set.len() as u64).unwrap_or(0);

    let mut out = std::io::BufWriter::new(std::io::stdout());

    // --- aggregate all ---
    let mut lang_all: BTreeMap<String, u64> = BTreeMap::new();
    let mut domain_all: BTreeMap<String, u64> = BTreeMap::new();
    let mut access_all: BTreeMap<String, u64> = BTreeMap::new();
    let mut cat_all: BTreeMap<String, u64> = BTreeMap::new();
    let mut tasks_all: BTreeMap<String, u64> = BTreeMap::new();

    // --- aggregate touched ---
    let mut lang_touched: BTreeMap<String, u64> = BTreeMap::new();
    let mut domain_touched: BTreeMap<String, u64> = BTreeMap::new();
    let mut access_touched: BTreeMap<String, u64> = BTreeMap::new();
    let mut cat_touched: BTreeMap<String, u64> = BTreeMap::new();
    let mut tasks_touched: BTreeMap<String, u64> = BTreeMap::new();

    for ds in &all {
        bump(&mut lang_all, ds.language());
        bump(&mut domain_all, ds.domain());
        bump(&mut access_all, format!("{:?}", ds.access_status()));
        for &c in ds.categories() {
            bump(&mut cat_all, c);
        }
        for &t in ds.tasks() {
            bump(&mut tasks_all, t);
        }

        if let Some(tset) = touched.as_ref() {
            let name = format!("{ds:?}");
            if tset.contains(&name) {
                bump(&mut lang_touched, ds.language());
                bump(&mut domain_touched, ds.domain());
                bump(&mut access_touched, format!("{:?}", ds.access_status()));
                for &c in ds.categories() {
                    bump(&mut cat_touched, c);
                }
                for &t in ds.tasks() {
                    bump(&mut tasks_touched, t);
                }
            }
        }
    }

    outln!(&mut out, "Datasets: {}", total);
    if let Some(p) = touched_report {
        if touched.is_some() {
            outln!(&mut out, "Touched: {} (from {})", touched_total, p.trim());
        }
    }

    fn print_top(
        out: &mut impl Write,
        title: &str,
        map: &BTreeMap<String, u64>,
        top: usize,
        denom: u64,
    ) -> Result<(), String> {
        outln!(out, "");
        outln!(out, "{} (top {})", title, top);
        let mut items: Vec<(&String, &u64)> = map.iter().collect();
        items.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
        for (k, v) in items.into_iter().take(top) {
            outln!(out, "  {:22} {:5} ({:4.1}%)", k, v, pct(*v, denom));
        }
        Ok(())
    }

    print_top(&mut out, "Languages (all)", &lang_all, 12, total)?;
    print_top(&mut out, "Domains (all)", &domain_all, 12, total)?;
    print_top(&mut out, "Access (all)", &access_all, 12, total)?;
    print_top(&mut out, "Categories (all)", &cat_all, 15, total)?;
    print_top(&mut out, "Tasks (all)", &tasks_all, 15, total)?;

    if touched.is_some() {
        print_top(
            &mut out,
            "Languages (touched)",
            &lang_touched,
            12,
            touched_total,
        )?;
        print_top(
            &mut out,
            "Domains (touched)",
            &domain_touched,
            12,
            touched_total,
        )?;
        print_top(
            &mut out,
            "Access (touched)",
            &access_touched,
            12,
            touched_total,
        )?;
        print_top(
            &mut out,
            "Categories (touched)",
            &cat_touched,
            15,
            touched_total,
        )?;
        print_top(
            &mut out,
            "Tasks (touched)",
            &tasks_touched,
            15,
            touched_total,
        )?;
    }

    if gaps && touched.is_some() {
        fn gap_table(
            out: &mut impl Write,
            title: &str,
            all: &BTreeMap<String, u64>,
            touched: &BTreeMap<String, u64>,
            total: u64,
            touched_total: u64,
        ) -> Result<(), String> {
            outln!(out, "");
            outln!(out, "Under-touched {} (min_all=6)", title);
            let mut rows: Vec<(f64, String, u64, u64, f64, f64)> = Vec::new();
            for (k, na) in all {
                if *na < 6 {
                    continue;
                }
                let nt = *touched.get(k).unwrap_or(&0);
                let pa = (*na as f64) / (total as f64);
                let pt = (nt as f64) / (touched_total as f64);
                rows.push((
                    (pt - pa) * 100.0,
                    k.clone(),
                    *na,
                    nt,
                    pa * 100.0,
                    pt * 100.0,
                ));
            }
            rows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            for (gap_pp, k, na, nt, pa, pt) in rows.into_iter().take(12) {
                outln!(
                    out,
                    "  {:22} all={:4} ({:4.1}%) touched={:4} ({:4.1}%) gap={:5.1}pp",
                    k,
                    na,
                    pa,
                    nt,
                    pt,
                    gap_pp
                );
            }
            Ok(())
        }

        gap_table(
            &mut out,
            "domains",
            &domain_all,
            &domain_touched,
            total,
            touched_total,
        )?;
        gap_table(
            &mut out,
            "access",
            &access_all,
            &access_touched,
            total,
            touched_total,
        )?;
        gap_table(
            &mut out,
            "categories",
            &cat_all,
            &cat_touched,
            total,
            touched_total,
        )?;
        gap_table(
            &mut out,
            "tasks",
            &tasks_all,
            &tasks_touched,
            total,
            touched_total,
        )?;
    }

    Ok(())
}
