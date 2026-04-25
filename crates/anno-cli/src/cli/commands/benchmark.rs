//! Benchmark command - Comprehensive evaluation across all task-dataset-backend combinations

use clap::{ArgAction, Parser, ValueEnum};

#[cfg(feature = "eval")]
use anno_eval::eval::loader::DatasetId;
#[cfg(feature = "eval")]
use anno_eval::eval::task_evaluator::TaskEvaluator;
#[cfg(feature = "eval")]
use anno_eval::eval::task_mapping::Task;

/// Comprehensive evaluation across all task-dataset-backend combinations
#[derive(Parser, Debug)]
pub struct BenchmarkArgs {
    /// Optional curated profile (panel) to make comparisons meaningful.
    ///
    /// If provided, and you do not explicitly pass `--tasks/--datasets/--backends`,
    /// the benchmark will use a curated slice instead of “everything”.
    #[arg(long, value_enum)]
    pub profile: Option<BenchmarkProfile>,

    /// Multi-seed mode: run the benchmark for each seed (comma-separated).
    ///
    /// If set, this runs one full benchmark per seed and writes per-seed artifacts by
    /// suffixing the `--output/--output-json` paths with `-seed<SEED>`.
    #[arg(long, value_delimiter = ',')]
    pub seeds: Option<Vec<u64>>,

    /// When multiple backends are requested, run heavy ONNX backends in separate processes.
    ///
    /// This reduces memory coupling (notably GLiNER) at the cost of multiple runs.
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    pub split_heavy_backends: bool,

    /// Tasks to evaluate (comma-separated: ner,coref,relation). Default: all
    #[arg(short, long, value_delimiter = ',')]
    pub tasks: Option<Vec<String>>,

    /// Datasets to use (comma-separated). Default: all suitable datasets
    #[arg(short, long, value_delimiter = ',')]
    pub datasets: Option<Vec<String>>,

    /// Backends to test (comma-separated). Default: all compatible backends
    #[arg(short, long, value_delimiter = ',')]
    pub backends: Option<Vec<String>>,

    /// Maximum examples per dataset (for quick testing)
    #[arg(short, long)]
    pub max_examples: Option<usize>,

    /// Random seed for sampling (for reproducibility and varied testing)
    #[arg(long)]
    pub seed: Option<u64>,

    /// Only use cached datasets (skip downloads)
    #[arg(long)]
    pub cached_only: bool,

    /// Output file for markdown report (default: stdout)
    #[arg(short, long)]
    pub output: Option<String>,

    /// Optional JSON output path (machine-readable results).
    ///
    /// This writes the full `ComprehensiveEvalResults` structure so other tools can
    /// aggregate/rank backends without parsing markdown.
    #[arg(long, value_name = "PATH")]
    pub output_json: Option<String>,
}

/// Curated benchmark profiles (panels) for meaningful comparisons.
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum BenchmarkProfile {
    /// Standard NER label space (PER/ORG/LOC/MISC-ish).
    NerStandard,
    /// Zero-shot NER on multilingual/fine-grained datasets (compare zero-shot backends only).
    NerZeroshotMultilingual,
    /// Coreference sanity (single dataset).
    CorefStandard,
    /// Relation extraction sanity (single dataset).
    RelationStandard,
}

fn suffix_path(p: &str, suffix: &str) -> String {
    match p.rsplit_once('.') {
        Some((stem, ext)) => format!("{stem}{suffix}.{ext}"),
        None => format!("{p}{suffix}"),
    }
}

fn is_heavy_backend(name: &str) -> bool {
    matches!(
        name,
        "gliner_onnx" | "gliner_multitask" | "gliner_poly" | "gliner_pii" | "gliner_relex"
    )
}

fn profile_defaults(profile: BenchmarkProfile) -> (Vec<String>, Vec<String>, Vec<String>) {
    match profile {
        BenchmarkProfile::NerStandard => (
            vec!["ner".to_string()],
            vec![
                "WikiGold".to_string(),
                "Wnut17".to_string(),
                "CoNLL2003Sample".to_string(),
            ],
            vec![
                "bert_onnx".to_string(),
                "stacked".to_string(),
                "heuristic".to_string(),
            ],
        ),
        BenchmarkProfile::NerZeroshotMultilingual => (
            vec!["ner".to_string()],
            vec![
                "WikiANN".to_string(),
                "MasakhaNER".to_string(),
                "MultiNERD".to_string(),
                "MultiCoNERv2".to_string(),
            ],
            vec!["gliner_onnx".to_string(), "nuner".to_string()],
        ),
        BenchmarkProfile::CorefStandard => (
            vec!["coref".to_string()],
            vec!["GAP".to_string()],
            vec!["coref_resolver".to_string()],
        ),
        BenchmarkProfile::RelationStandard => (
            vec!["relation".to_string()],
            vec!["DocRED".to_string()],
            vec!["tplinker".to_string()],
        ),
    }
}

/// Execute the benchmark command.
pub fn run(args: BenchmarkArgs) -> Result<(), String> {
    #[cfg(not(feature = "eval"))]
    {
        let _ = args;
        Err("Benchmark command requires --features eval".to_string())
    }

    #[cfg(feature = "eval")]
    {
        println!("=== Comprehensive Task-Dataset-Backend Evaluation ===\n");

        // Multi-seed: run one benchmark per seed and suffix outputs.
        if let Some(seeds) = &args.seeds {
            if seeds.is_empty() {
                return Err("--seeds was provided but empty".to_string());
            }
            for seed in seeds {
                let per = BenchmarkArgs {
                    profile: args.profile,
                    seeds: None,
                    split_heavy_backends: args.split_heavy_backends,
                    tasks: args.tasks.clone(),
                    datasets: args.datasets.clone(),
                    backends: args.backends.clone(),
                    max_examples: args.max_examples,
                    seed: Some(*seed),
                    cached_only: args.cached_only,
                    output: args
                        .output
                        .as_ref()
                        .map(|p| suffix_path(p, &format!("-seed{seed}"))),
                    output_json: args
                        .output_json
                        .as_ref()
                        .map(|p| suffix_path(p, &format!("-seed{seed}"))),
                };
                run(per)?;
            }
            return Ok(());
        }

        // Apply curated defaults if profile was provided and user didn't override.
        let (profile_tasks, profile_datasets, profile_backends) = match args.profile {
            Some(p) => profile_defaults(p),
            None => (Vec::new(), Vec::new(), Vec::new()),
        };

        let task_strs = args.tasks.clone().unwrap_or(profile_tasks);
        let dataset_strs = args.datasets.clone().unwrap_or(profile_datasets);
        let backend_strs = args.backends.clone().unwrap_or(profile_backends);

        // Parse tasks
        let tasks = if !task_strs.is_empty() {
            let mut parsed = Vec::new();
            for t in task_strs {
                match t.to_lowercase().as_str() {
                    "ner" | "ner_task" => parsed.push(Task::NER),
                    "coref" | "coreference" | "intradoc_coref" => parsed.push(Task::IntraDocCoref),
                    "relation" | "relation_extraction" => parsed.push(Task::RelationExtraction),
                    other => {
                        return Err(format!(
                            "Unknown task: {}. Use: ner, coref, relation",
                            other
                        ));
                    }
                }
            }
            parsed
        } else {
            Task::all().to_vec()
        };

        // Parse datasets
        let datasets = if !dataset_strs.is_empty() {
            let mut parsed = Vec::new();
            for d in dataset_strs {
                let dataset_id: DatasetId = d
                    .parse()
                    .map_err(|e| format!("Invalid dataset '{}': {}", d, e))?;
                parsed.push(dataset_id);
            }
            parsed
        } else {
            vec![] // Empty = use all suitable datasets
        };

        // Parse backends
        let backends = backend_strs;

        // Heavy backend splitting: run heavy ONNX backends in subprocesses to reduce memory coupling.
        //
        // If `--output-json` is provided, we merge subprocess JSON artifacts so the final report
        // still compares all requested backends in one table.
        if args.split_heavy_backends && backends.len() > 1 {
            let (heavy, light): (Vec<String>, Vec<String>) =
                backends.into_iter().partition(|b| is_heavy_backend(b));

            // Run each heavy backend as its own `anno benchmark` invocation (same args, one backend).
            for hb in &heavy {
                let exe = std::env::current_exe()
                    .map_err(|e| format!("Failed to locate current executable: {}", e))?;

                let mut cmd = std::process::Command::new(exe);
                cmd.arg("benchmark");
                if let Some(profile) = args.profile {
                    cmd.arg("--profile")
                        .arg(profile.to_possible_value().unwrap().get_name());
                }
                // Use canonical short codes for tasks (ensures we don't leak debug formatting).
                cmd.arg("--tasks")
                    .arg(tasks.iter().map(|t| t.code()).collect::<Vec<_>>().join(","));
                if !datasets.is_empty() {
                    cmd.arg("--datasets").arg(
                        datasets
                            .iter()
                            .map(|d| d.to_string())
                            .collect::<Vec<_>>()
                            .join(","),
                    );
                }
                cmd.arg("--backends").arg(hb);
                if let Some(max) = args.max_examples {
                    cmd.arg("--max-examples").arg(max.to_string());
                }
                if let Some(seed) = args.seed {
                    cmd.arg("--seed").arg(seed.to_string());
                }
                if args.cached_only {
                    cmd.arg("--cached-only");
                }
                if let Some(out) = &args.output {
                    cmd.arg("--output").arg(suffix_path(out, &format!("-{hb}")));
                }
                if let Some(outj) = &args.output_json {
                    cmd.arg("--output-json")
                        .arg(suffix_path(outj, &format!("-{hb}")));
                }
                // Disable splitting in child.
                cmd.arg("--split-heavy-backends=false");

                let status = cmd
                    .status()
                    .map_err(|e| format!("Failed to spawn benchmark subprocess: {}", e))?;
                if !status.success() {
                    return Err(format!("Heavy backend subprocess failed for '{hb}'"));
                }
            }

            // Continue with light backends in this process (if any).
            if light.is_empty() {
                return Ok(());
            }

            // If we cannot merge (no JSON path), fall back to the old behavior: emit
            // per-backend artifacts only.
            if args.output_json.is_none() {
                let per = BenchmarkArgs {
                    profile: args.profile,
                    seeds: None,
                    split_heavy_backends: false,
                    tasks: Some(tasks.iter().map(|t| t.code().to_string()).collect()),
                    datasets: if datasets.is_empty() {
                        None
                    } else {
                        Some(datasets.iter().map(|d| d.to_string()).collect())
                    },
                    backends: Some(light),
                    max_examples: args.max_examples,
                    seed: args.seed,
                    cached_only: args.cached_only,
                    output: args.output.clone(),
                    output_json: args.output_json.clone(),
                };
                return run(per);
            }

            // Otherwise: run light backends here, then merge heavy JSON artifacts into a single
            // combined report/JSON at the original `--output`/`--output-json` paths.
            use anno_eval::eval::config_builder::TaskEvalConfigBuilder;
            use anno_eval::eval::task_evaluator::{
                ComprehensiveEvalResults, EvalSummary, TaskEvalResult,
            };
            use std::collections::HashSet;

            fn summarize(results: &[TaskEvalResult]) -> EvalSummary {
                let skipped = results.iter().filter(|r| r.is_skipped()).count();
                let failed = results
                    .iter()
                    .filter(|r| !r.success && !r.is_skipped())
                    .count();

                let mut tasks: Vec<Task> = Vec::new();
                let mut datasets: Vec<DatasetId> = Vec::new();
                let mut backends: Vec<String> = Vec::new();

                for r in results {
                    if !tasks.contains(&r.task) {
                        tasks.push(r.task);
                    }
                    if !datasets.contains(&r.dataset) {
                        datasets.push(r.dataset);
                    }
                    if !backends.contains(&r.backend) {
                        backends.push(r.backend.clone());
                    }
                }

                EvalSummary {
                    total_combinations: results.len(),
                    successful: results.iter().filter(|r| r.success).count(),
                    failed,
                    skipped,
                    tasks,
                    datasets,
                    backends,
                }
            }

            let evaluator =
                TaskEvaluator::new().map_err(|e| format!("Failed to create evaluator: {}", e))?;

            let mut builder = TaskEvalConfigBuilder::new()
                .with_tasks(tasks.clone())
                .with_datasets(datasets.clone())
                .with_backends(light)
                .require_cached(args.cached_only)
                .with_confidence_intervals(true)
                .with_familiarity(true);

            if let Some(max) = args.max_examples {
                if max > 0 {
                    builder = builder.with_max_examples(max);
                }
            }
            if let Some(seed) = args.seed {
                builder = builder.with_seed(seed);
            }

            let config = builder.build();

            println!("Running comprehensive evaluation...");
            println!("Tasks: {:?}", config.tasks);
            if !config.datasets.is_empty() {
                println!("Datasets: {:?}", config.datasets);
            } else {
                println!("Datasets: all suitable datasets");
            }
            if !config.backends.is_empty() {
                println!("Backends: {:?}", config.backends);
            } else {
                println!("Backends: all compatible backends");
            }
            if let Some(max) = config.max_examples {
                println!("Max examples per dataset: {}", max);
            }
            if let Some(seed) = config.seed {
                println!("Random seed: {}", seed);
            }
            println!();

            let mut combined = evaluator
                .evaluate_all(config)
                .map_err(|e| format!("Evaluation failed: {}", e))?;

            let json_path = args
                .output_json
                .as_ref()
                .expect("checked output_json Some above");

            // Merge heavy artifacts (JSON) into the combined result set.
            // Deduplicate exact (task,dataset,backend) triples in case of retries.
            let mut seen: HashSet<(Task, DatasetId, String)> = combined
                .results
                .iter()
                .map(|r| (r.task, r.dataset, r.backend.clone()))
                .collect();

            for hb in &heavy {
                let p = suffix_path(json_path, &format!("-{hb}"));
                let raw = std::fs::read_to_string(&p)
                    .map_err(|e| format!("Failed to read heavy JSON artifact {}: {}", p, e))?;
                let parsed: ComprehensiveEvalResults = serde_json::from_str(&raw)
                    .map_err(|e| format!("Failed to parse heavy JSON artifact {}: {}", p, e))?;
                for r in parsed.results {
                    let key = (r.task, r.dataset, r.backend.clone());
                    if seen.insert(key) {
                        combined.results.push(r);
                    }
                }
            }

            combined.summary = summarize(&combined.results);

            // Print summary for the combined run.
            println!("=== Evaluation Summary ===");
            println!(
                "Total combinations: {}",
                combined.summary.total_combinations
            );
            println!("Successful: {}", combined.summary.successful);
            println!(
                "Skipped (feature not available): {}",
                combined.summary.skipped
            );
            println!("Failed (actual errors): {}", combined.summary.failed);
            println!("\nTasks evaluated: {}", combined.summary.tasks.len());
            println!("Datasets used: {}", combined.summary.datasets.len());
            println!("Backends tested: {}", combined.summary.backends.len());
            println!();

            // Write combined JSON and markdown to the original paths.
            let json = serde_json::to_string_pretty(&combined)
                .map_err(|e| format!("Failed to serialize results as JSON: {}", e))?;
            std::fs::write(json_path, json)
                .map_err(|e| format!("Failed to write JSON report to {}: {}", json_path, e))?;
            println!("JSON saved to: {}", json_path);

            let report = combined.to_markdown();
            if let Some(output_path) = &args.output {
                std::fs::write(output_path, &report)
                    .map_err(|e| format!("Failed to write report to {}: {}", output_path, e))?;
                println!("Report saved to: {}", output_path);
            } else {
                println!("=== Markdown Report ===");
                println!("{}", report);
            }

            return Ok(());
        }

        // Create evaluator
        let evaluator =
            TaskEvaluator::new().map_err(|e| format!("Failed to create evaluator: {}", e))?;

        // Configure evaluation using builder pattern
        use anno_eval::eval::config_builder::TaskEvalConfigBuilder;
        let mut builder = TaskEvalConfigBuilder::new()
            .with_tasks(tasks)
            .with_datasets(datasets)
            .with_backends(backends)
            .require_cached(args.cached_only)
            .with_confidence_intervals(true)
            .with_familiarity(true);

        // Set max_examples (None means "all examples", 0 also means "all examples")
        if let Some(max) = args.max_examples {
            if max > 0 {
                builder = builder.with_max_examples(max);
            }
            // If max == 0, don't set it (None = unlimited)
        }

        // Only set seed if provided (default is 42 in builder)
        if let Some(seed) = args.seed {
            builder = builder.with_seed(seed);
        }

        let config = builder.build();

        println!("Running comprehensive evaluation...");
        println!("Tasks: {:?}", config.tasks);
        if !config.datasets.is_empty() {
            println!("Datasets: {:?}", config.datasets);
        } else {
            println!("Datasets: all suitable datasets");
        }
        if !config.backends.is_empty() {
            println!("Backends: {:?}", config.backends);
        } else {
            println!("Backends: all compatible backends");
        }
        if let Some(max) = config.max_examples {
            println!("Max examples per dataset: {}", max);
        }
        if let Some(seed) = config.seed {
            println!("Random seed: {}", seed);
        }
        println!();

        // Run evaluation
        let results = evaluator
            .evaluate_all(config)
            .map_err(|e| format!("Evaluation failed: {}", e))?;

        // Print summary
        println!("=== Evaluation Summary ===");
        println!("Total combinations: {}", results.summary.total_combinations);
        println!("Successful: {}", results.summary.successful);
        println!(
            "Skipped (feature not available): {}",
            results.summary.skipped
        );
        println!("Failed (actual errors): {}", results.summary.failed);
        println!("\nTasks evaluated: {}", results.summary.tasks.len());
        println!("Datasets used: {}", results.summary.datasets.len());
        println!("Backends tested: {}", results.summary.backends.len());
        println!();

        // Generate markdown report
        let report = results.to_markdown();

        // Optional JSON artifact (stable-ish, for aggregation)
        if let Some(json_path) = &args.output_json {
            let json = serde_json::to_string_pretty(&results)
                .map_err(|e| format!("Failed to serialize results as JSON: {}", e))?;
            std::fs::write(json_path, json)
                .map_err(|e| format!("Failed to write JSON report to {}: {}", json_path, e))?;
            println!("JSON saved to: {}", json_path);
        }

        // Output report
        if let Some(output_path) = &args.output {
            std::fs::write(output_path, &report)
                .map_err(|e| format!("Failed to write report to {}: {}", output_path, e))?;
            println!("Report saved to: {}", output_path);
        } else {
            println!("=== Markdown Report ===");
            println!("{}", report);
        }

        Ok(())
    }
}
