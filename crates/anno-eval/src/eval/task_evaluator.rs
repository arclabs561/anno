//! Comprehensive Task-Dataset-Backend Evaluation System
//!
//! This module provides a unified evaluation framework that:
//! - Maps tasks to suitable datasets
//! - Maps datasets to compatible backends
//! - Runs evaluations across all valid combinations
//! - Generates comprehensive reports
//!
//! # Design Philosophy
//!
//! - **Trait-based**: Backend capabilities detected via trait implementations
//! - **Many-to-many**: Each task can use multiple datasets, each dataset can evaluate multiple tasks
//! - **Comprehensive**: Evaluates all valid task-dataset-backend combinations
//! - **Extensible**: Easy to add new tasks, datasets, or backends

#[cfg(any(feature = "onnx", feature = "gliner2-fastino"))]
use crate::eval::backend_factory::artifact_receipt_from_paths;
use crate::eval::backend_factory::{BackendConstructionReceipt, BackendFactory};
use crate::eval::backend_name::BackendName;
use crate::eval::loader::{DatasetId, DatasetLoader, LoadedDataset};
#[cfg(feature = "eval-profiling")]
use crate::eval::profiling;
use crate::eval::provenance::{
    artifact_status, canonical_entity_type_label, current_build_provenance,
    legacy_eval_run_provenance,
};
pub use crate::eval::provenance::{
    ArtifactProvenanceStatus, BackendRunProvenance, ClosedLabelDiagnostic, DatasetRunProvenance,
    EvalBuildProvenance, EvalRunProvenance, EvalRuntimeProvenance, EvaluationScheduling,
    ExecutionProvenanceStatus, InferenceSettingProvenance, NerInferenceProvenance,
    NerLabelPolicyReport,
};
use crate::eval::task_mapping::{
    dataset_tasks, get_task_backends, get_task_datasets, Task, TaskMapping,
};
use anno::backends::inference::ZeroShotNER;
use anno::{Entity, EntityType, Model, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};
use std::sync::Mutex;
use std::time::Instant;

/// Lock a std::sync::Mutex, recovering from poisoning.
fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(feature = "eval-parallel")]
fn observe_construction_receipt(
    observed: &std::sync::Arc<Mutex<Vec<BackendConstructionReceipt>>>,
    receipt: &BackendConstructionReceipt,
) {
    lock(observed).push(receipt.clone());
}

/// Deterministically choose one sentence for a bootstrap replicate.
fn bootstrap_index(state: &mut u64, sample_count: usize) -> usize {
    debug_assert!(sample_count > 0);
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as usize) % sample_count
}

/// Return the equal-tailed 95% percentile interval for bootstrap samples.
fn bootstrap_ci_95(samples: &mut [f64]) -> (f64, f64) {
    debug_assert!(!samples.is_empty());
    samples.sort_by(f64::total_cmp);
    let lower = ((samples.len() - 1) as f64 * 0.025).round() as usize;
    let upper = ((samples.len() - 1) as f64 * 0.975).round() as usize;
    (samples[lower], samples[upper])
}

// Type aliases for complex types
type PerExampleScores = Vec<(Vec<anno::Entity>, Vec<anno::Entity>, String)>;

// Constants for evaluation
/// 95% confidence interval z-score (normal distribution)
const DEFAULT_Z_SCORE_95: f64 = 1.96;
/// Number of deterministic sentence-bootstrap replicates for NER confidence intervals.
const CI_BOOTSTRAP_REPLICATES: usize = 1_000;
/// Fixed seed makes scorecards reproducible for an unchanged evaluation sample.
const CI_BOOTSTRAP_SEED: u64 = 0xC195_5EED;
/// Maximum number of examples for robustness testing (performance limit)
///
/// Used in `compute_robustness()` to limit the number of test cases processed.
#[cfg(feature = "eval")]
const ROBUSTNESS_TEST_LIMIT: usize = 50;

/// Stratified metrics across multiple dimensions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StratifiedMetrics {
    /// Metrics by entity type
    pub by_entity_type: HashMap<String, MetricWithCI>,
    /// Metrics by temporal stratum (if available)
    pub by_temporal_stratum: Option<HashMap<String, MetricWithCI>>,
    /// Metrics by surface form type (proper noun, common noun, pronoun)
    pub by_surface_form: Option<HashMap<String, MetricWithCI>>,
    /// Metrics by mention characteristics (capitalized, partial name, etc.)
    pub by_mention_char: Option<HashMap<String, MetricWithCI>>,
}

/// Metrics with confidence intervals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricWithCI {
    /// Mean value
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// 95% confidence interval (lower, upper)
    pub ci_95: (f64, f64),
    /// Sample size
    pub n: usize,
}

/// Confidence intervals for key metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceIntervals {
    /// F1 score CI
    pub f1_ci: (f64, f64),
    /// Precision CI
    pub precision_ci: (f64, f64),
    /// Recall CI
    pub recall_ci: (f64, f64),
}

/// Cached backend enum for thread-local storage (avoids Box<dyn Any> downcast issues).
#[allow(clippy::large_enum_variant)]
#[cfg(feature = "eval-parallel")]
enum CachedBackend {
    #[cfg(feature = "onnx")]
    NuNER(anno::backends::nuner::NuNER),
    #[cfg(feature = "onnx")]
    GLiNEROnnx(anno::backends::gliner_onnx::GLiNEROnnx),
    #[cfg(feature = "onnx")]
    GLiNERMultitaskOnnx(anno::backends::gliner_multitask::GLiNERMultitaskOnnx),
    #[cfg(feature = "gliner2-fastino")]
    GLiNER2Fastino(anno::backends::gliner2_fastino::GLiNER2Fastino),
    #[cfg(feature = "candle")]
    GLiNERCandle(anno::backends::gliner_candle::GLiNERCandle),
    #[cfg(feature = "onnx")]
    GLiNERPoly(anno::backends::gliner_poly::GLiNERPoly),
    UniversalNER(anno::backends::universal_ner::UniversalNER),
}

/// Configuration for task evaluation.
#[derive(Serialize, Deserialize)]
pub struct TaskEvalConfig {
    /// Which tasks to evaluate
    pub tasks: Vec<Task>,
    /// Which datasets to use (if empty, uses all suitable datasets for each task)
    pub datasets: Vec<DatasetId>,
    /// Which backends to test (if empty, uses all compatible backends)
    pub backends: Vec<String>,
    /// Maximum number of examples per dataset (for quick testing)
    pub max_examples: Option<usize>,
    /// Random seed for sampling (for reproducibility and varied testing)
    pub seed: Option<u64>,
    /// Whether to require a local cached dataset and forbid all download fallbacks.
    pub require_cached: bool,
    /// Confidence threshold for relation extraction (default: 0.5)
    pub relation_threshold: f32,
    /// Whether to run robustness testing (perturbations)
    pub robustness: bool,
    /// Whether to compute familiarity scores for zero-shot evaluations
    pub compute_familiarity: bool,
    /// Whether to compute temporal stratification (if dataset supports it)
    pub temporal_stratification: bool,
    /// Whether to compute confidence intervals for metrics
    pub confidence_intervals: bool,
    /// Optional custom coreference resolver (for use with matryoshka-box trained models)
    /// If None, resolver is created from backend_name using create_coref_resolver()
    /// Uses Arc to allow sharing across multiple evaluation calls
    #[serde(skip)]
    pub custom_coref_resolver:
        Option<std::sync::Arc<dyn crate::eval::coref_resolver::CoreferenceResolver>>,

    /// Coreference evaluation mode:
    /// - `false` (default): run NER to get mentions, then resolve coref on those mentions.
    /// - `true`: use GOLD mentions from the coref dataset and evaluate clustering only.
    ///
    /// This is critical for datasets like CorefUD where mentions include pronouns/nominals
    /// and empty nodes (zero anaphora) that typical NER backends do not emit.
    pub coref_use_gold_mentions: bool,
}

impl Default for TaskEvalConfig {
    fn default() -> Self {
        Self {
            tasks: Task::all().to_vec(),
            datasets: vec![],
            backends: vec![],
            max_examples: None,
            seed: Some(42),
            require_cached: false,
            relation_threshold: 0.5,
            robustness: false,
            compute_familiarity: true, // Default to true for zero-shot awareness
            temporal_stratification: false,
            confidence_intervals: true, // Default to true for better reporting
            custom_coref_resolver: None,
            coref_use_gold_mentions: false,
        }
    }
}

impl std::fmt::Debug for TaskEvalConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TaskEvalConfig")
            .field("tasks", &self.tasks)
            .field("datasets", &self.datasets)
            .field("backends", &self.backends)
            .field("max_examples", &self.max_examples)
            .field("seed", &self.seed)
            .field("require_cached", &self.require_cached)
            .field("relation_threshold", &self.relation_threshold)
            .field("robustness", &self.robustness)
            .field("compute_familiarity", &self.compute_familiarity)
            .field("temporal_stratification", &self.temporal_stratification)
            .field("confidence_intervals", &self.confidence_intervals)
            .field("coref_use_gold_mentions", &self.coref_use_gold_mentions)
            .field(
                "custom_coref_resolver",
                &if self.custom_coref_resolver.is_some() {
                    "Some(...)"
                } else {
                    "None"
                },
            )
            .finish()
    }
}

/// Results from evaluating a task-dataset-backend combination.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskEvalResult {
    /// Task being evaluated
    pub task: Task,
    /// Dataset used
    pub dataset: DatasetId,
    /// Backend name
    pub backend: String,
    /// Backend display name (may include composition details, e.g. `stacked(regex+heuristic)`).
    ///
    /// Best-effort: when absent, callers should fall back to `backend`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend_display: Option<String>,
    /// Random seed used for sampling/examples.
    pub seed: u64,
    /// Whether evaluation succeeded
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
    /// Metrics (task-specific, stored as JSON-serializable map)
    pub metrics: HashMap<String, f64>,
    /// Number of examples evaluated
    pub num_examples: usize,
    /// Time taken in milliseconds (if available)
    pub duration_ms: Option<f64>,
    /// Label shift/familiarity metrics (if computed for zero-shot)
    pub label_shift: Option<super::types::LabelShift>,
    /// Robustness scores (if robustness testing was enabled)
    #[cfg(feature = "eval")]
    pub robustness: Option<super::robustness::RobustnessResults>,
    #[cfg(not(feature = "eval"))]
    /// Robustness testing results (only available with `eval` feature).
    #[cfg(not(feature = "eval"))]
    pub robustness: Option<()>, // Stub when `eval` feature not enabled
    /// Stratified metrics by various dimensions
    pub stratified: Option<StratifiedMetrics>,
    /// Confidence intervals for key metrics (if computed)
    pub confidence_intervals: Option<ConfidenceIntervals>,
    /// KB version used (if available from dataset metadata)
    pub kb_version: Option<String>,
    /// Inputs and scoring policy actually used for this result.
    ///
    /// This is deliberately a receipt, not a configuration registry: fields that
    /// the active backend does not expose are represented as `unknown` rather
    /// than inferred from a cache layout or a backend name.
    #[serde(default = "legacy_eval_run_provenance")]
    pub provenance: EvalRunProvenance,
}

#[derive(Debug)]
struct BackendEvalOk {
    metrics: HashMap<String, f64>,
    backend_display: Option<String>,
    configured_labels: Option<Vec<String>>,
    ner_label_policy: Option<NerLabelPolicyReport>,
    construction_receipt: BackendConstructionReceipt,
    scheduling: EvaluationScheduling,
}

#[derive(Debug)]
struct NerEvaluationReceipt {
    metrics: HashMap<String, f64>,
    configured_labels: Option<Vec<String>>,
    label_policy: NerLabelPolicyReport,
    construction_receipt: BackendConstructionReceipt,
    scheduling: EvaluationScheduling,
}

struct ProvenanceInput<'a> {
    requested_backend_name: &'a str,
    effective_backend_name: &'a str,
    backend_display: Option<String>,
    dataset: &'a LoadedDataset,
    config: &'a TaskEvalConfig,
    configured_labels: Option<Vec<String>>,
    ner_label_policy: Option<NerLabelPolicyReport>,
    construction_receipt: BackendConstructionReceipt,
    scheduling: EvaluationScheduling,
}

#[cfg(not(feature = "eval-parallel"))]
struct CachedBackendAny {
    backend: Box<dyn std::any::Any>,
    construction_receipt: BackendConstructionReceipt,
}

impl TaskEvalResult {
    /// Check if this is a "skipped" result (feature not available or incompatible) vs actual failure
    pub fn is_skipped(&self) -> bool {
        if self.success {
            return false;
        }
        if let Some(ref err) = self.error {
            err.starts_with("incompatible:")
                || err.contains("Feature not available")
                || err.contains("requires '")
                || err.contains("Incompatible entity types")
        } else {
            false
        }
    }

    /// Get primary F1 metric for ranking
    pub fn primary_f1(&self) -> Option<f64> {
        self.metrics
            .get("f1")
            .or_else(|| self.metrics.get("conll_f1"))
            .or_else(|| self.metrics.get("strict_f1"))
            .copied()
    }
}

/// Comprehensive evaluation results across all combinations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComprehensiveEvalResults {
    /// Individual evaluation results
    pub results: Vec<TaskEvalResult>,
    /// Summary statistics
    pub summary: EvalSummary,
}

/// Summary statistics for comprehensive evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalSummary {
    /// Total combinations evaluated
    pub total_combinations: usize,
    /// Successful evaluations
    pub successful: usize,
    /// Failed evaluations (actual errors, not skipped)
    pub failed: usize,
    /// Skipped evaluations (feature not available, etc.)
    pub skipped: usize,
    /// Tasks evaluated
    pub tasks: Vec<Task>,
    /// Datasets used
    pub datasets: Vec<DatasetId>,
    /// Backends tested
    pub backends: Vec<String>,
}

/// Evaluator for task-dataset-backend combinations.
pub struct TaskEvaluator {
    loader: DatasetLoader,
    #[allow(dead_code)] // Reserved for future use
    mapping: TaskMapping,
    // Temporary storage for per-example scores (used during evaluation)
    // Cloned when needed to avoid borrow checker issues
    per_example_scores_cache: Mutex<Option<PerExampleScores>>,
    /// Evaluation history tracker (optional, for persistent result storage)
    history: Option<super::history::EvalHistory>,
}

impl TaskEvaluator {
    /// Access the evaluation history (if initialized).
    pub fn history(&self) -> Option<&super::history::EvalHistory> {
        self.history.as_ref()
    }

    /// True if this task has a real end-to-end evaluation path implemented in `TaskEvaluator`.
    ///
    /// Tasks may be "catalogued" (present in the dataset registry and task mapping) without
    /// having an evaluation pipeline yet; those tasks should not be scheduled by the matrix.
    pub fn is_task_supported(task: Task) -> bool {
        matches!(
            task,
            Task::NER
                | Task::DiscontinuousNER
                | Task::RelationExtraction
                | Task::IntraDocCoref
                | Task::InterDocCoref
                | Task::AbstractAnaphora
                | Task::TextClassification
                | Task::EventExtraction
                | Task::SpeechActClassification
                | Task::Temporal
                | Task::DiscourseRelations
                | Task::DiscourseSegmentation
        )
    }

    /// Create a new task evaluator.
    pub fn new() -> Result<Self> {
        // Resolution order (mirrors eval_history_jsonl_path() in matrix_muxer_ci):
        //   1. ANNO_EVAL_HISTORY — explicit override
        //   2. ANNO_CACHE_DIR   — CI-consistent cache root (avoids split-brain with the harness)
        //   3. platform cache   — dirs::cache_dir()/anno/eval-results.jsonl
        let history_path: std::path::PathBuf = std::env::var("ANNO_EVAL_HISTORY")
            .map(std::path::PathBuf::from)
            .or_else(|_| {
                std::env::var("ANNO_CACHE_DIR")
                    .map(|d| std::path::PathBuf::from(d).join("eval-results.jsonl"))
            })
            .unwrap_or_else(|_| {
                dirs::cache_dir()
                    .map(|d| d.join("anno").join("eval-results.jsonl"))
                    .unwrap_or_else(|| std::path::PathBuf::from("eval-results.jsonl"))
            });
        let history = super::history::EvalHistory::new(&history_path)
            .map_err(|e| {
                log::warn!("Failed to initialize eval history: {}", e);
                e
            })
            .ok();

        Ok(Self {
            loader: DatasetLoader::new()?,
            mapping: TaskMapping::build(),
            per_example_scores_cache: Mutex::new(None),
            history,
        })
    }

    /// Create a new task evaluator with a custom cache directory.
    ///
    /// Useful for testing with isolated caches.
    pub fn with_cache_dir(cache_dir: impl AsRef<std::path::Path>) -> Result<Self> {
        let cache_path = cache_dir.as_ref();

        // Use same directory for history if cache_dir is provided
        // If cache_dir is a file, use its parent; if it's a dir, use it directly
        let history_path = if cache_path.is_file() {
            cache_path
                .parent()
                .map(|p| p.join("eval-results.jsonl"))
                .unwrap_or_else(|| cache_path.with_file_name("eval-results.jsonl"))
        } else {
            cache_path.join("eval-results.jsonl")
        };
        let history = super::history::EvalHistory::new(&history_path)
            .map_err(|e| {
                log::warn!("Failed to initialize eval history: {}", e);
                e
            })
            .ok();

        Ok(Self {
            loader: DatasetLoader::with_cache_dir(cache_path)?,
            mapping: TaskMapping::build(),
            per_example_scores_cache: Mutex::new(None),
            history,
        })
    }

    fn sample_dataset_for_task(
        task: Task,
        dataset_data: &LoadedDataset,
        config: &TaskEvalConfig,
    ) -> (LoadedDataset, usize) {
        let total = dataset_data.sentences.len();
        let (sampled_data, sentences_to_use) = if let Some(max) = config.max_examples {
            if max >= total {
                (dataset_data.clone(), total)
            } else {
                // Task-aware, deterministic sampling:
                //
                // For NER-like tasks, preserve the corpus's positive/negative mix. Sampling only
                // positive sentences hides false positives and overstates precision.
                let seed = config.seed.unwrap_or(42);
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};
                let select = |base: Vec<usize>, count: usize| {
                    let mut indices: Vec<(usize, u64)> = base
                        .into_iter()
                        .map(|i| {
                            let mut hasher = DefaultHasher::new();
                            seed.hash(&mut hasher);
                            i.hash(&mut hasher);
                            (i, hasher.finish())
                        })
                        .collect();
                    indices.sort_by_key(|(_, hash)| *hash);
                    indices
                        .into_iter()
                        .take(count)
                        .map(|(i, _)| i)
                        .collect::<Vec<_>>()
                };

                let mut selected_indices = match task {
                    Task::NER | Task::DiscontinuousNER | Task::EventExtraction => {
                        let (positive, negative): (Vec<_>, Vec<_>) = dataset_data
                            .sentences
                            .iter()
                            .enumerate()
                            .partition(|(_, sentence)| !sentence.entities().is_empty());
                        let positive: Vec<usize> = positive.into_iter().map(|(i, _)| i).collect();
                        let negative: Vec<usize> = negative.into_iter().map(|(i, _)| i).collect();

                        if positive.is_empty() || negative.is_empty() {
                            select((0..total).collect(), max)
                        } else {
                            // Round to the corpus negative rate. With room for both strata, keep
                            // one representative from each so neither error mode disappears.
                            let negative_count = ((max * negative.len()) + (total / 2)) / total;
                            let negative_count = if max > 1 {
                                negative_count.clamp(1, max - 1)
                            } else {
                                negative_count
                            };
                            let mut selected = select(positive, max - negative_count);
                            selected.extend(select(negative, negative_count));
                            selected
                        }
                    }
                    _ => select((0..total).collect(), max),
                };
                selected_indices.sort_unstable();
                let sampled_sentences: Vec<_> = selected_indices
                    .iter()
                    .filter_map(|&i| dataset_data.sentences.get(i).cloned())
                    .collect();
                let sampled_dataset = LoadedDataset {
                    id: dataset_data.id,
                    sentences: sampled_sentences,
                    loaded_at: dataset_data.loaded_at.clone(),
                    source_url: dataset_data.source_url.clone(),
                    data_source: dataset_data.data_source,
                    temporal_metadata: dataset_data.temporal_metadata.clone(),
                    metadata: dataset_data.metadata.clone(),
                };
                let n = sampled_dataset.sentences.len();
                (sampled_dataset, n)
            }
        } else {
            (dataset_data.clone(), total)
        };

        (sampled_data, sentences_to_use)
    }

    fn evaluate_backend_on_loaded(
        &self,
        task: Task,
        dataset: DatasetId,
        backend_name: &str,
        sampled_data: &LoadedDataset,
        sentences_to_use: usize,
        config: &TaskEvalConfig,
    ) -> TaskEvalResult {
        let seed = config.seed.unwrap_or(42);
        let requested_backend_name = backend_name;
        // Use the same canonical spelling accepted by the factory and typed
        // backend API before compatibility checks and zero-shot dispatch.
        let backend_name = BackendName::try_parse(backend_name)
            .map(|name| name.as_str())
            .unwrap_or(backend_name);
        // Try to evaluate backend (handles backend creation internally)
        let start = Instant::now();
        match self.try_evaluate_backend(task, dataset, backend_name, sampled_data, config) {
            Ok(ok) => {
                let metrics = ok.metrics;
                let backend_display = ok.backend_display;
                let duration = start.elapsed().as_secs_f64() * 1000.0;
                let num_examples = if task.is_coref_family() {
                    metrics
                        .get("num_docs")
                        .copied()
                        .map(|n| n.max(0.0) as usize)
                        .unwrap_or(sentences_to_use)
                } else {
                    sentences_to_use
                };

                // Compute familiarity for zero-shot backends
                let label_shift = if config.compute_familiarity {
                    self.compute_familiarity_if_zero_shot(backend_name, sampled_data)
                } else {
                    None
                };

                // Run robustness testing if enabled
                #[cfg(feature = "eval")]
                let robustness_result: Option<
                    super::robustness::RobustnessResults,
                > = if config.robustness && matches!(task, Task::NER | Task::DiscontinuousNER) {
                    self.compute_robustness(backend_name, sampled_data, config)
                } else {
                    None
                };

                // Compute stratified metrics (use per-example scores if available)
                // Extract per-example scores once and reuse for both stratified metrics and confidence intervals
                let per_example_opt =
                    { lock::<Option<PerExampleScores>>(&self.per_example_scores_cache).clone() };

                let stratified = if matches!(task, Task::NER | Task::DiscontinuousNER) {
                    if let Some(per_example) = per_example_opt.as_ref() {
                        self.compute_stratified_metrics_from_scores(
                            sampled_data,
                            &metrics,
                            Some(per_example),
                        )
                    } else {
                        None
                    }
                } else {
                    None
                };

                // Compute confidence intervals if requested (use per-example scores if available)
                let confidence_intervals = if config.confidence_intervals {
                    per_example_opt.as_ref().and_then(|per_example| {
                        self.compute_confidence_intervals_from_scores(per_example)
                    })
                } else {
                    None
                };

                // Clear cache after use
                let mut cache = lock(&self.per_example_scores_cache);
                *cache = None;

                // Extract KB version if available
                let kb_version = Self::extract_kb_version(sampled_data);

                TaskEvalResult {
                    task,
                    dataset,
                    backend: backend_name.to_string(),
                    backend_display: backend_display.clone(),
                    seed,
                    success: true,
                    error: None,
                    metrics,
                    num_examples,
                    duration_ms: Some(duration),
                    label_shift,
                    #[cfg(feature = "eval")]
                    robustness: robustness_result,
                    #[cfg(not(feature = "eval"))]
                    robustness: None,
                    stratified,
                    confidence_intervals,
                    kb_version,
                    provenance: Self::run_provenance(ProvenanceInput {
                        requested_backend_name,
                        effective_backend_name: backend_name,
                        backend_display,
                        dataset: sampled_data,
                        config,
                        configured_labels: ok.configured_labels,
                        ner_label_policy: ok.ner_label_policy,
                        construction_receipt: ok.construction_receipt,
                        scheduling: ok.scheduling,
                    }),
                }
            }
            Err(e) => Self::failed_task_eval_result(
                task,
                dataset,
                seed,
                sentences_to_use,
                start.elapsed().as_secs_f64() * 1000.0,
                ProvenanceInput {
                    requested_backend_name,
                    effective_backend_name: backend_name,
                    backend_display: None,
                    dataset: sampled_data,
                    config,
                    configured_labels: None,
                    ner_label_policy: None,
                    construction_receipt: BackendConstructionReceipt::default(),
                    scheduling: EvaluationScheduling::Unknown {
                        reason: "evaluation failed before its scheduling strategy was recorded"
                            .to_string(),
                    },
                },
                e,
            ),
        }
    }

    fn failed_task_eval_result(
        task: Task,
        dataset: DatasetId,
        seed: u64,
        sentences_to_use: usize,
        duration_ms: f64,
        provenance: ProvenanceInput<'_>,
        error: impl std::fmt::Display,
    ) -> TaskEvalResult {
        TaskEvalResult {
            task,
            dataset,
            backend: provenance.effective_backend_name.to_string(),
            backend_display: None,
            seed,
            success: false,
            error: Some(error.to_string()),
            metrics: HashMap::new(),
            num_examples: sentences_to_use,
            duration_ms: Some(duration_ms),
            label_shift: None,
            #[cfg(feature = "eval")]
            robustness: None,
            #[cfg(not(feature = "eval"))]
            robustness: None,
            stratified: None,
            confidence_intervals: None,
            kb_version: None,
            provenance: Self::run_provenance(provenance),
        }
    }

    fn run_provenance(input: ProvenanceInput<'_>) -> EvalRunProvenance {
        EvalRunProvenance {
            schema_version: 1,
            build: current_build_provenance(),
            dataset: DatasetRunProvenance {
                source_url: input.dataset.source_url.clone(),
                data_source: input.dataset.data_source.to_string(),
                split: input.dataset.metadata.split.clone(),
                version: input.dataset.metadata.version.clone(),
                language: input.dataset.metadata.language.clone(),
                domain: input.dataset.metadata.domain.clone(),
                sentence_count: input.dataset.sentences.len(),
                entity_count: input.dataset.entity_count(),
            },
            backend: BackendRunProvenance {
                requested: input.requested_backend_name.to_string(),
                effective: input.effective_backend_name.to_string(),
                display_name: input.backend_display,
                configured_labels: input.configured_labels,
                model_artifacts: artifact_status(input.construction_receipt),
                execution: ExecutionProvenanceStatus::default(),
            },
            runtime: EvalRuntimeProvenance {
                seed: input.config.seed.unwrap_or(42),
                cached_only: input.config.require_cached,
                max_examples: input.config.max_examples,
                confidence_intervals: input.config.confidence_intervals,
                robustness: input.config.robustness,
                relation_threshold: input.config.relation_threshold,
                ner_inference: Self::ner_inference_provenance(input.effective_backend_name),
                scheduling: input.scheduling,
            },
            ner_label_policy: input.ner_label_policy,
        }
    }

    fn ner_inference_provenance(backend_name: &str) -> NerInferenceProvenance {
        let uses_evaluator_threshold = matches!(
            backend_name,
            "nuner"
                | "gliner_onnx"
                | "gliner"
                | "gliner2_fastino"
                | "gliner_candle"
                | "gliner_poly"
                | "universal_ner"
        );
        NerInferenceProvenance {
            threshold: if uses_evaluator_threshold {
                InferenceSettingProvenance::Known {
                    value: serde_json::json!(0.5),
                }
            } else {
                InferenceSettingProvenance::Unknown {
                    reason: "evaluator did not supply a NER threshold for this backend".to_string(),
                }
            },
            ..Default::default()
        }
    }

    /// Run comprehensive evaluation across all valid combinations.
    pub fn evaluate_all(&self, config: TaskEvalConfig) -> Result<ComprehensiveEvalResults> {
        let seed = config.seed.unwrap_or(42);
        let mut results = Vec::new();
        let mut tasks_evaluated = Vec::new();
        let mut datasets_used = Vec::new();
        let mut backends_tested: Vec<String> = Vec::new();
        let mut dataset_cache: HashMap<DatasetId, LoadedDataset> = HashMap::new();
        let mut sampled_cache: HashMap<(Task, DatasetId), (LoadedDataset, usize)> = HashMap::new();

        // Determine which tasks to evaluate
        let tasks = if config.tasks.is_empty() {
            Task::all().to_vec()
        } else {
            config.tasks.clone()
        };

        for task in &tasks {
            tasks_evaluated.push(*task);

            // Get suitable datasets for this task
            let datasets = if config.datasets.is_empty() {
                get_task_datasets(*task)
            } else {
                // Filter to datasets that support this task
                config
                    .datasets
                    .iter()
                    .filter(|d| dataset_tasks(**d).contains(task))
                    .copied()
                    .collect()
            };

            for dataset in &datasets {
                if !datasets_used.contains(dataset) {
                    datasets_used.push(*dataset);
                }
                // Get compatible backends for this task
                let backends: Vec<String> = if config.backends.is_empty() {
                    get_task_backends(*task)
                        .iter()
                        .map(|s| s.to_string())
                        .collect()
                } else {
                    // If the caller specifies explicit backends, still filter them per-task.
                    // Otherwise we waste time evaluating impossible combinations and inflate
                    // "expected failures" (which reduces signal from matrix sampling).
                    let allowed: std::collections::HashSet<&'static str> =
                        get_task_backends(*task).into_iter().collect();
                    config
                        .backends
                        .iter()
                        .filter(|b| allowed.contains(b.as_str()))
                        .cloned()
                        .collect()
                };

                // Further filter by dataset-level compatibility (entity types, etc.).
                // Track incompatible backends for better error reporting.
                let (compatible_backends, incompatible_backends): (Vec<String>, Vec<String>) =
                    backends
                        .into_iter()
                        .partition(|b| Self::is_backend_compatible(b, *dataset));

                // Add incompatible backends as results with clear error message
                for backend_name in &incompatible_backends {
                    if !backends_tested.contains(backend_name) {
                        backends_tested.push(backend_name.clone());
                    }
                    let dataset_entity_types = dataset.entity_types();
                    results.push(TaskEvalResult {
                        task: *task,
                        dataset: *dataset,
                        backend: backend_name.to_string(),
                        backend_display: None,
                        seed,
                        success: false,
                        error: Some(format!(
                            "incompatible: backend '{}' doesn't support dataset entity types: {:?}",
                            backend_name, dataset_entity_types
                        )),
                        metrics: HashMap::new(),
                        num_examples: 0,
                        duration_ms: None,
                        label_shift: None,
                        #[cfg(feature = "eval")]
                        robustness: None,
                        #[cfg(not(feature = "eval"))]
                        robustness: None,
                        stratified: None,
                        confidence_intervals: None,
                        kb_version: None,
                        provenance: EvalRunProvenance::default(),
                    });
                }

                if compatible_backends.is_empty() {
                    continue;
                }

                let backends = compatible_backends;

                // Load dataset once per dataset id and reuse across backends.
                if !dataset_cache.contains_key(dataset) {
                    let loaded: Result<LoadedDataset> = {
                        #[cfg(feature = "eval")]
                        {
                            let loadable = crate::eval::LoadableDatasetId::try_from(*dataset)
                                .map_err(|e| crate::Error::InvalidInput(format!("{}", e)))?;
                            if config.require_cached {
                                self.loader.load(loadable)
                            } else {
                                self.loader.load_or_download(loadable)
                            }
                        }
                        #[cfg(not(feature = "eval"))]
                        {
                            let loadable = crate::eval::LoadableDatasetId::try_from(*dataset)
                                .map_err(|e| crate::Error::InvalidInput(format!("{}", e)))?;
                            self.loader.load(loadable)
                        }
                    };
                    match loaded {
                        Ok(d) => {
                            dataset_cache.insert(*dataset, d);
                        }
                        Err(e) => {
                            for backend_name in &backends {
                                if !backends_tested.contains(backend_name) {
                                    backends_tested.push(backend_name.clone());
                                }
                                results.push(TaskEvalResult {
                                    task: *task,
                                    dataset: *dataset,
                                    backend: backend_name.to_string(),
                                    backend_display: None,
                                    seed,
                                    success: false,
                                    error: Some(format!("Failed to load dataset: {}", e)),
                                    metrics: HashMap::new(),
                                    num_examples: 0,
                                    duration_ms: None,
                                    label_shift: None,
                                    #[cfg(feature = "eval")]
                                    robustness: None,
                                    #[cfg(not(feature = "eval"))]
                                    robustness: None,
                                    stratified: None,
                                    confidence_intervals: None,
                                    kb_version: None,
                                    provenance: EvalRunProvenance::default(),
                                });
                            }
                            continue;
                        }
                    }
                }

                let dataset_data = dataset_cache.get(dataset).expect("cache populated");

                if dataset_data.sentences.is_empty() {
                    for backend_name in &backends {
                        if !backends_tested.contains(backend_name) {
                            backends_tested.push(backend_name.clone());
                        }
                        results.push(TaskEvalResult {
                            task: *task,
                            dataset: *dataset,
                            backend: backend_name.to_string(),
                            backend_display: None,
                            seed,
                            success: false,
                            error: Some(format!(
                                "Dataset '{}' is empty (no sentences found)",
                                dataset.name()
                            )),
                            metrics: HashMap::new(),
                            num_examples: 0,
                            duration_ms: None,
                            label_shift: None,
                            #[cfg(feature = "eval")]
                            robustness: None,
                            #[cfg(not(feature = "eval"))]
                            robustness: None,
                            stratified: None,
                            confidence_intervals: None,
                            kb_version: None,
                            provenance: EvalRunProvenance::default(),
                        });
                    }
                    continue;
                }

                sampled_cache.entry((*task, *dataset)).or_insert_with(|| {
                    let (sampled, n) = Self::sample_dataset_for_task(*task, dataset_data, &config);
                    (sampled, n)
                });
                let (sampled_data, sentences_to_use) = sampled_cache
                    .get(&(*task, *dataset))
                    .expect("sampled cache populated");

                for backend_name in &backends {
                    if !backends_tested.contains(backend_name) {
                        backends_tested.push(backend_name.clone());
                    }
                    results.push(self.evaluate_backend_on_loaded(
                        *task,
                        *dataset,
                        backend_name,
                        sampled_data,
                        *sentences_to_use,
                        &config,
                    ));
                }
            }
        }

        let skipped = results.iter().filter(|r| r.is_skipped()).count();
        let failed = results
            .iter()
            .filter(|r| !r.success && !r.is_skipped())
            .count();
        let summary = EvalSummary {
            total_combinations: results.len(),
            successful: results.iter().filter(|r| r.success).count(),
            failed,
            skipped,
            tasks: tasks_evaluated,
            datasets: datasets_used,
            backends: backends_tested,
        };

        #[cfg(feature = "eval-profiling")]
        profiling::print_summary();

        // Store results in history if available
        if let Some(ref history) = self.history {
            for result in &results {
                let entry = super::history::EvalHistoryEntry::from(result);
                if let Err(e) = history.append_entry(&entry) {
                    log::warn!("Failed to store result in history: {}", e);
                }
            }
        }

        Ok(ComprehensiveEvalResults { results, summary })
    }

    /// Check if backend is compatible with dataset entity types.
    ///
    /// - `stacked`: Compatible with most types (combines pattern+heuristic)
    /// - ML backends: Always compatible (zero-shot or trained)
    /// - `pattern`: Only structured entities (not named entities)
    /// - `heuristic`: Only Person, Organization, Location
    pub(crate) fn is_backend_compatible(backend_name: &str, dataset: DatasetId) -> bool {
        let entity_types = dataset.entity_types();
        let normalized_types: Vec<String> = entity_types.iter().map(|t| t.to_lowercase()).collect();

        match backend_name {
            // Stacked combines pattern+heuristic, so it's compatible with most types
            "stacked" => true,
            // Classical backends in this repo are trained/implemented for CoNLL-style tags.
            "crf" | "hmm" => {
                let supported = [
                    "person",
                    "per",
                    "organization",
                    "org",
                    "location",
                    "loc",
                    "misc",
                ];
                normalized_types
                    .iter()
                    .all(|t| supported.iter().any(|s| t == s || t.starts_with(s)))
            }
            // ML backends are zero-shot or trained, so compatible
            "bert_onnx" | "candle_ner" | "nuner" | "nuner_4k" | "b2ner" | "gliner_onnx"
            | "gliner_candle" | "gliner_multitask" | "gliner_pii" | "gliner_relex" | "w2ner"
            | "gliner_poly" | "deberta_v3" | "albert" | "universal_ner" => true,
            // Pattern only does structured entities (not named entities)
            "pattern" => {
                // RegexNER only extracts: Date, Time, Money, Percent, Email, URL, Phone
                // Not compatible with named entity datasets
                false
            }
            // Heuristic only does Person, Organization, Location
            "heuristic" => {
                let supported = [
                    "person",
                    "per",
                    "organization",
                    "org",
                    "location",
                    "loc",
                    "misc",
                ];
                normalized_types
                    .iter()
                    .all(|t| supported.iter().any(|s| t == s || t.starts_with(s)))
            }
            _ => true, // Unknown backends - assume compatible
        }
    }

    /// Evaluate a backend on a task with actual inference and metrics.
    ///
    /// This implementation:
    /// 1. Creates backend instance via `BackendFactory`
    /// 2. Runs inference on dataset examples
    /// 3. Computes task-specific metrics (P/R/F1 for NER, MUC/B³/CEAF for coref, etc.)
    /// 4. Returns metrics as a map
    fn try_evaluate_backend(
        &self,
        task: Task,
        dataset: DatasetId,
        backend_name: &str,
        dataset_data: &LoadedDataset,
        config: &TaskEvalConfig,
    ) -> Result<BackendEvalOk> {
        if config.temporal_stratification {
            return Err(crate::Error::FeatureNotAvailable(
                "Temporal stratification requires per-example timestamps, which this evaluator does not load"
                    .to_string(),
            ));
        }

        // Validate task-dataset compatibility
        let dataset_tasks = dataset_tasks(dataset);
        if !dataset_tasks.contains(&task) {
            return Err(crate::Error::InvalidInput(format!(
                "Dataset {:?} does not support task {:?}",
                dataset, task
            )));
        }

        // Validate task-backend compatibility
        let backend_tasks: Vec<String> = get_task_backends(task)
            .iter()
            .map(|s| s.to_string())
            .collect();
        if !backend_tasks.contains(&backend_name.to_string()) {
            return Err(crate::Error::InvalidInput(format!(
                "Backend '{}' does not support task {:?}",
                backend_name, task
            )));
        }

        // Run task-specific evaluation
        // Note: Coref tasks don't use BackendFactory (they use create_coref_resolver)
        match task {
            Task::NER
            | Task::DiscontinuousNER
            | Task::EventExtraction
            | Task::Temporal
            | Task::DiscourseSegmentation => {
                let defer_label_conditioned_load = !dataset.entity_types().is_empty()
                    && matches!(
                        backend_name,
                        "nuner"
                            | "gliner_onnx"
                            | "gliner"
                            | "gliner_multitask"
                            | "gliner2_fastino"
                            | "gliner_poly"
                            | "universal_ner"
                    );
                // `evaluate_ner_task` caches one label-aware instance per worker.
                // Constructing one here first would load GLiNER/Fastino twice and
                // make the recorded receipt describe an unused instance.
                let (backend, construction_receipt) = if defer_label_conditioned_load {
                    (None, BackendConstructionReceipt::default())
                } else {
                    let (backend, receipt) = BackendFactory::create_with_provenance(backend_name)?;
                    (Some(backend), receipt)
                };
                let backend_display = backend
                    .as_ref()
                    .map(|backend| {
                        let name = backend.name().trim();
                        if name.is_empty() || name.eq_ignore_ascii_case("unknown") {
                            backend_name.to_string()
                        } else {
                            name.to_string()
                        }
                    })
                    .or_else(|| Some(backend_name.to_string()));
                if let Some(backend) = backend.as_ref() {
                    if !backend.is_available() {
                        return Err(crate::Error::FeatureNotAvailable(format!(
                            "Backend '{}' is not available (feature not enabled or model not loaded)",
                            backend_name
                        )));
                    }
                }
                let ner = self.evaluate_ner_task(
                    backend_name,
                    backend.as_deref(),
                    dataset,
                    dataset_data,
                    config,
                )?;
                let NerEvaluationReceipt {
                    metrics,
                    configured_labels,
                    label_policy,
                    construction_receipt: zero_shot_receipt,
                    scheduling,
                } = ner;
                let construction_receipt = if zero_shot_receipt.model_artifacts.is_some() {
                    zero_shot_receipt
                } else {
                    construction_receipt
                };
                Ok(BackendEvalOk {
                    metrics,
                    backend_display,
                    configured_labels,
                    ner_label_policy: Some(label_policy),
                    construction_receipt,
                    scheduling,
                })
            }
            Task::IntraDocCoref | Task::InterDocCoref | Task::AbstractAnaphora => {
                // Coref tasks use create_coref_resolver, not BackendFactory
                // Skip BackendFactory::create() to avoid "Unknown backend" error
                let metrics = self.evaluate_coref_task(task, backend_name, dataset_data, config)?;
                Ok(BackendEvalOk {
                    metrics,
                    backend_display: None,
                    configured_labels: None,
                    ner_label_policy: None,
                    construction_receipt: BackendConstructionReceipt::default(),
                    scheduling: EvaluationScheduling::Sequential,
                })
            }
            Task::RelationExtraction => {
                // Relation extraction requires a Model backend
                let (backend, construction_receipt) =
                    BackendFactory::create_with_provenance(backend_name)?;
                let backend_display = {
                    let n = backend.name().trim();
                    if n.is_empty() || n.eq_ignore_ascii_case("unknown") {
                        Some(backend_name.to_string())
                    } else {
                        Some(n.to_string())
                    }
                };
                // Check availability before evaluation
                if !backend.is_available() {
                    return Err(crate::Error::FeatureNotAvailable(format!(
                        "Backend '{}' is not available (feature not enabled or model not loaded)",
                        backend_name
                    )));
                }
                let metrics =
                    self.evaluate_relation_task(backend_name, &*backend, dataset_data, config)?;
                Ok(BackendEvalOk {
                    metrics,
                    backend_display,
                    configured_labels: None,
                    ner_label_policy: None,
                    construction_receipt,
                    scheduling: EvaluationScheduling::Sequential,
                })
            }
            Task::TextClassification | Task::SpeechActClassification | Task::DiscourseRelations => {
                let metrics = self.evaluate_text_classification_task(
                    backend_name,
                    dataset,
                    dataset_data,
                    config,
                )?;
                Ok(BackendEvalOk {
                    metrics,
                    backend_display: None,
                    configured_labels: None,
                    ner_label_policy: None,
                    construction_receipt: BackendConstructionReceipt::default(),
                    scheduling: EvaluationScheduling::Sequential,
                })
            }
            _ => Err(crate::Error::InvalidInput(format!(
                "Task {} is catalogued but not yet supported by TaskEvaluator",
                task.code()
            ))),
        }
    }

    /// Evaluate NER task with actual inference.
    fn evaluate_ner_task(
        &self,
        backend_name: &str,
        backend: Option<&dyn Model>,
        dataset: DatasetId,
        dataset_data: &LoadedDataset,
        _config: &TaskEvalConfig,
    ) -> Result<NerEvaluationReceipt> {
        use crate::eval::metrics::compute_document_extraction_quality_metrics;
        use crate::eval::ner_metrics::{evaluate_entities, NerEvalResults};

        #[cfg(feature = "eval-profiling")]
        profiling::start("evaluate_ner_task");

        // Pre-allocate vectors with estimated capacity to reduce reallocations
        let estimated_entities = dataset_data.sentences.len() * 3; // Rough estimate: ~3 entities per sentence
        let mut all_gold = Vec::with_capacity(estimated_entities);
        let mut all_predicted = Vec::with_capacity(estimated_entities);
        let mut eval_results = NerEvalResults::new();
        let mut total_chars = 0;
        let start_time = Instant::now();

        // Track per-example scores for stratified metrics and confidence intervals
        // Always track for NER tasks (needed for per-type metrics)
        // Note: This function is only called for NER/DiscontinuousNER tasks
        let track_per_example = true;
        let mut per_example_scores: Vec<(Vec<Entity>, Vec<Entity>, String)> = Vec::new();

        // Extract dataset entity types and map to model-compatible labels
        let dataset_labels = dataset.entity_types();
        let mapped_labels = Self::map_dataset_labels_to_model(dataset_labels, backend_name);
        let dataset_label_space: BTreeSet<String> = dataset_labels
            .iter()
            .map(|label| canonical_entity_type_label(&EntityType::from_label(label)))
            .collect();

        // Debug: log mapped labels for zero-shot models
        if std::env::var("ANNO_DEBUG_LABELS").is_ok() {
            eprintln!(
                "DEBUG [{}]: dataset_labels={:?} mapped_labels={:?}",
                backend_name, dataset_labels, mapped_labels
            );
        }

        // Check if this is a zero-shot backend that needs custom labels
        let is_zero_shot = matches!(
            backend_name.to_lowercase().as_str(),
            "nuner"
                | "gliner_onnx"
                | "gliner_candle"
                | "gliner_multitask"
                | "gliner2_fastino"
                | "gliner_poly"
                | "universal_ner"
        );

        // Process sentences (parallel if rayon is available, sequential otherwise)
        let total_sentences = dataset_data.sentences.len();
        let mut zero_shot_construction_receipt = BackendConstructionReceipt::default();
        let scheduling;

        #[cfg(feature = "eval-parallel")]
        {
            use rayon::prelude::*;
            use std::cell::RefCell;
            use std::collections::HashSet;
            use std::sync::atomic::{AtomicUsize, Ordering};
            use std::sync::Arc;

            // For parallel processing, use thread-local storage to cache backends per thread
            // This avoids the need to share state across threads while still caching per thread
            // Using CachedBackend enum instead of Box<dyn Any> to avoid downcast issues
            thread_local! {
                // Store (normalized_name, backend_name_used_for_creation, backend, receipt).
                // Hashes belong to construction, so warm cache hits clone this receipt.
                // Using enum instead of Box<dyn Any> for type safety
                static THREAD_CACHED_BACKEND: RefCell<Option<(String, String, CachedBackend, BackendConstructionReceipt)>> = const { RefCell::new(None) };
            }

            // Normalize backend name to lowercase for consistent caching
            let backend_name_normalized = backend_name.to_lowercase();
            let backend_name_arc = Arc::new(backend_name_normalized);
            let mapped_labels_arc = Arc::new(mapped_labels.clone());
            let is_zero_shot_flag = is_zero_shot;
            let observed_construction_receipts = Arc::new(Mutex::new(Vec::new()));
            let worker_threads = Arc::new(Mutex::new(HashSet::new()));
            let configured_rayon_threads = rayon::current_num_threads();

            let progress_counter = AtomicUsize::new(0);
            let last_progress_percent = Arc::new(Mutex::new(0));
            let start_time_arc = Arc::new(Mutex::new(start_time));

            let all_results: Vec<_> = dataset_data.sentences
                .par_iter()
                .enumerate()
                .map(|(idx, sentence)| {
                    lock(&worker_threads).insert(std::thread::current().id());
                    let text = sentence.text();
                    let chars_count = text.chars().count();

                    // Extract gold entities (clone necessary for parallel processing)
                    let gold_entities: Vec<Entity> = sentence.entities().iter().map(|g| {
                        let mut entity = Entity::new(
                            g.text.clone(), // Clone necessary: sentence.entities() returns references
                            g.entity_type.clone(), // Clone necessary: sentence.entities() returns references
                            g.start,
                            g.end,
                            1.0,
                        );
                        entity.provenance = Some(crate::Provenance::ml("gold", 1.0));
                        entity
                    }).collect();

                    // Run inference - use thread-local cached backend for zero-shot models
                    let entities_result = if is_zero_shot_flag && !mapped_labels_arc.is_empty() {
                        THREAD_CACHED_BACKEND.with(|cache| {
                            let mut cached = cache.borrow_mut();
                            // Check if we have a cached backend for this backend_name (case-insensitive)
                            let backend_name_lower = backend_name_arc.as_str().to_lowercase();
                            if let Some((ref cached_name, ref _creation_name, ref backend, ref receipt)) = *cached {
                                if cached_name.to_lowercase() == backend_name_lower {
                                    observe_construction_receipt(
                                        &observed_construction_receipts,
                                        receipt,
                                    );
                                    // Use cached backend - no downcast needed, enum is type-safe
                                    return Self::extract_with_cached_backend(
                                        backend,
                                        &text,
                                        &mapped_labels_arc,
                                    );
                                }
                            }
                            // Create and cache new backend for this thread
                            let creation_name = backend_name_arc.as_str().to_string();
                            match Self::create_zero_shot_backend(backend_name_arc.as_str()) {
                                Ok(new_backend) => {
                                    let receipt = Self::cached_backend_construction_receipt(&new_backend);
                                    observe_construction_receipt(
                                        &observed_construction_receipts,
                                        &receipt,
                                    );
                                    let result = Self::extract_with_cached_backend(
                                        &new_backend,
                                        &text,
                                        &mapped_labels_arc,
                                    );
                                    // Store normalized (lowercase) name for matching, and creation name for reference
                                    *cached = Some((backend_name_lower, creation_name, new_backend, receipt));
                                    result
                                }
                                Err(e) => Err(e),
                            }
                        })
                    } else {
                        backend
                            .map(|backend| backend.extract_entities(&text, None))
                            .unwrap_or_else(|| {
                                Err(crate::Error::InvalidInput(format!(
                                    "Zero-shot backend '{backend_name}' requires dataset labels"
                                )))
                            })
                    };

                    // Update progress with time estimates
                    let processed = progress_counter.fetch_add(1, Ordering::Relaxed) + 1;
                    let current_percent = (processed * 100) / total_sentences;
                    let mut last_percent = lock(&last_progress_percent);
                    if current_percent >= *last_percent + 10 || processed.is_multiple_of(10) {
                        let elapsed = lock(&start_time_arc).elapsed();
                        let elapsed_secs = elapsed.as_secs_f64();
                        let rate = if elapsed_secs > 0.0 {
                            processed as f64 / elapsed_secs
                        } else {
                            0.0
                        };
                        let remaining = if rate > 0.0 {
                            ((total_sentences - processed) as f64 / rate) as u64
                        } else {
                            0
                        };
                        let remaining_str = if remaining > 0 {
                            format!(" (~{}s remaining)", remaining)
                        } else {
                            String::new()
                        };
                        eprint!("\rProcessing: {}/{} sentences ({:.0}%) for backend '{}' on dataset '{}'{}\x1b[K",
                            processed, total_sentences, current_percent, backend_name, dataset, remaining_str);
                        *last_percent = current_percent;
                    }

                    let text = sentence.text();
                    (idx, chars_count, gold_entities, entities_result, text.to_string())
                })
                .collect();

            let observed = lock(&observed_construction_receipts);
            if let Some(first) = observed.first() {
                if observed.iter().any(|receipt| receipt != first) {
                    return Err(crate::Error::Inference(
                        "parallel zero-shot backend instances selected different artifact receipts"
                            .to_string(),
                    ));
                }
                zero_shot_construction_receipt = first.clone();
            }
            scheduling = EvaluationScheduling::Rayon {
                configured_threads: configured_rayon_threads,
                effective_threads: lock(&worker_threads).len(),
            };

            // Final progress update with timing
            let total_elapsed = start_time.elapsed();
            let total_secs = total_elapsed.as_secs_f64();
            let (time_str, rate_str) = if total_secs >= 0.01 {
                (
                    format!("{:.2}s", total_secs),
                    format!("{:.1} sentences/s", total_sentences as f64 / total_secs),
                )
            } else {
                let ms = total_elapsed.as_millis();
                let time_str = if ms == 0 {
                    "<1ms".to_string()
                } else {
                    format!("{ms}ms")
                };
                (time_str, "n/a".to_string())
            };
            eprint!(
                "\rProcessing: {}/{} sentences (100.0%) for backend '{}' on dataset '{}' (completed in {}, {})\x1b[K",
                total_sentences,
                total_sentences,
                backend_name,
                dataset,
                time_str,
                rate_str
            );
            eprintln!(); // Newline after progress

            // Aggregate results and track per-example scores if needed
            for (idx, chars_count, gold_entities, entities_result, text) in all_results {
                total_chars += chars_count;

                match entities_result {
                    Ok(entities) => {
                        // Entity offsets are local to each sentence, so match before merging
                        // counts. Pooling entities would make equal offsets in separate
                        // sentences look like true positives.
                        eval_results.merge(&evaluate_entities(&gold_entities, &entities));
                        if track_per_example {
                            // Clone when tracking per-example (need to store in cache)
                            all_gold.extend(gold_entities.clone());
                            all_predicted.extend(entities.clone());
                            per_example_scores.push((gold_entities, entities, text));
                        } else {
                            // Move when not tracking (more efficient)
                            all_gold.extend(gold_entities);
                            all_predicted.extend(entities);
                        }
                    }
                    Err(e) => {
                        return Err(crate::Error::Inference(format!(
                            "Backend '{backend_name}' inference failed for sentence {}: {e}",
                            idx + 1
                        )));
                    }
                }
            }
        }

        #[cfg(not(feature = "eval-parallel"))]
        {
            scheduling = EvaluationScheduling::Sequential;
            // For zero-shot backends, create a cached instance once to avoid recreating for each sentence
            // Non-parallel path still uses Box<dyn Any> for backward compatibility
            let zero_shot_backend: Option<CachedBackendAny> =
                if is_zero_shot && !mapped_labels.is_empty() {
                    Some(Self::create_zero_shot_backend_any(backend_name)?)
                } else {
                    None
                };
            if let Some(cached) = zero_shot_backend.as_ref() {
                zero_shot_construction_receipt = cached.construction_receipt.clone();
            }

            // Sequential processing (fallback when rayon not available)
            for (idx, sentence) in dataset_data.sentences.iter().enumerate() {
                // Progress reporting every 10% or every 10 sentences, whichever is more frequent
                if idx % 10 == 0 || idx == total_sentences - 1 {
                    let progress = ((idx + 1) as f64 / total_sentences as f64) * 100.0;
                    let elapsed = start_time.elapsed();
                    let elapsed_secs = elapsed.as_secs_f64();
                    let rate = if elapsed_secs > 0.0 {
                        (idx + 1) as f64 / elapsed_secs
                    } else {
                        0.0
                    };
                    let remaining = if rate > 0.0 {
                        ((total_sentences.saturating_sub(idx).saturating_sub(1)) as f64 / rate)
                            as u64
                    } else {
                        0
                    };
                    let remaining_str = if remaining > 0 {
                        format!(" (~{}s remaining)", remaining)
                    } else {
                        String::new()
                    };
                    eprint!("\rProcessing: {}/{} sentences ({:.1}%) for backend '{}' on dataset '{}'{}\x1b[K",
                        idx + 1, total_sentences, progress, backend_name, dataset, remaining_str);
                }

                let text = sentence.text();
                total_chars += text.chars().count();

                #[cfg(feature = "eval-profiling")]
                profiling::start("extract_gold_entities");
                // Extract gold entities from sentence.
                let gold_entities: Vec<Entity> = sentence
                    .entities()
                    .iter()
                    .map(|g| {
                        let mut entity =
                            Entity::new(g.text.clone(), g.entity_type.clone(), g.start, g.end, 1.0);
                        entity.provenance = Some(crate::Provenance::ml("gold", 1.0));
                        entity
                    })
                    .collect();
                #[cfg(feature = "eval-profiling")]
                profiling::stop("extract_gold_entities");

                #[cfg(feature = "eval-profiling")]
                profiling::start("backend_inference");

                // Run inference (no prediction cache).
                let entities = {
                    let inference_start = Instant::now();
                    let result = if let Some(ref cached) = zero_shot_backend {
                        // Dereference Box to get &dyn Any (not &Box<dyn Any>)
                        Self::extract_with_cached_backend_any(
                            backend_name,
                            cached.backend.as_ref(),
                            &text,
                            &mapped_labels,
                        )
                    } else {
                        backend
                            .ok_or_else(|| {
                                crate::Error::InvalidInput(format!(
                                    "Zero-shot backend '{backend_name}' requires dataset labels"
                                ))
                            })?
                            .extract_entities(&text, None)
                    };
                    let _ = inference_start; // reserved for future profiling
                    result
                };

                #[cfg(feature = "eval-profiling")]
                profiling::stop("backend_inference");

                match entities {
                    Ok(entities) => {
                        // Match local sentence spans before aggregating counts across the
                        // dataset; offsets restart at zero for each AnnotatedSentence.
                        eval_results.merge(&evaluate_entities(&gold_entities, &entities));
                        if track_per_example {
                            // Clone when tracking per-example (need to store in cache)
                            let gold = gold_entities.clone();
                            all_gold.extend(gold_entities);
                            all_predicted.extend(entities.clone());
                            per_example_scores.push((gold, entities, text.to_string()));
                        } else {
                            // Move when not tracking (more efficient)
                            all_gold.extend(gold_entities);
                            all_predicted.extend(entities);
                        }
                    }
                    Err(e) => {
                        return Err(crate::Error::Inference(format!(
                            "Backend '{backend_name}' inference failed for sentence {}: {e}",
                            idx + 1
                        )));
                    }
                }
            }

            // Final progress update with timing
            let total_elapsed = start_time.elapsed();
            let total_secs = total_elapsed.as_secs_f64();
            let (time_str, rate_str) = if total_secs >= 0.01 {
                (
                    format!("{:.2}s", total_secs),
                    format!("{:.1} sentences/s", total_sentences as f64 / total_secs),
                )
            } else {
                let ms = total_elapsed.as_millis();
                let time_str = if ms == 0 {
                    "<1ms".to_string()
                } else {
                    format!("{ms}ms")
                };
                (time_str, "n/a".to_string())
            };
            eprint!(
                "\rProcessing: {}/{} sentences (100.0%) for backend '{}' on dataset '{}' (completed in {}, {})\x1b[K",
                total_sentences, total_sentences, backend_name, dataset, time_str, rate_str
            );
            eprintln!(); // Newline after progress
        }

        #[cfg(feature = "eval-profiling")]
        profiling::stop("evaluate_ner_task");

        #[cfg(feature = "eval-profiling")]
        profiling::start("compute_metrics");

        let elapsed = start_time.elapsed();
        let chars_per_second = if elapsed.as_secs_f64() > 0.0 {
            total_chars as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };

        #[cfg(feature = "eval-profiling")]
        profiling::stop("compute_metrics");
        let summary = eval_results.summary();

        // Build metrics map
        let mut metrics = HashMap::new();
        metrics.insert("precision".to_string(), summary.strict_precision);
        metrics.insert("recall".to_string(), summary.strict_recall);
        metrics.insert("f1".to_string(), summary.strict_f1);
        metrics.insert("exact_precision".to_string(), summary.exact_precision);
        metrics.insert("exact_recall".to_string(), summary.exact_recall);
        metrics.insert("exact_f1".to_string(), summary.exact_f1);
        metrics.insert("partial_precision".to_string(), summary.partial_precision);
        metrics.insert("partial_recall".to_string(), summary.partial_recall);
        metrics.insert("partial_f1".to_string(), summary.partial_f1);
        metrics.insert("type_precision".to_string(), summary.type_precision);
        metrics.insert("type_recall".to_string(), summary.type_recall);
        metrics.insert("type_f1".to_string(), summary.type_f1);
        metrics.insert("chars_per_second".to_string(), chars_per_second);
        metrics.insert("num_gold".to_string(), all_gold.len() as f64);
        metrics.insert("num_predicted".to_string(), all_predicted.len() as f64);

        // Primary metrics above deliberately score every emitted entity. The dataset's
        // label space is recorded only to account for broader outputs (for example,
        // DATE/MONEY emitted while scoring CoNLL) and to produce a separately named
        // closed-label diagnostic. Do not silently substitute this diagnostic for F1.
        let mut closed_label_results = NerEvalResults::new();
        let mut gold_outside_dataset_labels = 0usize;
        let mut predictions_outside_dataset_labels = 0usize;
        let mut closed_gold_count = 0usize;
        let mut closed_prediction_count = 0usize;
        for (gold, predicted, _) in &per_example_scores {
            let closed_gold: Vec<Entity> = gold
                .iter()
                .filter(|entity| {
                    dataset_label_space.contains(&canonical_entity_type_label(&entity.entity_type))
                })
                .cloned()
                .collect();
            let closed_predicted: Vec<Entity> = predicted
                .iter()
                .filter(|entity| {
                    dataset_label_space.contains(&canonical_entity_type_label(&entity.entity_type))
                })
                .cloned()
                .collect();
            gold_outside_dataset_labels += gold.len() - closed_gold.len();
            predictions_outside_dataset_labels += predicted.len() - closed_predicted.len();
            closed_gold_count += closed_gold.len();
            closed_prediction_count += closed_predicted.len();
            closed_label_results.merge(&evaluate_entities(&closed_gold, &closed_predicted));
        }
        let closed_summary = closed_label_results.summary();
        metrics.insert(
            "dataset_label_closed_precision".to_string(),
            closed_summary.strict_precision,
        );
        metrics.insert(
            "dataset_label_closed_recall".to_string(),
            closed_summary.strict_recall,
        );
        metrics.insert(
            "dataset_label_closed_f1".to_string(),
            closed_summary.strict_f1,
        );
        metrics.insert(
            "gold_outside_dataset_labels".to_string(),
            gold_outside_dataset_labels as f64,
        );
        metrics.insert(
            "predictions_outside_dataset_labels".to_string(),
            predictions_outside_dataset_labels as f64,
        );

        // CORE-KG-inspired diagnostics (heuristic): duplication + noise in predictions.
        // Sentence-local offsets and repeated mentions make cross-sentence duplicate
        // detection misleading, so scope duplicate detection to each sentence.
        let q = compute_document_extraction_quality_metrics(
            per_example_scores
                .iter()
                .map(|(_, predicted, _)| predicted.as_slice()),
        );
        metrics.insert("pred_duplication_rate".to_string(), q.duplication_rate);
        metrics.insert("pred_noise_rate".to_string(), q.noise_rate);
        metrics.insert("pred_duplicates".to_string(), q.duplicates as f64);
        metrics.insert("pred_noisy".to_string(), q.noisy as f64);

        // Store per-example scores for later use in stratified metrics and confidence intervals
        {
            // Use blocking lock for cache - it's not critical path and avoids "would block" errors
            // If lock fails (poisoned), just skip caching rather than failing the evaluation
            let mut cache_guard = lock(&self.per_example_scores_cache);
            if !per_example_scores.is_empty() {
                *cache_guard = Some(per_example_scores);
            } else {
                *cache_guard = None;
            }
            // If lock fails, continue without caching (non-critical)
        }

        Ok(NerEvaluationReceipt {
            metrics,
            configured_labels: is_zero_shot.then_some(mapped_labels),
            label_policy: NerLabelPolicyReport {
                primary_policy:
                    "strict span-and-type micro score over all emitted entities; no label filtering"
                        .to_string(),
                dataset_label_space: dataset_label_space.into_iter().collect(),
                operational_gold_outside_dataset_labels: gold_outside_dataset_labels,
                operational_predictions_outside_dataset_labels: predictions_outside_dataset_labels,
                dataset_label_closed_diagnostic: ClosedLabelDiagnostic {
                    strict_precision: closed_summary.strict_precision,
                    strict_recall: closed_summary.strict_recall,
                    strict_f1: closed_summary.strict_f1,
                    gold_entities_scored: closed_gold_count,
                    predicted_entities_scored: closed_prediction_count,
                },
            },
            construction_receipt: zero_shot_construction_receipt,
            scheduling,
        })
    }

    /// Map dataset entity type labels to model-compatible labels.
    ///
    /// Handles common label variations (e.g., "PER" → "person", "PERSON" → "person").
    /// Also handles domain-specific mappings (e.g., MIT Movie "Actor" → "person").
    /// Also limits labels for backends with restrictions (e.g., NuNER only supports 3 labels).
    /// Public for testing purposes.
    pub fn map_dataset_labels_to_model(dataset_labels: &[&str], backend_name: &str) -> Vec<String> {
        let backend_lower = backend_name.to_lowercase();

        // NuNER has a limitation - it fails with GatherElements errors when using more than
        // its default 3 labels. Always use the exact default labels in the exact order.
        // The order matters because the model internally maps label index to entity type.
        if backend_lower == "nuner" {
            // Must match NuNER::from_pretrained default_labels exactly: person, organization, location
            return vec![
                "person".to_string(),
                "organization".to_string(),
                "location".to_string(),
            ];
        }

        dataset_labels
            .iter()
            .map(|label| {
                // Normalize label to lowercase for matching
                let normalized = label.to_lowercase();
                match normalized.as_str() {
                    // Person variations
                    "per" | "person" => "person".to_string(),
                    // Organization variations
                    "org" | "organization" | "organisation" | "corporation" | "company" => {
                        "organization".to_string()
                    }
                    // Location variations (including WNUT geo-loc)
                    "loc" | "location" | "place" | "gpe" | "geo-loc" => "location".to_string(),
                    // Other common types
                    "misc" | "miscellaneous" | "other" => "misc".to_string(),
                    "date" => "date".to_string(),
                    "time" => "time".to_string(),
                    "money" | "currency" => "money".to_string(),
                    "percent" | "percentage" => "percent".to_string(),
                    "product" | "prod" => "product".to_string(),
                    "event" => "event".to_string(),
                    "facility" | "fac" => "facility".to_string(),
                    "work_of_art" | "workofart" => "work_of_art".to_string(),
                    "law" => "law".to_string(),
                    "language" => "language".to_string(),
                    "norp" => "norp".to_string(),
                    // Domain-specific mappings (MIT Movie, MIT Restaurant, etc.)
                    "actor" | "character" | "director" | "producer" | "writer" | "cast" => {
                        "person".to_string()
                    }
                    "restaurant_name" | "restaurant" | "cuisine" | "dish" | "food" => {
                        "organization".to_string()
                    }
                    "disease" | "disorder" | "syndrome" => "disease".to_string(),
                    "chemical" | "drug" | "medication" | "compound" => "chemical".to_string(),
                    // For zero-shot backends, preserve original labels (they can handle any type)
                    _ if matches!(
                        backend_lower.as_str(),
                        "gliner_onnx"
                            | "gliner_candle"
                            | "gliner_multitask"
                            | "gliner2_fastino"
                            | "gliner_poly"
                            | "universal_ner"
                    ) =>
                    {
                        label.to_lowercase()
                    }
                    // For other backends, try to map or use original
                    _ => label.to_lowercase(),
                }
            })
            .collect()
    }

    #[cfg(feature = "onnx")]
    fn gliner_onnx_construction_receipt(
        model: &anno::backends::gliner_onnx::GLiNEROnnx,
    ) -> BackendConstructionReceipt {
        let paths = model.artifact_paths();
        let mut files = vec![
            ("graph".to_string(), paths.graph.clone()),
            ("tokenizer".to_string(), paths.tokenizer.clone()),
        ];
        for (role, path) in [
            ("config", paths.config.as_ref()),
            ("label_encoder", paths.label_encoder.as_ref()),
            ("label_tokenizer", paths.label_tokenizer.as_ref()),
        ] {
            if let Some(path) = path {
                files.push((role.to_string(), path.clone()));
            }
        }
        BackendConstructionReceipt {
            model_artifacts: Some(artifact_receipt_from_paths(
                "gliner_onnx",
                model.model_name(),
                None,
                files,
            )),
        }
    }

    #[cfg(feature = "gliner2-fastino")]
    fn fastino_construction_receipt(
        model: &anno::backends::gliner2_fastino::GLiNER2Fastino,
    ) -> BackendConstructionReceipt {
        let paths = model.artifact_paths();
        let mut files = vec![("tokenizer".to_string(), paths.tokenizer)];
        if let Some(config) = paths.config {
            files.push(("config".to_string(), config));
        }
        files.extend(paths.graphs);
        BackendConstructionReceipt {
            model_artifacts: Some(artifact_receipt_from_paths(
                "gliner2_fastino",
                model.model_id(),
                model.model_revision(),
                files,
            )),
        }
    }

    #[cfg(feature = "eval-parallel")]
    fn cached_backend_construction_receipt(backend: &CachedBackend) -> BackendConstructionReceipt {
        match backend {
            #[cfg(feature = "onnx")]
            CachedBackend::GLiNEROnnx(model) => Self::gliner_onnx_construction_receipt(model),
            #[cfg(feature = "gliner2-fastino")]
            CachedBackend::GLiNER2Fastino(model) => Self::fastino_construction_receipt(model),
            _ => BackendConstructionReceipt::default(),
        }
    }

    /// Create a zero-shot backend instance for the sequential evaluator.
    ///
    /// This avoids recreating the model for every sentence, which causes ONNX errors.
    #[cfg(not(feature = "eval-parallel"))]
    fn create_zero_shot_backend_any(backend_name: &str) -> Result<CachedBackendAny> {
        Self::create_zero_shot_backend_impl(backend_name)
    }

    /// Create a zero-shot backend instance (returns enum for type safety).
    ///
    /// This avoids recreating the model for every sentence, which causes ONNX errors.
    #[cfg(feature = "eval-parallel")]
    fn create_zero_shot_backend(backend_name: &str) -> Result<CachedBackend> {
        match backend_name.to_lowercase().as_str() {
            #[cfg(feature = "onnx")]
            "nuner" => {
                use crate::DEFAULT_NUNER_MODEL;
                use anno::backends::nuner::NuNER;
                let nuner = NuNER::from_pretrained(DEFAULT_NUNER_MODEL)?;
                Ok(CachedBackend::NuNER(nuner))
            }
            #[cfg(not(feature = "onnx"))]
            "nuner" => Err(crate::Error::FeatureNotAvailable(
                "NuNER requires the 'onnx' feature".to_string(),
            )),
            #[cfg(feature = "onnx")]
            "gliner_onnx" | "gliner" => {
                use crate::DEFAULT_GLINER_MODEL;
                use anno::backends::gliner_onnx::GLiNEROnnx;
                let gliner = GLiNEROnnx::new(DEFAULT_GLINER_MODEL)?;
                Ok(CachedBackend::GLiNEROnnx(gliner))
            }
            #[cfg(not(feature = "onnx"))]
            "gliner_onnx" | "gliner" => Err(crate::Error::FeatureNotAvailable(
                "GLiNER requires the 'onnx' feature".to_string(),
            )),
            #[cfg(feature = "onnx")]
            "gliner_multitask" => {
                use crate::DEFAULT_GLINER_MULTITASK_MODEL;
                use anno::backends::gliner_multitask::GLiNERMultitaskOnnx;
                let gliner_multitask =
                    GLiNERMultitaskOnnx::from_pretrained(DEFAULT_GLINER_MULTITASK_MODEL)?;
                Ok(CachedBackend::GLiNERMultitaskOnnx(gliner_multitask))
            }
            #[cfg(not(feature = "onnx"))]
            "gliner_multitask" => Err(crate::Error::FeatureNotAvailable(
                "GLiNER multi-task requires the 'onnx' feature".to_string(),
            )),
            #[cfg(feature = "gliner2-fastino")]
            "gliner2_fastino" => {
                use anno::backends::gliner2_fastino::{
                    GLiNER2Fastino, GLiNER2FastinoConfig, SUPPORTED_GLINER2_FASTINO_MODEL,
                    SUPPORTED_GLINER2_FASTINO_REVISION,
                };
                Ok(CachedBackend::GLiNER2Fastino(
                    GLiNER2Fastino::from_pretrained_with_config(
                        SUPPORTED_GLINER2_FASTINO_MODEL,
                        GLiNER2FastinoConfig::default()
                            .with_model_revision(SUPPORTED_GLINER2_FASTINO_REVISION),
                    )?,
                ))
            }
            #[cfg(not(feature = "gliner2-fastino"))]
            "gliner2_fastino" => Err(crate::Error::FeatureNotAvailable(
                "GLiNER2 Fastino requires the 'gliner2-fastino' feature".to_string(),
            )),
            #[cfg(feature = "candle")]
            "gliner_candle" => {
                use crate::DEFAULT_GLINER_MODEL;
                use anno::backends::gliner_candle::GLiNERCandle;
                let gliner = GLiNERCandle::from_pretrained(DEFAULT_GLINER_MODEL)?;
                Ok(CachedBackend::GLiNERCandle(gliner))
            }
            #[cfg(not(feature = "candle"))]
            "gliner_candle" => Err(crate::Error::FeatureNotAvailable(
                "GLiNER Candle requires the 'candle' feature".to_string(),
            )),
            #[cfg(feature = "onnx")]
            "gliner_poly" => {
                use anno::backends::gliner_poly::GLiNERPoly;
                use anno::DEFAULT_GLINER_POLY_MODEL;
                let gliner_poly = GLiNERPoly::new(DEFAULT_GLINER_POLY_MODEL)?;
                Ok(CachedBackend::GLiNERPoly(gliner_poly))
            }
            #[cfg(not(feature = "onnx"))]
            "gliner_poly" => Err(crate::Error::FeatureNotAvailable(
                "GLiNER Poly requires the 'onnx' feature".to_string(),
            )),
            "universal_ner" => {
                use anno::backends::universal_ner::UniversalNER;
                let universal_ner = UniversalNER::new()?;
                Ok(CachedBackend::UniversalNER(universal_ner))
            }
            _ => Err(crate::Error::InvalidInput(format!(
                "Unknown zero-shot backend: {}",
                backend_name
            ))),
        }
    }

    /// Internal implementation that creates backend as Box<dyn Any> (for non-parallel path).
    #[cfg(not(feature = "eval-parallel"))]
    fn create_zero_shot_backend_impl(backend_name: &str) -> Result<CachedBackendAny> {
        match backend_name.to_lowercase().as_str() {
            "nuner" => {
                #[cfg(feature = "onnx")]
                {
                    use crate::DEFAULT_NUNER_MODEL;
                    use anno::backends::nuner::NuNER;
                    let nuner = NuNER::from_pretrained(DEFAULT_NUNER_MODEL)?;
                    Ok(CachedBackendAny {
                        backend: Box::new(nuner),
                        construction_receipt: BackendConstructionReceipt::default(),
                    })
                }
                #[cfg(not(feature = "onnx"))]
                {
                    Err(crate::Error::FeatureNotAvailable(
                        "NuNER requires the 'onnx' feature".to_string(),
                    ))
                }
            }
            "gliner_onnx" | "gliner" => {
                #[cfg(feature = "onnx")]
                {
                    use crate::DEFAULT_GLINER_MODEL;
                    use anno::backends::gliner_onnx::GLiNEROnnx;
                    let gliner = GLiNEROnnx::new(DEFAULT_GLINER_MODEL)?;
                    let construction_receipt = Self::gliner_onnx_construction_receipt(&gliner);
                    Ok(CachedBackendAny {
                        backend: Box::new(gliner),
                        construction_receipt,
                    })
                }
                #[cfg(not(feature = "onnx"))]
                {
                    Err(crate::Error::FeatureNotAvailable(
                        "GLiNER requires the 'onnx' feature".to_string(),
                    ))
                }
            }
            "gliner_multitask" => {
                #[cfg(feature = "onnx")]
                {
                    use crate::DEFAULT_GLINER_MULTITASK_MODEL;
                    use anno::backends::gliner_multitask::GLiNERMultitaskOnnx;
                    let gliner_multitask =
                        GLiNERMultitaskOnnx::from_pretrained(DEFAULT_GLINER_MULTITASK_MODEL)?;
                    Ok(CachedBackendAny {
                        backend: Box::new(gliner_multitask),
                        construction_receipt: BackendConstructionReceipt::default(),
                    })
                }
                #[cfg(not(feature = "onnx"))]
                {
                    Err(crate::Error::FeatureNotAvailable(
                        "GLiNER multi-task requires the 'onnx' feature".to_string(),
                    ))
                }
            }
            "gliner2_fastino" => {
                #[cfg(feature = "gliner2-fastino")]
                {
                    use anno::backends::gliner2_fastino::{
                        GLiNER2Fastino, GLiNER2FastinoConfig, SUPPORTED_GLINER2_FASTINO_MODEL,
                        SUPPORTED_GLINER2_FASTINO_REVISION,
                    };
                    let fastino = GLiNER2Fastino::from_pretrained_with_config(
                        SUPPORTED_GLINER2_FASTINO_MODEL,
                        GLiNER2FastinoConfig::default()
                            .with_model_revision(SUPPORTED_GLINER2_FASTINO_REVISION),
                    )?;
                    let construction_receipt = Self::fastino_construction_receipt(&fastino);
                    Ok(CachedBackendAny {
                        backend: Box::new(fastino),
                        construction_receipt,
                    })
                }
                #[cfg(not(feature = "gliner2-fastino"))]
                {
                    Err(crate::Error::FeatureNotAvailable(
                        "GLiNER2 Fastino requires the 'gliner2-fastino' feature".to_string(),
                    ))
                }
            }
            "gliner_candle" => {
                #[cfg(feature = "candle")]
                {
                    use crate::DEFAULT_GLINER_MODEL;
                    use anno::backends::gliner_candle::GLiNERCandle;
                    let gliner = GLiNERCandle::from_pretrained(DEFAULT_GLINER_MODEL)?;
                    Ok(CachedBackendAny {
                        backend: Box::new(gliner),
                        construction_receipt: BackendConstructionReceipt::default(),
                    })
                }
                #[cfg(not(feature = "candle"))]
                {
                    Err(crate::Error::FeatureNotAvailable(
                        "GLiNER Candle requires the 'candle' feature".to_string(),
                    ))
                }
            }
            "gliner_poly" => {
                #[cfg(feature = "onnx")]
                {
                    use anno::backends::gliner_poly::GLiNERPoly;
                    use anno::DEFAULT_GLINER_POLY_MODEL;
                    let gliner_poly = GLiNERPoly::new(DEFAULT_GLINER_POLY_MODEL)?;
                    Ok(CachedBackendAny {
                        backend: Box::new(gliner_poly),
                        construction_receipt: BackendConstructionReceipt::default(),
                    })
                }
                #[cfg(not(feature = "onnx"))]
                {
                    Err(crate::Error::FeatureNotAvailable(
                        "GLiNER Poly requires the 'onnx' feature".to_string(),
                    ))
                }
            }
            "universal_ner" => {
                use anno::backends::universal_ner::UniversalNER;
                let universal_ner = UniversalNER::new()?;
                Ok(CachedBackendAny {
                    backend: Box::new(universal_ner),
                    construction_receipt: BackendConstructionReceipt::default(),
                })
            }
            _ => Err(crate::Error::InvalidInput(format!(
                "Unknown zero-shot backend: {}",
                backend_name
            ))),
        }
    }

    /// Extract entities using cached zero-shot backend instance.
    #[allow(unused_variables)] // False positives - variables are used in feature-gated code
    #[cfg(feature = "eval-parallel")]
    fn extract_with_cached_backend(
        cached: &CachedBackend,
        text: &str,
        labels: &[String],
    ) -> Result<Vec<Entity>> {
        // Convert labels to &str slice
        let label_strs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

        match cached {
            #[cfg(feature = "onnx")]
            CachedBackend::NuNER(nuner) => nuner.extract(text, &label_strs, 0.5),
            #[cfg(feature = "onnx")]
            CachedBackend::GLiNEROnnx(gliner) => {
                let result = gliner.extract(text, &label_strs, 0.5);
                if std::env::var("ANNO_DEBUG_EXTRACT").is_ok() {
                    eprintln!(
                        "DEBUG gliner result: {:?}",
                        result.as_ref().map(|v| v.len())
                    );
                }
                result
            }
            #[cfg(feature = "onnx")]
            CachedBackend::GLiNERMultitaskOnnx(gliner_multitask) => {
                use anno::backends::gliner_multitask::TaskSchema;
                let schema = TaskSchema::new().with_entities(&label_strs);
                let result = gliner_multitask.extract(text, &schema)?;
                Ok(result.entities)
            }
            #[cfg(feature = "gliner2-fastino")]
            CachedBackend::GLiNER2Fastino(gliner) => {
                gliner.extract_with_types(text, &label_strs, 0.5)
            }
            #[cfg(feature = "candle")]
            CachedBackend::GLiNERCandle(gliner) => gliner.extract(text, &label_strs, 0.5),
            #[cfg(feature = "onnx")]
            CachedBackend::GLiNERPoly(gliner_poly) => {
                gliner_poly.extract_with_types(text, &label_strs, 0.5)
            }
            CachedBackend::UniversalNER(universal_ner) => {
                universal_ner.extract_with_types(text, &label_strs, 0.5)
            }
        }
    }

    /// Extract entities using cached zero-shot backend instance (Box<dyn Any> version for non-parallel path).
    #[allow(unused_variables)] // False positives - variables are used in feature-gated code
    #[cfg(not(feature = "eval-parallel"))]
    fn extract_with_cached_backend_any(
        backend_name: &str,
        cached: &dyn std::any::Any,
        text: &str,
        labels: &[String],
    ) -> Result<Vec<Entity>> {
        // Convert labels to &str slice
        let label_strs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

        match backend_name.to_lowercase().as_str() {
            "nuner" => {
                #[cfg(feature = "onnx")]
                {
                    if let Some(nuner) = cached.downcast_ref::<anno::backends::nuner::NuNER>() {
                        let result = nuner.extract(text, &label_strs, 0.5);
                        if std::env::var("ANNO_DEBUG_NUNER").is_ok() {
                            eprintln!(
                                "DEBUG nuner: text={:?} labels={:?} result={:?}",
                                text.chars().take(30).collect::<String>(),
                                label_strs,
                                result.as_ref().map(|v| v.len())
                            );
                        }
                        result
                    } else {
                        Err(crate::Error::InvalidInput(
                            "Failed to downcast cached NuNER backend".to_string(),
                        ))
                    }
                }
                #[cfg(not(feature = "onnx"))]
                {
                    Err(crate::Error::FeatureNotAvailable(
                        "NuNER requires the 'onnx' feature".to_string(),
                    ))
                }
            }
            "gliner_onnx" | "gliner" => {
                #[cfg(feature = "onnx")]
                {
                    if let Some(gliner) =
                        cached.downcast_ref::<anno::backends::gliner_onnx::GLiNEROnnx>()
                    {
                        gliner.extract(text, &label_strs, 0.5)
                    } else {
                        Err(crate::Error::InvalidInput(
                            "Failed to downcast cached GLiNER backend".to_string(),
                        ))
                    }
                }
                #[cfg(not(feature = "onnx"))]
                {
                    Err(crate::Error::FeatureNotAvailable(
                        "GLiNER requires the 'onnx' feature".to_string(),
                    ))
                }
            }
            "gliner_multitask" => {
                #[cfg(feature = "onnx")]
                {
                    use anno::backends::gliner_multitask::TaskSchema;
                    if let Some(gliner_multitask) =
                        cached
                            .downcast_ref::<anno::backends::gliner_multitask::GLiNERMultitaskOnnx>()
                    {
                        let schema = TaskSchema::new().with_entities(&label_strs);
                        let result = gliner_multitask.extract(text, &schema);
                        if std::env::var("ANNO_DEBUG_GLINER_MULTITASK").is_ok() {
                            eprintln!(
                                "DEBUG gliner_multitask: text={:?} labels={:?} result={:?}",
                                &text[..text.len().min(50)],
                                label_strs,
                                result.as_ref().map(|r| r.entities.len())
                            );
                        }
                        Ok(result?.entities)
                    } else {
                        if std::env::var("ANNO_DEBUG_GLINER_MULTITASK").is_ok() {
                            eprintln!("DEBUG gliner_multitask: downcast FAILED");
                        }
                        Err(crate::Error::InvalidInput(
                            "Failed to downcast cached GLiNER multi-task backend".to_string(),
                        ))
                    }
                }
                #[cfg(not(feature = "onnx"))]
                {
                    Err(crate::Error::FeatureNotAvailable(
                        "GLiNER multi-task requires the 'onnx' feature".to_string(),
                    ))
                }
            }
            "gliner2_fastino" => {
                #[cfg(feature = "gliner2-fastino")]
                {
                    if let Some(gliner) =
                        cached.downcast_ref::<anno::backends::gliner2_fastino::GLiNER2Fastino>()
                    {
                        gliner.extract_with_types(text, &label_strs, 0.5)
                    } else {
                        Err(crate::Error::InvalidInput(
                            "Failed to downcast cached GLiNER2 Fastino backend".to_string(),
                        ))
                    }
                }
                #[cfg(not(feature = "gliner2-fastino"))]
                {
                    Err(crate::Error::FeatureNotAvailable(
                        "GLiNER2 Fastino requires the 'gliner2-fastino' feature".to_string(),
                    ))
                }
            }
            "gliner_candle" => {
                #[cfg(feature = "candle")]
                {
                    if let Some(gliner) =
                        cached.downcast_ref::<anno::backends::gliner_candle::GLiNERCandle>()
                    {
                        gliner.extract(text, &label_strs, 0.5)
                    } else {
                        Err(crate::Error::InvalidInput(
                            "Failed to downcast cached GLiNER Candle backend".to_string(),
                        ))
                    }
                }
                #[cfg(not(feature = "candle"))]
                {
                    Err(crate::Error::FeatureNotAvailable(
                        "GLiNER Candle requires the 'candle' feature".to_string(),
                    ))
                }
            }
            "gliner_poly" => {
                #[cfg(feature = "onnx")]
                {
                    if let Some(gliner_poly) =
                        cached.downcast_ref::<anno::backends::gliner_poly::GLiNERPoly>()
                    {
                        gliner_poly.extract_with_types(text, &label_strs, 0.5)
                    } else {
                        Err(crate::Error::InvalidInput(
                            "Failed to downcast cached GLiNER Poly backend".to_string(),
                        ))
                    }
                }
                #[cfg(not(feature = "onnx"))]
                {
                    Err(crate::Error::FeatureNotAvailable(
                        "GLiNER Poly requires the 'onnx' feature".to_string(),
                    ))
                }
            }
            "universal_ner" => {
                if let Some(universal_ner) =
                    cached.downcast_ref::<anno::backends::universal_ner::UniversalNER>()
                {
                    universal_ner.extract_with_types(text, &label_strs, 0.5)
                } else {
                    Err(crate::Error::InvalidInput(
                        "Failed to downcast cached UniversalNER backend".to_string(),
                    ))
                }
            }
            _ => Err(crate::Error::InvalidInput(format!(
                "Unknown zero-shot backend: {}",
                backend_name
            ))),
        }
    }

    /// Evaluate coreference task.
    ///
    /// For `IntraDocCoref` and `AbstractAnaphora`, runs per-document coreference.
    /// For `InterDocCoref`, groups documents by topic and runs cross-document clustering.
    fn evaluate_coref_task(
        &self,
        task: Task,
        backend_name: &str,
        dataset_data: &LoadedDataset,
        config: &TaskEvalConfig,
    ) -> Result<HashMap<String, f64>> {
        use crate::eval::backend_factory::create_coref_resolver;
        use crate::eval::coref::entities_to_chains;
        use crate::eval::coref_metrics::{CorefEvaluation, WindowFragmentationStats};

        // Try to load coreference documents if dataset supports it
        let gold_docs = if dataset_data.id.is_coreference() {
            match self.loader.load_coref(dataset_data.id) {
                Ok(docs) => {
                    if docs.is_empty() {
                        // A cached-only run must not refresh an empty cache entry.
                        if config.require_cached {
                            return Err(crate::Error::InvalidInput(format!(
                                "Coreference dataset {:?} has no documents in the local cache; cached-only evaluation forbids downloading",
                                dataset_data.id
                            )));
                        }
                        // If load_coref returns empty, try downloading first.
                        #[cfg(feature = "eval")]
                        {
                            if let Err(e) = self.loader.load_or_download_coref(dataset_data.id) {
                                return Err(crate::Error::InvalidInput(format!(
                                    "Failed to load coreference dataset {:?}: {}",
                                    dataset_data.id, e
                                )));
                            }
                            // Retry after download
                            self.loader.load_coref(dataset_data.id)?
                        }
                        #[cfg(not(feature = "eval"))]
                        {
                            return Err(crate::Error::InvalidInput(format!(
                                "Coreference dataset {:?} not cached. Enable eval feature to auto-download.",
                                dataset_data.id
                            )));
                        }
                    } else {
                        docs
                    }
                }
                Err(e) => {
                    // A cached-only run must not refresh a missing or invalid cache entry.
                    if config.require_cached {
                        return Err(crate::Error::InvalidInput(format!(
                            "Coreference dataset {:?} is not usable from the local cache: {}; cached-only evaluation forbids downloading",
                            dataset_data.id, e
                        )));
                    }
                    // Try downloading if not cached.
                    #[cfg(feature = "eval")]
                    {
                        if let Err(dl_err) = self.loader.load_or_download_coref(dataset_data.id) {
                            return Err(crate::Error::InvalidInput(format!(
                                "Failed to load/download coreference dataset {:?}: {} (original: {})",
                                dataset_data.id, dl_err, e
                            )));
                        }
                        // Retry after download
                        self.loader.load_coref(dataset_data.id)?
                    }
                    #[cfg(not(feature = "eval"))]
                    {
                        return Err(crate::Error::InvalidInput(format!(
                            "Coreference dataset {:?} not cached: {}. Enable eval feature to auto-download.",
                            dataset_data.id, e
                        )));
                    }
                }
            }
        } else {
            // Not a coreference dataset - return error metrics
            let mut metrics = HashMap::new();
            metrics.insert(
                "num_sentences".to_string(),
                dataset_data.sentences.len() as f64,
            );
            metrics.insert("error".to_string(), 1.0);
            return Ok(metrics);
        };

        // IMPORTANT: `TaskEvalConfig.max_examples` is interpreted as *max documents* for
        // coreference datasets (not "sentences"). Without this, `benchmark --max-examples N`
        // still evaluates the full coref dataset, which can be extremely slow.
        let gold_docs = if let Some(max) = config.max_examples.filter(|m| *m > 0) {
            if max >= gold_docs.len() {
                gold_docs
            } else {
                let seed = config.seed.unwrap_or(42);
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};

                let mut indices: Vec<(usize, u64)> = (0..gold_docs.len())
                    .map(|i| {
                        let mut hasher = DefaultHasher::new();
                        seed.hash(&mut hasher);
                        i.hash(&mut hasher);
                        (i, hasher.finish())
                    })
                    .collect();
                indices.sort_by_key(|(_, hash)| *hash);

                let selected: std::collections::HashSet<usize> =
                    indices.into_iter().take(max).map(|(i, _)| i).collect();

                gold_docs
                    .into_iter()
                    .enumerate()
                    .filter_map(|(i, doc)| selected.contains(&i).then_some(doc))
                    .collect()
            }
        } else {
            gold_docs
        };

        // ---- InterDocCoref: cross-document clustering path ----
        if task == Task::InterDocCoref {
            return self.evaluate_inter_doc_coref(&gold_docs, backend_name, config);
        }

        // ---- IntraDocCoref / AbstractAnaphora: per-document path (unchanged) ----

        // Create coreference resolver (not a Model backend)
        // Use custom resolver if provided, otherwise create from backend_name
        let resolver: std::sync::Arc<dyn crate::eval::coref_resolver::CoreferenceResolver> =
            if let Some(ref custom_resolver) = config.custom_coref_resolver {
                // Use the custom resolver directly (e.g., TrainedBoxCorefResolver from matryoshka-box)
                custom_resolver.clone()
            } else {
                // Create resolver from backend_name (e.g., "coref_resolver", "box", etc.)
                std::sync::Arc::from(create_coref_resolver(backend_name)?)
            };

        let mut all_predicted_chains = Vec::new();
        let mut all_gold_chains = Vec::new();

        // Long-document stitching diagnostics (CorefInst-style window fragmentation).
        // We use a fixed default windowing scheme matching other long-doc configs in this repo.
        let frag_window_size: usize = 4000;
        let frag_window_overlap: usize = 256;
        let mut frag_multiwindow_gold_chains: usize = 0;
        let mut frag_fragmented_gold_chains: usize = 0;
        let mut frag_boundary_checks: usize = 0;
        let mut frag_boundary_splits: usize = 0;
        let mut frag_missing_mentions_in_multiwindow_chains: usize = 0;

        // IMPORTANT: Coref metrics in `coref_metrics.rs` key mentions only by (start,end).
        // If we concatenate multiple documents without offsetting spans, identical spans across docs
        // collide and corrupt metrics. We avoid this by assigning a monotonically increasing
        // character base offset per document.
        let mut cumulative_char_base: usize = 0;

        fn offset_chains(
            mut chains: Vec<crate::eval::coref::CorefChain>,
            base: usize,
        ) -> Vec<crate::eval::coref::CorefChain> {
            if base == 0 {
                return chains;
            }
            for chain in &mut chains {
                for m in &mut chain.mentions {
                    m.start = m.start.saturating_add(base);
                    m.end = m.end.saturating_add(base);
                    if let Some(hs) = m.head_start.as_mut() {
                        *hs = hs.saturating_add(base);
                    }
                    if let Some(he) = m.head_end.as_mut() {
                        *he = he.saturating_add(base);
                    }
                }
            }
            chains
        }

        for doc in &gold_docs {
            let doc_base = cumulative_char_base;
            let doc_char_len = doc.text.chars().count();
            cumulative_char_base =
                cumulative_char_base.saturating_add(doc_char_len.saturating_add(1));

            // Collect gold chains from the document
            all_gold_chains.extend(offset_chains(doc.chains.clone(), doc_base));

            // Check if this is a text-based coref backend (CorefBackend)
            // rather than an entity-based resolver (CoreferenceResolver).
            let is_text_based_coref = matches!(backend_name, "fcoref" | "f-coref" | "fastcoref");

            let predicted_chains = if is_text_based_coref {
                // Text-based coref: run directly on raw text, bypass NER extraction.
                // This is the proper path for neural coref models (FCoref, etc.)
                use crate::eval::backend_factory::create_coref_backend;
                match create_coref_backend(backend_name) {
                    Ok(coref_backend) => {
                        match coref_backend.resolve(&doc.text) {
                            Ok(clusters) => {
                                // Convert CorefCluster -> CorefChain
                                use crate::eval::coref::{CorefChain, Mention};
                                clusters
                                    .into_iter()
                                    .map(|cluster| {
                                        let mentions = cluster
                                            .spans
                                            .iter()
                                            .zip(cluster.mentions.iter())
                                            .map(|(&(start, end), text)| {
                                                Mention::new(text, start, end)
                                            })
                                            .collect();
                                        CorefChain {
                                            mentions,
                                            cluster_id: Some(anno::CanonicalId::new(
                                                cluster.id as u64,
                                            )),
                                            entity_type: None,
                                        }
                                    })
                                    .collect()
                            }
                            Err(e) => {
                                return Err(crate::Error::Inference(format!(
                                    "Coreference backend '{backend_name}' inference failed for document {:?}: {e}",
                                    doc.doc_id
                                )));
                            }
                        }
                    }
                    Err(e) => {
                        return Err(crate::Error::FeatureNotAvailable(format!(
                            "Failed to create coref backend '{}': {}",
                            backend_name, e
                        )));
                    }
                }
            } else if config.coref_use_gold_mentions {
                // Gold-mention mode: evaluate clustering only.
                //
                // We deliberately exclude zero-length mentions (CorefUD empty nodes) from the
                // resolver input because most resolvers operate on overt spans.
                let mut gold_entities: Vec<crate::Entity> = Vec::new();
                for chain in &doc.chains {
                    for m in &chain.mentions {
                        let is_zero =
                            m.mention_type == Some(anno::MentionType::Zero) || m.start == m.end;
                        if is_zero {
                            continue;
                        }
                        let et = m
                            .entity_type
                            .as_deref()
                            .map(|t| {
                                // Best-effort mapping from CorefUD etype (person/place/organization/...)
                                // to our coarse EntityType. Everything else becomes Other.
                                let tl = t.to_lowercase();
                                if tl.contains("person") {
                                    crate::EntityType::Person
                                } else if tl.contains("place") || tl.contains("loc") {
                                    crate::EntityType::Location
                                } else if tl.contains("org") {
                                    crate::EntityType::Organization
                                } else {
                                    crate::EntityType::custom(t, crate::EntityCategory::Misc)
                                }
                            })
                            .unwrap_or_else(|| {
                                crate::EntityType::custom("mention", crate::EntityCategory::Misc)
                            });

                        gold_entities.push(crate::Entity::new(&m.text, et, m.start, m.end, 1.0));
                    }
                }

                let resolved_entities = resolver.resolve(&gold_entities);
                entities_to_chains(&resolved_entities)
            } else {
                // End-to-end mode: extract mentions via NER backend, then cluster.
                // Use a NER backend to extract entities first (heuristic or stacked as default)
                let ner_backend_name = match backend_name {
                    // Coref resolvers are not NER backends. Pick a sensible default mention detector.
                    "coref_resolver" | "mention_ranking" | "box" => "stacked",
                    // If the user passed an actual NER backend name, allow it.
                    other => other,
                };
                let ner_backend = BackendFactory::create(ner_backend_name)?;

                match ner_backend.extract_entities(&doc.text, None) {
                    Ok(entities) => {
                        let resolved_entities = resolver.resolve(&entities);
                        entities_to_chains(&resolved_entities)
                    }
                    Err(e) => {
                        return Err(crate::Error::Inference(format!(
                            "NER backend '{ner_backend_name}' inference failed for coreference document {:?}: {e}",
                            doc.doc_id
                        )));
                    }
                }
            };

            if let Some(fs) = WindowFragmentationStats::compute(
                &predicted_chains,
                &doc.chains,
                frag_window_size,
                frag_window_overlap,
            ) {
                frag_multiwindow_gold_chains += fs.multiwindow_gold_chains;
                frag_fragmented_gold_chains += fs.fragmented_gold_chains;
                frag_boundary_checks += fs.boundary_checks;
                frag_boundary_splits += fs.boundary_splits;
                frag_missing_mentions_in_multiwindow_chains +=
                    fs.missing_mentions_in_multiwindow_chains;
            }

            all_predicted_chains.extend(offset_chains(predicted_chains, doc_base));
        }

        // Compute coreference metrics
        let eval = CorefEvaluation::compute(&all_predicted_chains, &all_gold_chains);

        let mut metrics = HashMap::new();
        metrics.insert("num_docs".to_string(), gold_docs.len() as f64);
        metrics.insert("muc_precision".to_string(), eval.muc.precision);
        metrics.insert("muc_recall".to_string(), eval.muc.recall);
        metrics.insert("muc_f1".to_string(), eval.muc.f1);
        metrics.insert("b3_precision".to_string(), eval.b_cubed.precision);
        metrics.insert("b3_recall".to_string(), eval.b_cubed.recall);
        metrics.insert("b3_f1".to_string(), eval.b_cubed.f1);
        metrics.insert("ceaf_e_precision".to_string(), eval.ceaf_e.precision);
        metrics.insert("ceaf_e_recall".to_string(), eval.ceaf_e.recall);
        metrics.insert("ceaf_e_f1".to_string(), eval.ceaf_e.f1);
        metrics.insert("ceaf_m_precision".to_string(), eval.ceaf_m.precision);
        metrics.insert("ceaf_m_recall".to_string(), eval.ceaf_m.recall);
        metrics.insert("ceaf_m_f1".to_string(), eval.ceaf_m.f1);

        // Add chain-length stratification metrics
        if let Some(ref chain_stats) = eval.chain_stats {
            metrics.insert(
                "chain_long_count".to_string(),
                chain_stats.long_chain_count as f64,
            );
            metrics.insert(
                "chain_short_count".to_string(),
                chain_stats.short_chain_count as f64,
            );
            metrics.insert(
                "chain_singleton_count".to_string(),
                chain_stats.singleton_count as f64,
            );
            metrics.insert("chain_long_f1".to_string(), chain_stats.long_chain_f1);
            metrics.insert("chain_short_f1".to_string(), chain_stats.short_chain_f1);
            metrics.insert("chain_singleton_f1".to_string(), chain_stats.singleton_f1);
        }
        metrics.insert("lea_precision".to_string(), eval.lea.precision);
        metrics.insert("lea_recall".to_string(), eval.lea.recall);
        metrics.insert("lea_f1".to_string(), eval.lea.f1);
        metrics.insert("blanc_precision".to_string(), eval.blanc.precision);
        metrics.insert("blanc_recall".to_string(), eval.blanc.recall);
        metrics.insert("blanc_f1".to_string(), eval.blanc.f1);
        metrics.insert("conll_f1".to_string(), eval.conll_f1);

        if let Some(z) = eval.zero_anaphor {
            metrics.insert("zero_precision".to_string(), z.precision);
            metrics.insert("zero_recall".to_string(), z.recall);
            metrics.insert("zero_f1".to_string(), z.f1);
            metrics.insert("zero_tp".to_string(), z.tp as f64);
            metrics.insert("zero_wl".to_string(), z.wl as f64);
            metrics.insert("zero_fp".to_string(), z.fp as f64);
            metrics.insert("zero_fn".to_string(), z.fn_ as f64);
            metrics.insert("zero_gold_anaphors".to_string(), z.gold_anaphors as f64);
            metrics.insert("zero_pred_anaphors".to_string(), z.pred_anaphors as f64);
        }

        if frag_multiwindow_gold_chains > 0 {
            metrics.insert(
                "window_multiwindow_gold_chains".to_string(),
                frag_multiwindow_gold_chains as f64,
            );
            metrics.insert(
                "window_fragmented_gold_chains".to_string(),
                frag_fragmented_gold_chains as f64,
            );
            metrics.insert(
                "window_fragmentation_rate".to_string(),
                frag_fragmented_gold_chains as f64 / frag_multiwindow_gold_chains as f64,
            );
            metrics.insert(
                "window_boundary_checks".to_string(),
                frag_boundary_checks as f64,
            );
            metrics.insert(
                "window_boundary_splits".to_string(),
                frag_boundary_splits as f64,
            );
            if frag_boundary_checks > 0 {
                metrics.insert(
                    "window_boundary_split_rate".to_string(),
                    frag_boundary_splits as f64 / frag_boundary_checks as f64,
                );
            }
            metrics.insert(
                "window_missing_mentions_in_multiwindow_chains".to_string(),
                frag_missing_mentions_in_multiwindow_chains as f64,
            );
            metrics.insert("window_size".to_string(), frag_window_size as f64);
            metrics.insert("window_overlap".to_string(), frag_window_overlap as f64);
        }
        metrics.insert("num_documents".to_string(), gold_docs.len() as f64);
        metrics.insert("num_gold_chains".to_string(), all_gold_chains.len() as f64);
        metrics.insert(
            "num_predicted_chains".to_string(),
            all_predicted_chains.len() as f64,
        );

        Ok(metrics)
    }

    /// Evaluate inter-document (cross-document) coreference.
    ///
    /// Groups `CorefDocument`s by topic (from metadata), builds `Topic` objects with
    /// gold `CrossDocCluster`s, and runs `evaluate_cross_document()`.
    fn evaluate_inter_doc_coref(
        &self,
        gold_docs: &[crate::eval::coref::CorefDocument],
        _backend_name: &str,
        _config: &TaskEvalConfig,
    ) -> Result<HashMap<String, f64>> {
        use crate::eval::cdcr::{CrossDocCluster, Document};
        use crate::eval::cluster_encoder::{CosineMergeScorer, HeuristicClusterEncoder};
        use crate::eval::cross_context_eval::{
            evaluate_cross_document, CrossContextEvalConfig, Topic,
        };

        // Group docs by topic (from metadata, or treat each doc as its own topic)
        let mut topics_map: HashMap<String, Vec<&crate::eval::coref::CorefDocument>> =
            HashMap::new();

        for doc in gold_docs {
            // Extract topic from doc_id (format: "topicN_fileM" from ECB+ parser)
            let topic_key = doc
                .doc_id
                .as_deref()
                .and_then(|id| id.split('_').next())
                .unwrap_or("default")
                .to_string();
            topics_map.entry(topic_key).or_default().push(doc);
        }

        // Build Topic objects
        let mut topics: Vec<Topic> = Vec::new();
        let mut topic_keys: Vec<_> = topics_map.keys().cloned().collect();
        topic_keys.sort();

        for topic_key in &topic_keys {
            let coref_docs = &topics_map[topic_key];
            let mut topic = Topic::new(topic_key);

            // Convert CorefDocuments to cdcr::Documents and build gold clusters
            // Each CorefChain in each doc that shares a chain across docs becomes a cross-doc cluster.
            // For ECB+, the chain IDs encode cross-doc identity.
            let mut chain_to_mentions: HashMap<String, Vec<(String, usize)>> = HashMap::new();

            for coref_doc in coref_docs {
                let doc_id = coref_doc
                    .doc_id
                    .clone()
                    .unwrap_or_else(|| format!("doc_{}", topic.documents.len()));

                // Build cdcr::Document with entities from gold mentions
                let mut entities: Vec<anno::Entity> = Vec::new();
                for (chain_idx, chain) in coref_doc.chains.iter().enumerate() {
                    for mention in &chain.mentions {
                        let et = mention
                            .entity_type
                            .as_deref()
                            .map(|t| {
                                let tl = t.to_lowercase();
                                if tl.contains("person") {
                                    anno::EntityType::Person
                                } else if tl.contains("loc") || tl.contains("place") {
                                    anno::EntityType::Location
                                } else if tl.contains("org") {
                                    anno::EntityType::Organization
                                } else {
                                    anno::EntityType::custom(t, anno::EntityCategory::Misc)
                                }
                            })
                            .unwrap_or(anno::EntityType::custom(
                                "mention",
                                anno::EntityCategory::Misc,
                            ));

                        let entity_idx = entities.len();
                        entities.push(anno::Entity::new(
                            &mention.text,
                            et,
                            mention.start,
                            mention.end,
                            1.0,
                        ));

                        // Track chain membership for cross-doc clustering
                        let chain_key = format!("{}_{}", topic_key, chain_idx);
                        chain_to_mentions
                            .entry(chain_key)
                            .or_default()
                            .push((doc_id.clone(), entity_idx));
                    }
                }

                let cdcr_doc = Document::new(&doc_id, &coref_doc.text).with_entities(entities);
                topic.add_document(cdcr_doc);
            }

            // Build gold CrossDocClusters from chain_to_mentions
            for mentions in chain_to_mentions.values() {
                if mentions.len() < 2 {
                    continue; // Skip singletons for cross-doc
                }
                let mut cluster = CrossDocCluster::new(topic.gold_clusters.len() as u64, "");
                cluster.mentions = mentions.clone();
                topic.add_gold_cluster(cluster);
            }

            topics.push(topic);
        }

        // Run cross-document evaluation
        let encoder = HeuristicClusterEncoder::new(64);
        let scorer = CosineMergeScorer::new();
        let config = CrossContextEvalConfig::default();

        let results = evaluate_cross_document(&topics, encoder, scorer, &config)?;

        // Convert to flat metrics HashMap
        let mut metrics = HashMap::new();
        metrics.insert("conll_f1".to_string(), results.conll_f1);
        metrics.insert("muc_f1".to_string(), results.muc.f1);
        metrics.insert("muc_precision".to_string(), results.muc.precision);
        metrics.insert("muc_recall".to_string(), results.muc.recall);
        metrics.insert("b3_f1".to_string(), results.b_cubed.f1);
        metrics.insert("b3_precision".to_string(), results.b_cubed.precision);
        metrics.insert("b3_recall".to_string(), results.b_cubed.recall);
        metrics.insert("ceaf_e_f1".to_string(), results.ceaf_e.f1);
        metrics.insert("ceaf_e_precision".to_string(), results.ceaf_e.precision);
        metrics.insert("ceaf_e_recall".to_string(), results.ceaf_e.recall);
        metrics.insert("lea_f1".to_string(), results.lea.f1);
        metrics.insert("lea_precision".to_string(), results.lea.precision);
        metrics.insert("lea_recall".to_string(), results.lea.recall);
        metrics.insert("num_topics".to_string(), topics.len() as f64);
        metrics.insert("num_documents".to_string(), results.num_contexts as f64);
        metrics.insert(
            "num_gold_clusters".to_string(),
            results.num_gold_clusters as f64,
        );
        metrics.insert(
            "num_pred_clusters".to_string(),
            results.num_pred_clusters as f64,
        );
        metrics.insert("purity".to_string(), results.avg_cluster_size);
        metrics.insert("time_ms".to_string(), results.time_ms);
        metrics.insert("is_cross_doc".to_string(), 1.0);

        Ok(metrics)
    }

    /// Evaluate relation extraction task.
    fn evaluate_relation_task(
        &self,
        backend_name: &str,
        backend: &dyn Model,
        dataset_data: &LoadedDataset,
        config: &TaskEvalConfig,
    ) -> Result<HashMap<String, f64>> {
        use crate::eval::relation::{
            evaluate_relations, RelationEvalConfig, RelationGold, RelationPrediction,
        };

        // Load gold relations from cache, with an opt-in download fallback.
        let relation_docs = match self.loader.load_relation(dataset_data.id) {
            Ok(docs) => docs,
            Err(e) => {
                if config.require_cached {
                    return Err(crate::Error::InvalidInput(format!(
                        "Relations for {:?} are not usable from the local cache: {}; cached-only evaluation forbids downloading",
                        dataset_data.id, e
                    )));
                }
                // If not cached, try downloading (if eval feature enabled).
                #[cfg(feature = "eval")]
                {
                    match self.loader.load_or_download_relation(dataset_data.id) {
                        Ok(docs) => docs,
                        Err(e) => {
                            return Err(crate::Error::InvalidInput(format!(
                                "Failed to load/download relations for {:?}: {e}",
                                dataset_data.id
                            )));
                        }
                    }
                }
                #[cfg(not(feature = "eval"))]
                {
                    return Err(crate::Error::InvalidInput(format!(
                        "Relations for {:?} are not cached and the 'eval' feature cannot download them",
                        dataset_data.id
                    )));
                }
            }
        };

        // Collect all gold relations
        let mut all_gold_relations: Vec<RelationGold> = Vec::new();
        for doc in &relation_docs {
            all_gold_relations.extend(doc.relations.iter().cloned());
        }

        // Extract predicted relations from backend
        let mut all_predicted_relations: Vec<RelationPrediction> = Vec::new();

        // Extract relations using RelationExtractor if backend supports it
        // GLiNER multi-task backends implement RelationExtractor
        use anno::backends::inference::RelationExtractor;

        // Try to create RelationExtractor instance for relation extraction backends
        let relation_extractor: Option<Box<dyn RelationExtractor>> = match backend_name {
            #[cfg(feature = "onnx")]
            "gliner_multitask" | "gliner_multitask_onnx" => {
                use crate::DEFAULT_GLINER_MULTITASK_MODEL;
                use anno::backends::gliner_multitask::GLiNERMultitaskOnnx;
                match GLiNERMultitaskOnnx::from_pretrained(DEFAULT_GLINER_MULTITASK_MODEL) {
                    Ok(extractor) => Some(Box::new(extractor) as Box<dyn RelationExtractor>),
                    Err(e) => {
                        eprintln!(
                            "Warning: Failed to create GLiNER multi-task (ONNX) for relation extraction: {}",
                            e
                        );
                        None
                    }
                }
            }
            #[cfg(all(feature = "candle", feature = "onnx"))]
            "gliner_multitask_candle" => {
                use crate::DEFAULT_GLINER_MULTITASK_MODEL;
                use anno::backends::gliner_multitask::GLiNERMultitaskCandle;
                match GLiNERMultitaskCandle::from_pretrained(DEFAULT_GLINER_MULTITASK_MODEL) {
                    Ok(extractor) => Some(Box::new(extractor) as Box<dyn RelationExtractor>),
                    Err(e) => {
                        eprintln!(
                            "Warning: Failed to create GLiNER multi-task (Candle) for relation extraction: {}",
                            e
                        );
                        None
                    }
                }
            }
            "tplinker" | "tplink" => {
                use anno::backends::tplinker::TPLinker;
                match TPLinker::new() {
                    Ok(extractor) => Some(Box::new(extractor) as Box<dyn RelationExtractor>),
                    Err(e) => {
                        eprintln!("Warning: Failed to create TPLinker: {e}");
                        None
                    }
                }
            }
            _ => None,
        };

        // Extract relations from each document
        let allow_oracle_entities = std::env::var("ANNO_RELATION_ORACLE_ENTITIES")
            .ok()
            .map(|v| {
                let v = v.trim().to_lowercase();
                v == "1" || v == "true" || v == "yes" || v == "y"
            })
            .unwrap_or(true);
        // TPLinker uses ONNX neural inference when the `onnx` feature is enabled,
        // with a heuristic fallback otherwise. For relation datasets like DocRED/CHisIEC
        // that provide gold entity spans/types, allow an optional “oracle entities” mode
        // so the eval is not dominated by mention detection mismatch.
        let tplinker_oracle_entities = std::env::var("ANNO_RELATION_TPLINKER_ORACLE_ENTITIES")
            .ok()
            .map(|v| {
                let v = v.trim().to_lowercase();
                v == "1" || v == "true" || v == "yes" || v == "y"
            })
            .unwrap_or(true);
        let mut oracle_docs_used: usize = 0;
        let mut oracle_tplinker_docs_used: usize = 0;

        for (doc_index, doc) in relation_docs.iter().enumerate() {
            let text = &doc.text;

            if let Some(ref rel_extractor) = relation_extractor {
                // Use RelationExtractor to extract relations
                // Get entity types and relation types from gold relations
                let entity_types: Vec<&str> = doc
                    .relations
                    .iter()
                    .flat_map(|r| vec![r.head_type.as_str(), r.tail_type.as_str()])
                    .collect::<std::collections::HashSet<_>>()
                    .into_iter()
                    .collect();

                let relation_types: Vec<&str> = doc
                    .relations
                    .iter()
                    .map(|r| r.relation_type.as_str())
                    .collect::<std::collections::HashSet<_>>()
                    .into_iter()
                    .collect();

                // Use configurable threshold from TaskEvalConfig
                match rel_extractor.extract_with_relations(
                    text,
                    &entity_types,
                    &relation_types,
                    config.relation_threshold,
                ) {
                    Ok(extraction) => {
                        // TPLinker baseline: optionally use gold entity spans/types as the candidate
                        // entity set, then run our lightweight relation heuristics. This avoids the
                        // baseline being “always junk” due purely to entity boundary mismatch.
                        if backend_name.starts_with("tplinker")
                            && allow_oracle_entities
                            && tplinker_oracle_entities
                            && !doc.relations.is_empty()
                        {
                            use anno::backends::inference::{
                                extract_relation_triples_simple, RelationExtractionConfig,
                            };
                            use anno::{Confidence, Entity as PredEntity, EntityType};
                            use std::collections::BTreeMap;

                            // Dedup entities by (start,end,type,text) while preserving a stable order.
                            let mut by_key: BTreeMap<(usize, usize, String, String), PredEntity> =
                                BTreeMap::new();
                            for r in &doc.relations {
                                for (ty, span, txt) in [
                                    (&r.head_type, r.head_span, &r.head_text),
                                    (&r.tail_type, r.tail_span, &r.tail_text),
                                ] {
                                    let (start, end) = span;
                                    let text_fallback: String = if !txt.is_empty() {
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
                                        .entry((start, end, ty.clone(), text_fallback))
                                        .or_insert(ent);
                                }
                            }
                            let oracle_entities: Vec<PredEntity> = by_key.into_values().collect();

                            let rel_strs: Vec<&str> = relation_types.iter().map(|s| &**s).collect();
                            let rel_cfg = RelationExtractionConfig {
                                threshold: Confidence::new(config.relation_threshold as f64),
                                max_span_distance: 120,
                                extract_triggers: false,
                            };
                            let triples = extract_relation_triples_simple(
                                &oracle_entities,
                                text,
                                &rel_strs,
                                &rel_cfg,
                            );
                            for t in &triples {
                                if let (Some(head), Some(tail)) = (
                                    oracle_entities.get(t.head_idx),
                                    oracle_entities.get(t.tail_idx),
                                ) {
                                    all_predicted_relations.push(RelationPrediction {
                                        head_span: (head.start(), head.end()),
                                        head_type: head.entity_type.as_label().to_string(),
                                        tail_span: (tail.start(), tail.end()),
                                        tail_type: tail.entity_type.as_label().to_string(),
                                        relation_type: t.relation_type.clone(),
                                        confidence: t.confidence.value() as f32,
                                    });
                                }
                            }
                            oracle_docs_used += 1;
                            oracle_tplinker_docs_used += 1;
                            continue;
                        }

                        // If the backend's NER produces no entities (common for cross-lingual
                        // datasets like CHisIEC when using an English GLiNER multi-task model), fall back to
                        // an “oracle entities” baseline: use the gold entity spans/types as the
                        // candidate entity set, then run our lightweight relation heuristics.
                        //
                        // This keeps the relation evaluation non-degenerate and makes the
                        // matrix/muxer signal usable, without pretending the NER step worked.
                        // Scope this fallback narrowly:
                        // - only for CHisIEC (cross-lingual classical Chinese)
                        // - only for GLiNER multi-task (English NER tends to produce zero entities there)
                        //
                        // This keeps the eval non-degenerate *without* collapsing backend
                        // differences for other arms (e.g. `tplinker`).
                        if dataset_data.id == DatasetId::CHisIEC
                            && backend_name.starts_with("gliner_multitask")
                            && allow_oracle_entities
                            && extraction.entities.is_empty()
                            && !doc.relations.is_empty()
                        {
                            use anno::backends::inference::{
                                extract_relation_triples_simple, RelationExtractionConfig,
                            };
                            use anno::{Confidence, Entity as PredEntity, EntityType};
                            use std::collections::BTreeMap;

                            // Dedup entities by (start,end,type,text) while preserving a stable order.
                            let mut by_key: BTreeMap<(usize, usize, String, String), PredEntity> =
                                BTreeMap::new();
                            for r in &doc.relations {
                                for (ty, span, txt) in [
                                    (&r.head_type, r.head_span, &r.head_text),
                                    (&r.tail_type, r.tail_span, &r.tail_text),
                                ] {
                                    let (start, end) = span;
                                    let text_fallback: String = if !txt.is_empty() {
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
                                        .entry((start, end, ty.clone(), text_fallback))
                                        .or_insert(ent);
                                }
                            }
                            let oracle_entities: Vec<PredEntity> = by_key.into_values().collect();

                            let rel_strs: Vec<&str> = relation_types.iter().map(|s| &**s).collect();
                            let rel_cfg = RelationExtractionConfig {
                                threshold: Confidence::new(config.relation_threshold as f64),
                                max_span_distance: 120,
                                extract_triggers: false,
                            };
                            let triples = extract_relation_triples_simple(
                                &oracle_entities,
                                text,
                                &rel_strs,
                                &rel_cfg,
                            );
                            for t in &triples {
                                if let (Some(head), Some(tail)) = (
                                    oracle_entities.get(t.head_idx),
                                    oracle_entities.get(t.tail_idx),
                                ) {
                                    all_predicted_relations.push(RelationPrediction {
                                        head_span: (head.start(), head.end()),
                                        head_type: head.entity_type.as_label().to_string(),
                                        tail_span: (tail.start(), tail.end()),
                                        tail_type: tail.entity_type.as_label().to_string(),
                                        relation_type: t.relation_type.clone(),
                                        confidence: t.confidence.value() as f32,
                                    });
                                }
                            }
                            oracle_docs_used += 1;
                            continue;
                        }

                        // Convert ExtractionWithRelations to RelationPrediction
                        for triple in &extraction.relations {
                            if let (Some(head), Some(tail)) = (
                                extraction.entities.get(triple.head_idx),
                                extraction.entities.get(triple.tail_idx),
                            ) {
                                all_predicted_relations.push(RelationPrediction {
                                    head_span: (head.start(), head.end()),
                                    head_type: head.entity_type.as_label().to_string(),
                                    tail_span: (tail.start(), tail.end()),
                                    tail_type: tail.entity_type.as_label().to_string(),
                                    relation_type: triple.relation_type.clone(),
                                    confidence: triple.confidence.value() as f32,
                                });
                            }
                        }
                    }
                    Err(e) => {
                        return Err(crate::Error::Inference(format!(
                            "Relation backend '{backend_name}' inference failed for document {}: {e}",
                            doc_index + 1
                        )));
                    }
                }
            } else {
                // Fallback: Extract entities and create proximity-based heuristic relations
                let entities = match backend.extract_entities(text, None) {
                    Ok(ents) => ents,
                    Err(e) => {
                        return Err(crate::Error::Inference(format!(
                            "Relation fallback backend '{backend_name}' entity inference failed for document {}: {e}",
                            doc_index + 1
                        )));
                    }
                };

                // Create proximity-based relations for nearby entity pairs
                if entities.len() >= 2 {
                    for i in 0..entities.len() {
                        for j in (i + 1)..entities.len().min(i + 3) {
                            let head = &entities[i];
                            let tail = &entities[j];

                            all_predicted_relations.push(RelationPrediction {
                                head_span: (head.start(), head.end()),
                                head_type: head.entity_type.as_label().to_string(),
                                tail_span: (tail.start(), tail.end()),
                                tail_type: tail.entity_type.as_label().to_string(),
                                relation_type: "RELATED".to_string(), // Proximity heuristic
                                confidence: 0.5,
                            });
                        }
                    }
                }
            }
        }

        // Evaluate relations
        // Relation datasets in `anno` (e.g. DocRED/CHisIEC) commonly use a richer entity-type
        // schema than our `EntityType` enum. Require span + relation-type agreement, but do not
        // hard-require entity-type string equality by default.
        let config = RelationEvalConfig {
            require_entity_type_match: false,
            ..RelationEvalConfig::default()
        };
        let metrics_result =
            evaluate_relations(&all_gold_relations, &all_predicted_relations, &config);

        let mut metrics = HashMap::new();
        metrics.insert(
            "boundary_precision".to_string(),
            metrics_result.boundary_precision,
        );
        metrics.insert(
            "boundary_recall".to_string(),
            metrics_result.boundary_recall,
        );
        metrics.insert("boundary_f1".to_string(), metrics_result.boundary_f1);
        metrics.insert(
            "strict_precision".to_string(),
            metrics_result.strict_precision,
        );
        metrics.insert("strict_recall".to_string(), metrics_result.strict_recall);
        metrics.insert("strict_f1".to_string(), metrics_result.strict_f1);
        metrics.insert(
            "num_gold_relations".to_string(),
            all_gold_relations.len() as f64,
        );
        metrics.insert(
            "num_predicted_relations".to_string(),
            all_predicted_relations.len() as f64,
        );
        metrics.insert("oracle_docs_used".to_string(), oracle_docs_used as f64);
        metrics.insert(
            "oracle_tplinker_docs_used".to_string(),
            oracle_tplinker_docs_used as f64,
        );
        metrics.insert(
            "num_sentences".to_string(),
            dataset_data.sentences.len() as f64,
        );

        Ok(metrics)
    }

    /// Evaluate text classification task.
    ///
    /// Loader encodes the gold label as the `B-<LABEL>` tag on the single token for each example.
    fn evaluate_text_classification_task(
        &self,
        backend_name: &str,
        dataset: DatasetId,
        dataset_data: &LoadedDataset,
        _config: &TaskEvalConfig,
    ) -> Result<HashMap<String, f64>> {
        // For now, only GLiNER multi-task is wired for classification in this repo.
        let backend_name_norm = backend_name.to_lowercase();
        if backend_name_norm != "gliner_multitask"
            && backend_name_norm != "gliner_multitask_onnx"
            && backend_name_norm != "gliner_multitask_candle"
        {
            return Err(crate::Error::InvalidInput(format!(
                "Text classification currently only supports gliner_multitask backends (got {})",
                backend_name
            )));
        }

        // Prefer registry class labels when available, otherwise derive from gold labels in the data.
        let mut labels: Vec<String> = dataset
            .entity_types()
            .iter()
            .map(|s| s.to_string())
            .collect();
        if labels.is_empty() {
            for s in &dataset_data.sentences {
                let tag = s.tokens.first().map(|t| t.ner_tag.as_str()).unwrap_or("O");
                let gold = tag
                    .strip_prefix("B-")
                    .or_else(|| tag.strip_prefix("I-"))
                    .unwrap_or(tag)
                    .trim();
                if gold.is_empty() || gold == "O" {
                    continue;
                }
                labels.push(gold.to_string());
            }
            labels.sort();
            labels.dedup();
        }
        if labels.is_empty() {
            return Err(crate::Error::InvalidInput(format!(
                "Dataset {:?} has no class labels (neither registry entity_types nor gold labels in loaded data)",
                dataset
            )));
        }
        // If we don't have any compiled gliner_multitask backend (neither onnx nor candle),
        // classification is not available even if `eval` is enabled.
        #[cfg(any(feature = "onnx", feature = "candle"))]
        {
            use crate::eval::metrics::ClassificationMetrics;

            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

            // Create backend instance for classification.
            #[cfg(feature = "onnx")]
            let extractor = if backend_name_norm == "gliner_multitask"
                || backend_name_norm == "gliner_multitask_onnx"
            {
                use crate::DEFAULT_GLINER_MULTITASK_MODEL;
                use anno::backends::gliner_multitask::GLiNERMultitaskOnnx;
                Some(GLiNERMultitaskOnnx::from_pretrained(
                    DEFAULT_GLINER_MULTITASK_MODEL,
                )?)
            } else {
                None
            };
            #[cfg(not(feature = "onnx"))]
            let extractor: Option<()> = None;

            #[cfg(all(feature = "candle", feature = "onnx"))]
            let extractor_candle = if backend_name_norm == "gliner_multitask_candle" {
                use crate::DEFAULT_GLINER_MULTITASK_MODEL;
                use anno::backends::gliner_multitask::GLiNERMultitaskCandle;
                Some(GLiNERMultitaskCandle::from_pretrained(
                    DEFAULT_GLINER_MULTITASK_MODEL,
                )?)
            } else {
                None
            };
            #[cfg(not(all(feature = "candle", feature = "onnx")))]
            let extractor_candle: Option<()> = None;

            if extractor.is_none() && extractor_candle.is_none() {
                return Err(crate::Error::FeatureNotAvailable(
                    "Text classification requires a gliner_multitask backend with 'onnx' (and optionally 'candle') enabled"
                        .to_string(),
                ));
            }

            #[cfg(feature = "onnx")]
            let schema = anno::backends::gliner_multitask::TaskSchema::new().with_classification(
                "topic",
                &label_refs,
                false,
            );
            #[cfg(not(feature = "onnx"))]
            let schema = ();
            #[cfg(not(feature = "onnx"))]
            let _ = (&label_refs, &schema);

            let mut m = ClassificationMetrics::new();
            for s in &dataset_data.sentences {
                let text = s.text();
                if text.trim().is_empty() {
                    continue;
                }
                let tag = s.tokens.first().map(|t| t.ner_tag.as_str()).unwrap_or("O");
                let gold = tag
                    .strip_prefix("B-")
                    .or_else(|| tag.strip_prefix("I-"))
                    .unwrap_or(tag)
                    .to_string();
                if gold.is_empty() || gold == "O" {
                    continue;
                }

                #[cfg(feature = "onnx")]
                let pred_labels: Vec<String> = if let Some(ref gliner_multitask) = extractor {
                    let r = gliner_multitask.extract(&text, &schema)?;
                    r.classifications
                        .get("topic")
                        .map(|c| c.labels.clone())
                        .unwrap_or_default()
                } else {
                    Vec::new()
                };
                #[cfg(all(feature = "candle", feature = "onnx"))]
                let pred_labels: Vec<String> = if let Some(ref gliner_multitask) = extractor_candle
                {
                    let r = gliner_multitask.extract(&text, &schema)?;
                    r.classifications
                        .get("topic")
                        .map(|c| c.labels.clone())
                        .unwrap_or_default()
                } else {
                    pred_labels
                };
                #[cfg(not(any(feature = "onnx", all(feature = "candle", feature = "onnx"))))]
                let pred_labels: Vec<String> = Vec::new();

                let pred = pred_labels
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "Unknown".to_string());
                m.add(&pred, &gold);
            }

            let mut metrics = HashMap::new();
            metrics.insert("accuracy".to_string(), m.accuracy());
            metrics.insert("macro_f1".to_string(), m.macro_f1());
            metrics.insert("micro_f1".to_string(), m.micro_f1());
            metrics.insert("weighted_f1".to_string(), m.weighted_f1());
            metrics.insert("num_examples".to_string(), m.total as f64);
            Ok(metrics)
        }

        #[cfg(not(any(feature = "onnx", feature = "candle")))]
        {
            Err(crate::Error::FeatureNotAvailable(
                "Text classification requires a gliner_multitask backend with 'onnx' or 'candle' enabled"
                    .to_string(),
            ))
        }
    }
}

impl Default for TaskEvaluator {
    /// Creates a default `TaskEvaluator`.
    ///
    /// # Panics
    ///
    /// This function will panic if `DatasetLoader::new()` fails.
    /// In production code, prefer using `TaskEvaluator::new()` which returns a `Result`.
    fn default() -> Self {
        Self::new().expect("Failed to create TaskEvaluator: DatasetLoader initialization failed. Use TaskEvaluator::new() for proper error handling.")
    }
}

/// Generate a markdown report from evaluation results.
impl ComprehensiveEvalResults {
    /// Convert evaluation results to a markdown-formatted report.
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();
        md.push_str("# Eval Report\n\n");

        let ner_policy_rows: Vec<_> = self
            .results
            .iter()
            .filter_map(|result| {
                result
                    .provenance
                    .ner_label_policy
                    .as_ref()
                    .map(|policy| (result, policy))
            })
            .collect();
        if !ner_policy_rows.is_empty() {
            md.push_str("## NER label policy\n\n");
            md.push_str("Primary NER P/R/F1 uses strict span-and-type matching over every entity emitted by the backend. The closed-label value below is a separately named diagnostic using the same scorer after retaining the dataset label space; it never replaces the primary score.\n\n");
            md.push_str("| Dataset | Backend | Pred outside dataset labels | Gold outside dataset labels | Closed-label F1 (diagnostic) |\n");
            md.push_str("|---------|---------|-----------------------------|-----------------------------|------------------------------|\n");
            for (result, policy) in ner_policy_rows {
                md.push_str(&format!(
                    "| {:?} | {} | {} | {} | {:.1} |\n",
                    result.dataset,
                    result.backend,
                    policy.operational_predictions_outside_dataset_labels,
                    policy.operational_gold_outside_dataset_labels,
                    policy.dataset_label_closed_diagnostic.strict_f1 * 100.0,
                ));
            }
            md.push('\n');
        }

        // Backend macro-averages by task (successful-only).
        //
        // This is intentionally “objective backing”: within a single run/config, report
        // mean primary metric per backend per task. (Do not mix tasks.)
        {
            use std::collections::HashMap;
            let mut by_task_backend: HashMap<(Task, String), Vec<f64>> = HashMap::new();
            for r in &self.results {
                if !r.success {
                    continue;
                }
                if let Some(v) = r.primary_f1() {
                    by_task_backend
                        .entry((r.task, r.backend.clone()))
                        .or_default()
                        .push(v * 100.0);
                }
            }

            if !by_task_backend.is_empty() {
                md.push_str("## Backend macro averages (successful only)\n\n");
                md.push_str("| Task | Backend | Avg primary metric | n |\n");
                md.push_str("|------|---------|--------------------|---|\n");

                let mut entries: Vec<(Task, String, f64, usize)> = by_task_backend
                    .into_iter()
                    .map(|((task, backend), vals)| {
                        let n = vals.len();
                        let avg = if n == 0 {
                            0.0
                        } else {
                            vals.iter().sum::<f64>() / (n as f64)
                        };
                        (task, backend, avg, n)
                    })
                    .collect();

                // Sort by task name, then avg descending.
                entries.sort_by(|a, b| match a.0.name().cmp(b.0.name()) {
                    std::cmp::Ordering::Equal => {
                        b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal)
                    }
                    other => other,
                });

                for (task, backend, avg, n) in entries {
                    md.push_str(&format!(
                        "| {} | {} | {:.1} | {} |\n",
                        task.name(),
                        backend,
                        avg,
                        n
                    ));
                }
                md.push('\n');
            }
        }

        // Dense summary line
        let avg_examples: f64 = self
            .results
            .iter()
            .filter(|r| r.success)
            .map(|r| r.num_examples as f64)
            .sum::<f64>()
            / self.summary.successful.max(1) as f64;
        let avg_time: f64 = self
            .results
            .iter()
            .filter_map(|r| r.duration_ms)
            .sum::<f64>()
            / self
                .results
                .iter()
                .filter(|r| r.duration_ms.is_some())
                .count()
                .max(1) as f64;

        md.push_str(&format!(
            "Total: {} | ✓: {} | ⊘: {} | ✗: {} | Avg examples: {:.0} | Avg time: {:.0}ms\n\n",
            self.summary.total_combinations,
            self.summary.successful,
            self.summary.skipped,
            self.summary.failed,
            avg_examples,
            avg_time
        ));

        // Failures first (most important for debugging)
        let failures: Vec<_> = self
            .results
            .iter()
            .filter(|r| !r.success && !r.is_skipped())
            .collect();

        if !failures.is_empty() {
            md.push_str("## Failures\n\n");
            md.push_str("| Task | Dataset | Backend | Error |\n");
            md.push_str("|------|---------|---------|-------|\n");
            for result in &failures {
                let error = result
                    .error
                    .as_ref()
                    .map(|e| e.replace('|', "\\|").replace('\n', " "))
                    .unwrap_or_else(|| "N/A".to_string());
                md.push_str(&format!(
                    "| {} | {:?} | {} | {} |\n",
                    result.task.name(),
                    result.dataset,
                    result.backend,
                    error
                ));
            }
            md.push('\n');
        }

        // Error patterns
        let mut error_patterns: HashMap<String, usize> = HashMap::new();
        for result in failures.iter() {
            if let Some(ref err) = result.error {
                // Extract error pattern (first 50 chars or key phrase)
                let pattern = if err.len() > 50 {
                    err.chars().take(50).collect::<String>() + "..."
                } else {
                    err.clone()
                };
                *error_patterns.entry(pattern).or_insert(0) += 1;
            }
        }

        if !error_patterns.is_empty() {
            md.push_str("## Error Patterns\n\n");
            let mut patterns: Vec<_> = error_patterns.iter().collect();
            patterns.sort_by(|a, b| b.1.cmp(a.1));
            for (pattern, count) in patterns {
                md.push_str(&format!("- [{}x] {}\n", count, pattern));
            }
            md.push('\n');
        }

        md.push_str("## Results\n\n");

        // Filter out skipped entries for cleaner report (show summary instead)
        let skipped_count = self.results.iter().filter(|r| r.is_skipped()).count();
        if skipped_count > 0 {
            md.push_str(&format!(
                "**Note**: {} combinations skipped (features not enabled or incompatible). Showing successful and failed results only.\n\n",
                skipped_count
            ));
        }

        // Add compatibility notes
        md.push_str("**Compatibility Notes**:\n");
        md.push_str("- `stacked`: Combines pattern+heuristic, supports structured entities (date/time/money/etc) and named entities (PER/ORG/LOC), but not biomedical types\n");
        md.push_str("- `pattern`: Only structured entities (date, time, money, percent, email, URL, phone)\n");
        md.push_str("- `heuristic`: Only named entities (Person, Organization, Location)\n");
        md.push_str("- `incompatible`: Backend doesn't support dataset entity types (expected for non-zero-shot backends on fine-grained datasets)\n");
        md.push_str("- `load-failed`: Dataset failed to download/load (HuggingFace API errors, network issues, etc.)\n");
        md.push_str("- `empty-dataset`: Dataset loaded but contains no sentences\n");
        md.push_str("- `0.0 F1` with N>0: Backend doesn't support dataset entity types\n");
        md.push_str("- `N=0` or `N=1`: Dataset parsing issue or insufficient data\n\n");

        // Group results by task, filtering out skipped
        let mut by_task: HashMap<Task, Vec<&TaskEvalResult>> = HashMap::new();
        for result in &self.results {
            if !result.is_skipped() {
                by_task.entry(result.task).or_default().push(result);
            }
        }

        for (task, mut results) in by_task {
            md.push_str(&format!("### {}\n\n", task.name()));

            // Sort results: successful first (by F1 descending), then skipped, then failed
            results.sort_by(|a, b| match (a.success, b.success) {
                (true, true) => {
                    let a_f1 = a.primary_f1().unwrap_or(0.0);
                    let b_f1 = b.primary_f1().unwrap_or(0.0);
                    b_f1.partial_cmp(&a_f1).unwrap_or(std::cmp::Ordering::Equal)
                }
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                (false, false) => match (a.is_skipped(), b.is_skipped()) {
                    (true, false) => std::cmp::Ordering::Less,
                    (false, true) => std::cmp::Ordering::Greater,
                    _ => std::cmp::Ordering::Equal,
                },
            });

            // Compact table headers
            let show_metrics = match task {
                Task::NER | Task::DiscontinuousNER => {
                    md.push_str("| Dataset | Backend | F1 | P | R | N | ms |\n");
                    md.push_str("|---------|---------|----|----|----|---|----|\n");
                    true
                }
                Task::IntraDocCoref | Task::InterDocCoref | Task::AbstractAnaphora => {
                    md.push_str("| Dataset | Backend | CoNLL | MUC | B³ | N | ms |\n");
                    md.push_str("|---------|---------|-------|-----|----|---|----|\n");
                    true
                }
                Task::RelationExtraction => {
                    md.push_str(
                        "| Dataset | Backend | Strict | Boundary | Gold-oracle docs | N | ms |\n",
                    );
                    md.push_str(
                        "|---------|---------|--------|----------|-------------|---|----|\n",
                    );
                    true
                }
                _ => {
                    md.push_str("| Dataset | Backend | N | ms |\n");
                    md.push_str("|---------|---------|---|----|\n");
                    false
                }
            };

            for result in results {
                let time_str = result
                    .duration_ms
                    .map(|d| format!("{:.0}", d))
                    .unwrap_or_else(|| "-".to_string());

                if show_metrics && result.success {
                    match task {
                        Task::NER | Task::DiscontinuousNER => {
                            let f1 = result.metrics.get("f1").map(|v| *v * 100.0).unwrap_or(0.0);
                            let p = result
                                .metrics
                                .get("precision")
                                .map(|v| *v * 100.0)
                                .unwrap_or(0.0);
                            let r = result
                                .metrics
                                .get("recall")
                                .map(|v| *v * 100.0)
                                .unwrap_or(0.0);

                            // Add familiarity note for zero-shot backends
                            let mut note_parts = Vec::new();
                            if let Some(ref label_shift) = result.label_shift {
                                if label_shift.is_inflated() {
                                    note_parts.push(format!(
                                        "⚠ familiarity={:.0}%",
                                        label_shift.familiarity * 100.0
                                    ));
                                }
                            }

                            // Add note for 0.0 F1 scores
                            let note = if f1 < 0.1 && result.num_examples > 0 {
                                // Check if it's an incompatible entity type issue
                                let dataset_entity_types = result.dataset.entity_types();
                                let backend_name = &result.backend;
                                if backend_name == "stacked"
                                    || backend_name == "heuristic"
                                    || backend_name == "pattern"
                                {
                                    // Stacked/heuristic/pattern have limited entity type support
                                    let normalized_types: Vec<String> = dataset_entity_types
                                        .iter()
                                        .map(|t| t.to_lowercase())
                                        .collect();
                                    let supports_structured = normalized_types.iter().any(|t| {
                                        t.contains("date")
                                            || t.contains("time")
                                            || t.contains("money")
                                            || t.contains("percent")
                                            || t.contains("email")
                                            || t.contains("url")
                                            || t.contains("phone")
                                    });
                                    let supports_named = normalized_types.iter().any(|t| {
                                        t.contains("person")
                                            || t.contains("organization")
                                            || t.contains("location")
                                    });
                                    let supports_biomedical = normalized_types.iter().any(|t| {
                                        t.contains("disease")
                                            || t.contains("chemical")
                                            || t.contains("gene")
                                            || t.contains("protein")
                                            || t.contains("anatomy")
                                    });

                                    if backend_name == "pattern" && !supports_structured {
                                        " (pattern: no structured entities)"
                                    } else if backend_name == "heuristic" && !supports_named {
                                        " (heuristic: no PER/ORG/LOC)"
                                    } else if backend_name == "stacked"
                                        && !supports_structured
                                        && !supports_named
                                    {
                                        if supports_biomedical {
                                            " (stacked: biomedical not supported)"
                                        } else {
                                            " (stacked: incompatible types)"
                                        }
                                    } else {
                                        ""
                                    }
                                } else if result.num_examples == 0 {
                                    " (N=0: no data)"
                                } else {
                                    ""
                                }
                            } else {
                                ""
                            };

                            md.push_str(&format!(
                                "| {:?} | {} | {:.1} | {:.1} | {:.1} | {} | {} |{}\n",
                                result.dataset,
                                result.backend,
                                f1,
                                p,
                                r,
                                result.num_examples,
                                time_str,
                                note
                            ));

                            // Add stratified metrics section if available
                            if let Some(ref stratified) = result.stratified {
                                if !stratified.by_entity_type.is_empty() {
                                    md.push_str("\n#### Stratified by Entity Type\n\n");
                                    md.push_str("| Type | F1 | CI 95% | N |\n");
                                    md.push_str("|------|----|--------|---|\n");
                                    let mut types: Vec<_> =
                                        stratified.by_entity_type.iter().collect();
                                    types.sort_by_key(|(k, _)| *k);
                                    for (type_str, metric_ci) in types {
                                        let ci_str = format!(
                                            "[{:.2}, {:.2}]",
                                            metric_ci.ci_95.0, metric_ci.ci_95.1
                                        );
                                        md.push_str(&format!(
                                            "| {} | {:.2} | {} | {} |\n",
                                            type_str, metric_ci.mean, ci_str, metric_ci.n
                                        ));
                                    }
                                    md.push('\n');
                                }
                            }

                            // Add temporal stratification if available
                            if let Some(ref stratified) = result.stratified {
                                if let Some(ref temporal) = stratified.by_temporal_stratum {
                                    if !temporal.is_empty() {
                                        md.push_str("\n#### Temporal Stratification\n\n");
                                        md.push_str("| Stratum | F1 | CI 95% | N |\n");
                                        md.push_str("|---------|----|--------|---|\n");
                                        for (stratum, metric) in temporal {
                                            md.push_str(&format!(
                                                "| {} | {:.2} | [{:.2}, {:.2}] | {} |\n",
                                                stratum,
                                                metric.mean,
                                                metric.ci_95.0,
                                                metric.ci_95.1,
                                                metric.n
                                            ));
                                        }
                                        md.push('\n');
                                    }
                                }
                            }

                            // Add confidence intervals if available
                            if let Some(ref ci) = result.confidence_intervals {
                                md.push_str(&format!(
                                    "\n**Confidence Intervals (95%)**: F1: [{:.2}, {:.2}], P: [{:.2}, {:.2}], R: [{:.2}, {:.2}]\n\n",
                                    ci.f1_ci.0, ci.f1_ci.1,
                                    ci.precision_ci.0, ci.precision_ci.1,
                                    ci.recall_ci.0, ci.recall_ci.1
                                ));
                            }
                        }
                        Task::IntraDocCoref | Task::InterDocCoref | Task::AbstractAnaphora => {
                            let conll = result
                                .metrics
                                .get("conll_f1")
                                .map(|v| *v * 100.0)
                                .unwrap_or(0.0);
                            let muc = result
                                .metrics
                                .get("muc_f1")
                                .map(|v| *v * 100.0)
                                .unwrap_or(0.0);
                            let b3 = result
                                .metrics
                                .get("b3_f1")
                                .map(|v| *v * 100.0)
                                .unwrap_or(0.0);

                            // Add note for 0.0 scores with low N
                            let note = if conll < 0.1 && result.num_examples <= 1 {
                                " (N≤1: insufficient data or parsing issue)"
                            } else {
                                ""
                            };

                            md.push_str(&format!(
                                "| {:?} | {} | {:.1} | {:.1} | {:.1} | {} | {} |{}\n",
                                result.dataset,
                                result.backend,
                                conll,
                                muc,
                                b3,
                                result.num_examples,
                                time_str,
                                note
                            ));

                            // Add chain-length stratification if available in metrics
                            if let Some(long_f1) = result.metrics.get("chain_long_f1") {
                                md.push_str("\n#### Chain-Length Stratification\n\n");
                                md.push_str("| Chain Type | Count | F1 |\n");
                                md.push_str("|------------|-------|----|\n");
                                if let Some(long_count) = result.metrics.get("chain_long_count") {
                                    md.push_str(&format!(
                                        "| Long (>10) | {:.0} | {:.2} |\n",
                                        long_count,
                                        long_f1 * 100.0
                                    ));
                                }
                                if let Some(short_f1) = result.metrics.get("chain_short_f1") {
                                    if let Some(short_count) =
                                        result.metrics.get("chain_short_count")
                                    {
                                        md.push_str(&format!(
                                            "| Short (2-10) | {:.0} | {:.2} |\n",
                                            short_count,
                                            short_f1 * 100.0
                                        ));
                                    }
                                }
                                if let Some(singleton_f1) = result.metrics.get("chain_singleton_f1")
                                {
                                    if let Some(singleton_count) =
                                        result.metrics.get("chain_singleton_count")
                                    {
                                        md.push_str(&format!(
                                            "| Singleton (1) | {:.0} | {:.2} |\n",
                                            singleton_count,
                                            singleton_f1 * 100.0
                                        ));
                                    }
                                }
                                md.push('\n');
                            }
                        }
                        Task::RelationExtraction => {
                            let strict = result
                                .metrics
                                .get("strict_f1")
                                .map(|v| *v * 100.0)
                                .unwrap_or(0.0);
                            let boundary = result
                                .metrics
                                .get("boundary_f1")
                                .map(|v| *v * 100.0)
                                .unwrap_or(0.0);
                            let oracle_docs = result
                                .metrics
                                .get("oracle_docs_used")
                                .copied()
                                .unwrap_or(0.0);
                            md.push_str(&format!(
                                "| {:?} | {} | {:.1} | {:.1} | {:.0} | {} | {} |\n",
                                result.dataset,
                                result.backend,
                                strict,
                                boundary,
                                oracle_docs,
                                result.num_examples,
                                time_str
                            ));
                        }
                        _ => {
                            md.push_str(&format!(
                                "| {:?} | {} | {} | {} |\n",
                                result.dataset, result.backend, result.num_examples, time_str
                            ));
                        }
                    }
                } else {
                    // Failed or skipped - show error
                    let status = if result.is_skipped() { "⊘" } else { "✗" };
                    let error_msg = if result.is_skipped() {
                        "no-feature".to_string()
                    } else {
                        result
                            .error
                            .as_ref()
                            .map(|e| {
                                // Categorize errors for better debugging
                                if e.starts_with("incompatible:") {
                                    "incompatible".to_string()
                                } else if e.contains("Unknown backend")
                                    || e.contains("unknown backend")
                                {
                                    "unknown-backend".to_string()
                                } else if e.contains("Failed to load")
                                    || e.contains("422")
                                    || e.contains("HuggingFace")
                                    || e.contains("API")
                                {
                                    "load-failed".to_string()
                                } else if e.contains("empty") || e.contains("no sentences") {
                                    "empty-dataset".to_string()
                                } else if e.contains("ONNX") || e.contains("onnx") {
                                    "onnx-error".to_string()
                                } else if e.contains("model")
                                    && (e.contains("not found") || e.contains("download"))
                                {
                                    "model-load-failed".to_string()
                                } else if e.contains("timeout") || e.contains("timed out") {
                                    "timeout".to_string()
                                } else if e.contains("not available")
                                    || e.contains("FeatureNotAvailable")
                                {
                                    "not-available".to_string()
                                } else if e.len() > 30 {
                                    e.chars().take(30).collect::<String>() + "..."
                                } else {
                                    e.clone()
                                }
                            })
                            .unwrap_or_else(|| "error".to_string())
                    };
                    md.push_str(&format!(
                        "| {:?} | {} | {} | {} | {} |\n",
                        result.dataset, result.backend, status, error_msg, time_str
                    ));
                }
            }
            md.push('\n');
        }

        // Backend summary (compact)
        let mut backend_stats: HashMap<String, (usize, usize, usize, f64)> = HashMap::new();
        for result in &self.results {
            let entry = backend_stats
                .entry(result.backend.clone())
                .or_insert((0, 0, 0, 0.0));
            if result.success {
                entry.0 += 1;
                if let Some(f1) = result.primary_f1() {
                    entry.3 += f1;
                }
            } else if result.is_skipped() {
                entry.1 += 1;
            } else {
                entry.2 += 1;
            }
        }

        if !backend_stats.is_empty() {
            md.push_str("## Backend Summary\n\n");
            md.push_str("| Backend | ✓ | ⊘ | ✗ | Avg F1 |\n");
            md.push_str("|---------|---|---|---|--------|\n");
            let mut backends: Vec<_> = backend_stats.iter().collect();
            backends.sort_by_key(|(_, (success, _, _, _))| *success);
            backends.reverse();
            for (backend, (success, skipped, failed, total_f1)) in backends {
                let avg_f1 = if *success > 0 {
                    total_f1 / *success as f64 * 100.0
                } else {
                    0.0
                };
                md.push_str(&format!(
                    "| {} | {} | {} | {} | {:.1} |\n",
                    backend, success, skipped, failed, avg_f1
                ));
            }
            md.push('\n');
        }

        md
    }
}

// =============================================================================
// Helper Functions for Advanced Evaluation Features
// =============================================================================

impl TaskEvaluator {
    /// Extract KB version from dataset metadata if available.
    ///
    /// Returns KB version string if temporal metadata contains it.
    fn extract_kb_version(dataset_data: &super::loader::LoadedDataset) -> Option<String> {
        dataset_data.temporal_metadata.as_ref()?.kb_version.clone()
    }

    /// Compute familiarity for zero-shot backends.
    ///
    /// Returns None if backend is not zero-shot or if familiarity cannot be computed.
    fn compute_familiarity_if_zero_shot(
        &self,
        backend_name: &str,
        dataset_data: &LoadedDataset,
    ) -> Option<super::types::LabelShift> {
        // Check if this is a zero-shot backend
        let is_zero_shot = matches!(
            backend_name.to_lowercase().as_str(),
            "nuner"
                | "gliner_onnx"
                | "gliner_candle"
                | "gliner_multitask"
                | "gliner2_fastino"
                | "gliner_poly"
                | "universal_ner"
        );

        if !is_zero_shot {
            return None;
        }

        // Extract dataset entity types
        let eval_types: Vec<String> = dataset_data
            .sentences
            .iter()
            .flat_map(|s| s.entities())
            .map(|e| e.entity_type.as_label().to_string())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        // For zero-shot backends, we don't have training types, so we use a heuristic:
        // Common entity types that zero-shot models are typically trained on
        let common_train_types = vec![
            "person".to_string(),
            "organization".to_string(),
            "location".to_string(),
            "PER".to_string(),
            "ORG".to_string(),
            "LOC".to_string(),
            "PERSON".to_string(),
            "ORGANIZATION".to_string(),
        ];

        Some(super::types::LabelShift::from_type_sets(
            &common_train_types,
            &eval_types,
        ))
    }

    /// Compute robustness testing results.
    ///
    /// # Performance Note
    ///
    /// This function creates a new backend instance and runs robustness tests on up to
    /// `ROBUSTNESS_TEST_LIMIT` examples. This is intentional - robustness testing requires
    /// running perturbations that may affect backend state.
    ///
    /// # Limitations
    ///
    /// - Limited to `ROBUSTNESS_TEST_LIMIT` examples for performance
    /// - Creates a new backend instance (doesn't reuse from main evaluation)
    #[cfg(feature = "eval")]
    pub fn compute_robustness(
        &self,
        backend_name: &str,
        dataset_data: &LoadedDataset,
        config: &TaskEvalConfig,
    ) -> Option<super::robustness::RobustnessResults> {
        use super::robustness::RobustnessEvaluator;
        use anno::Entity;

        // Create backend for robustness testing
        // NOTE: We create a new backend instance here rather than reusing from main evaluation
        // because robustness testing may modify backend state through perturbations
        let backend = match BackendFactory::create(backend_name) {
            Ok(b) => b,
            Err(_) => return None,
        };

        if !backend.is_available() {
            return None;
        }

        // Prepare test cases (limit to ROBUSTNESS_TEST_LIMIT for performance)
        let test_cases: Vec<(String, Vec<Entity>)> = dataset_data
            .sentences
            .iter()
            .take(ROBUSTNESS_TEST_LIMIT)
            .map(|s| {
                let gold: Vec<Entity> = s
                    .entities()
                    .iter()
                    .map(|g| {
                        let mut entity =
                            Entity::new(g.text.clone(), g.entity_type.clone(), g.start, g.end, 1.0);
                        entity.provenance = Some(crate::Provenance::ml("gold", 1.0));
                        entity
                    })
                    .collect();
                (s.text().to_string(), gold)
            })
            .collect();

        if test_cases.is_empty() {
            return None;
        }

        // Create robustness evaluator
        let evaluator = RobustnessEvaluator {
            seed: config.seed.unwrap_or(42),
            ..Default::default()
        };

        // Run robustness evaluation
        Some(evaluator.evaluate(backend.as_ref(), &test_cases))
    }

    /// Compute stratified metrics from per-example scores.
    ///
    /// Uses actual per-example F1/precision/recall to compute per-type metrics.
    /// This is the primary method when per-example scores are available.
    fn compute_stratified_metrics_from_scores(
        &self,
        dataset_data: &LoadedDataset,
        aggregate_metrics: &HashMap<String, f64>,
        per_example_scores: Option<&PerExampleScores>,
    ) -> Option<StratifiedMetrics> {
        use crate::eval::ner_metrics::evaluate_entities;

        // If we have per-example scores, use them for proper stratification
        if let Some(per_example) = per_example_scores {
            // Compute per-type metrics from per-example scores
            let mut by_type_scores: HashMap<String, Vec<(f64, f64, f64)>> = HashMap::new(); // (f1, precision, recall)

            for (gold, predicted, _text) in per_example {
                // Group by entity type and compute per-type metrics
                let mut type_groups: HashMap<String, (Vec<Entity>, Vec<Entity>)> = HashMap::new();

                // Group gold entities by type
                for entity in gold {
                    let type_str = canonical_entity_type_label(&entity.entity_type);
                    type_groups
                        .entry(type_str.clone())
                        .or_default()
                        .0
                        .push(entity.clone());
                }

                // Group predicted entities by type
                for entity in predicted {
                    let type_str = canonical_entity_type_label(&entity.entity_type);
                    type_groups
                        .entry(type_str)
                        .or_default()
                        .1
                        .push(entity.clone());
                }

                // Compute per-type metrics
                for (type_str, (type_gold, type_predicted)) in type_groups {
                    let result = evaluate_entities(&type_gold, &type_predicted);
                    let summary = result.summary();
                    by_type_scores.entry(type_str).or_default().push((
                        summary.strict_f1,
                        summary.strict_precision,
                        summary.strict_recall,
                    ));
                }
            }

            // Compute mean and CI for each type
            let mut by_entity_type = HashMap::new();
            for (type_str, scores) in by_type_scores {
                if scores.is_empty() {
                    continue;
                }

                let n = scores.len() as f64;
                let f1_mean = scores.iter().map(|(f1, _, _)| f1).sum::<f64>() / n;
                // Note: precision_mean and recall_mean computed but not used in CI (using F1 only for now)
                let _precision_mean = scores.iter().map(|(_, p, _)| p).sum::<f64>() / n;
                let _recall_mean = scores.iter().map(|(_, _, r)| r).sum::<f64>() / n;

                // Use sample variance (Bessel's correction: n-1) for unbiased estimate
                let f1_variance = if n > 1.0 {
                    scores
                        .iter()
                        .map(|(f1, _, _)| (f1 - f1_mean).powi(2))
                        .sum::<f64>()
                        / (n - 1.0)
                } else {
                    0.0
                };
                let f1_std_dev = f1_variance.sqrt();

                let z = DEFAULT_Z_SCORE_95;
                let margin = z * f1_std_dev / n.sqrt();

                by_entity_type.insert(
                    type_str,
                    MetricWithCI {
                        mean: f1_mean,
                        std_dev: f1_std_dev,
                        ci_95: (
                            (f1_mean - margin).clamp(0.0, 1.0),
                            (f1_mean + margin).clamp(0.0, 1.0),
                        ),
                        n: scores.len(),
                    },
                );
            }

            // This loader carries only dataset-level temporal metadata, not example timestamps.
            let by_temporal_stratum = None;

            return Some(StratifiedMetrics {
                by_entity_type,
                by_temporal_stratum,
                by_surface_form: None, // Would need proper noun detection
                by_mention_char: None, // Would need mention analysis
            });
        }

        // Fallback to simplified version using aggregate metrics
        self.compute_stratified_metrics(dataset_data, aggregate_metrics)
    }

    /// Compute pooled-micro confidence intervals from per-example scores.
    fn compute_confidence_intervals_from_scores(
        &self,
        per_example_scores: &[(Vec<Entity>, Vec<Entity>, String)],
    ) -> Option<ConfidenceIntervals> {
        use crate::eval::ner_metrics::{evaluate_entities, MucCounts};

        if per_example_scores.is_empty() {
            return None;
        }

        // Score each sentence once, then bootstrap by merging the already-scored
        // strict counts. Averaging per-sentence F1 would estimate macro F1, while
        // the scorecard's primary metrics are pooled micro metrics.
        let strict_counts: Vec<MucCounts> = per_example_scores
            .iter()
            .map(|(gold, predicted, _text)| evaluate_entities(gold, predicted).strict)
            .collect();
        let sample_count = strict_counts.len();
        let mut state = CI_BOOTSTRAP_SEED;
        let mut f1_samples = Vec::with_capacity(CI_BOOTSTRAP_REPLICATES);
        let mut precision_samples = Vec::with_capacity(CI_BOOTSTRAP_REPLICATES);
        let mut recall_samples = Vec::with_capacity(CI_BOOTSTRAP_REPLICATES);

        for _ in 0..CI_BOOTSTRAP_REPLICATES {
            let mut pooled = MucCounts::default();
            for _ in 0..sample_count {
                pooled.merge(&strict_counts[bootstrap_index(&mut state, sample_count)]);
            }
            f1_samples.push(pooled.f1_exact());
            precision_samples.push(pooled.precision_exact());
            recall_samples.push(pooled.recall_exact());
        }

        Some(ConfidenceIntervals {
            f1_ci: bootstrap_ci_95(&mut f1_samples),
            precision_ci: bootstrap_ci_95(&mut precision_samples),
            recall_ci: bootstrap_ci_95(&mut recall_samples),
        })
    }

    /// Aggregate scores cannot establish per-type performance or uncertainty.
    ///
    /// Returns `None`; run the evaluation pipeline with per-example observations
    /// to obtain measured stratification. The signature remains available for callers
    /// which previously requested an aggregate-only estimate.
    pub fn compute_stratified_metrics(
        &self,
        _dataset_data: &LoadedDataset,
        _metrics: &HashMap<String, f64>,
    ) -> Option<StratifiedMetrics> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::loader::DatasetId;

    #[cfg(feature = "eval-parallel")]
    #[test]
    fn parallel_receipt_observation_keeps_warm_cache_identity() {
        use std::cell::RefCell;
        use std::sync::Arc;

        thread_local! {
            static WARM_RECEIPT: RefCell<Option<BackendConstructionReceipt>> = const { RefCell::new(None) };
        }

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("one-thread Rayon pool");
        let observed = Arc::new(Mutex::new(Vec::new()));

        for _ in 0..2 {
            let observed = Arc::clone(&observed);
            pool.install(|| {
                WARM_RECEIPT.with(|cached| {
                    let mut cached = cached.borrow_mut();
                    let receipt = cached.get_or_insert_with(BackendConstructionReceipt::default);
                    observe_construction_receipt(&observed, receipt);
                });
            });
        }

        let observed = lock(&observed);
        assert_eq!(observed.len(), 2);
        assert_eq!(observed[0], observed[1]);
    }

    #[test]
    fn test_task_mapping_build() {
        let mapping = TaskMapping::build();
        assert!(!mapping.task_to_datasets.is_empty());
        assert!(!mapping.dataset_to_tasks.is_empty());
        assert!(!mapping.backend_to_tasks.is_empty());
        assert!(!mapping.task_to_backends.is_empty());
    }

    #[test]
    fn cached_only_uses_the_provided_empty_cache_without_downloading() {
        let cache = tempfile::tempdir().expect("temporary cache directory");
        let evaluator = TaskEvaluator::with_cache_dir(cache.path()).expect("TaskEvaluator");
        assert_eq!(evaluator.loader.cache_dir(), cache.path());

        let results = evaluator
            .evaluate_all(TaskEvalConfig {
                tasks: vec![Task::NER],
                datasets: vec![DatasetId::WikiGold],
                backends: vec!["stacked".to_string()],
                require_cached: true,
                ..Default::default()
            })
            .expect("cached-only evaluation records an unavailable dataset");

        assert_eq!(results.results.len(), 1);
        let result = &results.results[0];
        assert!(!result.success);
        assert!(
            result
                .error
                .as_deref()
                .is_some_and(|error| error.contains("not cached")),
            "cached-only evaluation should fail from the isolated cache before model or network fallback: {result:?}"
        );
    }

    #[test]
    fn test_type_mapping_domain_specific() {
        // Test domain-specific type mappings (MIT Movie, MIT Restaurant, etc.)
        use super::TaskEvaluator;

        // MIT Movie types should map Actor/Director → person
        let mit_movie_types = vec!["Actor", "Director", "Character"];
        let mapped = TaskEvaluator::map_dataset_labels_to_model(&mit_movie_types, "stacked");
        assert!(
            mapped.iter().any(|t| t == "person"),
            "MIT Movie Actor/Director should map to person"
        );

        // MIT Restaurant types should map Restaurant_Name → organization
        let mit_restaurant_types = vec!["Restaurant_Name", "Cuisine", "Dish"];
        let mapped = TaskEvaluator::map_dataset_labels_to_model(&mit_restaurant_types, "stacked");
        assert!(
            mapped.iter().any(|t| t == "organization"),
            "MIT Restaurant Restaurant_Name should map to organization"
        );

        // Biomedical types should map Disease → disease
        let bio_types = vec!["Disease", "Chemical", "Disorder"];
        let mapped = TaskEvaluator::map_dataset_labels_to_model(&bio_types, "stacked");
        assert!(
            mapped.iter().any(|t| t == "disease"),
            "Biomedical Disease should map to disease"
        );
        assert!(
            mapped.iter().any(|t| t == "chemical"),
            "Biomedical Chemical should map to chemical"
        );
    }

    #[test]
    fn test_classical_backend_dataset_compatibility_gate() {
        // CRF/HMM in this repo are CoNLL-style: they should be compatible with PER/LOC/ORG/MISC
        // datasets, but excluded from datasets with different type inventories (e.g. WNUT-17).
        assert!(TaskEvaluator::is_backend_compatible(
            "crf",
            DatasetId::CoNLL2003Sample
        ));
        assert!(TaskEvaluator::is_backend_compatible(
            "hmm",
            DatasetId::CoNLL2003Sample
        ));

        assert!(!TaskEvaluator::is_backend_compatible(
            "crf",
            DatasetId::Wnut17
        ));
        assert!(!TaskEvaluator::is_backend_compatible(
            "hmm",
            DatasetId::Wnut17
        ));
    }

    #[test]
    fn test_gliner_multitask_capabilities() {
        let tasks = crate::eval::task_mapping::backend_tasks("gliner_multitask");
        assert!(tasks.contains(&Task::NER));
        assert!(tasks.contains(&Task::RelationExtraction));
        assert!(tasks.contains(&Task::TextClassification));
    }

    #[test]
    fn ner_metrics_do_not_match_across_sentences_and_fail_on_inference_errors() {
        use crate::eval::loader::{
            AnnotatedSentence, AnnotatedToken, DataSource, DatasetMetadata, LoadedDataset,
        };
        let ds = LoadedDataset {
            id: DatasetId::CoNLL2003Sample,
            sentences: [("Anna", "B-PER"), ("rain", "O")]
                .into_iter()
                .map(|(text, tag)| AnnotatedSentence {
                    tokens: vec![AnnotatedToken {
                        text: text.into(),
                        ner_tag: tag.into(),
                    }],
                    source_dataset: DatasetId::CoNLL2003Sample,
                })
                .collect(),
            loaded_at: "test".into(),
            source_url: "fixture".into(),
            data_source: DataSource::Embedded,
            temporal_metadata: None,
            metadata: DatasetMetadata::default(),
        };
        let mut corpus = ds.clone();
        corpus.sentences = (0..10)
            .map(|i| ds.sentences[usize::from(i != 0)].clone())
            .collect();
        let mut config = TaskEvalConfig {
            max_examples: Some(4),
            ..Default::default()
        };
        let (sample, count) = TaskEvaluator::sample_dataset_for_task(Task::NER, &corpus, &config);
        assert_eq!(count, 4);
        assert_eq!(sample.sentences.len(), 4);
        assert_eq!(
            sample
                .sentences
                .iter()
                .filter(|sentence| sentence.entities().is_empty())
                .count(),
            3
        );
        config.max_examples = Some(0);
        assert!(
            TaskEvaluator::sample_dataset_for_task(Task::NER, &corpus, &config)
                .0
                .sentences
                .is_empty()
        );

        let model = anno::AnyModel::new(
            "wrong-sentence",
            "false positive on negative sentence",
            vec![anno::EntityType::Person],
            |text, _| {
                Ok(if text == "rain" {
                    vec![anno::Entity::new(text, anno::EntityType::Person, 0, 4, 1.0)]
                } else {
                    vec![]
                })
            },
        );
        let eval = TaskEvaluator::new().unwrap();
        let metrics = eval
            .evaluate_ner_task(
                "wrong-sentence",
                Some(&model),
                ds.id,
                &ds,
                &TaskEvalConfig::default(),
            )
            .unwrap();
        assert_eq!(metrics.metrics["f1"], 0.0);
        assert_eq!(metrics.metrics["precision"], 0.0);
        assert_eq!(metrics.metrics["recall"], 0.0);
        let broken = anno::AnyModel::new("broken", "inference failure", vec![], |_, _| {
            Err(anno::Error::Inference("fixture failure".into()))
        });
        let error = eval
            .evaluate_ner_task(
                "broken",
                Some(&broken),
                ds.id,
                &ds,
                &TaskEvalConfig::default(),
            )
            .unwrap_err();
        assert!(error.to_string().contains("sentence 1"));
        assert!(error.to_string().contains("fixture failure"));
    }

    #[test]
    fn test_event_extraction_can_be_scored_like_ner() {
        use crate::eval::loader::{
            AnnotatedSentence, AnnotatedToken, DataSource, DatasetMetadata, LoadedDataset,
        };
        use anno::{AnyModel, Entity, EntityType};

        // One example with an "event type" label encoded as BIO, like the loader parsers do.
        let ds = LoadedDataset {
            id: DatasetId::MAVEN,
            sentences: vec![AnnotatedSentence {
                tokens: vec![AnnotatedToken {
                    text: "boom".to_string(),
                    ner_tag: "B-EventType".to_string(),
                }],
                source_dataset: DatasetId::MAVEN,
            }],
            loaded_at: "now".to_string(),
            source_url: "test".to_string(),
            data_source: DataSource::Embedded,
            temporal_metadata: None,
            metadata: DatasetMetadata::default(),
        };

        // A trivial backend that predicts exactly that span/type.
        let ty = EntityType::from_label("EventType");
        let m = AnyModel::new(
            "event-dummy",
            "dummy event trigger extractor",
            vec![ty.clone()],
            move |_text, _lang| Ok(vec![Entity::new("boom", ty.clone(), 0, 4, 1.0)]),
        );

        let eval = TaskEvaluator::new().expect("TaskEvaluator::new");
        let metrics = eval
            .evaluate_ner_task(
                "event-dummy",
                Some(&m),
                DatasetId::MAVEN,
                &ds,
                &TaskEvalConfig::default(),
            )
            .expect("evaluate_ner_task");

        assert!(metrics.metrics.get("f1").copied().unwrap_or(0.0) >= 0.99);
    }

    #[test]
    fn ner_receipt_keeps_all_output_score_and_names_closed_label_diagnostic() {
        use crate::eval::loader::{
            AnnotatedSentence, AnnotatedToken, DataSource, DatasetMetadata, LoadedDataset,
        };
        use anno::{AnyModel, Entity, EntityType};

        let dataset = LoadedDataset {
            id: DatasetId::CoNLL2003Sample,
            sentences: vec![AnnotatedSentence {
                tokens: vec![
                    AnnotatedToken {
                        text: "Alice".into(),
                        ner_tag: "B-PER".into(),
                    },
                    AnnotatedToken {
                        text: "2024".into(),
                        ner_tag: "O".into(),
                    },
                ],
                source_dataset: DatasetId::CoNLL2003Sample,
            }],
            loaded_at: "test".to_string(),
            source_url: "fixture".to_string(),
            data_source: DataSource::Embedded,
            temporal_metadata: None,
            metadata: DatasetMetadata::default(),
        };
        let model = AnyModel::new("broader-output", "fixture", vec![], |_, _| {
            Ok(vec![
                Entity::new("Alice", EntityType::Person, 0, 5, 1.0),
                Entity::new("2024", EntityType::Date, 6, 10, 1.0),
            ])
        });

        let receipt = TaskEvaluator::new()
            .unwrap()
            .evaluate_ner_task(
                "broader-output",
                Some(&model),
                DatasetId::CoNLL2003Sample,
                &dataset,
                &TaskEvalConfig::default(),
            )
            .unwrap();

        assert_eq!(receipt.metrics["predictions_outside_dataset_labels"], 1.0);
        assert!(receipt.metrics["f1"] < receipt.metrics["dataset_label_closed_f1"]);
        assert_eq!(
            receipt.label_policy.primary_policy,
            "strict span-and-type micro score over all emitted entities; no label filtering"
        );
    }

    #[test]
    fn provenance_records_evaluator_supplied_zero_shot_threshold_only() {
        assert!(matches!(
            TaskEvaluator::ner_inference_provenance("gliner_onnx").threshold,
            InferenceSettingProvenance::Known { value } if value == serde_json::json!(0.5)
        ));
        assert!(matches!(
            TaskEvaluator::ner_inference_provenance("bert_onnx").threshold,
            InferenceSettingProvenance::Unknown { .. }
        ));
    }

    #[test]
    fn failed_result_does_not_claim_sequential_scheduling() {
        use crate::eval::loader::{DataSource, DatasetMetadata, LoadedDataset};

        let dataset = LoadedDataset {
            id: DatasetId::CoNLL2003Sample,
            sentences: vec![],
            loaded_at: "test".to_string(),
            source_url: "fixture".to_string(),
            data_source: DataSource::Embedded,
            temporal_metadata: None,
            metadata: DatasetMetadata::default(),
        };
        let result = TaskEvaluator::new()
            .expect("TaskEvaluator::new")
            .evaluate_backend_on_loaded(
                Task::NER,
                DatasetId::CoNLL2003Sample,
                "not-a-backend",
                &dataset,
                0,
                &TaskEvalConfig::default(),
            );

        assert!(!result.success);
        assert!(matches!(
            result.provenance.runtime.scheduling,
            EvaluationScheduling::Unknown { .. }
        ));
        assert!(!result.provenance.build.package_version.is_empty());
    }

    #[test]
    fn legacy_result_without_provenance_does_not_claim_current_build() {
        let mut json = serde_json::to_value(make_test_result(true, None, Some(0.5)))
            .expect("serialize current result");
        json.as_object_mut()
            .expect("result JSON object")
            .remove("provenance");

        let legacy: TaskEvalResult =
            serde_json::from_value(json).expect("deserialize legacy result");
        assert_eq!(legacy.provenance.schema_version, 0);
        assert_eq!(legacy.provenance.build.package_version, "unknown");
        assert!(legacy.provenance.build.source_revision.is_none());
        assert!(legacy.provenance.build.enabled_features.is_empty());
        assert!(matches!(
            legacy.provenance.runtime.scheduling,
            EvaluationScheduling::Unknown { .. }
        ));
    }

    #[test]
    fn stratified_metrics_group_normalized_custom_type_labels_together() {
        use crate::eval::loader::{
            AnnotatedSentence, AnnotatedToken, DataSource, DatasetMetadata, LoadedDataset,
        };
        use anno::EntityCategory;

        let scores = vec![(
            vec![Entity::new(
                "Asian",
                EntityType::custom("MISC", EntityCategory::Misc),
                0,
                5,
                1.0,
            )],
            vec![Entity::new(
                "Asian",
                EntityType::custom("misc", EntityCategory::Misc),
                0,
                5,
                1.0,
            )],
            "Asian".to_string(),
        )];
        let dataset = LoadedDataset {
            id: DatasetId::CoNLL2003Sample,
            sentences: vec![AnnotatedSentence {
                tokens: vec![AnnotatedToken {
                    text: "Asian".into(),
                    ner_tag: "B-MISC".into(),
                }],
                source_dataset: DatasetId::CoNLL2003Sample,
            }],
            loaded_at: "test".to_string(),
            source_url: "fixture".to_string(),
            data_source: DataSource::Embedded,
            temporal_metadata: None,
            metadata: DatasetMetadata::default(),
        };

        let stratified = TaskEvaluator::new()
            .unwrap()
            .compute_stratified_metrics_from_scores(&dataset, &HashMap::new(), Some(&scores))
            .unwrap();

        assert_eq!(stratified.by_entity_type.len(), 1);
        assert_eq!(stratified.by_entity_type["MISC"].mean, 1.0);
        // Gold labels and a global score do not establish per-type performance.
        assert!(TaskEvaluator::new()
            .unwrap()
            .compute_stratified_metrics(&dataset, &HashMap::from([("f1".into(), 0.8)]))
            .is_none());
    }

    // =========================================================================
    // MetricWithCI Tests
    // =========================================================================

    #[test]
    fn confidence_intervals_bootstrap_pooled_strict_counts() {
        use anno::EntityType;

        let good_sentence: Vec<_> = (0..100)
            .map(|index| {
                let start = index * 2;
                Entity::new(
                    format!("e{index}"),
                    EntityType::Person,
                    start,
                    start + 1,
                    1.0,
                )
            })
            .collect();
        let mut scores = Vec::new();
        for _ in 0..99 {
            scores.push((good_sentence.clone(), good_sentence.clone(), String::new()));
        }
        scores.push((
            vec![Entity::new("miss", EntityType::Person, 0, 1, 1.0)],
            Vec::new(),
            String::new(),
        ));

        let evaluator = TaskEvaluator::new().unwrap();
        let ci = evaluator
            .compute_confidence_intervals_from_scores(&scores)
            .unwrap();
        let repeated_ci = evaluator
            .compute_confidence_intervals_from_scores(&scores)
            .unwrap();

        // The aggregate has 9,900 correct matches out of 9,901 gold entities,
        // while the arithmetic mean of sentence F1 values is only 0.99. A pooled
        // sentence bootstrap must therefore remain near the aggregate metric.
        assert!(ci.f1_ci.0 > 0.995, "pooled CI was {:?}", ci.f1_ci);
        assert_eq!(ci.f1_ci, repeated_ci.f1_ci);
    }

    #[test]
    fn aggregate_only_metrics_do_not_report_confidence_intervals() {
        assert!(TaskEvaluator::new()
            .unwrap()
            .compute_confidence_intervals_from_scores(&[])
            .is_none());
    }

    #[test]
    fn test_metric_with_ci_structure() {
        let metric = MetricWithCI {
            mean: 0.8,
            std_dev: 0.05,
            ci_95: (0.75, 0.85),
            n: 10,
        };

        assert!((metric.mean - 0.8).abs() < 0.001);
        assert_eq!(metric.n, 10);
        assert!(metric.ci_95.0 < metric.mean);
        assert!(metric.ci_95.1 > metric.mean);
    }

    #[test]
    fn test_metric_with_ci_serialization() {
        let metric = MetricWithCI {
            mean: 0.75,
            std_dev: 0.1,
            ci_95: (0.65, 0.85),
            n: 50,
        };

        // Should serialize/deserialize correctly
        let json = serde_json::to_string(&metric).unwrap();
        let parsed: MetricWithCI = serde_json::from_str(&json).unwrap();

        assert!((parsed.mean - 0.75).abs() < 0.001);
        assert_eq!(parsed.n, 50);
    }

    // =========================================================================
    // StratifiedMetrics Tests
    // =========================================================================

    #[test]
    fn test_stratified_metrics_default() {
        let strat = StratifiedMetrics {
            by_entity_type: HashMap::new(),
            by_temporal_stratum: None,
            by_surface_form: None,
            by_mention_char: None,
        };

        assert!(strat.by_entity_type.is_empty());
        assert!(strat.by_temporal_stratum.is_none());
    }

    #[test]
    fn test_stratified_metrics_with_types() {
        let mut by_type = HashMap::new();
        by_type.insert(
            "person".to_string(),
            MetricWithCI {
                mean: 0.87,
                std_dev: 0.03,
                ci_95: (0.84, 0.90),
                n: 100,
            },
        );
        by_type.insert(
            "location".to_string(),
            MetricWithCI {
                mean: 0.78,
                std_dev: 0.05,
                ci_95: (0.73, 0.83),
                n: 80,
            },
        );

        let strat = StratifiedMetrics {
            by_entity_type: by_type,
            by_temporal_stratum: None,
            by_surface_form: None,
            by_mention_char: None,
        };

        assert_eq!(strat.by_entity_type.len(), 2);
        assert!(strat.by_entity_type.contains_key("person"));
        assert!(strat.by_entity_type.contains_key("location"));
    }

    // =========================================================================
    // TaskEvalResult Tests
    // =========================================================================

    fn make_test_result(success: bool, error: Option<&str>, f1: Option<f64>) -> TaskEvalResult {
        let mut metrics = HashMap::new();
        if let Some(f1_val) = f1 {
            metrics.insert("f1".to_string(), f1_val);
            metrics.insert("precision".to_string(), 0.8);
            metrics.insert("recall".to_string(), 0.75);
        }

        TaskEvalResult {
            task: Task::NER,
            dataset: DatasetId::WikiGold,
            backend: "stacked".to_string(),
            backend_display: Some("stacked(regex+heuristic)".to_string()),
            seed: 42,
            success,
            error: error.map(|s| s.to_string()),
            metrics,
            num_examples: 100,
            duration_ms: Some(500.0),
            label_shift: None,
            robustness: None,
            stratified: None,
            confidence_intervals: None,
            kb_version: None,
            provenance: EvalRunProvenance::default(),
        }
    }

    #[test]
    fn test_task_eval_result_success() {
        let result = make_test_result(true, None, Some(0.85));

        assert!(result.success);
        assert!(result.error.is_none());
        assert!(result.metrics.contains_key("f1"));
        assert!((result.metrics["f1"] - 0.85).abs() < 0.001);
    }

    #[test]
    fn test_task_eval_result_failure() {
        let result = make_test_result(false, Some("Model failed to load"), None);

        assert!(!result.success);
        assert!(result.error.is_some());
        assert_eq!(result.error.as_ref().unwrap(), "Model failed to load");
    }

    #[test]
    fn test_task_eval_result_is_skipped() {
        let skipped = TaskEvalResult {
            task: Task::NER,
            dataset: DatasetId::WikiGold,
            backend: "missing".to_string(),
            backend_display: None,
            seed: 42,
            success: false,
            error: Some("Feature not available".to_string()),
            metrics: HashMap::new(),
            num_examples: 0,
            duration_ms: None,
            label_shift: None,
            robustness: None,
            stratified: None,
            confidence_intervals: None,
            kb_version: None,
            provenance: EvalRunProvenance::default(),
        };

        assert!(skipped.is_skipped());
    }

    #[test]
    fn test_task_eval_result_not_skipped() {
        let not_skipped = TaskEvalResult {
            task: Task::NER,
            dataset: DatasetId::WikiGold,
            backend: "missing".to_string(),
            backend_display: None,
            seed: 42,
            success: false,
            error: Some("Connection timeout".to_string()),
            metrics: HashMap::new(),
            num_examples: 0,
            duration_ms: None,
            label_shift: None,
            robustness: None,
            stratified: None,
            confidence_intervals: None,
            kb_version: None,
            provenance: EvalRunProvenance::default(),
        };

        assert!(!not_skipped.is_skipped());
    }

    #[test]
    fn test_task_eval_result_primary_f1() {
        let result = make_test_result(true, None, Some(0.824));
        assert_eq!(result.primary_f1(), Some(0.824));
    }

    #[test]
    fn test_task_eval_result_primary_f1_missing() {
        let result = make_test_result(false, Some("Error"), None);
        assert_eq!(result.primary_f1(), None);
    }

    // =========================================================================
    // Task Mapping Tests
    // =========================================================================

    #[test]
    fn test_all_tasks_have_datasets() {
        let mapping = TaskMapping::build();

        // Just check that the mapping was built successfully
        assert!(
            !mapping.task_to_datasets.is_empty(),
            "Task mapping should have some tasks"
        );

        // Check that NER task has datasets (core task that should always have datasets)
        let ner_code = Task::NER.code();
        let datasets = mapping.datasets_for_task(ner_code);
        assert!(
            datasets.is_some() && !datasets.unwrap().is_empty(),
            "NER task should have at least one dataset"
        );
    }

    #[test]
    fn test_get_task_datasets_ner() {
        let datasets = get_task_datasets(Task::NER);
        assert!(!datasets.is_empty(), "NER should have datasets");
    }

    #[test]
    fn test_get_task_backends_ner() {
        let backends = get_task_backends(Task::NER);
        assert!(!backends.is_empty(), "NER should have backends");
    }

    #[test]
    fn test_dataset_tasks_wikigold() {
        let tasks = dataset_tasks(DatasetId::WikiGold);
        assert!(
            tasks.contains(&Task::NER),
            "WikiGold should support NER task"
        );
    }

    // =========================================================================
    // Type Mapping Edge Cases
    // =========================================================================

    #[test]
    fn test_type_mapping_preserves_standard_types() {
        let standard_types = vec!["PER", "LOC", "ORG", "MISC"];
        let mapped = TaskEvaluator::map_dataset_labels_to_model(&standard_types, "stacked");

        // Standard types should be recognized
        assert!(
            mapped.iter().any(|t| t == "person" || t == "PER"),
            "PER should map to person or stay as PER"
        );
    }

    #[test]
    fn test_type_mapping_unknown_types() {
        let unknown_types = vec!["UNKNOWN_TYPE_XYZ"];
        let mapped = TaskEvaluator::map_dataset_labels_to_model(&unknown_types, "stacked");

        // Unknown types should be preserved or mapped to misc/other
        assert!(!mapped.is_empty());
    }

    #[test]
    fn test_type_mapping_empty_input() {
        let empty_types: Vec<&str> = vec![];
        let mapped = TaskEvaluator::map_dataset_labels_to_model(&empty_types, "stacked");

        assert!(mapped.is_empty());
    }

    #[test]
    fn test_type_mapping_case_insensitive() {
        // Test that mapping handles case variations
        let types1 = vec!["Person", "PERSON", "person"];
        let mapped1 = TaskEvaluator::map_dataset_labels_to_model(&types1, "stacked");

        // All should map to the same canonical form
        assert!(mapped1.iter().all(|t| t.to_lowercase() == "person"));
    }

    // =========================================================================
    // ComprehensiveEvalResults Tests
    // =========================================================================

    #[test]
    fn test_comprehensive_eval_results_average_f1() {
        let results = [
            make_test_result(true, None, Some(0.8)),
            make_test_result(true, None, Some(0.6)),
        ];

        // Compute average F1
        let avg_f1: f64 = results.iter().filter_map(|r| r.primary_f1()).sum::<f64>()
            / results.iter().filter(|r| r.primary_f1().is_some()).count() as f64;
        assert!((avg_f1 - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_comprehensive_eval_results_mixed_success() {
        let results = [
            make_test_result(true, None, Some(0.824)),
            make_test_result(false, Some("Backend unavailable"), None),
        ];

        let success_count = results.iter().filter(|r| r.success).count();
        let failure_count = results.iter().filter(|r| !r.success).count();

        assert_eq!(success_count, 1);
        assert_eq!(failure_count, 1);
    }

    #[test]
    fn markdown_discloses_operational_and_closed_label_scores_separately() {
        let mut result = make_test_result(true, None, Some(0.5));
        result.provenance.ner_label_policy = Some(NerLabelPolicyReport {
            primary_policy:
                "strict span-and-type micro score over all emitted entities; no label filtering"
                    .to_string(),
            dataset_label_space: vec!["person".to_string()],
            operational_gold_outside_dataset_labels: 0,
            operational_predictions_outside_dataset_labels: 3,
            dataset_label_closed_diagnostic: ClosedLabelDiagnostic {
                strict_precision: 1.0,
                strict_recall: 1.0,
                strict_f1: 1.0,
                gold_entities_scored: 1,
                predicted_entities_scored: 1,
            },
        });
        let report = ComprehensiveEvalResults {
            results: vec![result],
            summary: EvalSummary {
                total_combinations: 1,
                successful: 1,
                failed: 0,
                skipped: 0,
                tasks: vec![Task::NER],
                datasets: vec![DatasetId::WikiGold],
                backends: vec!["stacked".to_string()],
            },
        }
        .to_markdown();

        assert!(report
            .contains("Primary NER P/R/F1 uses strict span-and-type matching over every entity"));
        assert!(report.contains("Closed-label F1 (diagnostic)"));
        assert!(report.contains("| WikiGold | stacked | 3 | 0 | 100.0 |"));
    }

    #[test]
    fn test_eval_summary_structure() {
        let summary = EvalSummary {
            total_combinations: 100,
            successful: 85,
            failed: 10,
            skipped: 5,
            tasks: vec![Task::NER],
            datasets: vec![DatasetId::WikiGold],
            backends: vec!["stacked".to_string()],
        };

        assert_eq!(summary.total_combinations, 100);
        assert_eq!(summary.successful + summary.failed + summary.skipped, 100);
        assert!(!summary.tasks.is_empty());
        assert!(!summary.backends.is_empty());
    }

    #[test]
    fn relation_markdown_discloses_oracle_entity_documents() {
        let mut result = make_test_result(true, None, None);
        result.task = Task::RelationExtraction;
        result.metrics.insert("strict_f1".to_string(), 0.5);
        result.metrics.insert("boundary_f1".to_string(), 0.75);
        result.metrics.insert("oracle_docs_used".to_string(), 2.0);

        let report = ComprehensiveEvalResults {
            results: vec![result],
            summary: EvalSummary {
                total_combinations: 1,
                successful: 1,
                failed: 0,
                skipped: 0,
                tasks: vec![Task::RelationExtraction],
                datasets: vec![DatasetId::WikiGold],
                backends: vec!["stacked".to_string()],
            },
        }
        .to_markdown();

        assert!(report.contains("Gold-oracle docs"));
        assert!(report.contains("| WikiGold | stacked | 50.0 | 75.0 | 2 |"));
    }
}
