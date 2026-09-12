//! Truthful, per-result evaluation receipts.
//!
//! This module keeps benchmark provenance close to the evaluation surface without
//! adding a global telemetry registry. Fields are populated only from the active
//! loader, evaluator, and constructed backend; unavailable facts are explicit.

use crate::eval::backend_factory::{BackendConstructionReceipt, ModelArtifactFileReceipt};
use anno::EntityType;
use serde::{Deserialize, Serialize};

/// Render entity types under the canonical label mapping used by scoring.
pub(crate) fn canonical_entity_type_label(entity_type: &EntityType) -> String {
    EntityType::from_label(entity_type.as_label())
        .as_label()
        .to_string()
}

/// Reproducibility receipt attached to a single task/dataset/backend result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalRunProvenance {
    /// Format version for consumers of the JSON benchmark artifact.
    pub schema_version: u32,
    /// Build identity available without reconstructing source state.
    pub build: EvalBuildProvenance,
    /// Dataset identity as it was loaded for this run.
    pub dataset: DatasetRunProvenance,
    /// Requested and effective backend names plus known configuration.
    pub backend: BackendRunProvenance,
    /// Sampling and evaluator settings that affect this result.
    pub runtime: EvalRuntimeProvenance,
    /// Present for NER-style tasks. The primary metrics always use this policy.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ner_label_policy: Option<NerLabelPolicyReport>,
}

impl Default for EvalRunProvenance {
    fn default() -> Self {
        Self {
            schema_version: 1,
            build: current_build_provenance(),
            dataset: DatasetRunProvenance::default(),
            backend: BackendRunProvenance::default(),
            runtime: EvalRuntimeProvenance::default(),
            ner_label_policy: None,
        }
    }
}

/// Deserialize results written before per-result provenance existed.
///
/// This must never use [`current_build_provenance`]: a missing field describes
/// an older observation, not the executable currently reading it.
pub(crate) fn legacy_eval_run_provenance() -> EvalRunProvenance {
    EvalRunProvenance {
        schema_version: 0,
        build: EvalBuildProvenance {
            package_version: "unknown".to_string(),
            source_revision: None,
            enabled_features: Vec::new(),
        },
        dataset: DatasetRunProvenance::default(),
        backend: BackendRunProvenance::default(),
        runtime: EvalRuntimeProvenance {
            scheduling: EvaluationScheduling::Unknown {
                reason: "historical result did not record scheduling provenance".to_string(),
            },
            ..Default::default()
        },
        ner_label_policy: None,
    }
}

/// Source and feature settings compiled into the evaluator.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EvalBuildProvenance {
    /// `anno-eval` package version compiled into this executable.
    pub package_version: String,
    /// Source revision injected by the build, when the build supplied one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_revision: Option<String>,
    /// Cargo features compiled into this evaluator artifact.
    pub enabled_features: Vec<String>,
}

/// Build information available without shelling out or guessing repository state.
pub(crate) fn current_build_provenance() -> EvalBuildProvenance {
    let mut enabled_features = Vec::new();
    macro_rules! record_feature {
        ($name:literal) => {
            #[cfg(feature = $name)]
            enabled_features.push($name.to_string());
        };
    }
    record_feature!("eval");
    record_feature!("parallel");
    record_feature!("eval-parallel");
    record_feature!("onnx");
    record_feature!("gliner2-fastino");
    record_feature!("candle");
    record_feature!("llm");
    record_feature!("hf-hub");
    record_feature!("discourse");
    record_feature!("eval-profiling");
    record_feature!("eval-bias");
    record_feature!("heuristic-fr");
    record_feature!("bundled-crf-weights");
    record_feature!("bundled-hmm-params");
    EvalBuildProvenance {
        package_version: env!("CARGO_PKG_VERSION").to_string(),
        source_revision: option_env!("ANNO_GIT_COMMIT").map(str::to_string),
        enabled_features,
    }
}

/// Dataset fields observed at load time rather than reconstructed later.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DatasetRunProvenance {
    /// Source URI returned by the loader for this run.
    pub source_url: String,
    /// Actual loader source, such as local cache or original URL.
    pub data_source: String,
    /// Dataset split selected by the loader.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub split: Option<String>,
    /// Usually `sha256:<digest>` when the cache manifest supplied it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    /// Dataset language when declared by its metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Dataset domain when declared by its metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub domain: Option<String>,
    /// Number of sampled sentences in this result.
    pub sentence_count: usize,
    /// Number of gold entities in the sampled sentences.
    pub entity_count: usize,
}

/// Backend fields known to the evaluator without guessing from a cache directory.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BackendRunProvenance {
    /// Backend name supplied by the caller.
    pub requested: String,
    /// Canonical backend name evaluated by the harness.
    pub effective: String,
    /// Backend-reported display name, if available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    /// Label order passed to a label-conditioned backend, if one was constructed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub configured_labels: Option<Vec<String>>,
    /// `unknown` until a backend provides an explicit artifact receipt.
    pub model_artifacts: ArtifactProvenanceStatus,
    /// `unknown` until a backend exposes its selected layers/provider truthfully.
    pub execution: ExecutionProvenanceStatus,
}

/// Explicit absence is preferable to a plausible-looking reconstructed model ID.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ArtifactProvenanceStatus {
    /// The backend retained the assets it actually selected.
    Known {
        /// Backend implementation that selected the assets.
        model_kind: String,
        /// Model identifier passed to the constructed backend.
        model_id: String,
        /// Pinned model revision when the constructor received one.
        #[serde(skip_serializing_if = "Option::is_none")]
        model_revision: Option<String>,
        /// Files selected by the constructed backend.
        files: Vec<ModelArtifactFileReceipt>,
    },
    /// The backend does not expose selected assets.
    Unknown {
        /// Why a trustworthy asset receipt is unavailable.
        reason: String,
    },
}

impl Default for ArtifactProvenanceStatus {
    fn default() -> Self {
        Self::Unknown {
            reason: "backend did not expose artifact paths and hashes".to_string(),
        }
    }
}

/// Explicit absence is preferable to claiming an accelerator or fallback layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ExecutionProvenanceStatus {
    /// The backend does not expose effective provider or layer selection.
    Unknown {
        /// Why a trustworthy execution receipt is unavailable.
        reason: String,
    },
}

impl Default for ExecutionProvenanceStatus {
    fn default() -> Self {
        Self::Unknown {
            reason: "backend did not expose effective provider or layer selection".to_string(),
        }
    }
}

/// How sentence inference was scheduled for a concrete result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "strategy", rename_all = "snake_case")]
pub enum EvaluationScheduling {
    /// Evaluation did not reach sentence inference.
    NotRun {
        /// Why no scheduling strategy was selected.
        reason: String,
    },
    /// Evaluation failed before the evaluator could record an actual strategy.
    Unknown {
        /// Why a completed scheduling receipt is unavailable.
        reason: String,
    },
    /// Sentences were evaluated serially in the evaluator's calling thread.
    Sequential,
    /// Sentences were evaluated by Rayon.
    Rayon {
        /// Worker count reported by the active Rayon pool before dispatch.
        configured_threads: usize,
        /// Distinct Rayon worker threads that ran at least one sentence.
        effective_threads: usize,
    },
}

impl Default for EvaluationScheduling {
    fn default() -> Self {
        Self::NotRun {
            reason: "evaluation did not reach sentence inference".to_string(),
        }
    }
}

/// Evaluator settings that can alter a result or its uncertainty estimate.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EvalRuntimeProvenance {
    /// Seed used for sampling and bootstrap operations.
    pub seed: u64,
    /// Whether the caller requested cached datasets only.
    pub cached_only: bool,
    /// Maximum sampled examples, when the caller limited the dataset.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_examples: Option<usize>,
    /// Whether confidence intervals were requested.
    pub confidence_intervals: bool,
    /// Whether robustness perturbations were requested.
    pub robustness: bool,
    /// Relation confidence threshold in effect for relation tasks.
    pub relation_threshold: f32,
    /// NER inference controls observed by this evaluator.
    pub ner_inference: NerInferenceProvenance,
    /// Observed sentence scheduling and actual worker participation.
    pub scheduling: EvaluationScheduling,
}

/// NER controls recorded only when the evaluator or backend exposes them.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NerInferenceProvenance {
    /// Threshold used to accept NER candidates.
    pub threshold: InferenceSettingProvenance,
    /// Long-input chunking setting used by the backend.
    pub chunking: InferenceSettingProvenance,
    /// Non-maximum-suppression setting used by the backend.
    pub nms: InferenceSettingProvenance,
}

/// A concrete inference control, or an explicit absence of a trustworthy value.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum InferenceSettingProvenance {
    /// The evaluator supplied this setting directly to inference.
    Known {
        /// JSON value that was supplied to inference.
        value: serde_json::Value,
    },
    /// Neither the evaluator nor the backend exposes a selected value.
    Unknown {
        /// Why this setting cannot be reported without guessing.
        reason: String,
    },
}

impl Default for InferenceSettingProvenance {
    fn default() -> Self {
        Self::Unknown {
            reason: "evaluator did not supply this NER setting and backend did not expose it"
                .to_string(),
        }
    }
}

/// The NER scoring contract and a separately named closed-label diagnostic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NerLabelPolicyReport {
    /// The operational score includes every output emitted by the backend.
    pub primary_policy: String,
    /// Canonical labels observed in the gold dataset.
    pub dataset_label_space: Vec<String>,
    /// Gold entities outside the dataset label space under the operational score.
    pub operational_gold_outside_dataset_labels: usize,
    /// Predictions outside the dataset label space under the operational score.
    pub operational_predictions_outside_dataset_labels: usize,
    /// A separate strict scorer run after retaining only the dataset label space.
    pub dataset_label_closed_diagnostic: ClosedLabelDiagnostic,
}

/// Strict NER scores calculated after retaining only dataset-label entities.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClosedLabelDiagnostic {
    /// Strict micro precision after retaining only dataset labels.
    pub strict_precision: f64,
    /// Strict micro recall after retaining only dataset labels.
    pub strict_recall: f64,
    /// Strict micro F1 after retaining only dataset labels.
    pub strict_f1: f64,
    /// Gold entity count retained for this diagnostic.
    pub gold_entities_scored: usize,
    /// Prediction count retained for this diagnostic.
    pub predicted_entities_scored: usize,
}

/// Convert an explicit construction receipt into its JSON report form.
pub(crate) fn artifact_status(receipt: BackendConstructionReceipt) -> ArtifactProvenanceStatus {
    receipt
        .model_artifacts
        .map(|receipt| ArtifactProvenanceStatus::Known {
            model_kind: receipt.model_kind,
            model_id: receipt.model_id,
            model_revision: receipt.model_revision,
            files: receipt.files,
        })
        .unwrap_or_default()
}
