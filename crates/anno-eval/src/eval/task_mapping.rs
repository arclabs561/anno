//! Task-Dataset-Backend Mapping System
//!
//! This module provides a cohesive system for mapping:
//! - Tasks (NER, NED, Coreference, etc.) → Datasets
//! - Datasets → Backends that can evaluate them
//! - Backends → Tasks they support (via trait inspection)
//!
//! # Design Philosophy
//!
//! - **Trait-based capabilities**: Backend capabilities are determined by trait implementations
//! - **Many-to-many relationships**: A dataset can support multiple tasks, a backend can support multiple tasks
//! - **Explicit capabilities**: Each backend declares what tasks it supports via traits
//! - **Dataset metadata**: Each dataset declares what tasks it can evaluate
//! - **Task requirements**: Each task declares what datasets are suitable
//!
//! # Trait-Based Capability Detection
//!
//! Backends are queried for capabilities using trait bounds:
//! - `Model` → NER capability
//! - `ZeroShotNER` → Zero-shot NER capability
//! - `RelationExtractor` → Relation extraction capability
//! - `DiscontinuousNER` → Discontinuous NER capability
//! - `CoreferenceResolver` → Coreference resolution capability

use crate::eval::loader::DatasetId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Re-export traits for capability detection
pub use crate::eval::coref_resolver::CoreferenceResolver as CoreferenceResolverTrait;
pub use anno::backends::inference::{
    DiscontinuousNER as DiscontinuousNERTrait, RelationExtractor as RelationExtractorTrait,
    ZeroShotNER as ZeroShotNERTrait,
};

/// Information extraction and NLP tasks supported by anno.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Task {
    /// Named Entity Recognition: extract entity spans with types
    NER,
    /// Named Entity Disambiguation: link entities to knowledge bases
    NED,
    /// Relation Extraction: extract entity-relation-entity triples
    RelationExtraction,
    /// Intra-document Coreference: resolve mentions within a document
    IntraDocCoref,
    /// Inter-document Coreference: resolve mentions across documents
    InterDocCoref,
    /// Abstract Anaphora: resolve pronouns to events/propositions
    AbstractAnaphora,
    /// Discontinuous NER: extract non-contiguous entity spans
    DiscontinuousNER,
    /// Event Extraction: extract event triggers and arguments
    EventExtraction,
    /// Text Classification: classify entire text or spans
    TextClassification,
    /// Sentiment analysis (a specialization of text classification)
    SentimentAnalysis,
    /// Part-of-speech tagging
    PosTagging,
    /// Machine translation
    MachineTranslation,
    /// Language modeling
    LanguageModeling,
    /// Temporal extraction / temporal information (catalogued from registries).
    ///
    /// Note: this is distinct from `HierarchicalExtraction`. Some registries use "temporal"
    /// to mean time expressions, event ordering, or temporal relations; we keep it separate
    /// so we don't mislabel temporal datasets as hierarchical.
    Temporal,
    /// Hierarchical Structure Extraction: extract nested structures
    HierarchicalExtraction,
    /// Discourse relations (e.g., PDTB-style)
    DiscourseRelations,
    /// Discourse coherence / coherence modeling
    DiscourseCoherence,
    /// Discourse segmentation
    DiscourseSegmentation,
    /// Speech act classification (dialogue acts)
    SpeechActClassification,
}

impl Task {
    /// All supported tasks.
    pub fn all() -> &'static [Task] {
        &[
            Task::NER,
            Task::NED,
            Task::RelationExtraction,
            Task::IntraDocCoref,
            Task::InterDocCoref,
            Task::AbstractAnaphora,
            Task::DiscontinuousNER,
            Task::EventExtraction,
            Task::TextClassification,
            Task::SentimentAnalysis,
            Task::PosTagging,
            Task::MachineTranslation,
            Task::LanguageModeling,
            Task::Temporal,
            Task::HierarchicalExtraction,
            Task::DiscourseRelations,
            Task::DiscourseCoherence,
            Task::DiscourseSegmentation,
            Task::SpeechActClassification,
        ]
    }

    /// Human-readable name for this task.
    pub fn name(&self) -> &'static str {
        match self {
            Task::NER => "Named Entity Recognition",
            Task::NED => "Named Entity Disambiguation",
            Task::RelationExtraction => "Relation Extraction",
            Task::IntraDocCoref => "Intra-document Coreference",
            Task::InterDocCoref => "Inter-document Coreference",
            Task::AbstractAnaphora => "Abstract Anaphora Resolution",
            Task::DiscontinuousNER => "Discontinuous NER",
            Task::EventExtraction => "Event Extraction",
            Task::TextClassification => "Text Classification",
            Task::SentimentAnalysis => "Sentiment Analysis",
            Task::PosTagging => "Part-of-speech Tagging",
            Task::MachineTranslation => "Machine Translation",
            Task::LanguageModeling => "Language Modeling",
            Task::Temporal => "Temporal",
            Task::HierarchicalExtraction => "Hierarchical Structure Extraction",
            Task::DiscourseRelations => "Discourse Relations",
            Task::DiscourseCoherence => "Discourse Coherence",
            Task::DiscourseSegmentation => "Discourse Segmentation",
            Task::SpeechActClassification => "Speech Act Classification",
        }
    }

    /// Short code for this task (for CLI/config).
    pub fn code(&self) -> &'static str {
        match self {
            Task::NER => "ner",
            Task::NED => "ned",
            Task::RelationExtraction => "re",
            Task::IntraDocCoref => "intra-coref",
            Task::InterDocCoref => "inter-coref",
            Task::AbstractAnaphora => "abstract-anaphora",
            Task::DiscontinuousNER => "discontinuous-ner",
            Task::EventExtraction => "events",
            Task::TextClassification => "classification",
            Task::SentimentAnalysis => "sentiment",
            Task::PosTagging => "pos",
            Task::MachineTranslation => "mt",
            Task::LanguageModeling => "lm",
            Task::Temporal => "temporal",
            Task::HierarchicalExtraction => "hierarchical",
            Task::DiscourseRelations => "discourse-relations",
            Task::DiscourseCoherence => "discourse-coherence",
            Task::DiscourseSegmentation => "discourse-segmentation",
            Task::SpeechActClassification => "speech-act-classification",
        }
    }

    /// Parse task from short code string.
    ///
    /// Supports many common aliases used in dataset registry:
    /// - NER: "ner", "sequence_labeling", "nested-ner", "mner", "pii_detection", "slot_filling"
    /// - NED: "ned", "el", "entity_linking", "entity-linking"
    /// - RelationExtraction: "re", "relation_extraction", "relation-extraction"
    /// - IntraDocCoref: "coref", "intra-coref", "intra_coref"
    /// - InterDocCoref: "inter-coref", "inter_coref", "cdcr", "event_coref"
    /// - AbstractAnaphora: "abstract-anaphora", "abstract_anaphora", "bridging", "discourse_deixis"
    /// - DiscontinuousNER: "discontinuous-ner", "discontinuous_ner", "dner"
    /// - EventExtraction: "events", "event_extraction", "event-extraction"
    /// - TextClassification: "classification", "text-classification", "bias_evaluation", "qa", "harmonic_analysis"
    /// - Temporal: "temporal", "timex"
    /// - HierarchicalExtraction: "hierarchical"
    pub fn from_code(code: &str) -> Option<Self> {
        match code.to_lowercase().as_str() {
            // NER family
            "ner" | "sequence_labeling" | "nested-ner" | "mner" | "pii_detection"
            | "slot_filling" => Some(Task::NER),

            // Entity Linking / Disambiguation
            "ned" | "el" | "entity_linking" | "entity-linking" => Some(Task::NED),

            // Relation Extraction
            "re" | "relation" | "relation_extraction" | "relation-extraction" => {
                Some(Task::RelationExtraction)
            }

            // Intra-document coreference
            "coref" | "intra-coref" | "intra_coref" | "intracoref" => Some(Task::IntraDocCoref),

            // Inter-document coreference (CDCR)
            "inter-coref" | "inter_coref" | "intercoref" | "cdcr" | "event_coref" => {
                Some(Task::InterDocCoref)
            }

            // Abstract Anaphora (includes bridging and discourse deixis)
            "abstract-anaphora" | "abstract_anaphora" | "bridging" | "discourse_deixis" => {
                Some(Task::AbstractAnaphora)
            }

            // Discontinuous NER
            "discontinuous-ner" | "discontinuous_ner" | "disc-ner" | "dner" => {
                Some(Task::DiscontinuousNER)
            }

            // Event Extraction
            "events" | "event" | "event_extraction" | "event-extraction" | "ee" => {
                Some(Task::EventExtraction)
            }

            // Text Classification (includes bias evaluation, QA, harmonic analysis)
            "classification"
            | "text_classification"
            | "text-classification"
            | "tc"
            | "bias_evaluation"
            | "qa"
            | "harmonic_analysis" => Some(Task::TextClassification),

            // Sentiment analysis (often listed separately in registries)
            "sentiment" | "sentiment_analysis" | "sentiment-analysis" => {
                Some(Task::SentimentAnalysis)
            }

            // POS tagging
            "pos" | "pos_tagging" | "pos-tagging" => Some(Task::PosTagging),

            // Machine translation
            "mt" | "machine_translation" | "machine-translation" | "translation" => {
                Some(Task::MachineTranslation)
            }

            // Language modeling
            "lm" | "language_modeling" | "language-modeling" => Some(Task::LanguageModeling),

            // Temporal
            "temporal" | "timex" | "time_expressions" | "time-expressions" => Some(Task::Temporal),

            // Hierarchical Extraction
            "hierarchical" | "hierarchical-extraction" | "he" => Some(Task::HierarchicalExtraction),

            // Discourse tasks (catalogued in the registry; evaluation may be stubbed for now)
            "discourse_relations" | "discourse-relations" => Some(Task::DiscourseRelations),
            "discourse_coherence" | "discourse-coherence" => Some(Task::DiscourseCoherence),
            "discourse_segmentation" | "discourse-segmentation" => {
                Some(Task::DiscourseSegmentation)
            }
            "speech_act_classification" | "speech-act-classification" => {
                Some(Task::SpeechActClassification)
            }

            _ => None,
        }
    }

    /// Check if this task is in the NER family (NER, discontinuous NER, NED).
    pub fn is_ner_family(&self) -> bool {
        matches!(self, Task::NER | Task::DiscontinuousNER | Task::NED)
    }

    /// Check if this task is in the coreference family.
    pub fn is_coref_family(&self) -> bool {
        matches!(
            self,
            Task::IntraDocCoref | Task::InterDocCoref | Task::AbstractAnaphora
        )
    }
}

/// Tasks that a dataset supports *for evaluation*.
///
/// Derived from the dataset registry's `DatasetId::tasks()` strings and mapped through
/// `Task::from_code` (which supports many aliases used in the registry).
///
/// We intentionally **do not** default unknown datasets to NER: that hides missing metadata and
/// creates misleading task×dataset×backend matrices.
pub fn dataset_tasks(dataset: DatasetId) -> Vec<Task> {
    // Keep the registry as the single source of truth for the string task codes,
    // but use its typed view to avoid duplicating parsing/alias logic here.
    dataset.tasks_typed()
}

/// Mapping from tasks to suitable datasets.
pub fn task_datasets(task: Task) -> &'static [DatasetId] {
    match task {
        Task::NER => &[
            DatasetId::WikiGold,
            DatasetId::Wnut17,
            DatasetId::MitMovie,
            DatasetId::MitRestaurant,
            DatasetId::CoNLL2003Sample,
            DatasetId::OntoNotesSample,
            DatasetId::MultiNERD,
            DatasetId::BC5CDR,
            DatasetId::NCBIDisease,
            DatasetId::GENIA,
            DatasetId::AnatEM,
            DatasetId::BC2GM,
            DatasetId::BC4CHEMD,
            DatasetId::TweetNER7,
            DatasetId::BroadTwitterCorpus,
            DatasetId::FabNER,
            DatasetId::FewNERD,
            DatasetId::CrossNER,
            DatasetId::UniversalNERBench,
            DatasetId::WikiANN,
            DatasetId::MultiCoNER,
            DatasetId::MultiCoNERv2,
            DatasetId::WikiNeural,
            DatasetId::PolyglotNER,
            DatasetId::UniversalNER,
            DatasetId::UNER,
            DatasetId::MSNER,
            DatasetId::BioMNER,
            DatasetId::LegNER,
            DatasetId::OntoNotes50,
            DatasetId::GermEval2014,
            DatasetId::HAREM,
            DatasetId::SemEval2013Task91,
            DatasetId::MUC6,
            DatasetId::MUC7,
            DatasetId::JNLPBA,
            DatasetId::BC2GMFull,
            DatasetId::CRAFT,
            DatasetId::FinNER,
            DatasetId::LegalNER,
            DatasetId::SciERCNER,
            // Constructed languages (CoNLL-U)
            DatasetId::TaggedPBCEsperanto,
            DatasetId::TaggedPBCKlingon,
            // Note: These variants were referenced but not added to enum
            // Using existing variants: CoNLL2003Sample, Wnut17, BC5CDR, NCBIDisease
        ],
        Task::DiscontinuousNER => &[DatasetId::CADEC, DatasetId::ShARe13, DatasetId::ShARe14],
        Task::RelationExtraction => &[
            DatasetId::DocRED,
            DatasetId::ReTACRED,
            DatasetId::NYTFB,
            DatasetId::WEBNLG,
            DatasetId::GoogleRE,
            DatasetId::BioRED,
            DatasetId::SciER,
            DatasetId::MixRED,
            DatasetId::CovEReD,
        ],
        Task::IntraDocCoref => &[
            DatasetId::GAP,
            DatasetId::PreCo,
            DatasetId::LitBank,
            DatasetId::HumanVoiceAgentInteraction,
        ],
        Task::InterDocCoref => &[DatasetId::ECBPlus, DatasetId::WikiCoref],
        Task::AbstractAnaphora => &[
            DatasetId::GAP,
            DatasetId::PreCo,
            DatasetId::LitBank,
            DatasetId::HumanVoiceAgentInteraction,
        ],
        Task::EventExtraction => &[
            DatasetId::ACE2005,
            DatasetId::MAVEN,
            DatasetId::MAVENArg,
            DatasetId::CASIE,
            DatasetId::RAMS,
        ],
        Task::NED => &[DatasetId::AIDA, DatasetId::TACKBP],
        Task::TextClassification => &[
            DatasetId::MasakhaNEWS,
            DatasetId::AGNews,
            DatasetId::DBPedia14,
            DatasetId::YahooAnswers,
            DatasetId::TREC,
            DatasetId::TweetTopic,
        ],
        // We catalog these tasks/datasets, but we don't currently run end-to-end evaluation
        // pipelines for them.
        Task::SentimentAnalysis => &[],
        Task::PosTagging => &[],
        Task::MachineTranslation => &[],
        Task::LanguageModeling => &[],
        Task::Temporal => &[DatasetId::TimexRecognitionSentenceOriginal],
        Task::HierarchicalExtraction => {
            // GLiNER2 can do hierarchical extraction, but we don't have dedicated datasets yet
            &[]
        }
        Task::DiscourseRelations => &[
            DatasetId::DisrptEngDepScidtbRels,
            DatasetId::DisrptDeuRstPccRels,
        ],
        Task::SpeechActClassification => &[DatasetId::ViraDialogActsLive],
        Task::DiscourseSegmentation => &[
            DatasetId::DisrptEngDepScidtbConlluSeg,
            DatasetId::DisrptDeuRstPccConlluSeg,
        ],
        Task::DiscourseCoherence => &[],
    }
}

/// Detect backend capabilities via trait inspection.
///
/// This function attempts to determine what tasks a backend supports
/// by checking if it implements relevant traits. For runtime detection,
/// use `detect_backend_capabilities` instead.
pub fn backend_tasks(backend_name: &str) -> &'static [Task] {
    match backend_name {
        // Regex-based backends
        "pattern" | "RegexNER" => &[Task::NER], // Only structured entities
        "heuristic" | "HeuristicNER" => &[Task::NER],
        "stacked" | "StackedNER" => &[Task::NER],
        "crf" | "CrfNER" => &[Task::NER],

        // ML-based NER backends (all implement Model)
        "bert_onnx" | "BertNEROnnx" => &[Task::NER],
        "candle_ner" | "CandleNER" => &[Task::NER],
        "nuner" | "NuNER" => &[Task::NER], // Also implements ZeroShotNER
        "deberta_v3" | "DeBERTaV3NER" => &[Task::NER],
        "albert" | "ALBERTNER" => &[Task::NER],

        // Zero-shot NER backends (implement Model + ZeroShotNER)
        "gliner_onnx" | "GLiNEROnnx" => &[Task::NER],
        "gliner_candle" | "GLiNERCandle" => &[Task::NER],
        "gliner_poly" | "GLiNERPoly" => &[Task::NER],
        "universal_ner" | "UniversalNER" => &[Task::NER],

        // Multi-task backends (GLiNER2 implements multiple traits)
        "gliner2" | "GLiNER2" | "GLiNER2Onnx" | "GLiNER2Candle" => &[
            Task::NER,
            Task::TextClassification,
            Task::SpeechActClassification,
            Task::Temporal,
            Task::HierarchicalExtraction,
            Task::DiscourseRelations,
            Task::DiscourseSegmentation,
            // Treat event trigger extraction as “entities” with event-type labels.
            // This makes event datasets runnable even before a dedicated event pipeline exists.
            Task::EventExtraction,
            Task::RelationExtraction, // Via RelationExtractor trait
        ],

        // Discontinuous NER backends (implement DiscontinuousNER trait)
        "w2ner" | "W2NER" => &[Task::NER, Task::DiscontinuousNER],

        // Joint entity-relation backends
        // TPLinker is a relation extraction model. It is not a general NER backend.
        "tplinker" | "TPLinker" => &[Task::RelationExtraction],

        // Coreference backends (implement CoreferenceResolver trait)
        //
        // Note: the same resolver interface is used for both intra-doc and inter-doc eval in
        // `TaskEvaluator` (it is invoked per document, and spans are offset to avoid collisions).
        "coref_resolver" | "CorefResolver" | "SimpleCorefResolver" | "DiscourseAwareResolver" => &[
            Task::IntraDocCoref,
            Task::InterDocCoref,
            Task::AbstractAnaphora,
        ],
        "mention_ranking" | "MentionRankingCoref" => &[Task::IntraDocCoref, Task::InterDocCoref],
        "box" | "box_coref" | "BoxCorefResolver" => &[Task::IntraDocCoref, Task::InterDocCoref],

        _ => &[],
    }
}

/// Runtime capability detection for a backend instance.
///
/// Uses backend name to determine capabilities (fallback when type_id isn't available).
/// For more accurate detection, use `detect_backend_capabilities_by_name` with the backend's name.
///
/// Note: Full runtime detection with trait objects is limited by Rust's type system.
/// A capability registry pattern would be more robust but requires backend registration.
pub fn detect_backend_capabilities(backend: &dyn crate::Model) -> Vec<Task> {
    // Use the backend's name to determine capabilities
    // This is a pragmatic approach given Rust's trait object limitations
    let backend_name = backend.name();
    detect_backend_capabilities_by_name(backend_name)
}

/// Capability detection using backend name (fallback when type_id isn't available).
///
/// This is less accurate than `detect_backend_capabilities` but works with trait objects.
pub fn detect_backend_capabilities_by_name(backend_name: &str) -> Vec<Task> {
    backend_tasks(backend_name).to_vec()
}

/// Get all tasks that a dataset supports.
pub fn get_dataset_tasks(dataset: DatasetId) -> Vec<Task> {
    dataset_tasks(dataset)
}

/// Get all datasets suitable for a task.
pub fn get_task_datasets(task: Task) -> Vec<DatasetId> {
    // Defensive: keep the curated task→datasets list, but only return datasets that
    // *actually* declare support for the task in the registry metadata.
    //
    // This prevents the benchmark/matrix from turning red due to mapping drift
    // (e.g., dataset listed for a task but missing `tasks: [...]` in the registry).
    task_datasets(task)
        .iter()
        .copied()
        .filter(|d| dataset_tasks(*d).contains(&task))
        .collect()
}

/// Get all backends that support a task.
///
/// For benchmarking, only returns "stacked" (which combines pattern+heuristic)
/// and ML backends, since individual pattern/heuristic backends are incomplete.
pub fn get_task_backends(task: Task) -> Vec<&'static str> {
    let mut backends = Vec::new();
    for backend in [
        // Builtins
        "stacked",
        "crf",
        "heuristic",
        // Pattern is intentionally excluded from NER eval by default (structured-only),
        // but it still advertises NER capability via `backend_tasks` and can be enabled by callers.
        // "pattern",
        // ML backends
        "bert_onnx",
        "candle_ner",
        "nuner",
        "gliner_onnx",
        "gliner_candle",
        "gliner2",
        "w2ner",
        // New backends
        "tplinker",
        "gliner_poly",
        "deberta_v3",
        "albert",
        "universal_ner",
        // Special backends
        "coref_resolver",
        "mention_ranking",
        "box",
    ] {
        if backend_tasks(backend).contains(&task) {
            backends.push(backend);
        }
    }
    backends
}

/// Comprehensive task-dataset-backend mapping.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskMapping {
    /// Tasks → Datasets
    pub task_to_datasets: HashMap<String, Vec<String>>,
    /// Datasets → Tasks
    pub dataset_to_tasks: HashMap<String, Vec<String>>,
    /// Backends → Tasks
    pub backend_to_tasks: HashMap<String, Vec<String>>,
    /// Tasks → Backends
    pub task_to_backends: HashMap<String, Vec<String>>,
}

impl TaskMapping {
    /// Build a complete mapping from all available data.
    pub fn build() -> Self {
        let mut task_to_datasets = HashMap::new();
        let mut dataset_to_tasks = HashMap::new();
        let mut backend_to_tasks = HashMap::new();
        let mut task_to_backends = HashMap::new();

        // Build task → datasets
        for task in Task::all() {
            let datasets = get_task_datasets(*task)
                .iter()
                .map(|d| format!("{:?}", d))
                .collect();
            task_to_datasets.insert(task.code().to_string(), datasets);
        }

        // Build dataset → tasks
        for dataset in DatasetId::all() {
            let tasks = get_dataset_tasks(*dataset)
                .iter()
                .map(|t| t.code().to_string())
                .collect();
            dataset_to_tasks.insert(format!("{:?}", dataset), tasks);
        }

        // Build backend → tasks
        for backend in [
            "pattern",
            "heuristic",
            "stacked",
            "hybrid",
            "bert_onnx",
            "candle_ner",
            "nuner",
            "gliner_onnx",
            "gliner_candle",
            "gliner2",
            "w2ner",
            "coref_resolver",
        ] {
            let tasks = backend_tasks(backend)
                .iter()
                .map(|t| t.code().to_string())
                .collect();
            backend_to_tasks.insert(backend.to_string(), tasks);
        }

        // Build task → backends
        for task in Task::all() {
            let backends: Vec<String> = get_task_backends(*task)
                .iter()
                .map(|s| s.to_string())
                .collect();
            task_to_backends.insert(task.code().to_string(), backends);
        }

        Self {
            task_to_datasets,
            dataset_to_tasks,
            backend_to_tasks,
            task_to_backends,
        }
    }

    /// Get datasets for a task.
    pub fn datasets_for_task(&self, task: &str) -> Option<&Vec<String>> {
        self.task_to_datasets.get(task)
    }

    /// Get tasks for a dataset.
    pub fn tasks_for_dataset(&self, dataset: &str) -> Option<&Vec<String>> {
        self.dataset_to_tasks.get(dataset)
    }

    /// Get tasks for a backend.
    pub fn tasks_for_backend(&self, backend: &str) -> Option<&Vec<String>> {
        self.backend_to_tasks.get(backend)
    }

    /// Get backends for a task.
    pub fn backends_for_task(&self, task: &str) -> Option<&Vec<String>> {
        self.task_to_backends.get(task)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_mapping() {
        let mapping = TaskMapping::build();
        assert!(mapping.datasets_for_task("ner").is_some());
        assert!(mapping.tasks_for_dataset("WikiGold").is_some());
        assert!(mapping.tasks_for_backend("gliner2").is_some());
        assert!(mapping.backends_for_task("ner").is_some());
    }

    #[test]
    fn test_gliner2_capabilities() {
        let tasks = backend_tasks("gliner2");
        assert!(tasks.contains(&Task::NER));
        assert!(tasks.contains(&Task::TextClassification));
        assert!(tasks.contains(&Task::HierarchicalExtraction));
        assert!(tasks.contains(&Task::RelationExtraction));
    }

    #[test]
    fn test_dataset_tasks_are_deduced_from_registry() {
        // Basic sanity: we should be able to derive at least one eval task from
        // the registry for a known dataset with explicit tasks.
        let tasks = dataset_tasks(DatasetId::WikiGold);
        assert!(tasks.contains(&Task::NER));
    }

    #[test]
    fn test_registry_task_codes_are_parseable() {
        // Ensures the task strings in the dataset registry don't silently drift away
        // from our enum mapping (which would make task selection/eval inconsistent).
        //
        // We only enforce this for *in-scope* task codes. The registry currently catalogs
        // some out-of-scope/non-text tasks (e.g., audio captioning) that we do not intend
        // to model in the eval task enum.
        fn is_in_scope_task_code(task_str: &str) -> bool {
            matches!(
                task_str.to_lowercase().as_str(),
                // NER family
                "ner"
                    | "sequence_labeling"
                    | "nested-ner"
                    | "mner"
                    | "pii_detection"
                    | "slot_filling"
                    // Coref family
                    | "coref"
                    | "intra-coref"
                    | "intra_coref"
                    | "intracoref"
                    | "inter-coref"
                    | "inter_coref"
                    | "intercoref"
                    | "cdcr"
                    | "event_coref"
                    // RE / events
                    | "re"
                    | "relation"
                    | "relation_extraction"
                    | "relation-extraction"
                    | "events"
                    | "event"
                    | "event_extraction"
                    | "event-extraction"
                    | "ee"
                    // Discourse / anaphora
                    | "abstract-anaphora"
                    | "abstract_anaphora"
                    | "bridging"
                    | "discourse_deixis"
                    | "discourse_relations"
                    | "discourse-relations"
                    | "discourse_coherence"
                    | "discourse-coherence"
                    | "discourse_segmentation"
                    | "discourse-segmentation"
                    | "speech_act_classification"
                    | "speech-act-classification"
                    // Classification-ish (text-only; not core scope but supported in Task enum)
                    | "classification"
                    | "text_classification"
                    | "text-classification"
                    | "tc"
                    | "bias_evaluation"
                    | "qa"
                    | "harmonic_analysis"
                    | "sentiment"
                    | "sentiment_analysis"
                    | "sentiment-analysis"
                    | "pos"
                    | "pos_tagging"
                    | "pos-tagging"
                    // MT/LM (text-only but not evaluated end-to-end yet)
                    | "mt"
                    | "machine_translation"
                    | "machine-translation"
                    | "translation"
                    | "lm"
                    | "language_modeling"
                    | "language-modeling"
            )
        }

        for ds in DatasetId::all() {
            for &task_str in ds.tasks() {
                if !is_in_scope_task_code(task_str) {
                    continue;
                }
                assert!(
                    Task::from_code(task_str).is_some(),
                    "Unrecognized task code '{}' on dataset {:?}",
                    task_str,
                    ds
                );
            }
        }
    }
}
