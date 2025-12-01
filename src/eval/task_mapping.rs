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
pub use crate::backends::inference::{
    DiscontinuousNER as DiscontinuousNERTrait, RelationExtractor as RelationExtractorTrait,
    ZeroShotNER as ZeroShotNERTrait,
};
pub use crate::eval::coref_resolver::CoreferenceResolver as CoreferenceResolverTrait;

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
    /// Hierarchical Structure Extraction: extract nested structures
    HierarchicalExtraction,
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
            Task::HierarchicalExtraction,
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
            Task::HierarchicalExtraction => "Hierarchical Structure Extraction",
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
            Task::HierarchicalExtraction => "hierarchical",
        }
    }
}

/// Mapping from datasets to tasks they support.
pub fn dataset_tasks(dataset: DatasetId) -> &'static [Task] {
    match dataset {
        // NER datasets
        DatasetId::WikiGold
        | DatasetId::Wnut17
        | DatasetId::MitMovie
        | DatasetId::MitRestaurant
        | DatasetId::CoNLL2003Sample
        | DatasetId::OntoNotesSample
        | DatasetId::MultiNERD
        | DatasetId::BC5CDR
        | DatasetId::NCBIDisease
        | DatasetId::GENIA
        | DatasetId::AnatEM
        | DatasetId::BC2GM
        | DatasetId::BC4CHEMD
        | DatasetId::TweetNER7
        | DatasetId::BroadTwitterCorpus
        | DatasetId::FabNER
        | DatasetId::FewNERD
        | DatasetId::CrossNER
        | DatasetId::UniversalNERBench
        | DatasetId::WikiANN
        | DatasetId::MultiCoNER
        | DatasetId::MultiCoNERv2
        | DatasetId::WikiNeural
        | DatasetId::PolyglotNER
        | DatasetId::UniversalNER
        | DatasetId::UNER
        | DatasetId::MSNER
        | DatasetId::BioMNER
        | DatasetId::LegNER => &[Task::NER],

        // Discontinuous NER datasets
        DatasetId::CADEC => &[Task::DiscontinuousNER, Task::NER],

        // Relation Extraction datasets
        DatasetId::DocRED
        | DatasetId::ReTACRED
        | DatasetId::NYTFB
        | DatasetId::WEBNLG
        | DatasetId::GoogleRE
        | DatasetId::BioRED
        | DatasetId::SciER
        | DatasetId::MixRED
        | DatasetId::CovEReD => &[Task::RelationExtraction],

        // Coreference datasets
        DatasetId::GAP | DatasetId::PreCo | DatasetId::LitBank => &[
            Task::IntraDocCoref,
            // Some coref datasets can also evaluate abstract anaphora
            Task::AbstractAnaphora,
        ],
        // Note: OntoNotes has both NER and coreference, but we only have the NER sample
        // Full OntoNotes would support: [Task::NER, Task::IntraDocCoref]
    }
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
        ],
        Task::DiscontinuousNER => {
            &[DatasetId::CADEC]
            // TODO: Add ShARe13, ShARe14 when available
        }
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
        Task::IntraDocCoref => &[DatasetId::GAP, DatasetId::PreCo, DatasetId::LitBank],
        Task::InterDocCoref => {
            // TODO: Add inter-doc coref datasets when available
            &[]
        }
        Task::AbstractAnaphora => &[DatasetId::GAP, DatasetId::PreCo, DatasetId::LitBank],
        Task::NED => {
            // TODO: Add NED datasets (e.g., AIDA, TAC-KBP) when available
            &[]
        }
        Task::EventExtraction => {
            // TODO: Add event extraction datasets (e.g., ACE 2005) when available
            &[]
        }
        Task::TextClassification => {
            // GLiNER2 can do classification, but we don't have dedicated datasets yet
            &[]
        }
        Task::HierarchicalExtraction => {
            // GLiNER2 can do hierarchical extraction, but we don't have dedicated datasets yet
            &[]
        }
    }
}

/// Detect backend capabilities via trait inspection.
///
/// This function attempts to determine what tasks a backend supports
/// by checking if it implements relevant traits. For runtime detection,
/// use `detect_backend_capabilities` instead.
pub fn backend_tasks(backend_name: &str) -> &'static [Task] {
    match backend_name {
        // Pattern-based backends
        "pattern" | "RegexNER" => &[Task::NER], // Only structured entities
        "heuristic" | "HeuristicNER" => &[Task::NER],
        "stacked" | "StackedNER" => &[Task::NER],

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
            Task::HierarchicalExtraction,
            Task::RelationExtraction, // Via RelationExtractor trait
        ],

        // Discontinuous NER backends (implement DiscontinuousNER trait)
        "w2ner" | "W2NER" => &[Task::NER, Task::DiscontinuousNER],

        // Joint entity-relation backends
        "tplinker" | "TPLinker" => &[Task::NER, Task::RelationExtraction],

        // Coreference backends (implement CoreferenceResolver trait)
        "coref_resolver" | "CorefResolver" | "SimpleCorefResolver" | "DiscourseAwareResolver" => {
            &[Task::IntraDocCoref, Task::AbstractAnaphora]
        }

        _ => &[],
    }
}

/// Runtime capability detection for a backend instance.
///
/// Uses trait object downcasting to detect what capabilities a backend has.
/// This is more accurate than string-based matching but requires runtime checks.
///
/// Note: This is a placeholder implementation. Full implementation would use
/// trait object downcasting or a capability registry.
pub fn detect_backend_capabilities<M: crate::Model>(_backend: &M) -> Vec<Task> {
    // Placeholder: return NER (all backends implement Model)
    // Full implementation would:
    // - Use Any::type_id() to check trait implementations
    // - Or maintain a capability registry
    vec![Task::NER]
}

/// Get all tasks that a dataset supports.
pub fn get_dataset_tasks(dataset: DatasetId) -> Vec<Task> {
    dataset_tasks(dataset).to_vec()
}

/// Get all datasets suitable for a task.
pub fn get_task_datasets(task: Task) -> Vec<DatasetId> {
    task_datasets(task).to_vec()
}

/// Get all backends that support a task.
///
/// For benchmarking, only returns "stacked" (which combines pattern+heuristic)
/// and ML backends, since individual pattern/heuristic backends are incomplete.
pub fn get_task_backends(task: Task) -> Vec<&'static str> {
    let mut backends = Vec::new();
    for backend in [
        // Only stacked (combines pattern+heuristic), not individual ones
        "stacked",
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
}
