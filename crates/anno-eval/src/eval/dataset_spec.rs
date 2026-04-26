//! Dataset specifications and metadata for NER evaluation.
//!
//! ## Why This Module Exists
//!
//! NER evaluation requires knowing what datasets exist, what entity types they
//! annotate, what format they're in, and what license governs their use. Without
//! a structured catalog:
//!
//! - Users reinvent the wheel finding/downloading the same datasets
//! - Entity type mappings differ across implementations (PER vs PERSON vs person)
//! - License compliance becomes guesswork
//! - Comparing results across papers requires manual dataset lookup
//!
//! This module provides a trait-based abstraction for dataset metadata, plus a
//! runtime registry for discovering and filtering datasets by task, language,
//! domain, or license.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                      DatasetSpec trait                      │
//! │  name(), id(), task(), languages(), entity_types(), ...     │
//! └─────────────────────────┬───────────────────────────────────┘
//!                           │ implemented by
//!           ┌───────────────┴───────────────┐
//!           │                               │
//!   ┌───────▼───────┐             ┌─────────▼─────────┐
//!   │ CustomDataset │             │   Built-in IDs    │
//!   │ (runtime)     │             │ (compile-time)    │
//!   └───────────────┘             └───────────────────┘
//! ```
//!
//! ## Example
//!
//! ```rust,ignore
//! use anno_eval::eval::{DatasetId, load_dataset};
//!
//! let conll = load_dataset(DatasetId::CoNLL2003, "test")?;
//! println!("Entity types: {:?}", DatasetId::CoNLL2003.entity_types());
//! ```
//!
//! Built-in datasets include CoNLL2003, OntoNotes5, WikiANN (176 languages),
//! BC5CDR (biomedical), AnnoCTR (cybersecurity), WNUT17 (social media), and more.
//!
//! # Custom Dataset
//!
//! ```rust
//! use anno_eval::eval::dataset_spec::{DatasetSpec, DatasetStats, Domain, License, ParserHint, Task};
//!
//! /// Example NER dataset spec (replace with your own dataset details).
//! struct ExampleNER;
//!
//! impl DatasetSpec for ExampleNER {
//!     fn name(&self) -> &str { "Example NER" }
//!     fn id(&self) -> &str { "example_ner_v1" }
//!     fn task(&self) -> Task { Task::NER }
//!     fn languages(&self) -> &[&str] { &["en"] }
//!     fn entity_types(&self) -> &[&str] { &["PER", "ORG", "LOC"] }
//!     fn parser_hint(&self) -> ParserHint { ParserHint::CoNLL }
//!     fn license(&self) -> License { License::CCBY }
//!     fn domain(&self) -> Domain { Domain::News }
//!     fn stats(&self) -> DatasetStats {
//!         DatasetStats { doc_count: Some(1_000), ..Default::default() }
//!     }
//! }
//! ```
//!
//! # Parser Hints
//!
//! Datasets come in various formats. [`ParserHint`] guides the loader:
//!
//! | Format | Description | Example Datasets |
//! |--------|-------------|------------------|
//! | `CoNLL` | Tab-separated BIO tags | CoNLL2003, WNUT17 |
//! | `JSON` | Structured JSON objects | LitBank, MultiNERD |
//! | `JSONL` | JSON Lines (one per doc) | WikiANN, Universal NER |
//! | `CoNLLU` | Universal Dependencies format | UD treebanks |
//! | `BRAT` | Standoff annotation format | custom annotations |
//!
//! # Licensing
//!
//! The [`License`] enum tracks data usage rights:
//!
//! - **Research**: Academic use only (e.g., LDC corpora)
//! - **CC-BY-4.0**: Attribution required, commercial OK
//! - **Apache-2.0**: Permissive, patent grant
//! - **Proprietary**: Internal/commercial datasets
//!
//! # Domain Coverage
//!
//! ```rust
//! use anno_eval::eval::dataset_spec::Domain;
//!
//! // Built-in domains
//! let domains = [
//!     Domain::News,           // CoNLL, OntoNotes
//!     Domain::Biomedical,     // BC5CDR, NCBI-Disease
//!     Domain::SocialMedia,    // WNUT17, Twitter
//!     Domain::Scientific,     // SciERC, WIESP
//!     Domain::Legal,          // E-NER SEC
//!     Domain::Cybersecurity,  // AnnoCTR
//!     Domain::Music,          // Distant Listening Corpus
//!     Domain::Literary,       // LitBank, etc.
//!     Domain::Historical,     // HIPE-2022, medieval corpora
//! ];
//! ```
//!
//! # Classical & Historical Languages
//!
//! For historical-language datasets, prefer ISO 639-3 codes and explicit provenance fields.
//! Keep examples here generic; put dataset-specific details (URLs, stats, citations) in the
//! dataset registry where they can be reviewed for correctness.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::hash::Hash;

// ============================================================================
// Task Enumeration
// ============================================================================

/// The primary NLP task a dataset is designed for.
///
/// A dataset may support multiple tasks (e.g., NER + Entity Linking),
/// but has one primary task that determines its structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Task {
    /// Named Entity Recognition (sequence labeling)
    NER,
    /// Intra-document coreference resolution
    IntraDocCoref,
    /// Inter-document (cross-document) coreference resolution
    InterDocCoref,
    /// Named Entity Disambiguation / Entity Linking to KB
    NED,
    /// Relation Extraction between entities
    RelationExtraction,
    /// Event extraction and argument role labeling
    EventExtraction,
    /// Discontinuous/nested NER (e.g., CADEC, ShARe)
    DiscontinuousNER,
    /// Visual document NER (forms, receipts, etc.)
    VisualNER,
    /// Temporal NER (diachronic entities, time expressions)
    TemporalNER,
    /// Sentiment/opinion target extraction
    AspectExtraction,
    /// Slot filling for dialogue systems
    SlotFilling,
    /// Part-of-speech tagging (often bundled with NER)
    POS,
    /// Dependency parsing
    DependencyParsing,
}

impl Task {
    /// Returns true if this task produces entity spans.
    #[must_use]
    pub const fn produces_entities(&self) -> bool {
        matches!(
            self,
            Self::NER
                | Self::DiscontinuousNER
                | Self::VisualNER
                | Self::TemporalNER
                | Self::AspectExtraction
                | Self::SlotFilling
        )
    }

    /// Returns true if this task involves coreference chains.
    #[must_use]
    pub const fn involves_coreference(&self) -> bool {
        matches!(self, Self::IntraDocCoref | Self::InterDocCoref)
    }

    /// Returns true if this task links to external knowledge bases.
    #[must_use]
    pub const fn involves_kb_linking(&self) -> bool {
        matches!(self, Self::NED)
    }

    /// Returns true if this task extracts relations between entities.
    #[must_use]
    pub const fn involves_relations(&self) -> bool {
        matches!(self, Self::RelationExtraction | Self::EventExtraction)
    }
}

impl fmt::Display for Task {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NER => write!(f, "NER"),
            Self::IntraDocCoref => write!(f, "Intra-Doc Coreference"),
            Self::InterDocCoref => write!(f, "Inter-Doc Coreference"),
            Self::NED => write!(f, "Named Entity Disambiguation"),
            Self::RelationExtraction => write!(f, "Relation Extraction"),
            Self::EventExtraction => write!(f, "Event Extraction"),
            Self::DiscontinuousNER => write!(f, "Discontinuous NER"),
            Self::VisualNER => write!(f, "Visual NER"),
            Self::TemporalNER => write!(f, "Temporal NER"),
            Self::AspectExtraction => write!(f, "Aspect Extraction"),
            Self::SlotFilling => write!(f, "Slot Filling"),
            Self::POS => write!(f, "POS Tagging"),
            Self::DependencyParsing => write!(f, "Dependency Parsing"),
        }
    }
}

impl std::str::FromStr for Task {
    type Err = String;

    /// Parse task from string (case-insensitive, supports common aliases).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use anno_eval::eval::dataset_spec::Task;
    ///
    /// assert_eq!("ner".parse::<Task>().unwrap(), Task::NER);
    /// assert_eq!("coref".parse::<Task>().unwrap(), Task::IntraDocCoref);
    /// assert_eq!("entity_linking".parse::<Task>().unwrap(), Task::NED);
    /// assert_eq!("RE".parse::<Task>().unwrap(), Task::RelationExtraction);
    /// ```
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "ner" | "named_entity_recognition" | "sequence_labeling" => Ok(Self::NER),
            "coref" | "coreference" | "intra_doc_coref" | "intradoccoref" => {
                Ok(Self::IntraDocCoref)
            }
            "cdcr" | "inter_doc_coref" | "interdoccoref" | "cross_doc_coref" => {
                Ok(Self::InterDocCoref)
            }
            "ned" | "el" | "entity_linking" | "disambiguation" => Ok(Self::NED),
            "re" | "relation_extraction" | "relations" => Ok(Self::RelationExtraction),
            "event" | "event_extraction" | "events" => Ok(Self::EventExtraction),
            "discontinuous" | "discontinuous_ner" | "nested" | "nested_ner" => {
                Ok(Self::DiscontinuousNER)
            }
            "visual" | "visual_ner" | "document_ner" => Ok(Self::VisualNER),
            "temporal" | "temporal_ner" | "timex" => Ok(Self::TemporalNER),
            "aspect" | "aspect_extraction" | "absa" => Ok(Self::AspectExtraction),
            "slot" | "slot_filling" | "intent" => Ok(Self::SlotFilling),
            "pos" | "pos_tagging" | "part_of_speech" => Ok(Self::POS),
            "dep" | "dependency" | "dependency_parsing" => Ok(Self::DependencyParsing),
            _ => Err(format!(
                "Unknown task: '{}'. Valid: ner, coref, ned, re, event, ...",
                s
            )),
        }
    }
}

impl Task {
    /// All task variants for iteration.
    pub const ALL: &'static [Task] = &[
        Task::NER,
        Task::IntraDocCoref,
        Task::InterDocCoref,
        Task::NED,
        Task::RelationExtraction,
        Task::EventExtraction,
        Task::DiscontinuousNER,
        Task::VisualNER,
        Task::TemporalNER,
        Task::AspectExtraction,
        Task::SlotFilling,
        Task::POS,
        Task::DependencyParsing,
    ];

    /// Short code for this task (lowercase, no spaces).
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::NER => "ner",
            Self::IntraDocCoref => "coref",
            Self::InterDocCoref => "cdcr",
            Self::NED => "el",
            Self::RelationExtraction => "re",
            Self::EventExtraction => "event",
            Self::DiscontinuousNER => "discontinuous",
            Self::VisualNER => "visual",
            Self::TemporalNER => "temporal",
            Self::AspectExtraction => "aspect",
            Self::SlotFilling => "slot",
            Self::POS => "pos",
            Self::DependencyParsing => "dep",
        }
    }
}

// ============================================================================
// Parser Hints
// ============================================================================

/// Hint for how to parse this dataset's format.
///
/// Used by the loader to select the appropriate parser without
/// requiring format auto-detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub enum ParserHint {
    /// CoNLL-style column format (BIO/IOB2 tags)
    #[default]
    CoNLL,
    /// CoNLL-U format (Universal Dependencies)
    CoNLLU,
    /// JSON with tokens and labels arrays
    JSON,
    /// JSON Lines (one JSON object per line)
    JSONL,
    /// HuggingFace datasets API format
    HuggingFaceAPI,
    /// BRAT standoff annotation format
    BRAT,
    /// XML-based format (TEI, etc.)
    XML,
    /// ACE/ERE XML format
    ACE,
    /// OntoNotes-style format
    OntoNotes,
    /// Custom format requiring manual parsing
    Custom,
}

impl ParserHint {
    /// File extensions typically associated with this format.
    #[must_use]
    pub const fn typical_extensions(&self) -> &'static [&'static str] {
        match self {
            Self::CoNLL => &["conll", "txt", "bio"],
            Self::CoNLLU => &["conllu"],
            Self::JSON => &["json"],
            Self::JSONL => &["jsonl", "ndjson"],
            Self::HuggingFaceAPI => &["json"],
            Self::BRAT => &["ann", "txt"],
            Self::XML | Self::ACE => &["xml", "sgml"],
            Self::OntoNotes => &["onf", "name"],
            Self::Custom => &[],
        }
    }
}

// ============================================================================
// License Information
// ============================================================================

/// License type for dataset usage.
///
/// Important for determining redistribution rights and
/// commercial usage restrictions.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub enum License {
    /// Creative Commons Attribution
    CCBY,
    /// Creative Commons Attribution-ShareAlike
    CCBYSA,
    /// Creative Commons Attribution-NonCommercial
    CCBYNC,
    /// Creative Commons Attribution-NonCommercial-ShareAlike
    CCBYNCSA,
    /// Creative Commons Zero (public domain)
    CC0,
    /// MIT License
    MIT,
    /// Apache 2.0 License
    Apache2,
    /// GNU General Public License
    GPL,
    /// Linguistic Data Consortium (requires membership)
    LDC,
    /// Research-only license
    ResearchOnly,
    /// Proprietary (restricted use; typically not redistributable)
    Proprietary,
    /// Unknown license
    #[default]
    Unknown,
    /// Other license with description
    Other(String),
}

impl License {
    /// Returns true if commercial use is allowed.
    #[must_use]
    pub fn allows_commercial(&self) -> bool {
        matches!(
            self,
            Self::CCBY | Self::CCBYSA | Self::CC0 | Self::MIT | Self::Apache2
        )
    }

    /// Returns true if the dataset can be freely redistributed.
    #[must_use]
    pub fn allows_redistribution(&self) -> bool {
        !matches!(self, Self::LDC | Self::Proprietary | Self::ResearchOnly)
    }
}

impl fmt::Display for License {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CCBY => write!(f, "CC BY 4.0"),
            Self::CCBYSA => write!(f, "CC BY-SA 4.0"),
            Self::CCBYNC => write!(f, "CC BY-NC 4.0"),
            Self::CCBYNCSA => write!(f, "CC BY-NC-SA 4.0"),
            Self::CC0 => write!(f, "CC0 (Public Domain)"),
            Self::MIT => write!(f, "MIT"),
            Self::Apache2 => write!(f, "Apache 2.0"),
            Self::GPL => write!(f, "GPL"),
            Self::LDC => write!(f, "LDC"),
            Self::ResearchOnly => write!(f, "Research Only"),
            Self::Proprietary => write!(f, "Proprietary"),
            Self::Unknown => write!(f, "Unknown"),
            Self::Other(s) => write!(f, "{s}"),
        }
    }
}

// ============================================================================
// Domain Information
// ============================================================================

/// Domain/genre of the dataset's source text.
///
/// Useful for domain adaptation and transfer learning decisions.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub enum Domain {
    /// News articles and journalism
    News,
    /// Biomedical and clinical text
    Biomedical,
    /// Scientific papers and abstracts
    Scientific,
    /// Legal documents and contracts
    Legal,
    /// Financial reports and news
    Financial,
    /// Social media (Twitter, Reddit, etc.)
    SocialMedia,
    /// Wikipedia and encyclopedic text
    Wikipedia,
    /// Literary fiction and novels
    Literary,
    /// Historical documents
    Historical,
    /// Conversational/dialogue text
    Dialogue,
    /// Technical documentation
    Technical,
    /// Web text (general)
    Web,
    /// Cybersecurity reports and threat intelligence
    Cybersecurity,
    /// Music-related text (lyrics, reviews, metadata)
    Music,
    /// Multiple domains
    #[default]
    Mixed,
    /// Other specific domain
    Other(String),
}

impl fmt::Display for Domain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::News => write!(f, "News"),
            Self::Biomedical => write!(f, "Biomedical"),
            Self::Scientific => write!(f, "Scientific"),
            Self::Legal => write!(f, "Legal"),
            Self::Financial => write!(f, "Financial"),
            Self::SocialMedia => write!(f, "Social Media"),
            Self::Wikipedia => write!(f, "Wikipedia"),
            Self::Literary => write!(f, "Literary"),
            Self::Historical => write!(f, "Historical"),
            Self::Dialogue => write!(f, "Dialogue"),
            Self::Technical => write!(f, "Technical"),
            Self::Web => write!(f, "Web"),
            Self::Cybersecurity => write!(f, "Cybersecurity"),
            Self::Music => write!(f, "Music"),
            Self::Mixed => write!(f, "Mixed"),
            Self::Other(s) => write!(f, "{s}"),
        }
    }
}

// ============================================================================
// Temporal Coverage
// ============================================================================

/// Temporal coverage of the dataset.
///
/// Important for understanding potential temporal bias and
/// for diachronic entity tracking.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct TemporalCoverage {
    /// Earliest document date (if known)
    pub start_year: Option<i32>,
    /// Latest document date (if known)
    pub end_year: Option<i32>,
    /// Whether the dataset includes explicit temporal annotations
    pub has_temporal_annotations: bool,
    /// Whether entities have validity periods (diachronic)
    pub has_diachronic_entities: bool,
}

// ============================================================================
// Dataset Statistics
// ============================================================================

/// Statistics about a dataset's size and composition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct DatasetStats {
    /// Number of documents/examples
    pub doc_count: Option<usize>,
    /// Number of entity mentions
    pub mention_count: Option<usize>,
    /// Number of unique entities (after coreference)
    pub entity_count: Option<usize>,
    /// Number of tokens
    pub token_count: Option<usize>,
    /// Train/dev/test split sizes
    pub split_sizes: Option<SplitSizes>,
}

/// Train/dev/test split sizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SplitSizes {
    /// Number of examples in training split
    pub train: usize,
    /// Number of examples in development/validation split
    pub dev: usize,
    /// Number of examples in test split
    pub test: usize,
}

// ============================================================================
// DatasetSpec Trait
// ============================================================================

/// Specification for a dataset that can be loaded and evaluated.
///
/// This trait is the foundation for both built-in datasets (via the
/// `DatasetId` enum) and custom user-defined datasets.
///
/// # Implementing Custom Datasets
///
/// ```rust
/// use anno_eval::eval::dataset_spec::*;
/// use std::path::PathBuf;
///
/// struct MyDataset {
///     path: PathBuf,
/// }
///
/// impl DatasetSpec for MyDataset {
///     fn name(&self) -> &str { "My Custom Dataset" }
///     fn id(&self) -> &str { "my_custom_v1" }
///     fn task(&self) -> Task { Task::NER }
///     fn languages(&self) -> &[&str] { &["en"] }
///     fn entity_types(&self) -> &[&str] { &["PER", "ORG", "LOC"] }
///     fn parser_hint(&self) -> ParserHint { ParserHint::CoNLL }
///     fn license(&self) -> License { License::Proprietary }
///
///     // Override to provide actual data path
///     fn local_path(&self) -> Option<&std::path::Path> {
///         Some(&self.path)
///     }
/// }
/// ```
pub trait DatasetSpec: Send + Sync {
    // ========================================================================
    // Required Methods
    // ========================================================================

    /// Human-readable name of the dataset.
    fn name(&self) -> &str;

    /// Unique identifier string (snake_case, no spaces).
    fn id(&self) -> &str;

    /// Primary task this dataset is designed for.
    fn task(&self) -> Task;

    /// ISO 639-1 language codes (e.g., "en", "zh", "de").
    ///
    /// Use `["multilingual"]` for datasets covering many languages.
    fn languages(&self) -> &[&str];

    /// Entity types annotated in this dataset.
    ///
    /// For NER: `["PER", "LOC", "ORG", "MISC"]`
    /// For biomedical: `["GENE", "DISEASE", "DRUG", "SPECIES"]`
    fn entity_types(&self) -> &[&str];

    /// Parser format hint for loading.
    fn parser_hint(&self) -> ParserHint;

    /// License governing dataset usage.
    fn license(&self) -> License;

    // ========================================================================
    // Optional Methods with Defaults
    // ========================================================================

    /// Detailed description of the dataset.
    fn description(&self) -> Option<&str> {
        None
    }

    /// Domain/genre of source text.
    fn domain(&self) -> Domain {
        Domain::Mixed
    }

    /// URL for downloading the dataset.
    fn download_url(&self) -> Option<&str> {
        None
    }

    /// Citation information (BibTeX or plain text).
    fn citation(&self) -> Option<&str> {
        None
    }

    /// DOI or other persistent identifier.
    fn doi(&self) -> Option<&str> {
        None
    }

    /// Local path if already downloaded.
    fn local_path(&self) -> Option<&std::path::Path> {
        None
    }

    /// Dataset statistics (counts, splits).
    fn stats(&self) -> DatasetStats {
        DatasetStats::default()
    }

    /// Temporal coverage information.
    fn temporal_coverage(&self) -> TemporalCoverage {
        TemporalCoverage::default()
    }

    /// Additional tasks supported beyond the primary task.
    fn secondary_tasks(&self) -> &[Task] {
        &[]
    }

    /// Whether this is a constructed/artificial language dataset.
    fn is_constructed_language(&self) -> bool {
        false
    }

    /// Whether this is a historical/ancient language dataset.
    fn is_historical(&self) -> bool {
        false
    }

    /// Whether this dataset requires special access (gated, auth, etc.).
    fn requires_auth(&self) -> bool {
        false
    }

    /// Version string (e.g., "1.0", "2024-01").
    fn version(&self) -> Option<&str> {
        None
    }

    /// Notes or caveats about the dataset.
    fn notes(&self) -> Option<&str> {
        None
    }

    // ========================================================================
    // Owned Variants (for runtime/custom datasets)
    // ========================================================================

    /// Get languages as owned Vec (for custom datasets that don't have static data).
    ///
    /// Default implementation converts from `languages()`.
    fn languages_vec(&self) -> Vec<String> {
        self.languages().iter().map(|s| (*s).to_string()).collect()
    }

    /// Get entity types as owned Vec (for custom datasets that don't have static data).
    ///
    /// Default implementation converts from `entity_types()`.
    fn entity_types_vec(&self) -> Vec<String> {
        self.entity_types()
            .iter()
            .map(|s| (*s).to_string())
            .collect()
    }

    // ========================================================================
    // Computed Properties
    // ========================================================================

    /// Check if this dataset is publicly available.
    fn is_public(&self) -> bool {
        self.license().allows_redistribution() && !self.requires_auth()
    }

    /// Check if this dataset supports a specific task.
    fn supports_task(&self, task: Task) -> bool {
        self.task() == task || self.secondary_tasks().contains(&task)
    }

    /// Check if this dataset covers a specific language.
    fn supports_language(&self, lang: &str) -> bool {
        let langs = self.languages_vec();
        langs.iter().any(|l| l == "multilingual" || l == lang)
    }

    /// Check if this dataset has a specific entity type.
    fn has_entity_type(&self, entity_type: &str) -> bool {
        self.entity_types_vec()
            .iter()
            .any(|t| t.eq_ignore_ascii_case(entity_type))
    }
}

// ============================================================================
// Custom Dataset Implementation
// ============================================================================

/// A custom dataset defined at runtime.
///
/// Use this when you need to load a dataset that isn't in the built-in
/// `DatasetId` enum.
///
/// # Example
///
/// ```rust
/// use anno_eval::eval::dataset_spec::{CustomDataset, Task, ParserHint, License, Domain};
/// use std::path::PathBuf;
///
/// let dataset = CustomDataset::new("my_ner_data", Task::NER)
///     .with_name("My Company NER Dataset")
///     .with_languages(&["en", "de"])
///     .with_entity_types(&["PRODUCT", "TEAM", "PROJECT"])
///     .with_parser(ParserHint::CoNLL)
///     .with_license(License::Proprietary)
///     .with_domain(Domain::Technical)
///     .with_path(PathBuf::from("/data/my_ner.conll"));
/// ```
#[derive(Debug, Clone)]
pub struct CustomDataset {
    id: String,
    name: String,
    task: Task,
    languages: Vec<String>,
    entity_types: Vec<String>,
    parser_hint: ParserHint,
    license: License,
    description: Option<String>,
    domain: Domain,
    download_url: Option<String>,
    local_path: Option<std::path::PathBuf>,
    stats: DatasetStats,
    temporal_coverage: TemporalCoverage,
    secondary_tasks: Vec<Task>,
    is_constructed: bool,
    is_historical: bool,
    requires_auth: bool,
    version: Option<String>,
    notes: Option<String>,
    citation: Option<String>,
}

impl CustomDataset {
    /// Create a new custom dataset with minimal required fields.
    #[must_use]
    pub fn new(id: impl Into<String>, task: Task) -> Self {
        let id = id.into();
        Self {
            name: id.clone(),
            id,
            task,
            languages: vec!["en".to_string()],
            entity_types: vec![],
            parser_hint: ParserHint::CoNLL,
            license: License::Unknown,
            description: None,
            domain: Domain::Mixed,
            download_url: None,
            local_path: None,
            stats: DatasetStats::default(),
            temporal_coverage: TemporalCoverage::default(),
            secondary_tasks: vec![],
            is_constructed: false,
            is_historical: false,
            requires_auth: false,
            version: None,
            notes: None,
            citation: None,
        }
    }

    /// Set the human-readable name.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set the languages covered.
    #[must_use]
    pub fn with_languages(mut self, langs: &[&str]) -> Self {
        self.languages = langs.iter().map(|s| (*s).to_string()).collect();
        self
    }

    /// Set the entity types.
    #[must_use]
    pub fn with_entity_types(mut self, types: &[&str]) -> Self {
        self.entity_types = types.iter().map(|s| (*s).to_string()).collect();
        self
    }

    /// Set the parser hint.
    #[must_use]
    pub fn with_parser(mut self, parser: ParserHint) -> Self {
        self.parser_hint = parser;
        self
    }

    /// Set the license.
    #[must_use]
    pub fn with_license(mut self, license: License) -> Self {
        self.license = license;
        self
    }

    /// Set the description.
    #[must_use]
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Set the domain.
    #[must_use]
    pub fn with_domain(mut self, domain: Domain) -> Self {
        self.domain = domain;
        self
    }

    /// Set the download URL.
    #[must_use]
    pub fn with_url(mut self, url: impl Into<String>) -> Self {
        self.download_url = Some(url.into());
        self
    }

    /// Set the local file path.
    #[must_use]
    pub fn with_path(mut self, path: std::path::PathBuf) -> Self {
        self.local_path = Some(path);
        self
    }

    /// Set dataset statistics.
    #[must_use]
    pub fn with_stats(mut self, stats: DatasetStats) -> Self {
        self.stats = stats;
        self
    }

    /// Set temporal coverage.
    #[must_use]
    pub fn with_temporal_coverage(mut self, coverage: TemporalCoverage) -> Self {
        self.temporal_coverage = coverage;
        self
    }

    /// Add secondary tasks.
    #[must_use]
    pub fn with_secondary_tasks(mut self, tasks: Vec<Task>) -> Self {
        self.secondary_tasks = tasks;
        self
    }

    /// Mark as constructed language.
    #[must_use]
    pub fn constructed(mut self) -> Self {
        self.is_constructed = true;
        self
    }

    /// Mark as historical language.
    #[must_use]
    pub fn historical(mut self) -> Self {
        self.is_historical = true;
        self
    }

    /// Mark as requiring authentication.
    #[must_use]
    pub fn requires_authentication(mut self) -> Self {
        self.requires_auth = true;
        self
    }

    /// Set version string.
    #[must_use]
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = Some(version.into());
        self
    }

    /// Get languages as owned strings (for custom datasets).
    #[must_use]
    pub fn languages_owned(&self) -> &[String] {
        &self.languages
    }

    /// Get entity types as owned strings (for custom datasets).
    #[must_use]
    pub fn entity_types_owned(&self) -> &[String] {
        &self.entity_types
    }

    /// Set notes.
    #[must_use]
    pub fn with_notes(mut self, notes: impl Into<String>) -> Self {
        self.notes = Some(notes.into());
        self
    }

    /// Set citation.
    #[must_use]
    pub fn with_citation(mut self, citation: impl Into<String>) -> Self {
        self.citation = Some(citation.into());
        self
    }
}

impl DatasetSpec for CustomDataset {
    fn name(&self) -> &str {
        &self.name
    }

    fn id(&self) -> &str {
        &self.id
    }

    fn task(&self) -> Task {
        self.task
    }

    fn languages(&self) -> &[&str] {
        // This is handled via cached_languages field
        // For CustomDataset, we use a different pattern - see languages_owned()
        static EMPTY: &[&str] = &[];
        EMPTY
    }

    fn entity_types(&self) -> &[&str] {
        // This is handled via cached_entity_types field
        // For CustomDataset, we use a different pattern - see entity_types_owned()
        static EMPTY: &[&str] = &[];
        EMPTY
    }

    fn parser_hint(&self) -> ParserHint {
        self.parser_hint
    }

    fn license(&self) -> License {
        self.license.clone()
    }

    fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    fn domain(&self) -> Domain {
        self.domain.clone()
    }

    fn download_url(&self) -> Option<&str> {
        self.download_url.as_deref()
    }

    fn local_path(&self) -> Option<&std::path::Path> {
        self.local_path.as_deref()
    }

    fn stats(&self) -> DatasetStats {
        self.stats.clone()
    }

    fn temporal_coverage(&self) -> TemporalCoverage {
        self.temporal_coverage.clone()
    }

    fn secondary_tasks(&self) -> &[Task] {
        &self.secondary_tasks
    }

    fn is_constructed_language(&self) -> bool {
        self.is_constructed
    }

    fn is_historical(&self) -> bool {
        self.is_historical
    }

    fn requires_auth(&self) -> bool {
        self.requires_auth
    }

    fn version(&self) -> Option<&str> {
        self.version.as_deref()
    }

    fn notes(&self) -> Option<&str> {
        self.notes.as_deref()
    }

    fn citation(&self) -> Option<&str> {
        self.citation.as_deref()
    }

    // Override to use owned data directly instead of converting
    fn languages_vec(&self) -> Vec<String> {
        self.languages.clone()
    }

    fn entity_types_vec(&self) -> Vec<String> {
        self.entity_types.clone()
    }
}

// ============================================================================
// Dataset Registry
// ============================================================================

/// Registry for dynamically registered custom datasets.
///
/// This allows users to register their own datasets at runtime
/// without modifying the built-in enum.
#[derive(Default)]
pub struct DatasetRegistry {
    datasets: std::collections::HashMap<String, Box<dyn DatasetSpec>>,
}

impl DatasetRegistry {
    /// Create a new empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a custom dataset.
    ///
    /// Returns the previous dataset with this ID if one existed.
    pub fn register(
        &mut self,
        dataset: impl DatasetSpec + 'static,
    ) -> Option<Box<dyn DatasetSpec>> {
        let id = dataset.id().to_string();
        self.datasets.insert(id, Box::new(dataset))
    }

    /// Get a dataset by ID.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&dyn DatasetSpec> {
        self.datasets.get(id).map(|b| &**b)
    }

    /// Remove a dataset by ID.
    pub fn unregister(&mut self, id: &str) -> Option<Box<dyn DatasetSpec>> {
        self.datasets.remove(id)
    }

    /// List all registered dataset IDs.
    #[must_use]
    pub fn list_ids(&self) -> Vec<&str> {
        self.datasets.keys().map(|s| s.as_str()).collect()
    }

    /// Number of registered datasets.
    #[must_use]
    pub fn len(&self) -> usize {
        self.datasets.len()
    }

    /// Check if registry is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.datasets.is_empty()
    }

    /// Iterate over all registered datasets.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &dyn DatasetSpec)> {
        self.datasets.iter().map(|(k, v)| (k.as_str(), &**v))
    }

    /// Filter datasets by task.
    pub fn by_task(&self, task: Task) -> impl Iterator<Item = &dyn DatasetSpec> {
        self.datasets
            .values()
            .filter(move |d| d.supports_task(task))
            .map(|b| &**b)
    }

    /// Filter datasets by language.
    pub fn by_language<'a>(&'a self, lang: &'a str) -> impl Iterator<Item = &'a dyn DatasetSpec> {
        self.datasets
            .values()
            .filter(move |d| d.supports_language(lang))
            .map(|b| &**b)
    }

    /// Filter datasets by domain.
    pub fn by_domain(&self, domain: Domain) -> impl Iterator<Item = &dyn DatasetSpec> {
        self.datasets
            .values()
            .filter(move |d| d.domain() == domain)
            .map(|b| &**b)
    }

    /// Filter datasets that are publicly available (no auth, redistributable license).
    pub fn public_only(&self) -> impl Iterator<Item = &dyn DatasetSpec> {
        self.datasets
            .values()
            .filter(|d| d.is_public())
            .map(|b| &**b)
    }

    /// Filter historical/ancient language datasets.
    pub fn historical(&self) -> impl Iterator<Item = &dyn DatasetSpec> {
        self.datasets
            .values()
            .filter(|d| d.is_historical())
            .map(|b| &**b)
    }

    /// Find datasets supporting a specific entity type.
    pub fn with_entity_type<'a>(
        &'a self,
        entity_type: &'a str,
    ) -> impl Iterator<Item = &'a dyn DatasetSpec> {
        self.datasets
            .values()
            .filter(move |d| d.has_entity_type(entity_type))
            .map(|b| &**b)
    }

    /// Get summary statistics about registered datasets.
    #[must_use]
    pub fn summary(&self) -> RegistrySummary {
        let mut tasks = std::collections::HashMap::new();
        let mut domains = std::collections::HashMap::new();
        let mut languages = std::collections::HashSet::new();

        for ds in self.datasets.values() {
            *tasks.entry(ds.task()).or_insert(0) += 1;
            *domains.entry(ds.domain()).or_insert(0) += 1;
            for lang in ds.languages_vec() {
                languages.insert(lang);
            }
        }

        RegistrySummary {
            total: self.datasets.len(),
            by_task: tasks,
            by_domain: domains,
            languages: languages.into_iter().collect(),
        }
    }
}

/// Summary statistics for a dataset registry.
#[derive(Debug, Clone)]
pub struct RegistrySummary {
    /// Total number of datasets.
    pub total: usize,
    /// Count by primary task.
    pub by_task: std::collections::HashMap<Task, usize>,
    /// Count by domain.
    pub by_domain: std::collections::HashMap<Domain, usize>,
    /// All languages covered.
    pub languages: Vec<String>,
}

impl fmt::Debug for DatasetRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DatasetRegistry")
            .field("count", &self.datasets.len())
            .field("ids", &self.list_ids())
            .finish()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_custom_dataset_creation() {
        let dataset = CustomDataset::new("test_ner", Task::NER)
            .with_name("Test NER Dataset")
            .with_languages(&["en", "de"])
            .with_entity_types(&["PER", "LOC", "ORG"])
            .with_license(License::MIT)
            .with_domain(Domain::News);

        assert_eq!(dataset.id(), "test_ner");
        assert_eq!(dataset.name(), "Test NER Dataset");
        assert_eq!(dataset.task(), Task::NER);
        // Use owned getters for custom datasets
        assert!(dataset.languages_owned().contains(&"en".to_string()));
        assert!(dataset.languages_owned().contains(&"de".to_string()));
        assert!(!dataset.languages_owned().contains(&"fr".to_string()));
        assert!(dataset
            .entity_types_owned()
            .iter()
            .any(|t| t.eq_ignore_ascii_case("PER")));
        assert!(dataset
            .entity_types_owned()
            .iter()
            .any(|t| t.eq_ignore_ascii_case("per"))); // case insensitive
        assert!(dataset.is_public());
    }

    #[test]
    fn test_registry() {
        let mut registry = DatasetRegistry::new();

        let dataset1 = CustomDataset::new("ds1", Task::NER)
            .with_name("Dataset 1")
            .with_languages(&["en"]);

        let dataset2 = CustomDataset::new("ds2", Task::IntraDocCoref)
            .with_name("Dataset 2")
            .with_languages(&["de"]);

        registry.register(dataset1);
        registry.register(dataset2);

        assert_eq!(registry.len(), 2);
        assert!(registry.get("ds1").is_some());
        assert!(registry.get("ds2").is_some());
        assert!(registry.get("ds3").is_none());

        let ner_datasets: Vec<_> = registry.by_task(Task::NER).collect();
        assert_eq!(ner_datasets.len(), 1);
        assert_eq!(ner_datasets[0].id(), "ds1");
    }

    #[test]
    fn test_task_properties() {
        assert!(Task::NER.produces_entities());
        assert!(!Task::IntraDocCoref.produces_entities());
        assert!(Task::IntraDocCoref.involves_coreference());
        assert!(Task::InterDocCoref.involves_coreference());
        assert!(!Task::NER.involves_coreference());
        assert!(Task::NED.involves_kb_linking());
        assert!(Task::RelationExtraction.involves_relations());
    }

    #[test]
    fn test_license_properties() {
        assert!(License::MIT.allows_commercial());
        assert!(License::MIT.allows_redistribution());
        assert!(!License::LDC.allows_redistribution());
        assert!(!License::ResearchOnly.allows_commercial());
    }

    #[test]
    fn test_parser_extensions() {
        assert!(ParserHint::CoNLL.typical_extensions().contains(&"conll"));
        assert!(ParserHint::JSONL.typical_extensions().contains(&"jsonl"));
    }

    #[test]
    fn test_task_from_str() {
        // Basic parsing
        assert_eq!("ner".parse::<Task>().expect("task parse"), Task::NER);
        assert_eq!("NER".parse::<Task>().expect("task parse"), Task::NER);
        assert_eq!(
            "coref".parse::<Task>().expect("task parse"),
            Task::IntraDocCoref
        );
        assert_eq!(
            "cdcr".parse::<Task>().expect("task parse"),
            Task::InterDocCoref
        );
        assert_eq!("el".parse::<Task>().expect("task parse"), Task::NED);
        assert_eq!(
            "entity_linking".parse::<Task>().expect("task parse"),
            Task::NED
        );
        assert_eq!(
            "re".parse::<Task>().expect("task parse"),
            Task::RelationExtraction
        );

        // Invalid task
        assert!("invalid_task".parse::<Task>().is_err());
    }

    #[test]
    fn test_task_code() {
        assert_eq!(Task::NER.code(), "ner");
        assert_eq!(Task::IntraDocCoref.code(), "coref");
        assert_eq!(Task::NED.code(), "el");
        assert_eq!(Task::RelationExtraction.code(), "re");
    }

    #[test]
    fn test_task_all_variants() {
        // Ensure ALL contains all variants
        assert!(Task::ALL.contains(&Task::NER));
        assert!(Task::ALL.contains(&Task::IntraDocCoref));
        assert!(Task::ALL.contains(&Task::NED));
        assert_eq!(Task::ALL.len(), 13); // Update if variants change
    }

    #[test]
    fn test_registry_filtering() {
        let mut registry = DatasetRegistry::new();

        // Add diverse datasets
        registry.register(
            CustomDataset::new("biomedical_ner", Task::NER)
                .with_languages(&["en"])
                .with_domain(Domain::Biomedical)
                .with_entity_types(&["DISEASE", "DRUG"]),
        );
        registry.register(
            CustomDataset::new("news_coref", Task::IntraDocCoref)
                .with_languages(&["en", "de"])
                .with_domain(Domain::News),
        );
        registry.register(
            CustomDataset::new("sanskrit_edl", Task::NED)
                .with_languages(&["sa"])
                .with_domain(Domain::Literary)
                .historical(),
        );

        // Test by_domain
        let bio: Vec<_> = registry.by_domain(Domain::Biomedical).collect();
        assert_eq!(bio.len(), 1);
        assert_eq!(bio[0].id(), "biomedical_ner");

        // Test by_language
        let german: Vec<_> = registry.by_language("de").collect();
        assert_eq!(german.len(), 1);
        assert_eq!(german[0].id(), "news_coref");

        // Test historical
        let historical: Vec<_> = registry.historical().collect();
        assert_eq!(historical.len(), 1);
        assert_eq!(historical[0].id(), "sanskrit_edl");

        // Test with_entity_type
        let disease: Vec<_> = registry.with_entity_type("DISEASE").collect();
        assert_eq!(disease.len(), 1);
    }

    #[test]
    fn test_registry_summary() {
        let mut registry = DatasetRegistry::new();
        registry.register(CustomDataset::new("a", Task::NER).with_languages(&["en"]));
        registry.register(CustomDataset::new("b", Task::NER).with_languages(&["de"]));
        registry.register(CustomDataset::new("c", Task::IntraDocCoref).with_languages(&["en"]));

        let summary = registry.summary();
        assert_eq!(summary.total, 3);
        assert_eq!(summary.by_task.get(&Task::NER), Some(&2));
        assert_eq!(summary.by_task.get(&Task::IntraDocCoref), Some(&1));
        assert!(summary.languages.contains(&"en".to_string()));
        assert!(summary.languages.contains(&"de".to_string()));
    }

    #[test]
    fn test_historical_custom_dataset_smoke() {
        // Keep tests generic; dataset-specific examples belong in the dataset registry.
        let ds = CustomDataset::new("historical_edl", Task::NED)
            .with_name("Historical EDL (example)")
            .with_languages(&["sa"])
            .with_entity_types(&["Person", "Location"])
            .with_parser(ParserHint::CoNLLU)
            .with_license(License::CCBY)
            .with_domain(Domain::Literary)
            .with_secondary_tasks(vec![Task::IntraDocCoref, Task::NER])
            .with_stats(DatasetStats {
                doc_count: Some(10),
                mention_count: Some(100),
                ..Default::default()
            })
            .with_citation("Example citation")
            .historical();

        assert_eq!(ds.task(), Task::NED);
        assert!(ds.supports_language("sa"));
        assert!(ds.is_historical());
        assert!(ds.is_public());
    }

    #[test]
    fn test_domain_display() {
        assert_eq!(format!("{}", Domain::Biomedical), "Biomedical");
        assert_eq!(format!("{}", Domain::Literary), "Literary");
        assert_eq!(format!("{}", Domain::Other("custom".into())), "custom");
    }

    #[test]
    fn test_license_display() {
        assert_eq!(format!("{}", License::CCBY), "CC BY 4.0");
        assert_eq!(format!("{}", License::MIT), "MIT");
        assert_eq!(format!("{}", License::LDC), "LDC");
    }

    #[test]
    fn test_temporal_coverage() {
        let cov = TemporalCoverage {
            start_year: Some(2010),
            end_year: Some(2020),
            has_temporal_annotations: true,
            has_diachronic_entities: false,
        };

        assert_eq!(cov.start_year, Some(2010));
        assert!(cov.has_temporal_annotations);
    }

    #[test]
    fn test_split_sizes() {
        let splits = SplitSizes {
            train: 1000,
            dev: 100,
            test: 200,
        };

        assert_eq!(splits.train + splits.dev + splits.test, 1300);
    }
}
