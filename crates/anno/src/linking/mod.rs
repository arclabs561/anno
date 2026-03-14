//! Entity Linking (NEL/NED) Module.
//!
//! Connects extracted mentions to knowledge base entries like Wikidata, DBpedia, or UMLS.
//!
//! # Architecture
//!
//! This module implements a **BLINK-style** entity linking pipeline:
//!
//! ```text
//! ┌─────────────┐     ┌────────────────────┐     ┌──────────────┐     ┌───────────────┐
//! │   Mention   │ ──► │ Candidate Generator│ ──► │  Bi-Encoder  │ ──► │ Cross-Encoder │
//! │  "Einstein" │     │  (High-recall)     │     │ (Dense rank) │     │  (Re-rank)    │
//! └─────────────┘     └────────────────────┘     └──────────────┘     └───────────────┘
//!                                                                             │
//!                                                                             ▼
//!                                                                     ┌───────────────┐
//!                                                                     │ NIL Detection │
//!                                                                     │ (Unlinkable?) │
//!                                                                     └───────────────┘
//! ```
//!
//! # Components
//!
//! | Component | Purpose | Module |
//! |-----------|---------|--------|
//! | **Candidate Generation** | High-recall retrieval of potential KB entries | `candidate` |
//! | **NIL Detection** | Identifies unlinkable mentions | `nil` |
//! | **Entity Linker** | Combines all components | `linker` |
//!
//! # Knowledge Bases Supported
//!
//! - **Wikidata** - Q-numbers (e.g., Q937 for Einstein)
//! - **DBpedia** - Wikipedia-derived structured data
//! - **UMLS** - Unified Medical Language System (biomedical)
//! - **Custom** - Pluggable generators for proprietary KBs
//!
//! # Quick Start
//!
//! ```rust
//! use anno::linking::{EntityLinker, Mention};
//! use anno::EntityType;
//!
//! // Create linker with default dictionary
//! let linker = EntityLinker::builder()
//!     .with_nil_threshold(0.3)
//!     .build();
//!
//! // Link a single mention
//! let linked = linker.link_one(
//!     "Albert Einstein",
//!     "He was a theoretical physicist.",
//!     Some(EntityType::Person)
//! );
//!
//! if let Some(entity) = linked {
//!     if !entity.is_nil {
//!         println!("Linked to: {:?}", entity.kb_id);
//!         println!("IRI: {:?}", entity.iri);
//!     }
//! }
//! ```
//!
//! # NIL Detection
//!
//! Not all mentions can be linked to a KB entry. NIL detection identifies these cases:
//!
//! | Reason | Description |
//! |--------|-------------|
//! | `NoCandidates` | Candidate generator found nothing |
//! | `LowConfidence` | Best candidate score below threshold |
//! | `TypeMismatch` | NER type incompatible with all candidates |
//! | `NoisyMention` | Mention appears to be noise (too short, numeric) |
//!
//! # Integration with Coalesce
//!
//! Entity linking feeds into cross-document coreference via `anno::coalesce`:
//!
//! ```text
//! Linked Entities ──► Coalesce ──► Identities (with KB IDs)
//! ```
//!
//! See [`crate::preprocess::reference`] for extracting KB links from text.

pub mod candidate;
pub mod confusables;
pub mod kb;
pub mod linker;
pub mod nil;
pub mod wikidata;

// Re-export main types for convenient access
pub use candidate::{
    string_similarity, type_compatibility, Candidate, CandidateGenerator, CandidateSource,
    DictionaryCandidateGenerator, SimilarityMetric,
};
pub use confusables::{
    ConfusableEntity, ConfusableReason, ConfusableRegistry, ConfusableSet, OcrConfusables,
    TrainingPair,
};
pub use kb::{
    CrossKBMapper, EntityURI, KnowledgeBase, NILCluster, UnifiedLinker, UnifiedLinkerBuilder,
    YAGOEntity,
};
pub use linker::{
    CandidateSummary, EntityLinker, EntityLinkerBuilder, LinkedEntity, LinkingResult, Mention,
};
pub use nil::{CandidateWithEmbedding, NilAction, NilAnalysis, NilDetector, NilReason};
pub use wikidata::{WikidataDictionary, WikidataEntity, WikidataTypeMapper};
