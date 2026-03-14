//! Joint Entity Analysis: Coreference + NER + Entity Linking
//!
//! Implements a structured CRF for joint modeling of:
//! - **Coreference resolution** (antecedent selection)
//! - **Named entity recognition** (semantic typing)
//! - **Entity linking** (Wikipedia/Wikidata grounding)
//!
//! Based on Durrett & Klein (2014): "A Joint Model for Entity Analysis"
//!
//! # Architecture
//!
//! The model is a factor graph with three variable types per mention:
//!
//! ```text
//! For each mention m_i:
//!   a_i ∈ {1,...,i-1,NEW}  -- antecedent (coreference)
//!   t_i ∈ EntityTypes      -- semantic type (NER)
//!   e_i ∈ WikiTitles       -- entity link
//!
//! Factors:
//!   ψ_coref(a_i)           -- unary coreference features
//!   ψ_ner(t_i)             -- unary NER features
//!   ψ_link(e_i)            -- unary linking features
//!   ψ_coref_ner(a_i,t_i,t_j)   -- consistent types across coref
//!   ψ_ner_link(t_i,e_i)        -- Wikipedia semantics ↔ NER type
//!   ψ_coref_link(a_i,e_i,e_j)  -- related entities across coref
//! ```
//!
//! # Key Insight
//!
//! Cross-task factors capture mutual constraints:
//!
//! - "The company" coreferent with "Dell" → Dell = ORGANIZATION (not person)
//! - Entity = ORGANIZATION → link to Dell Inc., not Michael Dell
//! - Coreferent mentions should link to related Wikipedia articles
//!
//! # Inference
//!
//! Uses loopy belief propagation with pruning:
//!
//! 1. **Prune** antecedent candidates using coarse mention-ranking model
//! 2. **Initialize** beliefs from unary factors
//! 3. **Iterate** message passing until convergence (3-5 iterations)
//! 4. **Decode** via minimum Bayes risk on marginals
//!
//! # Example
//!
//! ```rust,ignore
//! use anno::joint::{JointModel, JointConfig};
//! use anno::Entity;
//!
//! let model = JointModel::new(JointConfig::default())?;
//!
//! let text = "Dell posted revenues of $14B. The company expanded...";
//! let result = model.analyze(text)?;
//!
//! // result.entities: typed mentions
//! // result.chains: coreference clusters
//! // result.links: Wikipedia/Wikidata links
//! ```
//!
//! # References
//!
//! - Durrett & Klein (2014): "A Joint Model for Entity Analysis" TACL
//! - Durrett & Klein (2013): "Easy Victories and Uphill Battles in Coreference Resolution"
//! - Singh et al. (2013): "Joint Inference of Entities, Relations, and Coreference"
//! - Zhao et al. (2025): "Beyond Benchmarks: Building a Richer Cross-Document Event
//!   Coreference Dataset with Decontextualization" NAACL

#[cfg(feature = "analysis")]
mod cross_context;
mod factors;
mod inference;
mod learning;
mod providers;
mod types;

// Cross-context types (xCoRe integration) - requires eval for cluster_encoder
#[cfg(feature = "analysis")]
pub use cross_context::{
    Context, CrossContextJointConfig, CrossContextJointModel, CrossContextResult, GlobalCorefChain,
    GlobalEntity, WindowSplitter,
};

// Factor types and traits
pub use factors::{
    CorefLinkFactor, CorefLinkWeights, CorefNerFactor, CorefNerWeights, Factor, LinkNerFactor,
    LinkNerWeights, UnaryCorefFactor, UnaryLinkFactor, UnaryNerFactor, WikipediaKnowledgeStore,
};

// Inference types
pub use inference::{log_sum_exp, BeliefPropagation, InferenceConfig, Marginals, MessageSchedule};

// Core types
pub use types::{
    AntecedentValue, Assignment, CoarsePruner, CorefScoreProvider, DecontextualizedMention,
    EventCorefRelation, EventMention, JointConfig, JointMention, JointModel, JointModelBuilder,
    JointResult, JointVariable, LinkScoreProvider, LinkValue, MentionKind, NerScoreProvider,
    VariableDomain, VariableId, VariableType,
};

// Learning types
pub use learning::{DynamicBatchConfig, JointWeights, Trainer, TrainingConfig, TrainingExample};

// Score providers
pub use providers::{
    DictionaryLinkProvider, EntityLinkerProvider, HeuristicCorefProvider, ModelNerProvider,
};
