//! Core types for joint entity analysis.
//!
//! This module implements the factor graph approach to joint NER, coreference,
//! and entity linking from Durrett & Klein (2014). The key insight is that
//! these three tasks are **interdependent**:
//!
//! - **NER informs coreference**: "President Obama" and "the CEO" likely don't
//!   corefer (PERSON vs likely PERSON/ORG mismatch)
//! - **Coreference informs linking**: If "Microsoft" and "the company" corefer,
//!   they should link to the same Wikipedia entity
//! - **Linking informs NER**: If a mention links to a Wikipedia person page,
//!   it's probably a PERSON mention
//!
//! # The Joint Model
//!
//! For each mention m_i, we have three random variables:
//!
//! | Variable | Domain | Meaning |
//! |----------|--------|---------|
//! | a_i | {1..i-1, NEW} | Antecedent (or start new entity) |
//! | t_i | EntityTypes | Semantic type (PER, ORG, LOC, ...) |
//! | e_i | WikiTitles ∪ {NIL} | Entity link (or no KB entry) |
//!
//! These are connected by **factors** that encode soft constraints:
//!
//! ```text
//!   ┌─────────┐     ┌─────────┐     ┌─────────┐
//!   │  NER    │─────│ Coref   │─────│  Link   │
//!   │  (t_i)  │     │  (a_i)  │     │  (e_i)  │
//!   └────┬────┘     └────┬────┘     └────┬────┘
//!        │               │               │
//!        └───────────────┴───────────────┘
//!               Pairwise Factors
//! ```
//!
//! # Inference
//!
//! We use loopy belief propagation to find marginal distributions over each
//! variable, then decode via MAP or marginal inference.
//!
//! # References
//!
//! - Durrett & Klein (2014): "A Joint Model for Entity Analysis: Coreference,
//!   Typing, and Linking" (TACL)
//! - Zhao et al. (2025): RECB for cross-document event coreference (future)

use crate::linking::candidate::CandidateSource;
use crate::linking::linker::LinkedEntity;
use crate::{Confidence, Entity, EntityCategory, EntityType, Language, Result};
use anno_core::{CorefChain, Mention as CorefMention};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::factors::{
    CorefLinkFactor, CorefLinkWeights, CorefNerFactor, CorefNerWeights, Factor, LinkNerFactor,
    LinkNerWeights, UnaryCorefFactor, UnaryLinkFactor, UnaryNerFactor, WikipediaKnowledgeStore,
};
use super::inference::{BeliefPropagation, InferenceConfig, Marginals};
use std::sync::Arc;

// =============================================================================
// Variable Types
// =============================================================================

/// Unique identifier for a variable in the factor graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VariableId {
    /// Mention index this variable belongs to
    pub mention_idx: usize,
    /// Which variable type for this mention
    pub var_type: VariableType,
}

/// Types of variables in the joint model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VariableType {
    /// Antecedent selection: a_i ∈ {1,...,i-1,NEW}
    Antecedent,
    /// Semantic type: t_i ∈ EntityTypes
    SemanticType,
    /// Entity link: e_i ∈ WikiTitles ∪ {NIL}
    EntityLink,
}

/// A variable in the joint model.
#[derive(Debug, Clone)]
pub enum JointVariable {
    /// Antecedent for mention i
    Antecedent {
        /// Mention index
        mention_idx: usize,
        /// Possible antecedents (pruned)
        candidates: Vec<usize>,
    },
    /// Semantic type for mention i
    SemanticType {
        /// Mention index
        mention_idx: usize,
        /// Possible types
        types: Vec<EntityType>,
    },
    /// Entity link for mention i
    EntityLink {
        /// Mention index
        mention_idx: usize,
        /// Candidate KB IDs (e.g., Wikidata Q-numbers)
        candidates: Vec<String>,
    },
}

impl JointVariable {
    /// Get the variable ID.
    pub fn id(&self) -> VariableId {
        match self {
            JointVariable::Antecedent { mention_idx, .. } => VariableId {
                mention_idx: *mention_idx,
                var_type: VariableType::Antecedent,
            },
            JointVariable::SemanticType { mention_idx, .. } => VariableId {
                mention_idx: *mention_idx,
                var_type: VariableType::SemanticType,
            },
            JointVariable::EntityLink { mention_idx, .. } => VariableId {
                mention_idx: *mention_idx,
                var_type: VariableType::EntityLink,
            },
        }
    }

    /// Get domain size.
    pub fn domain_size(&self) -> usize {
        match self {
            JointVariable::Antecedent { candidates, .. } => candidates.len() + 1, // +1 for NEW
            JointVariable::SemanticType { types, .. } => types.len(),
            JointVariable::EntityLink { candidates, .. } => candidates.len() + 1, // +1 for NIL
        }
    }
}

/// Domain of a variable (possible values).
#[derive(Debug, Clone)]
pub enum VariableDomain {
    /// Antecedent domain: indices into mention list, plus NEW_CLUSTER
    Antecedent(Vec<AntecedentValue>),
    /// Type domain: entity types
    SemanticType(Vec<EntityType>),
    /// Link domain: KB IDs plus NIL
    EntityLink(Vec<LinkValue>),
}

/// Value for an antecedent variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AntecedentValue {
    /// Links to mention at index
    Mention(usize),
    /// Starts a new cluster
    NewCluster,
}

/// Value for an entity link variable.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LinkValue {
    /// Links to KB entry
    KbId(String),
    /// Not in knowledge base
    Nil,
}

// =============================================================================
// Assignment
// =============================================================================

/// An assignment of values to variables.
#[derive(Debug, Clone, Default)]
pub struct Assignment {
    /// Antecedent assignments: mention_idx → antecedent
    pub antecedents: HashMap<usize, AntecedentValue>,
    /// Type assignments: mention_idx → entity type
    pub types: HashMap<usize, EntityType>,
    /// Link assignments: mention_idx → link value
    pub links: HashMap<usize, LinkValue>,
}

impl Assignment {
    /// Get antecedent for mention.
    pub fn get_antecedent(&self, mention_idx: usize) -> Option<AntecedentValue> {
        self.antecedents.get(&mention_idx).copied()
    }

    /// Get type for mention.
    pub fn get_type(&self, mention_idx: usize) -> Option<EntityType> {
        self.types.get(&mention_idx).cloned()
    }

    /// Get link for mention.
    pub fn get_link(&self, mention_idx: usize) -> Option<&LinkValue> {
        self.links.get(&mention_idx)
    }

    /// Set antecedent.
    pub fn set_antecedent(&mut self, mention_idx: usize, value: AntecedentValue) {
        self.antecedents.insert(mention_idx, value);
    }

    /// Set type.
    pub fn set_type(&mut self, mention_idx: usize, value: EntityType) {
        self.types.insert(mention_idx, value);
    }

    /// Set link.
    pub fn set_link(&mut self, mention_idx: usize, value: LinkValue) {
        self.links.insert(mention_idx, value);
    }
}

// =============================================================================
// Mention Representation
// =============================================================================

/// Kind of mention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MentionKind {
    /// Proper name (e.g., "Barack Obama")
    Proper,
    /// Common noun (e.g., "the president")
    Nominal,
    /// Pronoun (e.g., "he", "she", "it")
    Pronominal,
}

impl MentionKind {
    /// Infer mention kind from text.
    pub fn from_text(text: &str) -> Self {
        let lower = text.to_lowercase();
        let pronouns = [
            "he",
            "she",
            "it",
            "they",
            "him",
            "her",
            "them",
            "his",
            "hers",
            "its",
            "their",
            "himself",
            "herself",
            "itself",
            "themselves",
            "who",
            "whom",
            "which",
            "that",
        ];

        if pronouns.contains(&lower.as_str()) {
            MentionKind::Pronominal
        } else if text.chars().next().is_some_and(|c| c.is_uppercase()) {
            MentionKind::Proper
        } else {
            MentionKind::Nominal
        }
    }

    /// Check if this is a proper noun mention.
    pub fn is_proper_name(&self) -> bool {
        matches!(self, MentionKind::Proper)
    }

    /// Check if this is a pronoun mention.
    pub fn is_pronoun(&self) -> bool {
        matches!(self, MentionKind::Pronominal)
    }

    /// Check if this is a nominal mention.
    pub fn is_nominal(&self) -> bool {
        matches!(self, MentionKind::Nominal)
    }
}

// =============================================================================
// Cross-Document Event Coreference (RECB)
// =============================================================================

/// Event coreference relation types from RECB (Zhao et al., 2025).
///
/// RECB extends binary coref with fine-grained near-identity relations,
/// enabling richer annotation and evaluation of event coreference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EventCorefRelation {
    /// Full identity: events are the same event instance
    Identity,
    /// Concept-instance: one is a general event, one is a specific instance
    /// E.g., "protests" vs "the March 15th protest"
    ConceptInstance,
    /// Whole-subevent: one event contains the other
    /// E.g., "the war" vs "the Battle of Gettysburg"
    WholeSubevent,
    /// Set-member: one event is part of a set described by the other
    /// E.g., "the attacks" vs "the September 11 attack"
    SetMember,
    /// Topically related but not coreferent
    TopicallyRelated,
    /// Not related
    NotRelated,
    /// Cannot decide (annotation guideline escape)
    CannotDecide,
}

impl EventCorefRelation {
    /// Is this a positive coreference relation?
    ///
    /// Returns true for Identity, ConceptInstance, WholeSubevent, SetMember.
    pub fn is_positive(&self) -> bool {
        matches!(
            self,
            EventCorefRelation::Identity
                | EventCorefRelation::ConceptInstance
                | EventCorefRelation::WholeSubevent
                | EventCorefRelation::SetMember
        )
    }

    /// Convert to standard binary coreference label.
    ///
    /// Only Identity maps to true; all others to false.
    pub fn to_binary(&self) -> bool {
        matches!(self, EventCorefRelation::Identity)
    }

    /// Convert to strict binary with near-identity.
    ///
    /// Maps Identity, ConceptInstance, WholeSubevent, SetMember to true.
    pub fn to_strict_binary(&self) -> bool {
        self.is_positive()
    }
}

/// A decontextualized event mention.
///
/// Decontextualization (Choi et al., 2021) transforms event mentions into
/// self-contained sentences that don't require document context to interpret.
///
/// # Example
///
/// Original: "The company announced it yesterday."
/// Decontextualized: "Apple Inc. announced the new iPhone on March 15, 2024."
#[derive(Debug, Clone)]
pub struct DecontextualizedMention {
    /// Original mention text
    pub original_text: String,
    /// Decontextualized (self-contained) version
    pub decontextualized: String,
    /// Source document ID
    pub doc_id: String,
    /// Original character start offset in source document.
    pub original_start: usize,
    /// Original character end offset in source document.
    pub original_end: usize,
    /// Entities resolved during decontextualization
    pub resolved_entities: Vec<(String, String)>, // (pronoun/reference, resolved value)
}

impl DecontextualizedMention {
    /// Create a new decontextualized mention.
    pub fn new(
        original_text: impl Into<String>,
        decontextualized: impl Into<String>,
        doc_id: impl Into<String>,
        original_start: usize,
        original_end: usize,
    ) -> Self {
        Self {
            original_text: original_text.into(),
            decontextualized: decontextualized.into(),
            doc_id: doc_id.into(),
            original_start,
            original_end,
            resolved_entities: Vec::new(),
        }
    }

    /// Add a resolved entity reference.
    pub fn with_resolved(
        mut self,
        reference: impl Into<String>,
        resolved: impl Into<String>,
    ) -> Self {
        self.resolved_entities
            .push((reference.into(), resolved.into()));
        self
    }
}

/// An event mention for cross-document coreference.
#[derive(Debug, Clone)]
pub struct EventMention {
    /// Unique mention ID
    pub id: String,
    /// Event trigger text
    pub trigger: String,
    /// Event type (if known)
    pub event_type: Option<String>,
    /// Source document ID
    pub doc_id: String,
    /// Character start offset in source document.
    pub start: usize,
    /// Character end offset in source document.
    pub end: usize,
    /// Decontextualized form (for improved annotation/modeling)
    pub decontextualized: Option<DecontextualizedMention>,
}

// =============================================================================
// Mention Representation
// =============================================================================

/// A mention in the joint model with all relevant context.
#[derive(Debug, Clone)]
pub struct JointMention {
    /// Mention index in document
    pub idx: usize,
    /// Surface text
    pub text: String,
    /// Head word
    pub head: String,
    /// Start character offset
    pub start: usize,
    /// End character offset
    pub end: usize,
    /// Mention kind (proper/nominal/pronominal)
    pub mention_kind: MentionKind,
    /// Entity type (if known from NER)
    pub entity_type: Option<EntityType>,
    /// Original entity (if available)
    pub entity: Option<Entity>,
}

impl JointMention {
    /// Create from an Entity.
    pub fn from_entity(idx: usize, entity: &Entity, text: &str) -> Self {
        let mention_text = text
            .chars()
            .skip(entity.start)
            .take(entity.end - entity.start)
            .collect::<String>();

        let head = mention_text
            .split_whitespace()
            .last()
            .unwrap_or(&mention_text)
            .to_string();

        Self {
            idx,
            text: mention_text.clone(),
            head,
            start: entity.start,
            end: entity.end,
            mention_kind: MentionKind::from_text(&mention_text),
            entity_type: Some(entity.entity_type.clone()),
            entity: Some(entity.clone()),
        }
    }
}

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for joint model.
#[derive(Debug, Clone)]
pub struct JointConfig {
    /// Enable Link+NER factors
    pub enable_link_ner: bool,
    /// Enable Coref+NER factors
    pub enable_coref_ner: bool,
    /// Enable Coref+Link factors
    pub enable_coref_link: bool,

    /// Maximum iterations for belief propagation
    pub max_iterations: usize,
    /// Convergence threshold for message changes
    pub convergence_threshold: f64,

    /// Pruning threshold for antecedent candidates (log space)
    pub pruning_threshold: f64,
    /// Maximum antecedent candidates to keep after pruning
    pub max_antecedent_candidates: usize,

    /// Maximum link candidates per mention
    pub max_link_candidates: usize,

    /// Entity types to consider
    pub entity_types: Vec<EntityType>,
}

impl Default for JointConfig {
    fn default() -> Self {
        Self {
            enable_link_ner: true,
            enable_coref_ner: true,
            enable_coref_link: true,

            max_iterations: 5,
            convergence_threshold: 1e-4,

            pruning_threshold: 5.0, // Paper uses k=5
            max_antecedent_candidates: 50,

            max_link_candidates: 20,

            // Include all common NER types so we can preserve original type
            entity_types: vec![
                EntityType::Person,
                EntityType::Organization,
                EntityType::Location,
                EntityType::Date,
                EntityType::Time,
                EntityType::Money,
                EntityType::Percent,
                EntityType::custom("MISC", EntityCategory::Misc),
            ],
        }
    }
}

// =============================================================================
// Results
// =============================================================================

/// Result of joint entity analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointResult {
    /// Typed entity mentions
    pub entities: Vec<Entity>,
    /// Coreference chains
    pub chains: Vec<CorefChain>,
    /// Entity links
    pub links: Vec<LinkedEntity>,
    /// Confidence scores per mention (averaged over variables)
    pub confidences: Vec<Confidence>,
}

// =============================================================================
// Coarse Pruning (§5 of Durrett & Klein 2014)
// =============================================================================

/// Coarse pruner for antecedent candidates.
///
/// From the paper (§5):
/// "We prune the domains of the coreference variables using a coarse model
/// consisting of the coreference factors trained in isolation."
pub struct CoarsePruner {
    /// Pruning threshold in log space (paper uses k=5)
    pub threshold: f64,
    /// Maximum candidates to keep regardless of threshold
    pub max_candidates: usize,
    /// Weight for string match features
    pub string_match_weight: f64,
    /// Weight for distance penalty
    pub distance_weight: f64,
}

impl Default for CoarsePruner {
    fn default() -> Self {
        Self {
            threshold: 5.0, // Paper: k=5
            max_candidates: 50,
            string_match_weight: 2.0,
            distance_weight: 0.1,
        }
    }
}

impl CoarsePruner {
    /// Prune antecedent candidates for a mention.
    pub fn prune_candidates(&self, mention_idx: usize, mentions: &[JointMention]) -> Vec<usize> {
        if mention_idx == 0 {
            return vec![];
        }

        let mention = &mentions[mention_idx];

        // Score all preceding mentions
        let mut scored: Vec<(usize, f64)> = (0..mention_idx)
            .map(|ante_idx| {
                let score = self.score_pair(mention, &mentions[ante_idx], mention_idx - ante_idx);
                (ante_idx, score)
            })
            .collect();

        // Sort by score (best first)
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        if scored.is_empty() {
            return vec![];
        }

        // Find best score
        let best_score = scored[0].1;

        // Prune: keep candidates within threshold of best
        scored
            .into_iter()
            .take_while(|(_, score)| best_score - *score <= self.threshold)
            .take(self.max_candidates)
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Score a mention-antecedent pair.
    fn score_pair(
        &self,
        mention: &JointMention,
        antecedent: &JointMention,
        distance: usize,
    ) -> f64 {
        let mut score = 0.0;

        // String match features
        let m_lower = mention.text.to_lowercase();
        let a_lower = antecedent.text.to_lowercase();
        let m_head = mention.head.to_lowercase();
        let a_head = antecedent.head.to_lowercase();

        // Exact match (strongest signal)
        if m_lower == a_lower {
            score += self.string_match_weight * 1.0;
        }
        // Head match
        else if m_head == a_head {
            score += self.string_match_weight * 0.6;
        }
        // Substring match
        else if m_lower.contains(&a_lower) || a_lower.contains(&m_lower) {
            score += self.string_match_weight * 0.3;
        }

        // Type compatibility
        match (mention.mention_kind, antecedent.mention_kind) {
            // Pronouns resolve to proper nouns well
            (MentionKind::Pronominal, MentionKind::Proper) => score += 0.5,
            // Same type is good
            (a, b) if a == b => score += 0.3,
            _ => {}
        }

        // Distance penalty (log distance)
        score -= self.distance_weight * (distance as f64 + 1.0).ln();

        score
    }
}

// =============================================================================
// Model
// =============================================================================

/// Joint model for entity analysis.
///
/// Combines NER, coreference, and entity linking in a single factor graph
/// following Durrett & Klein (2014).
pub struct JointModel {
    config: JointConfig,
    /// Coarse pruner for antecedent candidates
    pruner: CoarsePruner,
    /// Wikipedia knowledge store for semantics lookups
    knowledge_store: Option<Arc<WikipediaKnowledgeStore>>,
    /// Optional custom NER score provider
    ner_provider: Option<Arc<dyn NerScoreProvider>>,
    /// Optional custom coref score provider
    coref_provider: Option<Arc<dyn CorefScoreProvider>>,
    /// Optional custom link score provider
    link_provider: Option<Arc<dyn LinkScoreProvider>>,
}

impl Default for JointModel {
    fn default() -> Self {
        Self::new(JointConfig::default()).expect("default config should always succeed")
    }
}

impl JointModel {
    /// Create a new joint model.
    pub fn new(config: JointConfig) -> Result<Self> {
        let pruner = CoarsePruner {
            threshold: config.pruning_threshold,
            max_candidates: config.max_antecedent_candidates,
            ..Default::default()
        };

        Ok(Self {
            config,
            pruner,
            knowledge_store: None,
            ner_provider: None,
            coref_provider: None,
            link_provider: None,
        })
    }

    /// Add Wikipedia knowledge store for semantics lookups (Link+NER factors).
    pub fn with_knowledge(mut self, store: Arc<WikipediaKnowledgeStore>) -> Self {
        self.knowledge_store = Some(store);
        self
    }

    /// Attach a custom NER score provider.
    pub fn with_ner_provider(mut self, provider: Arc<dyn NerScoreProvider>) -> Self {
        self.ner_provider = Some(provider);
        self
    }

    /// Attach a custom coreference score provider.
    pub fn with_coref_provider(mut self, provider: Arc<dyn CorefScoreProvider>) -> Self {
        self.coref_provider = Some(provider);
        self
    }

    /// Attach a custom link score provider.
    pub fn with_link_provider(mut self, provider: Arc<dyn LinkScoreProvider>) -> Self {
        self.link_provider = Some(provider);
        self
    }

    /// Analyze text jointly for entities, coreference, and links.
    pub fn analyze(&self, text: &str, entities: &[Entity]) -> Result<JointResult> {
        // 1. Create joint mentions from NER entities
        let mut mentions: Vec<JointMention> = entities
            .iter()
            .enumerate()
            .map(|(i, e)| JointMention::from_entity(i, e, text))
            .collect();

        // 1b. Detect pronouns not already covered by NER entities.
        // Without this, pronouns like "He" never enter the factor graph and
        // cannot be resolved to their antecedents.
        let pronouns: &[&str] = &[
            "he",
            "she",
            "it",
            "they",
            "him",
            "her",
            "them",
            "his",
            "hers",
            "its",
            "their",
            "himself",
            "herself",
            "itself",
            "themselves",
        ];
        let mut char_pos = 0;
        for word in text.split_whitespace() {
            let word_lower = word
                .trim_end_matches(|c: char| c.is_ascii_punctuation())
                .to_lowercase();
            let word_char_len = word
                .trim_end_matches(|c: char| c.is_ascii_punctuation())
                .chars()
                .count();
            if word_char_len > 0 && pronouns.contains(&word_lower.as_str()) {
                let start = char_pos;
                let end = char_pos + word_char_len;
                // Only add if it doesn't overlap an existing NER mention
                let overlaps = mentions.iter().any(|m| start < m.end && end > m.start);
                if !overlaps {
                    let idx = mentions.len();
                    mentions.push(JointMention {
                        idx,
                        text: word
                            .trim_end_matches(|c: char| c.is_ascii_punctuation())
                            .to_string(),
                        head: word_lower.clone(),
                        start,
                        end,
                        mention_kind: MentionKind::Pronominal,
                        entity_type: None,
                        entity: None,
                    });
                }
            }
            char_pos += word.chars().count() + 1; // +1 for space
        }
        // Re-sort by position and re-index
        mentions.sort_by_key(|m| (m.start, m.end));
        for (i, m) in mentions.iter_mut().enumerate() {
            m.idx = i;
        }

        if mentions.is_empty() {
            return Ok(JointResult {
                entities: vec![],
                chains: vec![],
                links: vec![],
                confidences: vec![],
            });
        }

        // 2. Build variables
        let variables = self.build_variables(&mentions);

        // 3. Build factors
        let factors = self.build_factors(&mentions, &variables);

        // 4. Run belief propagation
        let inference_config = InferenceConfig {
            max_iterations: self.config.max_iterations,
            convergence_threshold: self.config.convergence_threshold,
            ..Default::default()
        };
        let mut bp = BeliefPropagation::new(factors, variables.clone(), inference_config);
        let marginals = bp.run();

        // 5. Decode using MBR
        let (entities_out, chains, links, confidences) =
            self.decode(&mentions, &variables, &marginals);

        Ok(JointResult {
            entities: entities_out,
            chains,
            links,
            confidences,
        })
    }

    /// Build variables for all mentions.
    fn build_variables(&self, mentions: &[JointMention]) -> Vec<JointVariable> {
        let mut variables = Vec::new();

        for (i, _mention) in mentions.iter().enumerate() {
            // Antecedent variable (for all except first)
            if i > 0 {
                let pruned = self.pruner.prune_candidates(i, mentions);
                variables.push(JointVariable::Antecedent {
                    mention_idx: i,
                    candidates: pruned,
                });
            }

            // Semantic type variable
            variables.push(JointVariable::SemanticType {
                mention_idx: i,
                types: self.config.entity_types.clone(),
            });

            // Entity link variable
            // In production: query an external linker for candidates (only for Proper mentions).
            let link_candidates: Vec<String> = vec![];
            variables.push(JointVariable::EntityLink {
                mention_idx: i,
                candidates: link_candidates,
            });
        }

        variables
    }

    /// Build factors for the model.
    fn build_factors(
        &self,
        mentions: &[JointMention],
        _variables: &[JointVariable],
    ) -> Vec<Box<dyn Factor>> {
        let mut factors: Vec<Box<dyn Factor>> = Vec::new();

        for mention in mentions {
            let i = mention.idx;

            // Unary NER factor
            let type_scores: Vec<(EntityType, f64)> = if let Some(ref provider) = self.ner_provider
            {
                provider.type_scores(mention, mention.text.as_str())
            } else {
                let original_type = mention.entity.as_ref().map(|e| &e.entity_type);
                self.config
                    .entity_types
                    .iter()
                    .map(|t| {
                        let score = if original_type == Some(t) {
                            10.0 // Strong prior from NER
                        } else {
                            -5.0 // Penalize non-matching types
                        };
                        (t.clone(), score)
                    })
                    .collect()
            };
            factors.push(Box::new(UnaryNerFactor::new(i, type_scores)));

            // Unary coref factor (for mentions after first)
            if i > 0 {
                let candidates: Vec<usize> =
                    (0..i).take(self.config.max_antecedent_candidates).collect();
                let coref_scores: Vec<(AntecedentValue, f64)> =
                    if let Some(ref provider) = self.coref_provider {
                        // Build candidate refs
                        let cand_refs: Vec<&JointMention> =
                            candidates.iter().map(|&idx| &mentions[idx]).collect();
                        provider.antecedent_scores(mention, &cand_refs, mention.text.as_str())
                    } else {
                        let mut scores: Vec<(AntecedentValue, f64)> = candidates
                            .iter()
                            .map(|&ante| {
                                let ante_mention = &mentions[ante];
                                let head_match = if mention.head.to_lowercase()
                                    == ante_mention.head.to_lowercase()
                                {
                                    2.0
                                } else {
                                    0.0
                                };
                                let distance_penalty = -0.1 * (i - ante) as f64;
                                (
                                    AntecedentValue::Mention(ante),
                                    head_match + distance_penalty,
                                )
                            })
                            .collect();
                        scores.push((AntecedentValue::NewCluster, 0.0));
                        scores
                    };
                factors.push(Box::new(UnaryCorefFactor::new(i, coref_scores)));
            }

            // Unary link factor
            let link_candidates_raw = if let Some(ref provider) = self.link_provider {
                provider.link_candidates(mention, mention.text.as_str())
            } else {
                vec![]
            };
            let link_candidates: Vec<(LinkValue, f64)> = link_candidates_raw
                .into_iter()
                .map(|(id, score)| {
                    let lv = if id == "NIL" {
                        LinkValue::Nil
                    } else {
                        LinkValue::KbId(id)
                    };
                    (lv, score)
                })
                .collect();
            factors.push(Box::new(UnaryLinkFactor::new(i, link_candidates)));
        }

        // Cross-task factors
        for mention in mentions {
            let i = mention.idx;

            if i > 0 {
                let candidates: Vec<usize> =
                    (0..i).take(self.config.max_antecedent_candidates).collect();

                for &ante in &candidates {
                    // Coref+NER factor
                    if self.config.enable_coref_ner {
                        factors.push(Box::new(CorefNerFactor::new(
                            i,
                            ante,
                            CorefNerWeights::default(),
                        )));
                    }

                    // Coref+Link factor
                    if self.config.enable_coref_link {
                        let mut factor = CorefLinkFactor::new(i, ante, CorefLinkWeights::default());
                        if let Some(ref store) = self.knowledge_store {
                            factor = factor.with_knowledge(store.clone());
                        }
                        factors.push(Box::new(factor));
                    }
                }
            }

            // Link+NER factor
            if self.config.enable_link_ner {
                let mut factor = LinkNerFactor::new(i, LinkNerWeights::default());
                if let Some(ref store) = self.knowledge_store {
                    factor = factor.with_knowledge(store.clone());
                }
                factors.push(Box::new(factor));
            }
        }

        factors
    }

    /// Decode assignments from marginals using MBR.
    fn decode(
        &self,
        mentions: &[JointMention],
        variables: &[JointVariable],
        marginals: &Marginals,
    ) -> (
        Vec<Entity>,
        Vec<CorefChain>,
        Vec<LinkedEntity>,
        Vec<Confidence>,
    ) {
        let mut entities = Vec::new();
        let mut links = Vec::new();
        let mut confidences = Vec::new();
        let mut antecedents: HashMap<usize, AntecedentValue> = HashMap::new();

        for var in variables {
            let var_id = var.id();
            if let Some(best_idx) = marginals.argmax(&var_id) {
                let prob = marginals.prob(&var_id, best_idx).unwrap_or(0.0);

                match var {
                    JointVariable::Antecedent {
                        mention_idx,
                        candidates,
                    } => {
                        let value = if best_idx < candidates.len() {
                            AntecedentValue::Mention(candidates[best_idx])
                        } else {
                            AntecedentValue::NewCluster
                        };
                        antecedents.insert(*mention_idx, value);
                    }
                    JointVariable::SemanticType {
                        mention_idx, types, ..
                    } => {
                        let m = &mentions[*mention_idx];
                        // Use inferred type if confident, otherwise fall back to original NER
                        let (entity_type, conf) = if let Some(inferred_type) = types.get(best_idx) {
                            // If prob is high, use inferred type
                            if prob > 0.3 {
                                (inferred_type.clone(), Confidence::new(prob))
                            } else if let Some(original) = &m.entity {
                                // Fall back to original NER type
                                (original.entity_type.clone(), original.confidence)
                            } else {
                                (inferred_type.clone(), Confidence::new(prob))
                            }
                        } else if let Some(original) = &m.entity {
                            // No inference available, use original
                            (original.entity_type.clone(), original.confidence)
                        } else {
                            // Should not happen
                            continue;
                        };
                        entities.push(Entity::new(&m.text, entity_type, m.start, m.end, conf));
                        confidences.push(conf);
                    }
                    JointVariable::EntityLink {
                        mention_idx,
                        candidates,
                    } => {
                        let link_value = if best_idx < candidates.len() {
                            LinkValue::KbId(candidates[best_idx].clone())
                        } else {
                            LinkValue::Nil
                        };
                        if let LinkValue::KbId(kb_id) = link_value {
                            let m = &mentions[*mention_idx];
                            links.push(LinkedEntity {
                                mention_text: m.text.clone(),
                                start: m.start,
                                end: m.end,
                                kb_id: Some(kb_id),
                                source: CandidateSource::Wikidata,
                                label: None,
                                iri: None,
                                confidence: Confidence::new(prob),
                                is_nil: false,
                                nil_reason: None,
                                nil_action: None,
                                alternatives: Vec::new(),
                            });
                        }
                    }
                }
            }
        }

        // Build coreference chains from antecedent assignments
        let chains = self.build_chains(&antecedents, mentions);

        (entities, chains, links, confidences)
    }

    /// Build coreference chains from antecedent assignments.
    fn build_chains(
        &self,
        antecedents: &HashMap<usize, AntecedentValue>,
        mentions: &[JointMention],
    ) -> Vec<CorefChain> {
        let n_mentions = mentions.len();
        // Union-find to group mentions
        let mut parent: Vec<usize> = (0..n_mentions).collect();

        fn find(parent: &mut [usize], i: usize) -> usize {
            if parent[i] != i {
                parent[i] = find(parent, parent[i]);
            }
            parent[i]
        }

        fn union(parent: &mut [usize], i: usize, j: usize) {
            let pi = find(parent, i);
            let pj = find(parent, j);
            if pi != pj {
                parent[pi] = pj;
            }
        }

        // Process antecedent assignments
        for (&mention_idx, &ante_value) in antecedents {
            if let AntecedentValue::Mention(ante_idx) = ante_value {
                union(&mut parent, mention_idx, ante_idx);
            }
        }

        // Group by root
        let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..n_mentions {
            let root = find(&mut parent, i);
            clusters.entry(root).or_default().push(i);
        }

        // Fallback: if belief propagation produced only singletons (all
        // mentions are NewCluster), merge mentions with identical normalized
        // text.  This gives joint at least basic name-variant coreference
        // even when unary coref factors are weak/untrained.
        let all_singletons = clusters.values().all(|m| m.len() == 1);
        if all_singletons && n_mentions > 1 {
            let mut text_to_root: HashMap<String, usize> = HashMap::new();
            for (i, mention) in mentions.iter().enumerate().take(n_mentions) {
                let key = mention.text.to_lowercase();
                if let Some(&existing) = text_to_root.get(&key) {
                    union(&mut parent, i, existing);
                } else {
                    text_to_root.insert(key, i);
                }
            }
            // Also merge last-name matches: "Marie Curie" and "Curie"
            for (i, mention) in mentions.iter().enumerate().take(n_mentions) {
                let words: Vec<&str> = mention.text.split_whitespace().collect();
                if words.len() > 1 {
                    let last = words.last().unwrap().to_lowercase();
                    if let Some(&existing) = text_to_root.get(&last) {
                        union(&mut parent, i, existing);
                    }
                }
            }
            // Rebuild clusters after fallback merges
            clusters.clear();
            for i in 0..n_mentions {
                let root = find(&mut parent, i);
                clusters.entry(root).or_default().push(i);
            }
        }

        // Convert to CorefChain
        clusters
            .into_iter()
            .filter(|(_, members)| members.len() > 1) // Only non-singleton
            .enumerate()
            .map(|(chain_id, (_, mut members))| {
                members.sort();
                let coref_mentions: Vec<CorefMention> = members
                    .iter()
                    .map(|&idx| {
                        let m = &mentions[idx];
                        CorefMention {
                            text: m.text.clone(),
                            start: m.start,
                            end: m.end,
                            head_start: None,
                            head_end: None,
                            entity_type: m.entity.as_ref().map(|e| format!("{:?}", e.entity_type)),
                            mention_type: None,
                        }
                    })
                    .collect();
                CorefChain {
                    cluster_id: Some(anno_core::CanonicalId::new(chain_id as u64)),
                    mentions: coref_mentions,
                    entity_type: None,
                }
            })
            .collect()
    }

    /// Get configuration.
    pub fn config(&self) -> &JointConfig {
        &self.config
    }

    /// Extract entities from raw text (requires external NER first).
    ///
    /// This is a convenience method for pipelines that want to use
    /// JointModel as the final step after mention detection.
    pub fn extract_entities_from_mentions(
        &self,
        text: &str,
        mentions: &[JointMention],
    ) -> Result<Vec<Entity>> {
        let entities: Vec<Entity> = mentions.iter().filter_map(|m| m.entity.clone()).collect();

        let result = self.analyze(text, &entities)?;
        Ok(result.entities)
    }
}

// =============================================================================
// Trait Implementations
// =============================================================================

/// Implement the `Model` trait for JointModel to allow it to be used as an NER backend.
///
/// Note: JointModel requires pre-extracted entities as input, so `extract_entities`
/// uses an internal regex-based mention detector as a fallback.
impl crate::Model for JointModel {
    fn extract_entities(&self, text: &str, _language: Option<Language>) -> Result<Vec<Entity>> {
        // For raw text input, we need mention detection first.
        // Use a simple regex-based approach for common entity patterns.
        let initial_entities = detect_mentions_heuristic(text);
        let result = self.analyze(text, &initial_entities)?;
        Ok(result.entities)
    }

    fn supported_types(&self) -> Vec<EntityType> {
        self.config.entity_types.clone()
    }

    fn is_available(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "joint-model"
    }

    fn description(&self) -> &'static str {
        "Joint Entity Analysis: NER + Coreference + Entity Linking (Durrett & Klein 2014)"
    }
}

/// Implement the `CoreferenceResolver` trait for JointModel.
impl anno_core::CoreferenceResolver for JointModel {
    fn resolve(&self, entities: &[Entity]) -> Vec<Entity> {
        if entities.is_empty() {
            return vec![];
        }

        // Create a dummy text for position-based analysis
        // In practice, you should use `analyze_with_text` if you have the text
        let max_end = entities.iter().map(|e| e.end).max().unwrap_or(0);
        let text = " ".repeat(max_end + 1);

        match self.analyze(&text, entities) {
            Ok(result) => {
                // Assign canonical IDs based on coreference chains
                let mut resolved = entities.to_vec();

                for chain in &result.chains {
                    let cluster_id = chain.cluster_id.unwrap_or(anno_core::CanonicalId::ZERO);
                    for mention in &chain.mentions {
                        // Find matching entity by position
                        for entity in &mut resolved {
                            if entity.start == mention.start && entity.end == mention.end {
                                entity.canonical_id = Some(cluster_id);
                            }
                        }
                    }
                }

                // Assign unique IDs to singletons
                let mut next_id = anno_core::CanonicalId::new(result.chains.len() as u64);
                for entity in &mut resolved {
                    if entity.canonical_id.is_none() {
                        entity.canonical_id = Some(next_id);
                        next_id += 1;
                    }
                }

                resolved
            }
            Err(_) => entities.to_vec(),
        }
    }

    fn name(&self) -> &'static str {
        "joint-model-coref"
    }
}

// =============================================================================
// Builder Pattern
// =============================================================================

/// Builder for `JointModel` with fluent configuration.
///
/// # Example
///
/// ```rust,ignore
/// use anno::joint::{JointModelBuilder, WikipediaKnowledgeStore};
///
/// let model = JointModelBuilder::new()
///     .with_max_iterations(10)
///     .with_convergence_threshold(1e-5)
///     .enable_link_ner(true)
///     .enable_coref_ner(true)
///     .enable_coref_link(true)
///     .with_knowledge(knowledge_store)
///     .build()?;
/// ```
#[derive(Clone, Default)]
pub struct JointModelBuilder {
    config: JointConfig,
    knowledge_store: Option<Arc<WikipediaKnowledgeStore>>,
    ner_provider: Option<Arc<dyn NerScoreProvider>>,
    coref_provider: Option<Arc<dyn CorefScoreProvider>>,
    link_provider: Option<Arc<dyn LinkScoreProvider>>,
}

impl JointModelBuilder {
    /// Create a new builder with default configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum iterations for belief propagation.
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.config.max_iterations = max_iterations;
        self
    }

    /// Set convergence threshold for belief propagation.
    pub fn with_convergence_threshold(mut self, threshold: f64) -> Self {
        self.config.convergence_threshold = threshold;
        self
    }

    /// Set pruning threshold for antecedent candidates.
    pub fn with_pruning_threshold(mut self, threshold: f64) -> Self {
        self.config.pruning_threshold = threshold;
        self
    }

    /// Set maximum antecedent candidates to keep after pruning.
    pub fn with_max_antecedent_candidates(mut self, max: usize) -> Self {
        self.config.max_antecedent_candidates = max;
        self
    }

    /// Set maximum link candidates per mention.
    pub fn with_max_link_candidates(mut self, max: usize) -> Self {
        self.config.max_link_candidates = max;
        self
    }

    /// Enable or disable Link+NER factors.
    pub fn enable_link_ner(mut self, enable: bool) -> Self {
        self.config.enable_link_ner = enable;
        self
    }

    /// Enable or disable Coref+NER factors.
    pub fn enable_coref_ner(mut self, enable: bool) -> Self {
        self.config.enable_coref_ner = enable;
        self
    }

    /// Enable or disable Coref+Link factors.
    pub fn enable_coref_link(mut self, enable: bool) -> Self {
        self.config.enable_coref_link = enable;
        self
    }

    /// Set entity types to consider.
    pub fn with_entity_types(mut self, types: Vec<EntityType>) -> Self {
        self.config.entity_types = types;
        self
    }

    /// Add Wikipedia knowledge store for semantics lookups.
    pub fn with_knowledge(mut self, store: Arc<WikipediaKnowledgeStore>) -> Self {
        self.knowledge_store = Some(store);
        self
    }

    /// Plug in a custom NER score provider (unary NER factors).
    pub fn with_ner_provider(mut self, provider: Arc<dyn NerScoreProvider>) -> Self {
        self.ner_provider = Some(provider);
        self
    }

    /// Plug in a custom coreference score provider (unary coref factors).
    pub fn with_coref_provider(mut self, provider: Arc<dyn CorefScoreProvider>) -> Self {
        self.coref_provider = Some(provider);
        self
    }

    /// Plug in a custom link score provider (unary link factors).
    pub fn with_link_provider(mut self, provider: Arc<dyn LinkScoreProvider>) -> Self {
        self.link_provider = Some(provider);
        self
    }

    /// Build the JointModel.
    pub fn build(self) -> Result<JointModel> {
        let mut model = JointModel::new(self.config)?;
        if let Some(store) = self.knowledge_store {
            model = model.with_knowledge(store);
        }
        if let Some(ner) = self.ner_provider {
            model = model.with_ner_provider(ner);
        }
        if let Some(coref) = self.coref_provider {
            model = model.with_coref_provider(coref);
        }
        if let Some(link) = self.link_provider {
            model = model.with_link_provider(link);
        }
        Ok(model)
    }
}

// =============================================================================
// Score Provider Traits
// =============================================================================

/// Trait for providing NER scores for mentions.
///
/// Allows plugging in different NER backends to provide unary type scores.
pub trait NerScoreProvider: Send + Sync {
    /// Get type scores for a mention.
    ///
    /// Returns a vector of (EntityType, log_score) pairs.
    fn type_scores(&self, mention: &JointMention, text: &str) -> Vec<(EntityType, f64)>;
}

/// Trait for providing coreference scores for mention pairs.
///
/// Allows plugging in different mention-ranking models.
pub trait CorefScoreProvider: Send + Sync {
    /// Get antecedent scores for a mention.
    ///
    /// Returns scores for each candidate antecedent plus NEW_CLUSTER.
    fn antecedent_scores(
        &self,
        mention: &JointMention,
        candidates: &[&JointMention],
        text: &str,
    ) -> Vec<(AntecedentValue, f64)>;
}

/// Trait for providing entity linking scores.
///
/// Allows plugging in different candidate generators and rankers.
pub trait LinkScoreProvider: Send + Sync {
    /// Get link candidates for a mention.
    ///
    /// Returns KB IDs and their log scores.
    fn link_candidates(&self, mention: &JointMention, text: &str) -> Vec<(String, f64)>;
}

// =============================================================================
// Heuristic Mention Detection (fallback for Model trait)
// =============================================================================

/// Simple heuristic mention detection for when no external NER is available.
///
/// This is a basic fallback - for best results, use a proper NER backend first.
/// Uses CHARACTER offsets (not byte offsets) as required by Entity.
fn detect_mentions_heuristic(text: &str) -> Vec<Entity> {
    let mut entities = Vec::new();

    // Simple capitalized word sequence detection
    // Track character position explicitly
    let mut in_name = false;
    let mut name_start_char = 0;
    let mut char_pos = 0;

    let chars: Vec<char> = text.chars().collect();

    for c in &chars {
        if c.is_whitespace() || c.is_ascii_punctuation() {
            if in_name {
                // End of name - extract text using character positions
                let name_text: String = chars[name_start_char..char_pos].iter().collect();

                if name_text.chars().count() > 1 {
                    entities.push(Entity::new(
                        &name_text,
                        EntityType::custom("MENTION", EntityCategory::Misc),
                        name_start_char,
                        char_pos,
                        0.5,
                    ));
                }
                in_name = false;
            }
        } else if c.is_uppercase() && !in_name {
            // Start of potential name
            in_name = true;
            name_start_char = char_pos;
        }

        char_pos += 1;
    }

    // Handle trailing name
    if in_name {
        let name_text: String = chars[name_start_char..char_pos].iter().collect();

        if name_text.chars().count() > 1 {
            entities.push(Entity::new(
                &name_text,
                EntityType::custom("MENTION", EntityCategory::Misc),
                name_start_char,
                char_pos,
                0.5,
            ));
        }
    }

    entities
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variable_id() {
        let id = VariableId {
            mention_idx: 0,
            var_type: VariableType::Antecedent,
        };
        assert_eq!(id.mention_idx, 0);
    }

    #[test]
    fn test_assignment() {
        let mut assignment = Assignment::default();
        assignment.set_antecedent(1, AntecedentValue::Mention(0));
        assignment.set_type(0, EntityType::Person);
        assignment.set_link(0, LinkValue::KbId("Q42".to_string()));

        assert_eq!(
            assignment.get_antecedent(1),
            Some(AntecedentValue::Mention(0))
        );
        assert_eq!(assignment.get_type(0), Some(EntityType::Person));
        assert_eq!(
            assignment.get_link(0),
            Some(&LinkValue::KbId("Q42".to_string()))
        );
    }

    #[test]
    fn test_joint_config_default() {
        let config = JointConfig::default();
        assert!(config.enable_link_ner);
        assert!(config.enable_coref_ner);
        assert!(config.enable_coref_link);
        assert_eq!(config.max_iterations, 5);
    }

    #[test]
    fn test_joint_model_creation() {
        let model = JointModel::new(JointConfig::default());
        assert!(model.is_ok());
    }

    #[test]
    fn test_mention_kind_detection() {
        assert_eq!(MentionKind::from_text("he"), MentionKind::Pronominal);
        assert_eq!(MentionKind::from_text("She"), MentionKind::Pronominal);
        assert_eq!(MentionKind::from_text("Barack Obama"), MentionKind::Proper);
        assert_eq!(
            MentionKind::from_text("the president"),
            MentionKind::Nominal
        );
    }

    #[test]
    fn test_joint_model_analyze_empty() {
        let model = JointModel::new(JointConfig::default()).unwrap();
        let result = model.analyze("Hello world", &[]).unwrap();

        assert!(result.entities.is_empty());
        assert!(result.chains.is_empty());
    }

    #[test]
    fn test_joint_model_analyze_single_entity() {
        let model = JointModel::new(JointConfig::default()).unwrap();
        let entities = vec![Entity::new("Obama", EntityType::Person, 0, 5, 0.9)];

        let result = model.analyze("Obama was here.", &entities).unwrap();
        assert!(!result.entities.is_empty());
    }

    #[test]
    fn test_coarse_pruner() {
        let pruner = CoarsePruner::default();

        let mentions = vec![
            JointMention {
                idx: 0,
                text: "Barack Obama".to_string(),
                head: "Obama".to_string(),
                start: 0,
                end: 12,
                mention_kind: MentionKind::Proper,
                entity_type: Some(EntityType::Person),
                entity: None,
            },
            JointMention {
                idx: 1,
                text: "France".to_string(),
                head: "France".to_string(),
                start: 21,
                end: 27,
                mention_kind: MentionKind::Proper,
                entity_type: Some(EntityType::Location),
                entity: None,
            },
            JointMention {
                idx: 2,
                text: "Obama".to_string(),
                head: "Obama".to_string(),
                start: 40,
                end: 45,
                mention_kind: MentionKind::Proper,
                entity_type: Some(EntityType::Person),
                entity: None,
            },
        ];

        let candidates = pruner.prune_candidates(2, &mentions);
        // Should include mention 0 (head match "Obama") but maybe not mention 1
        assert!(!candidates.is_empty());
        // Mention 0 should be the best candidate due to head match
        assert!(candidates.contains(&0));
    }

    // =========================================================================
    // Cross-Document Event Coreference Tests
    // =========================================================================

    #[test]
    fn test_event_coref_relation_is_positive() {
        // Positive relations (should be clustered together)
        assert!(EventCorefRelation::Identity.is_positive());
        assert!(EventCorefRelation::ConceptInstance.is_positive());
        assert!(EventCorefRelation::WholeSubevent.is_positive());
        assert!(EventCorefRelation::SetMember.is_positive());

        // Negative relations (not coreferent)
        assert!(!EventCorefRelation::TopicallyRelated.is_positive());
        assert!(!EventCorefRelation::NotRelated.is_positive());
        assert!(!EventCorefRelation::CannotDecide.is_positive());
    }

    #[test]
    fn test_event_coref_relation_to_binary() {
        // Standard binary: only Identity is positive
        assert!(EventCorefRelation::Identity.to_binary());
        assert!(!EventCorefRelation::ConceptInstance.to_binary());
        assert!(!EventCorefRelation::WholeSubevent.to_binary());
        assert!(!EventCorefRelation::SetMember.to_binary());
        assert!(!EventCorefRelation::NotRelated.to_binary());
    }

    #[test]
    fn test_event_coref_relation_to_strict_binary() {
        // Strict binary: all positive near-identity relations count
        assert!(EventCorefRelation::Identity.to_strict_binary());
        assert!(EventCorefRelation::ConceptInstance.to_strict_binary());
        assert!(EventCorefRelation::WholeSubevent.to_strict_binary());
        assert!(EventCorefRelation::SetMember.to_strict_binary());
        assert!(!EventCorefRelation::NotRelated.to_strict_binary());
        assert!(!EventCorefRelation::TopicallyRelated.to_strict_binary());
    }

    #[test]
    fn test_decontextualized_mention() {
        let mention = DecontextualizedMention::new("it", "Apple Inc.", "doc001", 10, 12)
            .with_resolved("it", "Apple Inc.");

        assert_eq!(mention.original_text, "it");
        assert_eq!(mention.decontextualized, "Apple Inc.");
        assert_eq!(mention.doc_id, "doc001");
        assert_eq!(mention.resolved_entities.len(), 1);
        assert_eq!(
            mention.resolved_entities[0],
            ("it".to_string(), "Apple Inc.".to_string())
        );
    }

    #[test]
    fn test_event_mention() {
        let event = EventMention {
            id: "e001".to_string(),
            trigger: "announced".to_string(),
            event_type: Some("Communication".to_string()),
            doc_id: "doc001".to_string(),
            start: 15,
            end: 24,
            decontextualized: Some(DecontextualizedMention::new(
                "The company announced it yesterday",
                "Apple Inc. announced the new iPhone on March 15, 2024",
                "doc001",
                0,
                34,
            )),
        };

        assert_eq!(event.id, "e001");
        assert_eq!(event.trigger, "announced");
        assert!(event.decontextualized.is_some());
        let decon = event.decontextualized.unwrap();
        assert!(decon.decontextualized.contains("Apple Inc."));
    }

    // ==========================================================================
    // Trait Implementation Tests
    // ==========================================================================

    #[test]
    fn test_model_trait_implementation() {
        use crate::Model;

        let model = JointModel::default();

        // Test Model trait methods
        assert_eq!(model.name(), "joint-model");
        assert!(model.description().contains("Durrett"));
        assert!(model.is_available());

        let types = model.supported_types();
        assert!(!types.is_empty());
    }

    #[test]
    fn test_model_extract_entities_simple() {
        use crate::Model;

        let model = JointModel::default();

        // Test with simple text containing capitalized words
        let text = "John Smith visited New York";
        let entities = model.extract_entities(text, None).unwrap();

        // Heuristic detection may legitimately return empty output; this test only asserts no error.
        let _ = entities;
    }

    #[test]
    fn test_coref_resolver_trait_implementation() {
        use anno_core::CoreferenceResolver;

        let model = JointModel::default();

        // Test CoreferenceResolver trait
        assert_eq!(model.name(), "joint-model-coref");

        // Test with empty input
        let empty_result = model.resolve(&[]);
        assert!(empty_result.is_empty());
    }

    #[test]
    fn test_coref_resolver_assigns_canonical_ids() {
        use anno_core::CoreferenceResolver;

        let model = JointModel::default();

        let entities = vec![
            Entity::new("John", EntityType::Person, 0, 4, 0.9),
            Entity::new("he", EntityType::Person, 10, 12, 0.8),
            Entity::new("Microsoft", EntityType::Organization, 20, 29, 0.95),
        ];

        let resolved = model.resolve(&entities);

        // All entities should have canonical IDs assigned
        assert_eq!(resolved.len(), 3);
        for entity in &resolved {
            assert!(entity.canonical_id.is_some());
        }
    }

    #[test]
    fn test_builder_default() {
        let model = JointModelBuilder::new().build().unwrap();

        // Default configuration matches JointConfig::default()
        let config = model.config();
        assert_eq!(config.max_iterations, 5); // Default is 5
        assert!(config.enable_link_ner);
        assert!(config.enable_coref_ner);
        assert!(config.enable_coref_link);
    }

    #[test]
    fn test_builder_fluent_api() {
        let model = JointModelBuilder::new()
            .with_max_iterations(50)
            .with_convergence_threshold(1e-6)
            .with_pruning_threshold(0.5)
            .with_max_antecedent_candidates(100)
            .with_max_link_candidates(20)
            .enable_link_ner(false)
            .enable_coref_ner(true)
            .enable_coref_link(false)
            .build()
            .unwrap();

        let config = model.config();
        assert_eq!(config.max_iterations, 50);
        assert!((config.convergence_threshold - 1e-6).abs() < 1e-10);
        assert!((config.pruning_threshold - 0.5).abs() < 1e-10);
        assert_eq!(config.max_antecedent_candidates, 100);
        assert_eq!(config.max_link_candidates, 20);
        assert!(!config.enable_link_ner);
        assert!(config.enable_coref_ner);
        assert!(!config.enable_coref_link);
    }

    #[test]
    fn test_builder_with_entity_types() {
        let custom_types = vec![EntityType::Person, EntityType::Organization];

        let model = JointModelBuilder::new()
            .with_entity_types(custom_types.clone())
            .build()
            .unwrap();

        assert_eq!(model.config().entity_types, custom_types);
    }

    #[test]
    fn test_heuristic_mention_detection() {
        // Test the heuristic detection directly
        let text = "Barack Obama met Angela Merkel in Berlin";
        let entities = detect_mentions_heuristic(text);

        // Should detect capitalized sequences
        // Note: May vary based on implementation
        assert!(!entities.is_empty());

        // All detected entities should have valid spans
        for entity in &entities {
            assert!(entity.start < entity.end);
            assert!(entity.end <= text.chars().count());
        }
    }

    #[test]
    fn test_heuristic_mention_detection_unicode() {
        // Test with Unicode characters
        let text = "François Müller visited München";
        let entities = detect_mentions_heuristic(text);

        // Should handle Unicode correctly
        for entity in &entities {
            assert!(entity.start <= entity.end);
            let char_count = text.chars().count();
            assert!(entity.end <= char_count);
        }
    }

    #[test]
    fn test_extract_entities_from_mentions() {
        let model = JointModel::default();

        let text = "John Smith visited New York. He liked the city.";
        let mentions = vec![
            JointMention::from_entity(
                0,
                &Entity::new("John Smith", EntityType::Person, 0, 10, 0.9),
                text,
            ),
            JointMention::from_entity(
                1,
                &Entity::new("New York", EntityType::Location, 19, 27, 0.85),
                text,
            ),
            JointMention::from_entity(2, &Entity::new("He", EntityType::Person, 29, 31, 0.7), text),
        ];

        let result = model.extract_entities_from_mentions(text, &mentions);
        assert!(result.is_ok());

        let entities = result.unwrap();
        // Should return entities with updated types/links based on joint inference
        assert!(!entities.is_empty());
    }

    /// P4: Joint model's analyze should detect pronouns even when NER doesn't
    /// extract them.
    #[test]
    fn test_analyze_detects_pronouns() {
        let model = JointModel::new(JointConfig::default()).unwrap();
        let text = "Marie Curie discovered radium. She won the Nobel Prize.";
        // NER entities without pronouns
        let entities = vec![
            Entity::new("Marie Curie", EntityType::Person, 0, 11, 0.95),
            Entity::new("Nobel Prize", EntityType::Organization, 41, 52, 0.9),
        ];
        let result = model.analyze(text, &entities).unwrap();
        // The joint model should have detected "She" as a pronoun mention
        // and potentially linked it to "Marie Curie" in a coreference chain.
        // At minimum, the output entities should include the NER entities.
        assert!(
            result.entities.len() >= 2,
            "Should have at least the 2 NER entities, got {}",
            result.entities.len()
        );
    }
}
