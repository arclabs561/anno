//! Factor definitions for the joint entity analysis model.
//!
//! In factor graphs, **factors** (also called "potential functions") encode
//! dependencies between variables. Each factor is a function ψ(X_S) over a
//! subset S of variables that returns a non-negative score indicating how
//! "good" that configuration is.
//!
//! The joint distribution is then:
//!
//! ```text
//! P(X) ∝ ∏_f ψ_f(X_{S_f})
//! ```
//!
//! # Factor Types in Joint Entity Analysis
//!
//! ## Unary Factors (single variable)
//!
//! These capture task-specific features for individual mentions:
//!
//! | Factor | Variable | What It Encodes |
//! |--------|----------|-----------------|
//! | `UnaryCorefFactor` | a_i | Mention-ranking features (string match, distance, etc.) |
//! | `UnaryNerFactor` | t_i | NER classifier features (word shape, context, etc.) |
//! | `UnaryLinkFactor` | e_i | Entity linking features (prior probability, name match, etc.) |
//!
//! ## Pairwise Cross-Task Factors
//!
//! These encode dependencies **between** different tasks:
//!
//! | Factor | Variables | What It Encodes |
//! |--------|-----------|-----------------|
//! | `LinkNerFactor` | (e_i, t_i) | Wikipedia type should match NER type |
//! | `CorefNerFactor` | (a_i, t_i) | Coreferent mentions should have same type |
//! | `CorefLinkFactor` | (a_i, e_i) | Coreferent mentions should link to same/related entities |
//!
//! # Example: LinkNer Factor
//!
//! If mention m links to Wikipedia page "Barack Obama" (a Person), and the NER
//! system types m as PERSON, the `LinkNerFactor` assigns high score. If NER
//! types m as ORG, the factor assigns low score (penalty).
//!
//! ```text
//! LinkNerFactor(e="Barack Obama", t=PERSON) → +3.0 (type match)
//! LinkNerFactor(e="Barack Obama", t=ORG)    → -2.0 (type mismatch)
//! ```
//!
//! # Weight Learning
//!
//! Factor weights are learned from annotated data using structured perceptron
//! or max-margin methods. See [`super::learning`] for training.
//!
//! # References
//!
//! - Durrett & Klein (2014): "A Joint Model for Entity Analysis" (TACL)
//! - Kschischang et al. (2001): Factor Graphs and the Sum-Product Algorithm

use super::types::{AntecedentValue, Assignment, LinkValue, VariableId, VariableType};
use crate::linking::wikidata::{WikidataDictionary, WikidataNERType};
use crate::EntityType;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

// =============================================================================
// Factor Trait
// =============================================================================

/// A factor (potential function) in the structured CRF.
///
/// Factors define the functions ψ(·) that encode soft constraints and
/// dependencies between variables. The joint distribution over all
/// variables is proportional to the product of all factor potentials:
///
/// ```text
/// P(a,t,e|x) ∝ ∏_f ψ_f(vars_f, x)
/// ```
///
/// where each factor ψ_f depends on a subset of variables (its **scope**).
///
/// # Implementation Notes
///
/// - Factors work in **log space** for numerical stability
/// - `log_potential` should return `-f64::INFINITY` for impossible configurations
/// - Factors must be `Send + Sync` for parallel inference
///
/// # Example
///
/// A simple type-consistency factor that prefers matching types:
///
/// ```ignore
/// impl Factor for TypeConsistencyFactor {
///     fn log_potential(&self, assignment: &Assignment) -> f64 {
///         let type_i = assignment.get(&self.var_type_i);
///         let type_j = assignment.get(&self.var_type_j);
///         if type_i == type_j {
///             self.match_weight // positive bonus
///         } else {
///             self.mismatch_penalty // negative penalty
///         }
///     }
/// }
/// ```
pub trait Factor: Send + Sync {
    /// The variables this factor touches (its "scope").
    ///
    /// A unary factor has scope of size 1.
    /// A pairwise factor has scope of size 2.
    /// Higher-order factors have larger scopes (but increase inference cost).
    fn scope(&self) -> &[VariableId];

    /// Log potential (unnormalized log probability) for an assignment.
    ///
    /// Returns the score θᵀφ(assignment) where θ are learned weights and
    /// φ are feature functions. Higher scores indicate more likely configurations.
    ///
    /// Should return `-f64::INFINITY` for logically impossible configurations.
    fn log_potential(&self, assignment: &Assignment) -> f64;

    /// Human-readable factor name for debugging and visualization.
    fn name(&self) -> &'static str;
}

// =============================================================================
// Unary Factors
// =============================================================================

/// Unary factor for coreference (antecedent selection).
///
/// Features from mention-ranking model: string match, distance, type compatibility.
#[derive(Debug, Clone)]
pub struct UnaryCorefFactor {
    /// Mention index
    pub mention_idx: usize,
    /// Variable ID
    scope: Vec<VariableId>,
    /// Precomputed scores for each antecedent candidate
    pub scores: Vec<(AntecedentValue, f64)>,
}

impl UnaryCorefFactor {
    /// Create a new unary coreference factor.
    pub fn new(mention_idx: usize, scores: Vec<(AntecedentValue, f64)>) -> Self {
        let scope = vec![VariableId {
            mention_idx,
            var_type: VariableType::Antecedent,
        }];
        Self {
            mention_idx,
            scope,
            scores,
        }
    }
}

impl Factor for UnaryCorefFactor {
    fn scope(&self) -> &[VariableId] {
        &self.scope
    }

    fn log_potential(&self, assignment: &Assignment) -> f64 {
        let antecedent = assignment.get_antecedent(self.mention_idx);
        antecedent
            .and_then(|a| self.scores.iter().find(|(v, _)| *v == a).map(|(_, s)| *s))
            .unwrap_or(f64::NEG_INFINITY)
    }

    fn name(&self) -> &'static str {
        "unary_coref"
    }
}

/// Unary factor for NER (semantic typing).
///
/// Features: token features, gazetteer matches, context.
#[derive(Debug, Clone)]
pub struct UnaryNerFactor {
    /// Mention index
    pub mention_idx: usize,
    /// Variable ID
    scope: Vec<VariableId>,
    /// Scores for each entity type
    pub scores: Vec<(EntityType, f64)>,
}

impl UnaryNerFactor {
    /// Create a new unary NER factor.
    pub fn new(mention_idx: usize, scores: Vec<(EntityType, f64)>) -> Self {
        let scope = vec![VariableId {
            mention_idx,
            var_type: VariableType::SemanticType,
        }];
        Self {
            mention_idx,
            scope,
            scores,
        }
    }
}

impl Factor for UnaryNerFactor {
    fn scope(&self) -> &[VariableId] {
        &self.scope
    }

    fn log_potential(&self, assignment: &Assignment) -> f64 {
        let entity_type = assignment.get_type(self.mention_idx);
        entity_type
            .and_then(|t| self.scores.iter().find(|(v, _)| *v == t).map(|(_, s)| *s))
            .unwrap_or(f64::NEG_INFINITY)
    }

    fn name(&self) -> &'static str {
        "unary_ner"
    }
}

/// Unary factor for entity linking.
///
/// Features: string match, prior probability, context similarity.
#[derive(Debug, Clone)]
pub struct UnaryLinkFactor {
    /// Mention index
    pub mention_idx: usize,
    /// Variable ID
    scope: Vec<VariableId>,
    /// Scores for each link candidate
    pub scores: Vec<(LinkValue, f64)>,
}

impl UnaryLinkFactor {
    /// Create a new unary link factor.
    pub fn new(mention_idx: usize, scores: Vec<(LinkValue, f64)>) -> Self {
        let scope = vec![VariableId {
            mention_idx,
            var_type: VariableType::EntityLink,
        }];
        Self {
            mention_idx,
            scope,
            scores,
        }
    }
}

impl Factor for UnaryLinkFactor {
    fn scope(&self) -> &[VariableId] {
        &self.scope
    }

    fn log_potential(&self, assignment: &Assignment) -> f64 {
        let link = assignment.get_link(self.mention_idx);
        link.and_then(|l| self.scores.iter().find(|(v, _)| v == l).map(|(_, s)| *s))
            .unwrap_or(f64::NEG_INFINITY)
    }

    fn name(&self) -> &'static str {
        "unary_link"
    }
}

// =============================================================================
// Wikipedia Knowledge Store (for cross-task factors)
// =============================================================================

/// Store of Wikipedia/Wikidata knowledge for joint inference.
///
/// Provides semantic information about entities:
/// - Type mappings (Q5 → Person, Q43229 → Organization)
/// - Outgoing links (for relatedness computation)
/// - Categories
#[derive(Debug, Clone, Default)]
pub struct WikipediaKnowledgeStore {
    /// Entity types by KB ID
    pub entity_types: HashMap<String, WikidataNERType>,
    /// Outgoing links by KB ID
    pub outlinks: HashMap<String, HashSet<String>>,
    /// Categories by KB ID
    pub categories: HashMap<String, Vec<String>>,
    /// Wikidata dictionary for type lookups
    pub dictionary: Option<Arc<WikidataDictionary>>,
}

impl WikipediaKnowledgeStore {
    /// Create a new empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create from a Wikidata dictionary.
    pub fn from_dictionary(dict: WikidataDictionary) -> Self {
        Self {
            dictionary: Some(Arc::new(dict)),
            ..Default::default()
        }
    }

    /// Get the entity type for a KB ID.
    pub fn get_type(&self, kb_id: &str) -> Option<WikidataNERType> {
        // First check explicit mappings
        if let Some(t) = self.entity_types.get(kb_id) {
            return Some(*t);
        }

        // Then check dictionary
        if let Some(ref dict) = self.dictionary {
            if let Some(entity) = dict.get(kb_id) {
                return entity.entity_type;
            }
        }

        None
    }

    /// Add a type mapping.
    pub fn add_type(&mut self, kb_id: &str, ner_type: WikidataNERType) {
        self.entity_types.insert(kb_id.to_string(), ner_type);
    }

    /// Add outgoing links for an entity.
    pub fn add_outlinks(&mut self, kb_id: &str, links: impl IntoIterator<Item = String>) {
        self.outlinks
            .entry(kb_id.to_string())
            .or_default()
            .extend(links);
    }

    /// Check if two entities share outlinks.
    pub fn shared_outlinks(&self, kb_id_a: &str, kb_id_b: &str) -> usize {
        let links_a = self.outlinks.get(kb_id_a);
        let links_b = self.outlinks.get(kb_id_b);

        match (links_a, links_b) {
            (Some(a), Some(b)) => a.intersection(b).count(),
            _ => 0,
        }
    }

    /// Check if one entity links to another.
    pub fn has_link(&self, from: &str, to: &str) -> bool {
        self.outlinks
            .get(from)
            .is_some_and(|links| links.contains(to))
    }

    /// Check if entities mutually link to each other.
    pub fn mutual_link(&self, kb_id_a: &str, kb_id_b: &str) -> bool {
        self.has_link(kb_id_a, kb_id_b) || self.has_link(kb_id_b, kb_id_a)
    }

    /// Compute relatedness score between two entities.
    ///
    /// Based on Wikipedia link structure (Milne & Witten 2008 style).
    pub fn relatedness(&self, kb_id_a: &str, kb_id_b: &str) -> f64 {
        if kb_id_a == kb_id_b {
            return 1.0;
        }

        let shared = self.shared_outlinks(kb_id_a, kb_id_b) as f64;
        let mutual = if self.mutual_link(kb_id_a, kb_id_b) {
            1.0
        } else {
            0.0
        };

        // Simple relatedness: normalized shared links + bonus for mutual
        let links_a = self.outlinks.get(kb_id_a).map_or(0, |l| l.len()) as f64;
        let links_b = self.outlinks.get(kb_id_b).map_or(0, |l| l.len()) as f64;

        if links_a + links_b == 0.0 {
            return mutual * 0.5;
        }

        let jaccard = shared / (links_a + links_b - shared).max(1.0);
        (jaccard + mutual * 0.3).min(1.0)
    }
}

// =============================================================================
// Cross-Task Factors
// =============================================================================

/// Factor coupling entity linking and NER.
///
/// Uses Wikipedia article semantics to inform NER type:
/// - Infobox type (e.g., "company" → ORGANIZATION)
/// - Categories (e.g., "American politicians" → PERSON)
/// - First sentence copula (e.g., "is a British city" → LOCATION)
///
/// # Example
///
/// If mention links to `Dell` (Wikipedia):
/// - Infobox type: company
/// - Categories: Computer companies, Technology companies
/// - → Strong signal for ORGANIZATION type
#[derive(Debug, Clone)]
pub struct LinkNerFactor {
    /// Mention index
    pub mention_idx: usize,
    /// Variable IDs (type and link for same mention)
    scope: Vec<VariableId>,
    /// Weights for type-link compatibility
    pub weights: LinkNerWeights,
    /// Knowledge store for lookups
    knowledge: Option<Arc<WikipediaKnowledgeStore>>,
}

/// Weights for Link+NER factor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkNerWeights {
    /// Weight for infobox/Wikidata type match
    pub type_match: f64,
    /// Weight for type mismatch (penalty, should be negative)
    pub type_mismatch: f64,
    /// Weight for category match
    pub category_match: f64,
    /// Weight for NIL link (no entity in KB)
    pub nil_bonus: f64,
}

impl Default for LinkNerWeights {
    fn default() -> Self {
        Self {
            type_match: 1.5,
            type_mismatch: -1.0,
            category_match: 0.5,
            nil_bonus: 0.0,
        }
    }
}

impl LinkNerFactor {
    /// Create a new Link+NER factor.
    pub fn new(mention_idx: usize, weights: LinkNerWeights) -> Self {
        let scope = vec![
            VariableId {
                mention_idx,
                var_type: VariableType::SemanticType,
            },
            VariableId {
                mention_idx,
                var_type: VariableType::EntityLink,
            },
        ];
        Self {
            mention_idx,
            scope,
            weights,
            knowledge: None,
        }
    }

    /// Set knowledge store.
    pub fn with_knowledge(mut self, knowledge: Arc<WikipediaKnowledgeStore>) -> Self {
        self.knowledge = Some(knowledge);
        self
    }

    /// Check if NER type matches Wikidata type.
    fn types_compatible(ner_type: &EntityType, wiki_type: WikidataNERType) -> bool {
        match wiki_type {
            WikidataNERType::Person => matches!(ner_type, EntityType::Person),
            WikidataNERType::Organization => matches!(ner_type, EntityType::Organization),
            WikidataNERType::Location | WikidataNERType::GeopoliticalEntity => {
                matches!(ner_type, EntityType::Location)
                    || matches!(ner_type, EntityType::Other(ref s) if s == "GPE")
            }
            WikidataNERType::Event => {
                matches!(ner_type, EntityType::Other(ref s) if s == "EVENT")
            }
            WikidataNERType::WorkOfArt => {
                matches!(ner_type, EntityType::Other(ref s) if s == "WORK_OF_ART")
            }
            WikidataNERType::Product => {
                matches!(ner_type, EntityType::Other(ref s) if s == "PRODUCT")
            }
            WikidataNERType::DateTime => {
                matches!(ner_type, EntityType::Other(ref s) if s == "DATE")
            }
            WikidataNERType::Miscellaneous => true, // MISC compatible with anything
        }
    }
}

impl Factor for LinkNerFactor {
    fn scope(&self) -> &[VariableId] {
        &self.scope
    }

    fn log_potential(&self, assignment: &Assignment) -> f64 {
        let entity_type = match assignment.get_type(self.mention_idx) {
            Some(t) => t,
            None => return 0.0,
        };

        let link = match assignment.get_link(self.mention_idx) {
            Some(l) => l,
            None => return 0.0,
        };

        // NIL links get a small bonus (or penalty depending on config)
        let kb_id = match link {
            LinkValue::KbId(id) => id,
            LinkValue::Nil => return self.weights.nil_bonus,
        };

        // Look up entity type from knowledge store
        let wiki_type = self.knowledge.as_ref().and_then(|k| k.get_type(kb_id));

        match wiki_type {
            Some(wt) => {
                if Self::types_compatible(&entity_type, wt) {
                    self.weights.type_match
                } else {
                    self.weights.type_mismatch
                }
            }
            None => 0.0, // No type info available
        }
    }

    fn name(&self) -> &'static str {
        "link_ner"
    }
}

/// Factor coupling coreference and NER.
///
/// Encourages consistent semantic types across coreference chains.
/// Only fires when mention i is linked to mention j (a_i = j).
///
/// # Features
///
/// - Type pair: (t_i, t_j) indicator features
/// - Monolexical: (t_i, head_j) and (t_j, head_i) features
///
/// # Example
///
/// If "he" → "John Smith" is a coref link:
/// - t("John Smith") = PERSON
/// - t("he") = PERSON
/// - Factor score high for matching types
#[derive(Debug, Clone)]
pub struct CorefNerFactor {
    /// Current mention index (i)
    pub mention_i: usize,
    /// Antecedent mention index (j)
    pub mention_j: usize,
    /// Variable IDs
    scope: Vec<VariableId>,
    /// Weights
    pub weights: CorefNerWeights,
    /// Head word of mention i for monolexical features.
    pub head_i: Option<String>,
    /// Head word of mention j for monolexical features.
    pub head_j: Option<String>,
    /// Monolexical feature lookup: (type, head) → weight
    pub monolexical_weights: HashMap<(String, String), f64>,
}

/// Weights for Coref+NER factor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorefNerWeights {
    /// Weight for same-type bonus
    pub type_match: f64,
    /// Weight for type mismatch penalty
    pub type_mismatch: f64,
    /// Weight for monolexical features (type + head word)
    pub monolexical: f64,
    /// Special weight for pronoun → proper name with matching type
    pub pronoun_proper_match: f64,
}

impl Default for CorefNerWeights {
    fn default() -> Self {
        Self {
            type_match: 1.0,
            type_mismatch: -0.5,
            monolexical: 0.3,
            pronoun_proper_match: 0.5,
        }
    }
}

impl CorefNerFactor {
    /// Create a new Coref+NER factor.
    pub fn new(mention_i: usize, mention_j: usize, weights: CorefNerWeights) -> Self {
        let scope = vec![
            VariableId {
                mention_idx: mention_i,
                var_type: VariableType::Antecedent,
            },
            VariableId {
                mention_idx: mention_i,
                var_type: VariableType::SemanticType,
            },
            VariableId {
                mention_idx: mention_j,
                var_type: VariableType::SemanticType,
            },
        ];
        Self {
            mention_i,
            mention_j,
            scope,
            weights,
            head_i: None,
            head_j: None,
            monolexical_weights: HashMap::new(),
        }
    }

    /// Set head words for monolexical features.
    pub fn with_heads(mut self, head_i: &str, head_j: &str) -> Self {
        self.head_i = Some(head_i.to_lowercase());
        self.head_j = Some(head_j.to_lowercase());
        self
    }

    /// Add monolexical weight.
    pub fn add_monolexical_weight(&mut self, type_name: &str, head: &str, weight: f64) {
        self.monolexical_weights
            .insert((type_name.to_string(), head.to_lowercase()), weight);
    }

    /// Load default monolexical weights (common patterns).
    pub fn with_default_monolexical(mut self) -> Self {
        // Person type + common person head words
        for head in ["mr", "mrs", "dr", "prof", "president", "ceo", "chairman"] {
            self.add_monolexical_weight("Person", head, 0.5);
        }

        // Organization type + common org head words
        for head in [
            "company",
            "corporation",
            "inc",
            "corp",
            "llc",
            "firm",
            "bank",
        ] {
            self.add_monolexical_weight("Organization", head, 0.5);
        }

        // Location type + common location head words
        for head in [
            "city", "country", "state", "province", "region", "river", "mountain",
        ] {
            self.add_monolexical_weight("Location", head, 0.5);
        }

        self
    }
}

impl Factor for CorefNerFactor {
    fn scope(&self) -> &[VariableId] {
        &self.scope
    }

    fn log_potential(&self, assignment: &Assignment) -> f64 {
        // Only fires if i→j is a coreference link
        let antecedent = assignment.get_antecedent(self.mention_i);
        if antecedent != Some(AntecedentValue::Mention(self.mention_j)) {
            return 0.0; // Factor doesn't contribute
        }

        let type_i = match assignment.get_type(self.mention_i) {
            Some(t) => t,
            None => return 0.0,
        };

        let type_j = match assignment.get_type(self.mention_j) {
            Some(t) => t,
            None => return 0.0,
        };

        let mut score = 0.0;

        // Type pair feature
        if type_i == type_j {
            score += self.weights.type_match;
        } else {
            score += self.weights.type_mismatch;
        }

        // Monolexical features: (type_i, head_j)
        if let Some(ref head_j) = self.head_j {
            let type_name = format!("{:?}", type_i);
            if let Some(&w) = self
                .monolexical_weights
                .get(&(type_name.clone(), head_j.clone()))
            {
                score += self.weights.monolexical * w;
            }
            // Also check simplified type name
            let simple_type = match type_i {
                EntityType::Person => "Person",
                EntityType::Organization => "Organization",
                EntityType::Location => "Location",
                EntityType::Other(ref s) => s.as_str(),
                _ => "Unknown",
            };
            if let Some(&w) = self
                .monolexical_weights
                .get(&(simple_type.to_string(), head_j.clone()))
            {
                score += self.weights.monolexical * w;
            }
        }

        // Monolexical features: (type_j, head_i)
        if let Some(ref head_i) = self.head_i {
            let type_name = format!("{:?}", type_j);
            if let Some(&w) = self
                .monolexical_weights
                .get(&(type_name.clone(), head_i.clone()))
            {
                score += self.weights.monolexical * w;
            }
            let simple_type = match type_j {
                EntityType::Person => "Person",
                EntityType::Organization => "Organization",
                EntityType::Location => "Location",
                EntityType::Other(ref s) => s.as_str(),
                _ => "Unknown",
            };
            if let Some(&w) = self
                .monolexical_weights
                .get(&(simple_type.to_string(), head_i.clone()))
            {
                score += self.weights.monolexical * w;
            }
        }

        score
    }

    fn name(&self) -> &'static str {
        "coref_ner"
    }
}

/// Factor coupling coreference and entity linking.
///
/// Encourages coreferent mentions to link to related Wikipedia articles.
/// Only fires when mention i is linked to mention j (a_i = j).
///
/// # Features
///
/// - Same title: e_i = e_j (same article)
/// - Shared outlinks: articles share outgoing links
/// - Mutual links: one article links to the other
///
/// # Example
///
/// If "the company" → "Dell" is a coref link:
/// - e("Dell") = Dell (company article)
/// - e("the company") = Dell (should link to same)
/// - Factor: high score for same entity
#[derive(Debug, Clone)]
pub struct CorefLinkFactor {
    /// Current mention index (i)
    pub mention_i: usize,
    /// Antecedent mention index (j)
    pub mention_j: usize,
    /// Variable IDs
    scope: Vec<VariableId>,
    /// Weights
    pub weights: CorefLinkWeights,
    /// Knowledge store for Wikipedia graph
    knowledge: Option<Arc<WikipediaKnowledgeStore>>,
}

/// Weights for Coref+Link factor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorefLinkWeights {
    /// Weight for same entity (alias for same_title for learning.rs compatibility)
    pub same_entity: f64,
    /// Weight for different entity (negative, penalty for coreferent mentions with different links)
    pub different_entity: f64,
    /// Weight for same title (kept for backward compatibility)
    pub same_title: f64,
    /// Weight for shared outlinks (scaled by count)
    pub shared_outlinks: f64,
    /// Weight for mutual links
    pub mutual_link: f64,
    /// Weight for both being NIL
    pub both_nil: f64,
    /// Penalty for one NIL one not
    pub nil_mismatch: f64,
}

impl Default for CorefLinkWeights {
    fn default() -> Self {
        Self {
            same_entity: 2.0,       // Same as same_title
            different_entity: -0.3, // Penalty for different links
            same_title: 2.0,        // Backward compat
            shared_outlinks: 0.1,
            mutual_link: 1.0,
            both_nil: 0.5,
            nil_mismatch: -0.3,
        }
    }
}

impl CorefLinkFactor {
    /// Create a new Coref+Link factor.
    pub fn new(mention_i: usize, mention_j: usize, weights: CorefLinkWeights) -> Self {
        let scope = vec![
            VariableId {
                mention_idx: mention_i,
                var_type: VariableType::Antecedent,
            },
            VariableId {
                mention_idx: mention_i,
                var_type: VariableType::EntityLink,
            },
            VariableId {
                mention_idx: mention_j,
                var_type: VariableType::EntityLink,
            },
        ];
        Self {
            mention_i,
            mention_j,
            scope,
            weights,
            knowledge: None,
        }
    }

    /// Set knowledge store.
    pub fn with_knowledge(mut self, knowledge: Arc<WikipediaKnowledgeStore>) -> Self {
        self.knowledge = Some(knowledge);
        self
    }
}

impl Factor for CorefLinkFactor {
    fn scope(&self) -> &[VariableId] {
        &self.scope
    }

    fn log_potential(&self, assignment: &Assignment) -> f64 {
        // Only fires if i→j is a coreference link
        let antecedent = assignment.get_antecedent(self.mention_i);
        if antecedent != Some(AntecedentValue::Mention(self.mention_j)) {
            return 0.0;
        }

        let link_i = match assignment.get_link(self.mention_i) {
            Some(l) => l,
            None => return 0.0,
        };

        let link_j = match assignment.get_link(self.mention_j) {
            Some(l) => l,
            None => return 0.0,
        };

        let mut score = 0.0;

        // Handle NIL cases
        match (link_i, link_j) {
            (LinkValue::Nil, LinkValue::Nil) => {
                return self.weights.both_nil;
            }
            (LinkValue::Nil, _) | (_, LinkValue::Nil) => {
                return self.weights.nil_mismatch;
            }
            (LinkValue::KbId(id_i), LinkValue::KbId(id_j)) => {
                // Same entity feature
                if id_i == id_j {
                    score += self.weights.same_entity;
                    return score; // Same entity, no need for relatedness
                }

                // Wikipedia graph features
                if let Some(ref knowledge) = self.knowledge {
                    // Shared outlinks
                    let shared = knowledge.shared_outlinks(id_i, id_j);
                    score += self.weights.shared_outlinks * (shared as f64).ln().max(0.0);

                    // Mutual links
                    if knowledge.mutual_link(id_i, id_j) {
                        score += self.weights.mutual_link;
                    }
                }
            }
        }

        score
    }

    fn name(&self) -> &'static str {
        "coref_link"
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unary_coref_factor() {
        let scores = vec![
            (AntecedentValue::NewCluster, -1.0),
            (AntecedentValue::Mention(0), 0.5),
        ];
        let factor = UnaryCorefFactor::new(1, scores);

        let mut assignment = Assignment::default();
        assignment.set_antecedent(1, AntecedentValue::Mention(0));

        let score = factor.log_potential(&assignment);
        assert!((score - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_coref_ner_factor_same_type() {
        let factor = CorefNerFactor::new(1, 0, CorefNerWeights::default());

        let mut assignment = Assignment::default();
        assignment.set_antecedent(1, AntecedentValue::Mention(0));
        assignment.set_type(0, EntityType::Person);
        assignment.set_type(1, EntityType::Person);

        let score = factor.log_potential(&assignment);
        assert!(score > 0.0, "Same types should have positive score");
    }

    #[test]
    fn test_coref_ner_factor_different_type() {
        let factor = CorefNerFactor::new(1, 0, CorefNerWeights::default());

        let mut assignment = Assignment::default();
        assignment.set_antecedent(1, AntecedentValue::Mention(0));
        assignment.set_type(0, EntityType::Person);
        assignment.set_type(1, EntityType::Organization);

        let score = factor.log_potential(&assignment);
        assert!(score < 0.0, "Different types should have penalty");
    }

    #[test]
    fn test_coref_ner_factor_no_link() {
        let factor = CorefNerFactor::new(1, 0, CorefNerWeights::default());

        let mut assignment = Assignment::default();
        assignment.set_antecedent(1, AntecedentValue::NewCluster); // NOT linked
        assignment.set_type(0, EntityType::Person);
        assignment.set_type(1, EntityType::Organization);

        let score = factor.log_potential(&assignment);
        assert!(
            (score - 0.0).abs() < 1e-6,
            "Factor should not fire when not linked"
        );
    }

    #[test]
    fn test_coref_ner_factor_with_heads() {
        let factor = CorefNerFactor::new(1, 0, CorefNerWeights::default())
            .with_heads("he", "president")
            .with_default_monolexical();

        let mut assignment = Assignment::default();
        assignment.set_antecedent(1, AntecedentValue::Mention(0));
        assignment.set_type(0, EntityType::Person);
        assignment.set_type(1, EntityType::Person);

        let score = factor.log_potential(&assignment);
        // Should include monolexical bonus for "president" + Person
        assert!(score > 1.0, "Should have monolexical bonus");
    }

    #[test]
    fn test_coref_link_factor_same_title() {
        let factor = CorefLinkFactor::new(1, 0, CorefLinkWeights::default());

        let mut assignment = Assignment::default();
        assignment.set_antecedent(1, AntecedentValue::Mention(0));
        assignment.set_link(0, LinkValue::KbId("Q42".to_string()));
        assignment.set_link(1, LinkValue::KbId("Q42".to_string()));

        let score = factor.log_potential(&assignment);
        assert!(score > 0.0, "Same title should have high score");
    }

    #[test]
    fn test_coref_link_factor_with_knowledge() {
        let mut knowledge = WikipediaKnowledgeStore::new();
        knowledge.add_outlinks("Q1", vec!["Q2".to_string(), "Q3".to_string()]);
        knowledge.add_outlinks("Q2", vec!["Q1".to_string(), "Q3".to_string()]);

        let factor = CorefLinkFactor::new(1, 0, CorefLinkWeights::default())
            .with_knowledge(Arc::new(knowledge));

        let mut assignment = Assignment::default();
        assignment.set_antecedent(1, AntecedentValue::Mention(0));
        assignment.set_link(0, LinkValue::KbId("Q1".to_string()));
        assignment.set_link(1, LinkValue::KbId("Q2".to_string()));

        let score = factor.log_potential(&assignment);
        // Should have positive score from mutual links and shared outlinks
        assert!(score > 0.0, "Related entities should have positive score");
    }

    #[test]
    fn test_link_ner_factor_type_match() {
        let mut knowledge = WikipediaKnowledgeStore::new();
        knowledge.add_type("Q937", WikidataNERType::Person);

        let factor =
            LinkNerFactor::new(0, LinkNerWeights::default()).with_knowledge(Arc::new(knowledge));

        let mut assignment = Assignment::default();
        assignment.set_type(0, EntityType::Person);
        assignment.set_link(0, LinkValue::KbId("Q937".to_string()));

        let score = factor.log_potential(&assignment);
        assert!(score > 0.0, "Type match should have positive score");
    }

    #[test]
    fn test_link_ner_factor_type_mismatch() {
        let mut knowledge = WikipediaKnowledgeStore::new();
        knowledge.add_type("Q937", WikidataNERType::Person);

        let factor =
            LinkNerFactor::new(0, LinkNerWeights::default()).with_knowledge(Arc::new(knowledge));

        let mut assignment = Assignment::default();
        assignment.set_type(0, EntityType::Organization); // Mismatch!
        assignment.set_link(0, LinkValue::KbId("Q937".to_string()));

        let score = factor.log_potential(&assignment);
        assert!(score < 0.0, "Type mismatch should have negative score");
    }

    #[test]
    fn test_link_ner_factor_nil() {
        let factor = LinkNerFactor::new(0, LinkNerWeights::default());

        let mut assignment = Assignment::default();
        assignment.set_type(0, EntityType::Person);
        assignment.set_link(0, LinkValue::Nil);

        let score = factor.log_potential(&assignment);
        // NIL should return nil_bonus (default 0)
        assert!((score - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_wikipedia_knowledge_store_relatedness() {
        let mut knowledge = WikipediaKnowledgeStore::new();
        knowledge.add_outlinks("A", vec!["C".to_string(), "D".to_string()]);
        knowledge.add_outlinks("B", vec!["C".to_string(), "E".to_string()]);

        // A and B share outlink to C
        let shared = knowledge.shared_outlinks("A", "B");
        assert_eq!(shared, 1);

        // Self-relatedness is 1.0
        let self_rel = knowledge.relatedness("A", "A");
        assert!((self_rel - 1.0).abs() < 1e-10);

        // A-B have some relatedness
        let rel = knowledge.relatedness("A", "B");
        assert!(rel > 0.0);
    }

    #[test]
    fn test_types_compatible() {
        assert!(LinkNerFactor::types_compatible(
            &EntityType::Person,
            WikidataNERType::Person
        ));
        assert!(LinkNerFactor::types_compatible(
            &EntityType::Organization,
            WikidataNERType::Organization
        ));
        assert!(LinkNerFactor::types_compatible(
            &EntityType::Location,
            WikidataNERType::Location
        ));
        assert!(!LinkNerFactor::types_compatible(
            &EntityType::Person,
            WikidataNERType::Organization
        ));
    }
}
