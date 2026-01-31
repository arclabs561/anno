//! Coreference resolution data structures.
//!
//! Provides types for representing coreference chains (clusters of mentions
//! that refer to the same entity) and utilities for working with them.
//!
//! # Terminology
//!
//! - **Mention**: A span of text referring to an entity (e.g., "John", "he", "the CEO")
//! - **Chain/Cluster**: A set of mentions that corefer (refer to the same entity)
//! - **Singleton**: A chain with only one mention (entity mentioned only once)
//! - **Antecedent**: An earlier mention that a pronoun/noun phrase refers to
//!
//! # Example
//!
//! ```rust
//! use anno_core::core::coref::{Mention, CorefChain, CorefDocument};
//!
//! // "John went to the store. He bought milk."
//! let john = Mention::new("John", 0, 4);
//! let he = Mention::new("He", 25, 27);
//!
//! let chain = CorefChain::new(vec![john, he]);
//! assert_eq!(chain.len(), 2);
//! assert!(!chain.is_singleton());
//! ```

use super::Entity;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// Re-export MentionType for convenience
pub use super::types::MentionType;

// =============================================================================
// Mention
// =============================================================================

/// A single mention (text span) that may corefer with other mentions.
///
/// Mentions are comparable by span position, not by text content.
/// Two mentions with identical text at different positions are distinct.
///
/// # Character vs Byte Offsets
///
/// `start` and `end` are **character** offsets, not byte offsets.
/// For "北京 Beijing", the character offsets are:
/// - "北" = 0..1 (but 3 bytes in UTF-8)
/// - "京" = 1..2 (but 3 bytes)
/// - " " = 2..3
/// - "Beijing" = 3..10
///
/// Use `text.chars().skip(start).take(end - start).collect()` to extract.
///
/// # Head Span
///
/// The `head_start`/`head_end` fields mark the syntactic head for head-match
/// evaluation (used in CEAF-e, LEA metrics). In "the former president of France",
/// the head is "president" - the noun that determines agreement.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Mention {
    /// The mention text (surface form).
    pub text: String,
    /// Start character offset (inclusive, 0-indexed).
    pub start: usize,
    /// End character offset (exclusive).
    pub end: usize,
    /// Head word start (for head-match metrics like CEAF).
    pub head_start: Option<usize>,
    /// Head word end.
    pub head_end: Option<usize>,
    /// Entity type if known (e.g., "PER", "ORG").
    pub entity_type: Option<String>,
    /// Mention category: Pronominal, Proper, Nominal, Zero.
    pub mention_type: Option<MentionType>,
}

impl Mention {
    /// `Mention::new("John", 0, 4)` creates a mention for "John" at characters 0..4.
    ///
    /// Offsets are character positions, not byte positions.
    #[must_use]
    pub fn new(text: impl Into<String>, start: usize, end: usize) -> Self {
        Self {
            text: text.into(),
            start,
            end,
            head_start: None,
            head_end: None,
            entity_type: None,
            mention_type: None,
        }
    }

    /// Mention with head span for head-match evaluation.
    ///
    /// The head is the syntactic nucleus: in "the former president", head is "president".
    ///
    /// ```
    /// # use anno_core::core::coref::Mention;
    /// let m = Mention::with_head("the former president", 0, 20, 11, 20);
    /// assert_eq!(m.head_start, Some(11)); // "president" starts at 11
    /// ```
    #[must_use]
    pub fn with_head(
        text: impl Into<String>,
        start: usize,
        end: usize,
        head_start: usize,
        head_end: usize,
    ) -> Self {
        Self {
            text: text.into(),
            start,
            end,
            head_start: Some(head_start),
            head_end: Some(head_end),
            entity_type: None,
            mention_type: None,
        }
    }

    /// Mention with type annotation for type-aware evaluation.
    ///
    /// ```
    /// # use anno_core::core::coref::Mention;
    /// # use anno_core::core::types::MentionType;
    /// let pronoun = Mention::with_type("he", 25, 27, MentionType::Pronominal);
    /// let proper = Mention::with_type("John Smith", 0, 10, MentionType::Proper);
    /// ```
    #[must_use]
    pub fn with_type(
        text: impl Into<String>,
        start: usize,
        end: usize,
        mention_type: MentionType,
    ) -> Self {
        Self {
            text: text.into(),
            start,
            end,
            head_start: None,
            head_end: None,
            entity_type: None,
            mention_type: Some(mention_type),
        }
    }

    /// True if spans share any characters: `[0,5)` overlaps `[3,8)`.
    #[must_use]
    pub fn overlaps(&self, other: &Mention) -> bool {
        self.start < other.end && other.start < self.end
    }

    /// True if spans are identical: same start AND end.
    #[must_use]
    pub fn span_matches(&self, other: &Mention) -> bool {
        self.start == other.start && self.end == other.end
    }

    /// Span length in characters. Returns 0 if `end <= start`.
    #[must_use]
    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    /// True if span has zero length.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// `(start, end)` tuple for use in hash sets and comparisons.
    #[must_use]
    pub fn span_id(&self) -> (usize, usize) {
        (self.start, self.end)
    }
}

impl std::fmt::Display for Mention {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "\"{}\" [{}-{})", self.text, self.start, self.end)
    }
}

// =============================================================================
// CorefChain (Cluster)
// =============================================================================

/// A coreference chain: mentions that all refer to the same entity.
///
/// ```
/// # use anno_core::core::coref::{CorefChain, Mention};
/// // "John went to the store. He bought milk."
/// //  ^^^^                    ^^
/// let john = Mention::new("John", 0, 4);
/// let he = Mention::new("He", 25, 27);
///
/// let chain = CorefChain::new(vec![john, he]);
/// assert_eq!(chain.len(), 2);
/// assert!(!chain.is_singleton());
/// ```
///
/// # Note
///
/// This type is for **evaluation and intermediate processing**. For production pipelines,
/// use [`Track`](super::grounded::Track) which integrates with the Signal/Track/Identity hierarchy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorefChain {
    /// Mentions in document order (sorted by start position).
    pub mentions: Vec<Mention>,
    /// Cluster ID from the source data, if any.
    pub cluster_id: Option<super::types::CanonicalId>,
    /// Entity type shared by all mentions (e.g., "PERSON").
    pub entity_type: Option<String>,
}

impl CorefChain {
    /// Build a chain from mentions. Sorts by position automatically.
    ///
    /// ```
    /// # use anno_core::core::coref::{CorefChain, Mention};
    /// let chain = CorefChain::new(vec![
    ///     Mention::new("she", 50, 53),
    ///     Mention::new("Dr. Smith", 0, 9),  // out of order
    /// ]);
    /// assert_eq!(chain.mentions[0].text, "Dr. Smith"); // sorted
    /// ```
    #[must_use]
    pub fn new(mut mentions: Vec<Mention>) -> Self {
        mentions.sort_by_key(|m| (m.start, m.end));
        Self {
            mentions,
            cluster_id: None,
            entity_type: None,
        }
    }

    /// Build a chain with an explicit cluster ID.
    #[must_use]
    pub fn with_id(
        mut mentions: Vec<Mention>,
        cluster_id: impl Into<super::types::CanonicalId>,
    ) -> Self {
        mentions.sort_by_key(|m| (m.start, m.end));
        Self {
            mentions,
            cluster_id: Some(cluster_id.into()),
            entity_type: None,
        }
    }

    /// A chain with exactly one mention (entity mentioned only once).
    #[must_use]
    pub fn singleton(mention: Mention) -> Self {
        Self {
            mentions: vec![mention],
            cluster_id: None,
            entity_type: None,
        }
    }

    /// Number of mentions. A chain with 3 mentions has 2 implicit "links".
    #[must_use]
    pub fn len(&self) -> usize {
        self.mentions.len()
    }

    /// True if chain has no mentions. Shouldn't happen in valid data.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.mentions.is_empty()
    }

    /// True if chain has exactly one mention (singleton entity).
    #[must_use]
    pub fn is_singleton(&self) -> bool {
        self.mentions.len() == 1
    }

    /// All pairwise links. For MUC: `n` mentions = `n*(n-1)/2` links.
    ///
    /// ```
    /// # use anno_core::core::coref::{CorefChain, Mention};
    /// let chain = CorefChain::new(vec![
    ///     Mention::new("A", 0, 1),
    ///     Mention::new("B", 2, 3),
    ///     Mention::new("C", 4, 5),
    /// ]);
    /// assert_eq!(chain.links().len(), 3); // A-B, A-C, B-C
    /// ```
    #[must_use]
    pub fn links(&self) -> Vec<(&Mention, &Mention)> {
        let mut links = Vec::new();
        for i in 0..self.mentions.len() {
            for j in (i + 1)..self.mentions.len() {
                links.push((&self.mentions[i], &self.mentions[j]));
            }
        }
        links
    }

    /// Number of coreference links.
    ///
    /// For a chain of n mentions: n*(n-1)/2 pairs, but only n-1 links needed
    /// to connect all mentions (spanning tree).
    #[must_use]
    pub fn link_count(&self) -> usize {
        if self.mentions.len() <= 1 {
            0
        } else {
            self.mentions.len() - 1
        }
    }

    /// Get all pairwise mention combinations (for B³, CEAF).
    #[must_use]
    pub fn all_pairs(&self) -> Vec<(&Mention, &Mention)> {
        self.links() // Same as links for non-directed pairs
    }

    /// Check if chain contains a mention with given span.
    #[must_use]
    pub fn contains_span(&self, start: usize, end: usize) -> bool {
        self.mentions
            .iter()
            .any(|m| m.start == start && m.end == end)
    }

    /// Get first mention (usually the most salient/representative).
    #[must_use]
    pub fn first(&self) -> Option<&Mention> {
        self.mentions.first()
    }

    /// Get set of mention span IDs for set operations.
    #[must_use]
    pub fn mention_spans(&self) -> HashSet<(usize, usize)> {
        self.mentions.iter().map(|m| m.span_id()).collect()
    }

    /// Get the canonical (representative) mention for this chain.
    ///
    /// Prefers proper nouns over other mention types, then longest mention.
    /// Falls back to first mention if no proper noun exists.
    #[must_use]
    pub fn canonical_mention(&self) -> Option<&Mention> {
        // Prefer proper noun mentions
        let proper = self
            .mentions
            .iter()
            .filter(|m| m.mention_type == Some(MentionType::Proper))
            .max_by_key(|m| m.text.len());

        if proper.is_some() {
            return proper;
        }

        // Fall back to longest mention (likely most informative)
        self.mentions.iter().max_by_key(|m| m.text.len())
    }

    /// Get the canonical ID for this chain (cluster_id if set).
    #[must_use]
    pub fn canonical_id(&self) -> Option<super::types::CanonicalId> {
        self.cluster_id
    }
}

impl std::fmt::Display for CorefChain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mentions: Vec<String> = self
            .mentions
            .iter()
            .map(|m| format!("\"{}\"", m.text))
            .collect();
        write!(f, "[{}]", mentions.join(", "))
    }
}

// =============================================================================
// CorefDocument
// =============================================================================

/// A document with coreference annotations.
///
/// Contains the source text and all coreference chains.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorefDocument {
    /// Document text.
    pub text: String,
    /// Document identifier.
    pub doc_id: Option<String>,
    /// Coreference chains (clusters).
    pub chains: Vec<CorefChain>,
    /// Whether singletons are included.
    pub includes_singletons: bool,
}

impl CorefDocument {
    /// Create a new document with chains.
    #[must_use]
    pub fn new(text: impl Into<String>, chains: Vec<CorefChain>) -> Self {
        Self {
            text: text.into(),
            doc_id: None,
            chains,
            includes_singletons: false,
        }
    }

    /// Create document with ID.
    #[must_use]
    pub fn with_id(
        text: impl Into<String>,
        doc_id: impl Into<String>,
        chains: Vec<CorefChain>,
    ) -> Self {
        Self {
            text: text.into(),
            doc_id: Some(doc_id.into()),
            chains,
            includes_singletons: false,
        }
    }

    /// Total number of mentions across all chains.
    #[must_use]
    pub fn mention_count(&self) -> usize {
        self.chains.iter().map(|c| c.len()).sum()
    }

    /// Number of chains (clusters).
    #[must_use]
    pub fn chain_count(&self) -> usize {
        self.chains.len()
    }

    /// Number of non-singleton chains.
    #[must_use]
    pub fn non_singleton_count(&self) -> usize {
        self.chains.iter().filter(|c| !c.is_singleton()).count()
    }

    /// Get all mentions in document order.
    #[must_use]
    pub fn all_mentions(&self) -> Vec<&Mention> {
        let mut mentions: Vec<&Mention> = self.chains.iter().flat_map(|c| &c.mentions).collect();
        mentions.sort_by_key(|m| (m.start, m.end));
        mentions
    }

    /// Find which chain contains a mention span.
    #[must_use]
    pub fn find_chain(&self, start: usize, end: usize) -> Option<&CorefChain> {
        self.chains.iter().find(|c| c.contains_span(start, end))
    }

    /// Build mention-to-chain index for fast lookup.
    #[must_use]
    pub fn mention_to_chain_index(&self) -> HashMap<(usize, usize), usize> {
        let mut index = HashMap::new();
        for (chain_idx, chain) in self.chains.iter().enumerate() {
            for mention in &chain.mentions {
                index.insert(mention.span_id(), chain_idx);
            }
        }
        index
    }

    /// Filter to only non-singleton chains.
    #[must_use]
    pub fn without_singletons(&self) -> Self {
        Self {
            text: self.text.clone(),
            doc_id: self.doc_id.clone(),
            chains: self
                .chains
                .iter()
                .filter(|c| !c.is_singleton())
                .cloned()
                .collect(),
            includes_singletons: false,
        }
    }
}

// =============================================================================
// Conversion from Entity to Mention
// =============================================================================

impl From<&Entity> for Mention {
    fn from(entity: &Entity) -> Self {
        Self {
            text: entity.text.clone(),
            start: entity.start,
            end: entity.end,
            head_start: None,
            head_end: None,
            entity_type: Some(entity.entity_type.as_label().to_string()),
            mention_type: None,
        }
    }
}

/// Convert entities with canonical_id to coreference chains.
///
/// Entities sharing the same `canonical_id` are grouped into a chain.
#[must_use]
pub fn entities_to_chains(entities: &[Entity]) -> Vec<CorefChain> {
    let mut clusters: HashMap<u64, Vec<Mention>> = HashMap::new();
    let mut singletons: Vec<Mention> = Vec::new();

    for entity in entities {
        let mention = Mention::from(entity);
        if let Some(canonical_id) = entity.canonical_id {
            clusters
                .entry(canonical_id.get())
                .or_default()
                .push(mention);
        } else {
            singletons.push(mention);
        }
    }

    let mut chains: Vec<CorefChain> = clusters
        .into_iter()
        .map(|(id, mentions)| CorefChain::with_id(mentions, id))
        .collect();

    // Add singletons as individual chains
    for mention in singletons {
        chains.push(CorefChain::singleton(mention));
    }

    chains
}

// =============================================================================
// CoreferenceResolver Trait
// =============================================================================

/// Trait for coreference resolution algorithms.
///
/// Implementors take a set of entity mentions and cluster them into
/// coreference chains (groups of mentions referring to the same entity).
///
/// # Design Philosophy
///
/// This trait lives in `anno::core` because:
/// 1. It depends only on core types (`Entity`, `CorefChain`)
/// 2. Multiple crates need to implement it (backends, eval)
/// 3. Keeping it here prevents circular dependencies
///
/// # Relationship to the Grounded Pipeline
///
/// `CoreferenceResolver` operates on the **evaluation/convenience layer** (`Entity`),
/// not the canonical **grounded pipeline** (`Signal` → `Track` → `Identity`).
///
/// | Layer | Type | `CoreferenceResolver` role |
/// |-------|------|----------------------------|
/// | Detection (L1) | `Entity` | Input: mentions to cluster |
/// | Coref (L2) | `Entity.canonical_id` | Output: cluster assignment |
/// | Linking (L3) | `Identity` | (not covered by this trait) |
///
/// For integration with `GroundedDocument`, use backends that produce
/// `Signal` + `Track` directly (e.g., `anno::backends::MentionRankingCoref`).
///
/// # Example Implementation
///
/// ```rust,ignore
/// use anno_core::{CoreferenceResolver, Entity, CorefChain};
///
/// struct ExactMatchResolver;
///
/// impl CoreferenceResolver for ExactMatchResolver {
///     fn resolve(&self, entities: &[Entity]) -> Vec<Entity> {
///         // Cluster entities with identical text
///         // ... implementation ...
///     }
///
///     fn name(&self) -> &'static str {
///         "exact-match"
///     }
/// }
/// ```
pub trait CoreferenceResolver: Send + Sync {
    /// Resolve coreference, assigning canonical IDs to entities.
    ///
    /// Each entity in the output will have a `canonical_id` field set.
    /// Entities with the same `canonical_id` are coreferent (refer to the
    /// same real-world entity).
    ///
    /// # Invariants
    ///
    /// - Every output entity has `canonical_id.is_some()`
    /// - Coreferent entities share the same `canonical_id`
    /// - Singleton mentions get unique `canonical_id` values
    fn resolve(&self, entities: &[Entity]) -> Vec<Entity>;

    /// Resolve directly to chains.
    ///
    /// A chain groups all mentions of the same entity together.
    /// This is often the desired output format for evaluation and
    /// downstream tasks.
    fn resolve_to_chains(&self, entities: &[Entity]) -> Vec<CorefChain> {
        let resolved = self.resolve(entities);
        entities_to_chains(&resolved)
    }

    /// Get resolver name.
    ///
    /// Used for logging, metrics, and result attribution.
    fn name(&self) -> &'static str;
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mention_creation() {
        let m = Mention::new("John", 0, 4);
        assert_eq!(m.text, "John");
        assert_eq!(m.start, 0);
        assert_eq!(m.end, 4);
        assert_eq!(m.len(), 4);
    }

    #[test]
    fn test_mention_overlap() {
        let m1 = Mention::new("John Smith", 0, 10);
        let m2 = Mention::new("Smith", 5, 10);
        let m3 = Mention::new("works", 11, 16);

        assert!(m1.overlaps(&m2));
        assert!(!m1.overlaps(&m3));
        assert!(!m2.overlaps(&m3));
    }

    #[test]
    fn test_chain_creation() {
        let mentions = vec![
            Mention::new("John", 0, 4),
            Mention::new("he", 20, 22),
            Mention::new("him", 40, 43),
        ];
        let chain = CorefChain::new(mentions);

        assert_eq!(chain.len(), 3);
        assert!(!chain.is_singleton());
        assert_eq!(chain.link_count(), 2); // Minimum links to connect
    }

    #[test]
    fn test_chain_links() {
        let mentions = vec![
            Mention::new("a", 0, 1),
            Mention::new("b", 2, 3),
            Mention::new("c", 4, 5),
        ];
        let chain = CorefChain::new(mentions);

        // All pairs: (a,b), (a,c), (b,c) = 3 pairs
        assert_eq!(chain.all_pairs().len(), 3);
    }

    #[test]
    fn test_singleton_chain() {
        let m = Mention::new("entity", 0, 6);
        let chain = CorefChain::singleton(m);

        assert!(chain.is_singleton());
        assert_eq!(chain.link_count(), 0);
        assert!(chain.all_pairs().is_empty());
    }

    #[test]
    fn test_document() {
        let text = "John went to the store. He bought milk.";
        let chain = CorefChain::new(vec![Mention::new("John", 0, 4), Mention::new("He", 24, 26)]);
        let doc = CorefDocument::new(text, vec![chain]);

        assert_eq!(doc.mention_count(), 2);
        assert_eq!(doc.chain_count(), 1);
        assert_eq!(doc.non_singleton_count(), 1);
    }

    #[test]
    fn test_mention_to_chain_index() {
        let chain1 = CorefChain::new(vec![Mention::new("John", 0, 4), Mention::new("he", 20, 22)]);
        let chain2 = CorefChain::new(vec![
            Mention::new("Mary", 5, 9),
            Mention::new("she", 30, 33),
        ]);
        let doc = CorefDocument::new("text", vec![chain1, chain2]);

        let index = doc.mention_to_chain_index();
        assert_eq!(index.get(&(0, 4)), Some(&0));
        assert_eq!(index.get(&(20, 22)), Some(&0));
        assert_eq!(index.get(&(5, 9)), Some(&1));
        assert_eq!(index.get(&(30, 33)), Some(&1));
    }
}
