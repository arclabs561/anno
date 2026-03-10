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
    ///
    /// ```
    /// use anno_core::Mention;
    ///
    /// let m = Mention::new("John", 0, 4);
    /// assert_eq!(m.text, "John");
    /// assert_eq!(m.len(), 4);
    /// assert_eq!(m.span_id(), (0, 4));
    /// ```
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

    /// Head-span tuple `(head_start, head_end)` for head-match scoring.
    ///
    /// Returns the head span if both `head_start` and `head_end` are set,
    /// otherwise falls back to the full span `(start, end)`.
    ///
    /// Head-match evaluation is standard in CRAC shared tasks: two mentions
    /// match if their syntactic heads overlap, even when full spans differ
    /// (e.g., "the president" vs "the former president of France" both have
    /// head "president").
    ///
    /// ```
    /// # use anno_core::core::coref::Mention;
    /// let m = Mention::with_head("the former president", 0, 20, 11, 20);
    /// assert_eq!(m.span_id_head(), (11, 20));
    ///
    /// let m2 = Mention::new("John", 0, 4);
    /// assert_eq!(m2.span_id_head(), (0, 4)); // falls back to full span
    /// ```
    #[must_use]
    pub fn span_id_head(&self) -> (usize, usize) {
        match (self.head_start, self.head_end) {
            (Some(hs), Some(he)) => (hs, he),
            _ => (self.start, self.end),
        }
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
    ///
    /// ```
    /// use anno_core::core::coref::{CorefDocument, CorefChain, Mention};
    ///
    /// let chain = CorefChain::new(vec![
    ///     Mention::new("John", 0, 4),
    ///     Mention::new("He", 24, 26),
    /// ]);
    /// let doc = CorefDocument::new("John went to the store. He bought milk.", vec![chain]);
    /// assert_eq!(doc.mention_count(), 2);
    /// assert_eq!(doc.chain_count(), 1);
    /// ```
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
            mention_type: entity.mention_type,
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
/// ```rust
/// use anno_core::{CoreferenceResolver, Entity, EntityType};
///
/// struct ExactMatchResolver;
///
/// impl CoreferenceResolver for ExactMatchResolver {
///     fn resolve(&self, entities: &[Entity]) -> Vec<Entity> {
///         // Trivially return entities unchanged for this example
///         entities.to_vec()
///     }
///
///     fn name(&self) -> &'static str {
///         "exact-match"
///     }
/// }
///
/// let resolver = ExactMatchResolver;
/// assert_eq!(resolver.name(), "exact-match");
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

    // =========================================================================
    // Edge case tests
    // =========================================================================

    #[test]
    fn test_unicode_mention_offsets() {
        // "北京 Beijing" — character offsets, not byte offsets.
        // "北" is 3 bytes in UTF-8 but 1 character.
        let m = Mention::new("北京", 0, 2); // 2 characters, not 6 bytes
        assert_eq!(m.len(), 2);
        assert_eq!(m.span_id(), (0, 2));
        assert!(!m.is_empty());
    }

    #[test]
    fn test_zero_length_mention() {
        // Zero anaphora / empty mention at position 5.
        let m = Mention::new("", 5, 5);
        assert!(m.is_empty());
        assert_eq!(m.len(), 0);
        assert_eq!(m.span_id(), (5, 5));
    }

    #[test]
    fn test_empty_chain() {
        let chain = CorefChain::new(vec![]);
        assert!(chain.is_empty());
        assert_eq!(chain.link_count(), 0);
        assert!(chain.all_pairs().is_empty());
        assert!(chain.first().is_none());
        assert!(chain.canonical_mention().is_none());
    }

    #[test]
    fn test_chain_sorting_out_of_order() {
        // Mentions given out of document order should be sorted by (start, end).
        let chain = CorefChain::new(vec![
            Mention::new("c", 20, 21),
            Mention::new("a", 0, 1),
            Mention::new("b", 10, 11),
        ]);
        assert_eq!(chain.mentions[0].text, "a");
        assert_eq!(chain.mentions[1].text, "b");
        assert_eq!(chain.mentions[2].text, "c");
    }

    #[test]
    fn test_chain_sorting_ties_broken_by_end() {
        // Same start, different end: shorter span first.
        let chain = CorefChain::new(vec![
            Mention::new("John Smith", 0, 10),
            Mention::new("John", 0, 4),
        ]);
        assert_eq!(chain.mentions[0].text, "John");
        assert_eq!(chain.mentions[1].text, "John Smith");
    }

    #[test]
    fn test_entities_to_chains_grouped() {
        use super::super::entity::EntityType;
        use super::super::types::CanonicalId;

        let e1 = super::super::Entity::new("John", EntityType::Person, 0, 4, 0.9)
            .with_canonical_id(1_u64);
        let e2 = super::super::Entity::new("he", EntityType::Person, 20, 22, 0.8)
            .with_canonical_id(1_u64);
        let e3 = super::super::Entity::new("Mary", EntityType::Person, 5, 9, 0.95)
            .with_canonical_id(2_u64);

        let chains = entities_to_chains(&[e1, e2, e3]);

        // Two canonical_ids -> two chains
        assert_eq!(chains.len(), 2);

        // Find the chain with cluster_id=1 (John + he)
        let chain1 = chains
            .iter()
            .find(|c| c.cluster_id == Some(CanonicalId::new(1)))
            .expect("chain with id=1");
        assert_eq!(chain1.len(), 2);

        // Find the chain with cluster_id=2 (Mary)
        let chain2 = chains
            .iter()
            .find(|c| c.cluster_id == Some(CanonicalId::new(2)))
            .expect("chain with id=2");
        assert_eq!(chain2.len(), 1);
    }

    #[test]
    fn test_entities_to_chains_singletons() {
        use super::super::entity::EntityType;

        // Entities without canonical_id become individual singleton chains.
        let e1 = super::super::Entity::new("Paris", EntityType::Location, 0, 5, 0.9);
        let e2 = super::super::Entity::new("London", EntityType::Location, 10, 16, 0.85);

        let chains = entities_to_chains(&[e1, e2]);
        assert_eq!(chains.len(), 2);
        assert!(chains.iter().all(|c| c.is_singleton()));
    }

    #[test]
    fn test_entities_to_chains_empty() {
        let chains = entities_to_chains(&[]);
        assert!(chains.is_empty());
    }

    #[test]
    fn test_without_singletons_filters() {
        let singleton = CorefChain::singleton(Mention::new("solo", 0, 4));
        let multi = CorefChain::new(vec![
            Mention::new("John", 10, 14),
            Mention::new("he", 20, 22),
        ]);
        let doc = CorefDocument::new("text", vec![singleton, multi]);

        let filtered = doc.without_singletons();
        assert_eq!(filtered.chain_count(), 1);
        assert_eq!(filtered.chains[0].len(), 2);
        assert!(!filtered.includes_singletons);
    }

    #[test]
    fn test_without_singletons_preserves_non_singletons() {
        let c1 = CorefChain::new(vec![Mention::new("a", 0, 1), Mention::new("b", 2, 3)]);
        let c2 = CorefChain::new(vec![
            Mention::new("x", 10, 11),
            Mention::new("y", 12, 13),
            Mention::new("z", 14, 15),
        ]);
        let doc = CorefDocument::new("text", vec![c1.clone(), c2.clone()]);

        let filtered = doc.without_singletons();
        assert_eq!(filtered.chain_count(), 2);
    }

    #[test]
    fn test_without_singletons_all_singletons() {
        let s1 = CorefChain::singleton(Mention::new("a", 0, 1));
        let s2 = CorefChain::singleton(Mention::new("b", 2, 3));
        let doc = CorefDocument::new("text", vec![s1, s2]);

        let filtered = doc.without_singletons();
        assert!(filtered.chains.is_empty());
    }

    #[test]
    fn test_overlaps_adjacent_non_overlapping() {
        // [0,5) and [5,10) are adjacent but NOT overlapping (half-open intervals).
        let m1 = Mention::new("hello", 0, 5);
        let m2 = Mention::new("world", 5, 10);
        assert!(!m1.overlaps(&m2));
        assert!(!m2.overlaps(&m1));
    }

    #[test]
    fn test_overlaps_nested() {
        // [0,10) fully contains [2,5).
        let outer = Mention::new("the big dog", 0, 10);
        let inner = Mention::new("big", 2, 5);
        assert!(outer.overlaps(&inner));
        assert!(inner.overlaps(&outer));
    }

    #[test]
    fn test_chain_with_id() {
        let chain = CorefChain::with_id(
            vec![Mention::new("John", 0, 4), Mention::new("he", 10, 12)],
            42_u64,
        );
        assert_eq!(
            chain.canonical_id(),
            Some(super::super::types::CanonicalId::new(42))
        );
        assert_eq!(
            chain.cluster_id,
            Some(super::super::types::CanonicalId::new(42))
        );
        // Mentions should still be sorted.
        assert_eq!(chain.mentions[0].text, "John");
    }
}

#[cfg(test)]
mod proptests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use proptest::prelude::*;

    /// Strategy to generate a Mention with bounded offsets.
    fn arb_mention(max_offset: usize) -> impl Strategy<Value = Mention> {
        (0usize..max_offset, 1usize..500)
            .prop_map(|(start, len)| Mention::new(format!("m_{}", start), start, start + len))
    }

    proptest! {
        /// Mentions sort by (start, end) consistently after CorefChain::new.
        #[test]
        fn mention_ordering_after_chain_construction(
            mentions in proptest::collection::vec(arb_mention(10000), 1..20),
        ) {
            let chain = CorefChain::new(mentions);
            for w in chain.mentions.windows(2) {
                prop_assert!(
                    (w[0].start, w[0].end) <= (w[1].start, w[1].end),
                    "mentions must be sorted by (start, end): ({},{}) vs ({},{})",
                    w[0].start, w[0].end, w[1].start, w[1].end
                );
            }
        }

        /// CorefChain constructed with at least one mention is never empty.
        #[test]
        fn coref_chain_non_empty(
            mentions in proptest::collection::vec(arb_mention(10000), 1..20),
        ) {
            let n = mentions.len();
            let chain = CorefChain::new(mentions);
            prop_assert!(!chain.is_empty());
            prop_assert_eq!(chain.len(), n);
        }

        /// CorefChain::singleton always produces a chain with exactly one mention.
        #[test]
        fn coref_chain_singleton_has_one(start in 0usize..10000, len in 1usize..500) {
            let m = Mention::new("x", start, start + len);
            let chain = CorefChain::singleton(m);
            prop_assert!(chain.is_singleton());
            prop_assert_eq!(chain.len(), 1);
            prop_assert_eq!(chain.link_count(), 0);
        }

        /// Mention::overlaps is symmetric.
        #[test]
        fn mention_overlap_symmetric(
            s1 in 0usize..10000, len1 in 1usize..500,
            s2 in 0usize..10000, len2 in 1usize..500,
        ) {
            let m1 = Mention::new("a", s1, s1 + len1);
            let m2 = Mention::new("b", s2, s2 + len2);
            prop_assert_eq!(m1.overlaps(&m2), m2.overlaps(&m1));
        }

        /// Mention serde roundtrip preserves all fields.
        #[test]
        fn mention_serde_roundtrip(
            start in 0usize..10000, len in 1usize..500,
        ) {
            let m = Mention::new(format!("mention_{}", start), start, start + len);
            let json = serde_json::to_string(&m).unwrap();
            let m2: Mention = serde_json::from_str(&json).unwrap();
            prop_assert_eq!(&m, &m2);
        }
    }
}
