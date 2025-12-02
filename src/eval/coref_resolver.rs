//! Simple coreference resolution for evaluation pipelines.
//!
//! Provides a minimal resolver to produce coreference chains from entities,
//! completing the loop between NER extraction and coreference evaluation metrics.
//!
//! # Design Philosophy
//!
//! This resolver is intentionally simple:
//! - Rule-based, no ML dependencies
//! - Good enough for evaluation pipelines
//! - Demonstrates how to connect NER → Coref metrics
//!
//! For production coreference, use a dedicated system like:
//! - Stanford CoreNLP
//! - AllenNLP coref
//! - Hugging Face neuralcoref
//!
//! # Gender Handling & Bias Considerations
//!
//! This resolver takes a **gender-aware but debiased** approach, informed by
//! research in NLP fairness (Rudinger 2018, Cao & Daumé 2019, Hossain 2023):
//!
//! 1. **No name-based gender inference**: We do NOT assume "Mary" is female
//!    or "John" is male. Such assumptions encode cultural stereotypes that
//!    harm transgender, non-binary, and gender-nonconforming individuals.
//!
//! 2. **Pronoun-only gender signals**: Gender is inferred ONLY from pronouns
//!    (he→masculine, she→feminine, they→neutral). Names get `None` gender.
//!
//! 3. **Singular "they" support**: Treated as gender-neutral, compatible with
//!    any antecedent. This reflects contemporary English usage.
//!
//! 4. **Neopronoun support**: Recognizes xe/xem, ze/zir, ey/em, fae/faer.
//!    These are used by non-binary individuals and should resolve correctly.
//!
//! ## Limitations
//!
//! - **No occupational bias mitigation**: WinoBias shows systems struggle with
//!   anti-stereotypical pronoun-occupation pairs. This simple resolver doesn't
//!   address that—use ML-based systems with debiasing for production.
//!
//! - **Neopronoun performance gap**: Research (MISGENDERED, ACL 2023) shows
//!   ML models achieve only ~7.7% accuracy on neopronouns out-of-the-box.
//!   Our rule-based approach handles them correctly but can't learn context.
//!
//! - **No intersectional analysis**: WinoIdentity shows bias compounds for
//!   doubly-disadvantaged groups. We don't measure or mitigate this.
//!
//! ## References
//!
//! - Rudinger et al. (2018): "Gender Bias in Coreference Resolution"
//! - Cao & Daumé (2019): "Toward Gender-Inclusive Coreference Resolution"
//! - Hossain et al. (2023): "MISGENDERED: Limits of LLMs in Understanding Pronouns"
//! - Devinney et al. (2022): "Theories of 'Gender' in NLP Bias Research"
//!
//! # Example
//!
//! ```rust
//! use anno::eval::coref_resolver::{SimpleCorefResolver, CorefConfig};
//! use anno::eval::coref::CorefChain;
//! use anno::{Entity, EntityType};
//!
//! let resolver = SimpleCorefResolver::default();
//!
//! let entities = vec![
//!     Entity::new("John Smith", EntityType::Person, 0, 10, 0.9),
//!     Entity::new("Smith", EntityType::Person, 45, 50, 0.85),
//!     Entity::new("he", EntityType::Person, 80, 82, 0.7),
//! ];
//!
//! let resolved = resolver.resolve(&entities);
//! // resolved[0].canonical_id == resolved[1].canonical_id == resolved[2].canonical_id
//! ```

use super::coref::CorefChain;
use crate::backends::box_embeddings::{BoxCorefConfig, BoxEmbedding};
use crate::{Entity, EntityType};
use std::collections::HashMap;

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for simple coreference resolver.
#[derive(Debug, Clone)]
pub struct CorefConfig {
    /// Similarity threshold for name matching (0.0-1.0)
    pub similarity_threshold: f64,
    /// Maximum sentence distance for pronoun resolution
    pub max_pronoun_distance: usize,
    /// Enable fuzzy name matching (e.g., "John Smith" ~ "J. Smith")
    pub fuzzy_matching: bool,
    /// Include singletons in output chains
    pub include_singletons: bool,
}

impl Default for CorefConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.7,
            max_pronoun_distance: 3,
            fuzzy_matching: true,
            include_singletons: true,
        }
    }
}

// =============================================================================
// Resolver
// =============================================================================

/// Simple rule-based coreference resolver.
///
/// Resolves coreference using three strategies:
/// 1. **Exact match**: Same surface form → same entity
/// 2. **Substring match**: "Smith" matches "John Smith"
/// 3. **Pronoun resolution**: "he/she" links to nearest person
///
/// This is sufficient for evaluation but not production-grade.
#[derive(Debug, Clone)]
pub struct SimpleCorefResolver {
    config: CorefConfig,
}

impl Default for SimpleCorefResolver {
    fn default() -> Self {
        Self::new(CorefConfig::default())
    }
}

impl SimpleCorefResolver {
    /// Create a new resolver with configuration.
    #[must_use]
    pub fn new(config: CorefConfig) -> Self {
        Self { config }
    }

    /// Resolve coreference for entities, assigning canonical IDs.
    ///
    /// Returns entities with `canonical_id` populated. Entities sharing
    /// the same `canonical_id` corefer (refer to the same real-world entity).
    #[must_use]
    pub fn resolve(&self, entities: &[Entity]) -> Vec<Entity> {
        if entities.is_empty() {
            return vec![];
        }

        let mut resolved = entities.to_vec();
        let mut next_cluster_id: u64 = 0;

        // Map from canonical form to cluster ID
        let mut canonical_to_cluster: HashMap<String, u64> = HashMap::new();

        // Process entities in order
        for i in 0..resolved.len() {
            let entity = &resolved[i];

            // Skip if already assigned
            if entity.canonical_id.is_some() {
                continue;
            }

            // Try to find a matching cluster
            let cluster_id =
                self.find_matching_cluster(entity, &resolved[..i], &canonical_to_cluster);

            let cluster_id = cluster_id.unwrap_or_else(|| {
                // Create new cluster
                let id = next_cluster_id;
                next_cluster_id += 1;
                id
            });

            // Assign cluster ID
            resolved[i].canonical_id = Some(cluster_id);

            // Update canonical form mapping
            let canonical = self.canonical_form(&resolved[i].text, &resolved[i].entity_type);
            canonical_to_cluster.insert(canonical, cluster_id);
        }

        resolved
    }

    /// Convert resolved entities directly to coreference chains.
    ///
    /// Convenience method that calls `resolve()` then groups into chains.
    #[must_use]
    pub fn resolve_to_chains(&self, entities: &[Entity]) -> Vec<CorefChain> {
        let resolved = self.resolve(entities);
        super::coref::entities_to_chains(&resolved)
    }

    /// Find a matching cluster for an entity.
    fn find_matching_cluster(
        &self,
        entity: &Entity,
        previous: &[Entity],
        canonical_map: &HashMap<String, u64>,
    ) -> Option<u64> {
        // Strategy 1: Pronoun resolution
        if self.is_pronoun(&entity.text) {
            return self.resolve_pronoun(entity, previous);
        }

        // Strategy 2: Exact canonical match
        let canonical = self.canonical_form(&entity.text, &entity.entity_type);
        if let Some(&cluster_id) = canonical_map.get(&canonical) {
            return Some(cluster_id);
        }

        // Strategy 3: Substring/fuzzy matching
        if self.config.fuzzy_matching {
            for (other_canonical, &cluster_id) in canonical_map {
                if self.names_match(&canonical, other_canonical) {
                    return Some(cluster_id);
                }
            }
        }

        None
    }

    /// Resolve a pronoun to its antecedent.
    ///
    /// # Gender Handling
    ///
    /// - Gendered pronouns (he/she) link to the nearest type-compatible entity
    /// - Neutral pronouns (they/them) link to any type-compatible entity
    /// - We do NOT infer gender from names to avoid encoding bias
    ///
    /// This means "they" can refer to anyone, and we don't assume
    /// "Mary" is female or "John" is male.
    fn resolve_pronoun(&self, pronoun: &Entity, previous: &[Entity]) -> Option<u64> {
        let pronoun_gender = self.infer_gender(&pronoun.text);

        // Look backwards for a compatible antecedent
        for entity in previous
            .iter()
            .rev()
            .take(self.config.max_pronoun_distance * 10)
        {
            // Skip other pronouns
            if self.is_pronoun(&entity.text) {
                continue;
            }

            // Must be a person (for he/she) or compatible type
            if !self.pronoun_compatible(&pronoun.text, &entity.entity_type) {
                continue;
            }

            // Gender compatibility check
            //
            // Key insight: We only know gender from PRONOUNS, not from names.
            // So entity_gender will almost always be None (can't infer from "John" or "Mary").
            // This is intentional - it avoids encoding stereotypical assumptions.
            //
            // - If pronoun is 'n' (they/them): compatible with any entity
            // - If pronoun is 'm' or 'f': compatible with any entity (we can't check name gender)
            // - If entity was a pronoun (already skipped above): would check gender match
            let entity_gender = self.infer_gender(&entity.text);

            match (pronoun_gender, entity_gender) {
                // Neutral pronoun: compatible with everything
                (Some('n'), _) => {}
                // Entity is neutral (they): compatible with any pronoun
                (_, Some('n')) => {}
                // Entity has known gender (was a pronoun in original text): must match
                (Some(pg), Some(eg)) => {
                    if pg != eg {
                        continue;
                    }
                }
                // Can't determine entity gender (it's a name): no filtering
                // This is the common case and avoids gender-from-name bias
                (_, None) => {}
                // Pronoun has no gender (shouldn't happen for pronouns): accept
                (None, Some(_)) => {}
            }

            // Found a compatible antecedent
            return entity.canonical_id;
        }

        None
    }

    /// Check if text is a pronoun.
    ///
    /// Recognizes:
    /// - Traditional binary pronouns (he/she)
    /// - Singular "they" (widely adopted for non-binary individuals)
    /// - Common neopronouns (xe, ze, ey, fae) per MISGENDERED dataset
    fn is_pronoun(&self, text: &str) -> bool {
        matches!(
            text.to_lowercase().as_str(),
            // Traditional gendered pronouns
            "he" | "she" | "him" | "her" | "his" | "hers" | "himself" | "herself" |
            // Singular they (gender-neutral, widely adopted)
            "they" | "them" | "their" | "theirs" | "themselves" | "themself" |
            // Impersonal pronouns
            "it" | "its" | "itself" |
            // Neopronouns: xe/xem/xyr (one of the most common)
            "xe" | "xem" | "xyr" | "xyrs" | "xemself" |
            // Neopronouns: ze/zir (Spivak-derived)
            "ze" | "hir" | "zir" | "hirs" | "zirs" | "hirself" | "zirself" |
            // Neopronouns: ey/em (from "they" minus "th")
            "ey" | "em" | "eir" | "eirs" | "emself" |
            // Neopronouns: fae/faer (nature-inspired)
            "fae" | "faer" | "faers" | "faeself"
        )
    }

    /// Check if a pronoun is compatible with an entity type.
    ///
    /// Person entities can take any personal pronoun including neopronouns.
    /// Organizations can take "they" (collective) or "it".
    /// Locations typically take "it".
    fn pronoun_compatible(&self, pronoun: &str, entity_type: &EntityType) -> bool {
        let lower = pronoun.to_lowercase();
        match entity_type {
            EntityType::Person => matches!(
                lower.as_str(),
                // Traditional
                "he" | "she" | "they" | "him" | "her" | "them" |
                "his" | "hers" | "their" | "theirs" |
                "himself" | "herself" | "themselves" | "themself" |
                // Neopronouns (all can refer to people)
                "xe" | "xem" | "xyr" | "xyrs" | "xemself" |
                "ze" | "hir" | "zir" | "hirs" | "zirs" | "hirself" | "zirself" |
                "ey" | "em" | "eir" | "eirs" | "emself" |
                "fae" | "faer" | "faers" | "faeself"
            ),
            EntityType::Organization => matches!(
                lower.as_str(),
                // Orgs use "it" or collective "they"
                "it" | "they" | "its" | "their" | "theirs" | "itself" | "themselves"
            ),
            EntityType::Location => matches!(lower.as_str(), "it" | "its" | "itself"),
            _ => matches!(lower.as_str(), "it" | "its" | "itself"),
        }
    }

    /// Infer gender from pronoun text.
    ///
    /// # Gender Bias Warning
    ///
    /// This method only infers gender from **pronouns**, not from names.
    /// Inferring gender from names (e.g., "Mary" → female) encodes cultural
    /// and stereotypical assumptions that don't hold universally.
    ///
    /// # Pronoun Categories
    ///
    /// - **Masculine** ('m'): he/him/his
    /// - **Feminine** ('f'): she/her/hers
    /// - **Neutral** ('n'): they/them, neopronouns (xe/ze/ey/fae)
    ///
    /// All neopronouns are treated as neutral since they explicitly signal
    /// non-binary identity. Per Cao & Daumé (2019), this avoids forcing
    /// binary categorization on non-binary individuals.
    ///
    /// # Returns
    ///
    /// - `Some('m')` for masculine pronouns
    /// - `Some('f')` for feminine pronouns
    /// - `Some('n')` for neutral/non-binary pronouns (including neopronouns)
    /// - `None` for names and other text (no assumption made)
    fn infer_gender(&self, text: &str) -> Option<char> {
        let lower = text.to_lowercase();
        match lower.as_str() {
            // Traditional masculine
            "he" | "him" | "his" | "himself" => Some('m'),

            // Traditional feminine
            "she" | "her" | "hers" | "herself" => Some('f'),

            // Singular "they" - gender-neutral, compatible with any antecedent
            "they" | "them" | "their" | "theirs" | "themselves" | "themself" => Some('n'),

            // Neopronouns - all treated as neutral ('n')
            // xe/xem set
            "xe" | "xem" | "xyr" | "xyrs" | "xemself" => Some('n'),
            // ze/zir set (includes "hir" which is distinct from "her")
            "ze" | "hir" | "zir" | "hirs" | "zirs" | "hirself" | "zirself" => Some('n'),
            // ey/em set
            "ey" | "em" | "eir" | "eirs" | "emself" => Some('n'),
            // fae/faer set
            "fae" | "faer" | "faers" | "faeself" => Some('n'),

            // Names and other text: NO gender inference
            // This is critical - assuming "Mary" → female encodes bias
            _ => None,
        }
    }

    /// Normalize text to canonical form for matching.
    fn canonical_form(&self, text: &str, entity_type: &EntityType) -> String {
        let normalized = text.to_lowercase().trim().to_string();

        // Prefix with type to avoid "Apple" (company) matching "apple" (fruit)
        format!("{}:{}", entity_type.as_label(), normalized)
    }

    /// Check if two canonical names match (substring or fuzzy).
    fn names_match(&self, name1: &str, name2: &str) -> bool {
        // Same type prefix required
        let (type1, text1) = name1.split_once(':').unwrap_or(("", name1));
        let (type2, text2) = name2.split_once(':').unwrap_or(("", name2));

        if type1 != type2 {
            return false;
        }

        // Exact match
        if text1 == text2 {
            return true;
        }

        // Substring match (one is part of the other)
        if text1.contains(text2) || text2.contains(text1) {
            return true;
        }

        // Last name match ("Smith" matches "John Smith")
        let words1: Vec<&str> = text1.split_whitespace().collect();
        let words2: Vec<&str> = text2.split_whitespace().collect();

        if words1.len() > 1 && words2.len() == 1 && words1.last() == words2.first() {
            return true;
        }
        if words2.len() > 1 && words1.len() == 1 && words2.last() == words1.first() {
            return true;
        }

        false
    }
}

// =============================================================================
// Trait Implementation
// =============================================================================

/// Trait for coreference resolvers.
///
/// Allows different resolution strategies to be used interchangeably.
pub trait CoreferenceResolver: Send + Sync {
    /// Resolve coreference, assigning canonical IDs to entities.
    fn resolve(&self, entities: &[Entity]) -> Vec<Entity>;

    /// Resolve directly to chains.
    fn resolve_to_chains(&self, entities: &[Entity]) -> Vec<CorefChain> {
        let resolved = self.resolve(entities);
        super::coref::entities_to_chains(&resolved)
    }

    /// Get resolver name.
    fn name(&self) -> &'static str;
}

impl CoreferenceResolver for SimpleCorefResolver {
    fn resolve(&self, entities: &[Entity]) -> Vec<Entity> {
        self.resolve(entities)
    }

    fn name(&self) -> &'static str {
        "simple-rule-based"
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn person(text: &str, start: usize) -> Entity {
        Entity::new(text, EntityType::Person, start, start + text.len(), 0.9)
    }

    fn org(text: &str, start: usize) -> Entity {
        Entity::new(
            text,
            EntityType::Organization,
            start,
            start + text.len(),
            0.9,
        )
    }

    #[test]
    fn test_exact_match() {
        let resolver = SimpleCorefResolver::default();
        let entities = vec![person("John Smith", 0), person("John Smith", 50)];

        let resolved = resolver.resolve(&entities);
        assert_eq!(resolved[0].canonical_id, resolved[1].canonical_id);
    }

    #[test]
    fn test_substring_match() {
        let resolver = SimpleCorefResolver::default();
        let entities = vec![person("John Smith", 0), person("Smith", 50)];

        let resolved = resolver.resolve(&entities);
        assert_eq!(resolved[0].canonical_id, resolved[1].canonical_id);
    }

    #[test]
    fn test_pronoun_resolution() {
        let resolver = SimpleCorefResolver::default();
        let entities = vec![person("John Smith", 0), person("he", 50)];

        let resolved = resolver.resolve(&entities);
        assert_eq!(resolved[0].canonical_id, resolved[1].canonical_id);
    }

    #[test]
    fn test_different_entities() {
        let resolver = SimpleCorefResolver::default();
        let entities = vec![person("John Smith", 0), person("Mary Jones", 50)];

        let resolved = resolver.resolve(&entities);
        assert_ne!(resolved[0].canonical_id, resolved[1].canonical_id);
    }

    #[test]
    fn test_type_matters() {
        let resolver = SimpleCorefResolver::default();
        let entities = vec![
            person("Apple", 0), // Person named Apple
            org("Apple", 50),   // Apple Inc.
        ];

        let resolved = resolver.resolve(&entities);
        // Different types should NOT match
        assert_ne!(resolved[0].canonical_id, resolved[1].canonical_id);
    }

    #[test]
    fn test_resolve_to_chains() {
        let resolver = SimpleCorefResolver::default();
        let entities = vec![person("John", 0), person("he", 20), person("Mary", 40)];

        let chains = resolver.resolve_to_chains(&entities);

        // John + he in one chain, Mary singleton
        assert_eq!(chains.len(), 2);

        let non_singletons: Vec<_> = chains.iter().filter(|c| !c.is_singleton()).collect();
        assert_eq!(non_singletons.len(), 1);
        assert_eq!(non_singletons[0].len(), 2);
    }

    // =========================================================================
    // Gender-inclusive pronoun tests
    // =========================================================================

    #[test]
    fn test_singular_they_resolves_to_any_person() {
        let resolver = SimpleCorefResolver::default();
        let entities = vec![
            person("Alex", 0), // Gender-ambiguous name
            person("they", 20),
        ];

        let resolved = resolver.resolve(&entities);
        assert_eq!(
            resolved[0].canonical_id, resolved[1].canonical_id,
            "Singular 'they' should resolve to any person"
        );
    }

    #[test]
    fn test_neopronoun_xe_resolves() {
        let resolver = SimpleCorefResolver::default();
        let entities = vec![person("Jordan", 0), person("xe", 30)];

        let resolved = resolver.resolve(&entities);
        assert_eq!(
            resolved[0].canonical_id, resolved[1].canonical_id,
            "Neopronoun 'xe' should resolve to any person"
        );
    }

    #[test]
    fn test_neopronoun_ze_resolves() {
        let resolver = SimpleCorefResolver::default();
        let entities = vec![
            person("Sam", 0),
            person("zir", 25), // possessive of ze
        ];

        let resolved = resolver.resolve(&entities);
        assert_eq!(
            resolved[0].canonical_id, resolved[1].canonical_id,
            "Neopronoun 'zir' should resolve to any person"
        );
    }

    #[test]
    fn test_neopronoun_fae_resolves() {
        let resolver = SimpleCorefResolver::default();
        let entities = vec![person("River", 0), person("faer", 30)];

        let resolved = resolver.resolve(&entities);
        assert_eq!(
            resolved[0].canonical_id, resolved[1].canonical_id,
            "Neopronoun 'faer' should resolve to any person"
        );
    }

    #[test]
    fn test_no_gender_inferred_from_names() {
        let resolver = SimpleCorefResolver::default();

        // Even stereotypically "female" names should not block "he"
        // This is intentional: we don't assume gender from names
        let entities = vec![
            person("Mary", 0),
            person("he", 20), // Should still link - we don't know Mary's pronouns
        ];

        let resolved = resolver.resolve(&entities);
        assert_eq!(
            resolved[0].canonical_id, resolved[1].canonical_id,
            "Should not infer gender from names (avoids stereotyping)"
        );
    }

    #[test]
    fn test_all_neopronouns_recognized() {
        let resolver = SimpleCorefResolver::default();

        // Test that all supported neopronouns are recognized
        let neopronouns = [
            "xe", "xem", "xyr", "xyrs", "xemself", "ze", "hir", "zir", "hirs", "zirs", "hirself",
            "zirself", "ey", "em", "eir", "eirs", "emself", "fae", "faer", "faers", "faeself",
        ];

        for pronoun in neopronouns {
            assert!(
                resolver.is_pronoun(pronoun),
                "Should recognize neopronoun: {}",
                pronoun
            );
        }
    }
}

// =============================================================================
// Discourse-Aware Coreference Resolution
// =============================================================================

#[cfg(feature = "discourse")]
use crate::discourse::{
    classify_shell_noun, DiscourseReferent, DiscourseScope, EventExtractor, EventMention,
    ReferentType,
};

/// Configuration for discourse-aware coreference resolution.
#[cfg(feature = "discourse")]
#[derive(Debug, Clone)]
pub struct DiscourseCorefConfig {
    /// Base coreference config
    pub base: CorefConfig,
    /// Enable shell noun detection
    pub detect_shell_nouns: bool,
    /// Maximum sentences back to search for antecedent
    pub max_sentence_distance: usize,
    /// Prefer clause-level antecedents for "this"
    pub prefer_clause_antecedent: bool,
}

#[cfg(feature = "discourse")]
impl Default for DiscourseCorefConfig {
    fn default() -> Self {
        Self {
            base: CorefConfig::default(),
            detect_shell_nouns: true,
            max_sentence_distance: 3,
            prefer_clause_antecedent: true,
        }
    }
}

/// Discourse-aware coreference resolver.
///
/// Extends `SimpleCorefResolver` with:
/// - **Event extraction** (NEW): Uses `EventExtractor` to detect event triggers
/// - Shell noun detection ("this problem" → abstract antecedent)
/// - Clause boundary awareness (prefer nearest clause for "this")
/// - Event/proposition candidate generation
///
/// # Research Background
///
/// Based on insights from:
/// - Kolhatkar & Hirst (2012): Shell noun resolution
/// - Marasović et al. (2017): Neural abstract anaphora
/// - Li & Ng (2022): dd-utt for discourse deixis
/// - ACE 2005 event ontology for trigger detection
///
/// # Example
///
/// ```rust,ignore
/// use anno::eval::coref_resolver::{DiscourseAwareResolver, DiscourseCorefConfig};
/// use anno::Entity;
///
/// let text = "Russia invaded Ukraine. This shocked the world.";
/// let resolver = DiscourseAwareResolver::new(DiscourseCorefConfig::default(), text);
///
/// // The resolver will:
/// // 1. Extract events: "invaded" (conflict:attack)
/// // 2. Detect "This" as a demonstrative
/// // 3. Link "This" to the invasion event (not Russia or Ukraine)
/// ```
#[cfg(feature = "discourse")]
#[derive(Debug, Clone)]
pub struct DiscourseAwareResolver {
    config: DiscourseCorefConfig,
    base_resolver: SimpleCorefResolver,
    scope: DiscourseScope,
    text: String,
    /// Extracted events from the text (NEW)
    events: Vec<EventMention>,
}

#[cfg(feature = "discourse")]
impl DiscourseAwareResolver {
    /// Create a new discourse-aware resolver.
    ///
    /// Automatically extracts events from the text using `EventExtractor`.
    #[must_use]
    pub fn new(config: DiscourseCorefConfig, text: &str) -> Self {
        let scope = DiscourseScope::analyze(text);

        // NEW: Extract events from text
        let extractor = EventExtractor::default();
        let events = extractor.extract(text);

        Self {
            base_resolver: SimpleCorefResolver::new(config.base.clone()),
            config,
            scope,
            text: text.to_string(),
            events,
        }
    }

    /// Get extracted events.
    #[must_use]
    pub fn events(&self) -> &[EventMention] {
        &self.events
    }

    /// Find an event mention containing or near a character offset.
    fn find_event_near(&self, offset: usize, max_distance: usize) -> Option<&EventMention> {
        // First try events in the same clause
        if let Some((clause_start, clause_end)) = self.scope.clause_at(offset) {
            for event in &self.events {
                if event.trigger_start >= clause_start && event.trigger_end <= clause_end {
                    return Some(event);
                }
            }
        }

        // Fall back to nearest event within distance
        self.events
            .iter()
            .filter(|e| {
                let dist = if e.trigger_end <= offset {
                    offset - e.trigger_end
                } else {
                    e.trigger_start.saturating_sub(offset)
                };
                dist <= max_distance
            })
            .min_by_key(|e| {
                if e.trigger_end <= offset {
                    offset - e.trigger_end
                } else {
                    e.trigger_start.saturating_sub(offset)
                }
            })
    }

    /// Resolve coreference with discourse awareness.
    ///
    /// First attempts standard entity coref, then handles abstract anaphora
    /// by searching for clause/sentence-level antecedents.
    #[must_use]
    pub fn resolve(&self, entities: &[Entity]) -> Vec<Entity> {
        // First pass: standard entity coref
        let mut resolved = self.base_resolver.resolve(entities);

        // Second pass: discourse-aware resolution for abstract anaphora
        for (i, entity) in resolved.iter_mut().enumerate() {
            if entity.canonical_id.is_some() {
                continue; // Already resolved
            }

            // Check for demonstrative pronouns and shell nouns
            if self.is_abstract_anaphor(&entity.text) {
                if let Some(antecedent) = self.find_discourse_antecedent(entity) {
                    // Assign a new cluster ID linking to the discourse referent
                    // For now, we just mark it as resolved (actual linking would need
                    // discourse referents to be first-class citizens)
                    let cluster_id = 10000 + i as u64; // High ID to distinguish
                    entity.canonical_id = Some(cluster_id);

                    // Store the antecedent info in normalized field as a temporary solution
                    // TODO: Make discourse referents first-class citizens with proper linking
                    // This format allows later extraction of discourse reference information
                    entity.normalized = Some(format!(
                        "DISCOURSE_REF:{:?}@{}-{}",
                        antecedent.referent_type, antecedent.start, antecedent.end
                    ));
                }
            }
        }

        resolved
    }

    /// Check if a mention is an abstract anaphor (demonstrative or shell noun phrase).
    fn is_abstract_anaphor(&self, text: &str) -> bool {
        let lower = text.to_lowercase();

        // Bare demonstratives
        if matches!(lower.as_str(), "this" | "that" | "it") {
            return true;
        }

        // Shell noun phrases: "this X" where X is a shell noun
        let words: Vec<&str> = lower.split_whitespace().collect();
        if words.len() >= 2 {
            let det = words[0];
            let noun = words
                .last()
                .expect("words has >= 2 elements")
                .trim_matches(|c: char| !c.is_alphanumeric());

            if matches!(det, "this" | "that" | "the" | "such")
                && classify_shell_noun(noun).is_some()
            {
                return true;
            }
        }

        false
    }

    /// Find a discourse-level antecedent for an abstract anaphor.
    ///
    /// NEW: First checks for extracted events, then falls back to span heuristics.
    pub fn find_discourse_antecedent(&self, anaphor: &Entity) -> Option<DiscourseReferent> {
        // NEW: First, check if there's an extracted event in a preceding clause
        // This is more accurate than span-based heuristics
        if let Some(event) = self.find_event_near(anaphor.start, 200) {
            // Found an event - create a discourse referent for it
            // find_event_clause_span returns character offsets
            let (clause_char_start, clause_char_end) = self.find_event_clause_span(event);

            // Extract text using character offsets
            let span_text: String = self
                .text
                .chars()
                .skip(clause_char_start)
                .take(clause_char_end.saturating_sub(clause_char_start))
                .collect();

            return Some(
                DiscourseReferent::new(ReferentType::Event, clause_char_start, clause_char_end)
                    .with_event(event.clone())
                    .with_text(span_text)
                    .with_confidence(0.85), // Higher confidence for extracted events
            );
        }

        // Fallback: Original span-based approach
        let candidates = self.scope.candidate_antecedent_spans(anaphor.start);

        // For shell nouns, prefer matching type
        let shell_class = if self.config.detect_shell_nouns {
            let lower = anaphor.text.to_lowercase();
            let last_word = lower.split_whitespace().last().map(|w| w.to_string());
            last_word
                .as_ref()
                .and_then(|w| classify_shell_noun(w.trim_matches(|c: char| !c.is_alphanumeric())))
        } else {
            None
        };

        // Score candidates by distance and type match
        for (start, end) in candidates
            .into_iter()
            .take(self.config.max_sentence_distance)
        {
            let span_text = self.scope.extract_span(&self.text, start, end);

            // Skip empty or very short spans
            if span_text.trim().len() < 3 {
                continue;
            }

            // Infer referent type from span
            let ref_type = self.infer_referent_type(span_text);

            // If shell noun, check type compatibility
            if let Some(class) = &shell_class {
                let expected_types = class.typical_antecedent_types();
                if !expected_types.contains(&ref_type) {
                    continue; // Type mismatch
                }
            }

            return Some(
                DiscourseReferent::new(ref_type, start, end)
                    .with_text(span_text)
                    .with_confidence(0.7),
            );
        }

        None
    }

    /// Find the clause span for an event mention.
    ///
    /// Returns character offsets (start, end) of the clause containing the event.
    fn find_event_clause_span(&self, event: &EventMention) -> (usize, usize) {
        // Try to get the clause containing the event (scope.clause_at returns char offsets)
        if let Some((start, end)) = self.scope.clause_at(event.trigger_start) {
            return (start, end);
        }

        // Fall back to sentence (scope.sentence_at returns char offsets)
        if let Some((start, end)) = self.scope.sentence_at(event.trigger_start) {
            return (start, end);
        }

        // Last resort: just the trigger span with some context
        // Note: trigger_start/trigger_end are character offsets, so we need to use char count
        let char_count = self.text.chars().count();
        let context_before = event.trigger_start.saturating_sub(30);
        let context_after = (event.trigger_end + 30).min(char_count);
        (context_before, context_after)
    }

    /// Infer the referent type from span text.
    ///
    /// Uses extracted events when available (NEW), falls back to heuristics.
    fn infer_referent_type(&self, text: &str) -> ReferentType {
        let lower = text.to_lowercase();

        // NEW: Check if span contains an extracted event trigger
        // This is more reliable than pure heuristics
        for event in &self.events {
            // Check if event trigger is within this text span
            if lower.contains(&event.trigger.to_lowercase()) {
                // Use event type to determine referent type
                if let Some(ref event_type) = event.trigger_type {
                    if event_type.starts_with("conflict:")
                        || event_type.starts_with("movement:")
                        || event_type.starts_with("transaction:")
                        || event_type.starts_with("justice:")
                        || event_type.starts_with("personnel:")
                        || event_type.starts_with("life:")
                        || event_type.starts_with("disaster:")
                        || event_type.starts_with("business:")
                    {
                        return ReferentType::Event;
                    }
                    if event_type.starts_with("economic:") {
                        return ReferentType::Situation;
                    }
                }
                return ReferentType::Event;
            }
        }

        // Fallback heuristics (original logic)

        // Look for event indicators (past tense verbs, action words)
        let event_indicators = [
            "ed ",
            " was ",
            " were ",
            " had ",
            " did ",
            " happened",
            " occurred",
        ];
        for ind in &event_indicators {
            if lower.contains(ind) {
                return ReferentType::Event;
            }
        }

        // Look for fact indicators
        let fact_indicators = [" is ", " are ", " equals ", " means "];
        for ind in &fact_indicators {
            if lower.contains(ind) {
                return ReferentType::Fact;
            }
        }

        // Look for proposition indicators (modals, subjunctive)
        let prop_indicators = [" might ", " may ", " could ", " would ", " should ", " if "];
        for ind in &prop_indicators {
            if lower.contains(ind) {
                return ReferentType::Proposition;
            }
        }

        // Look for situation indicators (ongoing states)
        let sit_indicators = [" while ", " as ", "ing ", " continues", " remains"];
        for ind in &sit_indicators {
            if lower.contains(ind) {
                return ReferentType::Situation;
            }
        }

        // Default to event for past-looking things
        ReferentType::Event
    }

    /// Get discourse referents found for abstract anaphora.
    #[must_use]
    pub fn get_discourse_referents(&self, entities: &[Entity]) -> Vec<(Entity, DiscourseReferent)> {
        let mut pairs = Vec::new();

        for entity in entities {
            if self.is_abstract_anaphor(&entity.text) {
                if let Some(referent) = self.find_discourse_antecedent(entity) {
                    pairs.push((entity.clone(), referent));
                }
            }
        }

        pairs
    }
}

#[cfg(feature = "discourse")]
impl CoreferenceResolver for DiscourseAwareResolver {
    fn resolve(&self, entities: &[Entity]) -> Vec<Entity> {
        self.resolve(entities)
    }

    fn name(&self) -> &'static str {
        "discourse-aware"
    }
}

// =============================================================================
// Box Embedding Coreference Resolver
// =============================================================================

/// Box-based coreference resolver.
///
/// Uses box embeddings to resolve coreference with explicit encoding of
/// logical invariants (transitivity, syntactic constraints).
///
/// # Algorithm
///
/// 1. Convert entities to box embeddings (if not already boxes)
/// 2. Compute pairwise coreference scores via conditional probability
/// 3. Cluster via transitive closure (box containment)
/// 4. Enforce syntactic constraints (Principle B/C) if enabled
///
/// # Example
///
/// ```rust,ignore
/// use anno::eval::coref_resolver::{BoxCorefResolver, BoxCorefConfig};
/// use anno::backends::box_embeddings::BoxEmbedding;
/// use anno::{Entity, EntityType};
///
/// let config = BoxCorefConfig::default();
/// let mut resolver = BoxCorefResolver::new(config);
///
/// let entities = vec![
///     Entity::new("John", EntityType::Person, 0, 4, 0.9),
///     Entity::new("he", EntityType::Person, 10, 12, 0.8),
/// ];
///
/// // Create box embeddings (in practice, these would be learned)
/// let boxes = vec![
///     BoxEmbedding::new(vec![0.0, 0.0], vec![1.0, 1.0]),
///     BoxEmbedding::new(vec![0.1, 0.1], vec![0.9, 0.9]),
/// ];
///
/// let resolved = resolver.resolve_with_boxes(&entities, &boxes);
/// ```
pub struct BoxCorefResolver {
    config: BoxCorefConfig,
}

impl BoxCorefResolver {
    /// Create a new box-based coreference resolver.
    #[must_use]
    pub fn new(config: BoxCorefConfig) -> Self {
        Self { config }
    }

    /// Resolve coreference using box embeddings.
    ///
    /// # Arguments
    ///
    /// * `entities` - Entities to resolve
    /// * `boxes` - Box embeddings for each entity (must match entities.len())
    ///
    /// # Returns
    ///
    /// Entities with `canonical_id` populated. Entities sharing the same
    /// `canonical_id` corefer.
    ///
    /// # Panics
    ///
    /// Panics if `boxes.len() != entities.len()`.
    pub fn resolve_with_boxes(&self, entities: &[Entity], boxes: &[BoxEmbedding]) -> Vec<Entity> {
        assert_eq!(
            entities.len(),
            boxes.len(),
            "entities and boxes must have same length"
        );

        if entities.is_empty() {
            return vec![];
        }

        let mut resolved = entities.to_vec();
        let mut next_cluster_id: u64 = 0;

        // Union-find for clustering
        let mut parent: Vec<usize> = (0..entities.len()).collect();

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

        // Compute pairwise coreference scores
        for i in 0..entities.len() {
            for j in (i + 1)..entities.len() {
                let score = boxes[i].coreference_score(&boxes[j]);

                // Check coreference threshold
                if score >= self.config.coreference_threshold {
                    // Enforce type compatibility
                    if entities[i].entity_type == entities[j].entity_type {
                        // Check syntactic constraints if enabled
                        if !self.config.enforce_syntactic_constraints
                            || self.check_syntactic_constraints(&entities[i], &entities[j], i, j)
                        {
                            union(&mut parent, i, j);
                        }
                    }
                }
            }
        }

        // Assign cluster IDs
        let mut cluster_map: HashMap<usize, u64> = HashMap::new();
        for i in 0..entities.len() {
            let root = find(&mut parent, i);
            let cluster_id = *cluster_map.entry(root).or_insert_with(|| {
                let id = next_cluster_id;
                next_cluster_id += 1;
                id
            });
            resolved[i].canonical_id = Some(cluster_id);
        }

        resolved
    }

    /// Check syntactic constraints (Principle B/C).
    ///
    /// Returns true if coreference is allowed, false if it violates constraints.
    fn check_syntactic_constraints(
        &self,
        entity_a: &Entity,
        entity_b: &Entity,
        _idx_a: usize,
        _idx_b: usize,
    ) -> bool {
        // Simplified: check if entities are in local domain
        // In full implementation, would use parse tree to determine c-command
        let distance = if entity_a.end <= entity_b.start {
            entity_b.start - entity_a.end
        } else {
            entity_a.start.saturating_sub(entity_b.end)
        };

        // Principle B: Pronoun cannot corefer with local entity (unless reflexive)
        if self.is_pronoun(&entity_a.text) && distance <= self.config.max_local_distance {
            // Would need parse tree to check if entity_b c-commands entity_a
            // For now, allow if not in same sentence (heuristic)
            return distance > 50; // Rough sentence boundary
        }

        // Principle C: R-expression (name) cannot be bound by c-commanding entity
        if self.is_rexpression(&entity_a.text) && distance <= self.config.max_local_distance {
            // Would need parse tree to check c-command
            // For now, allow if not too close (heuristic)
            return distance > 20;
        }

        true
    }

    /// Check if text is a pronoun.
    fn is_pronoun(&self, text: &str) -> bool {
        matches!(
            text.to_lowercase().as_str(),
            "he" | "she" | "they" | "him" | "her" | "them" | "it" | "this" | "that"
        )
    }

    /// Check if text is an R-expression (proper name).
    fn is_rexpression(&self, text: &str) -> bool {
        // Simple heuristic: capitalized words are likely R-expressions
        text.chars().next().map_or(false, |c| c.is_uppercase()) && text.len() > 1
    }
}

impl CoreferenceResolver for BoxCorefResolver {
    fn resolve(&self, entities: &[Entity]) -> Vec<Entity> {
        // Default implementation: requires boxes to be provided separately
        // In practice, boxes would come from a learned model or be computed
        // from entity embeddings. For now, return entities unchanged.
        // Use `resolve_with_boxes()` for actual resolution.
        entities.to_vec()
    }

    fn name(&self) -> &'static str {
        "box-embedding"
    }
}

// =============================================================================
// Box Embedding Utilities
// =============================================================================

/// Convert vector embeddings to box embeddings for coreference resolution.
///
/// This is a convenience function that creates boxes from entity embeddings
/// using a fixed radius. In practice, boxes would be learned from data.
///
/// # Arguments
///
/// * `embeddings` - Vector embeddings [num_entities, hidden_dim]
/// * `hidden_dim` - Hidden dimension of embeddings
/// * `radius` - Half-width of boxes (default: 0.1)
///
/// # Returns
///
/// Vector of box embeddings, one per entity.
pub fn vectors_to_boxes(
    embeddings: &[f32],
    hidden_dim: usize,
    radius: Option<f32>,
) -> Vec<BoxEmbedding> {
    let radius = radius.unwrap_or(0.1);
    let num_entities = embeddings.len() / hidden_dim;
    let mut boxes = Vec::with_capacity(num_entities);

    for i in 0..num_entities {
        let start = i * hidden_dim;
        let end = start + hidden_dim;
        let vector = &embeddings[start..end];
        boxes.push(BoxEmbedding::from_vector(vector, radius));
    }

    boxes
}

/// Resolve coreference using box embeddings derived from vector embeddings.
///
/// Convenience function that combines vector-to-box conversion with resolution.
///
/// # Arguments
///
/// * `entities` - Entities to resolve
/// * `embeddings` - Vector embeddings [num_entities, hidden_dim]
/// * `hidden_dim` - Hidden dimension
/// * `config` - Box coreference configuration
///
/// # Returns
///
/// Resolved entities with `canonical_id` populated.
pub fn resolve_with_box_embeddings(
    entities: &[Entity],
    embeddings: &[f32],
    hidden_dim: usize,
    config: BoxCorefConfig,
) -> Vec<Entity> {
    let radius = config.vector_to_box_radius;
    let boxes = vectors_to_boxes(embeddings, hidden_dim, radius);
    let resolver = BoxCorefResolver::new(config);
    resolver.resolve_with_boxes(entities, &boxes)
}

#[cfg(all(test, feature = "discourse"))]
mod discourse_tests {
    use super::*;

    #[test]
    fn test_shell_noun_detection() {
        let text = "The company failed. This problem worried investors.";
        let resolver = DiscourseAwareResolver::new(DiscourseCorefConfig::default(), text);

        assert!(resolver.is_abstract_anaphor("This problem"));
        assert!(resolver.is_abstract_anaphor("this"));
        assert!(!resolver.is_abstract_anaphor("John"));
    }

    #[test]
    fn test_discourse_antecedent_finding() {
        let text = "Russia invaded Ukraine. This shocked everyone.";
        let resolver = DiscourseAwareResolver::new(DiscourseCorefConfig::default(), text);

        let anaphor = Entity::new("This", EntityType::Other("demo".into()), 24, 28, 0.8);
        let referent = resolver.find_discourse_antecedent(&anaphor);

        assert!(referent.is_some(), "Should find discourse antecedent");
        let ref_unwrapped = referent.unwrap();
        assert!(ref_unwrapped.is_abstract(), "Should be abstract referent");
    }

    #[test]
    fn test_discourse_aware_resolution() {
        let text = "The CEO resigned suddenly. This decision shocked the board.";
        let config = DiscourseCorefConfig::default();
        let resolver = DiscourseAwareResolver::new(config, text);

        let entities = vec![
            Entity::new("CEO", EntityType::Person, 4, 7, 0.9),
            Entity::new(
                "This decision",
                EntityType::Other("shell".into()),
                28,
                41,
                0.8,
            ),
            Entity::new("board", EntityType::Organization, 53, 58, 0.85),
        ];

        // Check that abstract anaphor detection works
        assert!(
            resolver.is_abstract_anaphor("This decision"),
            "Should detect 'This decision' as abstract anaphor"
        );

        // Check discourse referent finding
        let pairs = resolver.get_discourse_referents(&entities);
        assert!(
            !pairs.is_empty(),
            "Should find discourse referents for shell nouns"
        );

        // Verify the referent is for the resignation event
        let (anaphor, referent) = &pairs[0];
        assert_eq!(anaphor.text, "This decision");
        assert!(referent.is_abstract(), "Referent should be abstract");
    }

    // === NEW TESTS FOR EVENT EXTRACTION INTEGRATION ===

    #[test]
    fn test_event_extraction_integration() {
        let text = "Russia invaded Ukraine in 2022. This caused a global crisis.";
        let resolver = DiscourseAwareResolver::new(DiscourseCorefConfig::default(), text);

        // Should extract the "invaded" event
        let events = resolver.events();
        assert!(!events.is_empty(), "Should extract events from text");

        let invasion_event = events.iter().find(|e| e.trigger == "invaded");
        assert!(
            invasion_event.is_some(),
            "Should find 'invaded' event trigger"
        );

        let event = invasion_event.unwrap();
        assert_eq!(event.trigger_type.as_deref(), Some("conflict:attack"));
    }

    #[test]
    fn test_event_based_antecedent_finding() {
        let text = "The earthquake struck at dawn. This destroyed thousands of homes.";
        let resolver = DiscourseAwareResolver::new(DiscourseCorefConfig::default(), text);

        // Check events were extracted
        let events = resolver.events();
        assert!(
            events.iter().any(|e| e.trigger == "struck"),
            "Should extract 'struck' event"
        );

        // The anaphor "This" should link to the earthquake event
        let anaphor = Entity::new("This", EntityType::Other("demo".into()), 31, 35, 0.8);
        let referent = resolver.find_discourse_antecedent(&anaphor);

        assert!(referent.is_some(), "Should find event antecedent");
        let ref_unwrapped = referent.unwrap();
        assert_eq!(ref_unwrapped.referent_type, ReferentType::Event);
        assert!(
            ref_unwrapped.event.is_some(),
            "Should attach event mention to referent"
        );
    }

    #[test]
    fn test_multiple_events_nearest_preferred() {
        let text = "Apple announced layoffs. Microsoft announced profits. This shocked analysts.";
        let resolver = DiscourseAwareResolver::new(DiscourseCorefConfig::default(), text);

        // Should have 2 announcement events
        let events = resolver.events();
        let announcements: Vec<_> = events.iter().filter(|e| e.trigger == "announced").collect();
        assert_eq!(announcements.len(), 2, "Should find 2 announcement events");

        // "This" should prefer the nearest (Microsoft announcement)
        let anaphor = Entity::new("This", EntityType::Other("demo".into()), 55, 59, 0.8);
        let referent = resolver.find_discourse_antecedent(&anaphor);

        assert!(referent.is_some(), "Should find antecedent");
        // The referent text should be from the Microsoft sentence (nearest)
        let ref_text = referent.unwrap().text.unwrap_or_default();
        assert!(
            ref_text.contains("Microsoft") || ref_text.contains("profits"),
            "Should prefer nearest event: {}",
            ref_text
        );
    }

    #[test]
    fn test_event_type_inference() {
        // Use a simpler trigger that's definitely in the lexicon
        let text = "The company fired 500 workers.";
        let resolver = DiscourseAwareResolver::new(DiscourseCorefConfig::default(), text);

        // Check event extraction (fired is in the lexicon)
        let events = resolver.events();
        assert!(
            !events.is_empty(),
            "Should extract firing event, got: {:?}",
            events
        );

        // Test referent type inference uses event type
        let ref_type = resolver.infer_referent_type(text);
        assert_eq!(
            ref_type,
            ReferentType::Event,
            "Should infer Event type from extracted event"
        );
    }
}
