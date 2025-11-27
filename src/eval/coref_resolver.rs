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
            let cluster_id = self.find_matching_cluster(
                entity,
                &resolved[..i],
                &canonical_to_cluster,
            );
            
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
    fn resolve_pronoun(&self, pronoun: &Entity, previous: &[Entity]) -> Option<u64> {
        let pronoun_gender = self.infer_gender(&pronoun.text);
        
        // Look backwards for a compatible antecedent
        for entity in previous.iter().rev().take(self.config.max_pronoun_distance * 10) {
            // Skip other pronouns
            if self.is_pronoun(&entity.text) {
                continue;
            }
            
            // Must be a person (for he/she) or compatible type
            if !self.pronoun_compatible(&pronoun.text, &entity.entity_type) {
                continue;
            }
            
            // Gender compatibility (if we can infer)
            if let Some(entity_gender) = self.infer_gender(&entity.text) {
                if let Some(pg) = pronoun_gender {
                    if pg != entity_gender {
                        continue;
                    }
                }
            }
            
            // Found a compatible antecedent
            return entity.canonical_id;
        }
        
        None
    }

    /// Check if text is a pronoun.
    fn is_pronoun(&self, text: &str) -> bool {
        matches!(
            text.to_lowercase().as_str(),
            "he" | "she" | "it" | "they" | "him" | "her" | "them" |
            "his" | "hers" | "its" | "their" | "theirs" |
            "himself" | "herself" | "itself" | "themselves"
        )
    }

    /// Check if a pronoun is compatible with an entity type.
    fn pronoun_compatible(&self, pronoun: &str, entity_type: &EntityType) -> bool {
        let lower = pronoun.to_lowercase();
        match entity_type {
            EntityType::Person => matches!(lower.as_str(), 
                "he" | "she" | "they" | "him" | "her" | "them" |
                "his" | "hers" | "their" | "theirs" |
                "himself" | "herself" | "themselves"
            ),
            EntityType::Organization => matches!(lower.as_str(),
                "it" | "they" | "its" | "their" | "theirs" | "itself" | "themselves"
            ),
            EntityType::Location => matches!(lower.as_str(),
                "it" | "its" | "itself"
            ),
            _ => matches!(lower.as_str(), "it" | "its" | "itself"),
        }
    }

    /// Infer gender from text (heuristic).
    fn infer_gender(&self, text: &str) -> Option<char> {
        let lower = text.to_lowercase();
        match lower.as_str() {
            "he" | "him" | "his" | "himself" => Some('m'),
            "she" | "her" | "hers" | "herself" => Some('f'),
            _ => None, // Can't infer from name without external knowledge
        }
    }

    /// Normalize text to canonical form for matching.
    fn canonical_form(&self, text: &str, entity_type: &EntityType) -> String {
        let normalized = text
            .to_lowercase()
            .trim()
            .to_string();
        
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
        
        if words1.len() > 1 && words2.len() == 1 {
            if words1.last() == words2.first() {
                return true;
            }
        }
        if words2.len() > 1 && words1.len() == 1 {
            if words2.last() == words1.first() {
                return true;
            }
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
        Entity::new(text, EntityType::Organization, start, start + text.len(), 0.9)
    }

    #[test]
    fn test_exact_match() {
        let resolver = SimpleCorefResolver::default();
        let entities = vec![
            person("John Smith", 0),
            person("John Smith", 50),
        ];
        
        let resolved = resolver.resolve(&entities);
        assert_eq!(resolved[0].canonical_id, resolved[1].canonical_id);
    }

    #[test]
    fn test_substring_match() {
        let resolver = SimpleCorefResolver::default();
        let entities = vec![
            person("John Smith", 0),
            person("Smith", 50),
        ];
        
        let resolved = resolver.resolve(&entities);
        assert_eq!(resolved[0].canonical_id, resolved[1].canonical_id);
    }

    #[test]
    fn test_pronoun_resolution() {
        let resolver = SimpleCorefResolver::default();
        let entities = vec![
            person("John Smith", 0),
            person("he", 50),
        ];
        
        let resolved = resolver.resolve(&entities);
        assert_eq!(resolved[0].canonical_id, resolved[1].canonical_id);
    }

    #[test]
    fn test_different_entities() {
        let resolver = SimpleCorefResolver::default();
        let entities = vec![
            person("John Smith", 0),
            person("Mary Jones", 50),
        ];
        
        let resolved = resolver.resolve(&entities);
        assert_ne!(resolved[0].canonical_id, resolved[1].canonical_id);
    }

    #[test]
    fn test_type_matters() {
        let resolver = SimpleCorefResolver::default();
        let entities = vec![
            person("Apple", 0),  // Person named Apple
            org("Apple", 50),    // Apple Inc.
        ];
        
        let resolved = resolver.resolve(&entities);
        // Different types should NOT match
        assert_ne!(resolved[0].canonical_id, resolved[1].canonical_id);
    }

    #[test]
    fn test_resolve_to_chains() {
        let resolver = SimpleCorefResolver::default();
        let entities = vec![
            person("John", 0),
            person("he", 20),
            person("Mary", 40),
        ];
        
        let chains = resolver.resolve_to_chains(&entities);
        
        // John + he in one chain, Mary singleton
        assert_eq!(chains.len(), 2);
        
        let non_singletons: Vec<_> = chains.iter().filter(|c| !c.is_singleton()).collect();
        assert_eq!(non_singletons.len(), 1);
        assert_eq!(non_singletons[0].len(), 2);
    }
}

