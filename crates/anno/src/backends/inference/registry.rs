//! Pre-computed label embeddings and semantic registry for zero-shot NER.
//!
//! `SemanticRegistry` stores relation labels and class embeddings used by
//! the encoder-based extraction pipeline.

use crate::Confidence;
use std::collections::HashMap;

// Semantic Registry (Pre-computed Label Embeddings)
// =============================================================================

/// A frozen, pre-computed registry of entity and relation types.
///
/// # Motivation
///
/// The `SemanticRegistry` is the "knowledge base" of a bi-encoder NER system.
/// It stores pre-computed embeddings for all entity/relation types, enabling:
///
/// - **Zero-shot**: Add new types without retraining
/// - **Speed**: Encode labels once, reuse forever
/// - **Semantics**: Rich descriptions enable better matching
///
/// # Architecture
///
/// ```text
/// ┌────────────────────────────────────────────────────────────────┐
/// │                     SemanticRegistry                           │
/// ├────────────────────────────────────────────────────────────────┤
/// │  labels: [                                                     │
/// │    { slug: "person", description: "named individual human" }   │
/// │    { slug: "organization", description: "company or group" }   │
/// │    { slug: "CEO_OF", description: "leads organization" }       │
/// │  ]                                                             │
/// │                                                                │
/// │  embeddings: [768 floats] [768 floats] [768 floats]            │
/// │              └────┬────┘  └────┬────┘  └────┬────┘             │
/// │                   ▲            ▲            ▲                  │
/// │              person        organization   CEO_OF               │
/// │                                                                │
/// │  label_index: { "person" → 0, "organization" → 1, ... }        │
/// └────────────────────────────────────────────────────────────────┘
/// ```
///
/// # Bi-Encoder Efficiency
///
/// The key insight from GLiNER is that label embeddings can be computed once
/// and reused across all inference requests:
///
/// | Approach | Cost per query | Benefit |
/// |----------|----------------|---------|
/// | Cross-encoder | O(N × L) | Better accuracy |
/// | Bi-encoder | O(N) + O(L) | Much faster, labels cached |
///
/// # Example
///
/// ```ignore
/// use anno::SemanticRegistry;
///
/// // Build registry (expensive, do once at startup)
/// let registry = SemanticRegistry::builder()
///     .add_entity("person", "A named individual human being")
///     .add_entity("organization", "A company, institution, or organized group")
///     .add_relation("CEO_OF", "Chief executive officer of an organization")
///     .build(&label_encoder)?;
///
/// // Use registry for all inference (cheap, cached embeddings)
/// for document in documents {
///     let entities = engine.extract(&document, &registry)?;
/// }
/// ```
///
/// # Adding Custom Types
///
/// ```ignore
/// // Domain-specific medical entities
/// let medical_registry = SemanticRegistry::builder()
///     .add_entity("drug", "A pharmaceutical compound or medication")
///     .add_entity("disease", "A medical condition or illness")
///     .add_entity("gene", "A genetic sequence encoding a protein")
///     .add_relation("TREATS", "Drug is used to treat disease")
///     .add_relation("CAUSES", "Factor causes or leads to condition")
///     .build(&label_encoder)?;
/// ```
#[derive(Debug, Clone)]
pub struct SemanticRegistry {
    /// Pre-computed embeddings for all labels.
    /// Shape: [num_labels, hidden_dim]
    /// Stored as flattened f32 for simplicity without tensor deps.
    pub embeddings: Vec<f32>,
    /// Hidden dimension of embeddings
    pub hidden_dim: usize,
    /// Metadata for each label (index corresponds to embedding row)
    pub labels: Vec<LabelDefinition>,
    /// Index mapping from label slug to embedding row
    pub label_index: HashMap<String, usize>,
}

/// Definition of a semantic label (entity type or relation type).
#[derive(Debug, Clone)]
pub struct LabelDefinition {
    /// Unique identifier (e.g., "person", "CEO_OF")
    pub slug: String,
    /// Human-readable description (used for encoding)
    pub description: String,
    /// Category: Entity or Relation
    pub category: LabelCategory,
    /// Expected source modality
    pub modality: ModalityHint,
    /// Minimum confidence threshold for this label
    pub threshold: Confidence,
}

/// Category of semantic label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LabelCategory {
    /// Named entity (Person, Organization, Location, etc.)
    Entity,
    /// Relation between entities (CEO_OF, LOCATED_IN, etc.)
    Relation,
    /// Attribute of an entity (date of birth, revenue, etc.)
    Attribute,
}

/// Hint for which modality this label applies to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ModalityHint {
    /// Text-only (most entity types)
    #[default]
    TextOnly,
    /// Works with both text and visual
    Any,
}

impl SemanticRegistry {
    /// Create a builder for constructing a registry.
    pub fn builder() -> SemanticRegistryBuilder {
        SemanticRegistryBuilder::new()
    }

    /// Get number of labels in the registry.
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    /// Check if registry is empty.
    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    /// Get embedding for a label by slug.
    pub fn get_embedding(&self, slug: &str) -> Option<&[f32]> {
        let idx = self.label_index.get(slug)?;
        let start = idx * self.hidden_dim;
        let end = start + self.hidden_dim;
        if end <= self.embeddings.len() {
            Some(&self.embeddings[start..end])
        } else {
            None
        }
    }

    /// Get all entity labels (excluding relations).
    pub fn entity_labels(&self) -> impl Iterator<Item = &LabelDefinition> {
        self.labels
            .iter()
            .filter(|l| l.category == LabelCategory::Entity)
    }

    /// Get all relation labels.
    pub fn relation_labels(&self) -> impl Iterator<Item = &LabelDefinition> {
        self.labels
            .iter()
            .filter(|l| l.category == LabelCategory::Relation)
    }

    /// Create a standard NER registry with common entity types.
    pub fn standard_ner(hidden_dim: usize) -> Self {
        // Placeholder embeddings - in real use, these would be encoder outputs
        let labels = vec![
            LabelDefinition {
                slug: "person".into(),
                description: "A named individual human being".into(),
                category: LabelCategory::Entity,
                modality: ModalityHint::TextOnly,
                threshold: Confidence::new(0.5),
            },
            LabelDefinition {
                slug: "organization".into(),
                description: "A company, institution, agency, or other group".into(),
                category: LabelCategory::Entity,
                modality: ModalityHint::TextOnly,
                threshold: Confidence::new(0.5),
            },
            LabelDefinition {
                slug: "location".into(),
                description: "A geographical place, city, country, or region".into(),
                category: LabelCategory::Entity,
                modality: ModalityHint::TextOnly,
                threshold: Confidence::new(0.5),
            },
            LabelDefinition {
                slug: "date".into(),
                description: "A calendar date or time expression".into(),
                category: LabelCategory::Entity,
                modality: ModalityHint::TextOnly,
                threshold: Confidence::new(0.5),
            },
            LabelDefinition {
                slug: "money".into(),
                description: "A monetary amount with currency".into(),
                category: LabelCategory::Entity,
                modality: ModalityHint::TextOnly,
                threshold: Confidence::new(0.5),
            },
        ];

        let num_labels = labels.len();
        let label_index: HashMap<String, usize> = labels
            .iter()
            .enumerate()
            .map(|(i, l)| (l.slug.clone(), i))
            .collect();

        // Initialize with zeros (placeholder)
        let embeddings = vec![0.0f32; num_labels * hidden_dim];

        Self {
            embeddings,
            hidden_dim,
            labels,
            label_index,
        }
    }
}

/// Builder for SemanticRegistry.
#[derive(Debug, Default)]
pub struct SemanticRegistryBuilder {
    labels: Vec<LabelDefinition>,
}

impl SemanticRegistryBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an entity type.
    pub fn add_entity(mut self, slug: &str, description: &str) -> Self {
        self.labels.push(LabelDefinition {
            slug: slug.into(),
            description: description.into(),
            category: LabelCategory::Entity,
            modality: ModalityHint::TextOnly,
            threshold: Confidence::new(0.5),
        });
        self
    }

    /// Add a relation type.
    pub fn add_relation(mut self, slug: &str, description: &str) -> Self {
        self.labels.push(LabelDefinition {
            slug: slug.into(),
            description: description.into(),
            category: LabelCategory::Relation,
            modality: ModalityHint::TextOnly,
            threshold: Confidence::new(0.5),
        });
        self
    }

    /// Add a label with full configuration.
    pub fn add_label(mut self, label: LabelDefinition) -> Self {
        self.labels.push(label);
        self
    }

    /// Build the registry with zero-initialized embeddings.
    ///
    /// Useful for tests and for pipelines that populate embeddings lazily.
    /// For production use with a real encoder, prefer `build_with_encoder`.
    pub fn build_zero(self, hidden_dim: usize) -> SemanticRegistry {
        let num_labels = self.labels.len();
        let label_index: HashMap<String, usize> = self
            .labels
            .iter()
            .enumerate()
            .map(|(i, l)| (l.slug.clone(), i))
            .collect();

        SemanticRegistry {
            embeddings: vec![0.0f32; num_labels * hidden_dim],
            hidden_dim,
            labels: self.labels,
            label_index,
        }
    }
}

// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // SemanticRegistry::standard_ner
    // =========================================================================

    #[test]
    fn standard_ner_has_five_labels() {
        let reg = SemanticRegistry::standard_ner(4);
        assert_eq!(reg.len(), 5);
        assert!(!reg.is_empty());
    }

    #[test]
    fn standard_ner_label_index_consistent() {
        let reg = SemanticRegistry::standard_ner(8);
        for (i, label) in reg.labels.iter().enumerate() {
            assert_eq!(
                reg.label_index.get(&label.slug),
                Some(&i),
                "label_index[{slug}] should map to {i}",
                slug = label.slug,
            );
        }
    }

    #[test]
    fn standard_ner_embedding_dimensions() {
        let dim = 16;
        let reg = SemanticRegistry::standard_ner(dim);
        assert_eq!(reg.hidden_dim, dim);
        assert_eq!(reg.embeddings.len(), reg.len() * dim);
    }

    #[test]
    fn standard_ner_all_entity_category() {
        let reg = SemanticRegistry::standard_ner(4);
        for label in &reg.labels {
            assert_eq!(label.category, LabelCategory::Entity);
        }
    }

    // =========================================================================
    // Embedding retrieval
    // =========================================================================

    #[test]
    fn get_embedding_returns_correct_slice() {
        let dim = 3;
        let mut reg = SemanticRegistry::standard_ner(dim);
        // Write recognizable values into the "organization" slot (index 1).
        let idx = reg.label_index["organization"];
        let start = idx * dim;
        reg.embeddings[start] = 1.0;
        reg.embeddings[start + 1] = 2.0;
        reg.embeddings[start + 2] = 3.0;

        let emb = reg.get_embedding("organization").unwrap();
        assert_eq!(emb, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn get_embedding_returns_none_for_unknown() {
        let reg = SemanticRegistry::standard_ner(4);
        assert!(reg.get_embedding("nonexistent").is_none());
    }

    // =========================================================================
    // Filtered iterators
    // =========================================================================

    #[test]
    fn entity_and_relation_iterators() {
        let reg = SemanticRegistry::builder()
            .add_entity("person", "a human")
            .add_relation("CEO_OF", "chief executive of")
            .add_entity("org", "an organization")
            .build_zero(4);

        let entities: Vec<_> = reg.entity_labels().collect();
        let relations: Vec<_> = reg.relation_labels().collect();

        assert_eq!(entities.len(), 2);
        assert_eq!(relations.len(), 1);
        assert_eq!(relations[0].slug, "CEO_OF");
    }

    // =========================================================================
    // Builder
    // =========================================================================

    #[test]
    fn builder_empty_produces_empty_registry() {
        let reg = SemanticRegistryBuilder::new().build_zero(8);
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
        assert_eq!(reg.embeddings.len(), 0);
    }

    #[test]
    fn builder_add_label_custom() {
        let label = LabelDefinition {
            slug: "drug".into(),
            description: "a pharmaceutical compound".into(),
            category: LabelCategory::Entity,
            modality: ModalityHint::Any,
            threshold: Confidence::new(0.3),
        };
        let reg = SemanticRegistry::builder().add_label(label).build_zero(2);
        assert_eq!(reg.len(), 1);
        assert_eq!(reg.labels[0].modality, ModalityHint::Any);
        assert!((reg.labels[0].threshold.value() - 0.3).abs() < f64::EPSILON);
    }

    #[test]
    fn builder_embeddings_are_zeroed() {
        let reg = SemanticRegistry::builder()
            .add_entity("a", "desc a")
            .add_entity("b", "desc b")
            .build_zero(4);
        assert!(reg.embeddings.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn builder_preserves_insertion_order() {
        let reg = SemanticRegistry::builder()
            .add_entity("alpha", "first")
            .add_relation("BETA", "second")
            .add_entity("gamma", "third")
            .build_zero(2);

        let slugs: Vec<&str> = reg.labels.iter().map(|l| l.slug.as_str()).collect();
        assert_eq!(slugs, vec!["alpha", "BETA", "gamma"]);
    }

    // =========================================================================
    // LabelCategory / ModalityHint
    // =========================================================================

    #[test]
    fn modality_hint_default_is_text_only() {
        assert_eq!(ModalityHint::default(), ModalityHint::TextOnly);
    }

    #[test]
    fn label_category_equality() {
        assert_eq!(LabelCategory::Entity, LabelCategory::Entity);
        assert_ne!(LabelCategory::Entity, LabelCategory::Relation);
        assert_ne!(LabelCategory::Relation, LabelCategory::Attribute);
    }
}
