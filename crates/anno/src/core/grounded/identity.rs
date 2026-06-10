use super::super::confidence::Confidence;
use super::super::types::{IdentityId, TypeLabel};
use super::track::TrackRef;
use serde::{Deserialize, Serialize};

// Re-export IdentityId so callers can use super::identity::IdentityId
pub use super::super::types::IdentityId as _IdentityIdReexport;

/// Source of identity formation.
///
/// Tracks how an identity was created, which affects how it should be
/// used and what operations are valid on it.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IdentitySource {
    /// Created from cross-document track clustering (inter-doc coref).
    /// No KB link yet - this is pure clustering.
    CrossDocCoref {
        /// Tracks that were clustered to form this identity
        track_refs: Vec<TrackRef>,
    },
    /// Linked from knowledge base (entity linking/NED).
    /// Single track or identity linked to KB.
    KnowledgeBase {
        /// Knowledge base name (e.g., "wikidata")
        kb_name: String,
        /// Knowledge base ID (e.g., "Q7186")
        kb_id: String,
    },
    /// Both: clustered from tracks AND linked to KB.
    /// This is the most complete identity.
    Hybrid {
        /// Tracks that were clustered
        track_refs: Vec<TrackRef>,
        /// Knowledge base name
        kb_name: String,
        /// Knowledge base ID
        kb_id: String,
    },
}

/// A global identity: a real-world entity linked to a knowledge base.
///
/// # The Modal Gap
///
/// There's a fundamental representational gap between:
/// - **Text mentions**: Contextual, variable surface forms ("Marie Curie", "she", "the scientist")
/// - **KB entities**: Canonical, static representations (Q7186 in Wikidata)
///
/// Bridging this gap requires:
/// 1. Learning aligned embeddings (text encoder ↔ KB encoder)
/// 2. Type consistency constraints
/// 3. Cross-encoder re-ranking for hard cases
///
/// # Design Philosophy
///
/// Identities are the "global truth" that tracks point to. They represent:
/// - A canonical name and description
/// - A knowledge base reference (if available)
/// - An embedding in the entity space (for similarity/clustering)
///
/// Identities can exist without KB links (for novel entities not in the KB).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Identity {
    /// Unique identifier
    pub id: IdentityId,
    /// Canonical name (the "official" name)
    pub canonical_name: String,
    /// Entity type/category.
    ///
    /// Stored as a `TypeLabel` to support both core and custom (domain) labels.
    pub entity_type: Option<TypeLabel>,
    /// Knowledge base reference (e.g., "Q7186" for Wikidata)
    pub kb_id: Option<String>,
    /// Knowledge base name (e.g., "wikidata", "umls")
    pub kb_name: Option<String>,
    /// Description from knowledge base
    pub description: Option<String>,
    /// Entity embedding in the KB/entity space
    /// This is aligned with the text encoder space for similarity computation
    pub embedding: Option<Vec<f32>>,
    /// Alias names (other known surface forms)
    pub aliases: Vec<String>,
    /// Confidence that this identity is correctly resolved
    pub confidence: Confidence,
    /// Source of identity formation (how it was created)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<IdentitySource>,
}

impl Identity {
    /// Create a new identity.
    #[must_use]
    pub fn new(id: impl Into<IdentityId>, canonical_name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            canonical_name: canonical_name.into(),
            entity_type: None,
            kb_id: None,
            kb_name: None,
            description: None,
            embedding: None,
            aliases: Vec::new(),
            confidence: Confidence::ONE,
            source: None,
        }
    }

    /// Create an identity from a knowledge base entry.
    #[must_use]
    pub fn from_kb(
        id: impl Into<IdentityId>,
        canonical_name: impl Into<String>,
        kb_name: impl Into<String>,
        kb_id: impl Into<String>,
    ) -> Self {
        let kb_name_str = kb_name.into();
        let kb_id_str = kb_id.into();
        Self {
            id: id.into(),
            canonical_name: canonical_name.into(),
            entity_type: None,
            kb_id: Some(kb_id_str.clone()),
            kb_name: Some(kb_name_str.clone()),
            description: None,
            embedding: None,
            aliases: Vec::new(),
            confidence: Confidence::ONE,
            source: Some(IdentitySource::KnowledgeBase {
                kb_name: kb_name_str,
                kb_id: kb_id_str,
            }),
        }
    }

    /// Add an alias.
    pub fn add_alias(&mut self, alias: impl Into<String>) {
        self.aliases.push(alias.into());
    }

    /// Get the identity's unique identifier.
    #[must_use]
    pub const fn id(&self) -> IdentityId {
        self.id
    }

    /// Get the canonical name.
    #[must_use]
    pub fn canonical_name(&self) -> &str {
        &self.canonical_name
    }

    /// Get the KB ID, if linked.
    #[must_use]
    pub fn kb_id(&self) -> Option<&str> {
        self.kb_id.as_deref()
    }

    /// Get the KB name, if linked.
    #[must_use]
    pub fn kb_name(&self) -> Option<&str> {
        self.kb_name.as_deref()
    }

    /// Get the aliases.
    #[must_use]
    pub fn aliases(&self) -> &[String] {
        &self.aliases
    }

    /// Get the confidence score.
    #[must_use]
    pub const fn confidence(&self) -> Confidence {
        self.confidence
    }

    /// Set the confidence score.
    pub fn set_confidence(&mut self, confidence: f32) {
        self.confidence = Confidence::new(confidence as f64);
    }

    /// Get the identity source.
    #[must_use]
    pub fn source(&self) -> Option<&IdentitySource> {
        self.source.as_ref()
    }

    /// Set the embedding.
    #[must_use]
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Set the entity type from a string.
    ///
    /// For new code, prefer [`Self::with_type_label`] which provides type safety.
    #[must_use]
    pub fn with_type(mut self, entity_type: impl Into<String>) -> Self {
        let s = entity_type.into();
        self.entity_type = Some(TypeLabel::from(s.as_str()));
        self
    }

    /// Set the entity type using a type-safe label.
    ///
    /// This is the preferred method for new code as it provides type safety
    /// and integrates with the core `EntityType` taxonomy.
    #[must_use]
    pub fn with_type_label(mut self, label: TypeLabel) -> Self {
        self.entity_type = Some(label);
        self
    }

    /// Get the entity type as a type-safe label.
    ///
    /// This converts the internal string representation to a `TypeLabel`,
    /// attempting to parse it as a core `EntityType` first.
    #[must_use]
    pub fn type_label(&self) -> Option<TypeLabel> {
        self.entity_type.clone()
    }

    /// Set description.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    // Note: from_cross_doc_cluster moved to anno crate (see anno/src/eval/cdcr.rs)
}
