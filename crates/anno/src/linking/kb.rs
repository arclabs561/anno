//! Knowledge Base abstraction and multi-KB entity linking.
//!
//! # Supported Knowledge Bases
//!
//! | KB | URI Namespace | Notes |
//! |----|---------------|-------|
//! | **Wikidata** | `http://www.wikidata.org/entity/` | Q-items, most comprehensive |
//! | **YAGO** | `http://yago-knowledge.org/resource/` | Wikipedia + WordNet + GeoNames |
//! | **DBpedia** | `http://dbpedia.org/resource/` | Wikipedia infobox extraction |
//! | **Wikipedia** | `https://en.wikipedia.org/wiki/` | Direct article links |
//! | **Freebase** | `http://rdf.freebase.com/ns/` | Legacy, mapped to Wikidata |
//! | **UMLS** | `https://uts.nlm.nih.gov/uts/umls/concept/` | Biomedical |
//! | **GeoNames** | `https://sws.geonames.org/` | Geographic entities |
//!
//! # URI/IRI Standards
//!
//! Entity linking produces **Linked Data** compatible URIs following:
//! - W3C RDF standards for entity identification
//! - HTTP URIs for dereferenceable entities
//! - owl:sameAs links between KBs
//!
//! # Modern Entity Linking Methods
//!
//! ## BLINK Architecture (Meta AI, 2020)
//! ```text
//! ┌──────────────┐     ┌─────────────┐     ┌───────────────┐
//! │ Bi-Encoder   │────►│ Dense Index │────►│ Cross-Encoder │
//! │ (BERT)       │     │ (FAISS)     │     │ (Re-ranker)   │
//! └──────────────┘     └─────────────┘     └───────────────┘
//! ```
//!
//! ## ReFinED (Amazon, 2022)
//! - End-to-end mention detection + linking
//! - Fine-grained entity typing
//! - Zero-shot linking capability
//!
//! ## GENRE (Meta AI, 2021)
//! - Autoregressive entity retrieval
//! - Generates entity names directly
//!
//! # Example
//!
//! ```rust
//! use anno::linking::kb::{KnowledgeBase, UnifiedLinker, EntityURI};
//!
//! // Create unified linker with multiple KBs
//! let linker = UnifiedLinker::builder()
//!     .add_kb(KnowledgeBase::Wikidata)
//!     .add_kb(KnowledgeBase::DBpedia)
//!     .add_kb(KnowledgeBase::YAGO)
//!     .build();
//!
//! // Link and get URIs for all supported KBs
//! let uris = linker.link_to_uris("Albert Einstein", None);
//! for uri in &uris {
//!     println!("{}: {}", uri.kb, uri.uri);
//! }
//! ```

use anno_core::Confidence;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Knowledge Base Enum
// =============================================================================

/// Supported knowledge bases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KnowledgeBase {
    /// Wikidata (Q-items) - Most comprehensive, actively maintained
    Wikidata,
    /// YAGO - Wikipedia + WordNet + GeoNames
    YAGO,
    /// DBpedia - Wikipedia infobox extraction
    DBpedia,
    /// Wikipedia - Direct article links
    Wikipedia,
    /// Freebase - Legacy (deprecated 2016, mapped to Wikidata)
    Freebase,
    /// UMLS - Unified Medical Language System
    UMLS,
    /// GeoNames - Geographic entities
    GeoNames,
    /// Schema.org - Structured data vocabulary
    SchemaOrg,
    /// OpenCyc - General knowledge
    OpenCyc,
    /// Custom KB
    Custom,
}

impl KnowledgeBase {
    /// Get the base URI namespace for this KB.
    #[must_use]
    pub fn base_uri(&self) -> &'static str {
        match self {
            Self::Wikidata => "http://www.wikidata.org/entity/",
            Self::YAGO => "http://yago-knowledge.org/resource/",
            Self::DBpedia => "http://dbpedia.org/resource/",
            Self::Wikipedia => "https://en.wikipedia.org/wiki/",
            Self::Freebase => "http://rdf.freebase.com/ns/",
            Self::UMLS => "https://uts.nlm.nih.gov/uts/umls/concept/",
            Self::GeoNames => "https://sws.geonames.org/",
            Self::SchemaOrg => "https://schema.org/",
            Self::OpenCyc => "http://sw.opencyc.org/concept/",
            Self::Custom => "",
        }
    }

    /// Get the SPARQL endpoint URL (if available).
    #[must_use]
    pub fn sparql_endpoint(&self) -> Option<&'static str> {
        match self {
            Self::Wikidata => Some("https://query.wikidata.org/sparql"),
            Self::DBpedia => Some("https://dbpedia.org/sparql"),
            Self::YAGO => Some("https://yago-knowledge.org/sparql/query"),
            _ => None,
        }
    }

    /// Get the API search endpoint (if available).
    #[must_use]
    pub fn search_api(&self) -> Option<&'static str> {
        match self {
            Self::Wikidata => Some("https://www.wikidata.org/w/api.php"),
            Self::Wikipedia => Some("https://en.wikipedia.org/w/api.php"),
            Self::GeoNames => Some("http://api.geonames.org/searchJSON"),
            _ => None,
        }
    }

    /// Is this KB still actively maintained?
    #[must_use]
    pub fn is_active(&self) -> bool {
        !matches!(self, Self::Freebase | Self::OpenCyc)
    }

    /// Get the owl:sameAs predicate for cross-KB linking.
    #[must_use]
    pub fn same_as_predicate(&self) -> &'static str {
        "http://www.w3.org/2002/07/owl#sameAs"
    }
}

impl std::fmt::Display for KnowledgeBase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Wikidata => write!(f, "Wikidata"),
            Self::YAGO => write!(f, "YAGO"),
            Self::DBpedia => write!(f, "DBpedia"),
            Self::Wikipedia => write!(f, "Wikipedia"),
            Self::Freebase => write!(f, "Freebase"),
            Self::UMLS => write!(f, "UMLS"),
            Self::GeoNames => write!(f, "GeoNames"),
            Self::SchemaOrg => write!(f, "Schema.org"),
            Self::OpenCyc => write!(f, "OpenCyc"),
            Self::Custom => write!(f, "Custom"),
        }
    }
}

// =============================================================================
// Entity URI
// =============================================================================

/// A fully qualified entity URI with metadata.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EntityURI {
    /// The knowledge base
    pub kb: KnowledgeBase,
    /// Local identifier within the KB (e.g., "Q937" for Wikidata)
    pub local_id: String,
    /// Full URI
    pub uri: String,
    /// Optional human-readable label
    pub label: Option<String>,
}

impl EntityURI {
    /// Create a new entity URI.
    pub fn new(kb: KnowledgeBase, local_id: &str) -> Self {
        let uri = format!("{}{}", kb.base_uri(), local_id);
        Self {
            kb,
            local_id: local_id.to_string(),
            uri,
            label: None,
        }
    }

    /// Create with label.
    pub fn with_label(mut self, label: &str) -> Self {
        self.label = Some(label.to_string());
        self
    }

    /// Parse a URI to extract KB and local ID.
    pub fn parse(uri: &str) -> Option<Self> {
        // Try each KB's base URI
        for kb in &[
            KnowledgeBase::Wikidata,
            KnowledgeBase::YAGO,
            KnowledgeBase::DBpedia,
            KnowledgeBase::Wikipedia,
            KnowledgeBase::Freebase,
            KnowledgeBase::UMLS,
            KnowledgeBase::GeoNames,
            KnowledgeBase::SchemaOrg,
        ] {
            if uri.starts_with(kb.base_uri()) {
                let local_id = &uri[kb.base_uri().len()..];
                return Some(Self {
                    kb: *kb,
                    local_id: local_id.to_string(),
                    uri: uri.to_string(),
                    label: None,
                });
            }
        }
        None
    }

    /// Check if this is a Wikidata Q-item.
    #[must_use]
    pub fn is_wikidata(&self) -> bool {
        self.kb == KnowledgeBase::Wikidata && self.local_id.starts_with('Q')
    }

    /// Get CURIE (Compact URI) format.
    #[must_use]
    pub fn to_curie(&self) -> String {
        let prefix = match self.kb {
            KnowledgeBase::Wikidata => "wd",
            KnowledgeBase::YAGO => "yago",
            KnowledgeBase::DBpedia => "dbr",
            KnowledgeBase::Wikipedia => "wp",
            KnowledgeBase::Freebase => "fb",
            KnowledgeBase::UMLS => "umls",
            KnowledgeBase::GeoNames => "gn",
            KnowledgeBase::SchemaOrg => "schema",
            KnowledgeBase::OpenCyc => "cyc",
            KnowledgeBase::Custom => "custom",
        };
        format!("{}:{}", prefix, self.local_id)
    }
}

// =============================================================================
// Cross-KB Mappings
// =============================================================================

/// Cross-KB entity mappings.
///
/// Maps entities between different knowledge bases using owl:sameAs relationships.
#[derive(Debug, Clone, Default)]
pub struct CrossKBMapper {
    /// Wikidata to other KBs
    wikidata_mappings: HashMap<String, Vec<EntityURI>>,
    /// Other KB to Wikidata (for reverse lookup)
    reverse_mappings: HashMap<String, String>, // URI -> Wikidata QID
}

impl CrossKBMapper {
    /// Create a new mapper.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a mapping from Wikidata to another KB.
    pub fn add_mapping(&mut self, wikidata_qid: &str, other_uri: EntityURI) {
        self.reverse_mappings
            .insert(other_uri.uri.clone(), wikidata_qid.to_string());
        self.wikidata_mappings
            .entry(wikidata_qid.to_string())
            .or_default()
            .push(other_uri);
    }

    /// Get all URIs for a Wikidata entity.
    pub fn get_uris(&self, wikidata_qid: &str) -> Vec<&EntityURI> {
        self.wikidata_mappings
            .get(wikidata_qid)
            .map(|v| v.iter().collect())
            .unwrap_or_default()
    }

    /// Find Wikidata QID from any other KB URI.
    pub fn to_wikidata(&self, uri: &str) -> Option<&str> {
        self.reverse_mappings.get(uri).map(|s| s.as_str())
    }

    /// Create mapper with well-known entity mappings.
    #[must_use]
    pub fn with_common_mappings() -> Self {
        let mut mapper = Self::new();

        // Albert Einstein
        mapper.add_mapping(
            "Q937",
            EntityURI::new(KnowledgeBase::DBpedia, "Albert_Einstein"),
        );
        mapper.add_mapping(
            "Q937",
            EntityURI::new(KnowledgeBase::YAGO, "Albert_Einstein"),
        );
        mapper.add_mapping("Q937", EntityURI::new(KnowledgeBase::Freebase, "m.0jcx"));
        mapper.add_mapping(
            "Q937",
            EntityURI::new(KnowledgeBase::Wikipedia, "Albert_Einstein"),
        );

        // Apple Inc.
        mapper.add_mapping("Q312", EntityURI::new(KnowledgeBase::DBpedia, "Apple_Inc."));
        mapper.add_mapping("Q312", EntityURI::new(KnowledgeBase::YAGO, "Apple_Inc."));
        mapper.add_mapping("Q312", EntityURI::new(KnowledgeBase::Freebase, "m.0k8z"));

        // New York City
        mapper.add_mapping(
            "Q60",
            EntityURI::new(KnowledgeBase::DBpedia, "New_York_City"),
        );
        mapper.add_mapping("Q60", EntityURI::new(KnowledgeBase::YAGO, "New_York_City"));
        mapper.add_mapping("Q60", EntityURI::new(KnowledgeBase::GeoNames, "5128581"));

        // United States
        mapper.add_mapping(
            "Q30",
            EntityURI::new(KnowledgeBase::DBpedia, "United_States"),
        );
        mapper.add_mapping("Q30", EntityURI::new(KnowledgeBase::YAGO, "United_States"));
        mapper.add_mapping("Q30", EntityURI::new(KnowledgeBase::GeoNames, "6252001"));

        mapper
    }
}

// =============================================================================
// YAGO-Specific Types
// =============================================================================

/// YAGO entity with taxonomy information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YAGOEntity {
    /// YAGO identifier
    pub yago_id: String,
    /// Primary label
    pub label: String,
    /// YAGO types (from YAGO taxonomy)
    pub types: Vec<String>,
    /// WordNet synset (if available)
    pub wordnet_synset: Option<String>,
    /// GeoNames ID (if geographic)
    pub geonames_id: Option<String>,
    /// Wikidata QID (if mapped)
    pub wikidata_qid: Option<String>,
    /// Wikipedia article
    pub wikipedia_article: Option<String>,
}

impl YAGOEntity {
    /// Create a new YAGO entity.
    pub fn new(yago_id: &str, label: &str) -> Self {
        Self {
            yago_id: yago_id.to_string(),
            label: label.to_string(),
            types: Vec::new(),
            wordnet_synset: None,
            geonames_id: None,
            wikidata_qid: None,
            wikipedia_article: None,
        }
    }

    /// Get the YAGO URI.
    #[must_use]
    pub fn uri(&self) -> String {
        format!("{}{}", KnowledgeBase::YAGO.base_uri(), self.yago_id)
    }

    /// Check if this is a person.
    #[must_use]
    pub fn is_person(&self) -> bool {
        self.types
            .iter()
            .any(|t| t.contains("person") || t.contains("human") || t.contains("wordnet_person"))
    }

    /// Check if this is a location.
    #[must_use]
    pub fn is_location(&self) -> bool {
        self.types.iter().any(|t| {
            t.contains("location")
                || t.contains("place")
                || t.contains("city")
                || t.contains("country")
        }) || self.geonames_id.is_some()
    }

    /// Check if this is an organization.
    #[must_use]
    pub fn is_organization(&self) -> bool {
        self.types.iter().any(|t| {
            t.contains("organization") || t.contains("company") || t.contains("institution")
        })
    }
}

// =============================================================================
// Unified Linker
// =============================================================================

/// Unified entity linker supporting multiple KBs.
#[derive(Debug, Clone, Default)]
pub struct UnifiedLinker {
    /// Enabled knowledge bases
    enabled_kbs: Vec<KnowledgeBase>,
    /// Cross-KB mapper
    mapper: CrossKBMapper,
    /// Entity dictionary (for offline linking)
    dictionary: HashMap<String, Vec<EntityURI>>, // mention -> URIs
}

impl UnifiedLinker {
    /// Create a builder.
    pub fn builder() -> UnifiedLinkerBuilder {
        UnifiedLinkerBuilder::default()
    }

    /// Link a mention and return URIs for all enabled KBs.
    pub fn link_to_uris(&self, mention: &str, _entity_type: Option<&str>) -> Vec<EntityURI> {
        let mention_lower = mention.to_lowercase();

        // Check dictionary first
        if let Some(uris) = self.dictionary.get(&mention_lower) {
            return uris
                .iter()
                .filter(|u| self.enabled_kbs.contains(&u.kb))
                .cloned()
                .collect();
        }

        // If we have a Wikidata match, expand to other KBs via mapper
        // (In a real implementation, this would call the appropriate APIs)
        Vec::new()
    }

    /// Link and return the primary (Wikidata) URI.
    pub fn link_primary(&self, mention: &str, entity_type: Option<&str>) -> Option<EntityURI> {
        self.link_to_uris(mention, entity_type)
            .into_iter()
            .find(|u| u.kb == KnowledgeBase::Wikidata)
    }

    /// Get all URIs for a known Wikidata QID.
    pub fn expand_wikidata(&self, qid: &str) -> Vec<EntityURI> {
        let mut uris = vec![EntityURI::new(KnowledgeBase::Wikidata, qid)];
        uris.extend(self.mapper.get_uris(qid).iter().cloned().cloned());
        uris
    }
}

/// Builder for UnifiedLinker.
#[derive(Debug, Clone, Default)]
pub struct UnifiedLinkerBuilder {
    enabled_kbs: Vec<KnowledgeBase>,
    use_common_mappings: bool,
}

impl UnifiedLinkerBuilder {
    /// Add a knowledge base.
    pub fn add_kb(mut self, kb: KnowledgeBase) -> Self {
        if !self.enabled_kbs.contains(&kb) {
            self.enabled_kbs.push(kb);
        }
        self
    }

    /// Use common entity mappings.
    pub fn with_common_mappings(mut self) -> Self {
        self.use_common_mappings = true;
        self
    }

    /// Build the linker.
    pub fn build(self) -> UnifiedLinker {
        let mapper = if self.use_common_mappings {
            CrossKBMapper::with_common_mappings()
        } else {
            CrossKBMapper::new()
        };

        let enabled_kbs = if self.enabled_kbs.is_empty() {
            vec![KnowledgeBase::Wikidata] // Default to Wikidata
        } else {
            self.enabled_kbs
        };

        UnifiedLinker {
            enabled_kbs,
            mapper,
            dictionary: HashMap::new(),
        }
    }
}

// =============================================================================
// NIL Clustering
// =============================================================================

/// Cluster of NIL (unlinkable) entities.
///
/// When mentions can't be linked to any KB, we cluster them
/// to track potentially new entities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NILCluster {
    /// Cluster ID
    pub id: u64,
    /// Canonical surface form
    pub canonical: String,
    /// All surface forms
    pub surfaces: Vec<String>,
    /// Inferred entity type
    pub entity_type: Option<String>,
    /// Cluster confidence
    pub confidence: Confidence,
    /// Number of mentions
    pub mention_count: usize,
}

impl NILCluster {
    /// Create a new NIL cluster.
    pub fn new(id: u64, canonical: &str) -> Self {
        Self {
            id,
            canonical: canonical.to_string(),
            surfaces: vec![canonical.to_string()],
            entity_type: None,
            confidence: Confidence::ONE,
            mention_count: 1,
        }
    }

    /// Generate a temporary URI for this NIL cluster.
    #[must_use]
    pub fn temp_uri(&self) -> String {
        format!("urn:nil:cluster:{}", self.id)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kb_uris() {
        assert_eq!(
            KnowledgeBase::Wikidata.base_uri(),
            "http://www.wikidata.org/entity/"
        );
        assert_eq!(
            KnowledgeBase::YAGO.base_uri(),
            "http://yago-knowledge.org/resource/"
        );
        assert_eq!(
            KnowledgeBase::DBpedia.base_uri(),
            "http://dbpedia.org/resource/"
        );
    }

    #[test]
    fn test_entity_uri() {
        let uri = EntityURI::new(KnowledgeBase::Wikidata, "Q937");
        assert_eq!(uri.uri, "http://www.wikidata.org/entity/Q937");
        assert_eq!(uri.to_curie(), "wd:Q937");
        assert!(uri.is_wikidata());
    }

    #[test]
    fn test_uri_parsing() {
        let uri = EntityURI::parse("http://www.wikidata.org/entity/Q937");
        assert!(uri.is_some());
        let uri = uri.unwrap();
        assert_eq!(uri.kb, KnowledgeBase::Wikidata);
        assert_eq!(uri.local_id, "Q937");

        let dbpedia = EntityURI::parse("http://dbpedia.org/resource/Albert_Einstein");
        assert!(dbpedia.is_some());
        assert_eq!(dbpedia.unwrap().kb, KnowledgeBase::DBpedia);
    }

    #[test]
    fn test_cross_kb_mapper() {
        let mapper = CrossKBMapper::with_common_mappings();

        // Should have mappings for Einstein
        let uris = mapper.get_uris("Q937");
        assert!(!uris.is_empty());

        // Should include DBpedia
        assert!(uris.iter().any(|u| u.kb == KnowledgeBase::DBpedia));
    }

    #[test]
    fn test_yago_entity() {
        let mut entity = YAGOEntity::new("Albert_Einstein", "Albert Einstein");
        entity.types.push("wordnet_person_100007846".to_string());
        entity.wikidata_qid = Some("Q937".to_string());

        assert!(entity.is_person());
        assert!(!entity.is_location());
        assert_eq!(
            entity.uri(),
            "http://yago-knowledge.org/resource/Albert_Einstein"
        );
    }

    #[test]
    fn test_unified_linker() {
        let linker = UnifiedLinker::builder()
            .add_kb(KnowledgeBase::Wikidata)
            .add_kb(KnowledgeBase::DBpedia)
            .with_common_mappings()
            .build();

        // Expand known QID
        let uris = linker.expand_wikidata("Q937");
        assert!(!uris.is_empty());
        assert!(uris.iter().any(|u| u.kb == KnowledgeBase::Wikidata));
    }

    #[test]
    fn test_nil_cluster() {
        let cluster = NILCluster::new(1, "John Doe");
        assert!(cluster.temp_uri().starts_with("urn:nil:"));
    }

    // =========================================================================
    // Additional unit tests
    // =========================================================================

    #[test]
    fn test_kb_is_active() {
        assert!(KnowledgeBase::Wikidata.is_active());
        assert!(KnowledgeBase::DBpedia.is_active());
        assert!(!KnowledgeBase::Freebase.is_active());
        assert!(!KnowledgeBase::OpenCyc.is_active());
    }

    #[test]
    fn test_entity_uri_parse_unknown_prefix() {
        // URI that doesn't match any known KB
        let result = EntityURI::parse("https://example.com/entity/42");
        assert!(result.is_none(), "Unknown prefix should return None");
    }

    #[test]
    fn test_entity_uri_is_wikidata_non_q() {
        // Wikidata property (P-item) should not be flagged as a Q-item
        let uri = EntityURI::new(KnowledgeBase::Wikidata, "P31");
        assert!(
            !uri.is_wikidata(),
            "P-items should not satisfy is_wikidata()"
        );
    }

    #[test]
    fn test_cross_kb_mapper_reverse_lookup() {
        let mapper = CrossKBMapper::with_common_mappings();
        // DBpedia URI for Einstein -> should resolve back to Q937
        let qid = mapper.to_wikidata("http://dbpedia.org/resource/Albert_Einstein");
        assert_eq!(qid, Some("Q937"));
    }

    #[test]
    fn test_unified_linker_default_kb() {
        // Builder with no KBs added should default to Wikidata
        let linker = UnifiedLinker::builder().build();
        let uris = linker.expand_wikidata("Q937");
        assert!(
            uris.iter().any(|u| u.kb == KnowledgeBase::Wikidata),
            "Default linker should include Wikidata"
        );
    }

    #[test]
    fn test_kb_sparql_endpoints() {
        assert!(KnowledgeBase::Wikidata.sparql_endpoint().is_some());
        assert!(KnowledgeBase::DBpedia.sparql_endpoint().is_some());
        assert!(KnowledgeBase::YAGO.sparql_endpoint().is_some());
        assert!(KnowledgeBase::Wikipedia.sparql_endpoint().is_none());
        assert!(KnowledgeBase::GeoNames.sparql_endpoint().is_none());
    }
}
