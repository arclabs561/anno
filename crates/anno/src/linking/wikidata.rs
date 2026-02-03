//! Wikidata utilities (offline).
//!
//! This module is intentionally **offline**: it does not call Wikidata’s APIs.
//! It provides:
//! - a small in-memory dictionary (`WikidataDictionary`) for demos/tests
//! - type mapping helpers (`WikidataTypeMapper`, `WikidataNERType`)
//!
//! # Example
//!
//! ```rust
//! use anno::linking::wikidata::WikidataDictionary;
//!
//! let dict = WikidataDictionary::with_common_entities();
//! let cands = dict.lookup("Einstein");
//! assert!(!cands.is_empty());
//! ```
//!
//! # Wikidata Type Mapping
//!
//! ```text
//! Wikidata Instance-of        → NER Type
//! ─────────────────────────────────────────
//! Q5 (human)                  → PER
//! Q43229 (organization)       → ORG
//! Q4830453 (business)         → ORG
//! Q515 (city)                 → LOC
//! Q6256 (country)             → LOC/GPE
//! Q35127 (website)            → PRODUCT
//! Q571 (book)                 → WORK_OF_ART
//! Q11424 (film)               → WORK_OF_ART
//! ```
//!
//! Note: if you need real Wikidata API integration, treat it as an external dependency and
//! keep network behavior explicit in your application.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for Wikidata linking.
#[derive(Debug, Clone)]
pub struct WikidataConfig {
    /// Wikidata API endpoint
    pub api_endpoint: String,
    /// Maximum candidates to retrieve
    pub max_candidates: usize,
    /// Minimum search score threshold
    pub min_score: f64,
    /// Languages for label retrieval (priority order)
    pub languages: Vec<String>,
    /// Request timeout in seconds
    pub timeout_secs: u64,
    /// Enable caching
    pub enable_cache: bool,
    /// Cache TTL in seconds
    pub cache_ttl: u64,
}

impl Default for WikidataConfig {
    fn default() -> Self {
        Self {
            api_endpoint: "https://www.wikidata.org/w/api.php".to_string(),
            max_candidates: 10,
            min_score: 0.0,
            languages: vec!["en".to_string(), "de".to_string(), "fr".to_string()],
            timeout_secs: 10,
            enable_cache: true,
            cache_ttl: 3600, // 1 hour
        }
    }
}

// =============================================================================
// Entity Types
// =============================================================================

/// A Wikidata entity (Q-item).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WikidataEntity {
    /// Q-identifier (e.g., "Q937")
    pub qid: String,
    /// Primary label in preferred language
    pub label: String,
    /// Short description
    pub description: Option<String>,
    /// Alternative names/aliases
    pub aliases: Vec<String>,
    /// Instance-of types (Q-IDs)
    pub instance_of: Vec<String>,
    /// Subclass-of types (Q-IDs)  
    pub subclass_of: Vec<String>,
    /// Number of Wikipedia sitelinks (popularity proxy)
    pub sitelinks: u32,
    /// Mapped NER entity type
    pub entity_type: Option<WikidataNERType>,
    /// Wikipedia URL (if available)
    pub wikipedia_url: Option<String>,
    /// Image URL (if available)
    pub image_url: Option<String>,
}

impl WikidataEntity {
    /// Create a new entity.
    pub fn new(qid: &str, label: &str) -> Self {
        Self {
            qid: qid.to_string(),
            label: label.to_string(),
            description: None,
            aliases: Vec::new(),
            instance_of: Vec::new(),
            subclass_of: Vec::new(),
            sitelinks: 0,
            entity_type: None,
            wikipedia_url: None,
            image_url: None,
        }
    }

    /// Get the Wikidata IRI.
    #[must_use]
    pub fn iri(&self) -> String {
        format!("http://www.wikidata.org/entity/{}", self.qid)
    }

    /// Check if entity matches a mention (label or alias).
    #[must_use]
    pub fn matches_mention(&self, mention: &str) -> bool {
        let mention_lower = mention.to_lowercase();

        if self.label.to_lowercase() == mention_lower {
            return true;
        }

        self.aliases
            .iter()
            .any(|a| a.to_lowercase() == mention_lower)
    }
}

/// Mapped NER type from Wikidata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WikidataNERType {
    /// Person (Q5: human)
    Person,
    /// Organization (Q43229, Q4830453, etc.)
    Organization,
    /// Location (Q515, Q6256, etc.)
    Location,
    /// Geopolitical entity (Q6256, Q3624078)
    GeopoliticalEntity,
    /// Event (Q1656682, Q18669875)
    Event,
    /// Product (Q2424752)
    Product,
    /// Work of art (Q838948)
    WorkOfArt,
    /// Date/time entity
    DateTime,
    /// Miscellaneous/other
    Miscellaneous,
}

impl WikidataNERType {
    /// Convert to standard anno EntityType string.
    #[must_use]
    pub fn to_entity_type_str(&self) -> &'static str {
        match self {
            Self::Person => "PER",
            Self::Organization => "ORG",
            Self::Location => "LOC",
            Self::GeopoliticalEntity => "GPE",
            Self::Event => "EVENT",
            Self::Product => "PRODUCT",
            Self::WorkOfArt => "WORK_OF_ART",
            Self::DateTime => "DATE",
            Self::Miscellaneous => "MISC",
        }
    }
}

// =============================================================================
// Type Mapping
// =============================================================================

/// Maps Wikidata types (Q-IDs) to NER types.
#[derive(Debug, Clone, Default)]
pub struct WikidataTypeMapper {
    /// Q-ID to NER type mapping
    mappings: HashMap<String, WikidataNERType>,
}

impl WikidataTypeMapper {
    /// Create a new mapper with default mappings.
    #[must_use]
    pub fn new() -> Self {
        let mut mappings = HashMap::new();

        // Person types
        mappings.insert("Q5".to_string(), WikidataNERType::Person); // human
        mappings.insert("Q215627".to_string(), WikidataNERType::Person); // person
        mappings.insert("Q95074".to_string(), WikidataNERType::Person); // fictional character

        // Organization types
        mappings.insert("Q43229".to_string(), WikidataNERType::Organization); // organization
        mappings.insert("Q4830453".to_string(), WikidataNERType::Organization); // business
        mappings.insert("Q783794".to_string(), WikidataNERType::Organization); // company
        mappings.insert("Q891723".to_string(), WikidataNERType::Organization); // public company
        mappings.insert("Q3918".to_string(), WikidataNERType::Organization); // university
        mappings.insert("Q7278".to_string(), WikidataNERType::Organization); // political party
        mappings.insert("Q476028".to_string(), WikidataNERType::Organization); // sports club
        mappings.insert("Q327333".to_string(), WikidataNERType::Organization); // government agency

        // Location types
        mappings.insert("Q515".to_string(), WikidataNERType::Location); // city
        mappings.insert("Q532".to_string(), WikidataNERType::Location); // village
        mappings.insert("Q5084".to_string(), WikidataNERType::Location); // hamlet
        mappings.insert("Q1549591".to_string(), WikidataNERType::Location); // big city
        mappings.insert("Q486972".to_string(), WikidataNERType::Location); // human settlement
        mappings.insert("Q82794".to_string(), WikidataNERType::Location); // geographic region
        mappings.insert("Q46831".to_string(), WikidataNERType::Location); // mountain range
        mappings.insert("Q8502".to_string(), WikidataNERType::Location); // mountain
        mappings.insert("Q4022".to_string(), WikidataNERType::Location); // river
        mappings.insert("Q23397".to_string(), WikidataNERType::Location); // lake

        // Geopolitical entity types
        mappings.insert("Q6256".to_string(), WikidataNERType::GeopoliticalEntity); // country
        mappings.insert("Q3624078".to_string(), WikidataNERType::GeopoliticalEntity); // sovereign state
        mappings.insert("Q7275".to_string(), WikidataNERType::GeopoliticalEntity); // state
        mappings.insert("Q35657".to_string(), WikidataNERType::GeopoliticalEntity); // administrative territorial entity

        // Event types
        mappings.insert("Q1656682".to_string(), WikidataNERType::Event); // event
        mappings.insert("Q18669875".to_string(), WikidataNERType::Event); // recurring event
        mappings.insert("Q198".to_string(), WikidataNERType::Event); // war
        mappings.insert("Q11483816".to_string(), WikidataNERType::Event); // natural disaster

        // Product types
        mappings.insert("Q2424752".to_string(), WikidataNERType::Product); // product
        mappings.insert("Q35127".to_string(), WikidataNERType::Product); // website
        mappings.insert("Q7889".to_string(), WikidataNERType::Product); // video game
        mappings.insert("Q22811662".to_string(), WikidataNERType::Product); // mobile app

        // Work of art types
        mappings.insert("Q838948".to_string(), WikidataNERType::WorkOfArt); // work of art
        mappings.insert("Q571".to_string(), WikidataNERType::WorkOfArt); // book
        mappings.insert("Q11424".to_string(), WikidataNERType::WorkOfArt); // film
        mappings.insert("Q7725634".to_string(), WikidataNERType::WorkOfArt); // literary work
        mappings.insert("Q105543609".to_string(), WikidataNERType::WorkOfArt); // musical work
        mappings.insert("Q134556".to_string(), WikidataNERType::WorkOfArt); // single (music)
        mappings.insert("Q482994".to_string(), WikidataNERType::WorkOfArt); // album

        Self { mappings }
    }

    /// Map a Wikidata type Q-ID to NER type.
    #[must_use]
    pub fn map_type(&self, qid: &str) -> Option<WikidataNERType> {
        self.mappings.get(qid).copied()
    }

    /// Map multiple types, returning the most specific match.
    #[must_use]
    pub fn map_types(&self, qids: &[String]) -> Option<WikidataNERType> {
        // Priority: Person > GPE > Org > Loc > Event > Work > Product > Misc
        let priority = [
            WikidataNERType::Person,
            WikidataNERType::GeopoliticalEntity,
            WikidataNERType::Organization,
            WikidataNERType::Location,
            WikidataNERType::Event,
            WikidataNERType::WorkOfArt,
            WikidataNERType::Product,
        ];

        for ptype in &priority {
            for qid in qids {
                if let Some(mapped) = self.map_type(qid) {
                    if &mapped == ptype {
                        return Some(mapped);
                    }
                }
            }
        }

        // Check if any match
        for qid in qids {
            if let Some(mapped) = self.map_type(qid) {
                return Some(mapped);
            }
        }

        None
    }

    /// Add a custom mapping.
    pub fn add_mapping(&mut self, qid: &str, ner_type: WikidataNERType) {
        self.mappings.insert(qid.to_string(), ner_type);
    }
}

// =============================================================================
// Search Result
// =============================================================================

/// A search result from Wikidata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WikidataSearchResult {
    /// Q-identifier
    pub qid: String,
    /// Primary label
    pub label: String,
    /// Short description
    pub description: Option<String>,
    /// Match score
    pub score: f64,
    /// Whether this is an exact match
    pub exact_match: bool,
}

// =============================================================================
// Linker (Offline/Dictionary-based)
// =============================================================================

/// Offline Wikidata linker using a pre-built dictionary.
///
/// For production use without API calls.
#[derive(Debug, Clone, Default)]
pub struct WikidataDictionary {
    /// Entity lookup by label (lowercase)
    by_label: HashMap<String, Vec<WikidataEntity>>,
    /// Entity lookup by Q-ID
    by_qid: HashMap<String, WikidataEntity>,
    /// Type mapper
    type_mapper: WikidataTypeMapper,
}

impl WikidataDictionary {
    /// Create a new empty dictionary.
    #[must_use]
    pub fn new() -> Self {
        Self {
            by_label: HashMap::new(),
            by_qid: HashMap::new(),
            type_mapper: WikidataTypeMapper::new(),
        }
    }

    /// Add an entity to the dictionary.
    pub fn add_entity(&mut self, mut entity: WikidataEntity) {
        // Map type
        entity.entity_type = self.type_mapper.map_types(&entity.instance_of);

        // Index by label
        let label_key = entity.label.to_lowercase();
        self.by_label
            .entry(label_key)
            .or_default()
            .push(entity.clone());

        // Index by aliases
        for alias in &entity.aliases {
            let alias_key = alias.to_lowercase();
            self.by_label
                .entry(alias_key)
                .or_default()
                .push(entity.clone());
        }

        // Index by Q-ID
        self.by_qid.insert(entity.qid.clone(), entity);
    }

    /// Look up entities by mention text.
    #[must_use]
    pub fn lookup(&self, mention: &str) -> Vec<&WikidataEntity> {
        let key = mention.to_lowercase();
        self.by_label
            .get(&key)
            .map_or(Vec::new(), |v| v.iter().collect())
    }

    /// Get entity by Q-ID.
    #[must_use]
    pub fn get(&self, qid: &str) -> Option<&WikidataEntity> {
        self.by_qid.get(qid)
    }

    /// Number of entities in dictionary.
    #[must_use]
    pub fn len(&self) -> usize {
        self.by_qid.len()
    }

    /// Check if dictionary is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.by_qid.is_empty()
    }

    /// Link a mention to the best matching entity.
    #[must_use]
    pub fn link(
        &self,
        mention: &str,
        expected_type: Option<WikidataNERType>,
    ) -> Option<&WikidataEntity> {
        let candidates = self.lookup(mention);

        if candidates.is_empty() {
            return None;
        }

        // If type expected, filter by type
        if let Some(etype) = expected_type {
            let filtered: Vec<_> = candidates
                .iter()
                .filter(|e| e.entity_type == Some(etype))
                .copied()
                .collect();

            if !filtered.is_empty() {
                // Return most popular from filtered
                return filtered.into_iter().max_by_key(|e| e.sitelinks);
            }
        }

        // Return most popular from all candidates
        candidates.into_iter().max_by_key(|e| e.sitelinks)
    }

    /// Create dictionary with some well-known entities.
    #[must_use]
    pub fn with_common_entities() -> Self {
        let mut dict = Self::new();

        // Add some well-known entities for demonstration
        let entities = vec![
            WikidataEntity {
                qid: "Q937".to_string(),
                label: "Albert Einstein".to_string(),
                description: Some("German-born theoretical physicist".to_string()),
                aliases: vec!["Einstein".to_string(), "A. Einstein".to_string()],
                instance_of: vec!["Q5".to_string()],
                subclass_of: Vec::new(),
                sitelinks: 500,
                entity_type: Some(WikidataNERType::Person),
                wikipedia_url: Some("https://en.wikipedia.org/wiki/Albert_Einstein".to_string()),
                image_url: None,
            },
            WikidataEntity {
                qid: "Q312".to_string(),
                label: "Apple Inc.".to_string(),
                description: Some("American multinational technology company".to_string()),
                aliases: vec!["Apple".to_string(), "Apple Computer".to_string()],
                instance_of: vec!["Q4830453".to_string()],
                subclass_of: Vec::new(),
                sitelinks: 400,
                entity_type: Some(WikidataNERType::Organization),
                wikipedia_url: Some("https://en.wikipedia.org/wiki/Apple_Inc.".to_string()),
                image_url: None,
            },
            WikidataEntity {
                qid: "Q60".to_string(),
                label: "New York City".to_string(),
                description: Some("Most populous city in the United States".to_string()),
                aliases: vec![
                    "NYC".to_string(),
                    "New York".to_string(),
                    "The Big Apple".to_string(),
                ],
                instance_of: vec!["Q515".to_string()],
                subclass_of: Vec::new(),
                sitelinks: 450,
                entity_type: Some(WikidataNERType::Location),
                wikipedia_url: Some("https://en.wikipedia.org/wiki/New_York_City".to_string()),
                image_url: None,
            },
            WikidataEntity {
                qid: "Q30".to_string(),
                label: "United States of America".to_string(),
                description: Some("Country primarily located in North America".to_string()),
                aliases: vec![
                    "USA".to_string(),
                    "United States".to_string(),
                    "US".to_string(),
                    "America".to_string(),
                ],
                instance_of: vec!["Q6256".to_string()],
                subclass_of: Vec::new(),
                sitelinks: 550,
                entity_type: Some(WikidataNERType::GeopoliticalEntity),
                wikipedia_url: Some("https://en.wikipedia.org/wiki/United_States".to_string()),
                image_url: None,
            },
        ];

        for entity in entities {
            dict.add_entity(entity);
        }

        dict
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_mapping() {
        let mapper = WikidataTypeMapper::new();

        assert_eq!(mapper.map_type("Q5"), Some(WikidataNERType::Person));
        assert_eq!(
            mapper.map_type("Q43229"),
            Some(WikidataNERType::Organization)
        );
        assert_eq!(mapper.map_type("Q515"), Some(WikidataNERType::Location));
        assert_eq!(
            mapper.map_type("Q6256"),
            Some(WikidataNERType::GeopoliticalEntity)
        );
        assert_eq!(mapper.map_type("Q99999999"), None);
    }

    #[test]
    fn test_entity_matches_mention() {
        let entity = WikidataEntity {
            qid: "Q937".to_string(),
            label: "Albert Einstein".to_string(),
            description: None,
            aliases: vec!["Einstein".to_string()],
            instance_of: Vec::new(),
            subclass_of: Vec::new(),
            sitelinks: 0,
            entity_type: None,
            wikipedia_url: None,
            image_url: None,
        };

        assert!(entity.matches_mention("Albert Einstein"));
        assert!(entity.matches_mention("einstein")); // Case insensitive
        assert!(entity.matches_mention("Einstein"));
        assert!(!entity.matches_mention("Albert"));
    }

    #[test]
    fn test_dictionary_lookup() {
        let dict = WikidataDictionary::with_common_entities();

        // Exact match
        let results = dict.lookup("Albert Einstein");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].qid, "Q937");

        // Alias match
        let results = dict.lookup("Einstein");
        assert_eq!(results.len(), 1);

        // Case insensitive
        let results = dict.lookup("EINSTEIN");
        assert_eq!(results.len(), 1);

        // No match
        let results = dict.lookup("Nonexistent Entity");
        assert!(results.is_empty());
    }

    #[test]
    fn test_dictionary_link_with_type() {
        let dict = WikidataDictionary::with_common_entities();

        // "Apple" could match org or fruit - with type filter it should return org
        let linked = dict.link("Apple", Some(WikidataNERType::Organization));
        assert!(linked.is_some());
        assert_eq!(linked.unwrap().qid, "Q312");
    }

    #[test]
    fn test_qid_lookup() {
        let dict = WikidataDictionary::with_common_entities();

        let entity = dict.get("Q60");
        assert!(entity.is_some());
        assert_eq!(entity.unwrap().label, "New York City");
    }

    #[test]
    fn test_entity_iri() {
        let entity = WikidataEntity::new("Q937", "Albert Einstein");
        assert_eq!(entity.iri(), "http://www.wikidata.org/entity/Q937");
    }

    #[test]
    fn test_ner_type_to_str() {
        assert_eq!(WikidataNERType::Person.to_entity_type_str(), "PER");
        assert_eq!(WikidataNERType::Organization.to_entity_type_str(), "ORG");
        assert_eq!(WikidataNERType::Location.to_entity_type_str(), "LOC");
        assert_eq!(
            WikidataNERType::GeopoliticalEntity.to_entity_type_str(),
            "GPE"
        );
    }
}
