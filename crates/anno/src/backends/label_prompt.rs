//! Label prompt normalization for zero-shot NER systems.
//!
//! GLiNER and similar label-text-conditioned models are sensitive to how
//! entity type labels are phrased. "Person", "person", "PERSON", "human",
//! "individual" may all give different results.
//!
//! This module provides tools to:
//! - Normalize label strings to canonical forms
//! - Expand labels to synonyms/aliases for better coverage
//! - Map between different ontologies (e.g., OntoNotes → CoNLL)
//!
//! # Research Background
//!
//! GLiNER critique (2025): "Performance is sensitive to how labels are phrased;
//! semantically similar but poorly written label prompts can cause large drops,
//! especially for fine-grained or rare types."
//!
//! # Usage
//!
//! ```rust
//! use anno::backends::label_prompt::{LabelNormalizer, StandardNormalizer};
//!
//! let normalizer = StandardNormalizer::default();
//!
//! // Canonical form
//! assert_eq!(normalizer.normalize("PERSON"), "person");
//! assert_eq!(normalizer.normalize("ORG"), "organization");
//!
//! // Expansions for better zero-shot coverage
//! let expansions = normalizer.expand("person");
//! // Returns: ["person", "human", "individual", "people", ...]
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Trait for label prompt normalization.
pub trait LabelNormalizer: Send + Sync {
    /// Normalize a label to its canonical form.
    fn normalize(&self, label: &str) -> String;

    /// Expand a label to synonyms/aliases for better coverage.
    fn expand(&self, label: &str) -> Vec<String>;

    /// Check if two labels are equivalent.
    fn equivalent(&self, a: &str, b: &str) -> bool {
        self.normalize(a) == self.normalize(b)
    }

    /// Get the canonical name for a label (for display).
    fn canonical_name(&self, label: &str) -> String {
        self.normalize(label)
    }
}

/// Standard label normalizer with common NER ontology mappings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StandardNormalizer {
    /// Canonical form mappings (alias → canonical)
    canonical_map: HashMap<String, String>,
    /// Expansion mappings (canonical → synonyms)
    expansion_map: HashMap<String, Vec<String>>,
    /// Whether to lowercase during normalization
    pub lowercase: bool,
}

impl Default for StandardNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

impl StandardNormalizer {
    /// Create a new standard normalizer with default mappings.
    #[must_use]
    pub fn new() -> Self {
        let mut canonical_map = HashMap::new();
        let mut expansion_map = HashMap::new();

        // Person variations
        for alias in &[
            "PER",
            "PERSON",
            "Person",
            "per",
            "people",
            "PEOPLE",
            "human",
            "HUMAN",
            "individual",
            "INDIVIDUAL",
            "B-PER",
            "I-PER",
            "S-PER",
            "E-PER",
        ] {
            canonical_map.insert(alias.to_lowercase(), "person".to_string());
        }
        expansion_map.insert(
            "person".to_string(),
            vec![
                "person".to_string(),
                "human".to_string(),
                "individual".to_string(),
                "human being".to_string(),
                "people".to_string(),
            ],
        );

        // Organization variations
        for alias in &[
            "ORG",
            "ORGANIZATION",
            "Organization",
            "org",
            "organisation",
            "ORGANISATION",
            "company",
            "COMPANY",
            "institution",
            "INSTITUTION",
            "B-ORG",
            "I-ORG",
            "S-ORG",
            "E-ORG",
            "CORP",
            "corporation",
        ] {
            canonical_map.insert(alias.to_lowercase(), "organization".to_string());
        }
        expansion_map.insert(
            "organization".to_string(),
            vec![
                "organization".to_string(),
                "organisation".to_string(),
                "company".to_string(),
                "institution".to_string(),
                "corporation".to_string(),
                "agency".to_string(),
                "group".to_string(),
            ],
        );

        // Location variations
        for alias in &[
            "LOC",
            "LOCATION",
            "Location",
            "loc",
            "GPE",
            "gpe",
            "place",
            "PLACE",
            "GEO",
            "geo",
            "geographic location",
            "B-LOC",
            "I-LOC",
            "S-LOC",
            "E-LOC",
            "B-GPE",
            "I-GPE",
            "FAC",
            "facility",
        ] {
            canonical_map.insert(alias.to_lowercase(), "location".to_string());
        }
        expansion_map.insert(
            "location".to_string(),
            vec![
                "location".to_string(),
                "place".to_string(),
                "geographic location".to_string(),
                "geopolitical entity".to_string(),
                "country".to_string(),
                "city".to_string(),
                "region".to_string(),
            ],
        );

        // Miscellaneous variations
        for alias in &[
            "MISC",
            "Misc",
            "misc",
            "miscellaneous",
            "MISCELLANEOUS",
            "OTHER",
            "other",
            "B-MISC",
            "I-MISC",
            "S-MISC",
            "E-MISC",
        ] {
            canonical_map.insert(alias.to_lowercase(), "miscellaneous".to_string());
        }
        expansion_map.insert(
            "miscellaneous".to_string(),
            vec![
                "miscellaneous".to_string(),
                "other entity".to_string(),
                "named entity".to_string(),
            ],
        );

        // Date/Time variations
        for alias in &[
            "DATE", "Date", "date", "TIME", "Time", "time", "DATETIME", "datetime", "temporal",
            "TEMPORAL", "B-DATE", "I-DATE", "B-TIME", "I-TIME",
        ] {
            canonical_map.insert(alias.to_lowercase(), "date".to_string());
        }
        expansion_map.insert(
            "date".to_string(),
            vec![
                "date".to_string(),
                "time".to_string(),
                "temporal expression".to_string(),
                "datetime".to_string(),
            ],
        );

        // Money variations
        for alias in &[
            "MONEY", "Money", "money", "CURRENCY", "currency", "monetary", "B-MONEY", "I-MONEY",
        ] {
            canonical_map.insert(alias.to_lowercase(), "money".to_string());
        }
        expansion_map.insert(
            "money".to_string(),
            vec![
                "money".to_string(),
                "monetary value".to_string(),
                "currency amount".to_string(),
                "price".to_string(),
            ],
        );

        // Event variations
        for alias in &[
            "EVENT",
            "Event",
            "event",
            "HAPPENING",
            "occurrence",
            "B-EVENT",
            "I-EVENT",
        ] {
            canonical_map.insert(alias.to_lowercase(), "event".to_string());
        }
        expansion_map.insert(
            "event".to_string(),
            vec![
                "event".to_string(),
                "occurrence".to_string(),
                "happening".to_string(),
                "incident".to_string(),
            ],
        );

        // Product variations
        for alias in &[
            "PRODUCT",
            "Product",
            "product",
            "PROD",
            "B-PRODUCT",
            "I-PRODUCT",
        ] {
            canonical_map.insert(alias.to_lowercase(), "product".to_string());
        }
        expansion_map.insert(
            "product".to_string(),
            vec![
                "product".to_string(),
                "commercial product".to_string(),
                "item".to_string(),
                "goods".to_string(),
            ],
        );

        // Work of art variations
        for alias in &[
            "WORK_OF_ART",
            "WorkOfArt",
            "work_of_art",
            "WORK",
            "artwork",
            "B-WORK_OF_ART",
            "I-WORK_OF_ART",
            "creative work",
        ] {
            canonical_map.insert(alias.to_lowercase(), "work_of_art".to_string());
        }
        expansion_map.insert(
            "work_of_art".to_string(),
            vec![
                "work of art".to_string(),
                "creative work".to_string(),
                "artwork".to_string(),
                "artistic creation".to_string(),
            ],
        );

        Self {
            canonical_map,
            expansion_map,
            lowercase: true,
        }
    }

    /// Add a custom mapping.
    pub fn add_mapping(&mut self, alias: &str, canonical: &str) {
        self.canonical_map
            .insert(alias.to_lowercase(), canonical.to_string());
    }

    /// Add custom expansions for a canonical label.
    pub fn add_expansions(&mut self, canonical: &str, expansions: Vec<String>) {
        self.expansion_map.insert(canonical.to_string(), expansions);
    }
}

impl LabelNormalizer for StandardNormalizer {
    fn normalize(&self, label: &str) -> String {
        let key = if self.lowercase {
            label.to_lowercase()
        } else {
            label.to_string()
        };

        // Strip BIO prefix if present
        let stripped = key
            .strip_prefix("b-")
            .or_else(|| key.strip_prefix("i-"))
            .or_else(|| key.strip_prefix("s-"))
            .or_else(|| key.strip_prefix("e-"))
            .unwrap_or(&key);

        self.canonical_map
            .get(stripped)
            .cloned()
            .unwrap_or_else(|| stripped.to_string())
    }

    fn expand(&self, label: &str) -> Vec<String> {
        let canonical = self.normalize(label);
        self.expansion_map
            .get(&canonical)
            .cloned()
            .unwrap_or_else(|| vec![canonical])
    }
}

/// Hierarchical entity type system.
///
/// Supports type hierarchies like: Person → Athlete → Tennis Player
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalTypeSystem {
    /// Parent → children mapping
    children: HashMap<String, Vec<String>>,
    /// Child → parent mapping
    parent: HashMap<String, String>,
    /// All types in the system
    all_types: Vec<String>,
}

impl Default for HierarchicalTypeSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl HierarchicalTypeSystem {
    /// Create a new empty type system.
    #[must_use]
    pub fn new() -> Self {
        Self {
            children: HashMap::new(),
            parent: HashMap::new(),
            all_types: Vec::new(),
        }
    }

    /// Create a type system with standard NER hierarchy.
    #[must_use]
    pub fn standard_ner() -> Self {
        let mut sys = Self::new();

        // Person hierarchy
        sys.add_type("person", None);
        sys.add_type("politician", Some("person"));
        sys.add_type("athlete", Some("person"));
        sys.add_type("artist", Some("person"));
        sys.add_type("scientist", Some("person"));
        sys.add_type("businessperson", Some("person"));

        // Organization hierarchy
        sys.add_type("organization", None);
        sys.add_type("company", Some("organization"));
        sys.add_type("government", Some("organization"));
        sys.add_type("educational", Some("organization"));
        sys.add_type("sports_team", Some("organization"));
        sys.add_type("political_party", Some("organization"));

        // Location hierarchy
        sys.add_type("location", None);
        sys.add_type("country", Some("location"));
        sys.add_type("city", Some("location"));
        sys.add_type("state", Some("location"));
        sys.add_type("facility", Some("location"));
        sys.add_type("natural_feature", Some("location"));

        sys
    }

    /// Add a type to the hierarchy.
    pub fn add_type(&mut self, type_name: &str, parent_type: Option<&str>) {
        let type_lower = type_name.to_lowercase();

        if !self.all_types.contains(&type_lower) {
            self.all_types.push(type_lower.clone());
        }

        if let Some(parent) = parent_type {
            let parent_lower = parent.to_lowercase();
            self.parent.insert(type_lower.clone(), parent_lower.clone());
            self.children
                .entry(parent_lower)
                .or_default()
                .push(type_lower);
        }
    }

    /// Get all ancestors of a type (from specific to general).
    #[must_use]
    pub fn ancestors(&self, type_name: &str) -> Vec<String> {
        let mut result = Vec::new();
        let mut current = type_name.to_lowercase();

        while let Some(parent) = self.parent.get(&current) {
            result.push(parent.clone());
            current = parent.clone();
        }

        result
    }

    /// Get all descendants of a type.
    #[must_use]
    pub fn descendants(&self, type_name: &str) -> Vec<String> {
        let mut result = Vec::new();
        let mut queue = vec![type_name.to_lowercase()];

        while let Some(current) = queue.pop() {
            if let Some(children) = self.children.get(&current) {
                for child in children {
                    result.push(child.clone());
                    queue.push(child.clone());
                }
            }
        }

        result
    }

    /// Check if type_a is a subtype of type_b.
    #[must_use]
    pub fn is_subtype(&self, type_a: &str, type_b: &str) -> bool {
        let a_lower = type_a.to_lowercase();
        let b_lower = type_b.to_lowercase();

        if a_lower == b_lower {
            return true;
        }

        self.ancestors(&a_lower).contains(&b_lower)
    }

    /// Get the most specific common ancestor of two types.
    #[must_use]
    pub fn common_ancestor(&self, type_a: &str, type_b: &str) -> Option<String> {
        let ancestors_a: std::collections::HashSet<_> = std::iter::once(type_a.to_lowercase())
            .chain(self.ancestors(type_a))
            .collect();

        let current = type_b.to_lowercase();
        if ancestors_a.contains(&current) {
            return Some(current);
        }

        self.ancestors(type_b)
            .into_iter()
            .find(|ancestor| ancestors_a.contains(ancestor))
            .map(|s| s.to_string())
    }

    /// Get all root types (types with no parent).
    #[must_use]
    pub fn roots(&self) -> Vec<String> {
        self.all_types
            .iter()
            .filter(|t| !self.parent.contains_key(*t))
            .cloned()
            .collect()
    }
}

/// Ontology mapper for cross-dataset type normalization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OntologyMapper {
    /// Source ontology name
    pub source: String,
    /// Target ontology name
    pub target: String,
    /// Mappings from source types to target types
    mappings: HashMap<String, String>,
}

impl OntologyMapper {
    /// Create a new ontology mapper.
    #[must_use]
    pub fn new(source: &str, target: &str) -> Self {
        Self {
            source: source.to_string(),
            target: target.to_string(),
            mappings: HashMap::new(),
        }
    }

    /// Create a CoNLL-2003 → OntoNotes mapper.
    #[must_use]
    pub fn conll_to_ontonotes() -> Self {
        let mut mapper = Self::new("conll2003", "ontonotes");
        mapper.add("PER", "PERSON");
        mapper.add("ORG", "ORG");
        mapper.add("LOC", "GPE"); // CoNLL LOC ≈ OntoNotes GPE for most cases
        mapper.add("MISC", "MISC"); // No direct equivalent
        mapper
    }

    /// Create an OntoNotes → CoNLL-2003 mapper.
    #[must_use]
    pub fn ontonotes_to_conll() -> Self {
        let mut mapper = Self::new("ontonotes", "conll2003");
        mapper.add("PERSON", "PER");
        mapper.add("ORG", "ORG");
        mapper.add("GPE", "LOC");
        mapper.add("LOC", "LOC");
        mapper.add("FAC", "LOC");
        mapper.add("NORP", "MISC");
        mapper.add("WORK_OF_ART", "MISC");
        mapper.add("EVENT", "MISC");
        mapper.add("PRODUCT", "MISC");
        mapper.add("LAW", "MISC");
        mapper.add("LANGUAGE", "MISC");
        // Numeric types typically not in CoNLL
        mapper.add("DATE", "MISC");
        mapper.add("TIME", "MISC");
        mapper.add("MONEY", "MISC");
        mapper.add("QUANTITY", "MISC");
        mapper.add("PERCENT", "MISC");
        mapper.add("CARDINAL", "MISC");
        mapper.add("ORDINAL", "MISC");
        mapper
    }

    /// Add a mapping.
    pub fn add(&mut self, source_type: &str, target_type: &str) {
        self.mappings
            .insert(source_type.to_string(), target_type.to_string());
    }

    /// Map a type from source to target ontology.
    #[must_use]
    pub fn map(&self, source_type: &str) -> Option<String> {
        self.mappings.get(source_type).cloned()
    }

    /// Map a type, falling back to original if no mapping exists.
    #[must_use]
    pub fn map_or_keep(&self, source_type: &str) -> String {
        self.map(source_type)
            .unwrap_or_else(|| source_type.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_normalizer() {
        let norm = StandardNormalizer::default();

        assert_eq!(norm.normalize("PER"), "person");
        assert_eq!(norm.normalize("PERSON"), "person");
        assert_eq!(norm.normalize("B-PER"), "person");

        assert_eq!(norm.normalize("ORG"), "organization");
        assert_eq!(norm.normalize("organisation"), "organization");

        assert_eq!(norm.normalize("LOC"), "location");
        assert_eq!(norm.normalize("GPE"), "location");
    }

    #[test]
    fn test_expansion() {
        let norm = StandardNormalizer::default();

        let expansions = norm.expand("PER");
        assert!(expansions.contains(&"person".to_string()));
        assert!(expansions.contains(&"human".to_string()));
    }

    #[test]
    fn test_hierarchical_types() {
        let sys = HierarchicalTypeSystem::standard_ner();

        assert!(sys.is_subtype("athlete", "person"));
        assert!(sys.is_subtype("person", "person"));
        assert!(!sys.is_subtype("person", "athlete"));

        let ancestors = sys.ancestors("athlete");
        assert_eq!(ancestors, vec!["person"]);

        let descendants = sys.descendants("person");
        assert!(descendants.contains(&"athlete".to_string()));
    }

    #[test]
    fn test_ontology_mapper() {
        let mapper = OntologyMapper::conll_to_ontonotes();

        assert_eq!(mapper.map("PER"), Some("PERSON".to_string()));
        assert_eq!(mapper.map("LOC"), Some("GPE".to_string()));
    }

    #[test]
    fn test_normalizer_bio_prefix_stripping() {
        let norm = StandardNormalizer::default();

        // All BIO prefixes should be stripped
        assert_eq!(norm.normalize("B-PER"), "person");
        assert_eq!(norm.normalize("I-PER"), "person");
        assert_eq!(norm.normalize("E-PER"), "person");
        assert_eq!(norm.normalize("S-PER"), "person");
    }

    #[test]
    fn test_normalizer_case_insensitive() {
        let norm = StandardNormalizer::default();

        assert_eq!(norm.normalize("per"), "person");
        assert_eq!(norm.normalize("Per"), "person");
        assert_eq!(norm.normalize("PER"), "person");
        assert_eq!(norm.normalize("PERSON"), "person");
    }

    #[test]
    fn test_expansion_all_types() {
        let norm = StandardNormalizer::default();

        // PER expansions
        let per = norm.expand("PER");
        assert!(per.len() >= 2);
        assert!(per.contains(&"person".to_string()));

        // ORG expansions
        let org = norm.expand("ORG");
        assert!(org.contains(&"organization".to_string()));

        // LOC expansions
        let loc = norm.expand("LOC");
        assert!(loc.contains(&"location".to_string()));
    }

    #[test]
    fn test_hierarchical_athletes() {
        let sys = HierarchicalTypeSystem::standard_ner();

        // athlete -> person
        assert!(sys.is_subtype("athlete", "person"));

        // politician -> person
        assert!(sys.is_subtype("politician", "person"));

        // transitivity: shouldn't match unrelated
        assert!(!sys.is_subtype("athlete", "organization"));
    }

    #[test]
    fn test_mapper_bidirectional() {
        let mapper = OntologyMapper::conll_to_ontonotes();

        // Known mappings
        assert_eq!(mapper.map("PER"), Some("PERSON".to_string()));
        assert_eq!(mapper.map("ORG"), Some("ORG".to_string()));
        assert_eq!(mapper.map("LOC"), Some("GPE".to_string()));
        assert_eq!(mapper.map("MISC"), Some("MISC".to_string()));

        // Unknown type returns None
        assert_eq!(mapper.map("UNKNOWN_TYPE"), None);
    }

    #[test]
    fn test_mapper_or_keep() {
        let mapper = OntologyMapper::conll_to_ontonotes();

        // Known type gets mapped
        assert_eq!(mapper.map_or_keep("PER"), "PERSON");

        // Unknown type gets kept as-is
        assert_eq!(mapper.map_or_keep("CUSTOM_TYPE"), "CUSTOM_TYPE");
    }
}
