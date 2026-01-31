//! Cross-schema entity type normalization.
//!
//! Normalizes labels across NER schemas: CoNLL `PER` = OntoNotes `PERSON` = spaCy `PERSON`.
//! Strips BIO prefixes automatically (`B-PER` → `PER`).
//!
//! # Example
//!
//! ```rust
//! use anno_core::core::ontology::{normalize, is_known, CoreType};
//!
//! assert_eq!(normalize("B-PER"), Some(CoreType::Person));
//! assert_eq!(normalize("PERSON"), Some(CoreType::Person));
//! assert_eq!(normalize("personne"), Some(CoreType::Person)); // French
//! assert!(is_known("ORG"));
//! ```
//!
//! For domain-specific types, use [`TypeMapper`](super::TypeMapper) instead of extending
//! the ontology.
//! - **OWL/RDF complexity**: Wrong abstraction level for runtime performance
//!
//! The goal is **practical interoperability**, not ontological completeness.

use std::collections::HashMap;
use std::sync::RwLock;

use serde::{Deserialize, Serialize};

// =============================================================================
// Core Label Normalization
// =============================================================================

/// Canonical entity type label.
///
/// This is the normalized form that all aliases map to.
/// Keep this list small (8-15 core types) per research recommendations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum CoreType {
    // === Named Entities (ML-required) ===
    /// Person names
    Person,
    /// Organizations, companies, agencies
    Organization,
    /// Locations, places, geopolitical entities
    Location,
    /// Miscellaneous named entities
    Misc,

    // === Temporal (pattern-detectable) ===
    /// Date expressions
    Date,
    /// Time expressions
    Time,

    // === Numeric (pattern-detectable) ===
    /// Monetary values
    Money,
    /// Percentages
    Percent,
    /// Quantities with units
    Quantity,
    /// Cardinal numbers
    Cardinal,
    /// Ordinal numbers (first, second)
    Ordinal,

    // === Contact (pattern-detectable) ===
    /// Email addresses
    Email,
    /// URLs/URIs
    Url,
    /// Phone numbers
    Phone,

    // === Extended (OntoNotes-style) ===
    /// Nationalities, religious/political groups
    Norp,
    /// Facilities (buildings, airports)
    Facility,
    /// Products
    Product,
    /// Events
    Event,
    /// Works of art
    WorkOfArt,
    /// Laws
    Law,
    /// Languages
    Language,

    // === Domain Extensions ===
    /// Domain-specific type (registered at runtime)
    Domain(&'static str),
}

impl CoreType {
    /// Get canonical label string.
    pub fn as_label(&self) -> &'static str {
        match self {
            CoreType::Person => "PER",
            CoreType::Organization => "ORG",
            CoreType::Location => "LOC",
            CoreType::Misc => "MISC",
            CoreType::Date => "DATE",
            CoreType::Time => "TIME",
            CoreType::Money => "MONEY",
            CoreType::Percent => "PERCENT",
            CoreType::Quantity => "QUANTITY",
            CoreType::Cardinal => "CARDINAL",
            CoreType::Ordinal => "ORDINAL",
            CoreType::Email => "EMAIL",
            CoreType::Url => "URL",
            CoreType::Phone => "PHONE",
            CoreType::Norp => "NORP",
            CoreType::Facility => "FAC",
            CoreType::Product => "PRODUCT",
            CoreType::Event => "EVENT",
            CoreType::WorkOfArt => "WORK_OF_ART",
            CoreType::Law => "LAW",
            CoreType::Language => "LANGUAGE",
            CoreType::Domain(s) => s,
        }
    }

    /// Is this type pattern-detectable (vs ML-required)?
    pub fn is_pattern_detectable(&self) -> bool {
        matches!(
            self,
            CoreType::Date
                | CoreType::Time
                | CoreType::Money
                | CoreType::Percent
                | CoreType::Quantity
                | CoreType::Cardinal
                | CoreType::Ordinal
                | CoreType::Email
                | CoreType::Url
                | CoreType::Phone
        )
    }
}

impl std::fmt::Display for CoreType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_label())
    }
}

// =============================================================================
// Label Normalizer
// =============================================================================

/// Normalizes entity type labels across schemas and languages.
///
/// This is a simple alias table, not a complex ontology graph.
/// Add aliases as needed; the lookup is O(1) via HashMap.
///
/// # Example
///
/// ```rust
/// use anno_core::core::ontology::{LabelNormalizer, CoreType};
///
/// let norm = LabelNormalizer::default();
///
/// // Different schemas resolve to same type
/// assert_eq!(norm.normalize("PER"), Some(CoreType::Person));
/// assert_eq!(norm.normalize("PERSON"), Some(CoreType::Person));
/// assert_eq!(norm.normalize("B-PER"), Some(CoreType::Person)); // BIO prefix stripped
///
/// // Cross-lingual
/// assert_eq!(norm.normalize("personne"), Some(CoreType::Person)); // French
/// ```
pub struct LabelNormalizer {
    aliases: RwLock<HashMap<String, CoreType>>,
}

impl Default for LabelNormalizer {
    fn default() -> Self {
        let norm = Self {
            aliases: RwLock::new(HashMap::new()),
        };
        norm.register_core_aliases();
        norm
    }
}

impl LabelNormalizer {
    /// Create empty normalizer (no aliases registered).
    pub fn new() -> Self {
        Self {
            aliases: RwLock::new(HashMap::new()),
        }
    }

    /// Register an alias for a core type.
    pub fn register(&self, alias: &str, core_type: CoreType) {
        let mut aliases = self.aliases.write().expect("LabelNormalizer lock poisoned");
        aliases.insert(alias.to_lowercase(), core_type);
    }

    /// Register multiple aliases for a core type.
    pub fn register_many(&self, aliases: &[&str], core_type: CoreType) {
        for alias in aliases {
            self.register(alias, core_type);
        }
    }

    /// Normalize a label to its core type.
    ///
    /// Handles BIO/BIOES prefixes automatically.
    pub fn normalize(&self, label: &str) -> Option<CoreType> {
        // Strip BIO/BIOES prefix
        let label = label
            .strip_prefix("B-")
            .or_else(|| label.strip_prefix("I-"))
            .or_else(|| label.strip_prefix("E-"))
            .or_else(|| label.strip_prefix("S-"))
            .or_else(|| label.strip_prefix("L-"))
            .or_else(|| label.strip_prefix("U-"))
            .unwrap_or(label);

        let aliases = self.aliases.read().expect("LabelNormalizer lock poisoned");
        aliases.get(&label.to_lowercase()).copied()
    }

    /// Check if a label is known.
    pub fn is_known(&self, label: &str) -> bool {
        self.normalize(label).is_some()
    }

    /// Register all core type aliases.
    fn register_core_aliases(&self) {
        // === Person ===
        self.register_many(
            &[
                "per",
                "person",
                "personne", // French
                "persona",  // Spanish/Italian
                "person",   // German (same)
                "pessoa",   // Portuguese
                "человек",  // Russian
                "人",       // Chinese
                "人物",     // Japanese
            ],
            CoreType::Person,
        );

        // === Organization ===
        self.register_many(
            &[
                "org",
                "organization",
                "organisation",   // British/French
                "organización",   // Spanish
                "organizzazione", // Italian
                "organização",    // Portuguese
                "組織",           // Japanese
            ],
            CoreType::Organization,
        );

        // === Location ===
        self.register_many(
            &[
                "loc", "location", "gpe", // Geopolitical entity (OntoNotes)
                "place", "lieu",  // French
                "lugar", // Spanish
                "ort",   // German
                "地点",  // Chinese/Japanese
            ],
            CoreType::Location,
        );

        // === Misc ===
        self.register_many(&["misc", "miscellaneous", "other", "o"], CoreType::Misc);

        // === Temporal ===
        self.register_many(&["date", "datum", "fecha", "日期"], CoreType::Date);
        self.register_many(&["time", "zeit", "hora", "時間"], CoreType::Time);

        // === Numeric ===
        self.register_many(
            &["money", "currency", "argent", "geld", "dinero"],
            CoreType::Money,
        );
        self.register_many(&["percent", "percentage"], CoreType::Percent);
        self.register_many(&["quantity", "qty"], CoreType::Quantity);
        self.register_many(&["cardinal", "number"], CoreType::Cardinal);
        self.register_many(&["ordinal"], CoreType::Ordinal);

        // === Contact ===
        self.register_many(&["email", "e-mail", "correo"], CoreType::Email);
        self.register_many(&["url", "uri", "link", "enlace"], CoreType::Url);
        self.register_many(&["phone", "telephone", "tel", "telefon"], CoreType::Phone);

        // === OntoNotes Extended ===
        self.register_many(&["norp", "nationality"], CoreType::Norp);
        self.register_many(&["fac", "facility", "building"], CoreType::Facility);
        self.register_many(&["product", "produkt", "producto"], CoreType::Product);
        self.register_many(&["event", "ereignis", "evento"], CoreType::Event);
        self.register_many(
            &["work_of_art", "creative-work", "artwork"],
            CoreType::WorkOfArt,
        );
        self.register_many(&["law", "legal", "ley", "gesetz"], CoreType::Law);
        self.register_many(
            &["language", "sprache", "idioma", "langue"],
            CoreType::Language,
        );
    }

    /// Register biomedical domain types.
    ///
    /// These are kept separate because they're domain-specific
    /// and shouldn't pollute the core type namespace.
    pub fn register_biomedical(&self) {
        // Map biomedical types to domain extensions
        // In practice, you'd use TypeMapper for domain-specific handling
        self.register("gene", CoreType::Domain("GENE"));
        self.register("dna", CoreType::Domain("GENE"));
        self.register("protein", CoreType::Domain("PROTEIN"));
        self.register("disease", CoreType::Domain("DISEASE"));
        self.register("chemical", CoreType::Domain("CHEMICAL"));
        self.register("drug", CoreType::Domain("DRUG"));
        self.register("cell_line", CoreType::Domain("CELL_LINE"));
        self.register("cell_type", CoreType::Domain("CELL_TYPE"));
        self.register("species", CoreType::Domain("SPECIES"));
        self.register("anatomy", CoreType::Domain("ANATOMY"));
    }

    /// Register legal domain types.
    pub fn register_legal(&self) {
        self.register("case_ref", CoreType::Domain("CASE_REF"));
        self.register("citation", CoreType::Domain("CITATION"));
        self.register("court", CoreType::Domain("COURT"));
        self.register("statute", CoreType::Domain("STATUTE"));
        self.register("judge", CoreType::Domain("JUDGE"));
    }

    /// Get all known aliases (for debugging/documentation).
    pub fn all_aliases(&self) -> Vec<(String, CoreType)> {
        let aliases = self.aliases.read().expect("LabelNormalizer lock poisoned");
        aliases.iter().map(|(k, v)| (k.clone(), *v)).collect()
    }
}

// =============================================================================
// Global Instance
// =============================================================================

use once_cell::sync::Lazy;

/// Global label normalizer with all core aliases pre-registered.
pub static NORMALIZER: Lazy<LabelNormalizer> = Lazy::new(LabelNormalizer::default);

/// Convenience function to normalize a label using the global normalizer.
pub fn normalize(label: &str) -> Option<CoreType> {
    NORMALIZER.normalize(label)
}

/// Check if a label is known.
pub fn is_known(label: &str) -> bool {
    NORMALIZER.is_known(label)
}

// =============================================================================
// Optional: External Ontology Links (for KB linking, not NER)
// =============================================================================

/// External identifier for entity linking (not for NER type classification).
///
/// These are useful for linking detected entities to knowledge bases,
/// but should NOT be used for type hierarchies in NER.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExternalId {
    /// Wikidata Q-item (e.g., Q5 for "human")
    Wikidata(String),
    /// DBpedia resource URI
    DBpedia(String),
    /// UMLS concept (medical)
    Umls(String),
    /// Custom external identifier
    Custom {
        /// The namespace/source of the identifier (e.g., "freebase", "geonames")
        source: String,
        /// The actual identifier value
        id: String,
    },
}

impl ExternalId {
    /// Create a Wikidata reference.
    pub fn wikidata(qid: &str) -> Self {
        ExternalId::Wikidata(qid.to_string())
    }

    /// Create a DBpedia reference.
    pub fn dbpedia(resource: &str) -> Self {
        ExternalId::DBpedia(resource.to_string())
    }

    /// Get the full IRI/URI.
    pub fn to_iri(&self) -> String {
        match self {
            ExternalId::Wikidata(q) => format!("http://www.wikidata.org/entity/{}", q),
            ExternalId::DBpedia(r) => format!("http://dbpedia.org/resource/{}", r),
            ExternalId::Umls(c) => format!("https://uts.nlm.nih.gov/uts/umls/concept/{}", c),
            ExternalId::Custom { source, id } => format!("{}:{}", source, id),
        }
    }
}

/// Well-known external IDs for core types.
///
/// These are provided for entity linking tasks, not for NER classification.
pub mod external_ids {
    use super::ExternalId;

    /// Wikidata ID for person (Q5 - human).
    pub fn person() -> ExternalId {
        ExternalId::wikidata("Q5")
    }

    /// Wikidata ID for organization (Q43229).
    pub fn organization() -> ExternalId {
        ExternalId::wikidata("Q43229")
    }

    /// Wikidata ID for location (Q618123 - geographical feature).
    pub fn location() -> ExternalId {
        ExternalId::wikidata("Q618123")
    }

    /// Wikidata ID for date (Q205892 - calendar date).
    pub fn date() -> ExternalId {
        ExternalId::wikidata("Q205892")
    }

    /// Wikidata ID for money (Q1368 - currency).
    pub fn money() -> ExternalId {
        ExternalId::wikidata("Q1368")
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_conll_labels() {
        let norm = LabelNormalizer::default();

        assert_eq!(norm.normalize("PER"), Some(CoreType::Person));
        assert_eq!(norm.normalize("ORG"), Some(CoreType::Organization));
        assert_eq!(norm.normalize("LOC"), Some(CoreType::Location));
        assert_eq!(norm.normalize("MISC"), Some(CoreType::Misc));
    }

    #[test]
    fn test_normalize_ontonotes_labels() {
        let norm = LabelNormalizer::default();

        assert_eq!(norm.normalize("PERSON"), Some(CoreType::Person));
        assert_eq!(norm.normalize("GPE"), Some(CoreType::Location));
        assert_eq!(norm.normalize("NORP"), Some(CoreType::Norp));
        assert_eq!(norm.normalize("FAC"), Some(CoreType::Facility));
    }

    #[test]
    fn test_bio_prefix_stripping() {
        let norm = LabelNormalizer::default();

        assert_eq!(norm.normalize("B-PER"), Some(CoreType::Person));
        assert_eq!(norm.normalize("I-PER"), Some(CoreType::Person));
        assert_eq!(norm.normalize("E-ORG"), Some(CoreType::Organization));
        assert_eq!(norm.normalize("S-LOC"), Some(CoreType::Location));
    }

    #[test]
    fn test_cross_lingual() {
        let norm = LabelNormalizer::default();

        // French
        assert_eq!(norm.normalize("personne"), Some(CoreType::Person));
        assert_eq!(norm.normalize("lieu"), Some(CoreType::Location));

        // Spanish
        assert_eq!(norm.normalize("persona"), Some(CoreType::Person));
        assert_eq!(norm.normalize("lugar"), Some(CoreType::Location));

        // German
        assert_eq!(norm.normalize("ort"), Some(CoreType::Location));
    }

    #[test]
    fn test_case_insensitive() {
        let norm = LabelNormalizer::default();

        assert_eq!(norm.normalize("per"), Some(CoreType::Person));
        assert_eq!(norm.normalize("PER"), Some(CoreType::Person));
        assert_eq!(norm.normalize("Per"), Some(CoreType::Person));
        assert_eq!(norm.normalize("PERSON"), Some(CoreType::Person));
        assert_eq!(norm.normalize("person"), Some(CoreType::Person));
    }

    #[test]
    fn test_biomedical_registration() {
        let norm = LabelNormalizer::default();
        norm.register_biomedical();

        assert!(norm.is_known("gene"));
        assert!(norm.is_known("protein"));
        assert!(norm.is_known("disease"));

        // Domain types resolve to Domain variant
        match norm.normalize("gene") {
            Some(CoreType::Domain(s)) => assert_eq!(s, "GENE"),
            _ => panic!("Expected Domain type"),
        }
    }

    #[test]
    fn test_pattern_detectable() {
        assert!(CoreType::Date.is_pattern_detectable());
        assert!(CoreType::Email.is_pattern_detectable());
        assert!(CoreType::Money.is_pattern_detectable());

        assert!(!CoreType::Person.is_pattern_detectable());
        assert!(!CoreType::Organization.is_pattern_detectable());
    }

    #[test]
    fn test_global_normalizer() {
        // Test convenience functions
        assert_eq!(normalize("PER"), Some(CoreType::Person));
        assert!(is_known("ORG"));
        assert!(!is_known("UNKNOWN_TYPE_XYZ"));
    }

    #[test]
    fn test_external_ids() {
        let qid = external_ids::person();
        assert_eq!(qid.to_iri(), "http://www.wikidata.org/entity/Q5");

        let dbp = ExternalId::dbpedia("Person");
        assert_eq!(dbp.to_iri(), "http://dbpedia.org/resource/Person");
    }
}
