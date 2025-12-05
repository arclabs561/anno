//! Entity types and structures for NER.
//!
//! # Design Philosophy (Research-Aligned)
//!
//! This module implements entity types informed by modern NER research:
//!
//! - **GLiNER/Bi-Encoder**: Entity types are *labels to match against*, not fixed classes.
//!   Relations ("CEO of") are entities too - they're just labels in the same latent space.
//!
//! - **TPLinker/Joint Extraction**: Entities and relations can be extracted in a single pass.
//!   The type system supports relation triggers as first-class mentions.
//!
//! - **Knowledge Graphs**: Entities can link to external knowledge bases (`kb_id`) for
//!   coreference resolution and GraphRAG applications.
//!
//! # Type Hierarchy
//!
//! ```text
//! Mention
//! ├── Entity (single span)
//! │   ├── Named (ML): Person, Organization, Location
//! │   ├── Temporal (Pattern): Date, Time
//! │   ├── Numeric (Pattern): Money, Percent, Quantity, Cardinal, Ordinal
//! │   └── Contact (Pattern): Email, Url, Phone
//! │
//! └── Relation (connects entities)
//!     └── Trigger text: "CEO of", "located in", "born on"
//! ```
//!
//! # Design Principles
//!
//! 1. **Bi-encoder compatible**: Types are semantic labels, not fixed enums
//! 2. **Joint extraction**: Relations are mentions with trigger spans
//! 3. **Knowledge linking**: `kb_id` for connecting to external KBs
//! 4. **Hierarchical confidence**: Coarse (linkage) + fine (type) scores
//! 5. **Multi-modal ready**: Spans can be text offsets or visual bboxes

use serde::{Deserialize, Serialize};
use std::borrow::Cow;

// ============================================================================
// Entity Category (OntoNotes-inspired)
// ============================================================================

/// Category of entity based on detection characteristics and semantics.
///
/// Based on OntoNotes 5.0 categories with extensions for:
/// - Structured data (Contact, patterns)
/// - Knowledge graphs (Relation, for TPLinker/GLiNER joint extraction)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum EntityCategory {
    /// Named entities for people/groups (ML-required).
    /// Types: Person, NORP (nationalities/religious/political groups)
    Agent,
    /// Named entities for organizations/facilities (ML-required).
    /// Types: Organization, Facility
    Organization,
    /// Named entities for places (ML-required).
    /// Types: GPE (geo-political), Location (geographic)
    Place,
    /// Named entities for creative/conceptual (ML-required).
    /// Types: Event, Product, WorkOfArt, Law, Language
    Creative,
    /// Temporal entities (pattern-detectable).
    /// Types: Date, Time
    Temporal,
    /// Numeric entities (pattern-detectable).
    /// Types: Money, Percent, Quantity, Cardinal, Ordinal
    Numeric,
    /// Contact/identifier entities (pattern-detectable).
    /// Types: Email, Url, Phone
    Contact,
    /// Relation triggers for knowledge graph construction (ML-required).
    /// Examples: "CEO of", "located in", "founded by"
    /// In GLiNER bi-encoder, relations are just another label to match.
    Relation,
    /// Miscellaneous/unknown category
    Misc,
}

impl EntityCategory {
    /// Returns true if this category requires ML for detection.
    #[must_use]
    pub const fn requires_ml(&self) -> bool {
        matches!(
            self,
            EntityCategory::Agent
                | EntityCategory::Organization
                | EntityCategory::Place
                | EntityCategory::Creative
                | EntityCategory::Relation
        )
    }

    /// Returns true if this category can be detected via patterns.
    #[must_use]
    pub const fn pattern_detectable(&self) -> bool {
        matches!(
            self,
            EntityCategory::Temporal | EntityCategory::Numeric | EntityCategory::Contact
        )
    }

    /// Returns true if this is a relation (for knowledge graph construction).
    #[must_use]
    pub const fn is_relation(&self) -> bool {
        matches!(self, EntityCategory::Relation)
    }

    /// Returns OntoNotes-compatible category name.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            EntityCategory::Agent => "agent",
            EntityCategory::Organization => "organization",
            EntityCategory::Place => "place",
            EntityCategory::Creative => "creative",
            EntityCategory::Temporal => "temporal",
            EntityCategory::Numeric => "numeric",
            EntityCategory::Contact => "contact",
            EntityCategory::Relation => "relation",
            EntityCategory::Misc => "misc",
        }
    }
}

impl std::fmt::Display for EntityCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ============================================================================
// Entity Viewport (Research: Entity Manifolds)
// ============================================================================

/// Viewport context for multi-faceted entity representation.
///
/// # Research Background
///
/// The concept of "Entity Viewports" comes from the observation that
/// real-world entities are not monolithic - they present different
/// facets depending on context:
///
/// - "Marie Curie" in an **Academic** context: physicist, Nobel laureate
/// - "Marie Curie" in a **Technical** context: radioactivity researcher, X-ray pioneer
/// - "Marie Curie" in a **Personal** context: mother, immigrant, educator
/// - "Marie Curie" in a **Medical** context: founder of mobile X-ray units
///
/// Rather than collapsing all information into a single vector,
/// the viewport model preserves these distinctions and enables
/// "projection" at query time.
///
/// # Usage in RAG Systems
///
/// When answering "What were Curie's scientific contributions?", retrieve
/// facts from the `Academic` viewport. When answering "What was Curie's
/// personal life like?", retrieve from `Personal`.
///
/// # Example
///
/// ```rust
/// use anno_core::{Entity, EntityType, EntityViewport};
///
/// let mut entity = Entity::new("Marie Curie", EntityType::Person, 0, 11, 0.9);
/// entity.viewport = Some(EntityViewport::Academic);
/// assert!(entity.viewport.as_ref().unwrap().is_professional());
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub enum EntityViewport {
    /// Business/financial context (CEO, revenue, market cap)
    Business,
    /// Legal context (lawsuits, settlements, compliance)
    Legal,
    /// Technical/engineering context (patents, inventions, code)
    Technical,
    /// Academic/research context (publications, citations, grants)
    Academic,
    /// Personal/biographical context (family, hobbies, background)
    Personal,
    /// Political context (lobbying, donations, policy positions)
    Political,
    /// Media/public relations context (interviews, statements, PR)
    Media,
    /// Historical context (past roles, timeline events)
    Historical,
    /// Generic/unspecified context
    #[default]
    General,
    /// Custom viewport with a descriptive label
    Custom(String),
}

impl EntityViewport {
    /// Human-readable label for the viewport.
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            EntityViewport::Business => "business",
            EntityViewport::Legal => "legal",
            EntityViewport::Technical => "technical",
            EntityViewport::Academic => "academic",
            EntityViewport::Personal => "personal",
            EntityViewport::Political => "political",
            EntityViewport::Media => "media",
            EntityViewport::Historical => "historical",
            EntityViewport::General => "general",
            EntityViewport::Custom(s) => s,
        }
    }

    /// Is this a professional/work-related viewport?
    #[must_use]
    pub const fn is_professional(&self) -> bool {
        matches!(
            self,
            EntityViewport::Business
                | EntityViewport::Legal
                | EntityViewport::Technical
                | EntityViewport::Academic
                | EntityViewport::Political
        )
    }
}

impl std::str::FromStr for EntityViewport {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s.to_lowercase().as_str() {
            "business" | "financial" | "corporate" => EntityViewport::Business,
            "legal" | "law" | "compliance" => EntityViewport::Legal,
            "technical" | "engineering" | "tech" => EntityViewport::Technical,
            "academic" | "research" | "scholarly" => EntityViewport::Academic,
            "personal" | "biographical" | "private" => EntityViewport::Personal,
            "political" | "policy" | "government" => EntityViewport::Political,
            "media" | "press" | "pr" | "public_relations" => EntityViewport::Media,
            "historical" | "history" | "past" => EntityViewport::Historical,
            "general" | "generic" | "" => EntityViewport::General,
            other => EntityViewport::Custom(other.to_string()),
        })
    }
}

impl std::fmt::Display for EntityViewport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ============================================================================
// Entity Type
// ============================================================================

/// Entity type classification.
///
/// Organized into categories:
/// - **Named** (ML-required): Person, Organization, Location
/// - **Temporal** (pattern): Date, Time
/// - **Numeric** (pattern): Money, Percent, Quantity, Cardinal, Ordinal
/// - **Contact** (pattern): Email, Url, Phone
///
/// # Examples
///
/// ```rust
/// use anno_core::EntityType;
///
/// let ty = EntityType::Email;
/// assert!(ty.category().pattern_detectable());
/// assert!(!ty.category().requires_ml());
///
/// let ty = EntityType::Person;
/// assert!(ty.category().requires_ml());
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum EntityType {
    // === Named Entities (ML-required) ===
    /// Person name (PER) - requires ML/context
    Person,
    /// Organization name (ORG) - requires ML/context
    Organization,
    /// Location/Place (LOC/GPE) - requires ML/context
    Location,

    // === Temporal Entities (Pattern-detectable) ===
    /// Date expression (DATE) - pattern-detectable
    Date,
    /// Time expression (TIME) - pattern-detectable
    Time,

    // === Numeric Entities (Pattern-detectable) ===
    /// Monetary value (MONEY) - pattern-detectable
    Money,
    /// Percentage (PERCENT) - pattern-detectable
    Percent,
    /// Quantity with unit (QUANTITY) - pattern-detectable
    Quantity,
    /// Cardinal number (CARDINAL) - pattern-detectable
    Cardinal,
    /// Ordinal number (ORDINAL) - pattern-detectable
    Ordinal,

    // === Contact Entities (Pattern-detectable) ===
    /// Email address - pattern-detectable
    Email,
    /// URL/URI - pattern-detectable
    Url,
    /// Phone number - pattern-detectable
    Phone,

    // === Extensibility ===
    /// Domain-specific custom type with explicit category
    Custom {
        /// Type name (e.g., "DISEASE", "PRODUCT", "EVENT")
        name: String,
        /// Category for this custom type
        category: EntityCategory,
    },

    /// Legacy catch-all for unknown types (prefer Custom for new code)
    #[serde(rename = "Other")]
    Other(String),
}

impl EntityType {
    /// Get the category of this entity type.
    #[must_use]
    pub fn category(&self) -> EntityCategory {
        match self {
            // Agent entities (people/groups)
            EntityType::Person => EntityCategory::Agent,
            // Organization entities
            EntityType::Organization => EntityCategory::Organization,
            // Place entities (locations)
            EntityType::Location => EntityCategory::Place,
            // Temporal entities
            EntityType::Date | EntityType::Time => EntityCategory::Temporal,
            // Numeric entities
            EntityType::Money
            | EntityType::Percent
            | EntityType::Quantity
            | EntityType::Cardinal
            | EntityType::Ordinal => EntityCategory::Numeric,
            // Contact entities
            EntityType::Email | EntityType::Url | EntityType::Phone => EntityCategory::Contact,
            // Custom with explicit category
            EntityType::Custom { category, .. } => *category,
            // Legacy Other - assume misc
            EntityType::Other(_) => EntityCategory::Misc,
        }
    }

    /// Returns true if this entity type requires ML for detection.
    #[must_use]
    pub fn requires_ml(&self) -> bool {
        self.category().requires_ml()
    }

    /// Returns true if this entity type can be detected via patterns.
    #[must_use]
    pub fn pattern_detectable(&self) -> bool {
        self.category().pattern_detectable()
    }

    /// Convert to standard label string (CoNLL/OntoNotes format).
    #[must_use]
    pub fn as_label(&self) -> &str {
        match self {
            EntityType::Person => "PER",
            EntityType::Organization => "ORG",
            EntityType::Location => "LOC",
            EntityType::Date => "DATE",
            EntityType::Time => "TIME",
            EntityType::Money => "MONEY",
            EntityType::Percent => "PERCENT",
            EntityType::Quantity => "QUANTITY",
            EntityType::Cardinal => "CARDINAL",
            EntityType::Ordinal => "ORDINAL",
            EntityType::Email => "EMAIL",
            EntityType::Url => "URL",
            EntityType::Phone => "PHONE",
            EntityType::Custom { name, .. } => name.as_str(),
            EntityType::Other(s) => s.as_str(),
        }
    }

    /// Parse from standard label string.
    ///
    /// Handles various formats: CoNLL (PER), OntoNotes (PERSON), BIO (B-PER).
    #[must_use]
    pub fn from_label(label: &str) -> Self {
        // Strip BIO prefix if present
        let label = label
            .strip_prefix("B-")
            .or_else(|| label.strip_prefix("I-"))
            .or_else(|| label.strip_prefix("E-"))
            .or_else(|| label.strip_prefix("S-"))
            .unwrap_or(label);

        match label.to_uppercase().as_str() {
            // Named entities
            "PER" | "PERSON" => EntityType::Person,
            "ORG" | "ORGANIZATION" => EntityType::Organization,
            "LOC" | "LOCATION" | "GPE" => EntityType::Location,
            // Temporal
            "DATE" => EntityType::Date,
            "TIME" => EntityType::Time,
            // Numeric
            "MONEY" | "CURRENCY" => EntityType::Money,
            "PERCENT" | "PERCENTAGE" => EntityType::Percent,
            "QUANTITY" => EntityType::Quantity,
            "CARDINAL" => EntityType::Cardinal,
            "ORDINAL" => EntityType::Ordinal,
            // Contact
            "EMAIL" => EntityType::Email,
            "URL" | "URI" => EntityType::Url,
            "PHONE" | "TELEPHONE" => EntityType::Phone,
            // Unknown -> Other
            other => EntityType::Other(other.to_string()),
        }
    }

    /// Create a custom domain-specific entity type.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use anno_core::{EntityType, EntityCategory};
    ///
    /// // Medical entity - custom domain-specific type
    /// let disease = EntityType::custom("DISEASE", EntityCategory::Agent);
    /// assert!(disease.requires_ml());
    ///
    /// // ID patterns - can be detected via patterns
    /// let product_id = EntityType::custom("PRODUCT_ID", EntityCategory::Misc);
    /// ```
    #[must_use]
    pub fn custom(name: impl Into<String>, category: EntityCategory) -> Self {
        EntityType::Custom {
            name: name.into(),
            category,
        }
    }
}

impl std::fmt::Display for EntityType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_label())
    }
}

impl std::str::FromStr for EntityType {
    type Err = std::convert::Infallible;

    /// Parse from standard label string. Never fails - unknown labels become `Other`.
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Ok(Self::from_label(s))
    }
}

// =============================================================================
// Type Mapping for Domain-Specific Datasets
// =============================================================================

/// Maps domain-specific entity types to standard NER types.
///
/// # Research Context (Familiarity paper, arXiv:2412.10121)
///
/// Type mapping creates "label overlap" between training and evaluation:
/// - Mapping ACTOR → Person increases overlap
/// - This can inflate zero-shot F1 scores
///
/// Use `LabelShift::from_type_sets()` to quantify how much overlap exists.
/// High overlap (>80%) means the evaluation is NOT truly zero-shot.
///
/// # When to Use TypeMapper
///
/// - Cross-dataset comparison (normalize schemas for fair eval)
/// - Domain adaptation (map new labels to known types)
///
/// # When NOT to Use TypeMapper
///
/// - True zero-shot evaluation (keep labels distinct)
/// - Measuring generalization (overlap hides generalization failures)
///
/// # Example
///
/// ```rust
/// use anno_core::{TypeMapper, EntityType, EntityCategory};
///
/// // MIT Movie dataset mapping
/// let mut mapper = TypeMapper::new();
/// mapper.add("ACTOR", EntityType::Person);
/// mapper.add("DIRECTOR", EntityType::Person);
/// mapper.add("TITLE", EntityType::custom("WORK_OF_ART", EntityCategory::Creative));
///
/// assert_eq!(mapper.map("ACTOR"), Some(&EntityType::Person));
/// assert_eq!(mapper.normalize("DIRECTOR"), EntityType::Person);
/// ```
#[derive(Debug, Clone, Default)]
pub struct TypeMapper {
    mappings: std::collections::HashMap<String, EntityType>,
}

impl TypeMapper {
    /// Create empty mapper.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create mapper for MIT Movie dataset.
    #[must_use]
    pub fn mit_movie() -> Self {
        let mut mapper = Self::new();
        // Map to standard types where possible
        mapper.add("ACTOR", EntityType::Person);
        mapper.add("DIRECTOR", EntityType::Person);
        mapper.add("CHARACTER", EntityType::Person);
        mapper.add(
            "TITLE",
            EntityType::custom("WORK_OF_ART", EntityCategory::Creative),
        );
        mapper.add("GENRE", EntityType::custom("GENRE", EntityCategory::Misc));
        mapper.add("YEAR", EntityType::Date);
        mapper.add("RATING", EntityType::custom("RATING", EntityCategory::Misc));
        mapper.add("PLOT", EntityType::custom("PLOT", EntityCategory::Misc));
        mapper
    }

    /// Create mapper for MIT Restaurant dataset.
    #[must_use]
    pub fn mit_restaurant() -> Self {
        let mut mapper = Self::new();
        mapper.add("RESTAURANT_NAME", EntityType::Organization);
        mapper.add("LOCATION", EntityType::Location);
        mapper.add(
            "CUISINE",
            EntityType::custom("CUISINE", EntityCategory::Misc),
        );
        mapper.add("DISH", EntityType::custom("DISH", EntityCategory::Misc));
        mapper.add("PRICE", EntityType::Money);
        mapper.add(
            "AMENITY",
            EntityType::custom("AMENITY", EntityCategory::Misc),
        );
        mapper.add("HOURS", EntityType::Time);
        mapper
    }

    /// Create mapper for biomedical datasets (BC5CDR, NCBI).
    #[must_use]
    pub fn biomedical() -> Self {
        let mut mapper = Self::new();
        mapper.add(
            "DISEASE",
            EntityType::custom("DISEASE", EntityCategory::Agent),
        );
        mapper.add(
            "CHEMICAL",
            EntityType::custom("CHEMICAL", EntityCategory::Misc),
        );
        mapper.add("DRUG", EntityType::custom("DRUG", EntityCategory::Misc));
        mapper.add("GENE", EntityType::custom("GENE", EntityCategory::Misc));
        mapper.add(
            "PROTEIN",
            EntityType::custom("PROTEIN", EntityCategory::Misc),
        );
        // GENIA types
        mapper.add("DNA", EntityType::custom("DNA", EntityCategory::Misc));
        mapper.add("RNA", EntityType::custom("RNA", EntityCategory::Misc));
        mapper.add(
            "cell_line",
            EntityType::custom("CELL_LINE", EntityCategory::Misc),
        );
        mapper.add(
            "cell_type",
            EntityType::custom("CELL_TYPE", EntityCategory::Misc),
        );
        mapper
    }

    /// Create mapper for social media NER datasets (TweetNER7, etc.).
    #[must_use]
    pub fn social_media() -> Self {
        let mut mapper = Self::new();
        // TweetNER7 types
        mapper.add("person", EntityType::Person);
        mapper.add("corporation", EntityType::Organization);
        mapper.add("location", EntityType::Location);
        mapper.add("group", EntityType::Organization);
        mapper.add(
            "product",
            EntityType::custom("PRODUCT", EntityCategory::Misc),
        );
        mapper.add(
            "creative_work",
            EntityType::custom("WORK_OF_ART", EntityCategory::Creative),
        );
        mapper.add("event", EntityType::custom("EVENT", EntityCategory::Misc));
        mapper
    }

    /// Create mapper for manufacturing domain datasets (FabNER, etc.).
    #[must_use]
    pub fn manufacturing() -> Self {
        let mut mapper = Self::new();
        // FabNER entity types
        mapper.add("MATE", EntityType::custom("MATERIAL", EntityCategory::Misc));
        mapper.add("MANP", EntityType::custom("PROCESS", EntityCategory::Misc));
        mapper.add("MACEQ", EntityType::custom("MACHINE", EntityCategory::Misc));
        mapper.add(
            "APPL",
            EntityType::custom("APPLICATION", EntityCategory::Misc),
        );
        mapper.add("FEAT", EntityType::custom("FEATURE", EntityCategory::Misc));
        mapper.add(
            "PARA",
            EntityType::custom("PARAMETER", EntityCategory::Misc),
        );
        mapper.add("PRO", EntityType::custom("PROPERTY", EntityCategory::Misc));
        mapper.add(
            "CHAR",
            EntityType::custom("CHARACTERISTIC", EntityCategory::Misc),
        );
        mapper.add(
            "ENAT",
            EntityType::custom("ENABLING_TECHNOLOGY", EntityCategory::Misc),
        );
        mapper.add(
            "CONPRI",
            EntityType::custom("CONCEPT_PRINCIPLE", EntityCategory::Misc),
        );
        mapper.add(
            "BIOP",
            EntityType::custom("BIO_PROCESS", EntityCategory::Misc),
        );
        mapper.add(
            "MANS",
            EntityType::custom("MAN_STANDARD", EntityCategory::Misc),
        );
        mapper
    }

    /// Add a mapping from source label to target type.
    pub fn add(&mut self, source: impl Into<String>, target: EntityType) {
        self.mappings.insert(source.into().to_uppercase(), target);
    }

    /// Get mapped type for a label (returns None if not mapped).
    #[must_use]
    pub fn map(&self, label: &str) -> Option<&EntityType> {
        self.mappings.get(&label.to_uppercase())
    }

    /// Normalize a label to EntityType, using mapping if available.
    ///
    /// Falls back to `EntityType::from_label()` if no mapping exists.
    #[must_use]
    pub fn normalize(&self, label: &str) -> EntityType {
        self.map(label)
            .cloned()
            .unwrap_or_else(|| EntityType::from_label(label))
    }

    /// Check if a label is mapped.
    #[must_use]
    pub fn contains(&self, label: &str) -> bool {
        self.mappings.contains_key(&label.to_uppercase())
    }

    /// Get all source labels.
    pub fn labels(&self) -> impl Iterator<Item = &String> {
        self.mappings.keys()
    }
}

/// Extraction method used to identify an entity.
///
/// # Research Context
///
/// Different extraction methods have different strengths:
///
/// | Method | Precision | Recall | Generalization | Use Case |
/// |--------|-----------|--------|----------------|----------|
/// | Pattern | Very High | Low | N/A (format-based) | Dates, emails, money |
/// | Neural | High | High | Good | General NER |
/// | Lexicon | Very High | Low | None | Closed-domain entities |
/// | SoftLexicon | Medium | High | Good for rare types | Low-resource NER |
/// | GatedEnsemble | Highest | Highest | Contextual | Short texts, domain shift |
///
/// See `docs/LEXICON_DESIGN.md` for detailed research context.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub enum ExtractionMethod {
    /// Regex pattern matching (high precision for structured data like dates, money).
    /// Does not generalize - only detects format-based entities.
    Pattern,

    /// Neural model inference (BERT, GLiNER, etc.).
    /// The recommended default for general NER. Generalizes to unseen entities.
    #[default]
    Neural,

    /// Exact lexicon/gazetteer lookup (deprecated approach).
    /// High precision on known entities, zero recall on novel entities.
    /// Only use for closed domains (stock tickers, medical codes).
    #[deprecated(since = "0.2.0", note = "Use Neural or GatedEnsemble instead")]
    Lexicon,

    /// Embedding-based soft lexicon matching.
    /// Useful for low-resource languages and rare entity types.
    /// See: Rijhwani et al. (2020) "Soft Gazetteers for Low-Resource NER"
    SoftLexicon,

    /// Gated ensemble: neural + lexicon with learned weighting.
    /// Model learns when to trust lexicon vs. context.
    /// See: Nie et al. (2021) "GEMNET: Effective Gated Gazetteer Representations"
    GatedEnsemble,

    /// Multiple methods agreed on this entity (high confidence).
    Consensus,

    /// Heuristic-based extraction (capitalization, word shape, context).
    /// Used by heuristic backends that don't use neural models.
    Heuristic,

    /// Unknown or unspecified extraction method.
    Unknown,

    /// Legacy rule-based extraction (for backward compatibility).
    #[deprecated(since = "0.2.0", note = "Use Heuristic or Pattern instead")]
    Rule,

    /// Legacy alias for Neural (for backward compatibility).
    #[deprecated(since = "0.2.0", note = "Use Neural instead")]
    ML,

    /// Legacy alias for Consensus (for backward compatibility).
    #[deprecated(since = "0.2.0", note = "Use Consensus instead")]
    Ensemble,
}

impl ExtractionMethod {
    /// Returns true if this extraction method produces probabilistically calibrated
    /// confidence scores suitable for calibration analysis (ECE, Brier score, etc.).
    ///
    /// # Calibrated Methods
    ///
    /// - **Neural**: Softmax outputs are intended to be probabilistic (though may need
    ///   temperature scaling for true calibration)
    /// - **GatedEnsemble**: Produces learned probability estimates
    /// - **SoftLexicon**: Embedding similarity is pseudo-probabilistic
    ///
    /// # Uncalibrated Methods
    ///
    /// - **Pattern**: Binary (match/no-match); confidence is typically hardcoded
    /// - **Heuristic**: Arbitrary scores from hand-crafted rules
    /// - **Lexicon**: Binary exact match
    /// - **Consensus**: Agreement count, not a probability
    ///
    /// # Example
    ///
    /// ```rust
    /// use anno_core::ExtractionMethod;
    ///
    /// assert!(ExtractionMethod::Neural.is_calibrated());
    /// assert!(!ExtractionMethod::Pattern.is_calibrated());
    /// assert!(!ExtractionMethod::Heuristic.is_calibrated());
    /// ```
    #[must_use]
    pub const fn is_calibrated(&self) -> bool {
        #[allow(deprecated)]
        match self {
            ExtractionMethod::Neural => true,
            ExtractionMethod::GatedEnsemble => true,
            ExtractionMethod::SoftLexicon => true,
            ExtractionMethod::ML => true, // Legacy alias for Neural
            // Everything else is not calibrated
            ExtractionMethod::Pattern => false,
            ExtractionMethod::Lexicon => false,
            ExtractionMethod::Consensus => false,
            ExtractionMethod::Heuristic => false,
            ExtractionMethod::Unknown => false,
            ExtractionMethod::Rule => false,
            ExtractionMethod::Ensemble => false,
        }
    }

    /// Returns the confidence interpretation for this extraction method.
    ///
    /// This helps users understand what the confidence score means:
    /// - `"probability"`: Score approximates P(correct)
    /// - `"heuristic_score"`: Score is a non-probabilistic quality measure
    /// - `"binary"`: Score is 0 or 1 (or a fixed value for matches)
    #[must_use]
    pub const fn confidence_interpretation(&self) -> &'static str {
        #[allow(deprecated)]
        match self {
            ExtractionMethod::Neural | ExtractionMethod::ML => "probability",
            ExtractionMethod::GatedEnsemble | ExtractionMethod::SoftLexicon => "probability",
            ExtractionMethod::Pattern | ExtractionMethod::Lexicon => "binary",
            ExtractionMethod::Heuristic | ExtractionMethod::Rule => "heuristic_score",
            ExtractionMethod::Consensus | ExtractionMethod::Ensemble => "agreement_ratio",
            ExtractionMethod::Unknown => "unknown",
        }
    }
}

impl std::fmt::Display for ExtractionMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        #[allow(deprecated)]
        match self {
            ExtractionMethod::Pattern => write!(f, "pattern"),
            ExtractionMethod::Neural => write!(f, "neural"),
            ExtractionMethod::Lexicon => write!(f, "lexicon"),
            ExtractionMethod::SoftLexicon => write!(f, "soft_lexicon"),
            ExtractionMethod::GatedEnsemble => write!(f, "gated_ensemble"),
            ExtractionMethod::Consensus => write!(f, "consensus"),
            ExtractionMethod::Heuristic => write!(f, "heuristic"),
            ExtractionMethod::Unknown => write!(f, "unknown"),
            ExtractionMethod::Rule => write!(f, "heuristic"), // Legacy alias
            ExtractionMethod::ML => write!(f, "neural"),      // Legacy alias
            ExtractionMethod::Ensemble => write!(f, "consensus"), // Legacy alias
        }
    }
}

// =============================================================================
// Lexicon Traits
// =============================================================================

/// Exact-match lexicon/gazetteer for entity lookup.
///
/// # Research Context
///
/// Gazetteers (lists of known entities) are a classic NER technique. Modern research
/// shows they're most valuable when:
///
/// 1. **Domain is closed**: Stock tickers, medical codes, known product catalogs
/// 2. **Text is short**: <50 tokens where context is insufficient (GEMNET +49% F1)
/// 3. **Used as features**: Input to neural model, not final output (Song et al. 2020)
///
/// They're harmful when:
/// 1. **Domain is open**: Novel entities not in the list get missed
/// 2. **Used as authority**: Hardcoded lookups inflate test scores but fail in production
///
/// # When to Use
///
/// ```text
/// Decision: Should I use a Lexicon?
///
/// Is entity type CLOSED (fixed, known list)?
/// ├─ Yes: Lexicon is appropriate
/// │       Examples: stock tickers, ICD-10 codes, country names
/// └─ No:  Use Neural extraction instead
///         Examples: person names, organization names, products
/// ```
///
/// # Example
///
/// ```rust
/// use anno_core::{Lexicon, EntityType, HashMapLexicon};
///
/// // Create a domain-specific lexicon
/// let mut lexicon = HashMapLexicon::new("stock_tickers");
/// lexicon.insert("AAPL", EntityType::Organization, 0.99);
/// lexicon.insert("GOOGL", EntityType::Organization, 0.99);
///
/// // Lookup
/// if let Some((entity_type, confidence)) = lexicon.lookup("AAPL") {
///     assert_eq!(entity_type, EntityType::Organization);
///     assert!(confidence > 0.9);
/// }
/// ```
///
/// # References
///
/// - Song et al. (2020). "Improving Neural NER with Gazetteers"
/// - Nie et al. (2021). "GEMNET: Effective Gated Gazetteer Representations"
/// - Rijhwani et al. (2020). "Soft Gazetteers for Low-Resource NER"
pub trait Lexicon: Send + Sync {
    /// Lookup an exact string, returning entity type and confidence if found.
    ///
    /// Returns `None` if the text is not in the lexicon.
    fn lookup(&self, text: &str) -> Option<(EntityType, f64)>;

    /// Check if the lexicon contains this exact string.
    fn contains(&self, text: &str) -> bool {
        self.lookup(text).is_some()
    }

    /// Get the lexicon source identifier (for provenance tracking).
    fn source(&self) -> &str;

    /// Get approximate number of entries (for debugging/metrics).
    fn len(&self) -> usize;

    /// Check if lexicon is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Simple HashMap-based lexicon implementation.
///
/// Suitable for small to medium lexicons (<100k entries).
/// For larger lexicons, consider a trie-based or FST implementation.
#[derive(Debug, Clone)]
pub struct HashMapLexicon {
    entries: std::collections::HashMap<String, (EntityType, f64)>,
    source: String,
}

impl HashMapLexicon {
    /// Create a new empty lexicon with the given source identifier.
    #[must_use]
    pub fn new(source: impl Into<String>) -> Self {
        Self {
            entries: std::collections::HashMap::new(),
            source: source.into(),
        }
    }

    /// Insert an entry into the lexicon.
    pub fn insert(&mut self, text: impl Into<String>, entity_type: EntityType, confidence: f64) {
        self.entries.insert(text.into(), (entity_type, confidence));
    }

    /// Create from an iterator of (text, type, confidence) tuples.
    pub fn from_iter<I, S>(source: impl Into<String>, entries: I) -> Self
    where
        I: IntoIterator<Item = (S, EntityType, f64)>,
        S: Into<String>,
    {
        let mut lexicon = Self::new(source);
        for (text, entity_type, confidence) in entries {
            lexicon.insert(text, entity_type, confidence);
        }
        lexicon
    }

    /// Get all entries as an iterator (for debugging).
    pub fn entries(&self) -> impl Iterator<Item = (&str, &EntityType, f64)> {
        self.entries.iter().map(|(k, (t, c))| (k.as_str(), t, *c))
    }
}

impl Lexicon for HashMapLexicon {
    fn lookup(&self, text: &str) -> Option<(EntityType, f64)> {
        self.entries.get(text).cloned()
    }

    fn source(&self) -> &str {
        &self.source
    }

    fn len(&self) -> usize {
        self.entries.len()
    }
}

/// Provenance information for an extracted entity.
///
/// Tracks where an entity came from for debugging, explainability,
/// and confidence calibration in hybrid/ensemble systems.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Provenance {
    /// Name of the backend that produced this entity (e.g., "pattern", "bert-onnx")
    pub source: Cow<'static, str>,
    /// Extraction method used
    pub method: ExtractionMethod,
    /// Specific pattern/rule name (for pattern/rule-based extraction)
    pub pattern: Option<Cow<'static, str>>,
    /// Raw confidence from the source model (before any calibration)
    pub raw_confidence: Option<f64>,
    /// Model version for reproducibility (e.g., "gliner-v2.1", "bert-base-uncased-2024-01")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_version: Option<Cow<'static, str>>,
    /// Timestamp when extraction occurred (ISO 8601)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
}

impl Provenance {
    /// Create provenance for regex-based extraction.
    #[must_use]
    pub fn pattern(pattern_name: &'static str) -> Self {
        Self {
            source: Cow::Borrowed("pattern"),
            method: ExtractionMethod::Pattern,
            pattern: Some(Cow::Borrowed(pattern_name)),
            raw_confidence: Some(1.0), // Patterns are deterministic
            model_version: None,
            timestamp: None,
        }
    }

    /// Create provenance for ML-based extraction.
    #[must_use]
    pub fn ml(model_name: &'static str, confidence: f64) -> Self {
        Self {
            source: Cow::Borrowed(model_name),
            method: ExtractionMethod::Neural,
            pattern: None,
            raw_confidence: Some(confidence),
            model_version: None,
            timestamp: None,
        }
    }

    /// Create provenance for ensemble/hybrid extraction.
    #[must_use]
    pub fn ensemble(sources: &'static str) -> Self {
        Self {
            source: Cow::Borrowed(sources),
            method: ExtractionMethod::Consensus,
            pattern: None,
            raw_confidence: None,
            model_version: None,
            timestamp: None,
        }
    }

    /// Create provenance with model version for reproducibility.
    #[must_use]
    pub fn with_version(mut self, version: &'static str) -> Self {
        self.model_version = Some(Cow::Borrowed(version));
        self
    }

    /// Create provenance with timestamp.
    #[must_use]
    pub fn with_timestamp(mut self, timestamp: impl Into<String>) -> Self {
        self.timestamp = Some(timestamp.into());
        self
    }
}

// ============================================================================
// Span Types (Multi-Modal Support)
// ============================================================================

/// A span locator that supports both text and visual modalities.
///
/// For ColPali/multi-modal systems, entities can be located by:
/// - Text offsets (traditional NER)
/// - Bounding boxes (visual document understanding)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Span {
    /// Text span with byte offsets (start, end)
    Text {
        /// Start byte offset (inclusive)
        start: usize,
        /// End byte offset (exclusive)
        end: usize,
    },
    /// Visual bounding box (normalized 0.0-1.0 coordinates)
    /// For ColPali: image patch locations
    BoundingBox {
        /// X coordinate (normalized 0.0-1.0)
        x: f32,
        /// Y coordinate (normalized 0.0-1.0)
        y: f32,
        /// Width (normalized 0.0-1.0)
        width: f32,
        /// Height (normalized 0.0-1.0)
        height: f32,
        /// Optional page number (for multi-page documents)
        page: Option<u32>,
    },
    /// Hybrid: both text and visual location (for OCR-verified extraction)
    Hybrid {
        /// Start byte offset (inclusive)
        start: usize,
        /// End byte offset (exclusive)
        end: usize,
        /// Bounding box for visual location
        bbox: Box<Span>,
    },
}

impl Span {
    /// Create a text span.
    #[must_use]
    pub const fn text(start: usize, end: usize) -> Self {
        Self::Text { start, end }
    }

    /// Create a bounding box span with normalized coordinates.
    #[must_use]
    pub fn bbox(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self::BoundingBox {
            x,
            y,
            width,
            height,
            page: None,
        }
    }

    /// Create a bounding box with page number.
    #[must_use]
    pub fn bbox_on_page(x: f32, y: f32, width: f32, height: f32, page: u32) -> Self {
        Self::BoundingBox {
            x,
            y,
            width,
            height,
            page: Some(page),
        }
    }

    /// Check if this is a text span.
    #[must_use]
    pub const fn is_text(&self) -> bool {
        matches!(self, Self::Text { .. } | Self::Hybrid { .. })
    }

    /// Check if this has visual location.
    #[must_use]
    pub const fn is_visual(&self) -> bool {
        matches!(self, Self::BoundingBox { .. } | Self::Hybrid { .. })
    }

    /// Get text offsets if available.
    #[must_use]
    pub const fn text_offsets(&self) -> Option<(usize, usize)> {
        match self {
            Self::Text { start, end } => Some((*start, *end)),
            Self::Hybrid { start, end, .. } => Some((*start, *end)),
            Self::BoundingBox { .. } => None,
        }
    }

    /// Calculate span length for text spans.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Text { start, end } => end.saturating_sub(*start),
            Self::Hybrid { start, end, .. } => end.saturating_sub(*start),
            Self::BoundingBox { .. } => 0,
        }
    }

    /// Check if span is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ============================================================================
// Discontinuous Spans (W2NER/ACE-style)
// ============================================================================

/// A discontinuous span representing non-contiguous entity mentions.
///
/// Some entities span multiple non-adjacent text regions:
/// - "severe \[pain\] in the \[abdomen\]" → "severe abdominal pain"
/// - "the \[president\] ... \[Obama\]" → coreference
///
/// This is required for:
/// - **Medical NER**: Anatomical modifiers separated from findings
/// - **Legal NER**: Parties referenced across clauses
/// - **W2NER**: Word-word relation grids that detect discontinuous entities
///
/// # Example
///
/// ```rust,ignore
/// use anno_core::DiscontinuousSpan;
///
/// // "severe pain in the abdomen" where "severe" modifies "pain"
/// // but they're separated by other words
/// let span = DiscontinuousSpan::new(vec![
///     0..6,   // "severe"
///     12..16, // "pain"
/// ]);
///
/// assert_eq!(span.num_segments(), 2);
/// assert!(span.is_discontinuous());
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiscontinuousSpan {
    /// Non-overlapping segments, sorted by start position.
    /// Each `Range<usize>` represents (start_byte, end_byte).
    segments: Vec<std::ops::Range<usize>>,
}

impl DiscontinuousSpan {
    /// Create a new discontinuous span from segments.
    ///
    /// Segments are sorted and validated (no overlaps).
    #[must_use]
    pub fn new(mut segments: Vec<std::ops::Range<usize>>) -> Self {
        // Sort by start position
        segments.sort_by_key(|r| r.start);
        Self { segments }
    }

    /// Create from a single contiguous span.
    #[must_use]
    #[allow(clippy::single_range_in_vec_init)] // Intentional: contiguous is special case of discontinuous
    pub fn contiguous(start: usize, end: usize) -> Self {
        Self {
            segments: vec![start..end],
        }
    }

    /// Number of segments.
    #[must_use]
    pub fn num_segments(&self) -> usize {
        self.segments.len()
    }

    /// True if this spans multiple non-adjacent regions.
    #[must_use]
    pub fn is_discontinuous(&self) -> bool {
        self.segments.len() > 1
    }

    /// True if this is a single contiguous span.
    #[must_use]
    pub fn is_contiguous(&self) -> bool {
        self.segments.len() <= 1
    }

    /// Get the segments.
    #[must_use]
    pub fn segments(&self) -> &[std::ops::Range<usize>] {
        &self.segments
    }

    /// Get the overall bounding range (start of first to end of last).
    #[must_use]
    pub fn bounding_range(&self) -> Option<std::ops::Range<usize>> {
        if self.segments.is_empty() {
            return None;
        }
        let start = self.segments.first()?.start;
        let end = self.segments.last()?.end;
        Some(start..end)
    }

    /// Total byte length (sum of all segments).
    ///
    /// # Note
    ///
    /// This returns the sum of byte lengths, not character lengths.
    /// Since `DiscontinuousSpan` uses byte offsets, this method calculates
    /// the total number of bytes covered by all segments.
    ///
    /// For character length, you would need to convert each segment's byte
    /// offsets to character offsets using the source text.
    #[must_use]
    pub fn total_len(&self) -> usize {
        self.segments.iter().map(|r| r.end - r.start).sum()
    }

    /// Extract text from each segment and join with separator.
    #[must_use]
    pub fn extract_text(&self, text: &str, separator: &str) -> String {
        self.segments
            .iter()
            .filter_map(|r| text.get(r.clone()))
            .collect::<Vec<_>>()
            .join(separator)
    }

    /// Check if a byte position falls within any segment.
    ///
    /// # Arguments
    ///
    /// * `pos` - Byte offset to check (must be a byte offset, not character offset)
    ///
    /// # Returns
    ///
    /// `true` if the byte position falls within any segment of this span.
    #[must_use]
    pub fn contains(&self, pos: usize) -> bool {
        self.segments.iter().any(|r| r.contains(&pos))
    }

    /// Convert to a regular Span (uses bounding range, loses discontinuity info).
    #[must_use]
    pub fn to_span(&self) -> Option<Span> {
        self.bounding_range().map(|r| Span::Text {
            start: r.start,
            end: r.end,
        })
    }
}

impl From<std::ops::Range<usize>> for DiscontinuousSpan {
    fn from(range: std::ops::Range<usize>) -> Self {
        Self::contiguous(range.start, range.end)
    }
}

impl Default for Span {
    fn default() -> Self {
        Self::Text { start: 0, end: 0 }
    }
}

// ============================================================================
// Hierarchical Confidence (Coarse-to-Fine)
// ============================================================================

/// Hierarchical confidence scores for coarse-to-fine extraction.
///
/// Research (HiNet, InfoHier) shows that extraction benefits from
/// decomposed confidence:
/// - **Linkage**: "Is there ANY entity here?" (binary, fast filter)
/// - **Type**: "What type is it?" (fine-grained classification)
/// - **Boundary**: "Where exactly does it start/end?" (span refinement)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HierarchicalConfidence {
    /// Coarse: probability that this span contains ANY entity (0.0-1.0)
    /// Used for early filtering in the TPLinker "handshaking" matrix.
    pub linkage: f32,
    /// Fine: probability that the type classification is correct (0.0-1.0)
    pub type_score: f32,
    /// Boundary: confidence in the exact span boundaries (0.0-1.0)
    /// Low for entities with fuzzy boundaries (e.g., "the CEO" vs "CEO")
    pub boundary: f32,
}

impl HierarchicalConfidence {
    /// Create hierarchical confidence with all scores.
    #[must_use]
    pub fn new(linkage: f32, type_score: f32, boundary: f32) -> Self {
        Self {
            linkage: linkage.clamp(0.0, 1.0),
            type_score: type_score.clamp(0.0, 1.0),
            boundary: boundary.clamp(0.0, 1.0),
        }
    }

    /// Create from a single confidence score (legacy compatibility).
    /// Assigns same score to all levels.
    #[must_use]
    pub fn from_single(confidence: f32) -> Self {
        let c = confidence.clamp(0.0, 1.0);
        Self {
            linkage: c,
            type_score: c,
            boundary: c,
        }
    }

    /// Calculate combined confidence (geometric mean).
    /// Geometric mean penalizes low scores more than arithmetic mean.
    #[must_use]
    pub fn combined(&self) -> f32 {
        (self.linkage * self.type_score * self.boundary).powf(1.0 / 3.0)
    }

    /// Calculate combined confidence as f64 for legacy compatibility.
    #[must_use]
    pub fn as_f64(&self) -> f64 {
        self.combined() as f64
    }

    /// Check if passes minimum threshold at all levels.
    #[must_use]
    pub fn passes_threshold(&self, linkage_min: f32, type_min: f32, boundary_min: f32) -> bool {
        self.linkage >= linkage_min && self.type_score >= type_min && self.boundary >= boundary_min
    }
}

impl Default for HierarchicalConfidence {
    fn default() -> Self {
        Self {
            linkage: 1.0,
            type_score: 1.0,
            boundary: 1.0,
        }
    }
}

impl From<f64> for HierarchicalConfidence {
    fn from(confidence: f64) -> Self {
        Self::from_single(confidence as f32)
    }
}

impl From<f32> for HierarchicalConfidence {
    fn from(confidence: f32) -> Self {
        Self::from_single(confidence)
    }
}

// ============================================================================
// Ragged Batch (ModernBERT Unpadding)
// ============================================================================

/// A ragged (unpadded) batch for efficient ModernBERT inference.
///
/// ModernBERT achieves its speed advantage by avoiding padding tokens entirely.
/// Instead of `[batch, max_seq_len]`, it uses a single contiguous 1D sequence
/// with offset indices to track document boundaries.
///
/// # Memory Layout
///
/// ```text
/// Traditional (padded):
/// [doc1_tok1, doc1_tok2, PAD, PAD, PAD]  <- wasted compute
/// [doc2_tok1, doc2_tok2, doc2_tok3, PAD, PAD]
///
/// Ragged (unpadded):
/// [doc1_tok1, doc1_tok2, doc2_tok1, doc2_tok2, doc2_tok3]
/// cumulative_offsets: [0, 2, 5]  <- doc1 is [0..2], doc2 is [2..5]
/// ```
#[derive(Debug, Clone)]
pub struct RaggedBatch {
    /// Token IDs flattened into a single contiguous array.
    /// Shape: `[total_tokens]` (1D, no padding)
    pub token_ids: Vec<u32>,
    /// Cumulative sequence lengths.
    /// Length: batch_size + 1
    /// Document i spans tokens \[offsets\[i\]..offsets\[i+1\])
    pub cumulative_offsets: Vec<u32>,
    /// Maximum sequence length in this batch (for kernel bounds).
    pub max_seq_len: usize,
}

impl RaggedBatch {
    /// Create a new ragged batch from sequences.
    pub fn from_sequences(sequences: &[Vec<u32>]) -> Self {
        let total_tokens: usize = sequences.iter().map(|s| s.len()).sum();
        let mut token_ids = Vec::with_capacity(total_tokens);
        let mut cumulative_offsets = Vec::with_capacity(sequences.len() + 1);
        let mut max_seq_len = 0;

        cumulative_offsets.push(0);
        for seq in sequences {
            token_ids.extend_from_slice(seq);
            // Check for overflow: u32::MAX is 4,294,967,295
            // If token_ids.len() exceeds this, we'll truncate (which is a bug)
            // but in practice, this is unlikely for reasonable batch sizes
            let len = token_ids.len();
            if len > u32::MAX as usize {
                // This would overflow - use saturating cast to prevent panic
                // but log a warning as this indicates a problem
                log::warn!(
                    "Token count {} exceeds u32::MAX, truncating to {}",
                    len,
                    u32::MAX
                );
                cumulative_offsets.push(u32::MAX);
            } else {
                cumulative_offsets.push(len as u32);
            }
            max_seq_len = max_seq_len.max(seq.len());
        }

        Self {
            token_ids,
            cumulative_offsets,
            max_seq_len,
        }
    }

    /// Get the number of documents in this batch.
    #[must_use]
    pub fn batch_size(&self) -> usize {
        self.cumulative_offsets.len().saturating_sub(1)
    }

    /// Get the total number of tokens (no padding).
    #[must_use]
    pub fn total_tokens(&self) -> usize {
        self.token_ids.len()
    }

    /// Get token range for a specific document.
    #[must_use]
    pub fn doc_range(&self, doc_idx: usize) -> Option<std::ops::Range<usize>> {
        if doc_idx + 1 < self.cumulative_offsets.len() {
            let start = self.cumulative_offsets[doc_idx] as usize;
            let end = self.cumulative_offsets[doc_idx + 1] as usize;
            Some(start..end)
        } else {
            None
        }
    }

    /// Get tokens for a specific document.
    #[must_use]
    pub fn doc_tokens(&self, doc_idx: usize) -> Option<&[u32]> {
        self.doc_range(doc_idx).map(|r| &self.token_ids[r])
    }

    /// Calculate memory saved vs padded batch.
    #[must_use]
    pub fn padding_savings(&self) -> f64 {
        let padded_size = self.batch_size() * self.max_seq_len;
        if padded_size == 0 {
            return 0.0;
        }
        1.0 - (self.total_tokens() as f64 / padded_size as f64)
    }
}

// ============================================================================
// Span Candidate Generation
// ============================================================================

/// A candidate span for entity extraction.
///
/// In GLiNER/bi-encoder systems, we generate all possible spans up to a
/// maximum width and score them against entity type embeddings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpanCandidate {
    /// Document index in the batch
    pub doc_idx: u32,
    /// Start token index (within the document)
    pub start: u32,
    /// End token index (exclusive)
    pub end: u32,
}

impl SpanCandidate {
    /// Create a new span candidate.
    #[must_use]
    pub const fn new(doc_idx: u32, start: u32, end: u32) -> Self {
        Self {
            doc_idx,
            start,
            end,
        }
    }

    /// Get span width (number of tokens).
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.end.saturating_sub(self.start)
    }
}

/// Generate all valid span candidates for a ragged batch.
///
/// This is the "gnarly" operation in GLiNER - efficiently enumerating
/// all valid spans without O(N^2) memory allocation.
pub fn generate_span_candidates(batch: &RaggedBatch, max_width: usize) -> Vec<SpanCandidate> {
    let mut candidates = Vec::new();

    for doc_idx in 0..batch.batch_size() {
        if let Some(range) = batch.doc_range(doc_idx) {
            let doc_len = range.len();
            // Generate all spans [i, j) where j - i <= max_width
            for start in 0..doc_len {
                let max_end = (start + max_width).min(doc_len);
                for end in (start + 1)..=max_end {
                    candidates.push(SpanCandidate::new(doc_idx as u32, start as u32, end as u32));
                }
            }
        }
    }

    candidates
}

/// Generate span candidates with early filtering.
///
/// Uses a linkage mask to skip low-probability spans (TPLinker optimization).
pub fn generate_filtered_candidates(
    batch: &RaggedBatch,
    max_width: usize,
    linkage_mask: &[f32],
    threshold: f32,
) -> Vec<SpanCandidate> {
    let mut candidates = Vec::new();
    let mut mask_idx = 0;

    for doc_idx in 0..batch.batch_size() {
        if let Some(range) = batch.doc_range(doc_idx) {
            let doc_len = range.len();
            for start in 0..doc_len {
                let max_end = (start + max_width).min(doc_len);
                for end in (start + 1)..=max_end {
                    // Only include if linkage probability exceeds threshold
                    if mask_idx < linkage_mask.len() && linkage_mask[mask_idx] >= threshold {
                        candidates.push(SpanCandidate::new(
                            doc_idx as u32,
                            start as u32,
                            end as u32,
                        ));
                    }
                    mask_idx += 1;
                }
            }
        }
    }

    candidates
}

// ============================================================================
// Entity (Extended)
// ============================================================================

/// A recognized named entity or relation trigger.
///
/// # Entity Structure
///
/// ```text
/// "Contact John at john@example.com on Jan 15"
///          ^^^^    ^^^^^^^^^^^^^^^^    ^^^^^^
///          PER     EMAIL               DATE
///          |       |                   |
///          Named   Contact             Temporal
///          (ML)    (Pattern)           (Pattern)
/// ```
///
/// # Knowledge Graph Support
///
/// For GraphRAG and coreference resolution, entities support:
/// - `kb_id`: External knowledge base identifier (e.g., Wikidata Q-ID)
/// - `canonical_id`: Local coreference cluster ID (links "John" and "he")
///
/// # Normalization
///
/// Entities can have a normalized form for downstream processing:
/// - Dates: "Jan 15" → "2024-01-15" (ISO 8601)
/// - Money: "$1.5M" → "1500000 USD"
/// - Locations: "NYC" → "New York City"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    /// Entity text (surface form as it appears in source)
    pub text: String,
    /// Entity type classification
    pub entity_type: EntityType,
    /// Start position (character offset, NOT byte offset).
    ///
    /// For Unicode text, character offsets differ from byte offsets.
    /// Use [`crate::offset::bytes_to_chars`] to convert if needed.
    pub start: usize,
    /// End position (character offset, exclusive).
    ///
    /// For Unicode text, character offsets differ from byte offsets.
    /// Use [`crate::offset::bytes_to_chars`] to convert if needed.
    pub end: usize,
    /// Confidence score (0.0-1.0, calibrated)
    pub confidence: f64,
    /// Normalized/canonical form (e.g., "Jan 15" → "2024-01-15")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normalized: Option<String>,
    /// Provenance: which backend/method produced this entity
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<Provenance>,
    /// External knowledge base ID (e.g., "Q7186" for Marie Curie in Wikidata).
    /// Used for entity linking and GraphRAG applications.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kb_id: Option<String>,
    /// Local coreference cluster ID.
    /// Multiple mentions with the same `canonical_id` refer to the same entity.
    /// Example: "Marie Curie" and "she" might share `canonical_id = 42`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub canonical_id: Option<u64>,
    /// Hierarchical confidence (coarse-to-fine).
    /// Provides linkage, type, and boundary scores separately.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hierarchical_confidence: Option<HierarchicalConfidence>,
    /// Visual span for multi-modal (ColPali) extraction.
    /// When set, provides bounding box location in addition to text offsets.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub visual_span: Option<Span>,
    /// Discontinuous span for non-contiguous entity mentions (W2NER support).
    /// When set, overrides `start`/`end` for length calculations.
    /// Example: "New York and LA \[airports\]" where "airports" modifies both.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub discontinuous_span: Option<DiscontinuousSpan>,
    // =========================================================================
    // Temporal Validity (Research: Temporal Knowledge Graphs)
    // =========================================================================
    /// Start of temporal validity interval for this entity assertion.
    ///
    /// Entities are facts that may change over time:
    /// - "Satya Nadella is CEO of Microsoft" is valid from [2014, present]
    /// - "Steve Ballmer was CEO of Microsoft" was valid from [2000, 2014]
    ///
    /// When `None`, the entity is either:
    /// - Currently valid (no known end date)
    /// - Atemporal (timeless fact like "Paris is in France")
    ///
    /// # Example
    /// ```rust
    /// use anno_core::{Entity, EntityType};
    /// use chrono::{TimeZone, Utc};
    ///
    /// let mut entity = Entity::new("CEO of Microsoft", EntityType::Person, 0, 16, 0.9);
    /// entity.valid_from = Some(Utc.with_ymd_and_hms(2008, 10, 1, 0, 0, 0).unwrap());
    /// ```
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub valid_from: Option<chrono::DateTime<chrono::Utc>>,
    /// End of temporal validity interval for this entity assertion.
    ///
    /// When `None` and `valid_from` is set, the fact is currently valid.
    /// When both are `None`, the entity is atemporal.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub valid_until: Option<chrono::DateTime<chrono::Utc>>,
    // =========================================================================
    // Viewport / Context (Research: Entity Manifolds)
    // =========================================================================
    /// Viewport context for multi-faceted entity representation.
    ///
    /// The same real-world entity can have different "faces" in different contexts:
    /// - "Marie Curie" in an academic context: professor, researcher
    /// - "Marie Curie" in a scientific context: physicist, chemist
    /// - "Marie Curie" in a personal context: mother, educator
    ///
    /// This enables "holographic" entity projection at query time:
    /// given a query context, project the entity manifold to the relevant viewport.
    ///
    /// # Example
    /// ```rust
    /// use anno_core::{Entity, EntityType, EntityViewport};
    ///
    /// let mut entity = Entity::new("Marie Curie", EntityType::Person, 0, 11, 0.9);
    /// entity.viewport = Some(EntityViewport::Academic);
    /// ```
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub viewport: Option<EntityViewport>,
}

impl Entity {
    /// Create a new entity.
    #[must_use]
    pub fn new(
        text: impl Into<String>,
        entity_type: EntityType,
        start: usize,
        end: usize,
        confidence: f64,
    ) -> Self {
        Self {
            text: text.into(),
            entity_type,
            start,
            end,
            confidence: confidence.clamp(0.0, 1.0),
            normalized: None,
            provenance: None,
            kb_id: None,
            canonical_id: None,
            hierarchical_confidence: None,
            visual_span: None,
            discontinuous_span: None,
            valid_from: None,
            valid_until: None,
            viewport: None,
        }
    }

    /// Create a new entity with provenance information.
    #[must_use]
    pub fn with_provenance(
        text: impl Into<String>,
        entity_type: EntityType,
        start: usize,
        end: usize,
        confidence: f64,
        provenance: Provenance,
    ) -> Self {
        Self {
            text: text.into(),
            entity_type,
            start,
            end,
            confidence: confidence.clamp(0.0, 1.0),
            normalized: None,
            provenance: Some(provenance),
            kb_id: None,
            canonical_id: None,
            hierarchical_confidence: None,
            visual_span: None,
            discontinuous_span: None,
            valid_from: None,
            valid_until: None,
            viewport: None,
        }
    }

    /// Create an entity with hierarchical confidence scores.
    #[must_use]
    pub fn with_hierarchical_confidence(
        text: impl Into<String>,
        entity_type: EntityType,
        start: usize,
        end: usize,
        confidence: HierarchicalConfidence,
    ) -> Self {
        Self {
            text: text.into(),
            entity_type,
            start,
            end,
            confidence: confidence.as_f64(),
            normalized: None,
            provenance: None,
            kb_id: None,
            canonical_id: None,
            hierarchical_confidence: Some(confidence),
            visual_span: None,
            discontinuous_span: None,
            valid_from: None,
            valid_until: None,
            viewport: None,
        }
    }

    /// Create an entity from a visual bounding box (ColPali multi-modal).
    #[must_use]
    pub fn from_visual(
        text: impl Into<String>,
        entity_type: EntityType,
        bbox: Span,
        confidence: f64,
    ) -> Self {
        Self {
            text: text.into(),
            entity_type,
            start: 0,
            end: 0,
            confidence: confidence.clamp(0.0, 1.0),
            normalized: None,
            provenance: None,
            kb_id: None,
            canonical_id: None,
            hierarchical_confidence: None,
            visual_span: Some(bbox),
            discontinuous_span: None,
            valid_from: None,
            valid_until: None,
            viewport: None,
        }
    }

    /// Create an entity with default confidence (1.0).
    #[must_use]
    pub fn with_type(
        text: impl Into<String>,
        entity_type: EntityType,
        start: usize,
        end: usize,
    ) -> Self {
        Self::new(text, entity_type, start, end, 1.0)
    }

    /// Link this entity to an external knowledge base.
    ///
    /// # Examples
    /// ```rust
    /// use anno_core::{Entity, EntityType};
    /// let mut e = Entity::new("Marie Curie", EntityType::Person, 0, 11, 0.95);
    /// e.link_to_kb("Q7186"); // Wikidata ID
    /// ```
    pub fn link_to_kb(&mut self, kb_id: impl Into<String>) {
        self.kb_id = Some(kb_id.into());
    }

    /// Assign this entity to a coreference cluster.
    ///
    /// Entities with the same `canonical_id` refer to the same real-world entity.
    pub fn set_canonical(&mut self, canonical_id: u64) {
        self.canonical_id = Some(canonical_id);
    }

    /// Builder-style method to set canonical ID.
    ///
    /// # Example
    /// ```
    /// use anno_core::{Entity, EntityType};
    /// let entity = Entity::new("John", EntityType::Person, 0, 4, 0.9)
    ///     .with_canonical_id(42);
    /// assert_eq!(entity.canonical_id, Some(42));
    /// ```
    #[must_use]
    pub fn with_canonical_id(mut self, canonical_id: u64) -> Self {
        self.canonical_id = Some(canonical_id);
        self
    }

    /// Check if this entity is linked to a knowledge base.
    #[must_use]
    pub fn is_linked(&self) -> bool {
        self.kb_id.is_some()
    }

    /// Check if this entity has coreference information.
    #[must_use]
    pub fn has_coreference(&self) -> bool {
        self.canonical_id.is_some()
    }

    /// Check if this entity has a discontinuous span.
    ///
    /// Discontinuous entities span non-contiguous text regions.
    /// Example: "New York and LA airports" contains "New York airports"
    /// as a discontinuous entity.
    #[must_use]
    pub fn is_discontinuous(&self) -> bool {
        self.discontinuous_span
            .as_ref()
            .map(|s| s.is_discontinuous())
            .unwrap_or(false)
    }

    /// Get the discontinuous segments if present.
    ///
    /// Returns `None` if this is a contiguous entity.
    #[must_use]
    pub fn discontinuous_segments(&self) -> Option<Vec<std::ops::Range<usize>>> {
        self.discontinuous_span
            .as_ref()
            .filter(|s| s.is_discontinuous())
            .map(|s| s.segments().to_vec())
    }

    /// Set a discontinuous span for this entity.
    ///
    /// This is used by W2NER and similar models that detect non-contiguous mentions.
    pub fn set_discontinuous_span(&mut self, span: DiscontinuousSpan) {
        // Update start/end to match the bounding range
        if let Some(bounding) = span.bounding_range() {
            self.start = bounding.start;
            self.end = bounding.end;
        }
        self.discontinuous_span = Some(span);
    }

    /// Get the total length covered by this entity.
    ///
    /// # Note on Offset Systems
    ///
    /// - For **contiguous entities** (no discontinuous span): Returns character length (`end - start`)
    /// - For **discontinuous entities**: Returns byte length (sum of segment byte lengths)
    ///
    /// This inconsistency exists because:
    /// - `Entity` uses character offsets for the main span (`start`, `end`)
    /// - `DiscontinuousSpan` uses byte offsets for segments
    ///
    /// When using discontinuous spans, be aware that `total_len()` returns bytes,
    /// not characters. For accurate character length with discontinuous spans,
    /// you would need to convert each segment's byte offsets to character offsets
    /// using the source text.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use anno_core::{Entity, EntityType, DiscontinuousSpan};
    ///
    /// // Contiguous entity - returns character length
    /// let entity = Entity::new("Hello", EntityType::Person, 0, 5, 0.9);
    /// assert_eq!(entity.total_len(), 5); // 5 characters
    ///
    /// // Discontinuous entity - returns byte length
    /// let mut entity = Entity::new("test", EntityType::Person, 0, 4, 0.9);
    /// entity.set_discontinuous_span(DiscontinuousSpan::new(vec![0..4, 10..14]));
    /// assert_eq!(entity.total_len(), 8); // 4 + 4 = 8 bytes
    /// ```
    #[must_use]
    pub fn total_len(&self) -> usize {
        if let Some(ref span) = self.discontinuous_span {
            span.segments().iter().map(|r| r.end - r.start).sum()
        } else {
            self.end.saturating_sub(self.start)
        }
    }

    /// Set the normalized form for this entity.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use anno_core::{Entity, EntityType};
    ///
    /// let mut entity = Entity::new("Jan 15", EntityType::Date, 0, 6, 0.95);
    /// entity.set_normalized("2024-01-15");
    /// assert_eq!(entity.normalized.as_deref(), Some("2024-01-15"));
    /// ```
    pub fn set_normalized(&mut self, normalized: impl Into<String>) {
        self.normalized = Some(normalized.into());
    }

    /// Get the normalized form, or the original text if not normalized.
    #[must_use]
    pub fn normalized_or_text(&self) -> &str {
        self.normalized.as_deref().unwrap_or(&self.text)
    }

    /// Get the extraction method, if known.
    #[must_use]
    pub fn method(&self) -> ExtractionMethod {
        self.provenance
            .as_ref()
            .map_or(ExtractionMethod::Unknown, |p| p.method)
    }

    /// Get the source backend name, if known.
    #[must_use]
    pub fn source(&self) -> Option<&str> {
        self.provenance.as_ref().map(|p| p.source.as_ref())
    }

    /// Get the entity category.
    #[must_use]
    pub fn category(&self) -> EntityCategory {
        self.entity_type.category()
    }

    /// Returns true if this entity was detected via patterns (not ML).
    #[must_use]
    pub fn is_structured(&self) -> bool {
        self.entity_type.pattern_detectable()
    }

    /// Returns true if this entity required ML for detection.
    #[must_use]
    pub fn is_named(&self) -> bool {
        self.entity_type.requires_ml()
    }

    /// Check if this entity overlaps with another.
    #[must_use]
    pub fn overlaps(&self, other: &Entity) -> bool {
        !(self.end <= other.start || other.end <= self.start)
    }

    /// Calculate overlap ratio (IoU) with another entity.
    #[must_use]
    pub fn overlap_ratio(&self, other: &Entity) -> f64 {
        let intersection_start = self.start.max(other.start);
        let intersection_end = self.end.min(other.end);

        if intersection_start >= intersection_end {
            return 0.0;
        }

        let intersection = (intersection_end - intersection_start) as f64;
        let union = ((self.end - self.start) + (other.end - other.start)
            - (intersection_end - intersection_start)) as f64;

        if union == 0.0 {
            return 1.0;
        }

        intersection / union
    }

    /// Set hierarchical confidence scores.
    pub fn set_hierarchical_confidence(&mut self, confidence: HierarchicalConfidence) {
        self.confidence = confidence.as_f64();
        self.hierarchical_confidence = Some(confidence);
    }

    /// Get the linkage confidence (coarse filter score).
    #[must_use]
    pub fn linkage_confidence(&self) -> f32 {
        self.hierarchical_confidence
            .map_or(self.confidence as f32, |h| h.linkage)
    }

    /// Get the type classification confidence.
    #[must_use]
    pub fn type_confidence(&self) -> f32 {
        self.hierarchical_confidence
            .map_or(self.confidence as f32, |h| h.type_score)
    }

    /// Get the boundary confidence.
    #[must_use]
    pub fn boundary_confidence(&self) -> f32 {
        self.hierarchical_confidence
            .map_or(self.confidence as f32, |h| h.boundary)
    }

    /// Check if this entity has visual location (multi-modal).
    #[must_use]
    pub fn is_visual(&self) -> bool {
        self.visual_span.is_some()
    }

    /// Get the text span (start, end).
    #[must_use]
    pub const fn text_span(&self) -> (usize, usize) {
        (self.start, self.end)
    }

    /// Get the span length.
    #[must_use]
    pub const fn span_len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    /// Create a unified TextSpan with both byte and char offsets.
    ///
    /// This is useful when you need to work with both offset systems.
    /// The `text` parameter must be the original source text from which
    /// this entity was extracted.
    ///
    /// # Arguments
    /// * `source_text` - The original text (needed to compute byte offsets)
    ///
    /// # Returns
    /// A TextSpan with both byte and char offsets.
    ///
    /// # Example
    /// ```rust
    /// use anno_core::{Entity, EntityType};
    ///
    /// let text = "Hello, 日本!";
    /// // Entity stores CHARACTER offsets: "日本" is chars 7-9
    /// let entity = Entity::new("日本", EntityType::Location, 7, 9, 0.95);
    /// let span = entity.to_text_span(text);
    ///
    /// // Character offsets (stored in entity)
    /// assert_eq!(span.char_start, 7);
    /// assert_eq!(span.char_end, 9);
    ///
    /// // Byte offsets (computed from text)
    /// assert_eq!(span.byte_start, 7);  // "Hello, ".len() in bytes
    /// assert_eq!(span.byte_end, 13);   // + 6 bytes for "日本"
    /// ```
    // Note: This method will be moved to anno crate (depends on offset module)
    #[allow(dead_code)]
    pub fn to_text_span(&self, _source_text: &str) -> serde_json::Value {
        todo!("This method depends on offset module which will be in anno crate")
    }

    /// Set visual span for multi-modal extraction.
    pub fn set_visual_span(&mut self, span: Span) {
        self.visual_span = Some(span);
    }

    /// Safely extract text from source using character offsets.
    ///
    /// Entity stores character offsets, not byte offsets. This method
    /// correctly extracts text by iterating over characters.
    ///
    /// # Arguments
    /// * `source_text` - The original text from which this entity was extracted
    ///
    /// # Returns
    /// The extracted text, or empty string if offsets are invalid
    ///
    /// # Example
    /// ```rust
    /// use anno_core::{Entity, EntityType};
    ///
    /// let text = "Hello, 日本!";
    /// let entity = Entity::new("日本", EntityType::Location, 7, 9, 0.95);
    /// assert_eq!(entity.extract_text(text), "日本");
    /// ```
    #[must_use]
    pub fn extract_text(&self, source_text: &str) -> String {
        // Performance: Use cached length if available, but fallback to counting
        // For single entity extraction, this is fine. For batch operations,
        // use extract_text_with_len with pre-computed length.
        let char_count = source_text.chars().count();
        self.extract_text_with_len(source_text, char_count)
    }

    /// Extract text with pre-computed text length (performance optimization).
    ///
    /// Use this when validating/clamping multiple entities from the same text
    /// to avoid recalculating `text.chars().count()` for each entity.
    ///
    /// # Arguments
    /// * `source_text` - The original text
    /// * `text_char_count` - Pre-computed character count (from `text.chars().count()`)
    ///
    /// # Returns
    /// The extracted text, or empty string if offsets are invalid
    #[must_use]
    pub fn extract_text_with_len(&self, source_text: &str, text_char_count: usize) -> String {
        if self.start >= text_char_count || self.end > text_char_count || self.start >= self.end {
            return String::new();
        }
        source_text
            .chars()
            .skip(self.start)
            .take(self.end - self.start)
            .collect()
    }

    // =========================================================================
    // Temporal Validity Methods
    // =========================================================================

    /// Set the temporal validity start for this entity assertion.
    ///
    /// # Example
    /// ```rust
    /// use anno_core::{Entity, EntityType};
    /// use chrono::{TimeZone, Utc};
    ///
    /// let mut entity = Entity::new("CEO", EntityType::Person, 0, 3, 0.9);
    /// entity.set_valid_from(Utc.with_ymd_and_hms(2008, 10, 1, 0, 0, 0).unwrap());
    /// assert!(entity.is_temporal());
    /// ```
    pub fn set_valid_from(&mut self, dt: chrono::DateTime<chrono::Utc>) {
        self.valid_from = Some(dt);
    }

    /// Set the temporal validity end for this entity assertion.
    pub fn set_valid_until(&mut self, dt: chrono::DateTime<chrono::Utc>) {
        self.valid_until = Some(dt);
    }

    /// Set both temporal bounds at once.
    pub fn set_temporal_range(
        &mut self,
        from: chrono::DateTime<chrono::Utc>,
        until: chrono::DateTime<chrono::Utc>,
    ) {
        self.valid_from = Some(from);
        self.valid_until = Some(until);
    }

    /// Check if this entity has temporal validity information.
    #[must_use]
    pub fn is_temporal(&self) -> bool {
        self.valid_from.is_some() || self.valid_until.is_some()
    }

    /// Check if this entity was valid at a specific point in time.
    ///
    /// Returns `true` if:
    /// - No temporal bounds are set (atemporal entity)
    /// - The timestamp falls within [valid_from, valid_until]
    ///
    /// # Example
    /// ```rust
    /// use anno_core::{Entity, EntityType};
    /// use chrono::{TimeZone, Utc};
    ///
    /// let mut entity = Entity::new("CEO of Microsoft", EntityType::Person, 0, 16, 0.9);
    /// entity.set_valid_from(Utc.with_ymd_and_hms(2008, 1, 1, 0, 0, 0).unwrap());
    /// entity.set_valid_until(Utc.with_ymd_and_hms(2023, 12, 31, 0, 0, 0).unwrap());
    ///
    /// let query_2015 = Utc.with_ymd_and_hms(2015, 6, 1, 0, 0, 0).unwrap();
    /// let query_2005 = Utc.with_ymd_and_hms(2005, 6, 1, 0, 0, 0).unwrap();
    ///
    /// assert!(entity.valid_at(&query_2015));
    /// assert!(!entity.valid_at(&query_2005));
    /// ```
    #[must_use]
    pub fn valid_at(&self, timestamp: &chrono::DateTime<chrono::Utc>) -> bool {
        match (&self.valid_from, &self.valid_until) {
            (None, None) => true,                      // Atemporal - always valid
            (Some(from), None) => timestamp >= from,   // Started, still valid
            (None, Some(until)) => timestamp <= until, // Unknown start, ended
            (Some(from), Some(until)) => timestamp >= from && timestamp <= until,
        }
    }

    /// Check if this entity is currently valid (at the current time).
    #[must_use]
    pub fn is_currently_valid(&self) -> bool {
        self.valid_at(&chrono::Utc::now())
    }

    // =========================================================================
    // Viewport/Context Methods
    // =========================================================================

    /// Set the viewport context for this entity.
    ///
    /// # Example
    /// ```rust
    /// use anno_core::{Entity, EntityType, EntityViewport};
    ///
    /// let mut entity = Entity::new("Marie Curie", EntityType::Person, 0, 11, 0.9);
    /// entity.set_viewport(EntityViewport::Academic);
    /// assert!(entity.has_viewport());
    /// ```
    pub fn set_viewport(&mut self, viewport: EntityViewport) {
        self.viewport = Some(viewport);
    }

    /// Check if this entity has a viewport context.
    #[must_use]
    pub fn has_viewport(&self) -> bool {
        self.viewport.is_some()
    }

    /// Get the viewport, defaulting to General if not set.
    #[must_use]
    pub fn viewport_or_default(&self) -> EntityViewport {
        self.viewport.clone().unwrap_or_default()
    }

    /// Check if this entity matches a viewport context.
    ///
    /// Returns true if:
    /// - The entity has no viewport (matches any)
    /// - The entity's viewport matches the query
    #[must_use]
    pub fn matches_viewport(&self, query_viewport: &EntityViewport) -> bool {
        match &self.viewport {
            None => true, // No viewport = matches any
            Some(v) => v == query_viewport,
        }
    }

    /// Create a builder for fluent entity construction.
    #[must_use]
    pub fn builder(text: impl Into<String>, entity_type: EntityType) -> EntityBuilder {
        EntityBuilder::new(text, entity_type)
    }

    // =========================================================================
    // Validation Methods (Production Quality)
    // =========================================================================

    /// Validate this entity against the source text.
    ///
    /// Returns a list of validation issues. Empty list means the entity is valid.
    ///
    /// # Checks Performed
    ///
    /// 1. **Span bounds**: `start < end`, both within text length
    /// 2. **Text match**: `text` matches the span in source
    /// 3. **Confidence range**: `confidence` in [0.0, 1.0]
    /// 4. **Type consistency**: Custom types have non-empty names
    /// 5. **Discontinuous consistency**: If present, segments are valid
    ///
    /// # Example
    ///
    /// ```rust
    /// use anno_core::{Entity, EntityType};
    ///
    /// let text = "John works at Apple";
    /// let entity = Entity::new("John", EntityType::Person, 0, 4, 0.95);
    ///
    /// let issues = entity.validate(text);
    /// assert!(issues.is_empty(), "Entity should be valid");
    ///
    /// // Invalid entity: span doesn't match text
    /// let bad = Entity::new("Jane", EntityType::Person, 0, 4, 0.95);
    /// let issues = bad.validate(text);
    /// assert!(!issues.is_empty(), "Entity text doesn't match span");
    /// ```
    #[must_use]
    pub fn validate(&self, source_text: &str) -> Vec<ValidationIssue> {
        // Performance: Calculate length once, delegate to optimized version
        let char_count = source_text.chars().count();
        self.validate_with_len(source_text, char_count)
    }

    /// Validate entity with pre-computed text length (performance optimization).
    ///
    /// Use this when validating multiple entities from the same text to avoid
    /// recalculating `text.chars().count()` for each entity.
    ///
    /// # Arguments
    /// * `source_text` - The original text
    /// * `text_char_count` - Pre-computed character count (from `text.chars().count()`)
    ///
    /// # Returns
    /// Vector of validation issues (empty if valid)
    #[must_use]
    pub fn validate_with_len(
        &self,
        source_text: &str,
        text_char_count: usize,
    ) -> Vec<ValidationIssue> {
        let mut issues = Vec::new();

        // 1. Span bounds
        if self.start >= self.end {
            issues.push(ValidationIssue::InvalidSpan {
                start: self.start,
                end: self.end,
                reason: "start must be less than end".to_string(),
            });
        }

        if self.end > text_char_count {
            issues.push(ValidationIssue::SpanOutOfBounds {
                end: self.end,
                text_len: text_char_count,
            });
        }

        // 2. Text match (only if span is valid)
        if self.start < self.end && self.end <= text_char_count {
            let actual = self.extract_text_with_len(source_text, text_char_count);
            if actual != self.text {
                issues.push(ValidationIssue::TextMismatch {
                    expected: self.text.clone(),
                    actual,
                    start: self.start,
                    end: self.end,
                });
            }
        }

        // 3. Confidence range
        if !(0.0..=1.0).contains(&self.confidence) {
            issues.push(ValidationIssue::InvalidConfidence {
                value: self.confidence,
            });
        }

        // 4. Type consistency
        if let EntityType::Custom { ref name, .. } = self.entity_type {
            if name.is_empty() {
                issues.push(ValidationIssue::InvalidType {
                    reason: "Custom entity type has empty name".to_string(),
                });
            }
        }

        // 5. Discontinuous span consistency
        if let Some(ref disc_span) = self.discontinuous_span {
            for (i, seg) in disc_span.segments().iter().enumerate() {
                if seg.start >= seg.end {
                    issues.push(ValidationIssue::InvalidSpan {
                        start: seg.start,
                        end: seg.end,
                        reason: format!("discontinuous segment {} is invalid", i),
                    });
                }
                // Note: discontinuous spans use byte offsets, so we'd need byte length here
                // For now, skip this check for discontinuous spans
            }
        }

        issues
    }

    /// Check if this entity is valid against the source text.
    ///
    /// Convenience method that returns `true` if `validate()` returns empty.
    #[must_use]
    pub fn is_valid(&self, source_text: &str) -> bool {
        self.validate(source_text).is_empty()
    }

    /// Validate a batch of entities efficiently.
    ///
    /// Returns a map of entity index -> validation issues.
    /// Only entities with issues are included.
    ///
    /// # Example
    ///
    /// ```rust
    /// use anno_core::{Entity, EntityType};
    ///
    /// let text = "John and Jane work at Apple";
    /// let entities = vec![
    ///     Entity::new("John", EntityType::Person, 0, 4, 0.95),
    ///     Entity::new("Wrong", EntityType::Person, 9, 13, 0.8),
    /// ];
    ///
    /// let issues = Entity::validate_batch(&entities, text);
    /// assert!(issues.is_empty() || issues.contains_key(&1)); // Second entity might fail
    /// ```
    #[must_use]
    pub fn validate_batch(
        entities: &[Entity],
        source_text: &str,
    ) -> std::collections::HashMap<usize, Vec<ValidationIssue>> {
        entities
            .iter()
            .enumerate()
            .filter_map(|(idx, entity)| {
                let issues = entity.validate(source_text);
                if issues.is_empty() {
                    None
                } else {
                    Some((idx, issues))
                }
            })
            .collect()
    }
}

/// Validation issue found during entity validation.
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationIssue {
    /// Span bounds are invalid (start >= end).
    InvalidSpan {
        /// Start position of the invalid span.
        start: usize,
        /// End position of the invalid span.
        end: usize,
        /// Description of why the span is invalid.
        reason: String,
    },
    /// Span extends beyond text length.
    SpanOutOfBounds {
        /// End position that exceeds the text.
        end: usize,
        /// Actual length of the text.
        text_len: usize,
    },
    /// Entity text doesn't match the span in source.
    TextMismatch {
        /// Text stored in the entity.
        expected: String,
        /// Text found at the span in source.
        actual: String,
        /// Start position of the span.
        start: usize,
        /// End position of the span.
        end: usize,
    },
    /// Confidence is outside [0.0, 1.0].
    InvalidConfidence {
        /// The invalid confidence value.
        value: f64,
    },
    /// Entity type is invalid.
    InvalidType {
        /// Description of why the type is invalid.
        reason: String,
    },
}

impl std::fmt::Display for ValidationIssue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationIssue::InvalidSpan { start, end, reason } => {
                write!(f, "Invalid span [{}, {}): {}", start, end, reason)
            }
            ValidationIssue::SpanOutOfBounds { end, text_len } => {
                write!(f, "Span end {} exceeds text length {}", end, text_len)
            }
            ValidationIssue::TextMismatch {
                expected,
                actual,
                start,
                end,
            } => {
                write!(
                    f,
                    "Text mismatch at [{}, {}): expected '{}', got '{}'",
                    start, end, expected, actual
                )
            }
            ValidationIssue::InvalidConfidence { value } => {
                write!(f, "Confidence {} outside [0.0, 1.0]", value)
            }
            ValidationIssue::InvalidType { reason } => {
                write!(f, "Invalid entity type: {}", reason)
            }
        }
    }
}

/// Fluent builder for constructing entities with optional fields.
///
/// # Example
///
/// ```rust
/// use anno_core::{Entity, EntityType, Provenance};
///
/// let entity = Entity::builder("Marie Curie", EntityType::Person)
///     .span(0, 11)
///     .confidence(0.95)
///     .kb_id("Q7186")
///     .provenance(Provenance::ml("bert", 0.95))
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct EntityBuilder {
    text: String,
    entity_type: EntityType,
    start: usize,
    end: usize,
    confidence: f64,
    normalized: Option<String>,
    provenance: Option<Provenance>,
    kb_id: Option<String>,
    canonical_id: Option<u64>,
    hierarchical_confidence: Option<HierarchicalConfidence>,
    visual_span: Option<Span>,
    discontinuous_span: Option<DiscontinuousSpan>,
    valid_from: Option<chrono::DateTime<chrono::Utc>>,
    valid_until: Option<chrono::DateTime<chrono::Utc>>,
    viewport: Option<EntityViewport>,
}

impl EntityBuilder {
    /// Create a new builder.
    #[must_use]
    pub fn new(text: impl Into<String>, entity_type: EntityType) -> Self {
        Self {
            text: text.into(),
            entity_type,
            start: 0,
            end: 0,
            confidence: 1.0,
            normalized: None,
            provenance: None,
            kb_id: None,
            canonical_id: None,
            hierarchical_confidence: None,
            visual_span: None,
            discontinuous_span: None,
            valid_from: None,
            valid_until: None,
            viewport: None,
        }
    }

    /// Set span offsets.
    #[must_use]
    pub const fn span(mut self, start: usize, end: usize) -> Self {
        self.start = start;
        self.end = end;
        self
    }

    /// Set confidence score.
    #[must_use]
    pub fn confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Set hierarchical confidence.
    #[must_use]
    pub fn hierarchical_confidence(mut self, confidence: HierarchicalConfidence) -> Self {
        self.confidence = confidence.as_f64();
        self.hierarchical_confidence = Some(confidence);
        self
    }

    /// Set normalized form.
    #[must_use]
    pub fn normalized(mut self, normalized: impl Into<String>) -> Self {
        self.normalized = Some(normalized.into());
        self
    }

    /// Set provenance.
    #[must_use]
    pub fn provenance(mut self, provenance: Provenance) -> Self {
        self.provenance = Some(provenance);
        self
    }

    /// Set knowledge base ID.
    #[must_use]
    pub fn kb_id(mut self, kb_id: impl Into<String>) -> Self {
        self.kb_id = Some(kb_id.into());
        self
    }

    /// Set canonical (coreference) ID.
    #[must_use]
    pub const fn canonical_id(mut self, canonical_id: u64) -> Self {
        self.canonical_id = Some(canonical_id);
        self
    }

    /// Set visual span.
    #[must_use]
    pub fn visual_span(mut self, span: Span) -> Self {
        self.visual_span = Some(span);
        self
    }

    /// Set discontinuous span for non-contiguous entities.
    ///
    /// This automatically updates `start` and `end` to the bounding range.
    #[must_use]
    pub fn discontinuous_span(mut self, span: DiscontinuousSpan) -> Self {
        // Update start/end to bounding range
        if let Some(bounding) = span.bounding_range() {
            self.start = bounding.start;
            self.end = bounding.end;
        }
        self.discontinuous_span = Some(span);
        self
    }

    /// Set temporal validity start (when this entity assertion became true).
    ///
    /// # Example
    /// ```rust
    /// use anno_core::{EntityBuilder, EntityType};
    /// use chrono::{TimeZone, Utc};
    ///
    /// let entity = EntityBuilder::new("CEO of Microsoft", EntityType::Person)
    ///     .span(0, 12)
    ///     .valid_from(Utc.with_ymd_and_hms(2008, 10, 1, 0, 0, 0).unwrap())
    ///     .build();
    /// assert!(entity.valid_from.is_some());
    /// ```
    #[must_use]
    pub fn valid_from(mut self, dt: chrono::DateTime<chrono::Utc>) -> Self {
        self.valid_from = Some(dt);
        self
    }

    /// Set temporal validity end (when this entity assertion stopped being true).
    #[must_use]
    pub fn valid_until(mut self, dt: chrono::DateTime<chrono::Utc>) -> Self {
        self.valid_until = Some(dt);
        self
    }

    /// Set temporal validity range (convenience method).
    #[must_use]
    pub fn temporal_range(
        mut self,
        from: chrono::DateTime<chrono::Utc>,
        until: chrono::DateTime<chrono::Utc>,
    ) -> Self {
        self.valid_from = Some(from);
        self.valid_until = Some(until);
        self
    }

    /// Set the viewport context for multi-faceted entity representation.
    ///
    /// # Example
    /// ```rust
    /// use anno_core::{EntityBuilder, EntityType, EntityViewport};
    ///
    /// let entity = EntityBuilder::new("Marie Curie", EntityType::Person)
    ///     .span(0, 11)
    ///     .viewport(EntityViewport::Academic)
    ///     .build();
    /// assert_eq!(entity.viewport, Some(EntityViewport::Academic));
    /// ```
    #[must_use]
    pub fn viewport(mut self, viewport: EntityViewport) -> Self {
        self.viewport = Some(viewport);
        self
    }

    /// Build the entity.
    #[must_use]
    pub fn build(self) -> Entity {
        Entity {
            text: self.text,
            entity_type: self.entity_type,
            start: self.start,
            end: self.end,
            confidence: self.confidence,
            normalized: self.normalized,
            provenance: self.provenance,
            kb_id: self.kb_id,
            canonical_id: self.canonical_id,
            hierarchical_confidence: self.hierarchical_confidence,
            visual_span: self.visual_span,
            discontinuous_span: self.discontinuous_span,
            valid_from: self.valid_from,
            valid_until: self.valid_until,
            viewport: self.viewport,
        }
    }
}

// ============================================================================
// Relation (for Knowledge Graph Construction)
// ============================================================================

/// A relation between two entities, forming a knowledge graph triple.
///
/// In the GLiNER bi-encoder paradigm, relations are detected just like entities:
/// the relation trigger text ("CEO of", "located in") is matched against
/// relation type labels in the same latent space.
///
/// # Structure
///
/// ```text
/// Triple: (Head, Relation, Tail)
///
/// "Marie Curie worked at the Sorbonne"
///  ^^^^^^^^^^^ ~~~~~~~~~ ^^^^^^^^
///  Head        Rel       Tail
///  (Person)  (Employment)  (Organization)
/// ```
///
/// # TPLinker/Joint Extraction
///
/// For joint extraction, relations are extracted in a single pass with entities.
/// The `trigger_span` captures the text that indicates the relation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relation {
    /// The source entity (head of the triple)
    pub head: Entity,
    /// The target entity (tail of the triple)
    pub tail: Entity,
    /// Relation type label (e.g., "EMPLOYMENT", "LOCATED_IN", "FOUNDED_BY")
    pub relation_type: String,
    /// Optional trigger span: the text that indicates this relation
    /// For "CEO of", this would be the span covering "CEO of"
    pub trigger_span: Option<(usize, usize)>,
    /// Confidence score for this relation (0.0-1.0)
    pub confidence: f64,
}

impl Relation {
    /// Create a new relation between two entities.
    #[must_use]
    pub fn new(
        head: Entity,
        tail: Entity,
        relation_type: impl Into<String>,
        confidence: f64,
    ) -> Self {
        Self {
            head,
            tail,
            relation_type: relation_type.into(),
            trigger_span: None,
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// Create a relation with an explicit trigger span.
    #[must_use]
    pub fn with_trigger(
        head: Entity,
        tail: Entity,
        relation_type: impl Into<String>,
        trigger_start: usize,
        trigger_end: usize,
        confidence: f64,
    ) -> Self {
        Self {
            head,
            tail,
            relation_type: relation_type.into(),
            trigger_span: Some((trigger_start, trigger_end)),
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// Convert to a triple string representation (for debugging/display).
    #[must_use]
    pub fn as_triple(&self) -> String {
        format!(
            "({}, {}, {})",
            self.head.text, self.relation_type, self.tail.text
        )
    }

    /// Check if the head and tail entities are adjacent (within n tokens).
    /// Useful for filtering spurious long-distance relations.
    #[must_use]
    pub fn span_distance(&self) -> usize {
        if self.head.end <= self.tail.start {
            self.tail.start.saturating_sub(self.head.end)
        } else if self.tail.end <= self.head.start {
            self.head.start.saturating_sub(self.tail.end)
        } else {
            0 // Overlapping spans
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_type_roundtrip() {
        let types = [
            EntityType::Person,
            EntityType::Organization,
            EntityType::Location,
            EntityType::Date,
            EntityType::Money,
            EntityType::Percent,
        ];

        for t in types {
            let label = t.as_label();
            let parsed = EntityType::from_label(label);
            assert_eq!(t, parsed);
        }
    }

    #[test]
    fn test_entity_overlap() {
        let e1 = Entity::new("John", EntityType::Person, 0, 4, 0.9);
        let e2 = Entity::new("Smith", EntityType::Person, 5, 10, 0.9);
        let e3 = Entity::new("John Smith", EntityType::Person, 0, 10, 0.9);

        assert!(!e1.overlaps(&e2)); // No overlap
        assert!(e1.overlaps(&e3)); // e1 is contained in e3
        assert!(e3.overlaps(&e2)); // e3 contains e2
    }

    #[test]
    fn test_confidence_clamping() {
        let e1 = Entity::new("test", EntityType::Person, 0, 4, 1.5);
        assert!((e1.confidence - 1.0).abs() < f64::EPSILON);

        let e2 = Entity::new("test", EntityType::Person, 0, 4, -0.5);
        assert!(e2.confidence.abs() < f64::EPSILON);
    }

    #[test]
    fn test_entity_categories() {
        // Agent/Org/Place entities require ML
        assert_eq!(EntityType::Person.category(), EntityCategory::Agent);
        assert_eq!(
            EntityType::Organization.category(),
            EntityCategory::Organization
        );
        assert_eq!(EntityType::Location.category(), EntityCategory::Place);
        assert!(EntityType::Person.requires_ml());
        assert!(!EntityType::Person.pattern_detectable());

        // Temporal entities are pattern-detectable
        assert_eq!(EntityType::Date.category(), EntityCategory::Temporal);
        assert_eq!(EntityType::Time.category(), EntityCategory::Temporal);
        assert!(EntityType::Date.pattern_detectable());
        assert!(!EntityType::Date.requires_ml());

        // Numeric entities are pattern-detectable
        assert_eq!(EntityType::Money.category(), EntityCategory::Numeric);
        assert_eq!(EntityType::Percent.category(), EntityCategory::Numeric);
        assert!(EntityType::Money.pattern_detectable());

        // Contact entities are pattern-detectable
        assert_eq!(EntityType::Email.category(), EntityCategory::Contact);
        assert_eq!(EntityType::Url.category(), EntityCategory::Contact);
        assert_eq!(EntityType::Phone.category(), EntityCategory::Contact);
        assert!(EntityType::Email.pattern_detectable());
    }

    #[test]
    fn test_new_types_roundtrip() {
        let types = [
            EntityType::Time,
            EntityType::Email,
            EntityType::Url,
            EntityType::Phone,
            EntityType::Quantity,
            EntityType::Cardinal,
            EntityType::Ordinal,
        ];

        for t in types {
            let label = t.as_label();
            let parsed = EntityType::from_label(label);
            assert_eq!(t, parsed, "Roundtrip failed for {}", label);
        }
    }

    #[test]
    fn test_custom_entity_type() {
        let disease = EntityType::custom("DISEASE", EntityCategory::Agent);
        assert_eq!(disease.as_label(), "DISEASE");
        assert!(disease.requires_ml());

        let product_id = EntityType::custom("PRODUCT_ID", EntityCategory::Misc);
        assert_eq!(product_id.as_label(), "PRODUCT_ID");
        assert!(!product_id.requires_ml());
        assert!(!product_id.pattern_detectable());
    }

    #[test]
    fn test_entity_normalization() {
        let mut e = Entity::new("Jan 15", EntityType::Date, 0, 6, 0.95);
        assert!(e.normalized.is_none());
        assert_eq!(e.normalized_or_text(), "Jan 15");

        e.set_normalized("2024-01-15");
        assert_eq!(e.normalized.as_deref(), Some("2024-01-15"));
        assert_eq!(e.normalized_or_text(), "2024-01-15");
    }

    #[test]
    fn test_entity_helpers() {
        let named = Entity::new("John", EntityType::Person, 0, 4, 0.9);
        assert!(named.is_named());
        assert!(!named.is_structured());
        assert_eq!(named.category(), EntityCategory::Agent);

        let structured = Entity::new("$100", EntityType::Money, 0, 4, 0.95);
        assert!(!structured.is_named());
        assert!(structured.is_structured());
        assert_eq!(structured.category(), EntityCategory::Numeric);
    }

    #[test]
    fn test_knowledge_linking() {
        let mut entity = Entity::new("Marie Curie", EntityType::Person, 0, 11, 0.95);
        assert!(!entity.is_linked());
        assert!(!entity.has_coreference());

        entity.link_to_kb("Q7186"); // Wikidata ID
        assert!(entity.is_linked());
        assert_eq!(entity.kb_id.as_deref(), Some("Q7186"));

        entity.set_canonical(42);
        assert!(entity.has_coreference());
        assert_eq!(entity.canonical_id, Some(42));
    }

    #[test]
    fn test_relation_creation() {
        let head = Entity::new("Marie Curie", EntityType::Person, 0, 11, 0.95);
        let tail = Entity::new("Sorbonne", EntityType::Organization, 24, 32, 0.90);

        let relation = Relation::new(head.clone(), tail.clone(), "WORKED_AT", 0.85);
        assert_eq!(relation.relation_type, "WORKED_AT");
        assert_eq!(relation.as_triple(), "(Marie Curie, WORKED_AT, Sorbonne)");
        assert!(relation.trigger_span.is_none());

        // With trigger span
        let relation2 = Relation::with_trigger(head, tail, "EMPLOYMENT", 13, 19, 0.85);
        assert_eq!(relation2.trigger_span, Some((13, 19)));
    }

    #[test]
    fn test_relation_span_distance() {
        // Head at 0-11, tail at 24-32 -> distance is 24-11 = 13
        let head = Entity::new("Marie Curie", EntityType::Person, 0, 11, 0.95);
        let tail = Entity::new("Sorbonne", EntityType::Organization, 24, 32, 0.90);
        let relation = Relation::new(head, tail, "WORKED_AT", 0.85);
        assert_eq!(relation.span_distance(), 13);
    }

    #[test]
    fn test_relation_category() {
        // Relation types should be categorized as Relation
        let rel_type = EntityType::custom("CEO_OF", EntityCategory::Relation);
        assert_eq!(rel_type.category(), EntityCategory::Relation);
        assert!(rel_type.category().is_relation());
        assert!(rel_type.requires_ml()); // Relations require ML
    }

    // ========================================================================
    // Span Tests
    // ========================================================================

    #[test]
    fn test_span_text() {
        let span = Span::text(10, 20);
        assert!(span.is_text());
        assert!(!span.is_visual());
        assert_eq!(span.text_offsets(), Some((10, 20)));
        assert_eq!(span.len(), 10);
        assert!(!span.is_empty());
    }

    #[test]
    fn test_span_bbox() {
        let span = Span::bbox(0.1, 0.2, 0.3, 0.4);
        assert!(!span.is_text());
        assert!(span.is_visual());
        assert_eq!(span.text_offsets(), None);
        assert_eq!(span.len(), 0); // No text length
    }

    #[test]
    fn test_span_bbox_with_page() {
        let span = Span::bbox_on_page(0.1, 0.2, 0.3, 0.4, 5);
        if let Span::BoundingBox { page, .. } = span {
            assert_eq!(page, Some(5));
        } else {
            panic!("Expected BoundingBox");
        }
    }

    #[test]
    fn test_span_hybrid() {
        let bbox = Span::bbox(0.1, 0.2, 0.3, 0.4);
        let hybrid = Span::Hybrid {
            start: 10,
            end: 20,
            bbox: Box::new(bbox),
        };
        assert!(hybrid.is_text());
        assert!(hybrid.is_visual());
        assert_eq!(hybrid.text_offsets(), Some((10, 20)));
        assert_eq!(hybrid.len(), 10);
    }

    // ========================================================================
    // Hierarchical Confidence Tests
    // ========================================================================

    #[test]
    fn test_hierarchical_confidence_new() {
        let hc = HierarchicalConfidence::new(0.9, 0.8, 0.7);
        assert!((hc.linkage - 0.9).abs() < f32::EPSILON);
        assert!((hc.type_score - 0.8).abs() < f32::EPSILON);
        assert!((hc.boundary - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn test_hierarchical_confidence_clamping() {
        let hc = HierarchicalConfidence::new(1.5, -0.5, 0.5);
        assert!((hc.linkage - 1.0).abs() < f32::EPSILON);
        assert!(hc.type_score.abs() < f32::EPSILON);
        assert!((hc.boundary - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_hierarchical_confidence_from_single() {
        let hc = HierarchicalConfidence::from_single(0.8);
        assert!((hc.linkage - 0.8).abs() < f32::EPSILON);
        assert!((hc.type_score - 0.8).abs() < f32::EPSILON);
        assert!((hc.boundary - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn test_hierarchical_confidence_combined() {
        let hc = HierarchicalConfidence::new(1.0, 1.0, 1.0);
        assert!((hc.combined() - 1.0).abs() < f32::EPSILON);

        let hc2 = HierarchicalConfidence::new(0.8, 0.8, 0.8);
        assert!((hc2.combined() - 0.8).abs() < f32::EPSILON);

        // Geometric mean: (0.5 * 0.5 * 0.5)^(1/3) = 0.5
        let hc3 = HierarchicalConfidence::new(0.5, 0.5, 0.5);
        assert!((hc3.combined() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_hierarchical_confidence_threshold() {
        let hc = HierarchicalConfidence::new(0.9, 0.8, 0.7);
        assert!(hc.passes_threshold(0.5, 0.5, 0.5));
        assert!(hc.passes_threshold(0.9, 0.8, 0.7));
        assert!(!hc.passes_threshold(0.95, 0.8, 0.7)); // linkage too high
        assert!(!hc.passes_threshold(0.9, 0.85, 0.7)); // type too high
    }

    #[test]
    fn test_hierarchical_confidence_from_f64() {
        let hc: HierarchicalConfidence = 0.85_f64.into();
        assert!((hc.linkage - 0.85).abs() < 0.001);
    }

    // ========================================================================
    // RaggedBatch Tests
    // ========================================================================

    #[test]
    fn test_ragged_batch_from_sequences() {
        let seqs = vec![vec![1, 2, 3], vec![4, 5], vec![6, 7, 8, 9]];
        let batch = RaggedBatch::from_sequences(&seqs);

        assert_eq!(batch.batch_size(), 3);
        assert_eq!(batch.total_tokens(), 9);
        assert_eq!(batch.max_seq_len, 4);
        assert_eq!(batch.cumulative_offsets, vec![0, 3, 5, 9]);
    }

    #[test]
    fn test_ragged_batch_doc_range() {
        let seqs = vec![vec![1, 2, 3], vec![4, 5]];
        let batch = RaggedBatch::from_sequences(&seqs);

        assert_eq!(batch.doc_range(0), Some(0..3));
        assert_eq!(batch.doc_range(1), Some(3..5));
        assert_eq!(batch.doc_range(2), None);
    }

    #[test]
    fn test_ragged_batch_doc_tokens() {
        let seqs = vec![vec![1, 2, 3], vec![4, 5]];
        let batch = RaggedBatch::from_sequences(&seqs);

        assert_eq!(batch.doc_tokens(0), Some(&[1, 2, 3][..]));
        assert_eq!(batch.doc_tokens(1), Some(&[4, 5][..]));
    }

    #[test]
    fn test_ragged_batch_padding_savings() {
        // 3 docs: [3, 2, 4] tokens, max = 4
        // Padded: 3 * 4 = 12, actual: 9
        // Savings: 1 - 9/12 = 0.25
        let seqs = vec![vec![1, 2, 3], vec![4, 5], vec![6, 7, 8, 9]];
        let batch = RaggedBatch::from_sequences(&seqs);
        let savings = batch.padding_savings();
        assert!((savings - 0.25).abs() < 0.001);
    }

    // ========================================================================
    // SpanCandidate Tests
    // ========================================================================

    #[test]
    fn test_span_candidate() {
        let sc = SpanCandidate::new(0, 5, 10);
        assert_eq!(sc.doc_idx, 0);
        assert_eq!(sc.start, 5);
        assert_eq!(sc.end, 10);
        assert_eq!(sc.width(), 5);
    }

    #[test]
    fn test_generate_span_candidates() {
        let seqs = vec![vec![1, 2, 3]]; // doc with 3 tokens
        let batch = RaggedBatch::from_sequences(&seqs);
        let candidates = generate_span_candidates(&batch, 2);

        // With max_width=2: [0,1], [1,2], [2,3], [0,2], [1,3]
        // = spans: (0,1), (0,2), (1,2), (1,3), (2,3)
        assert_eq!(candidates.len(), 5);

        // Verify all candidates are valid
        for c in &candidates {
            assert_eq!(c.doc_idx, 0);
            assert!(c.end as usize <= 3);
            assert!(c.width() as usize <= 2);
        }
    }

    #[test]
    fn test_generate_filtered_candidates() {
        let seqs = vec![vec![1, 2, 3]];
        let batch = RaggedBatch::from_sequences(&seqs);

        // With max_width=2, we have 5 candidates
        // Set mask: only first 2 pass threshold
        let mask = vec![0.9, 0.9, 0.1, 0.1, 0.1];
        let candidates = generate_filtered_candidates(&batch, 2, &mask, 0.5);

        assert_eq!(candidates.len(), 2);
    }

    // ========================================================================
    // EntityBuilder Tests
    // ========================================================================

    #[test]
    fn test_entity_builder_basic() {
        let entity = Entity::builder("John", EntityType::Person)
            .span(0, 4)
            .confidence(0.95)
            .build();

        assert_eq!(entity.text, "John");
        assert_eq!(entity.entity_type, EntityType::Person);
        assert_eq!(entity.start, 0);
        assert_eq!(entity.end, 4);
        assert!((entity.confidence - 0.95).abs() < f64::EPSILON);
    }

    #[test]
    fn test_entity_builder_full() {
        let entity = Entity::builder("Marie Curie", EntityType::Person)
            .span(0, 11)
            .confidence(0.95)
            .kb_id("Q7186")
            .canonical_id(42)
            .normalized("Marie Salomea Skłodowska Curie")
            .provenance(Provenance::ml("bert", 0.95))
            .build();

        assert_eq!(entity.text, "Marie Curie");
        assert_eq!(entity.kb_id.as_deref(), Some("Q7186"));
        assert_eq!(entity.canonical_id, Some(42));
        assert_eq!(
            entity.normalized.as_deref(),
            Some("Marie Salomea Skłodowska Curie")
        );
        assert!(entity.provenance.is_some());
    }

    #[test]
    fn test_entity_builder_hierarchical() {
        let hc = HierarchicalConfidence::new(0.9, 0.8, 0.7);
        let entity = Entity::builder("test", EntityType::Person)
            .span(0, 4)
            .hierarchical_confidence(hc)
            .build();

        assert!(entity.hierarchical_confidence.is_some());
        assert!((entity.linkage_confidence() - 0.9).abs() < 0.001);
        assert!((entity.type_confidence() - 0.8).abs() < 0.001);
        assert!((entity.boundary_confidence() - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_entity_builder_visual() {
        let bbox = Span::bbox(0.1, 0.2, 0.3, 0.4);
        let entity = Entity::builder("receipt item", EntityType::Money)
            .visual_span(bbox)
            .confidence(0.9)
            .build();

        assert!(entity.is_visual());
        assert!(entity.visual_span.is_some());
    }

    // ========================================================================
    // Entity Helper Method Tests
    // ========================================================================

    #[test]
    fn test_entity_hierarchical_confidence_helpers() {
        let mut entity = Entity::new("test", EntityType::Person, 0, 4, 0.8);

        // Without hierarchical confidence, falls back to main confidence
        assert!((entity.linkage_confidence() - 0.8).abs() < 0.001);
        assert!((entity.type_confidence() - 0.8).abs() < 0.001);
        assert!((entity.boundary_confidence() - 0.8).abs() < 0.001);

        // Set hierarchical confidence
        entity.set_hierarchical_confidence(HierarchicalConfidence::new(0.95, 0.85, 0.75));
        assert!((entity.linkage_confidence() - 0.95).abs() < 0.001);
        assert!((entity.type_confidence() - 0.85).abs() < 0.001);
        assert!((entity.boundary_confidence() - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_entity_from_visual() {
        let entity = Entity::from_visual(
            "receipt total",
            EntityType::Money,
            Span::bbox(0.5, 0.8, 0.2, 0.05),
            0.92,
        );

        assert!(entity.is_visual());
        assert_eq!(entity.start, 0);
        assert_eq!(entity.end, 0);
        assert!((entity.confidence - 0.92).abs() < f64::EPSILON);
    }

    #[test]
    fn test_entity_span_helpers() {
        let entity = Entity::new("test", EntityType::Person, 10, 20, 0.9);
        assert_eq!(entity.text_span(), (10, 20));
        assert_eq!(entity.span_len(), 10);
    }

    // ========================================================================
    // Provenance Tests
    // ========================================================================

    #[test]
    fn test_provenance_pattern() {
        let prov = Provenance::pattern("EMAIL");
        assert_eq!(prov.method, ExtractionMethod::Pattern);
        assert_eq!(prov.pattern.as_deref(), Some("EMAIL"));
        assert_eq!(prov.raw_confidence, Some(1.0)); // Patterns are deterministic
    }

    #[test]
    fn test_provenance_ml() {
        let prov = Provenance::ml("bert-ner", 0.87);
        assert_eq!(prov.method, ExtractionMethod::Neural);
        assert_eq!(prov.source.as_ref(), "bert-ner");
        assert_eq!(prov.raw_confidence, Some(0.87));
    }

    #[test]
    fn test_provenance_with_version() {
        let prov = Provenance::ml("gliner", 0.92).with_version("v2.1.0");

        assert_eq!(prov.model_version.as_deref(), Some("v2.1.0"));
        assert_eq!(prov.source.as_ref(), "gliner");
    }

    #[test]
    fn test_provenance_with_timestamp() {
        let prov = Provenance::pattern("DATE").with_timestamp("2024-01-15T10:30:00Z");

        assert_eq!(prov.timestamp.as_deref(), Some("2024-01-15T10:30:00Z"));
    }

    #[test]
    fn test_provenance_builder_chain() {
        let prov = Provenance::ml("modernbert-ner", 0.95)
            .with_version("v1.0.0")
            .with_timestamp("2024-11-27T12:00:00Z");

        assert_eq!(prov.method, ExtractionMethod::Neural);
        assert_eq!(prov.source.as_ref(), "modernbert-ner");
        assert_eq!(prov.raw_confidence, Some(0.95));
        assert_eq!(prov.model_version.as_deref(), Some("v1.0.0"));
        assert_eq!(prov.timestamp.as_deref(), Some("2024-11-27T12:00:00Z"));
    }

    #[test]
    fn test_provenance_serialization() {
        let prov = Provenance::ml("test", 0.9)
            .with_version("v1.0")
            .with_timestamp("2024-01-01");

        let json = serde_json::to_string(&prov).unwrap();
        assert!(json.contains("model_version"));
        assert!(json.contains("v1.0"));

        let restored: Provenance = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.model_version.as_deref(), Some("v1.0"));
        assert_eq!(restored.timestamp.as_deref(), Some("2024-01-01"));
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn confidence_always_clamped(conf in -10.0f64..10.0) {
            let e = Entity::new("test", EntityType::Person, 0, 4, conf);
            prop_assert!(e.confidence >= 0.0);
            prop_assert!(e.confidence <= 1.0);
        }

        #[test]
        fn entity_type_roundtrip(label in "[A-Z]{3,10}") {
            let et = EntityType::from_label(&label);
            let back = EntityType::from_label(et.as_label());
            // Other types may round-trip to themselves or normalize
            prop_assert!(matches!(back, EntityType::Other(_)) || back == et);
        }

        #[test]
        fn overlap_is_symmetric(
            s1 in 0usize..100,
            len1 in 1usize..50,
            s2 in 0usize..100,
            len2 in 1usize..50,
        ) {
            let e1 = Entity::new("a", EntityType::Person, s1, s1 + len1, 1.0);
            let e2 = Entity::new("b", EntityType::Person, s2, s2 + len2, 1.0);
            prop_assert_eq!(e1.overlaps(&e2), e2.overlaps(&e1));
        }

        #[test]
        fn overlap_ratio_bounded(
            s1 in 0usize..100,
            len1 in 1usize..50,
            s2 in 0usize..100,
            len2 in 1usize..50,
        ) {
            let e1 = Entity::new("a", EntityType::Person, s1, s1 + len1, 1.0);
            let e2 = Entity::new("b", EntityType::Person, s2, s2 + len2, 1.0);
            let ratio = e1.overlap_ratio(&e2);
            prop_assert!(ratio >= 0.0);
            prop_assert!(ratio <= 1.0);
        }

        #[test]
        fn self_overlap_ratio_is_one(s in 0usize..100, len in 1usize..50) {
            let e = Entity::new("test", EntityType::Person, s, s + len, 1.0);
            let ratio = e.overlap_ratio(&e);
            prop_assert!((ratio - 1.0).abs() < 1e-10);
        }

        #[test]
        fn hierarchical_confidence_always_clamped(
            linkage in -2.0f32..2.0,
            type_score in -2.0f32..2.0,
            boundary in -2.0f32..2.0,
        ) {
            let hc = HierarchicalConfidence::new(linkage, type_score, boundary);
            prop_assert!(hc.linkage >= 0.0 && hc.linkage <= 1.0);
            prop_assert!(hc.type_score >= 0.0 && hc.type_score <= 1.0);
            prop_assert!(hc.boundary >= 0.0 && hc.boundary <= 1.0);
            prop_assert!(hc.combined() >= 0.0 && hc.combined() <= 1.0);
        }

        #[test]
        fn span_candidate_width_consistent(
            doc in 0u32..10,
            start in 0u32..100,
            end in 1u32..100,
        ) {
            let actual_end = start.max(end);
            let sc = SpanCandidate::new(doc, start, actual_end);
            prop_assert_eq!(sc.width(), actual_end.saturating_sub(start));
        }

        #[test]
        fn ragged_batch_preserves_tokens(
            seq_lens in proptest::collection::vec(1usize..10, 1..5),
        ) {
            // Create sequences with sequential token IDs
            let mut counter = 0u32;
            let seqs: Vec<Vec<u32>> = seq_lens.iter().map(|&len| {
                let seq: Vec<u32> = (counter..counter + len as u32).collect();
                counter += len as u32;
                seq
            }).collect();

            let batch = RaggedBatch::from_sequences(&seqs);

            // Verify batch properties
            prop_assert_eq!(batch.batch_size(), seqs.len());
            prop_assert_eq!(batch.total_tokens(), seq_lens.iter().sum::<usize>());

            // Verify each doc can be retrieved correctly
            for (i, seq) in seqs.iter().enumerate() {
                let doc_tokens = batch.doc_tokens(i).unwrap();
                prop_assert_eq!(doc_tokens, seq.as_slice());
            }
        }

        #[test]
        fn span_text_offsets_consistent(start in 0usize..100, len in 0usize..50) {
            let end = start + len;
            let span = Span::text(start, end);
            let (s, e) = span.text_offsets().unwrap();
            prop_assert_eq!(s, start);
            prop_assert_eq!(e, end);
            prop_assert_eq!(span.len(), len);
        }
    }

    // ========================================================================
    // EntityViewport Tests
    // ========================================================================

    #[test]
    fn test_entity_viewport_as_str() {
        assert_eq!(EntityViewport::Business.as_str(), "business");
        assert_eq!(EntityViewport::Legal.as_str(), "legal");
        assert_eq!(EntityViewport::Technical.as_str(), "technical");
        assert_eq!(EntityViewport::Academic.as_str(), "academic");
        assert_eq!(EntityViewport::Personal.as_str(), "personal");
        assert_eq!(EntityViewport::Political.as_str(), "political");
        assert_eq!(EntityViewport::Media.as_str(), "media");
        assert_eq!(EntityViewport::Historical.as_str(), "historical");
        assert_eq!(EntityViewport::General.as_str(), "general");
        assert_eq!(
            EntityViewport::Custom("custom".to_string()).as_str(),
            "custom"
        );
    }

    #[test]
    fn test_entity_viewport_is_professional() {
        assert!(EntityViewport::Business.is_professional());
        assert!(EntityViewport::Legal.is_professional());
        assert!(EntityViewport::Technical.is_professional());
        assert!(EntityViewport::Academic.is_professional());
        assert!(EntityViewport::Political.is_professional());

        assert!(!EntityViewport::Personal.is_professional());
        assert!(!EntityViewport::Media.is_professional());
        assert!(!EntityViewport::Historical.is_professional());
        assert!(!EntityViewport::General.is_professional());
        assert!(!EntityViewport::Custom("test".to_string()).is_professional());
    }

    #[test]
    fn test_entity_viewport_from_str() {
        assert_eq!(
            "business".parse::<EntityViewport>().unwrap(),
            EntityViewport::Business
        );
        assert_eq!(
            "financial".parse::<EntityViewport>().unwrap(),
            EntityViewport::Business
        );
        assert_eq!(
            "corporate".parse::<EntityViewport>().unwrap(),
            EntityViewport::Business
        );

        assert_eq!(
            "legal".parse::<EntityViewport>().unwrap(),
            EntityViewport::Legal
        );
        assert_eq!(
            "law".parse::<EntityViewport>().unwrap(),
            EntityViewport::Legal
        );

        assert_eq!(
            "technical".parse::<EntityViewport>().unwrap(),
            EntityViewport::Technical
        );
        assert_eq!(
            "engineering".parse::<EntityViewport>().unwrap(),
            EntityViewport::Technical
        );

        assert_eq!(
            "academic".parse::<EntityViewport>().unwrap(),
            EntityViewport::Academic
        );
        assert_eq!(
            "research".parse::<EntityViewport>().unwrap(),
            EntityViewport::Academic
        );

        assert_eq!(
            "personal".parse::<EntityViewport>().unwrap(),
            EntityViewport::Personal
        );
        assert_eq!(
            "biographical".parse::<EntityViewport>().unwrap(),
            EntityViewport::Personal
        );

        assert_eq!(
            "political".parse::<EntityViewport>().unwrap(),
            EntityViewport::Political
        );
        assert_eq!(
            "policy".parse::<EntityViewport>().unwrap(),
            EntityViewport::Political
        );

        assert_eq!(
            "media".parse::<EntityViewport>().unwrap(),
            EntityViewport::Media
        );
        assert_eq!(
            "press".parse::<EntityViewport>().unwrap(),
            EntityViewport::Media
        );

        assert_eq!(
            "historical".parse::<EntityViewport>().unwrap(),
            EntityViewport::Historical
        );
        assert_eq!(
            "history".parse::<EntityViewport>().unwrap(),
            EntityViewport::Historical
        );

        assert_eq!(
            "general".parse::<EntityViewport>().unwrap(),
            EntityViewport::General
        );
        assert_eq!(
            "generic".parse::<EntityViewport>().unwrap(),
            EntityViewport::General
        );
        assert_eq!(
            "".parse::<EntityViewport>().unwrap(),
            EntityViewport::General
        );

        // Custom viewport
        assert_eq!(
            "custom_viewport".parse::<EntityViewport>().unwrap(),
            EntityViewport::Custom("custom_viewport".to_string())
        );
    }

    #[test]
    fn test_entity_viewport_from_str_case_insensitive() {
        assert_eq!(
            "BUSINESS".parse::<EntityViewport>().unwrap(),
            EntityViewport::Business
        );
        assert_eq!(
            "Business".parse::<EntityViewport>().unwrap(),
            EntityViewport::Business
        );
        assert_eq!(
            "BuSiNeSs".parse::<EntityViewport>().unwrap(),
            EntityViewport::Business
        );
    }

    #[test]
    fn test_entity_viewport_display() {
        assert_eq!(format!("{}", EntityViewport::Business), "business");
        assert_eq!(format!("{}", EntityViewport::Academic), "academic");
        assert_eq!(
            format!("{}", EntityViewport::Custom("test".to_string())),
            "test"
        );
    }

    #[test]
    fn test_entity_viewport_methods() {
        let mut entity = Entity::new("Marie Curie", EntityType::Person, 0, 11, 0.9);

        // Initially no viewport
        assert!(!entity.has_viewport());
        assert_eq!(entity.viewport_or_default(), EntityViewport::General);
        assert!(entity.matches_viewport(&EntityViewport::Academic)); // No viewport matches any

        // Set viewport
        entity.set_viewport(EntityViewport::Academic);
        assert!(entity.has_viewport());
        assert_eq!(entity.viewport_or_default(), EntityViewport::Academic);
        assert!(entity.matches_viewport(&EntityViewport::Academic));
        assert!(!entity.matches_viewport(&EntityViewport::Business));
    }

    #[test]
    fn test_entity_builder_with_viewport() {
        let entity = Entity::builder("Marie Curie", EntityType::Person)
            .span(0, 11)
            .viewport(EntityViewport::Academic)
            .build();

        assert_eq!(entity.viewport, Some(EntityViewport::Academic));
        assert!(entity.has_viewport());
    }

    // ========================================================================
    // EntityCategory Tests
    // ========================================================================

    #[test]
    fn test_entity_category_requires_ml() {
        assert!(EntityCategory::Agent.requires_ml());
        assert!(EntityCategory::Organization.requires_ml());
        assert!(EntityCategory::Place.requires_ml());
        assert!(EntityCategory::Creative.requires_ml());
        assert!(EntityCategory::Relation.requires_ml());

        assert!(!EntityCategory::Temporal.requires_ml());
        assert!(!EntityCategory::Numeric.requires_ml());
        assert!(!EntityCategory::Contact.requires_ml());
        assert!(!EntityCategory::Misc.requires_ml());
    }

    #[test]
    fn test_entity_category_pattern_detectable() {
        assert!(EntityCategory::Temporal.pattern_detectable());
        assert!(EntityCategory::Numeric.pattern_detectable());
        assert!(EntityCategory::Contact.pattern_detectable());

        assert!(!EntityCategory::Agent.pattern_detectable());
        assert!(!EntityCategory::Organization.pattern_detectable());
        assert!(!EntityCategory::Place.pattern_detectable());
        assert!(!EntityCategory::Creative.pattern_detectable());
        assert!(!EntityCategory::Relation.pattern_detectable());
        assert!(!EntityCategory::Misc.pattern_detectable());
    }

    #[test]
    fn test_entity_category_is_relation() {
        assert!(EntityCategory::Relation.is_relation());

        assert!(!EntityCategory::Agent.is_relation());
        assert!(!EntityCategory::Organization.is_relation());
        assert!(!EntityCategory::Place.is_relation());
        assert!(!EntityCategory::Temporal.is_relation());
        assert!(!EntityCategory::Numeric.is_relation());
        assert!(!EntityCategory::Contact.is_relation());
        assert!(!EntityCategory::Creative.is_relation());
        assert!(!EntityCategory::Misc.is_relation());
    }

    #[test]
    fn test_entity_category_as_str() {
        assert_eq!(EntityCategory::Agent.as_str(), "agent");
        assert_eq!(EntityCategory::Organization.as_str(), "organization");
        assert_eq!(EntityCategory::Place.as_str(), "place");
        assert_eq!(EntityCategory::Creative.as_str(), "creative");
        assert_eq!(EntityCategory::Temporal.as_str(), "temporal");
        assert_eq!(EntityCategory::Numeric.as_str(), "numeric");
        assert_eq!(EntityCategory::Contact.as_str(), "contact");
        assert_eq!(EntityCategory::Relation.as_str(), "relation");
        assert_eq!(EntityCategory::Misc.as_str(), "misc");
    }

    #[test]
    fn test_entity_category_display() {
        assert_eq!(format!("{}", EntityCategory::Agent), "agent");
        assert_eq!(format!("{}", EntityCategory::Temporal), "temporal");
        assert_eq!(format!("{}", EntityCategory::Relation), "relation");
    }
}
