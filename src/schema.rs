//! Schema harmonization for multi-dataset NER training.
//!
//! # The Schema Misalignment Problem
//!
//! Different NER datasets use incompatible label schemas:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │ Dataset        │ "Marie Curie"  │ "Paris"      │ "Americans"        │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │ CoNLL-2003     │ PER            │ LOC          │ MISC               │
//! │ OntoNotes 5.0  │ PERSON         │ GPE          │ NORP               │
//! │ Science Corpus │ SCIENTIST      │ LOCATION     │ —                  │
//! │ MultiNERD      │ PER            │ LOC          │ —                  │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! **Naive concatenation causes ~30% F1 degradation** (CyberNER 2025, ESNERA 2025).
//!
//! # Solution: Canonical Ontology
//!
//! This module defines a single source of truth for entity types, with explicit
//! mappings from each dataset schema. Information loss is documented and intentional.
//!
//! # Usage
//!
//! ```rust
//! use anno::schema::{CanonicalType, DatasetSchema, SchemaMapper};
//!
//! // Create mapper for OntoNotes
//! let mapper = SchemaMapper::for_dataset(DatasetSchema::OntoNotes);
//!
//! // Map dataset-specific label to canonical type
//! let canonical = mapper.to_canonical("NORP");
//! assert_eq!(canonical.name(), "GROUP");  // Not "ORG"!
//!
//! // Check information loss
//! let loss = mapper.information_loss("FAC");
//! assert!(loss.is_some());  // FAC → LOCATION loses "man-made" semantics
//! ```
//!
//! # Research References
//!
//! - CyberNER (2025): Schema harmonization for cyber threat NER
//! - ESNERA (2025): Entity schema normalization for evaluation
//! - OntoNotes 5.0 Guidelines: <https://catalog.ldc.upenn.edu/docs/LDC2013T19/OntoNotes-Release-5.0.pdf>

use crate::{EntityCategory, EntityType};
use std::collections::HashMap;

// =============================================================================
// Canonical Entity Types
// =============================================================================

/// Canonical entity type in the unified schema.
///
/// This is the single source of truth. All dataset-specific labels map here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CanonicalType {
    // === Agent Types (people and groups) ===
    /// Individual person (maps from: PER, PERSON, ACTOR, DIRECTOR, etc.)
    Person,
    /// Group of people by nationality/religion/politics (maps from: NORP)
    /// NOT the same as Organization!
    Group,

    // === Organization Types ===
    /// Formal organization (maps from: ORG, ORGANIZATION, CORPORATION)
    Organization,

    // === Location Types (with semantic preservation) ===
    /// Geopolitical entity - has government (maps from: GPE, COUNTRY, CITY)
    GeopoliticalEntity,
    /// Natural location (maps from: LOC, LOCATION - mountains, rivers)
    NaturalLocation,
    /// Man-made facility (maps from: FAC, FACILITY - buildings, airports)
    Facility,
    /// Generic location (fallback when distinction unknown)
    Location,

    // === Temporal Types ===
    /// Date expression
    Date,
    /// Time expression
    Time,

    // === Numeric Types ===
    /// Monetary value
    Money,
    /// Percentage
    Percent,
    /// Quantity with unit
    Quantity,
    /// Cardinal number
    Cardinal,
    /// Ordinal number
    Ordinal,

    // === Creative/Legal ===
    /// Creative work (maps from: WORK_OF_ART, TITLE, creative-work)
    CreativeWork,
    /// Product (maps from: PRODUCT, PROD)
    Product,
    /// Event (maps from: EVENT, EVE)
    Event,
    /// Law or legal document
    Law,
    /// Language
    Language,

    // === Domain-Specific (Biomedical) ===
    /// Disease or medical condition
    Disease,
    /// Chemical compound
    Chemical,
    /// Gene
    Gene,
    /// Drug
    Drug,

    // === Domain-Specific (Other) ===
    /// Animal
    Animal,
    /// Plant
    Plant,
    /// Food item
    Food,

    // === Fallback ===
    /// Miscellaneous (maps from: MISC, unknown types)
    Misc,
}

impl CanonicalType {
    /// Get the canonical name.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Person => "PERSON",
            Self::Group => "GROUP",
            Self::Organization => "ORG",
            Self::GeopoliticalEntity => "GPE",
            Self::NaturalLocation => "LOC",
            Self::Facility => "FAC",
            Self::Location => "LOCATION",
            Self::Date => "DATE",
            Self::Time => "TIME",
            Self::Money => "MONEY",
            Self::Percent => "PERCENT",
            Self::Quantity => "QUANTITY",
            Self::Cardinal => "CARDINAL",
            Self::Ordinal => "ORDINAL",
            Self::CreativeWork => "WORK_OF_ART",
            Self::Product => "PRODUCT",
            Self::Event => "EVENT",
            Self::Law => "LAW",
            Self::Language => "LANGUAGE",
            Self::Disease => "DISEASE",
            Self::Chemical => "CHEMICAL",
            Self::Gene => "GENE",
            Self::Drug => "DRUG",
            Self::Animal => "ANIMAL",
            Self::Plant => "PLANT",
            Self::Food => "FOOD",
            Self::Misc => "MISC",
        }
    }

    /// Get the category for this canonical type.
    #[must_use]
    pub fn category(&self) -> EntityCategory {
        match self {
            Self::Person | Self::Group => EntityCategory::Agent,
            Self::Organization => EntityCategory::Organization,
            Self::GeopoliticalEntity | Self::NaturalLocation | Self::Facility | Self::Location => {
                EntityCategory::Place
            }
            Self::Date | Self::Time => EntityCategory::Temporal,
            Self::Money | Self::Percent | Self::Quantity | Self::Cardinal | Self::Ordinal => {
                EntityCategory::Numeric
            }
            Self::CreativeWork | Self::Product | Self::Event | Self::Law | Self::Language => {
                EntityCategory::Creative
            }
            Self::Disease | Self::Chemical | Self::Gene | Self::Drug => EntityCategory::Agent,
            Self::Animal | Self::Plant | Self::Food => EntityCategory::Misc,
            Self::Misc => EntityCategory::Misc,
        }
    }

    /// Convert to the legacy EntityType for compatibility.
    #[must_use]
    pub fn to_entity_type(&self) -> EntityType {
        match self {
            Self::Person => EntityType::Person,
            Self::Group => EntityType::custom("GROUP", EntityCategory::Agent),
            Self::Organization => EntityType::Organization,
            Self::GeopoliticalEntity => EntityType::custom("GPE", EntityCategory::Place),
            Self::NaturalLocation => EntityType::Location,
            Self::Facility => EntityType::custom("FAC", EntityCategory::Place),
            Self::Location => EntityType::Location,
            Self::Date => EntityType::Date,
            Self::Time => EntityType::Time,
            Self::Money => EntityType::Money,
            Self::Percent => EntityType::Percent,
            Self::Quantity => EntityType::Quantity,
            Self::Cardinal => EntityType::Cardinal,
            Self::Ordinal => EntityType::Ordinal,
            Self::CreativeWork => EntityType::custom("WORK_OF_ART", EntityCategory::Creative),
            Self::Product => EntityType::custom("PRODUCT", EntityCategory::Misc),
            Self::Event => EntityType::custom("EVENT", EntityCategory::Misc),
            Self::Law => EntityType::custom("LAW", EntityCategory::Misc),
            Self::Language => EntityType::custom("LANGUAGE", EntityCategory::Misc),
            Self::Disease => EntityType::custom("DISEASE", EntityCategory::Agent),
            Self::Chemical => EntityType::custom("CHEMICAL", EntityCategory::Misc),
            Self::Gene => EntityType::custom("GENE", EntityCategory::Misc),
            Self::Drug => EntityType::custom("DRUG", EntityCategory::Misc),
            Self::Animal => EntityType::custom("ANIMAL", EntityCategory::Misc),
            Self::Plant => EntityType::custom("PLANT", EntityCategory::Misc),
            Self::Food => EntityType::custom("FOOD", EntityCategory::Misc),
            Self::Misc => EntityType::Other("MISC".to_string()),
        }
    }
}

// =============================================================================
// Dataset Schemas
// =============================================================================

/// Known dataset schemas for automatic mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DatasetSchema {
    /// CoNLL-2003: PER, LOC, ORG, MISC
    CoNLL2003,
    /// OntoNotes 5.0: 18 types including GPE, NORP, FAC
    OntoNotes,
    /// MultiNERD: 15 types
    MultiNERD,
    /// FewNERD: 8 coarse + 66 fine types
    FewNERD,
    /// CrossNER: Domain-specific types
    CrossNER,
    /// BC5CDR: Chemical, Disease
    BC5CDR,
    /// NCBI Disease: Disease only
    NCBIDisease,
    /// MIT Movie: Actor, Director, Title, etc.
    MITMovie,
    /// MIT Restaurant: Restaurant_Name, Cuisine, etc.
    MITRestaurant,
    /// WNUT-17: person, location, corporation, product, creative-work, group
    WNUT17,
}

impl DatasetSchema {
    /// Get the entity labels used by this dataset.
    #[must_use]
    pub fn labels(&self) -> &'static [&'static str] {
        match self {
            Self::CoNLL2003 => &["PER", "LOC", "ORG", "MISC"],
            Self::OntoNotes => &[
                "PERSON",
                "NORP",
                "FAC",
                "ORG",
                "GPE",
                "LOC",
                "PRODUCT",
                "EVENT",
                "WORK_OF_ART",
                "LAW",
                "LANGUAGE",
                "DATE",
                "TIME",
                "PERCENT",
                "MONEY",
                "QUANTITY",
                "ORDINAL",
                "CARDINAL",
            ],
            Self::MultiNERD => &[
                "PER", "LOC", "ORG", "ANIM", "BIO", "CEL", "DIS", "EVE", "FOOD", "INST", "MEDIA",
                "MYTH", "PLANT", "TIME", "VEHI",
            ],
            Self::FewNERD => &[
                "person",
                "organization",
                "location",
                "building",
                "art",
                "product",
                "event",
                "other",
            ],
            Self::CrossNER => &[
                "politician",
                "election",
                "political_party",
                "country",
                "location",
                "organization",
                "person",
                "misc",
            ],
            Self::BC5CDR => &["Chemical", "Disease"],
            Self::NCBIDisease => &["Disease"],
            Self::MITMovie => &[
                "Actor",
                "Director",
                "Genre",
                "Title",
                "Year",
                "Song",
                "Character",
                "Plot",
                "Rating",
            ],
            Self::MITRestaurant => &[
                "Amenity",
                "Cuisine",
                "Dish",
                "Hours",
                "Location",
                "Price",
                "Rating",
                "Restaurant_Name",
            ],
            Self::WNUT17 => &[
                "person",
                "location",
                "corporation",
                "product",
                "creative-work",
                "group",
            ],
        }
    }
}

// =============================================================================
// Information Loss Tracking
// =============================================================================

/// Documents information lost during schema mapping.
#[derive(Debug, Clone)]
pub struct InformationLoss {
    /// The original fine-grained label
    pub original: String,
    /// The coarse canonical type it maps to
    pub canonical: CanonicalType,
    /// What semantic information is lost
    pub lost_semantics: &'static str,
}

// =============================================================================
// Schema Mapper
// =============================================================================

/// Maps dataset-specific labels to canonical types.
#[derive(Debug, Clone)]
pub struct SchemaMapper {
    /// The source dataset schema
    pub source_schema: DatasetSchema,
    /// Label → CanonicalType mapping
    mappings: HashMap<String, CanonicalType>,
    /// Label → InformationLoss (if any)
    losses: HashMap<String, InformationLoss>,
}

impl SchemaMapper {
    /// Create a mapper for a specific dataset.
    #[must_use]
    pub fn for_dataset(schema: DatasetSchema) -> Self {
        let mut mapper = Self {
            source_schema: schema,
            mappings: HashMap::new(),
            losses: HashMap::new(),
        };

        match schema {
            DatasetSchema::CoNLL2003 => {
                mapper.add("PER", CanonicalType::Person);
                mapper.add("LOC", CanonicalType::Location);
                mapper.add("ORG", CanonicalType::Organization);
                mapper.add("MISC", CanonicalType::Misc);
            }
            DatasetSchema::OntoNotes => {
                // Person types
                mapper.add("PERSON", CanonicalType::Person);

                // CRITICAL: NORP is NOT Organization!
                mapper.add_with_loss(
                    "NORP",
                    CanonicalType::Group,
                    "Nationalities/religions/politics - distinct from formal organizations",
                );

                // Location types - preserve distinctions
                mapper.add("GPE", CanonicalType::GeopoliticalEntity);
                mapper.add_with_loss(
                    "LOC",
                    CanonicalType::NaturalLocation,
                    "Natural locations (mountains, rivers)",
                );
                mapper.add_with_loss(
                    "FAC",
                    CanonicalType::Facility,
                    "Man-made structures (buildings, bridges)",
                );

                // Organization
                mapper.add("ORG", CanonicalType::Organization);

                // Temporal
                mapper.add("DATE", CanonicalType::Date);
                mapper.add("TIME", CanonicalType::Time);

                // Numeric
                mapper.add("MONEY", CanonicalType::Money);
                mapper.add("PERCENT", CanonicalType::Percent);
                mapper.add("QUANTITY", CanonicalType::Quantity);
                mapper.add("CARDINAL", CanonicalType::Cardinal);
                mapper.add("ORDINAL", CanonicalType::Ordinal);

                // Creative/Legal
                mapper.add("PRODUCT", CanonicalType::Product);
                mapper.add("EVENT", CanonicalType::Event);
                mapper.add("WORK_OF_ART", CanonicalType::CreativeWork);
                mapper.add("LAW", CanonicalType::Law);
                mapper.add("LANGUAGE", CanonicalType::Language);
            }
            DatasetSchema::MultiNERD => {
                mapper.add("PER", CanonicalType::Person);
                mapper.add("LOC", CanonicalType::Location);
                mapper.add("ORG", CanonicalType::Organization);
                mapper.add("ANIM", CanonicalType::Animal);
                mapper.add_with_loss("BIO", CanonicalType::Misc, "Biological entities");
                mapper.add_with_loss("CEL", CanonicalType::Misc, "Celestial bodies");
                mapper.add("DIS", CanonicalType::Disease);
                mapper.add("EVE", CanonicalType::Event);
                mapper.add("FOOD", CanonicalType::Food);
                mapper.add_with_loss("INST", CanonicalType::Misc, "Instruments");
                mapper.add_with_loss("MEDIA", CanonicalType::CreativeWork, "Media works");
                mapper.add_with_loss("MYTH", CanonicalType::Misc, "Mythological entities");
                mapper.add("PLANT", CanonicalType::Plant);
                mapper.add("TIME", CanonicalType::Time);
                mapper.add_with_loss("VEHI", CanonicalType::Product, "Vehicles");
            }
            DatasetSchema::FewNERD => {
                mapper.add("person", CanonicalType::Person);
                mapper.add("organization", CanonicalType::Organization);
                mapper.add("location", CanonicalType::Location);
                mapper.add_with_loss("building", CanonicalType::Facility, "Buildings/structures");
                mapper.add("art", CanonicalType::CreativeWork);
                mapper.add("product", CanonicalType::Product);
                mapper.add("event", CanonicalType::Event);
                mapper.add("other", CanonicalType::Misc);
            }
            DatasetSchema::CrossNER => {
                mapper.add_with_loss("politician", CanonicalType::Person, "Political role lost");
                mapper.add_with_loss(
                    "election",
                    CanonicalType::Event,
                    "Election specificity lost",
                );
                mapper.add_with_loss(
                    "political_party",
                    CanonicalType::Organization,
                    "Political nature lost",
                );
                mapper.add("country", CanonicalType::GeopoliticalEntity);
                mapper.add("location", CanonicalType::Location);
                mapper.add("organization", CanonicalType::Organization);
                mapper.add("person", CanonicalType::Person);
                mapper.add("misc", CanonicalType::Misc);
            }
            DatasetSchema::BC5CDR => {
                mapper.add("Chemical", CanonicalType::Chemical);
                mapper.add("Disease", CanonicalType::Disease);
            }
            DatasetSchema::NCBIDisease => {
                mapper.add("Disease", CanonicalType::Disease);
            }
            DatasetSchema::MITMovie => {
                mapper.add_with_loss("Actor", CanonicalType::Person, "Acting role lost");
                mapper.add_with_loss("Director", CanonicalType::Person, "Directing role lost");
                mapper.add_with_loss("Character", CanonicalType::Person, "Fictional status lost");
                mapper.add("Title", CanonicalType::CreativeWork);
                mapper.add("Year", CanonicalType::Date);
                mapper.add_with_loss("Song", CanonicalType::CreativeWork, "Song vs film lost");
                mapper.add_with_loss("Genre", CanonicalType::Misc, "Genre semantics lost");
                mapper.add_with_loss("Plot", CanonicalType::Misc, "Plot description lost");
                mapper.add_with_loss("Rating", CanonicalType::Misc, "Rating semantics lost");
            }
            DatasetSchema::MITRestaurant => {
                mapper.add("Restaurant_Name", CanonicalType::Organization);
                mapper.add("Location", CanonicalType::Location);
                mapper.add_with_loss("Cuisine", CanonicalType::Misc, "Cuisine type lost");
                mapper.add_with_loss("Dish", CanonicalType::Food, "Dish specifics lost");
                mapper.add("Price", CanonicalType::Money);
                mapper.add_with_loss("Amenity", CanonicalType::Misc, "Amenity type lost");
                mapper.add("Hours", CanonicalType::Time);
                mapper.add_with_loss("Rating", CanonicalType::Misc, "Rating semantics lost");
            }
            DatasetSchema::WNUT17 => {
                mapper.add("person", CanonicalType::Person);
                mapper.add("location", CanonicalType::Location);
                mapper.add("corporation", CanonicalType::Organization);
                mapper.add("product", CanonicalType::Product);
                mapper.add("creative-work", CanonicalType::CreativeWork);
                mapper.add("group", CanonicalType::Group);
            }
        }

        mapper
    }

    /// Add a simple mapping (no information loss).
    fn add(&mut self, label: &str, canonical: CanonicalType) {
        self.mappings.insert(label.to_uppercase(), canonical);
    }

    /// Add a mapping with documented information loss.
    fn add_with_loss(
        &mut self,
        label: &str,
        canonical: CanonicalType,
        lost_semantics: &'static str,
    ) {
        let upper = label.to_uppercase();
        self.mappings.insert(upper.clone(), canonical);
        self.losses.insert(
            upper.clone(),
            InformationLoss {
                original: label.to_string(),
                canonical,
                lost_semantics,
            },
        );
    }

    /// Map a dataset label to canonical type.
    #[must_use]
    pub fn to_canonical(&self, label: &str) -> CanonicalType {
        self.mappings
            .get(&label.to_uppercase())
            .copied()
            .unwrap_or(CanonicalType::Misc)
    }

    /// Get information loss for a label (if any).
    #[must_use]
    pub fn information_loss(&self, label: &str) -> Option<&InformationLoss> {
        self.losses.get(&label.to_uppercase())
    }

    /// Map to EntityType for compatibility with existing code.
    #[must_use]
    pub fn to_entity_type(&self, label: &str) -> EntityType {
        self.to_canonical(label).to_entity_type()
    }

    /// Get all mappings that have information loss.
    pub fn all_losses(&self) -> impl Iterator<Item = &InformationLoss> {
        self.losses.values()
    }

    /// Calculate label overlap with another schema.
    ///
    /// Used to detect if "zero-shot" evaluation is actually fair.
    /// High overlap (>80%) means evaluation inflates scores.
    #[must_use]
    pub fn label_overlap(&self, other: &SchemaMapper) -> f64 {
        let self_canonicals: std::collections::HashSet<_> =
            self.mappings.values().copied().collect();
        let other_canonicals: std::collections::HashSet<_> =
            other.mappings.values().copied().collect();

        let intersection = self_canonicals.intersection(&other_canonicals).count();
        let union = self_canonicals.union(&other_canonicals).count();

        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }
}

// =============================================================================
// Unified Mapping Function (replaces all the ad-hoc ones)
// =============================================================================

/// Unified label mapping - THE SINGLE SOURCE OF TRUTH.
///
/// Replaces:
/// - `EntityType::from_label()` (partial)
/// - `map_entity_type()` in loader.rs
/// - `map_label_to_entity_type()` in datasets.rs
/// - `string_to_entity_type()` in bio_adapter.rs
///
/// # Arguments
/// * `label` - The entity type label from any dataset
/// * `schema` - Optional source schema for precise mapping
///
/// # Returns
/// The canonical EntityType
#[must_use]
pub fn map_to_canonical(label: &str, schema: Option<DatasetSchema>) -> EntityType {
    let label = label
        .strip_prefix("B-")
        .or_else(|| label.strip_prefix("I-"))
        .or_else(|| label.strip_prefix("E-"))
        .or_else(|| label.strip_prefix("S-"))
        .or_else(|| label.strip_prefix("L-"))
        .or_else(|| label.strip_prefix("U-"))
        .unwrap_or(label);

    if let Some(schema) = schema {
        SchemaMapper::for_dataset(schema).to_entity_type(label)
    } else {
        // Fallback: use heuristic mapping
        map_label_heuristic(label)
    }
}

/// Heuristic mapping when schema is unknown.
fn map_label_heuristic(label: &str) -> EntityType {
    match label.to_uppercase().as_str() {
        // Person types
        "PER" | "PERSON" | "ACTOR" | "DIRECTOR" | "CHARACTER" | "POLITICIAN" => EntityType::Person,

        // NORP - distinct from ORG!
        "NORP" | "GROUP" | "NATIONALITY" | "RELIGION" => {
            EntityType::custom("GROUP", EntityCategory::Agent)
        }

        // Organization types
        "ORG" | "ORGANIZATION" | "ORGANISATION" | "CORPORATION" | "COMPANY" | "POLITICAL_PARTY"
        | "RESTAURANT_NAME" => EntityType::Organization,

        // Location types - preserve GPE/FAC when possible
        "GPE" | "COUNTRY" | "CITY" | "STATE" => EntityType::custom("GPE", EntityCategory::Place),
        "FAC" | "FACILITY" | "BUILDING" => EntityType::custom("FAC", EntityCategory::Place),
        "LOC" | "LOCATION" | "GEO" => EntityType::Location,

        // Temporal
        "DATE" | "YEAR" => EntityType::Date,
        "TIME" | "HOURS" => EntityType::Time,

        // Numeric
        "MONEY" | "PRICE" | "CURRENCY" => EntityType::Money,
        "PERCENT" | "PERCENTAGE" => EntityType::Percent,
        "QUANTITY" => EntityType::Quantity,
        "CARDINAL" => EntityType::Cardinal,
        "ORDINAL" => EntityType::Ordinal,

        // Creative/Legal
        "PRODUCT" | "PROD" => EntityType::custom("PRODUCT", EntityCategory::Misc),
        "EVENT" | "EVE" | "ELECTION" => EntityType::custom("EVENT", EntityCategory::Misc),
        "WORK_OF_ART" | "CREATIVE-WORK" | "TITLE" | "SONG" | "ART" | "MEDIA" => {
            EntityType::custom("WORK_OF_ART", EntityCategory::Creative)
        }
        "LAW" => EntityType::custom("LAW", EntityCategory::Misc),
        "LANGUAGE" => EntityType::custom("LANGUAGE", EntityCategory::Misc),

        // Biomedical
        "DISEASE" | "DIS" => EntityType::custom("DISEASE", EntityCategory::Agent),
        "CHEMICAL" => EntityType::custom("CHEMICAL", EntityCategory::Misc),
        "GENE" => EntityType::custom("GENE", EntityCategory::Misc),
        "DRUG" => EntityType::custom("DRUG", EntityCategory::Misc),

        // Other domain types
        "ANIM" | "ANIMAL" => EntityType::custom("ANIMAL", EntityCategory::Misc),
        "PLANT" => EntityType::custom("PLANT", EntityCategory::Misc),
        "FOOD" | "DISH" | "CUISINE" => EntityType::custom("FOOD", EntityCategory::Misc),
        "VEHI" | "VEHICLE" => EntityType::custom("VEHICLE", EntityCategory::Misc),

        // Contact
        "EMAIL" => EntityType::Email,
        "URL" | "URI" => EntityType::Url,
        "PHONE" | "TELEPHONE" => EntityType::Phone,

        // Misc fallback
        "MISC" | "MISCELLANEOUS" | "O" | "OTHER" => EntityType::Other("MISC".to_string()),

        // Unknown - preserve original
        other => EntityType::Other(other.to_string()),
    }
}

// =============================================================================
// Coarse Schema for Training
// =============================================================================

/// Coarse-grained schema for multi-dataset training.
///
/// Use this when training on concatenated datasets to avoid label conflicts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CoarseType {
    /// Any person entity
    Person,
    /// Any organization entity
    Organization,
    /// Any location entity
    Location,
    /// Any temporal entity
    DateTime,
    /// Any numeric entity
    Numeric,
    /// Everything else
    Other,
}

impl CoarseType {
    /// Map from canonical type.
    #[must_use]
    pub fn from_canonical(ct: CanonicalType) -> Self {
        match ct {
            CanonicalType::Person | CanonicalType::Group => Self::Person,
            CanonicalType::Organization => Self::Organization,
            CanonicalType::GeopoliticalEntity
            | CanonicalType::NaturalLocation
            | CanonicalType::Facility
            | CanonicalType::Location => Self::Location,
            CanonicalType::Date | CanonicalType::Time => Self::DateTime,
            CanonicalType::Money
            | CanonicalType::Percent
            | CanonicalType::Quantity
            | CanonicalType::Cardinal
            | CanonicalType::Ordinal => Self::Numeric,
            _ => Self::Other,
        }
    }

    /// Map from any label.
    #[must_use]
    pub fn from_label(label: &str) -> Self {
        let canonical = SchemaMapper::for_dataset(DatasetSchema::OntoNotes).to_canonical(label);
        Self::from_canonical(canonical)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_norp_is_not_organization() {
        let mapper = SchemaMapper::for_dataset(DatasetSchema::OntoNotes);
        let norp = mapper.to_canonical("NORP");
        let org = mapper.to_canonical("ORG");

        assert_eq!(norp, CanonicalType::Group);
        assert_eq!(org, CanonicalType::Organization);
        assert_ne!(norp, org, "NORP should NOT map to Organization!");
    }

    #[test]
    fn test_location_distinctions_preserved() {
        let mapper = SchemaMapper::for_dataset(DatasetSchema::OntoNotes);

        assert_eq!(
            mapper.to_canonical("GPE"),
            CanonicalType::GeopoliticalEntity
        );
        assert_eq!(mapper.to_canonical("LOC"), CanonicalType::NaturalLocation);
        assert_eq!(mapper.to_canonical("FAC"), CanonicalType::Facility);
    }

    #[test]
    fn test_information_loss_documented() {
        let mapper = SchemaMapper::for_dataset(DatasetSchema::OntoNotes);

        let fac_loss = mapper.information_loss("FAC");
        assert!(fac_loss.is_some());
        let loss_text = fac_loss.unwrap().lost_semantics.to_lowercase();
        // Check that loss contains info about structures/buildings
        assert!(loss_text.contains("structure") || loss_text.contains("building"));
    }

    #[test]
    fn test_conll_to_ontonotes_overlap() {
        let conll = SchemaMapper::for_dataset(DatasetSchema::CoNLL2003);
        let ontonotes = SchemaMapper::for_dataset(DatasetSchema::OntoNotes);

        let overlap = conll.label_overlap(&ontonotes);
        // CoNLL has 4 types, OntoNotes has 18 - expect low overlap
        assert!(overlap < 0.5);
    }

    #[test]
    fn test_unified_mapping_strips_bio() {
        let et = map_to_canonical("B-PER", None);
        assert_eq!(et, EntityType::Person);

        let et = map_to_canonical("I-ORG", None);
        assert_eq!(et, EntityType::Organization);
    }

    #[test]
    fn test_coarse_schema() {
        assert_eq!(
            CoarseType::from_canonical(CanonicalType::Person),
            CoarseType::Person
        );
        assert_eq!(
            CoarseType::from_canonical(CanonicalType::Group),
            CoarseType::Person
        );
        assert_eq!(
            CoarseType::from_canonical(CanonicalType::GeopoliticalEntity),
            CoarseType::Location
        );
    }
}
