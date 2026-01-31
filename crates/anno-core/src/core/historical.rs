//! Historical provenance types for ancient/historical text processing.
//!
//! # Research Context
//!
//! When processing ancient languages and historical texts, entities require
//! additional metadata that modern NLP typically ignores:
//!
//! - **Temporal provenance**: When was the text written? BCE/CE dates.
//! - **Epigraphic medium**: Stone inscription vs papyrus vs clay tablet.
//! - **Script/writing system**: Cuneiform, hieroglyphic, Linear B, etc.
//! - **Archaeological context**: Where found, current location, preservation.
//!
//! This module provides types for capturing this metadata, inspired by:
//! - Sommerschield et al. (2023): "Machine Learning for Ancient Languages"
//! - Digital epigraphy standards (EpiDoc, CIDOC-CRM)
//! - Ancient language corpora (ORACC, Perseus, TLG)
//!
//! # Example
//!
//! ```rust
//! use anno_core::core::historical::{HistoricalProvenance, EpigraphicMedium, HistoricalDate, Era};
//!
//! let provenance = HistoricalProvenance::new()
//!     .with_date(HistoricalDate::range_bce(1500, 1150))
//!     .with_medium(EpigraphicMedium::ClayTablet)
//!     .with_script("Cypro-Minoan")
//!     .with_find_spot("Enkomi, Cyprus")
//!     .with_corpus("ENKO");
//!
//! // Check if entity is from Bronze Age
//! assert!(provenance.is_bronze_age());
//! ```

use serde::{Deserialize, Serialize};

// =============================================================================
// Historical Date
// =============================================================================

/// Era designation for historical dates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum Era {
    /// Before Common Era (= BC)
    BCE,
    /// Common Era (= AD)
    #[default]
    CE,
}

impl std::fmt::Display for Era {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Era::BCE => write!(f, "BCE"),
            Era::CE => write!(f, "CE"),
        }
    }
}

/// A historical date, possibly imprecise.
///
/// Ancient dates are often imprecise (e.g., "circa 1500 BCE", "15th century BCE").
/// This type captures both point-in-time and range dates.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HistoricalDate {
    /// Start year (negative for BCE in internal representation)
    pub year_start: i32,
    /// End year (if a range)
    pub year_end: Option<i32>,
    /// Era for display purposes
    pub era: Era,
    /// Whether the date is approximate ("circa")
    pub circa: bool,
    /// Textual note (e.g., "Late Bronze Age")
    pub note: Option<String>,
}

impl HistoricalDate {
    /// Create a point-in-time CE date.
    pub fn ce(year: i32) -> Self {
        Self {
            year_start: year,
            year_end: None,
            era: Era::CE,
            circa: false,
            note: None,
        }
    }

    /// Create a point-in-time BCE date.
    pub fn bce(year: i32) -> Self {
        Self {
            year_start: -year.abs(),
            year_end: None,
            era: Era::BCE,
            circa: false,
            note: None,
        }
    }

    /// Create a range of years (BCE).
    ///
    /// Note: `start` and `end` should be positive; internally stored as negative.
    pub fn range_bce(start: i32, end: i32) -> Self {
        Self {
            year_start: -start.abs(),
            year_end: Some(-end.abs()),
            era: Era::BCE,
            circa: false,
            note: None,
        }
    }

    /// Create a range of years (CE).
    pub fn range_ce(start: i32, end: i32) -> Self {
        Self {
            year_start: start,
            year_end: Some(end),
            era: Era::CE,
            circa: false,
            note: None,
        }
    }

    /// Mark as approximate ("circa").
    pub fn circa(mut self) -> Self {
        self.circa = true;
        self
    }

    /// Add a textual note.
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.note = Some(note.into());
        self
    }

    /// Check if this date is in the Bronze Age (roughly 3300-1200 BCE).
    pub fn is_bronze_age(&self) -> bool {
        // Bronze Age: ~3300 BCE to ~1200 BCE
        self.year_start <= -1200 && self.year_start >= -3300
    }

    /// Check if this date is in the Iron Age (roughly 1200-500 BCE).
    pub fn is_iron_age(&self) -> bool {
        self.year_start <= -500 && self.year_start >= -1200
    }

    /// Check if this is an ancient date (before 500 CE).
    pub fn is_ancient(&self) -> bool {
        self.year_start < 500
    }

    /// Get the midpoint year (useful for sorting/comparison).
    pub fn midpoint(&self) -> i32 {
        match self.year_end {
            Some(end) => (self.year_start + end) / 2,
            None => self.year_start,
        }
    }
}

impl std::fmt::Display for HistoricalDate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let prefix = if self.circa { "c. " } else { "" };

        let display_year = |y: i32| -> (i32, Era) {
            if y < 0 {
                (-y, Era::BCE)
            } else {
                (y, Era::CE)
            }
        };

        let (start_abs, start_era) = display_year(self.year_start);

        if let Some(end) = self.year_end {
            let (end_abs, _) = display_year(end);
            write!(f, "{}{}-{} {}", prefix, start_abs, end_abs, start_era)?;
        } else {
            write!(f, "{}{} {}", prefix, start_abs, start_era)?;
        }

        if let Some(ref note) = self.note {
            write!(f, " ({})", note)?;
        }

        Ok(())
    }
}

// =============================================================================
// Epigraphic Medium
// =============================================================================

/// The physical medium on which text is written.
///
/// Different media require different OCR/HTR approaches and have
/// characteristic preservation patterns.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum EpigraphicMedium {
    /// Clay tablet (Mesopotamian cuneiform, Linear B, etc.)
    ClayTablet,
    /// Stone inscription (Greek/Roman epigraphy, Egyptian hieroglyphs)
    Stone,
    /// Papyrus (Egyptian, Greek papyri)
    Papyrus,
    /// Parchment/vellum (medieval manuscripts)
    Parchment,
    /// Metal (bronze, lead tablets)
    Metal,
    /// Pottery/ceramic (ostraca, vessel inscriptions)
    Pottery,
    /// Wax tablet (Roman/medieval note-taking)
    WaxTablet,
    /// Wood (wooden tablets, bamboo slips)
    Wood,
    /// Seal impression (cylinder seals, stamp seals)
    Seal,
    /// Coin (numismatic inscriptions)
    Coin,
    /// Other medium with description
    Other(String),
}

impl EpigraphicMedium {
    /// Get typical preservation characteristics.
    pub fn preservation_notes(&self) -> &'static str {
        match self {
            EpigraphicMedium::ClayTablet => "Surcernos fire; damaged by water",
            EpigraphicMedium::Stone => "Durable; may have erosion/damage",
            EpigraphicMedium::Papyrus => "Fragile; surcernos in dry climates only",
            EpigraphicMedium::Parchment => "Durable; may have damage/palimpsest",
            EpigraphicMedium::Metal => "Durable; may have corrosion",
            EpigraphicMedium::Pottery => "Durable; often fragmentary",
            EpigraphicMedium::WaxTablet => "Extremely rare survival",
            EpigraphicMedium::Wood => "Rare survival except in dry/waterlogged contexts",
            EpigraphicMedium::Seal => "Durable; small scale",
            EpigraphicMedium::Coin => "Durable; standardized format",
            EpigraphicMedium::Other(_) => "Variable preservation",
        }
    }

    /// Whether this medium typically requires specialized OCR/HTR.
    pub fn requires_specialized_ocr(&self) -> bool {
        matches!(
            self,
            EpigraphicMedium::ClayTablet | EpigraphicMedium::Papyrus | EpigraphicMedium::Seal
        )
    }
}

impl std::fmt::Display for EpigraphicMedium {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EpigraphicMedium::ClayTablet => write!(f, "Clay tablet"),
            EpigraphicMedium::Stone => write!(f, "Stone"),
            EpigraphicMedium::Papyrus => write!(f, "Papyrus"),
            EpigraphicMedium::Parchment => write!(f, "Parchment"),
            EpigraphicMedium::Metal => write!(f, "Metal"),
            EpigraphicMedium::Pottery => write!(f, "Pottery/Ostracon"),
            EpigraphicMedium::WaxTablet => write!(f, "Wax tablet"),
            EpigraphicMedium::Wood => write!(f, "Wood"),
            EpigraphicMedium::Seal => write!(f, "Seal"),
            EpigraphicMedium::Coin => write!(f, "Coin"),
            EpigraphicMedium::Other(s) => write!(f, "{}", s),
        }
    }
}

// =============================================================================
// Writing System
// =============================================================================

/// Ancient writing system classification.
///
/// Important for understanding character-level processing requirements.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum WritingSystem {
    /// Alphabetic (Greek, Latin, Phoenician)
    Alphabetic,
    /// Syllabic (Linear A/B, Cypro-Minoan, Cherokee)
    Syllabic,
    /// Logographic (Chinese, Egyptian hieroglyphs)
    Logographic,
    /// Logosyllabic (Cuneiform, Maya)
    Logosyllabic,
    /// Abjad - consonantal alphabet (Hebrew, Arabic, Phoenician)
    Abjad,
    /// Abugida - consonant-vowel combinations (Brahmic scripts)
    Abugida,
    /// Undeciphered (script system unknown)
    Undeciphered,
    /// Other with description
    Other(String),
}

impl WritingSystem {
    /// Whether this system is fully deciphered.
    pub fn is_deciphered(&self) -> bool {
        !matches!(self, WritingSystem::Undeciphered)
    }

    /// Whether word boundaries are typically explicit.
    pub fn has_word_boundaries(&self) -> bool {
        matches!(self, WritingSystem::Alphabetic | WritingSystem::Abjad)
    }
}

impl std::fmt::Display for WritingSystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WritingSystem::Alphabetic => write!(f, "Alphabetic"),
            WritingSystem::Syllabic => write!(f, "Syllabic"),
            WritingSystem::Logographic => write!(f, "Logographic"),
            WritingSystem::Logosyllabic => write!(f, "Logosyllabic"),
            WritingSystem::Abjad => write!(f, "Abjad (consonantal)"),
            WritingSystem::Abugida => write!(f, "Abugida"),
            WritingSystem::Undeciphered => write!(f, "Undeciphered"),
            WritingSystem::Other(s) => write!(f, "{}", s),
        }
    }
}

// =============================================================================
// Historical Provenance
// =============================================================================

/// Full provenance information for historical/ancient text.
///
/// Captures the archaeological, temporal, and linguistic context
/// that is essential for proper interpretation of ancient texts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct HistoricalProvenance {
    /// Date or date range of the text
    pub date: Option<HistoricalDate>,
    /// Physical medium
    pub medium: Option<EpigraphicMedium>,
    /// Script name (e.g., "Cypro-Minoan", "Linear B", "Demotic")
    pub script: Option<String>,
    /// Writing system classification
    pub writing_system: Option<WritingSystem>,
    /// Language (if known)
    pub language: Option<String>,
    /// Find spot / provenance (e.g., "Enkomi, Cyprus")
    pub find_spot: Option<String>,
    /// Current location (museum, collection)
    pub current_location: Option<String>,
    /// Corpus/catalog identifier (e.g., "ENKO", "KN", "P.Oxy.")
    pub corpus: Option<String>,
    /// Object number within corpus
    pub object_number: Option<String>,
    /// Publication reference
    pub publication: Option<String>,
    /// Preservation state (0.0 = destroyed, 1.0 = perfect)
    pub preservation: Option<f64>,
    /// Whether text is fragmentary
    pub fragmentary: bool,
    /// Additional notes
    pub notes: Option<String>,
}

impl HistoricalProvenance {
    /// Create empty provenance.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the date.
    pub fn with_date(mut self, date: HistoricalDate) -> Self {
        self.date = Some(date);
        self
    }

    /// Set the medium.
    pub fn with_medium(mut self, medium: EpigraphicMedium) -> Self {
        self.medium = Some(medium);
        self
    }

    /// Set the script name.
    pub fn with_script(mut self, script: impl Into<String>) -> Self {
        self.script = Some(script.into());
        self
    }

    /// Set the writing system.
    pub fn with_writing_system(mut self, system: WritingSystem) -> Self {
        self.writing_system = Some(system);
        self
    }

    /// Set the language.
    pub fn with_language(mut self, lang: impl Into<String>) -> Self {
        self.language = Some(lang.into());
        self
    }

    /// Set the find spot.
    pub fn with_find_spot(mut self, spot: impl Into<String>) -> Self {
        self.find_spot = Some(spot.into());
        self
    }

    /// Set the current location.
    pub fn with_current_location(mut self, loc: impl Into<String>) -> Self {
        self.current_location = Some(loc.into());
        self
    }

    /// Set the corpus identifier.
    pub fn with_corpus(mut self, corpus: impl Into<String>) -> Self {
        self.corpus = Some(corpus.into());
        self
    }

    /// Set the object number.
    pub fn with_object_number(mut self, num: impl Into<String>) -> Self {
        self.object_number = Some(num.into());
        self
    }

    /// Set preservation state (0.0-1.0).
    pub fn with_preservation(mut self, pres: f64) -> Self {
        self.preservation = Some(pres.clamp(0.0, 1.0));
        self
    }

    /// Mark as fragmentary.
    pub fn fragmentary(mut self) -> Self {
        self.fragmentary = true;
        self
    }

    /// Add notes.
    pub fn with_notes(mut self, notes: impl Into<String>) -> Self {
        self.notes = Some(notes.into());
        self
    }

    /// Check if from Bronze Age.
    pub fn is_bronze_age(&self) -> bool {
        self.date
            .as_ref()
            .map(|d| d.is_bronze_age())
            .unwrap_or(false)
    }

    /// Check if from Iron Age.
    pub fn is_iron_age(&self) -> bool {
        self.date.as_ref().map(|d| d.is_iron_age()).unwrap_or(false)
    }

    /// Check if ancient (before 500 CE).
    pub fn is_ancient(&self) -> bool {
        self.date.as_ref().map(|d| d.is_ancient()).unwrap_or(false)
    }

    /// Check if the text is in an undeciphered script.
    pub fn is_undeciphered(&self) -> bool {
        self.writing_system == Some(WritingSystem::Undeciphered)
    }

    /// Format a citation string.
    pub fn citation(&self) -> String {
        let mut parts = Vec::new();

        if let Some(ref corpus) = self.corpus {
            if let Some(ref num) = self.object_number {
                parts.push(format!("{} {}", corpus, num));
            } else {
                parts.push(corpus.clone());
            }
        }

        if let Some(ref date) = self.date {
            parts.push(date.to_string());
        }

        if let Some(ref spot) = self.find_spot {
            parts.push(spot.clone());
        }

        parts.join(", ")
    }
}

impl std::fmt::Display for HistoricalProvenance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.citation())
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_historical_date_bce() {
        let date = HistoricalDate::bce(1500);
        assert_eq!(date.year_start, -1500);
        assert!(date.is_bronze_age());
        assert!(!date.is_iron_age());
    }

    #[test]
    fn test_historical_date_range() {
        let date = HistoricalDate::range_bce(1500, 1150);
        assert_eq!(date.year_start, -1500);
        assert_eq!(date.year_end, Some(-1150));
        assert!(date.is_bronze_age());
    }

    #[test]
    fn test_historical_date_display() {
        let date = HistoricalDate::range_bce(1500, 1150).circa();
        let s = format!("{}", date);
        assert!(s.contains("c."));
        assert!(s.contains("1500"));
        assert!(s.contains("BCE"));
    }

    #[test]
    fn test_historical_date_ce() {
        let date = HistoricalDate::ce(2024);
        assert_eq!(date.year_start, 2024);
        assert!(!date.is_ancient());
    }

    #[test]
    fn test_epigraphic_medium() {
        let medium = EpigraphicMedium::ClayTablet;
        assert!(medium.requires_specialized_ocr());
        assert!(medium.preservation_notes().contains("fire"));
    }

    #[test]
    fn test_writing_system() {
        let undeciphered = WritingSystem::Undeciphered;
        assert!(!undeciphered.is_deciphered());

        let alphabetic = WritingSystem::Alphabetic;
        assert!(alphabetic.has_word_boundaries());
    }

    #[test]
    fn test_historical_provenance_builder() {
        let prov = HistoricalProvenance::new()
            .with_date(HistoricalDate::range_bce(1500, 1150))
            .with_medium(EpigraphicMedium::ClayTablet)
            .with_script("Cypro-Minoan")
            .with_writing_system(WritingSystem::Undeciphered)
            .with_find_spot("Enkomi, Cyprus")
            .with_corpus("ENKO")
            .with_object_number("001")
            .fragmentary();

        assert!(prov.is_bronze_age());
        assert!(prov.is_undeciphered());
        assert!(prov.fragmentary);
        assert_eq!(prov.script, Some("Cypro-Minoan".to_string()));
    }

    #[test]
    fn test_historical_provenance_citation() {
        let prov = HistoricalProvenance::new()
            .with_corpus("ENKO")
            .with_object_number("001")
            .with_date(HistoricalDate::range_bce(1500, 1150))
            .with_find_spot("Enkomi, Cyprus");

        let citation = prov.citation();
        assert!(citation.contains("ENKO 001"));
        assert!(citation.contains("BCE"));
        assert!(citation.contains("Enkomi"));
    }

    #[test]
    fn test_midpoint() {
        let point = HistoricalDate::bce(1500);
        assert_eq!(point.midpoint(), -1500);

        let range = HistoricalDate::range_bce(1500, 1200);
        assert_eq!(range.midpoint(), -1350);
    }

    #[test]
    fn test_era_display() {
        assert_eq!(format!("{}", Era::BCE), "BCE");
        assert_eq!(format!("{}", Era::CE), "CE");
    }

    #[test]
    fn test_preservation() {
        let prov = HistoricalProvenance::new().with_preservation(0.75);
        assert_eq!(prov.preservation, Some(0.75));

        // Clamping
        let clamped = HistoricalProvenance::new().with_preservation(1.5);
        assert_eq!(clamped.preservation, Some(1.0));
    }

    #[test]
    fn test_epic_literature_provenance() {
        // Mahābhārata-style provenance: composed ~400 BCE - 400 CE
        // but set in mythological time (Dwapara Yuga)
        let mahabharata = HistoricalProvenance::new()
            .with_date(HistoricalDate::range_bce(400, -400).with_note("composition period"))
            .with_script("Devanagari")
            .with_corpus("Mahābhārata");

        // The composition period spans BCE to CE
        assert!(!mahabharata.is_bronze_age());
        assert_eq!(mahabharata.script, Some("Devanagari".to_string()));
    }

    #[test]
    fn test_circa_display() {
        let approx = HistoricalDate::bce(1500).circa();
        let display = format!("{}", approx);
        assert!(display.contains("c."));
        assert!(display.contains("1500"));
    }

    #[test]
    fn test_date_ordering() {
        let bronze = HistoricalDate::bce(1500);
        let iron = HistoricalDate::bce(800);
        let modern = HistoricalDate::ce(2000);

        // Midpoint ordering works for chronological sorting
        assert!(bronze.midpoint() < iron.midpoint());
        assert!(iron.midpoint() < modern.midpoint());
    }
}
