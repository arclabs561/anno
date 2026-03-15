//! Temporal entity tracking, parsing, and diachronic NER.
//!
//! # The Problem: Entities Change Over Time
//!
//! Traditional NER treats entities as static facts, but the world changes:
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────────────────┐
//! │                     ENTITIES ARE NOT STATIC                                │
//! ├────────────────────────────────────────────────────────────────────────────┤
//! │                                                                            │
//! │  "CEO of Microsoft"                                                        │
//! │  ─────────────────                                                         │
//! │                                                                            │
//! │  2000:        Steve Ballmer                                                │
//! │  2014-today:  Satya Nadella                                                │
//! │                                                                            │
//! │  "Capital of Germany"                                                      │
//! │  ────────────────────                                                      │
//! │                                                                            │
//! │  1949-1990:   Bonn (West Germany)                                          │
//! │  1990-today:  Berlin (unified Germany)                                     │
//! │                                                                            │
//! │  "USSR"                                                                    │
//! │  ─────                                                                     │
//! │                                                                            │
//! │  1922-1991:   Existed as a country                                         │
//! │  1991-today:  Historical reference only                                    │
//! │                                                                            │
//! └────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Temporal Entity Operations
//!
//! This module provides:
//!
//! 1. **Point-in-time queries**: Which entities were valid at timestamp T?
//! 2. **Entity evolution**: How did entity E change over time?
//! 3. **Temporal alignment**: Link entities across documents with different dates
//! 4. **Version tracking**: Track multiple values for the same slot over time
//!
//! # Example
//!
//! ```rust
//! use anno::temporal::{TemporalEntityTracker, TemporalQuery, EntityTimeline};
//! use anno::{Entity, EntityType};
//! use chrono::{TimeZone, Utc};
//!
//! let mut tracker = TemporalEntityTracker::new();
//!
//! // Add entities with temporal validity
//! let mut ballmer = Entity::new("Steve Ballmer", EntityType::Person, 0, 13, 0.9);
//! ballmer.set_valid_from(Utc.with_ymd_and_hms(2000, 1, 13, 0, 0, 0).unwrap());
//! ballmer.set_valid_until(Utc.with_ymd_and_hms(2014, 2, 4, 0, 0, 0).unwrap());
//! ballmer.normalized = Some("CEO_OF_MICROSOFT".into());
//! tracker.add_entity(ballmer);
//!
//! let mut nadella = Entity::new("Satya Nadella", EntityType::Person, 0, 13, 0.95);
//! nadella.set_valid_from(Utc.with_ymd_and_hms(2014, 2, 4, 0, 0, 0).unwrap());
//! nadella.normalized = Some("CEO_OF_MICROSOFT".into());
//! tracker.add_entity(nadella);
//!
//! // Query: Who was CEO in 2010?
//! let query_2010 = Utc.with_ymd_and_hms(2010, 6, 1, 0, 0, 0).unwrap();
//! let result = tracker.query_at(&query_2010);
//! assert!(result.iter().any(|e| e.text.contains("Ballmer")));
//!
//! // Query: Who was CEO in 2020?
//! let query_2020 = Utc.with_ymd_and_hms(2020, 6, 1, 0, 0, 0).unwrap();
//! let result = tracker.query_at(&query_2020);
//! assert!(result.iter().any(|e| e.text.contains("Nadella")));
//! ```
//!
//! # Cultural Assumptions and Limitations
//!
//! **This module's default implementation assumes Western/Gregorian temporal concepts.**
//!
//! The trait-based design allows extending to other temporal ontologies, but
//! users should be aware of these built-in assumptions:
//!
//! | Assumption | Western View | Alternative Views |
//! |------------|--------------|-------------------|
//! | Time structure | Linear, unidirectional | Cyclical (Hindu yugas, Mayan), spiral |
//! | Reference point | Fixed (CE/BCE, Unix epoch) | Event-based ("when the rains came") |
//! | Calendar | Gregorian (solar) | Lunar (Islamic), lunisolar (Hebrew, Chinese) |
//! | Granularity | Clock-based (hours, minutes) | Event-based, seasonal, relational |
//! | Precision | Valued and expected | May be culturally inappropriate |
//!
//! ## Non-Western Temporal Concepts
//!
//! ### African Temporal Philosophies
//!
//! Many African cultures conceptualize time differently from the Western linear model:
//!
//! - **Event-based time**: Time is marked by significant events, not abstract units.
//!   "After the harvest" or "when the chief visited" may be more meaningful than dates.
//! - **Relational time**: Time understood through social relationships and activities
//!   rather than clock positions.
//! - **Cyclical/seasonal**: Agricultural and ceremonial cycles structure time.
//! - **Ubuntu temporality**: Time as fundamentally social and communal.
//!
//! The Swahili concept of "sasa" (present) and "zamani" (past that shapes present)
//! differs from Western past/present/future trichotomy.
//!
//! ### East Asian Calendars
//!
//! - **Chinese calendar**: Lunisolar with 60-year cycles (干支), zodiac years
//! - **Japanese eras**: Named periods tied to imperial reigns (令和, Reiwa)
//! - **Korean**: Dangun calendar alongside Gregorian
//!
//! ### South Asian Concepts
//!
//! - **Hindu yugas**: Cosmic time cycles spanning millions of years
//! - **Tithi**: Lunar days used for religious observances
//! - **Panchang**: Five-limbed calendar system
//!
//! ### Islamic Calendar
//!
//! - **Hijri calendar**: Purely lunar, 12 months of 29-30 days
//! - Religious dates drift through Gregorian seasons
//!
//! ### Indigenous Temporal Systems
//!
//! Many indigenous cultures use:
//! - Seasonal markers ("when salmon run")
//! - Astronomical events ("after the first frost")
//! - Generational time ("in my grandmother's time")
//! - Dreamtime (Australian Aboriginal non-linear temporality)
//!
//! ## Extending for Non-Western Time
//!
//! Implement the `TemporalOntology` trait to add support for different
//! temporal systems. The trait design intentionally avoids assuming:
//! - Linear time
//! - Fixed reference points
//! - Gregorian calendar
//! - Clock-based precision
//!
//! See the trait documentation for examples.
//!
//! # Research Background
//!
//! Based on:
//! - Campos et al. (2014): "Survey of Temporal Information Extraction Research"
//! - Kanhabua & Nørvåg (2012): "A Survey of Time-aware Information Access"
//! - Berberich et al. (2010): "Timetravel: Temporal Web Search"
//! - Mbiti, John S. (1969): "African Religions and Philosophy" (African time concepts)
//! - Adjaye, Joseph K. (1994): "Time in the Black Experience"

use crate::{Entity, EntityType};
use anno_core::Confidence;
use chrono::{DateTime, Duration, NaiveDate, TimeZone, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Temporal Ontology Traits
// =============================================================================

/// A temporal reference within a specific ontology.
///
/// This represents "a point or region in time" according to some temporal system.
/// It intentionally does NOT assume:
/// - Linear time (can represent cyclical or event-based time)
/// - Fixed granularity (can be fuzzy, range, or precise)
/// - Gregorian calendar (can be any calendar system)
///
/// # Design Philosophy
///
/// The trait uses associated types rather than concrete DateTime to allow:
/// - Event-based references ("after the harvest")
/// - Cyclical references ("Year of the Dragon")
/// - Fuzzy references ("recently")
/// - Composite references ("the third Monday of Ramadan")
pub trait TemporalReference: Clone + std::fmt::Debug {
    /// Can this reference be grounded to UTC?
    ///
    /// Returns `false` for:
    /// - Event-based time ("when the war ended") without known dates
    /// - Recurring patterns ("every Monday") without a specific instance
    /// - Purely relational time ("in my grandfather's time")
    fn is_groundable(&self) -> bool;

    /// Attempt to convert to a UTC range.
    ///
    /// Returns `None` if this reference cannot be mapped to Gregorian time.
    /// Even groundable references may have wide ranges (e.g., "the 90s" → 10 years).
    fn to_utc_range(&self) -> Option<(DateTime<Utc>, DateTime<Utc>)>;

    /// Get the original text of this reference.
    fn source_text(&self) -> &str;

    /// Confidence in the interpretation (0.0 to 1.0).
    ///
    /// Lower for ambiguous expressions like "soon" or "long ago".
    fn confidence(&self) -> f64;
}

/// A temporal ontology defines how time is conceptualized and parsed.
///
/// Different cultures and contexts have fundamentally different notions of time.
/// This trait allows implementing parsers and reasoners for any temporal system.
///
/// # Built-in Implementations
///
/// - [`GregorianOntology`]: Western/ISO 8601 time (default)
///
/// # Example: Custom Ontology
///
/// ```rust,ignore
/// use anno::temporal::{TemporalOntology, TemporalReference};
///
/// /// Swahili temporal expressions
/// struct SwahiliOntology;
///
/// #[derive(Clone, Debug)]
/// enum SwahiliTime {
///     /// "kesho" - tomorrow
///     Kesho,
///     /// "jana" - yesterday
///     Jana,
///     /// "sasa" - now/present (but conceptually broader than "now")
///     Sasa,
///     /// "zamani" - the past that shapes the present
///     Zamani,
///     /// Event-based: "wakati wa mavuno" - harvest time
///     SeasonalEvent(String),
/// }
///
/// impl TemporalOntology for SwahiliOntology {
///     type Reference = SwahiliTime;
///     type Error = String;
///
///     fn parse(&self, text: &str, context: Option<&TemporalContext>) -> Result<Self::Reference, Self::Error> {
///         match text.to_lowercase().as_str() {
///             "kesho" => Ok(SwahiliTime::Kesho),
///             "jana" => Ok(SwahiliTime::Jana),
///             "sasa" => Ok(SwahiliTime::Sasa),
///             "zamani" => Ok(SwahiliTime::Zamani),
///             _ if text.contains("mavuno") => Ok(SwahiliTime::SeasonalEvent("harvest".into())),
///             _ => Err(format!("Unknown temporal expression: {}", text)),
///         }
///     }
///
///     fn supports_linear_time(&self) -> bool {
///         // Swahili time concepts are not strictly linear
///         false
///     }
/// }
/// ```
pub trait TemporalOntology {
    /// The type of temporal reference this ontology produces.
    type Reference: TemporalReference;

    /// Error type for parsing failures.
    type Error: std::fmt::Debug;

    /// Parse a text expression into a temporal reference.
    ///
    /// The `context` parameter provides:
    /// - Document date (for relative expressions)
    /// - Geographic location (for local calendars)
    /// - Previous references (for anaphora like "the next day")
    fn parse(
        &self,
        text: &str,
        context: Option<&TemporalContext>,
    ) -> Result<Self::Reference, Self::Error>;

    /// Does this ontology assume linear, unidirectional time?
    ///
    /// Returns `false` for cyclical (Hindu yugas), event-based (African),
    /// or non-linear (Aboriginal Dreamtime) temporal systems.
    fn supports_linear_time(&self) -> bool {
        true
    }

    /// Does this ontology support conversion to UTC?
    ///
    /// Returns `false` for purely event-based or mythological time systems.
    fn supports_utc_conversion(&self) -> bool {
        true
    }

    /// Get the name of this temporal system for documentation.
    fn name(&self) -> &str;

    /// Get supported language codes (ISO 639-1).
    fn supported_languages(&self) -> &[&str] {
        &["en"]
    }
}

/// Context for temporal parsing.
///
/// Provides information needed to resolve relative and context-dependent
/// temporal expressions.
#[derive(Debug, Clone, Default)]
pub struct TemporalContext {
    /// The publication/utterance date of the document.
    ///
    /// Used to resolve "yesterday", "next week", etc.
    pub document_date: Option<DateTime<Utc>>,

    /// Geographic location for local calendar conversion.
    ///
    /// Some calendars (Islamic, Hebrew) depend on location for precise dates.
    pub location: Option<String>,

    /// Previously mentioned temporal references (for anaphora resolution).
    ///
    /// "On Monday... the next day..." → Tuesday
    pub previous_references: Vec<String>,

    /// The language of the text being parsed.
    pub language: Option<String>,

    /// Cultural context hints.
    ///
    /// E.g., "academic" (fall semester = Sep-Dec), "fiscal" (Q1 = different dates)
    pub domain: Option<String>,
}

impl TemporalContext {
    /// Create a context with just a document date.
    #[must_use]
    pub fn with_document_date(date: DateTime<Utc>) -> Self {
        Self {
            document_date: Some(date),
            ..Default::default()
        }
    }

    /// Create a context with document date and language.
    #[must_use]
    pub fn with_date_and_language(date: DateTime<Utc>, language: impl Into<String>) -> Self {
        Self {
            document_date: Some(date),
            language: Some(language.into()),
            ..Default::default()
        }
    }
}

// =============================================================================
// Gregorian Ontology (Default Western Implementation)
// =============================================================================

/// Western/Gregorian temporal ontology.
///
/// This is the default implementation, handling:
/// - ISO 8601 dates and times
/// - Common English temporal expressions
/// - Relative references (yesterday, next week)
/// - Fuzzy references (recently, soon)
///
/// **Limitations**: This implementation embeds Western assumptions about time.
/// See the module documentation for non-Western alternatives.
#[derive(Debug, Clone, Default)]
pub struct GregorianOntology;

impl TemporalOntology for GregorianOntology {
    type Reference = GregorianReference;
    type Error = String;

    fn parse(
        &self,
        text: &str,
        context: Option<&TemporalContext>,
    ) -> Result<Self::Reference, Self::Error> {
        // Delegate to the existing parse_temporal_expression function
        let abstract_expr = parse_temporal_expression(text);

        // If we have context, try to ground relative expressions
        let grounded = if let Some(ctx) = context {
            if let Some(doc_date) = ctx.document_date {
                abstract_expr.ground(&doc_date)
            } else {
                Some(abstract_expr)
            }
        } else {
            Some(abstract_expr)
        };

        grounded
            .map(|expr| GregorianReference {
                text: text.to_string(),
                expression: expr,
            })
            .ok_or_else(|| format!("Could not parse temporal expression: {}", text))
    }

    fn name(&self) -> &str {
        "Gregorian (Western)"
    }

    fn supported_languages(&self) -> &[&str] {
        &["en", "de", "fr", "es", "it", "pt", "nl"]
    }
}

/// A temporal reference in the Gregorian system.
#[derive(Debug, Clone)]
pub struct GregorianReference {
    /// Original text
    pub text: String,
    /// Parsed abstract expression
    pub expression: AbstractTemporalExpression,
}

impl TemporalReference for GregorianReference {
    fn is_groundable(&self) -> bool {
        self.expression.granularity.is_groundable() || self.expression.grounded_range.is_some()
    }

    fn to_utc_range(&self) -> Option<(DateTime<Utc>, DateTime<Utc>)> {
        self.expression.grounded_range
    }

    fn source_text(&self) -> &str {
        &self.text
    }

    fn confidence(&self) -> f64 {
        self.expression.grounding_confidence.value()
    }
}

// =============================================================================
// Calendar System Traits
// =============================================================================

/// A calendar system for date representation.
///
/// Different from [`TemporalOntology`] in that calendars are specifically
/// about date representation, while ontologies are about temporal concepts.
///
/// # Built-in Implementations
///
/// - [`GregorianCalendar`]: Standard Western calendar
///
/// # Example: Islamic Calendar
///
/// ```rust,ignore
/// struct HijriCalendar;
///
/// impl CalendarSystem for HijriCalendar {
///     type Date = HijriDate;
///
///     fn to_gregorian(&self, date: &Self::Date) -> Option<NaiveDate> {
///         // Islamic calendar is purely lunar (354 or 355 days/year)
///         // Conversion requires astronomical calculation or lookup tables
///         todo!()
///     }
///
///     fn from_gregorian(&self, date: &NaiveDate) -> Option<Self::Date> {
///         todo!()
///     }
/// }
/// ```
pub trait CalendarSystem {
    /// The date type for this calendar.
    type Date: Clone + std::fmt::Debug;

    /// Convert to Gregorian date.
    fn to_gregorian(&self, date: &Self::Date) -> Option<NaiveDate>;

    /// Convert from Gregorian date.
    #[allow(clippy::wrong_self_convention)]
    fn from_gregorian(&self, date: &NaiveDate) -> Option<Self::Date>;

    /// Get the calendar name.
    fn name(&self) -> &str;

    /// Is this calendar lunar, solar, or lunisolar?
    fn calendar_type(&self) -> CalendarType {
        CalendarType::Solar
    }
}

/// Type of calendar system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalendarType {
    /// Solar calendar (e.g., Gregorian)
    Solar,
    /// Lunar calendar (e.g., Islamic Hijri)
    Lunar,
    /// Lunisolar calendar (e.g., Hebrew, Chinese)
    Lunisolar,
    /// Other (e.g., Mayan long count)
    Other,
}

/// Standard Gregorian calendar implementation.
#[derive(Debug, Clone, Default)]
pub struct GregorianCalendar;

impl CalendarSystem for GregorianCalendar {
    type Date = NaiveDate;

    fn to_gregorian(&self, date: &Self::Date) -> Option<NaiveDate> {
        Some(*date)
    }

    fn from_gregorian(&self, date: &NaiveDate) -> Option<Self::Date> {
        Some(*date)
    }

    fn name(&self) -> &str {
        "Gregorian"
    }
}

// =============================================================================
// Core Types
// =============================================================================

/// A temporal scope for queries.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TemporalScope {
    /// Query at a specific point in time
    PointInTime(DateTime<Utc>),
    /// Query across a time range
    Range {
        /// Start of the time range (inclusive)
        start: DateTime<Utc>,
        /// End of the time range (exclusive)
        end: DateTime<Utc>,
    },
    /// Query for entities valid at any point
    AnyTime,
    /// Query for currently valid entities (no end date)
    Current,
}

impl TemporalScope {
    /// Check if an entity is valid within this scope.
    #[must_use]
    pub fn contains(&self, entity: &Entity) -> bool {
        match self {
            Self::PointInTime(ts) => entity.valid_at(ts),
            Self::Range { start, end } => {
                // Entity overlaps with range if:
                // entity_start <= end AND (entity_end is None OR entity_end >= start)
                let entity_starts_before_end = match entity.valid_from.as_ref() {
                    None => true,
                    Some(ef) => ef <= end,
                };
                let entity_ends_after_start = match entity.valid_until.as_ref() {
                    None => true,
                    Some(eu) => eu >= start,
                };
                entity_starts_before_end && entity_ends_after_start
            }
            Self::AnyTime => true,
            Self::Current => entity.valid_until.is_none(),
        }
    }
}

/// A temporal query for entity lookup.
#[derive(Debug, Clone)]
pub struct TemporalQuery {
    /// The temporal scope
    pub scope: TemporalScope,
    /// Optional entity type filter
    pub entity_type: Option<EntityType>,
    /// Optional slot/role filter (e.g., "CEO_OF_MICROSOFT")
    pub slot: Option<String>,
    /// Include superseded (past) values
    pub include_historical: bool,
}

impl TemporalQuery {
    /// Create a point-in-time query.
    #[must_use]
    pub fn at(timestamp: DateTime<Utc>) -> Self {
        Self {
            scope: TemporalScope::PointInTime(timestamp),
            entity_type: None,
            slot: None,
            include_historical: false,
        }
    }

    /// Create a range query.
    #[must_use]
    pub fn between(start: DateTime<Utc>, end: DateTime<Utc>) -> Self {
        Self {
            scope: TemporalScope::Range { start, end },
            entity_type: None,
            slot: None,
            include_historical: true,
        }
    }

    /// Create a query for current values.
    #[must_use]
    pub fn current() -> Self {
        Self {
            scope: TemporalScope::Current,
            entity_type: None,
            slot: None,
            include_historical: false,
        }
    }

    /// Filter by entity type.
    #[must_use]
    pub fn with_type(mut self, entity_type: EntityType) -> Self {
        self.entity_type = Some(entity_type);
        self
    }

    /// Filter by slot/role.
    #[must_use]
    pub fn with_slot(mut self, slot: impl Into<String>) -> Self {
        self.slot = Some(slot.into());
        self
    }

    /// Include historical values.
    #[must_use]
    pub fn include_historical(mut self) -> Self {
        self.include_historical = true;
        self
    }
}

// =============================================================================
// Entity Timeline
// =============================================================================

/// Timeline of values for a single slot/role over time.
///
/// Example: "CEO of Microsoft" slot has values:
/// - 2000-2014: Steve Ballmer
/// - 2014-present: Satya Nadella
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityTimeline {
    /// The slot/role this timeline tracks
    pub slot: String,
    /// Values over time, sorted by start date
    pub versions: Vec<TimelineEntry>,
}

/// A single entry in an entity timeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineEntry {
    /// The entity value for this period
    pub entity: Entity,
    /// Optional source/provenance
    pub source: Option<String>,
    /// Whether this was inferred vs. explicitly stated
    pub inferred: bool,
}

impl EntityTimeline {
    /// Create a new timeline for a slot.
    #[must_use]
    pub fn new(slot: impl Into<String>) -> Self {
        Self {
            slot: slot.into(),
            versions: Vec::new(),
        }
    }

    /// Add an entity to the timeline.
    pub fn add(&mut self, entity: Entity, source: Option<String>) {
        self.versions.push(TimelineEntry {
            entity,
            source,
            inferred: false,
        });
        // Sort by valid_from (None goes first as "unknown/always")
        self.versions
            .sort_by(|a, b| match (&a.entity.valid_from, &b.entity.valid_from) {
                (None, None) => std::cmp::Ordering::Equal,
                (None, Some(_)) => std::cmp::Ordering::Less,
                (Some(_), None) => std::cmp::Ordering::Greater,
                (Some(a_from), Some(b_from)) => a_from.cmp(b_from),
            });
    }

    /// Get the value at a specific point in time.
    #[must_use]
    pub fn value_at(&self, timestamp: &DateTime<Utc>) -> Option<&Entity> {
        self.versions
            .iter()
            .rfind(|v| v.entity.valid_at(timestamp))
            .map(|v| &v.entity)
    }

    /// Get the current value (no end date).
    #[must_use]
    pub fn current(&self) -> Option<&Entity> {
        self.versions
            .iter()
            .rfind(|v| v.entity.valid_until.is_none())
            .map(|v| &v.entity)
    }

    /// Get all historical values.
    #[must_use]
    pub fn history(&self) -> Vec<&Entity> {
        self.versions.iter().map(|v| &v.entity).collect()
    }

    /// Check if there are gaps in the timeline.
    #[must_use]
    pub fn has_gaps(&self) -> bool {
        if self.versions.len() < 2 {
            return false;
        }

        for i in 0..self.versions.len() - 1 {
            let current = &self.versions[i];
            let next = &self.versions[i + 1];

            // If current has an end and next has a start, check for gap
            if let (Some(end), Some(start)) = (&current.entity.valid_until, &next.entity.valid_from)
            {
                if end < start {
                    return true;
                }
            }
        }
        false
    }

    /// Check if there are overlapping values.
    #[must_use]
    pub fn has_overlaps(&self) -> bool {
        if self.versions.len() < 2 {
            return false;
        }

        for i in 0..self.versions.len() - 1 {
            let current = &self.versions[i];
            let next = &self.versions[i + 1];

            // Overlap if current's end > next's start (or current has no end)
            if let Some(next_start) = &next.entity.valid_from {
                if current.entity.valid_until.is_none() {
                    return true; // Current is still valid when next starts
                }
                if let Some(curr_end) = &current.entity.valid_until {
                    if curr_end > next_start {
                        return true;
                    }
                }
            }
        }
        false
    }
}

// =============================================================================
// Temporal Entity Tracker
// =============================================================================

/// Tracks entities over time with temporal validity.
///
/// Provides point-in-time queries and evolution tracking.
#[derive(Debug, Clone, Default)]
pub struct TemporalEntityTracker {
    /// All tracked entities
    entities: Vec<Entity>,
    /// Timelines by slot/role
    timelines: HashMap<String, EntityTimeline>,
}

impl TemporalEntityTracker {
    /// Create a new tracker.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an entity to track.
    pub fn add_entity(&mut self, entity: Entity) {
        // If entity has a normalized slot, add to timeline
        if let Some(ref slot) = entity.normalized {
            let timeline = self
                .timelines
                .entry(slot.clone())
                .or_insert_with(|| EntityTimeline::new(slot));
            timeline.add(entity.clone(), None);
        }

        self.entities.push(entity);
    }

    /// Add an entity with explicit slot.
    pub fn add_entity_with_slot(&mut self, entity: Entity, slot: impl Into<String>) {
        let slot = slot.into();
        let timeline = self
            .timelines
            .entry(slot.clone())
            .or_insert_with(|| EntityTimeline::new(&slot));
        timeline.add(entity.clone(), None);

        self.entities.push(entity);
    }

    /// Query entities valid at a specific timestamp.
    #[must_use]
    pub fn query_at(&self, timestamp: &DateTime<Utc>) -> Vec<&Entity> {
        self.entities
            .iter()
            .filter(|e| e.valid_at(timestamp))
            .collect()
    }

    /// Execute a temporal query.
    #[must_use]
    pub fn query(&self, query: &TemporalQuery) -> Vec<&Entity> {
        self.entities
            .iter()
            .filter(|e| {
                // Check temporal scope
                if !query.scope.contains(e) {
                    return false;
                }

                // Check entity type
                if let Some(ref et) = query.entity_type {
                    if &e.entity_type != et {
                        return false;
                    }
                }

                // Check slot
                if let Some(ref slot) = query.slot {
                    if e.normalized.as_ref() != Some(slot) {
                        return false;
                    }
                }

                true
            })
            .collect()
    }

    /// Get the timeline for a specific slot.
    #[must_use]
    pub fn timeline(&self, slot: &str) -> Option<&EntityTimeline> {
        self.timelines.get(slot)
    }

    /// Get all known slots.
    #[must_use]
    pub fn slots(&self) -> Vec<&str> {
        self.timelines.keys().map(|s| s.as_str()).collect()
    }

    /// Get entities that changed within a time range.
    #[must_use]
    pub fn changed_between(&self, start: &DateTime<Utc>, end: &DateTime<Utc>) -> Vec<&Entity> {
        self.entities
            .iter()
            .filter(|e| {
                // Entity changed if its valid_from or valid_until is within range
                let started_in_range = e
                    .valid_from
                    .as_ref()
                    .is_some_and(|vf| vf >= start && vf <= end);
                let ended_in_range = e
                    .valid_until
                    .as_ref()
                    .is_some_and(|vu| vu >= start && vu <= end);
                started_in_range || ended_in_range
            })
            .collect()
    }

    /// Get count of temporal vs atemporal entities.
    #[must_use]
    pub fn temporal_stats(&self) -> TemporalStats {
        let mut stats = TemporalStats::default();

        for entity in &self.entities {
            stats.total += 1;
            if entity.is_temporal() {
                stats.temporal += 1;
                if entity.valid_until.is_none() {
                    stats.currently_valid += 1;
                } else {
                    stats.historical += 1;
                }
            } else {
                stats.atemporal += 1;
            }
        }

        stats
    }
}

/// Statistics about temporal entities.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TemporalStats {
    /// Total entities
    pub total: usize,
    /// Entities with temporal bounds
    pub temporal: usize,
    /// Entities without temporal bounds (timeless facts)
    pub atemporal: usize,
    /// Temporal entities that are currently valid
    pub currently_valid: usize,
    /// Temporal entities that have ended
    pub historical: usize,
}

// =============================================================================
// Temporal Alignment
// =============================================================================

/// Aligns entities across documents with different publication dates.
///
/// When processing news from different dates, the same role might have
/// different values depending on when the document was written.
#[derive(Debug, Clone)]
pub struct TemporalAligner {
    /// Document timestamp to use as reference
    pub document_date: Option<DateTime<Utc>>,
    /// Whether to infer validity from document date
    pub infer_from_document_date: bool,
    /// Default validity duration for inferred entities
    pub default_duration: Option<Duration>,
}

impl Default for TemporalAligner {
    fn default() -> Self {
        Self {
            document_date: None,
            infer_from_document_date: true,
            default_duration: None,
        }
    }
}

impl TemporalAligner {
    /// Create a new aligner for a specific document date.
    #[must_use]
    pub fn for_document(date: DateTime<Utc>) -> Self {
        Self {
            document_date: Some(date),
            infer_from_document_date: true,
            default_duration: None,
        }
    }

    /// Annotate an entity with temporal information based on document date.
    ///
    /// If the entity doesn't have temporal bounds, this can infer them
    /// from the document date.
    pub fn annotate(&self, entity: &mut Entity) {
        if !self.infer_from_document_date {
            return;
        }

        // Don't override existing temporal bounds
        if entity.is_temporal() {
            return;
        }

        // If we have a document date, use it to infer validity
        if let Some(doc_date) = &self.document_date {
            // For "current state" assertions (e.g., "X is CEO"),
            // assume valid from document date with unknown end
            entity.valid_from = Some(*doc_date);

            // If we have a default duration, set end date too
            if let Some(duration) = &self.default_duration {
                entity.valid_until = Some(*doc_date + *duration);
            }
        }
    }

    /// Align multiple entities from different document dates.
    ///
    /// Returns entities grouped by their inferred "slot" (if any).
    pub fn align(&self, entities: Vec<(Entity, DateTime<Utc>)>) -> TemporalEntityTracker {
        let mut tracker = TemporalEntityTracker::new();

        for (mut entity, doc_date) in entities {
            // Create a temporary aligner for this document
            let aligner = Self::for_document(doc_date);
            aligner.annotate(&mut entity);
            tracker.add_entity(entity);
        }

        tracker
    }
}

// =============================================================================
// Temporal Relation Types
// =============================================================================

/// Temporal relations between events/entities.
///
/// Based on Allen's interval algebra.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TemporalRelation {
    /// A is completely before B
    Before,
    /// A meets B (end of A = start of B)
    Meets,
    /// A overlaps with start of B
    Overlaps,
    /// A starts when B starts but ends before
    Starts,
    /// A is completely during B
    During,
    /// A ends when B ends but starts after
    Finishes,
    /// A is identical to B
    Equal,
    // Inverses
    /// B is completely before A
    After,
    /// B meets A
    MetBy,
    /// B overlaps with end of A
    OverlappedBy,
    /// B starts when A starts
    StartedBy,
    /// B is during A
    Contains,
    /// B finishes when A finishes
    FinishedBy,
}

impl TemporalRelation {
    /// Compute the temporal relation between two entities.
    #[must_use]
    pub fn between(a: &Entity, b: &Entity) -> Option<Self> {
        // Clone to owned values for consistent comparison
        let a_start = *a.valid_from.as_ref()?;
        let b_start = *b.valid_from.as_ref()?;

        // If either has no end, treat as ongoing
        let a_end = a.valid_until.unwrap_or_else(Utc::now);
        let b_end = b.valid_until.unwrap_or_else(Utc::now);

        // Allen's interval algebra relations
        // All comparisons now on owned DateTime<Utc> values
        if a_end < b_start {
            Some(Self::Before)
        } else if a_end == b_start {
            Some(Self::Meets)
        } else if a_start < b_start && a_end > b_start && a_end < b_end {
            Some(Self::Overlaps)
        } else if a_start == b_start && a_end < b_end {
            Some(Self::Starts)
        } else if a_start > b_start && a_end < b_end {
            Some(Self::During)
        } else if a_start > b_start && a_end == b_end {
            Some(Self::Finishes)
        } else if a_start == b_start && a_end == b_end {
            Some(Self::Equal)
        } else if a_start > b_end {
            Some(Self::After)
        } else if a_start == b_end {
            Some(Self::MetBy)
        } else if b_start < a_start && b_end > a_start && b_end < a_end {
            Some(Self::OverlappedBy)
        } else if b_start == a_start && b_end > a_end {
            Some(Self::StartedBy)
        } else if b_start < a_start && b_end > a_end {
            Some(Self::Contains)
        } else if b_start < a_start && b_end == a_end {
            Some(Self::FinishedBy)
        } else {
            None
        }
    }

    /// Check if two entities are concurrent (overlap in time).
    #[must_use]
    pub fn is_concurrent(a: &Entity, b: &Entity) -> bool {
        matches!(
            Self::between(a, b),
            Some(Self::Overlaps)
                | Some(Self::Starts)
                | Some(Self::During)
                | Some(Self::Finishes)
                | Some(Self::Equal)
                | Some(Self::OverlappedBy)
                | Some(Self::StartedBy)
                | Some(Self::Contains)
                | Some(Self::FinishedBy)
        )
    }
}

// =============================================================================
// Utility Functions for Date/Time Parsing
// =============================================================================

/// Normalize a date string to ISO 8601 format (YYYY-MM-DD).
/// Returns None if the date cannot be parsed.
#[must_use]
pub fn normalize_date(text: &str) -> Option<String> {
    let text = text.trim();

    // Try Japanese format: YYYY年MM月DD日
    if let Some(date) = parse_japanese_date(text) {
        return Some(date.format("%Y-%m-%d").to_string());
    }

    // Try EU dot format: DD.MM.YYYY
    if let Some(date) = parse_eu_dot_date(text) {
        return Some(date.format("%Y-%m-%d").to_string());
    }

    // Try common date formats
    let formats = [
        "%Y-%m-%d",  // 2024-01-15
        "%Y/%m/%d",  // 2024/01/15
        "%d-%m-%Y",  // 15-01-2024
        "%d/%m/%Y",  // 15/01/2024
        "%B %d, %Y", // January 15, 2024
        "%b %d, %Y", // Jan 15, 2024
        "%d %B %Y",  // 15 January 2024
        "%d %b %Y",  // 15 Jan 2024
        "%m/%d/%Y",  // 01/15/2024 (US format)
    ];

    for fmt in &formats {
        if let Ok(date) = NaiveDate::parse_from_str(text, fmt) {
            return Some(date.format("%Y-%m-%d").to_string());
        }
    }

    // Try year-only
    if let Ok(year) = text.parse::<i32>() {
        if (1000..=2100).contains(&year) {
            return Some(format!("{year}-01-01"));
        }
    }

    None
}

/// Parse a date string into a `DateTime<Utc>`.
/// Returns None if the date cannot be parsed.
#[must_use]
pub fn parse_date(text: &str) -> Option<DateTime<Utc>> {
    let text = text.trim();

    // Try Japanese format: YYYY年MM月DD日
    if let Some(date) = parse_japanese_date(text) {
        if let Some(dt) = date.and_hms_opt(0, 0, 0) {
            return Some(Utc.from_utc_datetime(&dt));
        }
    }

    // Try EU dot format: DD.MM.YYYY
    if let Some(date) = parse_eu_dot_date(text) {
        if let Some(dt) = date.and_hms_opt(0, 0, 0) {
            return Some(Utc.from_utc_datetime(&dt));
        }
    }

    let formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%d %B %Y",
        "%d %b %Y",
        "%m/%d/%Y",
    ];

    for fmt in &formats {
        if let Ok(date) = NaiveDate::parse_from_str(text, fmt) {
            if let Some(dt) = date.and_hms_opt(0, 0, 0) {
                return Some(Utc.from_utc_datetime(&dt));
            }
        }
    }

    // Try year-only
    if let Ok(year) = text.parse::<i32>() {
        if (1000..=2100).contains(&year) {
            if let Some(date) = NaiveDate::from_ymd_opt(year, 1, 1) {
                if let Some(dt) = date.and_hms_opt(0, 0, 0) {
                    return Some(Utc.from_utc_datetime(&dt));
                }
            }
        }
    }

    None
}

/// Parse Japanese date format: YYYY年MM月DD日
fn parse_japanese_date(text: &str) -> Option<NaiveDate> {
    // Match pattern: digits + 年 + digits + 月 + digits + 日
    let text = text.trim();

    // Find the year part (before 年)
    let year_end = text.find('年')?;
    let year: i32 = text[..year_end].parse().ok()?;

    // Find the month part (between 年 and 月)
    let month_start = year_end + '年'.len_utf8();
    let month_end = text[month_start..].find('月')? + month_start;
    let month: u32 = text[month_start..month_end].parse().ok()?;

    // Find the day part (between 月 and 日)
    let day_start = month_end + '月'.len_utf8();
    let day_end = text[day_start..].find('日')? + day_start;
    let day: u32 = text[day_start..day_end].parse().ok()?;

    NaiveDate::from_ymd_opt(year, month, day)
}

/// Parse EU dot format: DD.MM.YYYY
fn parse_eu_dot_date(text: &str) -> Option<NaiveDate> {
    let parts: Vec<&str> = text.split('.').collect();
    if parts.len() != 3 {
        return None;
    }

    let day: u32 = parts[0].parse().ok()?;
    let month: u32 = parts[1].parse().ok()?;
    let year: i32 = parts[2].parse().ok()?;

    // Handle 2-digit years
    let year = if year < 100 {
        if year > 50 {
            1900 + year
        } else {
            2000 + year
        }
    } else {
        year
    };

    NaiveDate::from_ymd_opt(year, month, day)
}

/// Normalize a time string to ISO 8601 format (HH:MM).
/// Returns None if the time cannot be parsed.
#[must_use]
pub fn normalize_time(text: &str) -> Option<String> {
    let text = text.trim().to_uppercase();

    // Handle 12-hour format with AM/PM
    let (time_part, is_pm) = if text.ends_with("PM") {
        (text.trim_end_matches("PM").trim(), true)
    } else if text.ends_with("AM") {
        (text.trim_end_matches("AM").trim(), false)
    } else {
        (text.as_str(), false)
    };

    // Parse HH:MM or HH:MM:SS
    let parts: Vec<&str> = time_part.split(':').collect();
    match parts.len() {
        2 => {
            let hour: u32 = parts[0].parse().ok()?;
            let min: u32 = parts[1].parse().ok()?;
            let adjusted_hour = if is_pm && hour != 12 {
                hour + 12
            } else if !is_pm && hour == 12 {
                0
            } else {
                hour
            };
            if adjusted_hour < 24 && min < 60 {
                return Some(format!("{adjusted_hour:02}:{min:02}"));
            }
        }
        3 => {
            let hour: u32 = parts[0].parse().ok()?;
            let min: u32 = parts[1].parse().ok()?;
            let sec: u32 = parts[2].parse().ok()?;
            let adjusted_hour = if is_pm && hour != 12 {
                hour + 12
            } else if !is_pm && hour == 12 {
                0
            } else {
                hour
            };
            if adjusted_hour < 24 && min < 60 && sec < 60 {
                // Return HH:MM format (without seconds) for consistency
                return Some(format!("{adjusted_hour:02}:{min:02}"));
            }
        }
        _ => {}
    }

    None
}

// =============================================================================
// Abstract Temporal Expressions
// =============================================================================

/// Granularity of a temporal expression.
///
/// Temporal expressions exist at different levels of specificity,
/// analogous to how entities exist at different levels of abstraction
/// in a hierarchical clustering layer (currently archived out of the main workspace).
///
/// ```text
/// ┌─────────────────────────────────────────────────────────────────────┐
/// │                    TEMPORAL GRANULARITY HIERARCHY                   │
/// ├─────────────────────────────────────────────────────────────────────┤
/// │                                                                     │
/// │  Level 0: Instant      "2024-01-15T14:30:00Z"                       │
/// │  Level 1: Day          "January 15, 2024"                           │
/// │  Level 2: Week         "the week of Jan 15"                         │
/// │  Level 3: Month        "January 2024"                               │
/// │  Level 4: Quarter      "Q1 2024"                                    │
/// │  Level 5: Year         "2024"                                       │
/// │  Level 6: Decade       "the 2020s"                                  │
/// │  Level 7: Century      "21st century"                               │
/// │  Level 8: Era          "modern era", "post-WWII"                    │
/// │                                                                     │
/// └─────────────────────────────────────────────────────────────────────┘
/// ```
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default,
)]
pub enum TemporalGranularity {
    /// Precise timestamp with time component
    Instant,
    /// Single day
    Day,
    /// Week (7-day period)
    Week,
    /// Calendar month
    Month,
    /// Fiscal/calendar quarter
    Quarter,
    /// Calendar year
    Year,
    /// Decade (e.g., "the 90s")
    Decade,
    /// Century (e.g., "19th century")
    Century,
    /// Historical era (e.g., "Renaissance", "Cold War")
    Era,
    /// Unknown or unspecified granularity
    #[default]
    Unknown,
}

impl TemporalGranularity {
    /// Get the numeric level (0 = most specific, higher = more abstract).
    #[must_use]
    pub fn level(&self) -> u8 {
        match self {
            Self::Instant => 0,
            Self::Day => 1,
            Self::Week => 2,
            Self::Month => 3,
            Self::Quarter => 4,
            Self::Year => 5,
            Self::Decade => 6,
            Self::Century => 7,
            Self::Era => 8,
            Self::Unknown => 255,
        }
    }

    /// Can this granularity be converted to a concrete DateTime?
    #[must_use]
    pub fn is_groundable(&self) -> bool {
        matches!(
            self,
            Self::Instant | Self::Day | Self::Week | Self::Month | Self::Quarter | Self::Year
        )
    }
}

/// Type of temporal expression based on how it relates to absolute time.
///
/// ```text
/// ┌─────────────────────────────────────────────────────────────────────┐
/// │                    TEMPORAL EXPRESSION TYPES                        │
/// ├─────────────────────────────────────────────────────────────────────┤
/// │                                                                     │
/// │  Absolute:    "January 15, 2024"     → directly mappable            │
/// │  Relative:    "yesterday", "next week" → needs document date        │
/// │  Anchored:    "before the war"       → needs event reference        │
/// │  Recurring:   "every Monday"         → pattern, not single point    │
/// │  Fuzzy:       "recently", "soon"     → vague, probabilistic         │
/// │  Partial:     "in the morning"       → missing date component       │
/// │                                                                     │
/// └─────────────────────────────────────────────────────────────────────┘
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TemporalExpressionType {
    /// Directly maps to calendar time: "2024-01-15", "January 2024"
    Absolute,
    /// Relative to document/utterance time: "yesterday", "next week", "3 days ago"
    Relative {
        /// Direction from anchor (negative = past, positive = future)
        offset_days: i32,
        /// The reference point (if known)
        anchor: Option<Box<DateTime<Utc>>>,
    },
    /// Anchored to an event rather than calendar: "before the war", "after graduation"
    EventAnchored {
        /// The anchor event description
        event: String,
        /// Temporal relation to the event
        relation: TemporalRelation,
    },
    /// Recurring pattern: "every Monday", "annually", "on weekends"
    Recurring {
        /// Pattern description
        pattern: String,
        /// Frequency (if extractable)
        frequency: Option<RecurrenceFrequency>,
    },
    /// Fuzzy/vague: "recently", "soon", "in the past", "long ago"
    Fuzzy {
        /// Direction (past/future/unknown)
        direction: FuzzyDirection,
        /// Approximate distance (if inferable)
        approximate_days: Option<(i32, i32)>, // (min, max) range
    },
    /// Partial specification: "in the morning", "on Tuesday" (missing year/date)
    Partial {
        /// What components are specified
        specified: PartialTimeComponents,
    },
}

/// Direction for fuzzy temporal expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FuzzyDirection {
    /// Past: "recently", "long ago"
    Past,
    /// Future: "soon", "eventually"
    Future,
    /// Unknown/either: "sometime"
    Unknown,
}

/// Recurrence frequency for recurring patterns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecurrenceFrequency {
    /// Daily
    Daily,
    /// Weekly (specific day)
    Weekly,
    /// Biweekly
    Biweekly,
    /// Monthly
    Monthly,
    /// Quarterly
    Quarterly,
    /// Annually
    Annually,
    /// Custom/irregular
    Custom,
}

/// Components specified in a partial temporal expression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct PartialTimeComponents {
    /// Year specified
    pub year: bool,
    /// Month specified
    pub month: bool,
    /// Day specified
    pub day: bool,
    /// Day of week specified
    pub weekday: bool,
    /// Hour specified
    pub hour: bool,
    /// Minute specified
    pub minute: bool,
}

/// An abstract temporal expression with full metadata.
///
/// This is the temporal analog to abstract entities in a hierarchical clustering layer -
/// it captures not just when something happened, but how precisely
/// we know when, and what kind of temporal reference it is.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbstractTemporalExpression {
    /// Original text of the temporal expression
    pub text: String,
    /// Type of temporal expression
    pub expression_type: TemporalExpressionType,
    /// Granularity level
    pub granularity: TemporalGranularity,
    /// Grounded time range (if resolvable)
    /// For "January 2024", this would be (2024-01-01, 2024-01-31)
    pub grounded_range: Option<(DateTime<Utc>, DateTime<Utc>)>,
    /// Confidence in the grounding (0.0 to 1.0)
    pub grounding_confidence: Confidence,
    /// Whether this requires external context to resolve
    pub requires_context: bool,
}

impl AbstractTemporalExpression {
    /// Create a new absolute temporal expression.
    #[must_use]
    pub fn absolute(text: impl Into<String>, granularity: TemporalGranularity) -> Self {
        Self {
            text: text.into(),
            expression_type: TemporalExpressionType::Absolute,
            granularity,
            grounded_range: None,
            grounding_confidence: Confidence::ONE,
            requires_context: false,
        }
    }

    /// Create a relative temporal expression.
    #[must_use]
    pub fn relative(text: impl Into<String>, offset_days: i32) -> Self {
        Self {
            text: text.into(),
            expression_type: TemporalExpressionType::Relative {
                offset_days,
                anchor: None,
            },
            granularity: TemporalGranularity::Day,
            grounded_range: None,
            grounding_confidence: Confidence::ZERO, // Needs grounding
            requires_context: true,
        }
    }

    /// Create a fuzzy temporal expression.
    #[must_use]
    pub fn fuzzy(text: impl Into<String>, direction: FuzzyDirection) -> Self {
        Self {
            text: text.into(),
            expression_type: TemporalExpressionType::Fuzzy {
                direction,
                approximate_days: None,
            },
            granularity: TemporalGranularity::Unknown,
            grounded_range: None,
            grounding_confidence: Confidence::ZERO,
            requires_context: true,
        }
    }

    /// Ground this expression relative to a document date.
    ///
    /// For relative expressions like "yesterday", this resolves to an absolute time.
    #[must_use]
    pub fn ground(&self, document_date: &DateTime<Utc>) -> Option<Self> {
        let mut grounded = self.clone();

        match &self.expression_type {
            TemporalExpressionType::Relative { offset_days, .. } => {
                let target = *document_date + Duration::days(i64::from(*offset_days));
                let start = target
                    .date_naive()
                    .and_hms_opt(0, 0, 0)
                    .map(|dt| Utc.from_utc_datetime(&dt))?;
                let end = target
                    .date_naive()
                    .and_hms_opt(23, 59, 59)
                    .map(|dt| Utc.from_utc_datetime(&dt))?;

                grounded.grounded_range = Some((start, end));
                grounded.grounding_confidence = Confidence::new(0.95);
                grounded.requires_context = false;
                Some(grounded)
            }
            TemporalExpressionType::Fuzzy {
                direction,
                approximate_days,
            } => {
                // For fuzzy expressions, create a probabilistic range
                let (min_days, max_days) = approximate_days.unwrap_or(match direction {
                    FuzzyDirection::Past => (-365, -1),
                    FuzzyDirection::Future => (1, 365),
                    FuzzyDirection::Unknown => (-365, 365),
                });

                let start = *document_date + Duration::days(i64::from(min_days));
                let end = *document_date + Duration::days(i64::from(max_days));

                grounded.grounded_range = Some((start, end));
                grounded.grounding_confidence = Confidence::new(0.3); // Low confidence for fuzzy
                grounded.requires_context = false;
                Some(grounded)
            }
            _ => Some(grounded), // Already absolute or not groundable
        }
    }

    /// Check if this expression overlaps with another.
    #[must_use]
    pub fn overlaps(&self, other: &Self) -> bool {
        match (&self.grounded_range, &other.grounded_range) {
            (Some((s1, e1)), Some((s2, e2))) => s1 <= e2 && s2 <= e1,
            _ => false, // Can't determine overlap without grounded ranges
        }
    }

    /// Get the midpoint of this temporal expression (if grounded).
    #[must_use]
    pub fn midpoint(&self) -> Option<DateTime<Utc>> {
        self.grounded_range
            .map(|(start, end)| start + Duration::seconds((end - start).num_seconds() / 2))
    }
}

/// Parse a temporal expression and determine its type and granularity.
///
/// This is a lightweight parser for common patterns. For production use,
/// consider integrating with SUTime, HeidelTime, or similar.
#[must_use]
pub fn parse_temporal_expression(text: &str) -> AbstractTemporalExpression {
    let text_lower = text.to_lowercase();
    let text_trimmed = text.trim();

    // Check for relative expressions
    if let Some(expr) = parse_relative_expression(&text_lower) {
        return expr;
    }

    // Check for fuzzy expressions
    if let Some(expr) = parse_fuzzy_expression(&text_lower) {
        return expr;
    }

    // Check for recurring patterns
    if let Some(expr) = parse_recurring_expression(&text_lower) {
        return expr;
    }

    // Try to parse as absolute date and determine granularity
    if let Some(_normalized) = normalize_date(text_trimmed) {
        let granularity = infer_granularity(text_trimmed);
        let mut expr = AbstractTemporalExpression::absolute(text_trimmed, granularity);

        // Try to ground it
        if let Some(dt) = parse_date(text_trimmed) {
            let (start, end) = granularity_to_range(&dt, granularity);
            expr.grounded_range = Some((start, end));
        }

        return expr;
    }

    // Fallback: unknown expression
    AbstractTemporalExpression {
        text: text_trimmed.to_string(),
        expression_type: TemporalExpressionType::Partial {
            specified: PartialTimeComponents::default(),
        },
        granularity: TemporalGranularity::Unknown,
        grounded_range: None,
        grounding_confidence: Confidence::ZERO,
        requires_context: true,
    }
}

fn parse_relative_expression(text: &str) -> Option<AbstractTemporalExpression> {
    let patterns = [
        ("yesterday", -1),
        ("today", 0),
        ("tomorrow", 1),
        ("day before yesterday", -2),
        ("day after tomorrow", 2),
    ];

    for (pattern, offset) in patterns {
        if text.contains(pattern) {
            return Some(AbstractTemporalExpression::relative(text, offset));
        }
    }

    // Check for "N days ago" / "in N days"
    if text.contains("ago") {
        if let Some(n) = extract_number(text) {
            if text.contains("day") {
                return Some(AbstractTemporalExpression::relative(text, -(n as i32)));
            } else if text.contains("week") {
                return Some(AbstractTemporalExpression::relative(text, -(n as i32) * 7));
            } else if text.contains("month") {
                return Some(AbstractTemporalExpression::relative(text, -(n as i32) * 30));
            }
        }
    }

    if text.starts_with("in ") || text.starts_with("next ") {
        if let Some(n) = extract_number(text) {
            if text.contains("day") {
                return Some(AbstractTemporalExpression::relative(text, n as i32));
            } else if text.contains("week") {
                return Some(AbstractTemporalExpression::relative(text, n as i32 * 7));
            }
        }
        // "next week", "next month"
        if text.contains("week") {
            return Some(AbstractTemporalExpression::relative(text, 7));
        }
        if text.contains("month") {
            return Some(AbstractTemporalExpression::relative(text, 30));
        }
    }

    if text.starts_with("last ") {
        if text.contains("week") {
            return Some(AbstractTemporalExpression::relative(text, -7));
        }
        if text.contains("month") {
            return Some(AbstractTemporalExpression::relative(text, -30));
        }
    }

    None
}

fn parse_fuzzy_expression(text: &str) -> Option<AbstractTemporalExpression> {
    let past_patterns = ["recently", "lately", "long ago", "in the past", "earlier"];
    let future_patterns = ["soon", "eventually", "in the future", "later"];

    for pattern in past_patterns {
        if text.contains(pattern) {
            return Some(AbstractTemporalExpression::fuzzy(
                text,
                FuzzyDirection::Past,
            ));
        }
    }

    for pattern in future_patterns {
        if text.contains(pattern) {
            return Some(AbstractTemporalExpression::fuzzy(
                text,
                FuzzyDirection::Future,
            ));
        }
    }

    if text.contains("sometime") || text.contains("someday") {
        return Some(AbstractTemporalExpression::fuzzy(
            text,
            FuzzyDirection::Unknown,
        ));
    }

    None
}

fn parse_recurring_expression(text: &str) -> Option<AbstractTemporalExpression> {
    let frequency = if text.contains("daily") || text.contains("every day") {
        Some(RecurrenceFrequency::Daily)
    } else if text.contains("weekly") || text.contains("every week") {
        Some(RecurrenceFrequency::Weekly)
    } else if text.contains("monthly") || text.contains("every month") {
        Some(RecurrenceFrequency::Monthly)
    } else if text.contains("annually") || text.contains("every year") || text.contains("yearly") {
        Some(RecurrenceFrequency::Annually)
    } else if text.starts_with("every ") || text.starts_with("on ") && text.contains("s") {
        // "every Monday", "on Mondays"
        Some(RecurrenceFrequency::Weekly)
    } else {
        None
    };

    frequency.map(|freq| AbstractTemporalExpression {
        text: text.to_string(),
        expression_type: TemporalExpressionType::Recurring {
            pattern: text.to_string(),
            frequency: Some(freq),
        },
        granularity: TemporalGranularity::Unknown,
        grounded_range: None,
        grounding_confidence: Confidence::ZERO,
        requires_context: true,
    })
}

fn extract_number(text: &str) -> Option<u32> {
    // Simple number extraction
    for word in text.split_whitespace() {
        if let Ok(n) = word.parse::<u32>() {
            return Some(n);
        }
    }
    // Word numbers
    let word_numbers = [
        ("one", 1),
        ("two", 2),
        ("three", 3),
        ("four", 4),
        ("five", 5),
        ("six", 6),
        ("seven", 7),
        ("eight", 8),
        ("nine", 9),
        ("ten", 10),
    ];
    for (word, n) in word_numbers {
        if text.contains(word) {
            return Some(n);
        }
    }
    None
}

fn infer_granularity(text: &str) -> TemporalGranularity {
    // Check for time component
    if text.contains(':') || text.contains("am") || text.contains("pm") {
        return TemporalGranularity::Instant;
    }

    // Check for day-level precision
    if text.chars().filter(|c| c.is_ascii_digit()).count() >= 6 {
        // Has enough digits for YYYY-MM-DD or similar
        return TemporalGranularity::Day;
    }

    // Check for month-level patterns
    let months = [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
    ];
    let text_lower = text.to_lowercase();

    for month in months {
        if text_lower.contains(month) {
            // If there's a day number too, it's Day granularity
            if text.chars().filter(|c| c.is_ascii_digit()).count() >= 2 {
                // Has day number
                let digits: String = text.chars().filter(|c| c.is_ascii_digit()).collect();
                if digits.len() <= 4 {
                    // Just year or just day
                    if digits.len() == 4 {
                        return TemporalGranularity::Month;
                    }
                }
                return TemporalGranularity::Day;
            }
            return TemporalGranularity::Month;
        }
    }

    // Check for quarter
    if text_lower.contains("q1")
        || text_lower.contains("q2")
        || text_lower.contains("q3")
        || text_lower.contains("q4")
    {
        return TemporalGranularity::Quarter;
    }

    // Check for century (before decade, since "21st century" contains digits)
    if text_lower.contains("century") {
        return TemporalGranularity::Century;
    }

    // Check for decade (e.g., "1990s", "the 90s")
    if text_lower.contains("'s") || text_lower.ends_with("0s") {
        if let Ok(decade) = text
            .chars()
            .filter(|c| c.is_ascii_digit())
            .collect::<String>()
            .parse::<u32>()
        {
            if decade < 100 || (1900..=2100).contains(&decade) {
                return TemporalGranularity::Decade;
            }
        }
    }

    // Check for era
    if text_lower.contains("era") || text_lower.contains("age") || text_lower.contains("period") {
        return TemporalGranularity::Era;
    }

    // Default: if it's just a 4-digit year
    if text.chars().filter(|c| c.is_ascii_digit()).count() == 4 {
        return TemporalGranularity::Year;
    }

    TemporalGranularity::Unknown
}

fn granularity_to_range(
    dt: &DateTime<Utc>,
    granularity: TemporalGranularity,
) -> (DateTime<Utc>, DateTime<Utc>) {
    use chrono::Datelike;

    let date = dt.date_naive();

    match granularity {
        TemporalGranularity::Instant => (*dt, *dt),
        TemporalGranularity::Day => {
            let start = date
                .and_hms_opt(0, 0, 0)
                .map(|d| Utc.from_utc_datetime(&d))
                .unwrap_or(*dt);
            let end = date
                .and_hms_opt(23, 59, 59)
                .map(|d| Utc.from_utc_datetime(&d))
                .unwrap_or(*dt);
            (start, end)
        }
        TemporalGranularity::Week => {
            let weekday = date.weekday().num_days_from_monday();
            let start_date = date - Duration::days(i64::from(weekday));
            let end_date = start_date + Duration::days(6);
            let start = start_date
                .and_hms_opt(0, 0, 0)
                .map(|d| Utc.from_utc_datetime(&d))
                .unwrap_or(*dt);
            let end = end_date
                .and_hms_opt(23, 59, 59)
                .map(|d| Utc.from_utc_datetime(&d))
                .unwrap_or(*dt);
            (start, end)
        }
        TemporalGranularity::Month => {
            let start_date = NaiveDate::from_ymd_opt(date.year(), date.month(), 1).unwrap_or(date);
            let end_date = if date.month() == 12 {
                NaiveDate::from_ymd_opt(date.year() + 1, 1, 1).unwrap_or(date) - Duration::days(1)
            } else {
                NaiveDate::from_ymd_opt(date.year(), date.month() + 1, 1).unwrap_or(date)
                    - Duration::days(1)
            };
            let start = start_date
                .and_hms_opt(0, 0, 0)
                .map(|d| Utc.from_utc_datetime(&d))
                .unwrap_or(*dt);
            let end = end_date
                .and_hms_opt(23, 59, 59)
                .map(|d| Utc.from_utc_datetime(&d))
                .unwrap_or(*dt);
            (start, end)
        }
        TemporalGranularity::Quarter => {
            let quarter = (date.month() - 1) / 3;
            let start_month = quarter * 3 + 1;
            let end_month = start_month + 2;
            let start_date = NaiveDate::from_ymd_opt(date.year(), start_month, 1).unwrap_or(date);
            let end_date = if end_month == 12 {
                NaiveDate::from_ymd_opt(date.year(), 12, 31).unwrap_or(date)
            } else {
                NaiveDate::from_ymd_opt(date.year(), end_month + 1, 1).unwrap_or(date)
                    - Duration::days(1)
            };
            let start = start_date
                .and_hms_opt(0, 0, 0)
                .map(|d| Utc.from_utc_datetime(&d))
                .unwrap_or(*dt);
            let end = end_date
                .and_hms_opt(23, 59, 59)
                .map(|d| Utc.from_utc_datetime(&d))
                .unwrap_or(*dt);
            (start, end)
        }
        TemporalGranularity::Year => {
            let start_date = NaiveDate::from_ymd_opt(date.year(), 1, 1).unwrap_or(date);
            let end_date = NaiveDate::from_ymd_opt(date.year(), 12, 31).unwrap_or(date);
            let start = start_date
                .and_hms_opt(0, 0, 0)
                .map(|d| Utc.from_utc_datetime(&d))
                .unwrap_or(*dt);
            let end = end_date
                .and_hms_opt(23, 59, 59)
                .map(|d| Utc.from_utc_datetime(&d))
                .unwrap_or(*dt);
            (start, end)
        }
        TemporalGranularity::Decade => {
            let decade_start = (date.year() / 10) * 10;
            let start_date = NaiveDate::from_ymd_opt(decade_start, 1, 1).unwrap_or(date);
            let end_date = NaiveDate::from_ymd_opt(decade_start + 9, 12, 31).unwrap_or(date);
            let start = start_date
                .and_hms_opt(0, 0, 0)
                .map(|d| Utc.from_utc_datetime(&d))
                .unwrap_or(*dt);
            let end = end_date
                .and_hms_opt(23, 59, 59)
                .map(|d| Utc.from_utc_datetime(&d))
                .unwrap_or(*dt);
            (start, end)
        }
        TemporalGranularity::Century => {
            let century_start = (date.year() / 100) * 100;
            let start_date = NaiveDate::from_ymd_opt(century_start, 1, 1).unwrap_or(date);
            let end_date = NaiveDate::from_ymd_opt(century_start + 99, 12, 31).unwrap_or(date);
            let start = start_date
                .and_hms_opt(0, 0, 0)
                .map(|d| Utc.from_utc_datetime(&d))
                .unwrap_or(*dt);
            let end = end_date
                .and_hms_opt(23, 59, 59)
                .map(|d| Utc.from_utc_datetime(&d))
                .unwrap_or(*dt);
            (start, end)
        }
        TemporalGranularity::Era | TemporalGranularity::Unknown => {
            // Can't determine bounds for era/unknown
            (*dt, *dt)
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn make_entity(text: &str, from: DateTime<Utc>, until: Option<DateTime<Utc>>) -> Entity {
        let mut e = Entity::new(text, EntityType::Person, 0, text.len(), 0.9);
        e.valid_from = Some(from);
        e.valid_until = until;
        e
    }

    #[test]
    fn test_point_in_time_query() {
        let mut tracker = TemporalEntityTracker::new();

        let ballmer = make_entity(
            "Steve Ballmer",
            Utc.with_ymd_and_hms(2000, 1, 13, 0, 0, 0).unwrap(),
            Some(Utc.with_ymd_and_hms(2014, 2, 4, 0, 0, 0).unwrap()),
        );
        tracker.add_entity(ballmer);

        let nadella = make_entity(
            "Satya Nadella",
            Utc.with_ymd_and_hms(2014, 2, 4, 0, 0, 0).unwrap(),
            None,
        );
        tracker.add_entity(nadella);

        // 2010: Should get Ballmer
        let query_2010 = Utc.with_ymd_and_hms(2010, 6, 1, 0, 0, 0).unwrap();
        let result = tracker.query_at(&query_2010);
        assert_eq!(result.len(), 1);
        assert!(result[0].text.contains("Ballmer"));

        // 2020: Should get Nadella
        let query_2020 = Utc.with_ymd_and_hms(2020, 6, 1, 0, 0, 0).unwrap();
        let result = tracker.query_at(&query_2020);
        assert_eq!(result.len(), 1);
        assert!(result[0].text.contains("Nadella"));
    }

    #[test]
    fn test_entity_timeline() {
        let mut timeline = EntityTimeline::new("CEO_OF_MICROSOFT");

        let mut ballmer = make_entity(
            "Steve Ballmer",
            Utc.with_ymd_and_hms(2000, 1, 13, 0, 0, 0).unwrap(),
            Some(Utc.with_ymd_and_hms(2014, 2, 4, 0, 0, 0).unwrap()),
        );
        ballmer.normalized = Some("CEO_OF_MICROSOFT".into());
        timeline.add(ballmer, None);

        let mut nadella = make_entity(
            "Satya Nadella",
            Utc.with_ymd_and_hms(2014, 2, 4, 0, 0, 0).unwrap(),
            None,
        );
        nadella.normalized = Some("CEO_OF_MICROSOFT".into());
        timeline.add(nadella, None);

        // Check historical values
        assert_eq!(timeline.history().len(), 2);

        // Check current value
        let current = timeline.current();
        assert!(current.is_some());
        assert!(current.unwrap().text.contains("Nadella"));

        // Check value at specific time
        let query_2012 = Utc.with_ymd_and_hms(2012, 1, 1, 0, 0, 0).unwrap();
        let val_2012 = timeline.value_at(&query_2012);
        assert!(val_2012.is_some());
        assert!(val_2012.unwrap().text.contains("Ballmer"));
    }

    #[test]
    fn test_temporal_scope() {
        let entity = make_entity(
            "Test",
            Utc.with_ymd_and_hms(2010, 1, 1, 0, 0, 0).unwrap(),
            Some(Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap()),
        );

        // Point in time - within range
        let scope = TemporalScope::PointInTime(Utc.with_ymd_and_hms(2015, 1, 1, 0, 0, 0).unwrap());
        assert!(scope.contains(&entity));

        // Point in time - before range
        let scope = TemporalScope::PointInTime(Utc.with_ymd_and_hms(2005, 1, 1, 0, 0, 0).unwrap());
        assert!(!scope.contains(&entity));

        // Range - overlapping
        let scope = TemporalScope::Range {
            start: Utc.with_ymd_and_hms(2008, 1, 1, 0, 0, 0).unwrap(),
            end: Utc.with_ymd_and_hms(2012, 1, 1, 0, 0, 0).unwrap(),
        };
        assert!(scope.contains(&entity));

        // Current - entity has end date
        let scope = TemporalScope::Current;
        assert!(!scope.contains(&entity));
    }

    #[test]
    fn test_temporal_relation() {
        let a = make_entity(
            "A",
            Utc.with_ymd_and_hms(2010, 1, 1, 0, 0, 0).unwrap(),
            Some(Utc.with_ymd_and_hms(2015, 1, 1, 0, 0, 0).unwrap()),
        );
        let b = make_entity(
            "B",
            Utc.with_ymd_and_hms(2016, 1, 1, 0, 0, 0).unwrap(),
            Some(Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap()),
        );

        // A ends before B starts
        let rel = TemporalRelation::between(&a, &b);
        assert_eq!(rel, Some(TemporalRelation::Before));

        // B starts after A ends
        let rel = TemporalRelation::between(&b, &a);
        assert_eq!(rel, Some(TemporalRelation::After));
    }

    #[test]
    fn test_temporal_stats() {
        let mut tracker = TemporalEntityTracker::new();

        // Add temporal entity
        let temporal = make_entity(
            "Temporal",
            Utc.with_ymd_and_hms(2010, 1, 1, 0, 0, 0).unwrap(),
            Some(Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap()),
        );
        tracker.add_entity(temporal);

        // Add currently valid entity
        let current = make_entity(
            "Current",
            Utc.with_ymd_and_hms(2015, 1, 1, 0, 0, 0).unwrap(),
            None,
        );
        tracker.add_entity(current);

        // Add atemporal entity
        let atemporal = Entity::new("Atemporal", EntityType::Person, 0, 9, 0.9);
        tracker.add_entity(atemporal);

        let stats = tracker.temporal_stats();
        assert_eq!(stats.total, 3);
        assert_eq!(stats.temporal, 2);
        assert_eq!(stats.atemporal, 1);
        assert_eq!(stats.currently_valid, 1);
        assert_eq!(stats.historical, 1);
    }

    #[test]
    fn test_timeline_gaps_and_overlaps() {
        let mut timeline = EntityTimeline::new("TEST");

        // Add with gap
        let e1 = make_entity(
            "E1",
            Utc.with_ymd_and_hms(2010, 1, 1, 0, 0, 0).unwrap(),
            Some(Utc.with_ymd_and_hms(2012, 1, 1, 0, 0, 0).unwrap()),
        );
        timeline.add(e1, None);

        let e2 = make_entity(
            "E2",
            Utc.with_ymd_and_hms(2015, 1, 1, 0, 0, 0).unwrap(), // Gap: 2012-2015
            Some(Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap()),
        );
        timeline.add(e2, None);

        assert!(timeline.has_gaps());
        assert!(!timeline.has_overlaps());
    }

    // =========================================================================
    // Abstract Temporal Expression Tests
    // =========================================================================

    #[test]
    fn test_granularity_ordering() {
        assert!(TemporalGranularity::Instant.level() < TemporalGranularity::Day.level());
        assert!(TemporalGranularity::Day.level() < TemporalGranularity::Month.level());
        assert!(TemporalGranularity::Month.level() < TemporalGranularity::Year.level());
        assert!(TemporalGranularity::Year.level() < TemporalGranularity::Decade.level());
    }

    #[test]
    fn test_parse_relative_expression() {
        let expr = parse_temporal_expression("yesterday");
        assert!(matches!(
            expr.expression_type,
            TemporalExpressionType::Relative {
                offset_days: -1,
                ..
            }
        ));
        assert!(expr.requires_context);

        let expr = parse_temporal_expression("tomorrow");
        assert!(matches!(
            expr.expression_type,
            TemporalExpressionType::Relative { offset_days: 1, .. }
        ));

        let expr = parse_temporal_expression("3 days ago");
        assert!(matches!(
            expr.expression_type,
            TemporalExpressionType::Relative {
                offset_days: -3,
                ..
            }
        ));
    }

    #[test]
    fn test_parse_fuzzy_expression() {
        let expr = parse_temporal_expression("recently");
        assert!(matches!(
            expr.expression_type,
            TemporalExpressionType::Fuzzy {
                direction: FuzzyDirection::Past,
                ..
            }
        ));

        let expr = parse_temporal_expression("soon");
        assert!(matches!(
            expr.expression_type,
            TemporalExpressionType::Fuzzy {
                direction: FuzzyDirection::Future,
                ..
            }
        ));

        let expr = parse_temporal_expression("sometime");
        assert!(matches!(
            expr.expression_type,
            TemporalExpressionType::Fuzzy {
                direction: FuzzyDirection::Unknown,
                ..
            }
        ));
    }

    #[test]
    fn test_parse_recurring_expression() {
        let expr = parse_temporal_expression("every Monday");
        assert!(matches!(
            expr.expression_type,
            TemporalExpressionType::Recurring { .. }
        ));

        let expr = parse_temporal_expression("daily");
        if let TemporalExpressionType::Recurring { frequency, .. } = expr.expression_type {
            assert_eq!(frequency, Some(RecurrenceFrequency::Daily));
        } else {
            panic!("Expected Recurring expression");
        }
    }

    #[test]
    fn test_ground_relative_expression() {
        use chrono::Datelike;

        let expr = AbstractTemporalExpression::relative("yesterday", -1);
        let doc_date = Utc.with_ymd_and_hms(2024, 6, 15, 12, 0, 0).unwrap();

        let grounded = expr.ground(&doc_date).unwrap();
        assert!(grounded.grounded_range.is_some());

        let (start, _end) = grounded.grounded_range.unwrap();
        assert_eq!(start.day(), 14); // June 14
    }

    #[test]
    fn test_infer_granularity() {
        assert_eq!(infer_granularity("2024-01-15"), TemporalGranularity::Day);
        assert_eq!(
            infer_granularity("January 2024"),
            TemporalGranularity::Month
        );
        assert_eq!(infer_granularity("2024"), TemporalGranularity::Year);
        assert_eq!(infer_granularity("Q1 2024"), TemporalGranularity::Quarter);
        // "21st century" contains "century" but also matches decade pattern due to "21st"
        // Fix the check order in infer_granularity to prioritize explicit keywords
        assert_eq!(
            infer_granularity("the 21st century"),
            TemporalGranularity::Century
        );
        assert_eq!(infer_granularity("the 90s"), TemporalGranularity::Decade);
    }

    #[test]
    fn test_granularity_to_range() {
        use chrono::{Datelike, Timelike};

        let dt = Utc.with_ymd_and_hms(2024, 6, 15, 12, 0, 0).unwrap();

        // Day granularity should span the whole day
        let (start, end) = granularity_to_range(&dt, TemporalGranularity::Day);
        assert_eq!(start.hour(), 0);
        assert_eq!(end.hour(), 23);

        // Month granularity should span the whole month
        let (start, end) = granularity_to_range(&dt, TemporalGranularity::Month);
        assert_eq!(start.day(), 1);
        assert_eq!(end.day(), 30); // June has 30 days

        // Year granularity should span the whole year
        let (start, end) = granularity_to_range(&dt, TemporalGranularity::Year);
        assert_eq!(start.month(), 1);
        assert_eq!(end.month(), 12);
    }

    #[test]
    fn test_abstract_expression_overlap() {
        let jan_2024 = AbstractTemporalExpression {
            text: "January 2024".to_string(),
            expression_type: TemporalExpressionType::Absolute,
            granularity: TemporalGranularity::Month,
            grounded_range: Some((
                Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
                Utc.with_ymd_and_hms(2024, 1, 31, 23, 59, 59).unwrap(),
            )),
            grounding_confidence: Confidence::ONE,
            requires_context: false,
        };

        let jan_15 = AbstractTemporalExpression {
            text: "January 15, 2024".to_string(),
            expression_type: TemporalExpressionType::Absolute,
            granularity: TemporalGranularity::Day,
            grounded_range: Some((
                Utc.with_ymd_and_hms(2024, 1, 15, 0, 0, 0).unwrap(),
                Utc.with_ymd_and_hms(2024, 1, 15, 23, 59, 59).unwrap(),
            )),
            grounding_confidence: Confidence::ONE,
            requires_context: false,
        };

        let feb_2024 = AbstractTemporalExpression {
            text: "February 2024".to_string(),
            expression_type: TemporalExpressionType::Absolute,
            granularity: TemporalGranularity::Month,
            grounded_range: Some((
                Utc.with_ymd_and_hms(2024, 2, 1, 0, 0, 0).unwrap(),
                Utc.with_ymd_and_hms(2024, 2, 29, 23, 59, 59).unwrap(),
            )),
            grounding_confidence: Confidence::ONE,
            requires_context: false,
        };

        // Jan 15 is within January
        assert!(jan_2024.overlaps(&jan_15));
        assert!(jan_15.overlaps(&jan_2024));

        // January and February don't overlap
        assert!(!jan_2024.overlaps(&feb_2024));
    }
}

// =============================================================================
// Jiff Integration (Optional)
// =============================================================================

/// Jiff datetime interoperability.
///
/// When the `jiff-time` feature is enabled, this module provides conversion
/// utilities between `chrono` and `jiff` datetime types, allowing seamless
/// integration with the modern `jiff` datetime library.
///
/// # Why Jiff?
///
/// While `chrono` is the established datetime library in Rust, `jiff` offers:
/// - Better handling of civil time vs. absolute time
/// - Cleaner timezone arithmetic
/// - More ergonomic span/duration types
/// - Stricter correctness guarantees
///
/// # Example
///
/// ```rust,ignore
/// use anno::temporal::jiff_interop::{chrono_to_jiff, jiff_to_chrono};
/// use chrono::Utc;
/// use jiff::Timestamp;
///
/// let chrono_dt = Utc::now();
/// let jiff_ts = chrono_to_jiff(&chrono_dt);
///
/// // Use jiff for calculations
/// let jiff_future = jiff_ts.checked_add(jiff::Span::new().days(30)).unwrap();
///
/// // Convert back to chrono for storage
/// let chrono_future = jiff_to_chrono(&jiff_future);
/// ```
#[cfg(feature = "jiff-time")]
pub mod jiff_interop {
    use chrono::{DateTime, TimeZone, Utc};
    use jiff::{Span, Timestamp, ToSpan, Zoned};

    /// Convert a chrono `DateTime<Utc>` to a jiff `Timestamp`.
    #[must_use]
    pub fn chrono_to_jiff(dt: &DateTime<Utc>) -> Timestamp {
        Timestamp::from_second(dt.timestamp())
            .expect("chrono DateTime should be valid jiff Timestamp")
    }

    /// Convert a jiff `Timestamp` to a chrono `DateTime<Utc>`.
    #[must_use]
    pub fn jiff_to_chrono(ts: &Timestamp) -> DateTime<Utc> {
        Utc.timestamp_opt(ts.as_second(), 0)
            .single()
            .expect("jiff Timestamp should be valid chrono DateTime")
    }

    /// Convert a chrono `Duration` to a jiff `Span`.
    #[must_use]
    pub fn duration_to_span(d: &chrono::Duration) -> Span {
        d.num_seconds().seconds()
    }

    /// Convert a jiff `Span` to a chrono `Duration`.
    ///
    /// Note: This only preserves the total duration, not the civil components.
    #[must_use]
    pub fn span_to_duration(s: &Span) -> chrono::Duration {
        // Convert span to total seconds (approximate for civil spans)
        let total = s.total(jiff::Unit::Second).unwrap_or(0.0) as i64;
        chrono::Duration::seconds(total)
    }

    /// A temporal entity tracker that uses jiff internally.
    ///
    /// This provides a more ergonomic API for temporal operations while
    /// maintaining compatibility with anno's chrono-based Entity type.
    #[derive(Debug, Clone)]
    pub struct JiffTemporalTracker {
        entities: Vec<(crate::Entity, Option<Timestamp>, Option<Timestamp>)>,
    }

    impl JiffTemporalTracker {
        /// Create a new tracker.
        #[must_use]
        pub fn new() -> Self {
            Self {
                entities: Vec::new(),
            }
        }

        /// Add an entity with jiff timestamps.
        pub fn add(
            &mut self,
            entity: crate::Entity,
            from: Option<Timestamp>,
            until: Option<Timestamp>,
        ) {
            self.entities.push((entity, from, until));
        }

        /// Add an entity, converting from chrono timestamps.
        pub fn add_from_chrono(&mut self, entity: crate::Entity) {
            let from = entity.valid_from.as_ref().map(chrono_to_jiff);
            let until = entity.valid_until.as_ref().map(chrono_to_jiff);
            self.entities.push((entity, from, until));
        }

        /// Query entities valid at a jiff timestamp.
        #[must_use]
        pub fn at(&self, ts: &Timestamp) -> Vec<&crate::Entity> {
            self.entities
                .iter()
                .filter(|(_, from, until)| {
                    let after_start = from.is_none_or(|f| ts >= &f);
                    let before_end = until.is_none_or(|u| ts < &u);
                    after_start && before_end
                })
                .map(|(e, _, _)| e)
                .collect()
        }

        /// Query entities valid within a jiff span from now.
        #[must_use]
        pub fn within(&self, span: Span) -> Vec<&crate::Entity> {
            let now = Timestamp::now();
            let end = now.checked_add(span).unwrap_or(now);

            self.entities
                .iter()
                .filter(|(_, from, until)| {
                    let from = from.unwrap_or(Timestamp::MIN);
                    let until = until.unwrap_or(Timestamp::MAX);
                    // Overlap check
                    from <= end && until >= now
                })
                .map(|(e, _, _)| e)
                .collect()
        }

        /// Convert to a standard TemporalEntityTracker.
        #[must_use]
        pub fn to_chrono_tracker(&self) -> super::TemporalEntityTracker {
            let mut tracker = super::TemporalEntityTracker::new();
            for (entity, from, until) in &self.entities {
                let mut entity = entity.clone();
                entity.valid_from = from.map(|f| jiff_to_chrono(&f));
                entity.valid_until = until.map(|u| jiff_to_chrono(&u));
                tracker.add_entity(entity);
            }
            tracker
        }
    }

    impl Default for JiffTemporalTracker {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Parse a date string using jiff's flexible parser.
    ///
    /// Jiff has excellent parsing support for various date formats.
    #[must_use]
    pub fn parse_date_jiff(text: &str) -> Option<Timestamp> {
        // Try parsing as a zoned datetime first
        if let Ok(zoned) = text.parse::<Zoned>() {
            return Some(zoned.timestamp());
        }

        // Try parsing as a timestamp
        if let Ok(ts) = text.parse::<Timestamp>() {
            return Some(ts);
        }

        // Try civil date parsing
        if let Ok(date) = text.parse::<jiff::civil::Date>() {
            // Convert to timestamp at midnight UTC
            let dt = date.at(0, 0, 0, 0);
            if let Ok(ts) = dt.to_zoned(jiff::tz::TimeZone::UTC) {
                return Some(ts.timestamp());
            }
        }

        None
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::{Entity, EntityType};

        #[test]
        fn test_chrono_jiff_roundtrip() {
            let chrono_now = Utc::now();
            let jiff_ts = chrono_to_jiff(&chrono_now);
            let chrono_back = jiff_to_chrono(&jiff_ts);

            // Should be within 1 second (we lose sub-second precision)
            assert!((chrono_now - chrono_back).num_seconds().abs() < 1);
        }

        #[test]
        fn test_jiff_tracker_query() {
            let mut tracker = JiffTemporalTracker::new();

            let entity = Entity::new("Test", EntityType::Person, 0, 4, 0.9);
            let now = Timestamp::now();
            // Use hours instead of days (Timestamp doesn't support calendar units)
            let past = now.checked_sub(720.hours()).unwrap(); // ~30 days

            tracker.add(entity, Some(past), Some(now));

            // Query in the middle - should find it
            let mid = now.checked_sub(360.hours()).unwrap(); // ~15 days
            let results = tracker.at(&mid);
            assert_eq!(results.len(), 1);

            // Query in the future - should not find it
            let future = now.checked_add(360.hours()).unwrap();
            let results = tracker.at(&future);
            assert_eq!(results.len(), 0);
        }

        #[test]
        fn test_parse_date_jiff() {
            // ISO 8601
            assert!(parse_date_jiff("2024-01-15").is_some());

            // With time
            assert!(parse_date_jiff("2024-01-15T10:30:00Z").is_some());
        }
    }
}
