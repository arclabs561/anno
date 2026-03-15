//! Grounded entity representation with unified Signal → Track → Identity hierarchy.
//!
//! # Research Motivation
//!
//! Note: `anno` is text-first. The broad `Location` substrate here is intentionally future-facing.
//! See `docs/LOCATION.md` in the repo for the philosophy and practical guidance.
//!
//! Traditional NER systems conflate three distinct levels of entity processing:
//!
//! 1. **Signal Detection** (Level 1): "There's something here" - localization + classification
//! 2. **Track Formation** (Level 2): "These mentions are the same entity within this document"
//! 3. **Identity Resolution** (Level 3): "This entity is Q7186 in Wikidata"
//!
//! This conflation causes issues:
//! - Embedding models struggle when a single `Entity` type represents both mentions and KB entries
//! - Cross-document coreference requires different similarity metrics than within-document
//! - The "modal gap" between text spans and KB entities creates representation mismatches
//!
//! # The Isomorphism: Vision Detection ↔ NER
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                    VISION                    TEXT (NER)                 │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │ Localization Unit  │ BoundingBox (x,y,w,h)  │ TextSpan (start,end)     │
//! │ Signal             │ Detection              │ Mention                  │
//! │ Track (Level 2)    │ Tracklet (MOT)         │ CorefChain              │
//! │ Identity (Level 3) │ Face Recognition       │ Entity Linking          │
//! │ Region Proposal    │ RPN / DETR queries     │ Span enumeration        │
//! │ Modality           │ Iconic (physics)       │ Symbolic (convention)   │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! The key insight: **detection is modality-agnostic**. Whether detecting "Steve Jobs"
//! in text or a face in an image, the fundamental operation is:
//!
//! ```text
//! Detection = Localization (where?) × Classification (what?)
//! ```
//!
//! # Semiotic Gap: Icon vs Symbol
//!
//! A crucial nuance distinguishes text from vision:
//!
//! - **Iconic signs** (vision): The signifier physically resembles the signified.
//!   A photo of a cat looks like a cat. Detection is about physics/geometry.
//!
//! - **Symbolic signs** (text): The signifier is arbitrary convention.
//!   "cat" doesn't look like a cat. Detection requires learning cultural codes.
//!
//! This explains why text NER requires more sophisticated linguistic features
//! (negation, quantification, recursion) that have no visual analogue.
//!
//! # Architecture: Entity-Centric Representation
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                      GroundedDocument                                   │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                         │
//! │  identities: HashMap<IdentityId, Identity>                              │
//! │       │                                                                 │
//! │       └──► Identity { kb_id, canonical_name, embedding, ... }           │
//! │                 │                                                       │
//! │  tracks: HashMap<TrackId, Track<S>>                                     │
//! │       │                                                                 │
//! │       └──► Track { identity_id, signals: Vec<SignalRef>, ... }          │
//! │                 │                                                       │
//! │  signals: Vec<Signal<S>>                                                │
//! │       │                                                                 │
//! │       └──► Signal { location: S, label, confidence, ... }               │
//! │                                                                         │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! This entity-centric design enables:
//! - Efficient streaming/incremental coreference (signals → tracks incrementally)
//! - Clear separation of detection, clustering, and linking
//! - Unified treatment of text and visual signals
//!
//! # References
//!
//! - GLiNER: Bi-encoder span-label matching for zero-shot NER
//! - DETR: End-to-end object detection with transformers
//! - Pix2Seq: "Everything is a token" - bounding boxes as spatial tokens
//! - CDLKT: Cross-document Language-Knowledge Transfer
//! - Groma: Grounded multimodal assistant

use super::confidence::Confidence;
use super::entity::{
    DiscontinuousSpan, Entity, EntityType, HierarchicalConfidence, Provenance, Span,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Modality: The Semiotic Distinction
// =============================================================================

/// The semiotic modality of a signal source.
///
/// This captures a fundamental distinction in how meaning is encoded:
///
/// - **Iconic**: Physical resemblance (photos, audio waveforms)
/// - **Symbolic**: Arbitrary convention (text, notation)
/// - **Indexical**: Causal connection (smoke → fire, but rare in our domain)
///
/// # Why This Matters
///
/// The modality affects what linguistic features are relevant:
///
/// | Feature | Iconic (Vision) | Symbolic (Text) |
/// |---------|-----------------|-----------------|
/// | Negation | No analogue | "not a doctor" |
/// | Quantification | Approximate | "every/some/no" |
/// | Recursion | Rare | Nested NPs |
/// | Compositionality | Limited | Full |
///
/// Detection in iconic modalities is more about geometry and physics.
/// Detection in symbolic modalities requires cultural/linguistic knowledge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum Modality {
    /// Iconic sign: signifier resembles signified (images, audio).
    /// Detection is primarily geometric/physical.
    Iconic,
    /// Symbolic sign: arbitrary convention (text, notation).
    /// Detection requires linguistic/cultural knowledge.
    #[default]
    Symbolic,
    /// Hybrid: OCR text in images, captions, etc.
    /// Has both iconic (visual layout) and symbolic (text content) aspects.
    Hybrid,
}

// =============================================================================
// Location: The Universal Localization Unit
// =============================================================================

/// A location in text.
///
/// Two variants:
/// - `Text`: contiguous character span `[start, end)`
/// - `Discontinuous`: non-contiguous character regions
///
/// Use [`to_span()`](Self::to_span) to convert `Text` to [`entity::Span`].
///
/// [`entity::Span`]: super::entity::Span
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Location {
    /// Text span: 1D interval [start, end) in character offsets.
    Text {
        /// Start character offset (inclusive)
        start: usize,
        /// End character offset (exclusive)
        end: usize,
    },
    /// Discontinuous text span: non-contiguous regions.
    Discontinuous {
        /// Multiple text intervals
        segments: Vec<(usize, usize)>,
    },
}

impl Location {
    /// Create a text location.
    #[must_use]
    pub const fn text(start: usize, end: usize) -> Self {
        Self::Text { start, end }
    }

    /// Get the modality of this location.
    #[must_use]
    pub const fn modality(&self) -> Modality {
        match self {
            Self::Text { .. } | Self::Discontinuous { .. } => Modality::Symbolic,
        }
    }

    /// Get text offsets if this is a text location.
    #[must_use]
    pub fn text_offsets(&self) -> Option<(usize, usize)> {
        match self {
            Self::Text { start, end } => Some((*start, *end)),
            Self::Discontinuous { segments } => {
                let start = segments.iter().map(|(s, _)| *s).min()?;
                let end = segments.iter().map(|(_, e)| *e).max()?;
                Some((start, end))
            }
        }
    }

    /// Check if two locations overlap.
    #[must_use]
    pub fn overlaps(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Text { start: s1, end: e1 }, Self::Text { start: s2, end: e2 }) => {
                s1 < e2 && s2 < e1
            }
            _ => false, // Different types don't overlap
        }
    }

    /// Calculate IoU (Intersection over Union) for compatible location types.
    ///
    /// Returns None if the locations are incompatible (e.g., text vs bbox).
    #[must_use]
    pub fn iou(&self, other: &Self) -> Option<f64> {
        match (self, other) {
            (Self::Text { start: s1, end: e1 }, Self::Text { start: s2, end: e2 }) => {
                let intersection_start = (*s1).max(*s2);
                let intersection_end = (*e1).min(*e2);
                if intersection_start >= intersection_end {
                    return Some(0.0);
                }
                let intersection = (intersection_end - intersection_start) as f64;
                let union = ((*e1).max(*e2) - (*s1).min(*s2)) as f64;
                if union == 0.0 {
                    Some(0.0)
                } else {
                    Some(intersection / union)
                }
            }
            _ => None,
        }
    }
}

impl Default for Location {
    fn default() -> Self {
        Self::Text { start: 0, end: 0 }
    }
}

impl From<&Span> for Location {
    fn from(span: &Span) -> Self {
        match span {
            Span::Text { start, end } => Self::Text {
                start: *start,
                end: *end,
            },
            // BoundingBox and Hybrid spans have no Location equivalent;
            // extract text offsets where available, otherwise default.
            Span::BoundingBox { .. } => Self::Text { start: 0, end: 0 },
            Span::Hybrid { start, end, .. } => Self::Text {
                start: *start,
                end: *end,
            },
        }
    }
}

impl From<Span> for Location {
    fn from(span: Span) -> Self {
        Self::from(&span)
    }
}

/// Convert `Location` to `Span` where possible.
///
/// - `Location::Text` -> `Span::Text`
/// - `Location::Discontinuous` -> `None` (use `DiscontinuousSpan` instead)
impl Location {
    /// Try to convert this Location to a Span.
    ///
    /// Returns `None` for `Location::Discontinuous`.
    #[must_use]
    pub fn to_span(&self) -> Option<Span> {
        match self {
            Self::Text { start, end } => Some(Span::Text {
                start: *start,
                end: *end,
            }),
            Self::Discontinuous { .. } => None,
        }
    }
}

// =============================================================================
// Signal (Level 1): Raw Detection
// =============================================================================

// SignalId is now a newtype in super::types::ids for type safety
pub use super::types::SignalId;

/// A raw detection signal: the atomic unit of entity extraction.
///
/// # The Detection Equation
///
/// Every signal is the product of two factors:
///
/// ```text
/// Signal = Localization × Classification
///        = "where is it?" × "what is it?"
/// ```
///
/// This is true whether detecting faces in images, named entities in text,
/// or objects in LiDAR point clouds.
///
/// # Design Philosophy
///
/// Signals are intentionally minimal. They capture:
/// 1. **Where**: Location in the source medium
/// 2. **What**: Classification label + confidence
/// 3. **Provenance**: How it was detected
///
/// What they explicitly do NOT capture:
/// - Coreference relationships (→ Track)
/// - Knowledge base links (→ Identity)
/// - Semantic embeddings (computed lazily if needed)
///
/// This separation enables efficient streaming pipelines where signals
/// are produced incrementally and consumed by downstream track/identity
/// formation without blocking.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Signal<L = Location> {
    /// Unique identifier within the document
    pub id: SignalId,
    /// Location in the source medium
    pub location: L,
    /// Surface form (the actual text or image patch)
    pub surface: String,
    /// Classification label (e.g., "Person", "Organization", "PER").
    ///
    /// Stored as a `TypeLabel` to support both core taxonomy types and domain-specific labels.
    pub label: super::types::TypeLabel,
    /// Detection confidence in [0, 1]
    pub confidence: Confidence,
    /// Hierarchical confidence if available (linkage/type/boundary)
    pub hierarchical: Option<HierarchicalConfidence>,
    /// Provenance: which detector produced this signal
    pub provenance: Option<Provenance>,
    /// Semiotic modality (derived from location, but can be overridden)
    pub modality: Modality,
    /// Normalized form (e.g., "Jan 15" → "2024-01-15")
    pub normalized: Option<String>,
    /// Whether this signal is negated (e.g., "not a doctor")
    pub negated: bool,
    /// Quantification if applicable (e.g., "every employee")
    pub quantifier: Option<Quantifier>,
}

/// Quantification type for symbolic signals.
///
/// Only meaningful for text/symbolic modality where linguistic
/// quantification is possible. Has no visual analogue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Quantifier {
    /// Universal: "every", "all", "each"
    Universal,
    /// Existential: "some", "a", "certain"
    Existential,
    /// Negation: "no", "none"
    None,
    /// Specific: definite reference ("the")
    Definite,
    /// Approximate: "approximately", "about", "roughly"
    Approximate,
    /// Lower bound: "at least", "no fewer than"
    MinBound,
    /// Upper bound: "at most", "no more than", "up to"
    MaxBound,
    /// Bare: no explicit quantifier
    Bare,
}

impl<L> Signal<L> {
    /// Create a new signal.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier (will be overwritten when added to a document)
    /// * `location` - Where this signal was detected
    /// * `surface` - The actual text/content of the detection
    /// * `label` - Classification label (e.g., "Person", "Organization")
    /// * `confidence` - Detection confidence in `[0, 1]`
    #[must_use]
    pub fn new(
        id: impl Into<SignalId>,
        location: L,
        surface: impl Into<String>,
        label: impl Into<super::types::TypeLabel>,
        confidence: f32,
    ) -> Self {
        Self {
            id: id.into(),
            location,
            surface: surface.into(),
            label: label.into(),
            confidence: Confidence::new(confidence as f64),
            hierarchical: None,
            provenance: None,
            modality: Modality::default(),
            normalized: None,
            negated: false,
            quantifier: None,
        }
    }

    /// Get the classification label as a string.
    #[must_use]
    pub fn label(&self) -> &str {
        self.label.as_str()
    }

    /// Get the classification label as a type-safe `TypeLabel`.
    #[must_use]
    pub fn type_label(&self) -> super::types::TypeLabel {
        self.label.clone()
    }

    /// Get the surface form.
    #[must_use]
    pub fn surface(&self) -> &str {
        &self.surface
    }

    /// Check if this signal is above a confidence threshold.
    #[must_use]
    pub fn is_confident(&self, threshold: f32) -> bool {
        self.confidence >= threshold as f64
    }

    /// Set the modality.
    #[must_use]
    pub fn with_modality(mut self, modality: Modality) -> Self {
        self.modality = modality;
        self
    }

    /// Mark as negated.
    #[must_use]
    pub fn negated(mut self) -> Self {
        self.negated = true;
        self
    }

    /// Set quantifier.
    #[must_use]
    pub fn with_quantifier(mut self, q: Quantifier) -> Self {
        self.quantifier = Some(q);
        self
    }

    /// Set provenance.
    #[must_use]
    pub fn with_provenance(mut self, p: Provenance) -> Self {
        self.provenance = Some(p);
        self
    }
}

impl Signal<Location> {
    /// Get text offsets if this is a text signal.
    #[must_use]
    pub fn text_offsets(&self) -> Option<(usize, usize)> {
        self.location.text_offsets()
    }

    /// Validate that this signal's location matches its surface text.
    ///
    /// Returns `None` if valid, or a description of the mismatch.
    ///
    /// # Example
    ///
    /// ```rust
    /// use anno_core::{Signal, Location};
    ///
    /// let text = "Lynn Conway worked at IBM.";
    /// let good = Signal::new(0, Location::text(0, 11), "Lynn Conway", "PER", 0.9);
    /// assert!(good.validate_against(text).is_none());
    ///
    /// let bad = Signal::new(0, Location::text(0, 5), "Lynn Conway", "PER", 0.9);
    /// assert!(bad.validate_against(text).is_some());
    /// ```
    #[must_use]
    pub fn validate_against(&self, source_text: &str) -> Option<SignalValidationError> {
        let (start, end) = self.location.text_offsets()?;

        let char_count = source_text.chars().count();

        // Check bounds
        if end > char_count {
            return Some(SignalValidationError::OutOfBounds {
                signal_id: self.id,
                end,
                text_len: char_count,
            });
        }

        if start >= end {
            return Some(SignalValidationError::InvalidSpan {
                signal_id: self.id,
                start,
                end,
            });
        }

        // Extract actual text at offsets
        let actual: String = source_text.chars().skip(start).take(end - start).collect();

        if actual != self.surface {
            return Some(SignalValidationError::TextMismatch {
                signal_id: self.id,
                expected: self.surface.clone(),
                actual,
                start,
                end,
            });
        }

        None
    }

    /// Check if this signal is valid against the given source text.
    #[must_use]
    pub fn is_valid(&self, source_text: &str) -> bool {
        self.validate_against(source_text).is_none()
    }

    /// Create a signal by finding text in source (safe construction).
    ///
    /// Returns `None` if the surface text is not found in source.
    ///
    /// # Example
    ///
    /// ```rust
    /// use anno_core::{Signal, Location};
    ///
    /// let text = "Lynn Conway worked at IBM.";
    /// let signal = Signal::<Location>::from_text(text, "Lynn Conway", "PER", 0.95);
    /// assert!(signal.is_some());
    /// assert_eq!(signal.expect("signal should exist").text_offsets(), Some((0, 11)));
    /// ```
    #[must_use]
    pub fn from_text(
        source: &str,
        surface: &str,
        label: impl Into<super::types::TypeLabel>,
        confidence: f32,
    ) -> Option<Self> {
        Self::from_text_nth(source, surface, label, confidence, 0)
    }

    /// Create a signal by finding the nth occurrence of text in source.
    #[must_use]
    pub fn from_text_nth(
        source: &str,
        surface: &str,
        label: impl Into<super::types::TypeLabel>,
        confidence: f32,
        occurrence: usize,
    ) -> Option<Self> {
        // Find nth occurrence using char offsets
        for (count, (byte_idx, _)) in source.match_indices(surface).enumerate() {
            if count == occurrence {
                // Convert byte offset to char offset
                let start = source[..byte_idx].chars().count();
                let end = start + surface.chars().count();

                return Some(Self::new(
                    SignalId::ZERO,
                    Location::text(start, end),
                    surface,
                    label,
                    confidence,
                ));
            }
        }

        None
    }
}

/// Validation error for a signal.
#[derive(Debug, Clone, PartialEq)]
pub enum SignalValidationError {
    /// Signal's end offset exceeds text length.
    OutOfBounds {
        /// Signal ID
        signal_id: SignalId,
        /// End offset that exceeds text
        end: usize,
        /// Actual text length in chars
        text_len: usize,
    },
    /// Signal has invalid span (start >= end).
    InvalidSpan {
        /// Signal ID
        signal_id: SignalId,
        /// Start offset
        start: usize,
        /// End offset
        end: usize,
    },
    /// Signal's surface text doesn't match text at offsets.
    TextMismatch {
        /// Signal ID
        signal_id: SignalId,
        /// Surface text stored in signal
        expected: String,
        /// Actual text found at offsets
        actual: String,
        /// Start offset
        start: usize,
        /// End offset
        end: usize,
    },
}

impl std::fmt::Display for SignalValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OutOfBounds {
                signal_id,
                end,
                text_len,
            } => {
                write!(
                    f,
                    "S{}: end offset {} exceeds text length {}",
                    signal_id, end, text_len
                )
            }
            Self::InvalidSpan {
                signal_id,
                start,
                end,
            } => {
                write!(f, "S{}: invalid span [{}, {})", signal_id, start, end)
            }
            Self::TextMismatch {
                signal_id,
                expected,
                actual,
                start,
                end,
            } => {
                write!(
                    f,
                    "S{}: text mismatch at [{}, {}): expected '{}', found '{}'",
                    signal_id, start, end, expected, actual
                )
            }
        }
    }
}

impl std::error::Error for SignalValidationError {}

/// Convert an [`Entity`] to a [`Signal<Location>`].
///
/// Uses `Location::Text` for the span and preserves `normalized`, `provenance`,
/// and `hierarchical_confidence` fields. Discontinuous and visual spans are not
/// handled; use [`GroundedDocument::from_entities`] for full fidelity.
impl From<&Entity> for Signal<Location> {
    fn from(e: &Entity) -> Self {
        let mut signal = Signal::new(
            SignalId::ZERO,
            Location::text(e.start(), e.end()),
            &e.text,
            e.entity_type.as_label(),
            f32::from(e.confidence),
        );
        signal.normalized = e.normalized.clone();
        signal.provenance = e.provenance.clone();
        signal.hierarchical = e.hierarchical_confidence;
        signal
    }
}

// =============================================================================
// Track (Level 2): Within-Document Coreference
// =============================================================================

// TrackId is now a newtype in super::types::ids for type safety
pub use super::types::TrackId;

/// A reference to a signal within a track.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SignalRef {
    /// Signal ID
    pub signal_id: SignalId,
    /// Position in document order (for antecedent relationships)
    pub position: u32,
}

/// A reference to a track in a specific document.
///
/// Used for cross-document operations where we need to reference
/// tracks without copying them. This enables efficient inter-document
/// coreference resolution.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TrackRef {
    /// Document ID containing the track
    pub doc_id: String,
    /// Track ID within that document
    pub track_id: TrackId,
}

/// A track: a cluster of signals referring to the same entity within a document.
///
/// # Terminology Mapping
///
/// | Vision | NLP |
/// |--------|-----|
/// | Tracklet | CorefChain |
/// | Object track | Entity cluster |
/// | Re-identification | Coreference resolution |
///
/// # Design Philosophy
///
/// Tracks are the bridge between raw signals and global identities.
/// They answer: "which signals in THIS document refer to the same entity?"
///
/// Key properties:
/// - **Document-scoped**: A track only exists within one document
/// - **Homogeneous type**: All signals in a track should have compatible types
/// - **Representative**: The track has a "canonical" signal (usually the first proper mention)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Track {
    /// Unique identifier within the document
    pub id: TrackId,
    /// Signal references in this track (document order)
    pub signals: Vec<SignalRef>,
    /// Entity type (consensus from signals).
    ///
    /// This is a `TypeLabel` to support both core taxonomy types and domain-specific labels.
    pub entity_type: Option<super::types::TypeLabel>,
    /// Canonical surface form (the "best" name for this entity)
    pub canonical_surface: String,
    /// Link to global identity (Level 3), if resolved
    pub identity_id: Option<IdentityId>,
    /// Confidence that signals are correctly clustered
    pub cluster_confidence: Confidence,
    /// Optional embedding for track-level representation
    /// (aggregated from signal embeddings)
    pub embedding: Option<Vec<f32>>,
}

impl Track {
    /// Create a new track.
    #[must_use]
    pub fn new(id: impl Into<TrackId>, canonical_surface: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            signals: Vec::new(),
            entity_type: None,
            canonical_surface: canonical_surface.into(),
            identity_id: None,
            cluster_confidence: Confidence::ONE,
            embedding: None,
        }
    }

    /// Add a signal to this track.
    pub fn add_signal(&mut self, signal_id: impl Into<SignalId>, position: u32) {
        let signal_id = signal_id.into();
        self.signals.push(SignalRef {
            signal_id,
            position,
        });
    }

    /// Get the number of mentions in this track.
    #[must_use]
    pub fn len(&self) -> usize {
        self.signals.len()
    }

    /// Check if this track is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.signals.is_empty()
    }

    /// Check if this is a singleton (single mention).
    #[must_use]
    pub fn is_singleton(&self) -> bool {
        self.signals.len() == 1
    }

    /// Get the track's unique identifier.
    #[must_use]
    pub const fn id(&self) -> TrackId {
        self.id
    }

    /// Get the signal references in this track.
    #[must_use]
    pub fn signals(&self) -> &[SignalRef] {
        &self.signals
    }

    /// Get the canonical surface form.
    #[must_use]
    pub fn canonical_surface(&self) -> &str {
        &self.canonical_surface
    }

    /// Get the linked identity ID, if any.
    #[must_use]
    pub const fn identity_id(&self) -> Option<IdentityId> {
        self.identity_id
    }

    /// Get the cluster confidence score.
    #[must_use]
    pub const fn cluster_confidence(&self) -> Confidence {
        self.cluster_confidence
    }

    /// Set the cluster confidence score.
    pub fn set_cluster_confidence(&mut self, confidence: f32) {
        self.cluster_confidence = Confidence::new(confidence as f64);
    }

    /// Link this track to a global identity (mutable setter).
    pub fn set_identity_id(&mut self, identity_id: IdentityId) {
        self.identity_id = Some(identity_id);
    }

    /// Unlink this track from its identity.
    pub fn clear_identity_id(&mut self) {
        self.identity_id = None;
    }

    /// Link this track to a global identity.
    #[must_use]
    pub fn with_identity(mut self, identity_id: IdentityId) -> Self {
        self.identity_id = Some(identity_id);
        self
    }

    /// Set the entity type from a string.
    ///
    /// For new code, prefer [`Self::with_type_label`] which provides type safety.
    #[must_use]
    pub fn with_type(mut self, entity_type: impl Into<String>) -> Self {
        let s = entity_type.into();
        self.entity_type = Some(super::types::TypeLabel::from(s.as_str()));
        self
    }

    /// Set the entity type using a type-safe label.
    ///
    /// This is the preferred method for new code as it provides type safety
    /// and integrates with the core `EntityType` taxonomy.
    ///
    /// # Example
    ///
    /// ```rust
    /// use anno_core::{Track, TypeLabel, EntityType};
    ///
    /// let track = Track::new(0, "Marie Curie")
    ///     .with_type_label(TypeLabel::Core(EntityType::Person));
    /// ```
    #[must_use]
    pub fn with_type_label(mut self, label: super::types::TypeLabel) -> Self {
        self.entity_type = Some(label);
        self
    }

    /// Get the entity type as a type-safe label.
    ///
    /// This converts the internal string representation to a `TypeLabel`,
    /// attempting to parse it as a core `EntityType` first.
    #[must_use]
    pub fn type_label(&self) -> Option<super::types::TypeLabel> {
        self.entity_type.clone()
    }

    /// Set the embedding for this track.
    #[must_use]
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Get the spread (distance from first to last mention).
    ///
    /// Requires document to resolve signal positions.
    pub fn compute_spread(&self, doc: &GroundedDocument) -> Option<usize> {
        if self.signals.is_empty() {
            return Some(0);
        }

        let positions: Vec<usize> = self
            .signals
            .iter()
            .filter_map(|sr| {
                doc.signals
                    .iter()
                    .find(|s| s.id == sr.signal_id)
                    .and_then(|s| s.location.text_offsets())
                    .map(|(start, _)| start)
            })
            .collect();

        if positions.is_empty() {
            return None;
        }

        let min_pos = *positions.iter().min().expect("positions non-empty");
        let max_pos = *positions.iter().max().expect("positions non-empty");
        Some(max_pos.saturating_sub(min_pos))
    }

    /// Collect all surface form variations from signals.
    ///
    /// Requires document to resolve signal surfaces.
    pub fn collect_variations(&self, doc: &GroundedDocument) -> Vec<String> {
        let mut variations: std::collections::HashSet<String> = std::collections::HashSet::new();

        for sr in &self.signals {
            if let Some(signal) = doc.signals.iter().find(|s| s.id == sr.signal_id) {
                variations.insert(signal.surface.clone());
            }
        }

        variations.into_iter().collect()
    }

    /// Get confidence statistics across all signals.
    ///
    /// Returns (min, max, mean) confidence values.
    pub fn confidence_stats(&self, doc: &GroundedDocument) -> Option<(f32, f32, f32)> {
        let confidences: Vec<f32> = self
            .signals
            .iter()
            .filter_map(|sr| {
                doc.signals
                    .iter()
                    .find(|s| s.id == sr.signal_id)
                    .map(|s| s.confidence.value() as f32)
            })
            .collect();

        if confidences.is_empty() {
            return None;
        }

        let min = confidences.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = confidences
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let mean = confidences.iter().sum::<f32>() / confidences.len() as f32;

        Some((min, max, mean))
    }

    /// Compute aggregate statistics for this track.
    ///
    /// Returns a `TrackStats` struct with comprehensive aggregate features.
    pub fn compute_stats(&self, doc: &GroundedDocument, text_len: usize) -> TrackStats {
        let chain_length = self.signals.len();
        let spread = self.compute_spread(doc).unwrap_or(0);
        let variations = self.collect_variations(doc);
        let (min_conf, max_conf, mean_conf) = self.confidence_stats(doc).unwrap_or((0.0, 0.0, 0.0));

        // Compute first/last positions
        let positions: Vec<usize> = self
            .signals
            .iter()
            .filter_map(|sr| {
                doc.signals
                    .iter()
                    .find(|s| s.id == sr.signal_id)
                    .and_then(|s| s.location.text_offsets())
                    .map(|(start, _)| start)
            })
            .collect();

        let first_position = positions.iter().min().copied().unwrap_or(0);
        let last_position = positions.iter().max().copied().unwrap_or(0);
        let relative_spread = if text_len > 0 {
            spread as f64 / text_len as f64
        } else {
            0.0
        };

        TrackStats {
            chain_length,
            variation_count: variations.len(),
            variations,
            spread,
            relative_spread,
            first_position,
            last_position,
            min_confidence: min_conf,
            max_confidence: max_conf,
            mean_confidence: mean_conf,
            has_embedding: self.embedding.is_some(),
        }
    }
}

/// Aggregate statistics for a track (coreference chain).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrackStats {
    /// Number of mentions in the track.
    pub chain_length: usize,
    /// Number of unique surface form variations.
    pub variation_count: usize,
    /// All surface form variations.
    pub variations: Vec<String>,
    /// Spread in characters (first to last mention).
    pub spread: usize,
    /// Spread as fraction of document length.
    pub relative_spread: f64,
    /// Position of first mention.
    pub first_position: usize,
    /// Position of last mention.
    pub last_position: usize,
    /// Minimum confidence across mentions.
    pub min_confidence: f32,
    /// Maximum confidence across mentions.
    pub max_confidence: f32,
    /// Mean confidence across mentions.
    pub mean_confidence: f32,
    /// Whether this track has an embedding.
    pub has_embedding: bool,
}

// =============================================================================
// Identity (Level 3): Cross-Document Entity Linking
// =============================================================================

// IdentityId is now a newtype in super::types::ids for type safety
pub use super::types::IdentityId;

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
    pub entity_type: Option<super::types::TypeLabel>,
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
        self.entity_type = Some(super::types::TypeLabel::from(s.as_str()));
        self
    }

    /// Set the entity type using a type-safe label.
    ///
    /// This is the preferred method for new code as it provides type safety
    /// and integrates with the core `EntityType` taxonomy.
    #[must_use]
    pub fn with_type_label(mut self, label: super::types::TypeLabel) -> Self {
        self.entity_type = Some(label);
        self
    }

    /// Get the entity type as a type-safe label.
    ///
    /// This converts the internal string representation to a `TypeLabel`,
    /// attempting to parse it as a core `EntityType` first.
    #[must_use]
    pub fn type_label(&self) -> Option<super::types::TypeLabel> {
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

// =============================================================================
// GroundedDocument: The Container
// =============================================================================

/// Wire format for [`GroundedDocument`] — contains only the persisted fields.
/// Internal indexes are rebuilt automatically via [`GroundedDocument::rebuild_indexes`]
/// during deserialization.
#[derive(Deserialize)]
struct GroundedDocumentWire {
    id: String,
    text: String,
    signals: Vec<Signal<Location>>,
    tracks: HashMap<TrackId, Track>,
    identities: HashMap<IdentityId, Identity>,
}

impl From<GroundedDocumentWire> for GroundedDocument {
    fn from(wire: GroundedDocumentWire) -> Self {
        let mut doc = Self {
            id: wire.id,
            text: wire.text,
            signals: wire.signals,
            tracks: wire.tracks,
            identities: wire.identities,
            signal_to_track: HashMap::new(),
            track_to_identity: HashMap::new(),
            next_signal_id: SignalId::ZERO,
            next_track_id: TrackId::ZERO,
            next_identity_id: IdentityId::ZERO,
        };
        doc.rebuild_indexes();
        doc
    }
}

/// A document with grounded entity annotations using the three-level hierarchy.
///
/// # Entity-Centric Design
///
/// Traditional document representations store entities as a flat list.
/// This design uses an entity-centric representation where:
///
/// 1. **Signals** are the atomic detections (Level 1)
/// 2. **Tracks** cluster signals into within-document entities (Level 2)
/// 3. **Identities** link tracks to global KB entities (Level 3)
///
/// This enables efficient:
/// - Streaming signal processing (add signals incrementally)
/// - Incremental coreference (cluster signals as they arrive)
/// - Lazy entity linking (resolve identities only when needed)
///
/// # Usage
///
/// ```rust
/// use anno_core::{GroundedDocument, Signal, Track, Identity, Location};
///
/// let mut doc = GroundedDocument::new("doc1", "Marie Curie won the Nobel Prize. She was a physicist.");
///
/// // Add signals (Level 1)
/// doc.add_signal(Signal::new(0, Location::text(0, 11), "Marie Curie", "Person", 0.95));
/// doc.add_signal(Signal::new(1, Location::text(33, 36), "She", "Person", 0.88));
///
/// // Form track (Level 2)
/// let mut track = Track::new(0, "Marie Curie");
/// track.add_signal(0, 0);
/// track.add_signal(1, 1);
/// doc.add_track(track);
///
/// // Link identity (Level 3)
/// let identity = Identity::from_kb(0, "Marie Curie", "wikidata", "Q7186");
/// doc.add_identity(identity);
/// doc.link_track_to_identity(0, 0);
/// ```
///
/// # Invariants
///
/// `GroundedDocument` maintains internal indices (`signal_to_track`, `track_to_identity`)
/// that must be consistent with the public collections. The following invariants hold:
///
/// 1. **Signal ID uniqueness**: All signals in `signals` have distinct `id` values.
/// 2. **Track signal references**: Every `SignalRef` in a `Track.signals` points to
///    a valid signal ID in `signals`.
/// 3. **Signal-to-track consistency**: If `signal_to_track[s] == t`, then the track `t`
///    contains a `SignalRef` pointing to `s`.
/// 4. **Track-to-identity consistency**: If `track_to_identity[t] == i`, then
///    `tracks[t].identity_id == Some(i)` and `identities` contains `i`.
/// 5. **Signal offsets validity**: Signal text locations should match `self.text`.
///
/// **Prefer mutation via provided methods** (`add_signal`, `add_track`, `add_signal_to_track`,
/// `link_track_to_identity`) rather than direct field manipulation to preserve invariants.
///
/// Use [`validate_invariants()`](Self::validate_invariants) to check structural consistency
/// after external modifications.
///
/// ## Serialization
///
/// Internal indexes (`signal_to_track`, `track_to_identity`, counter fields) are **not**
/// serialized. They are rebuilt automatically on deserialization via [`rebuild_indexes`](Self::rebuild_indexes).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(from = "GroundedDocumentWire")]
pub struct GroundedDocument {
    /// Document identifier
    pub id: String,
    /// Raw text content
    pub text: String,
    /// Level 1: Raw signals (detections)
    pub signals: Vec<Signal<Location>>,
    /// Level 2: Tracks (within-document coreference chains)
    pub tracks: HashMap<TrackId, Track>,
    /// Level 3: Global identities (KB-linked entities)
    pub identities: HashMap<IdentityId, Identity>,
    /// Index: signal_id → track_id (for efficient lookup).
    /// Not serialized; rebuilt on deserialization.
    #[serde(skip)]
    signal_to_track: HashMap<SignalId, TrackId>,
    /// Index: track_id → identity_id (for efficient lookup).
    /// Not serialized; rebuilt on deserialization.
    #[serde(skip)]
    track_to_identity: HashMap<TrackId, IdentityId>,
    /// Next signal ID (for auto-incrementing).
    /// Not serialized; rebuilt on deserialization.
    #[serde(skip)]
    next_signal_id: SignalId,
    /// Next track ID.
    /// Not serialized; rebuilt on deserialization.
    #[serde(skip)]
    next_track_id: TrackId,
    /// Next identity ID.
    /// Not serialized; rebuilt on deserialization.
    #[serde(skip)]
    next_identity_id: IdentityId,
}

impl GroundedDocument {
    /// Create a new grounded document.
    #[must_use]
    pub fn new(id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
            signals: Vec::new(),
            tracks: HashMap::new(),
            identities: HashMap::new(),
            signal_to_track: HashMap::new(),
            track_to_identity: HashMap::new(),
            next_signal_id: SignalId::ZERO,
            next_track_id: TrackId::ZERO,
            next_identity_id: IdentityId::ZERO,
        }
    }

    /// Rebuild all internal indexes from the public data fields.
    ///
    /// Call this after deserializing a `GroundedDocument` or after directly mutating the
    /// `signals`, `tracks`, or `identities` fields. The method recomputes:
    /// - `signal_to_track` from each track's signal list
    /// - `track_to_identity` from each track's `identity_id`
    /// - `next_signal_id`, `next_track_id`, `next_identity_id` counters
    pub fn rebuild_indexes(&mut self) {
        self.signal_to_track.clear();
        self.track_to_identity.clear();

        for (&track_id, track) in &self.tracks {
            for sig_ref in &track.signals {
                self.signal_to_track.insert(sig_ref.signal_id, track_id);
            }
            if let Some(identity_id) = track.identity_id {
                self.track_to_identity.insert(track_id, identity_id);
            }
        }

        self.next_signal_id = self
            .signals
            .iter()
            .map(|s| s.id)
            .max()
            .map_or(SignalId::ZERO, |id| id + 1);
        self.next_track_id = self
            .tracks
            .keys()
            .copied()
            .max()
            .map_or(TrackId::ZERO, |id| id + 1);
        self.next_identity_id = self
            .identities
            .keys()
            .copied()
            .max()
            .map_or(IdentityId::ZERO, |id| id + 1);
    }

    // -------------------------------------------------------------------------
    // Signal operations (Level 1)
    // -------------------------------------------------------------------------

    /// Add a signal and return its ID.
    pub fn add_signal(&mut self, mut signal: Signal<Location>) -> SignalId {
        let id = self.next_signal_id;
        signal.id = id;
        self.signals.push(signal);
        self.next_signal_id += 1;
        id
    }

    /// Get a signal by ID.
    #[must_use]
    pub fn get_signal(&self, id: impl Into<SignalId>) -> Option<&Signal<Location>> {
        let id = id.into();
        self.signals.iter().find(|s| s.id == id)
    }

    /// Get all signals.
    pub fn signals(&self) -> &[Signal<Location>] {
        &self.signals
    }

    // -------------------------------------------------------------------------
    // Track operations (Level 2)
    // -------------------------------------------------------------------------

    /// Add a track and return its ID.
    pub fn add_track(&mut self, mut track: Track) -> TrackId {
        let id = self.next_track_id;
        track.id = id;

        // Update signal → track index
        for signal_ref in &track.signals {
            self.signal_to_track.insert(signal_ref.signal_id, id);
        }

        self.tracks.insert(id, track);
        self.next_track_id += 1;
        id
    }

    /// Get a track by ID.
    #[must_use]
    pub fn get_track(&self, id: impl Into<TrackId>) -> Option<&Track> {
        self.tracks.get(&id.into())
    }

    /// Get a mutable reference to a track by ID.
    #[must_use]
    pub fn get_track_mut(&mut self, id: impl Into<TrackId>) -> Option<&mut Track> {
        self.tracks.get_mut(&id.into())
    }

    /// Add a signal to an existing track.
    ///
    /// This properly updates the signal_to_track index.
    /// Returns true if the signal was added, false if track doesn't exist.
    pub fn add_signal_to_track(
        &mut self,
        signal_id: impl Into<SignalId>,
        track_id: impl Into<TrackId>,
        position: u32,
    ) -> bool {
        let signal_id = signal_id.into();
        let track_id = track_id.into();
        if let Some(track) = self.tracks.get_mut(&track_id) {
            track.add_signal(signal_id, position);
            self.signal_to_track.insert(signal_id, track_id);
            true
        } else {
            false
        }
    }

    /// Get the track containing a signal.
    #[must_use]
    pub fn track_for_signal(&self, signal_id: SignalId) -> Option<&Track> {
        let track_id = self.signal_to_track.get(&signal_id)?;
        self.tracks.get(track_id)
    }

    /// Get all tracks.
    pub fn tracks(&self) -> impl Iterator<Item = &Track> {
        self.tracks.values()
    }

    // -------------------------------------------------------------------------
    // Identity operations (Level 3)
    // -------------------------------------------------------------------------

    /// Add an identity and return its ID.
    pub fn add_identity(&mut self, mut identity: Identity) -> IdentityId {
        let id = self.next_identity_id;
        identity.id = id;
        self.identities.insert(id, identity);
        self.next_identity_id += 1;
        id
    }

    /// Link a track to an identity.
    pub fn link_track_to_identity(
        &mut self,
        track_id: impl Into<TrackId>,
        identity_id: impl Into<IdentityId>,
    ) {
        let track_id = track_id.into();
        let identity_id = identity_id.into();
        if let Some(track) = self.tracks.get_mut(&track_id) {
            track.identity_id = Some(identity_id);
            self.track_to_identity.insert(track_id, identity_id);
        }
    }

    /// Get an identity by ID.
    #[must_use]
    pub fn get_identity(&self, id: IdentityId) -> Option<&Identity> {
        self.identities.get(&id)
    }

    /// Get the identity for a track.
    #[must_use]
    pub fn identity_for_track(&self, track_id: TrackId) -> Option<&Identity> {
        let identity_id = self.track_to_identity.get(&track_id)?;
        self.identities.get(identity_id)
    }

    /// Get the identity for a signal (transitively through track).
    #[must_use]
    pub fn identity_for_signal(&self, signal_id: SignalId) -> Option<&Identity> {
        let track_id = self.signal_to_track.get(&signal_id)?;
        self.identity_for_track(*track_id)
    }

    /// Get all identities.
    pub fn identities(&self) -> impl Iterator<Item = &Identity> {
        self.identities.values()
    }

    /// Get a TrackRef for a track in this document.
    ///
    /// Returns `None` if the track doesn't exist in this document.
    /// This validates that the track is still present (tracks can be removed).
    #[must_use]
    pub fn track_ref(&self, track_id: TrackId) -> Option<TrackRef> {
        // Validate that the track actually exists
        if self.tracks.contains_key(&track_id) {
            Some(TrackRef {
                doc_id: self.id.clone(),
                track_id,
            })
        } else {
            None
        }
    }

    // -------------------------------------------------------------------------
    // Conversion utilities
    // -------------------------------------------------------------------------

    /// Convert to legacy Entity format for backwards compatibility.
    #[must_use]
    pub fn to_entities(&self) -> Vec<Entity> {
        self.signals
            .iter()
            .map(|signal| {
                let (start, end) = signal.location.text_offsets().unwrap_or((0, 0));
                let track = self.track_for_signal(signal.id);
                let identity = track.and_then(|t| self.identity_for_track(t.id));

                {
                    let mut entity = Entity::new(
                        signal.surface.clone(),
                        EntityType::from_label(signal.label.as_str()),
                        start,
                        end,
                        signal.confidence,
                    );
                    entity.normalized = signal.normalized.clone();
                    entity.provenance = signal.provenance.clone();
                    entity.kb_id = identity.and_then(|i| i.kb_id.clone());
                    entity.canonical_id = track.map(|t| super::types::CanonicalId::new(t.id.get()));
                    entity.hierarchical_confidence = signal.hierarchical;
                    if let Location::Discontinuous { segments } = &signal.location {
                        entity.set_discontinuous_span(DiscontinuousSpan::new(
                            segments.iter().map(|(s, e)| (*s)..(*e)).collect(),
                        ));
                    }
                    entity
                }
            })
            .collect()
    }

    /// Create from legacy Entity slice.
    #[must_use]
    pub fn from_entities(
        id: impl Into<String>,
        text: impl Into<String>,
        entities: &[Entity],
    ) -> Self {
        let mut doc = Self::new(id, text);

        // Group entities by canonical_id to form tracks.
        //
        // IMPORTANT: Entities without a `canonical_id` are *not* coreferent by default.
        // They must each form their own singleton track (otherwise all NER mentions would
        // collapse into one giant track).
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        enum TrackKey {
            Canonical(super::types::CanonicalId),
            Singleton(usize),
        }

        let mut tracks_map: HashMap<TrackKey, Vec<SignalId>> = HashMap::new();
        let mut signal_to_entity_idx: HashMap<SignalId, usize> = HashMap::new();

        for (idx, entity) in entities.iter().enumerate() {
            let location = if let Some(disc) = &entity.discontinuous_span {
                Location::Discontinuous {
                    segments: disc.segments().iter().map(|r| (r.start, r.end)).collect(),
                }
            } else if let Some(visual) = &entity.visual_span {
                Location::from(visual)
            } else {
                Location::text(entity.start(), entity.end())
            };

            let mut signal = Signal::new(
                SignalId::new(idx as u64),
                location,
                &entity.text,
                entity.entity_type.as_label(),
                f32::from(entity.confidence),
            );
            signal.normalized = entity.normalized.clone();
            signal.provenance = entity.provenance.clone();
            signal.hierarchical = entity.hierarchical_confidence;

            let signal_id = doc.add_signal(signal);
            signal_to_entity_idx.insert(signal_id, idx);

            let key = match entity.canonical_id {
                Some(cid) => TrackKey::Canonical(cid),
                None => TrackKey::Singleton(idx),
            };
            tracks_map.entry(key).or_default().push(signal_id);
        }

        // Create tracks from grouped signals
        for (_key, signal_ids) in tracks_map {
            if let Some(first_signal) = signal_ids.first().and_then(|id| doc.get_signal(*id)) {
                let mut track = Track::new(doc.next_track_id, &first_signal.surface);
                track.entity_type =
                    Some(super::types::TypeLabel::from(first_signal.label.as_str()));

                for (pos, &signal_id) in signal_ids.iter().enumerate() {
                    track.add_signal(signal_id, pos as u32);
                }

                // If any member entity is linked to a KB entry, create an identity and link it.
                // (We intentionally do this even for singleton tracks without canonical_id.)
                let kb_id = signal_ids.iter().find_map(|sid| {
                    let ent_idx = signal_to_entity_idx.get(sid).copied()?;
                    entities.get(ent_idx)?.kb_id.clone()
                });
                if let Some(kb_id) = kb_id {
                    let identity = Identity::from_kb(
                        doc.next_identity_id,
                        &track.canonical_surface,
                        "unknown",
                        kb_id,
                    );
                    let identity_id = doc.add_identity(identity);
                    track = track.with_identity(identity_id);
                }

                doc.add_track(track);
            }
        }

        doc
    }

    /// Get signals filtered by label.
    #[must_use]
    pub fn signals_with_label(&self, label: &str) -> Vec<&Signal<Location>> {
        let want = super::types::TypeLabel::from(label);
        self.signals.iter().filter(|s| s.label == want).collect()
    }

    /// Get signals above a confidence threshold.
    #[must_use]
    pub fn confident_signals(&self, threshold: f32) -> Vec<&Signal<Location>> {
        self.signals
            .iter()
            .filter(|s| s.confidence >= threshold as f64)
            .collect()
    }

    /// Get tracks that are linked to an identity.
    pub fn linked_tracks(&self) -> impl Iterator<Item = &Track> {
        self.tracks.values().filter(|t| t.identity_id.is_some())
    }

    /// Get tracks that are NOT linked to any identity (need resolution).
    pub fn unlinked_tracks(&self) -> impl Iterator<Item = &Track> {
        self.tracks.values().filter(|t| t.identity_id.is_none())
    }

    /// Count of signals that are not yet assigned to any track.
    #[must_use]
    pub fn untracked_signal_count(&self) -> usize {
        self.signals
            .iter()
            .filter(|s| !self.signal_to_track.contains_key(&s.id))
            .count()
    }

    /// Get untracked signals (need coreference resolution).
    #[must_use]
    pub fn untracked_signals(&self) -> Vec<&Signal<Location>> {
        self.signals
            .iter()
            .filter(|s| !self.signal_to_track.contains_key(&s.id))
            .collect()
    }

    // -------------------------------------------------------------------------
    // Advanced Query Methods
    // -------------------------------------------------------------------------

    /// Get signals filtered by modality.
    #[must_use]
    pub fn signals_by_modality(&self, modality: Modality) -> Vec<&Signal<Location>> {
        self.signals
            .iter()
            .filter(|s| s.modality == modality)
            .collect()
    }

    /// Get all text-based signals (symbolic modality).
    #[must_use]
    pub fn text_signals(&self) -> Vec<&Signal<Location>> {
        self.signals_by_modality(Modality::Symbolic)
    }

    /// Get all visual signals (iconic modality).
    #[must_use]
    pub fn visual_signals(&self) -> Vec<&Signal<Location>> {
        self.signals_by_modality(Modality::Iconic)
    }

    /// Find signals that overlap with a given location.
    #[must_use]
    pub fn overlapping_signals(&self, location: &Location) -> Vec<&Signal<Location>> {
        self.signals
            .iter()
            .filter(|s| s.location.overlaps(location))
            .collect()
    }

    /// Find signals within a text range.
    #[must_use]
    pub fn signals_in_range(&self, start: usize, end: usize) -> Vec<&Signal<Location>> {
        self.signals
            .iter()
            .filter(|s| {
                if let Some((s_start, s_end)) = s.location.text_offsets() {
                    s_start >= start && s_end <= end
                } else {
                    false
                }
            })
            .collect()
    }

    /// Get signals that are negated.
    #[must_use]
    pub fn negated_signals(&self) -> Vec<&Signal<Location>> {
        self.signals.iter().filter(|s| s.negated).collect()
    }

    /// Get signals with a specific quantifier.
    #[must_use]
    pub fn quantified_signals(&self, quantifier: Quantifier) -> Vec<&Signal<Location>> {
        self.signals
            .iter()
            .filter(|s| s.quantifier == Some(quantifier))
            .collect()
    }

    // -------------------------------------------------------------------------
    // Validation
    // -------------------------------------------------------------------------

    /// Validate all signals against the document text.
    ///
    /// Returns a list of validation errors. Empty means all valid.
    ///
    /// # Example
    ///
    /// ```rust
    /// use anno_core::{GroundedDocument, Signal, Location};
    ///
    /// let mut doc = GroundedDocument::new("test", "Marie Curie was a physicist.");
    /// doc.add_signal(Signal::new(0, Location::text(0, 11), "Marie Curie", "PER", 0.9));
    /// assert!(doc.validate().is_empty());
    ///
    /// // Bad signal: wrong text at offset
    /// doc.add_signal(Signal::new(0, Location::text(0, 5), "WRONG", "PER", 0.9));
    /// assert!(!doc.validate().is_empty());
    /// ```
    #[must_use]
    pub fn validate(&self) -> Vec<SignalValidationError> {
        self.signals
            .iter()
            .filter_map(|s| s.validate_against(&self.text))
            .collect()
    }

    /// Validate structural invariants of the document.
    ///
    /// Returns a list of invariant violations. An empty list means the document
    /// is structurally consistent.
    ///
    /// This checks:
    /// 1. Signal ID uniqueness
    /// 2. Track signal references point to existing signals
    /// 3. `signal_to_track` index consistency
    /// 4. `track_to_identity` index consistency
    /// 5. Track identity references point to existing identities
    ///
    /// Use this after any direct field manipulation to ensure consistency.
    ///
    /// # Example
    ///
    /// ```rust
    /// use anno_core::{GroundedDocument, Signal, Location};
    ///
    /// let mut doc = GroundedDocument::new("test", "Marie Curie was a physicist.");
    /// doc.add_signal(Signal::new(0, Location::text(0, 11), "Marie Curie", "PER", 0.9));
    /// assert!(doc.validate_invariants().is_empty());
    /// ```
    #[must_use]
    pub fn validate_invariants(&self) -> Vec<String> {
        let mut errors = Vec::new();

        // 1. Signal ID uniqueness
        let mut seen_ids = std::collections::HashSet::new();
        for signal in &self.signals {
            if !seen_ids.insert(signal.id) {
                errors.push(format!("Duplicate signal ID: {}", signal.id));
            }
        }

        // Build signal ID set for reference checks
        let signal_ids: std::collections::HashSet<_> = self.signals.iter().map(|s| s.id).collect();

        // 2. Track signal references point to existing signals
        for (track_id, track) in &self.tracks {
            for signal_ref in &track.signals {
                if !signal_ids.contains(&signal_ref.signal_id) {
                    errors.push(format!(
                        "Track {} references non-existent signal {}",
                        track_id, signal_ref.signal_id
                    ));
                }
            }
        }

        // 3. signal_to_track consistency
        for (signal_id, track_id) in &self.signal_to_track {
            // Check track exists
            if let Some(track) = self.tracks.get(track_id) {
                // Check track contains the signal reference
                if !track.signals.iter().any(|r| r.signal_id == *signal_id) {
                    errors.push(format!(
                        "signal_to_track[{}] = {} but track doesn't contain signal",
                        signal_id, track_id
                    ));
                }
            } else {
                errors.push(format!(
                    "signal_to_track[{}] = {} but track doesn't exist",
                    signal_id, track_id
                ));
            }
        }

        // 4. track_to_identity consistency
        for (track_id, identity_id) in &self.track_to_identity {
            // Check track exists and has matching identity_id
            if let Some(track) = self.tracks.get(track_id) {
                if track.identity_id != Some(*identity_id) {
                    errors.push(format!(
                        "track_to_identity[{}] = {} but track.identity_id = {:?}",
                        track_id, identity_id, track.identity_id
                    ));
                }
            } else {
                errors.push(format!(
                    "track_to_identity[{}] = {} but track doesn't exist",
                    track_id, identity_id
                ));
            }

            // Check identity exists
            if !self.identities.contains_key(identity_id) {
                errors.push(format!(
                    "track_to_identity[{}] = {} but identity doesn't exist",
                    track_id, identity_id
                ));
            }
        }

        // 5. Track identity references point to existing identities
        for (track_id, track) in &self.tracks {
            if let Some(identity_id) = track.identity_id {
                if !self.identities.contains_key(&identity_id) {
                    errors.push(format!(
                        "Track {} references non-existent identity {}",
                        track_id, identity_id
                    ));
                }
            }
        }

        errors
    }

    /// Check if all structural invariants hold.
    #[must_use]
    pub fn invariants_hold(&self) -> bool {
        self.validate_invariants().is_empty()
    }

    /// Check if all signals are valid against document text.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.signals.iter().all(|s| s.is_valid(&self.text))
    }

    /// Add a signal, validating it first.
    ///
    /// Returns `Err` if the signal's offsets don't match the document text.
    pub fn add_signal_validated(
        &mut self,
        signal: Signal<Location>,
    ) -> Result<SignalId, SignalValidationError> {
        if let Some(err) = signal.validate_against(&self.text) {
            return Err(err);
        }
        Ok(self.add_signal(signal))
    }

    /// Add a signal by finding text in document (safe construction).
    ///
    /// Returns the signal ID, or `None` if text not found.
    ///
    /// # Example
    ///
    /// ```rust
    /// use anno_core::GroundedDocument;
    ///
    /// let mut doc = GroundedDocument::new("test", "Marie Curie was a physicist.");
    /// let id = doc.add_signal_from_text("Marie Curie", "PER", 0.95);
    /// assert!(id.is_some());
    /// ```
    pub fn add_signal_from_text(
        &mut self,
        surface: &str,
        label: impl Into<super::types::TypeLabel>,
        confidence: f32,
    ) -> Option<SignalId> {
        let signal = Signal::from_text(&self.text, surface, label, confidence)?;
        Some(self.add_signal(signal))
    }

    /// Add a signal by finding the nth occurrence of text.
    pub fn add_signal_from_text_nth(
        &mut self,
        surface: &str,
        label: impl Into<super::types::TypeLabel>,
        confidence: f32,
        occurrence: usize,
    ) -> Option<SignalId> {
        let signal = Signal::from_text_nth(&self.text, surface, label, confidence, occurrence)?;
        Some(self.add_signal(signal))
    }

    // -------------------------------------------------------------------------
    // Statistics
    // -------------------------------------------------------------------------

    /// Get statistics about the document.
    #[must_use]
    pub fn stats(&self) -> DocumentStats {
        let signal_count = self.signals.len();
        let track_count = self.tracks.len();
        let identity_count = self.identities.len();

        let linked_track_count = self
            .tracks
            .values()
            .filter(|t| t.identity_id.is_some())
            .count();
        let untracked_count = self.untracked_signal_count();

        let avg_track_size = if track_count > 0 {
            self.tracks.values().map(|t| t.len()).sum::<usize>() as f32 / track_count as f32
        } else {
            0.0
        };

        let singleton_count = self.tracks.values().filter(|t| t.is_singleton()).count();

        let avg_confidence = if signal_count > 0 {
            self.signals
                .iter()
                .map(|s| s.confidence.value() as f32)
                .sum::<f32>()
                / signal_count as f32
        } else {
            0.0
        };

        let negated_count = self.signals.iter().filter(|s| s.negated).count();

        // Count by modality
        let symbolic_count = self
            .signals
            .iter()
            .filter(|s| s.modality == Modality::Symbolic)
            .count();
        let iconic_count = self
            .signals
            .iter()
            .filter(|s| s.modality == Modality::Iconic)
            .count();
        let hybrid_count = self
            .signals
            .iter()
            .filter(|s| s.modality == Modality::Hybrid)
            .count();

        DocumentStats {
            signal_count,
            track_count,
            identity_count,
            linked_track_count,
            untracked_count,
            avg_track_size,
            singleton_count,
            avg_confidence,
            negated_count,
            symbolic_count,
            iconic_count,
            hybrid_count,
        }
    }

    // -------------------------------------------------------------------------
    // Batch Operations
    // -------------------------------------------------------------------------

    /// Add multiple signals at once.
    ///
    /// Returns the IDs of all added signals.
    pub fn add_signals(
        &mut self,
        signals: impl IntoIterator<Item = Signal<Location>>,
    ) -> Vec<SignalId> {
        signals.into_iter().map(|s| self.add_signal(s)).collect()
    }

    /// Create a track from a list of signal IDs.
    ///
    /// Automatically sets positions based on order.
    pub fn create_track_from_signals(
        &mut self,
        canonical: impl Into<String>,
        signal_ids: &[SignalId],
    ) -> Option<TrackId> {
        if signal_ids.is_empty() {
            return None;
        }

        let mut track = Track::new(TrackId::ZERO, canonical);
        for (pos, &id) in signal_ids.iter().enumerate() {
            track.add_signal(id, pos as u32);
        }
        Some(self.add_track(track))
    }

    /// Merge multiple tracks into one.
    ///
    /// The resulting track has all signals from the input tracks.
    /// The canonical surface comes from the first track.
    pub fn merge_tracks(&mut self, track_ids: &[TrackId]) -> Option<TrackId> {
        if track_ids.is_empty() {
            return None;
        }

        // Collect all signals from tracks to merge
        let mut all_signals: Vec<SignalRef> = Vec::new();
        let mut canonical = String::new();
        let mut entity_type = None;

        for &track_id in track_ids {
            if let Some(track) = self.tracks.get(&track_id) {
                if canonical.is_empty() {
                    canonical = track.canonical_surface.clone();
                    entity_type = track.entity_type.clone();
                }
                all_signals.extend(track.signals.iter().cloned());
            }
        }

        if all_signals.is_empty() {
            return None;
        }

        // Sort by position
        all_signals.sort_by_key(|s| s.position);

        // Remove old tracks
        for &track_id in track_ids {
            self.tracks.remove(&track_id);
        }

        // Create new merged track
        let mut new_track = Track::new(TrackId::ZERO, canonical);
        new_track.entity_type = entity_type;
        for (pos, signal_ref) in all_signals.iter().enumerate() {
            new_track.add_signal(signal_ref.signal_id, pos as u32);
        }

        Some(self.add_track(new_track))
    }

    /// Find all pairs of overlapping signals (potential duplicates or nested entities).
    #[must_use]
    pub fn find_overlapping_signal_pairs(&self) -> Vec<(SignalId, SignalId)> {
        let mut pairs = Vec::new();
        let signals: Vec<_> = self.signals.iter().collect();

        for i in 0..signals.len() {
            for j in (i + 1)..signals.len() {
                if signals[i].location.overlaps(&signals[j].location) {
                    pairs.push((signals[i].id, signals[j].id));
                }
            }
        }

        pairs
    }
}

/// Statistics about a grounded document.
#[derive(Debug, Clone, Copy, Default)]
pub struct DocumentStats {
    /// Total number of signals
    pub signal_count: usize,
    /// Total number of tracks
    pub track_count: usize,
    /// Total number of identities
    pub identity_count: usize,
    /// Number of tracks linked to identities
    pub linked_track_count: usize,
    /// Number of signals not in any track
    pub untracked_count: usize,
    /// Average signals per track
    pub avg_track_size: f32,
    /// Number of singleton tracks (single mention)
    pub singleton_count: usize,
    /// Average signal confidence
    pub avg_confidence: f32,
    /// Number of negated signals
    pub negated_count: usize,
    /// Number of symbolic (text) signals
    pub symbolic_count: usize,
    /// Number of iconic (visual) signals
    pub iconic_count: usize,
    /// Number of hybrid signals
    pub hybrid_count: usize,
}

impl std::fmt::Display for DocumentStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Document Statistics:")?;
        writeln!(
            f,
            "  Signals: {} (avg confidence: {:.2})",
            self.signal_count, self.avg_confidence
        )?;
        writeln!(
            f,
            "  Tracks: {} (avg size: {:.1}, singletons: {})",
            self.track_count, self.avg_track_size, self.singleton_count
        )?;
        writeln!(
            f,
            "  Identities: {} ({} tracks linked)",
            self.identity_count, self.linked_track_count
        )?;
        writeln!(f, "  Untracked signals: {}", self.untracked_count)?;
        writeln!(
            f,
            "  Modalities: {} symbolic, {} iconic, {} hybrid",
            self.symbolic_count, self.iconic_count, self.hybrid_count
        )?;
        if self.negated_count > 0 {
            writeln!(f, "  Negated: {}", self.negated_count)?;
        }
        Ok(())
    }
}

// =============================================================================
// Spatial Index for Efficient Range Queries
// =============================================================================

/// A simple interval tree node for text span indexing.
///
/// This provides O(log n + k) lookup for signals within a text range,
/// where k is the number of results. Much faster than O(n) linear scan
/// for documents with many signals.
#[derive(Debug, Clone)]
struct IntervalNode {
    /// Signal ID
    signal_id: SignalId,
    /// Start offset (inclusive)
    start: usize,
    /// End offset (exclusive)
    end: usize,
    /// Maximum end in this subtree (for efficient pruning)
    max_end: usize,
    /// Left child
    left: Option<Box<IntervalNode>>,
    /// Right child
    right: Option<Box<IntervalNode>>,
}

impl IntervalNode {
    fn new(signal_id: SignalId, start: usize, end: usize) -> Self {
        Self {
            signal_id,
            start,
            end,
            max_end: end,
            left: None,
            right: None,
        }
    }

    fn insert(&mut self, signal_id: SignalId, start: usize, end: usize) {
        self.max_end = self.max_end.max(end);

        if start < self.start {
            if let Some(ref mut left) = self.left {
                left.insert(signal_id, start, end);
            } else {
                self.left = Some(Box::new(IntervalNode::new(signal_id, start, end)));
            }
        } else if let Some(ref mut right) = self.right {
            right.insert(signal_id, start, end);
        } else {
            self.right = Some(Box::new(IntervalNode::new(signal_id, start, end)));
        }
    }

    fn query_overlap(&self, query_start: usize, query_end: usize, results: &mut Vec<SignalId>) {
        // Check if this interval overlaps with query
        if self.start < query_end && query_start < self.end {
            results.push(self.signal_id);
        }

        // Check left subtree if it could contain overlapping intervals
        if let Some(ref left) = self.left {
            if left.max_end > query_start {
                left.query_overlap(query_start, query_end, results);
            }
        }

        // Check right subtree if query could overlap
        if let Some(ref right) = self.right {
            if self.start < query_end {
                right.query_overlap(query_start, query_end, results);
            }
        }
    }

    fn query_containing(&self, query_start: usize, query_end: usize, results: &mut Vec<SignalId>) {
        // Check if this interval fully contains the query
        if self.start <= query_start && self.end >= query_end {
            results.push(self.signal_id);
        }

        // Check left subtree if it could contain the range
        if let Some(ref left) = self.left {
            if left.max_end >= query_end {
                left.query_containing(query_start, query_end, results);
            }
        }

        // Check right subtree
        if let Some(ref right) = self.right {
            if self.start <= query_start {
                right.query_containing(query_start, query_end, results);
            }
        }
    }

    fn query_contained_in(
        &self,
        range_start: usize,
        range_end: usize,
        results: &mut Vec<SignalId>,
    ) {
        // Check if this interval is fully contained in range
        if self.start >= range_start && self.end <= range_end {
            results.push(self.signal_id);
        }

        // Check left subtree
        if let Some(ref left) = self.left {
            left.query_contained_in(range_start, range_end, results);
        }

        // Check right subtree if it could have contained intervals
        if let Some(ref right) = self.right {
            if self.start < range_end {
                right.query_contained_in(range_start, range_end, results);
            }
        }
    }
}

/// Spatial index for text signals using an interval tree.
///
/// Enables efficient queries:
/// - `query_overlap(start, end)`: Find signals that overlap with range
/// - `query_containing(start, end)`: Find signals that fully contain range
/// - `query_contained_in(start, end)`: Find signals fully within range
///
/// # Performance
///
/// - Build: O(n log n)
/// - Query: O(log n + k) where k is result count
/// - Space: O(n)
///
/// For documents with >100 signals, this provides significant speedup
/// over linear scan for range queries.
#[derive(Debug, Clone, Default)]
pub struct TextSpatialIndex {
    root: Option<IntervalNode>,
    size: usize,
}

impl TextSpatialIndex {
    /// Create a new empty index.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Build index from signals in a document.
    #[must_use]
    pub fn from_signals(signals: &[Signal<Location>]) -> Self {
        let mut index = Self::new();
        for signal in signals {
            if let Some((start, end)) = signal.location.text_offsets() {
                index.insert(signal.id, start, end);
            }
        }
        index
    }

    /// Insert a text span into the index.
    pub fn insert(&mut self, signal_id: SignalId, start: usize, end: usize) {
        if let Some(ref mut root) = self.root {
            root.insert(signal_id, start, end);
        } else {
            self.root = Some(IntervalNode::new(signal_id, start, end));
        }
        self.size += 1;
    }

    /// Find signals that overlap with the given range.
    #[must_use]
    pub fn query_overlap(&self, start: usize, end: usize) -> Vec<SignalId> {
        let mut results = Vec::new();
        if let Some(ref root) = self.root {
            root.query_overlap(start, end, &mut results);
        }
        results
    }

    /// Find signals that fully contain the given range.
    #[must_use]
    pub fn query_containing(&self, start: usize, end: usize) -> Vec<SignalId> {
        let mut results = Vec::new();
        if let Some(ref root) = self.root {
            root.query_containing(start, end, &mut results);
        }
        results
    }

    /// Find signals fully contained within the given range.
    #[must_use]
    pub fn query_contained_in(&self, start: usize, end: usize) -> Vec<SignalId> {
        let mut results = Vec::new();
        if let Some(ref root) = self.root {
            root.query_contained_in(start, end, &mut results);
        }
        results
    }

    /// Number of entries in the index.
    #[must_use]
    pub fn len(&self) -> usize {
        self.size
    }

    /// Check if the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
}

impl GroundedDocument {
    /// Build a spatial index for efficient text range queries.
    ///
    /// This is useful for documents with many signals where you need
    /// to frequently query by text position.
    ///
    /// # Example
    ///
    /// ```rust
    /// use anno_core::{GroundedDocument, Signal, Location};
    ///
    /// let mut doc = GroundedDocument::new("doc", "Some text with entities.");
    /// doc.add_signal(Signal::new(0, Location::text(0, 4), "Some", "T", 0.9));
    /// doc.add_signal(Signal::new(0, Location::text(10, 14), "with", "T", 0.9));
    ///
    /// let index = doc.build_text_index();
    /// let in_range = index.query_contained_in(0, 20);
    /// assert_eq!(in_range.len(), 2);
    /// ```
    #[must_use]
    pub fn build_text_index(&self) -> TextSpatialIndex {
        TextSpatialIndex::from_signals(&self.signals)
    }

    /// Query signals using the spatial index (builds index if needed).
    ///
    /// For repeated queries, build the index once with `build_text_index()`
    /// and reuse it.
    #[must_use]
    pub fn query_signals_in_range_indexed(
        &self,
        start: usize,
        end: usize,
    ) -> Vec<&Signal<Location>> {
        let index = self.build_text_index();
        let ids = index.query_contained_in(start, end);
        ids.iter().filter_map(|&id| self.get_signal(id)).collect()
    }

    /// Query overlapping signals using spatial index.
    #[must_use]
    pub fn query_overlapping_signals_indexed(
        &self,
        start: usize,
        end: usize,
    ) -> Vec<&Signal<Location>> {
        let index = self.build_text_index();
        let ids = index.query_overlap(start, end);
        ids.iter().filter_map(|&id| self.get_signal(id)).collect()
    }

    /// Convert this grounded document into a coreference document for evaluation.
    ///
    /// This is a lightweight bridge between the production pipeline types
    /// (Signal/Track/Identity) and the evaluation-oriented coreference types
    /// (`CorefDocument`, `CorefChain`, `Mention`).
    ///
    /// - Each [`Track`] becomes a [`super::coref::CorefChain`]
    /// - Each track mention is derived from the track's signal locations
    /// - Non-text signals (iconic-only locations) are skipped
    ///
    /// Note: Mention typing (proper/nominal/pronominal) is left unset; callers
    /// doing mention-type evaluation should compute that separately.
    #[must_use]
    pub fn to_coref_document(&self) -> super::coref::CorefDocument {
        use super::coref::{CorefChain, CorefDocument, Mention};
        use std::collections::HashMap;

        // Build a fast index for signal lookup.
        let signal_by_id: HashMap<SignalId, &Signal<Location>> =
            self.signals.iter().map(|s| (s.id, s)).collect();

        let mut chains: Vec<CorefChain> = Vec::new();

        for track in self.tracks.values() {
            let mut mentions: Vec<Mention> = Vec::new();

            for sref in &track.signals {
                let Some(signal) = signal_by_id.get(&sref.signal_id) else {
                    continue;
                };

                let Some((start, end)) = signal.location.text_offsets() else {
                    continue;
                };

                let mut m = Mention::new(signal.surface.clone(), start, end);
                m.entity_type = Some(signal.label.to_string());
                mentions.push(m);
            }

            if mentions.is_empty() {
                continue;
            }

            let mut chain = CorefChain::new(mentions);
            chain.entity_type = track.entity_type.as_ref().map(|t| t.to_string());
            chains.push(chain);
        }

        // Deterministic ordering: sort by earliest mention.
        chains.sort_by_key(|c| c.mentions.first().map(|m| m.start).unwrap_or(usize::MAX));

        CorefDocument::with_id(&self.text, &self.id, chains)
    }
}

// =============================================================================
// HTML Visualization (Brutalist/Functional Style)
// =============================================================================

/// Generate an HTML visualization of a grounded document.
///
/// Brutalist design: monospace, dense tables, no decoration, raw data.
pub fn render_document_html(doc: &GroundedDocument) -> String {
    let mut html = String::new();
    let stats = doc.stats();

    html.push_str(r#"<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="color-scheme" content="dark light">
<title>grounded::GroundedDocument</title>
<style>
:root{
  /* Allow UA widgets (inputs/scrollbars) to match the theme */
  color-scheme: light dark;
  /* Dark (default) */
  --bg:#0a0a0a;
  --panel-bg:#0d0d0d;
  --text:#b0b0b0;
  --text-strong:#fff;
  --muted:#666;
  --border:#222;
  --border-strong:#333;
  --hover:#111;
  --input-bg:#080808;
  --active:#fff;
  --track-strong:rgba(255,255,255,0.35);
  --track-soft:rgba(255,255,255,0.18);
  /* Entity colors (dark) */
  --per-bg:#1a1a2e; --per-br:#4a4a8a; --per-tx:#8888cc;
  --org-bg:#1a2e1a; --org-br:#4a8a4a; --org-tx:#88cc88;
  --loc-bg:#2e2e1a; --loc-br:#8a8a4a; --loc-tx:#cccc88;
  --mis-bg:#1a1a1a; --mis-br:#4a4a4a; --mis-tx:#999;
  --dat-bg:#2e1a1a; --dat-br:#8a4a4a; --dat-tx:#cc8888;
  --badge-y-bg:#1a2e1a; --badge-y-tx:#4a8a4a; --badge-y-br:#2a4a2a;
  --badge-n-bg:#2e2e1a; --badge-n-tx:#8a8a4a; --badge-n-br:#4a4a2a;
}
@media (prefers-color-scheme: light){
  :root{
    --bg:#ffffff;
    --panel-bg:#f7f7f7;
    --text:#222;
    --text-strong:#000;
    --muted:#555;
    --border:#d6d6d6;
    --border-strong:#c6c6c6;
    --hover:#f0f0f0;
    --input-bg:#ffffff;
    --active:#000;
    --track-strong:rgba(0,0,0,0.25);
    --track-soft:rgba(0,0,0,0.12);
    /* Entity colors (light) */
    --per-bg:#e9e9ff; --per-br:#6c6cff; --per-tx:#2b2b7a;
    --org-bg:#e9f7e9; --org-br:#2f8a2f; --org-tx:#1f5a1f;
    --loc-bg:#fff7db; --loc-br:#8a7a2f; --loc-tx:#5a4d12;
    --mis-bg:#f2f2f2; --mis-br:#8a8a8a; --mis-tx:#333;
    --dat-bg:#ffe9e9; --dat-br:#8a2f2f; --dat-tx:#5a1f1f;
    --badge-y-bg:#e9f7e9; --badge-y-tx:#1f5a1f; --badge-y-br:#9ad19a;
    --badge-n-bg:#fff7db; --badge-n-tx:#5a4d12; --badge-n-br:#e2d39a;
  }
}
html[data-theme='dark']{
  --bg:#0a0a0a; --panel-bg:#0d0d0d; --text:#b0b0b0; --text-strong:#fff;
  --muted:#666; --border:#222; --border-strong:#333; --hover:#111;
  --input-bg:#080808; --active:#fff;
  --track-strong:rgba(255,255,255,0.35); --track-soft:rgba(255,255,255,0.18);
  --per-bg:#1a1a2e; --per-br:#4a4a8a; --per-tx:#8888cc;
  --org-bg:#1a2e1a; --org-br:#4a8a4a; --org-tx:#88cc88;
  --loc-bg:#2e2e1a; --loc-br:#8a8a4a; --loc-tx:#cccc88;
  --mis-bg:#1a1a1a; --mis-br:#4a4a4a; --mis-tx:#999;
  --dat-bg:#2e1a1a; --dat-br:#8a4a4a; --dat-tx:#cc8888;
  --badge-y-bg:#1a2e1a; --badge-y-tx:#4a8a4a; --badge-y-br:#2a4a2a;
  --badge-n-bg:#2e2e1a; --badge-n-tx:#8a8a4a; --badge-n-br:#4a4a2a;
}
html[data-theme='light']{
  --bg:#ffffff; --panel-bg:#f7f7f7; --text:#222; --text-strong:#000;
  --muted:#555; --border:#d6d6d6; --border-strong:#c6c6c6; --hover:#f0f0f0;
  --input-bg:#ffffff; --active:#000;
  --track-strong:rgba(0,0,0,0.25); --track-soft:rgba(0,0,0,0.12);
  --per-bg:#e9e9ff; --per-br:#6c6cff; --per-tx:#2b2b7a;
  --org-bg:#e9f7e9; --org-br:#2f8a2f; --org-tx:#1f5a1f;
  --loc-bg:#fff7db; --loc-br:#8a7a2f; --loc-tx:#5a4d12;
  --mis-bg:#f2f2f2; --mis-br:#8a8a8a; --mis-tx:#333;
  --dat-bg:#ffe9e9; --dat-br:#8a2f2f; --dat-tx:#5a1f1f;
  --badge-y-bg:#e9f7e9; --badge-y-tx:#1f5a1f; --badge-y-br:#9ad19a;
  --badge-n-bg:#fff7db; --badge-n-tx:#5a4d12; --badge-n-br:#e2d39a;
}

*{box-sizing:border-box;margin:0;padding:0}
body{font:12px/1.4 monospace;background:var(--bg);color:var(--text);padding:8px}
h1,h2,h3{color:var(--text-strong);font-weight:normal;border-bottom:1px solid var(--border-strong);padding:4px 0;margin:16px 0 8px}
h1{font-size:14px}h2{font-size:12px}h3{font-size:11px;color:var(--muted)}
 a{color:inherit}
 a:hover{text-decoration:underline}
table{width:100%;border-collapse:collapse;font-size:11px;margin:4px 0}
th,td{padding:4px 8px;text-align:left;border:1px solid var(--border)}
th{background:var(--hover);color:var(--muted);font-weight:normal;text-transform:uppercase;font-size:10px}
tr:hover{background:var(--hover)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:8px}
.panel{border:1px solid var(--border);background:var(--panel-bg);padding:8px}
.panel-h{display:flex;align-items:center;gap:8px}
.toggle{cursor:pointer;user-select:none;color:var(--muted);border:1px solid var(--border);background:var(--bg);padding:2px 6px;font-size:10px}
.panel-collapsed table,.panel-collapsed .panel-body{display:none}
.toolbar{display:flex;gap:8px;align-items:center;margin:8px 0 0}
.toolbar input{width:100%;max-width:520px;background:var(--input-bg);border:1px solid var(--border);color:var(--text);padding:6px 8px;font:12px monospace}
.muted{color:var(--muted)}
.panel-body{white-space:pre-wrap;word-break:break-word}
.text-box{background:var(--input-bg);border:1px solid var(--border);padding:8px;white-space:pre-wrap;word-break:break-word;line-height:1.6}
.e{padding:1px 2px;border-bottom:1px solid}
.seg{cursor:pointer}
.e-per{background:var(--per-bg);border-color:var(--per-br);color:var(--per-tx)}
.e-org{background:var(--org-bg);border-color:var(--org-br);color:var(--org-tx)}
.e-loc{background:var(--loc-bg);border-color:var(--loc-br);color:var(--loc-tx)}
.e-misc{background:var(--mis-bg);border-color:var(--mis-br);color:var(--mis-tx)}
.e-date{background:var(--dat-bg);border-color:var(--dat-br);color:var(--dat-tx)}
.e-track{box-shadow:inset 0 0 0 1px var(--track-strong)}
.e-track-hover{box-shadow:inset 0 0 0 1px var(--track-soft)}
.e-active{outline:2px solid var(--active);outline-offset:1px}
.conf{color:var(--muted);font-size:10px}
.badge{display:inline-block;padding:1px 4px;font-size:9px;text-transform:uppercase}
.badge-y{background:var(--badge-y-bg);color:var(--badge-y-tx);border:1px solid var(--badge-y-br)}
.badge-n{background:var(--badge-n-bg);color:var(--badge-n-tx);border:1px solid var(--badge-n-br)}
.stats{display:flex;gap:16px;padding:8px 0;border-bottom:1px solid var(--border);margin-bottom:8px}
.stat{text-align:center}.stat-v{font-size:18px;color:var(--text-strong)}.stat-l{font-size:9px;color:var(--muted);text-transform:uppercase}
.id{color:var(--muted);font-size:9px}
.kb{color:var(--muted)}
.arrow{color:var(--muted)}
</style>
</head>
<body>
"#);

    // Header with stats
    html.push_str(&format!(
        r#"<div class="panel-h" style="justify-content:space-between"><h1>doc_id="{}" len={}</h1><span class="toggle" id="theme-toggle" title="toggle theme (auto → dark → light)">theme: auto</span></div>"#,
        html_escape(&doc.id),
        doc.text.len()
    ));

    html.push_str(r#"<div class="stats">"#);
    html.push_str(&format!(
        r#"<div class="stat"><div class="stat-v">{}</div><div class="stat-l">signals</div></div>"#,
        stats.signal_count
    ));
    html.push_str(&format!(
        r#"<div class="stat"><div class="stat-v">{}</div><div class="stat-l">tracks</div></div>"#,
        stats.track_count
    ));
    html.push_str(&format!(r#"<div class="stat"><div class="stat-v">{}</div><div class="stat-l">identities</div></div>"#, stats.identity_count));
    html.push_str(&format!(r#"<div class="stat"><div class="stat-v">{:.2}</div><div class="stat-l">avg_conf</div></div>"#, stats.avg_confidence));
    html.push_str(&format!(
        r#"<div class="stat"><div class="stat-v">{}</div><div class="stat-l">linked</div></div>"#,
        stats.linked_track_count
    ));
    html.push_str(&format!(r#"<div class="stat"><div class="stat-v">{}</div><div class="stat-l">untracked</div></div>"#, stats.untracked_count));
    if stats.iconic_count > 0 || stats.hybrid_count > 0 {
        html.push_str(&format!(r#"<div class="stat"><div class="stat-v">{}/{}/{}</div><div class="stat-l">sym/ico/hyb</div></div>"#,
            stats.symbolic_count, stats.iconic_count, stats.hybrid_count));
    }
    html.push_str(r#"</div>"#);

    // Annotated text
    html.push_str(r#"<h2>text</h2>"#);
    html.push_str(r#"<div class="text-box">"#);
    html.push_str(&annotate_text_html(
        &doc.text,
        doc.signals(),
        &doc.signal_to_track,
    ));
    html.push_str(r#"</div>"#);

    // Selection panel (filled by JS)
    html.push_str(
        r#"<h2>selection</h2><div class="panel" id="selection-panel" role="region" aria-label="selection"><div class="panel-h"><h3>selection</h3><span class="muted" id="selection-hint" role="status" aria-live="polite">click a mention / row to see coref track details</span></div><pre class="panel-body" id="selection-body" role="textbox" aria-readonly="true" aria-label="selection details">—</pre></div>"#,
    );

    // Grid layout for three levels
    html.push_str(r#"<div class="grid">"#);

    // Level 1: Signals table
    html.push_str(r#"<div class="panel" id="panel-signals"><div class="panel-h"><h3>signals (level 1)</h3><span class="toggle" data-toggle="panel-signals">toggle</span></div><div class="toolbar"><input id="signal-filter" type="text" placeholder="filter signals: id / label / surface (e.g. 'PER', 'S12', 'Paris')" /><span class="muted" id="signal-filter-count"></span></div><table id="signals-table">"#);
    html.push_str(r#"<tr><th>id</th><th>span</th><th>surface</th><th>label</th><th>conf</th><th>track</th></tr>"#);
    for signal in doc.signals() {
        let (span, start_opt, end_opt) = if let Some((s, e)) = signal.location.text_offsets() {
            (format!("[{},{})", s, e), Some(s), Some(e))
        } else {
            ("bbox".to_string(), None, None)
        };
        let track_id_num = doc.signal_to_track.get(&signal.id).copied();
        let track_id = track_id_num
            .map(|t| format!("T{}", t))
            .unwrap_or_else(|| "-".to_string());
        let track_attr = track_id_num
            .map(|t| format!(r#" data-track="{}""#, t))
            .unwrap_or_default();
        let offs_attr = match (start_opt, end_opt) {
            (Some(s), Some(e)) => format!(r#" data-start="{}" data-end="{}""#, s, e),
            _ => String::new(),
        };
        let neg = if signal.negated { " NEG" } else { "" };
        html.push_str(&format!(
            r#"<tr data-sid="S{sid}" data-label="{label}" data-surface="{surface}"{track_attr}{offs_attr} data-conf="{conf:.2}"><td class="id"><a href='#S{sid}'>S{sid}</a></td><td>{span}</td><td>{surface}</td><td>{label}{neg}</td><td class="conf">{conf:.2}</td><td class="id">{track}</td></tr>"#,
            sid = signal.id,
            span = span,
            surface = html_escape(&signal.surface),
            label = html_escape(signal.label.as_str()),
            neg = neg,
            conf = signal.confidence.value(),
            track = track_id,
            track_attr = track_attr,
            offs_attr = offs_attr
        ));
    }
    html.push_str(r#"</table></div>"#);

    // Level 2: Tracks table
    html.push_str(r#"<div class="panel" id="panel-tracks"><div class="panel-h"><h3>tracks (level 2)</h3><span class="toggle" data-toggle="panel-tracks">toggle</span></div><table id="tracks-table">"#);
    html.push_str(r#"<tr><th>id</th><th>canonical</th><th>type</th><th>|S|</th><th>signals</th><th>identity</th></tr>"#);
    for track in doc.tracks() {
        let entity_type = track
            .entity_type
            .as_ref()
            .map(|t| t.as_str())
            .unwrap_or("-");
        let signals: Vec<String> = track
            .signals
            .iter()
            .map(|s| format!("S{}", s.signal_id))
            .collect();
        let identity = doc
            .identity_for_track(track.id)
            .map(|i| format!("I{}", i.id))
            .unwrap_or_else(|| "-".to_string());
        let linked_badge = if track.identity_id.is_some() {
            r#"<span class="badge badge-y">y</span>"#
        } else {
            r#"<span class="badge badge-n">n</span>"#
        };
        html.push_str(&format!(
            r#"<tr data-tid="{tid}"><td class="id">T{tid}</td><td>{canonical_surface}</td><td>{etype}</td><td>{n}</td><td class="id">{sigs}</td><td class="id">{ident} {badge}</td></tr>"#,
            tid = track.id,
            canonical_surface = html_escape(&track.canonical_surface),
            etype = html_escape(entity_type),
            n = track.len(),
            sigs = html_escape(&signals.join(" ")),
            ident = identity,
            badge = linked_badge
        ));
    }
    html.push_str(r#"</table></div>"#);

    // Level 3: Identities table
    html.push_str(r#"<div class="panel" id="panel-identities"><div class="panel-h"><h3>identities (level 3)</h3><span class="toggle" data-toggle="panel-identities">toggle</span></div><table>"#);
    html.push_str(r#"<tr><th>id</th><th>name</th><th>type</th><th>kb</th><th>kb_id</th><th>aliases</th></tr>"#);
    for identity in doc.identities() {
        let kb = identity.kb_name.as_deref().unwrap_or("-");
        let kb_id = identity.kb_id.as_deref().unwrap_or("-");
        let entity_type = identity
            .entity_type
            .as_ref()
            .map(|t| t.as_str())
            .unwrap_or("-");
        let aliases = if identity.aliases.is_empty() {
            "-".to_string()
        } else {
            identity.aliases.join(", ")
        };
        html.push_str(&format!(
            r#"<tr><td class="id">I{}</td><td>{}</td><td>{}</td><td class="kb">{}</td><td class="kb">{}</td><td>{}</td></tr>"#,
            identity.id, html_escape(&identity.canonical_name), entity_type, kb, kb_id, html_escape(&aliases)
        ));
    }
    html.push_str(r#"</table></div>"#);

    html.push_str(r#"</div>"#); // end grid

    // Signal-Track-Identity mapping (compact view)
    html.push_str(r#"<h2>hierarchy trace</h2><div class="panel"><table>"#);
    html.push_str(r#"<tr><th>signal</th><th></th><th>track</th><th></th><th>identity</th><th>kb_id</th></tr>"#);
    for signal in doc.signals() {
        let track = doc.track_for_signal(signal.id);
        let identity = doc.identity_for_signal(signal.id);

        let track_str = track
            .map(|t| format!("T{} \"{}\"", t.id, html_escape(&t.canonical_surface)))
            .unwrap_or_else(|| "-".to_string());
        let identity_str = identity
            .map(|i| format!("I{} \"{}\"", i.id, html_escape(&i.canonical_name)))
            .unwrap_or_else(|| "-".to_string());
        let kb_str = identity
            .and_then(|i| i.kb_id.as_ref())
            .map(|s| s.as_str())
            .unwrap_or("-");

        html.push_str(&format!(
            r#"<tr><td>S{} "{}"</td><td class="arrow">→</td><td>{}</td><td class="arrow">→</td><td>{}</td><td class="kb">{}</td></tr>"#,
            signal.id, html_escape(&signal.surface), track_str, identity_str, kb_str
        ));
    }
    html.push_str(r#"</table></div>"#);

    // Minimal JS: click a signal row → highlight that mention in the text box.
    // Also support filtering signals by substring match.
    html.push_str(r#"<script>
(() => {
  // Index signal metadata from the signals table, and map signal/track → text elements.
  const signalMeta = new Map();
  document.querySelectorAll('#signals-table tr[data-sid]').forEach((row) => {
    const sid = row.getAttribute('data-sid');
    if (!sid) return;
    signalMeta.set(sid, {
      sid,
      label: row.getAttribute('data-label') || '',
      surface: row.getAttribute('data-surface') || '',
      conf: row.getAttribute('data-conf') || '',
      start: row.getAttribute('data-start'),
      end: row.getAttribute('data-end'),
      track: row.getAttribute('data-track'),
    });
  });

  const signalEls = new Map();
  const addSignalEl = (sid, el) => {
    if (!sid || !el) return;
    const arr = signalEls.get(sid) || [];
    arr.push(el);
    signalEls.set(sid, arr);
  };
  // Old-style inline spans (non-overlapping renderer).
  document.querySelectorAll('span.e[data-sid]').forEach((el) => {
    addSignalEl(el.getAttribute('data-sid'), el);
  });
  // Segmented spans (overlap/discontinuous-safe renderer).
  document.querySelectorAll('span.seg[data-sids]').forEach((el) => {
    const raw = (el.getAttribute('data-sids') || '').trim();
    if (!raw) return;
    raw.split(/\s+/).filter(Boolean).forEach((sid) => addSignalEl(sid, el));
  });

  const trackEls = new Map();
  for (const [sid, els] of signalEls.entries()) {
    const meta = signalMeta.get(sid);
    const tid = meta ? meta.track : null;
    if (!tid) continue;
    const arr = trackEls.get(tid) || [];
    els.forEach((el) => arr.push(el));
    trackEls.set(tid, arr);
  }

  const selectionBody = document.getElementById('selection-body');
  const selectionHint = document.getElementById('selection-hint');
  const defaultHint = selectionHint ? (selectionHint.textContent || '') : '';
  const setSelection = (text) => {
    if (!selectionBody) return;
    selectionBody.textContent = text;
  };
  const setHint = (text) => {
    if (!selectionHint) return;
    selectionHint.textContent = text || defaultHint;
  };

  // Theme toggle: auto (prefers-color-scheme) → dark → light.
  const themeBtn = document.getElementById('theme-toggle');
  const themeKey = 'anno-theme';
  const applyTheme = (theme) => {
    const t = theme || 'auto';
    if (t === 'auto') {
      delete document.documentElement.dataset.theme;
    } else {
      document.documentElement.dataset.theme = t;
    }
    if (themeBtn) themeBtn.textContent = `theme: ${t}`;
  };
  const readTheme = () => {
    try { return localStorage.getItem(themeKey) || 'auto'; } catch (_) { return 'auto'; }
  };
  const writeTheme = (t) => {
    try { localStorage.setItem(themeKey, t); } catch (_) { /* ignore */ }
  };
  applyTheme(readTheme());
  if (themeBtn) {
    themeBtn.addEventListener('click', () => {
      const cur = readTheme();
      const next = cur === 'auto' ? 'dark' : (cur === 'dark' ? 'light' : 'auto');
      writeTheme(next);
      applyTheme(next);
    });
  }

  let activeSignalEls = [];
  let activeSignalRow = null;
  const clearActive = () => {
    if (activeSignalEls && activeSignalEls.length) {
      activeSignalEls.forEach((el) => el.classList.remove('e-active'));
    }
    if (activeSignalRow) activeSignalRow.classList.remove('e-active');
    activeSignalEls = [];
    activeSignalRow = null;
  };

  let activeTrack = null;
  let hoverTrack = null;

  const removeTrackClass = (tid, cls) => {
    if (!tid) return;
    const els = trackEls.get(tid);
    if (!els) return;
    els.forEach((el) => el.classList.remove(cls));
  };

  const addTrackClass = (tid, cls) => {
    if (!tid) return;
    const els = trackEls.get(tid);
    if (!els) return;
    els.forEach((el) => el.classList.add(cls));
  };

  const trackSize = (tid) => {
    const els = tid ? trackEls.get(tid) : null;
    return els ? els.length : 0;
  };

  const getTrackSelectionText = (tid) => {
    if (!tid) return 'track: - (untracked)';
    const row = document.querySelector(`#tracks-table tr[data-tid='${tid}']`);
    if (!row) return `track T${tid}`;
    const cells = row.querySelectorAll('td');
    const canonical = (cells[1]?.textContent || '').trim();
    const etype = (cells[2]?.textContent || '').trim();
    const count = (cells[3]?.textContent || '').trim();
    const sigs = (cells[4]?.textContent || '').trim();
    const lines = [];
    lines.push(`track T${tid} canonical="${canonical}" type="${etype}" mentions=${count}`);
    if (sigs) lines.push(`track signals: ${sigs}`);
    return lines.join('\n');
  };

  const renderTrackSelection = (tid) => setSelection(getTrackSelectionText(tid));

  const renderSignalSelectionBySid = (sid) => {
    const meta = signalMeta.get(sid);
    const label = meta ? (meta.label || '') : '';
    const conf = meta ? (meta.conf || '') : '';
    const start = meta ? meta.start : null;
    const end = meta ? meta.end : null;
    const tid = meta ? meta.track : null;
    const lines = [];
    if (start !== null && end !== null) {
      lines.push(`signal ${sid} label=${label} conf=${conf} span=[${start},${end})`);
    } else {
      lines.push(`signal ${sid} label=${label} conf=${conf}`);
    }
    if (meta && meta.surface) lines.push(`surface: ${meta.surface}`);
    lines.push('');
    lines.push(getTrackSelectionText(tid));
    setSelection(lines.join('\n'));
  };

  const setActiveTrack = (tid) => {
    const next = tid || null;
    if (activeTrack === next) return;
    removeTrackClass(activeTrack, 'e-track');
    activeTrack = next;
    if (activeTrack) addTrackClass(activeTrack, 'e-track');
    if (hoverTrack && activeTrack && hoverTrack === activeTrack) {
      removeTrackClass(hoverTrack, 'e-track-hover');
    }
  };

  const setHoverTrack = (tid) => {
    const next = tid || null;
    if (hoverTrack === next) return;
    removeTrackClass(hoverTrack, 'e-track-hover');
    hoverTrack = next;
    if (!hoverTrack) {
      setHint('');
      return;
    }
    if (activeTrack && hoverTrack === activeTrack) {
      setHint(`selected track T${hoverTrack} (${trackSize(hoverTrack)} mentions)`);
      return;
    }
    addTrackClass(hoverTrack, 'e-track-hover');
    setHint(`hover track T${hoverTrack} (${trackSize(hoverTrack)} mentions)`);
  };

  const emitToParentSpan = (start, end) => {
    try {
      if (!window.parent || window.parent === window) return;
      if (start === null || end === null) return;
      window.parent.postMessage({ type: 'anno:activate-span', start: Number(start), end: Number(end) }, '*');
    } catch (_) {
      // ignore: best-effort bridge for iframe containers
    }
  };

  const activateBySpan = (start, end, emit) => {
    if (start === null || end === null || start === undefined || end === undefined) return;
    // Prefer an exact signal span if present; otherwise fall back to the table row metadata.
    const el = document.querySelector(`span.e[data-sid][data-start='${start}'][data-end='${end}']`);
    if (el) {
      const sid = el.getAttribute('data-sid');
      if (sid) activateSignal(sid, emit);
      return;
    }
    const row = document.querySelector(`#signals-table tr[data-start='${start}'][data-end='${end}']`);
    if (!row) return;
    const sid = row.getAttribute('data-sid');
    if (!sid) return;
    activateSignal(sid, emit);
  };

  const activateSignal = (sid, emit) => {
    clearActive();
    const els = signalEls.get(sid) || [];
    if (!els.length) return;
    els.forEach((el) => el.classList.add('e-active'));
    activeSignalEls = els;
    const row = document.querySelector(`#signals-table tr[data-sid='${sid}']`);
    if (row) {
      row.classList.add('e-active');
      activeSignalRow = row;
    }
    const primaryEl = els[0];
    primaryEl.scrollIntoView({ block: 'center', behavior: 'smooth' });
    const meta = signalMeta.get(sid);
    const tid = meta ? meta.track : primaryEl.getAttribute('data-track');
    setActiveTrack(tid);
    renderSignalSelectionBySid(sid);
    if (emit && meta && meta.start !== null && meta.end !== null) {
      emitToParentSpan(meta.start, meta.end);
    }
  };

  // Table click
  const signalsTable = document.getElementById('signals-table');
  if (signalsTable) {
    signalsTable.addEventListener('click', (ev) => {
      const a = ev.target && ev.target.closest ? ev.target.closest("a[href^='#S']") : null;
      const row = ev.target && ev.target.closest ? ev.target.closest('tr[data-sid]') : null;
      const sid = (a && a.getAttribute('href') ? a.getAttribute('href').slice(1) : null) || (row ? row.getAttribute('data-sid') : null);
      if (!sid) return;
      ev.preventDefault();
      activateSignal(sid, true);
      history.replaceState(null, '', '#' + sid);
    });

    // Hover a signals row → preview track highlight
    signalsTable.addEventListener('mouseover', (ev) => {
      const row = ev.target && ev.target.closest ? ev.target.closest('tr[data-sid]') : null;
      if (!row) return;
      const tid = row.getAttribute('data-track');
      setHoverTrack(tid);
    });
    signalsTable.addEventListener('mouseout', (ev) => {
      const to = ev.relatedTarget;
      if (to && signalsTable.contains(to)) return;
      setHoverTrack(null);
    });
  }

  // Clicking an inline entity should also toggle active highlight.
  const pickPrimarySid = (el) => {
    if (!el) return null;
    const p = el.getAttribute('data-primary');
    if (p) return p;
    const raw = (el.getAttribute('data-sids') || '').trim();
    if (!raw) return null;
    const sids = raw.split(/\s+/).filter(Boolean);
    if (!sids.length) return null;
    // Prefer the shortest mention span from metadata.
    let best = sids[0];
    let bestLen = null;
    for (const sid of sids) {
      const meta = signalMeta.get(sid);
      const s = meta && meta.start !== null ? Number(meta.start) : null;
      const e = meta && meta.end !== null ? Number(meta.end) : null;
      const len = (s !== null && e !== null) ? (e - s) : null;
      if (len === null) continue;
      if (bestLen === null || len < bestLen) {
        best = sid;
        bestLen = len;
      }
    }
    return best;
  };

  document.addEventListener('click', (ev) => {
    const span = ev.target && ev.target.closest ? ev.target.closest('span.e[data-sid]') : null;
    if (span) {
      activateSignal(span.getAttribute('data-sid'), true);
      return;
    }
    const seg = ev.target && ev.target.closest ? ev.target.closest('span.seg[data-sids]') : null;
    if (!seg) return;
    activateSignal(pickPrimarySid(seg), true);
  });

  // Hover an inline entity → preview highlight its track
  document.addEventListener('mouseover', (ev) => {
    const span = ev.target && ev.target.closest ? ev.target.closest('span.e[data-sid]') : null;
    if (span) {
      setHoverTrack(span.getAttribute('data-track'));
      return;
    }
    const seg = ev.target && ev.target.closest ? ev.target.closest('span.seg[data-sids]') : null;
    if (!seg) return;
    const sid = pickPrimarySid(seg);
    const meta = sid ? signalMeta.get(sid) : null;
    setHoverTrack(meta ? meta.track : null);
  });
  document.addEventListener('mouseout', (ev) => {
    const span = ev.target && ev.target.closest ? ev.target.closest('span.e[data-sid]') : null;
    const seg = ev.target && ev.target.closest ? ev.target.closest('span.seg[data-sids]') : null;
    if (!span && !seg) return;
    const to = ev.relatedTarget;
    if (to && to.closest && (to.closest('span.e[data-sid]') || to.closest('span.seg[data-sids]'))) return;
    setHoverTrack(null);
  });

  // Clicking a track row → select track (highlight + details)
  const tracksTable = document.getElementById('tracks-table');
  if (tracksTable) {
    tracksTable.addEventListener('click', (ev) => {
      const row = ev.target && ev.target.closest ? ev.target.closest('tr[data-tid]') : null;
      if (!row) return;
      const tid = row.getAttribute('data-tid');
      setActiveTrack(tid);
      renderTrackSelection(tid);
    });
    tracksTable.addEventListener('mouseover', (ev) => {
      const row = ev.target && ev.target.closest ? ev.target.closest('tr[data-tid]') : null;
      if (!row) return;
      setHoverTrack(row.getAttribute('data-tid'));
    });
    tracksTable.addEventListener('mouseout', (ev) => {
      const to = ev.relatedTarget;
      if (to && tracksTable.contains(to)) return;
      setHoverTrack(null);
    });
  }

  // Filter
  const input = document.getElementById('signal-filter');
  const countEl = document.getElementById('signal-filter-count');
  if (input && signalsTable) {
    const update = () => {
      const q = (input.value || '').trim().toLowerCase();
      let shown = 0;
      const rows = signalsTable.querySelectorAll('tr[data-sid]');
      rows.forEach(row => {
        const sid = (row.getAttribute('data-sid') || '').toLowerCase();
        const label = (row.getAttribute('data-label') || '').toLowerCase();
        const surface = (row.getAttribute('data-surface') || '').toLowerCase();
        const ok = !q || sid.includes(q) || label.includes(q) || surface.includes(q);
        row.style.display = ok ? '' : 'none';
        if (ok) shown += 1;
      });
      if (countEl) countEl.textContent = shown + ' shown';
    };
    input.addEventListener('input', update);
    update();
  }

  // Panel toggles
  document.querySelectorAll('[data-toggle]').forEach(btn => {
    btn.addEventListener('click', () => {
      const id = btn.getAttribute('data-toggle');
      const panel = id ? document.getElementById(id) : null;
      if (!panel) return;
      panel.classList.toggle('panel-collapsed');
    });
  });

  // If URL hash is #S123, focus it.
  const hash = (location.hash || '').slice(1);
  if (hash && hash.startsWith('S')) activateSignal(hash, false);

  // Optional: allow parent pages (e.g., dataset explorers) to sync selection across iframes.
  window.addEventListener('message', (ev) => {
    const data = ev && ev.data ? ev.data : null;
    if (!data || data.type !== 'anno:activate-span') return;
    if (typeof data.start !== 'number' || typeof data.end !== 'number') return;
    activateBySpan(data.start, data.end, false);
  });
})();
</script>"#);

    html.push_str(r#"</body></html>"#);
    html
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn annotate_text_html(
    text: &str,
    signals: &[Signal<Location>],
    signal_to_track: &std::collections::HashMap<SignalId, TrackId>,
) -> String {
    let char_count = text.chars().count();
    if char_count == 0 {
        return String::new();
    }

    #[derive(Debug, Clone)]
    struct SigMeta {
        sid: String,
        label: String,
        conf: f64,
        track_id: Option<TrackId>,
        covered_len: usize,
    }

    #[derive(Debug, Clone)]
    struct Event {
        pos: usize,
        meta_idx: usize,
        delta: i32, // -1 end, +1 start
    }

    // Collect text segments for each signal (supports discontinuous spans).
    let mut metas: Vec<SigMeta> = Vec::new();
    let mut events: Vec<Event> = Vec::new();
    let mut boundaries: Vec<usize> = vec![0, char_count];

    for s in signals {
        let raw_segments: Vec<(usize, usize)> = match &s.location {
            Location::Text { start, end } => vec![(*start, *end)],
            Location::Discontinuous { segments } => segments.clone(),
        };
        if raw_segments.is_empty() {
            continue;
        }

        let mut cleaned: Vec<(usize, usize)> = Vec::new();
        let mut covered_len = 0usize;
        for (start, end) in raw_segments {
            let start = start.min(char_count);
            let end = end.min(char_count);
            if start >= end {
                continue;
            }
            covered_len = covered_len.saturating_add(end - start);
            cleaned.push((start, end));
        }
        if cleaned.is_empty() {
            continue;
        }

        let meta_idx = metas.len();
        let track_id = signal_to_track.get(&s.id).copied();
        metas.push(SigMeta {
            sid: format!("S{}", s.id),
            label: s.label.to_string(),
            conf: s.confidence.value(),
            track_id,
            covered_len,
        });

        for (start, end) in cleaned {
            boundaries.push(start);
            boundaries.push(end);
            events.push(Event {
                pos: start,
                meta_idx,
                delta: 1,
            });
            events.push(Event {
                pos: end,
                meta_idx,
                delta: -1,
            });
        }
    }

    if metas.is_empty() {
        return html_escape(text);
    }

    boundaries.sort_unstable();
    boundaries.dedup();
    events.sort_by(|a, b| a.pos.cmp(&b.pos).then_with(|| a.delta.cmp(&b.delta)));

    let mut active_counts: Vec<u32> = vec![0; metas.len()];
    let mut active: Vec<usize> = Vec::new();
    let mut ev_idx = 0usize;

    let mut result = String::new();

    for bi in 0..boundaries.len().saturating_sub(1) {
        let pos = boundaries[bi];
        // Apply all events at this boundary.
        while ev_idx < events.len() && events[ev_idx].pos == pos {
            let e = &events[ev_idx];
            let idx = e.meta_idx;
            if e.delta < 0 {
                if active_counts[idx] > 0 {
                    active_counts[idx] -= 1;
                    if active_counts[idx] == 0 {
                        active.retain(|&x| x != idx);
                    }
                }
            } else {
                active_counts[idx] += 1;
                if active_counts[idx] == 1 {
                    active.push(idx);
                }
            }
            ev_idx += 1;
        }

        let next = boundaries[bi + 1];
        if next <= pos {
            continue;
        }

        let seg_text: String = text.chars().skip(pos).take(next - pos).collect();
        if active.is_empty() {
            result.push_str(&html_escape(&seg_text));
            continue;
        }

        // Determine primary (for coloring + click default): shortest covered len, then highest conf.
        let primary_idx = active
            .iter()
            .copied()
            .min_by(|a, b| {
                metas[*a]
                    .covered_len
                    .cmp(&metas[*b].covered_len)
                    .then_with(|| {
                        metas[*b]
                            .conf
                            .partial_cmp(&metas[*a].conf)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
            })
            .unwrap_or(active[0]);
        let primary = &metas[primary_idx];

        let class = match primary.label.to_uppercase().as_str() {
            "PER" | "PERSON" => "e-per",
            "ORG" | "ORGANIZATION" | "COMPANY" => "e-org",
            "LOC" | "LOCATION" | "GPE" => "e-loc",
            "DATE" | "TIME" => "e-date",
            _ => "e-misc",
        };

        let mut sids: Vec<&str> = active.iter().map(|i| metas[*i].sid.as_str()).collect();
        sids.sort_unstable();
        let data_sids = sids.join(" ");

        let mut title = format!(
            "sids=[{}] primary={} [{}..{})",
            data_sids, primary.sid, pos, next
        );
        if let Some(t) = primary.track_id {
            title.push_str(&format!(" track=T{}", t));
        }

        result.push_str(&format!(
            r#"<span class="e seg {class}" data-sids="{sids}" data-start="{start}" data-end="{end}" data-primary="{primary}" title="{title}">{text}</span>"#,
            class = class,
            sids = html_escape(&data_sids),
            start = pos,
            end = next,
            primary = html_escape(&primary.sid),
            title = html_escape(&title),
            text = html_escape(&seg_text),
        ));
    }

    result
}

// =============================================================================
// Eval Comparison HTML Rendering
// =============================================================================

/// Comparison between gold (ground truth) and predicted entities.
#[derive(Debug, Clone)]
pub struct EvalComparison {
    /// Document text
    pub text: String,
    /// Gold/ground truth signals
    pub gold: Vec<Signal<Location>>,
    /// Predicted signals
    pub predicted: Vec<Signal<Location>>,
    /// Match results
    pub matches: Vec<EvalMatch>,
}

/// Result of matching a gold or predicted signal.
#[derive(Debug, Clone)]
pub enum EvalMatch {
    /// Exact match: gold and predicted align perfectly.
    Correct {
        /// Gold signal ID
        gold_id: SignalId,
        /// Predicted signal ID
        pred_id: SignalId,
    },
    /// Type mismatch: same span, different label.
    TypeMismatch {
        /// Gold signal ID
        gold_id: SignalId,
        /// Predicted signal ID
        pred_id: SignalId,
        /// Gold label
        gold_label: String,
        /// Predicted label
        pred_label: String,
    },
    /// Boundary error: overlapping but not exact span.
    BoundaryError {
        /// Gold signal ID
        gold_id: SignalId,
        /// Predicted signal ID
        pred_id: SignalId,
        /// Intersection over Union
        iou: f64,
    },
    /// False positive: predicted with no gold match.
    Spurious {
        /// Predicted signal ID
        pred_id: SignalId,
    },
    /// False negative: gold with no prediction.
    Missed {
        /// Gold signal ID
        gold_id: SignalId,
    },
}

impl EvalComparison {
    /// Create a comparison from gold and predicted entities.
    ///
    /// # Example
    ///
    /// ```rust
    /// use anno_core::core::grounded::{EvalComparison};
    /// use anno_core::{Signal, Location};
    ///
    /// let text = "Marie Curie won the Nobel Prize.";
    /// let gold = vec![
    ///     Signal::new(0, Location::text(0, 11), "Marie Curie", "PER", 1.0),
    ///     Signal::new(1, Location::text(20, 31), "Nobel Prize", "AWARD", 1.0),
    /// ];
    /// let pred = vec![
    ///     Signal::new(0, Location::text(0, 11), "Marie Curie", "PER", 0.95),
    /// ];
    /// let cmp = EvalComparison::compare(text, gold, pred);
    /// assert_eq!(cmp.matches.len(), 2); // 1 correct, 1 missed
    /// ```
    #[must_use]
    pub fn compare(
        text: &str,
        gold: Vec<Signal<Location>>,
        predicted: Vec<Signal<Location>>,
    ) -> Self {
        let mut matches = Vec::new();
        let mut gold_matched = vec![false; gold.len()];
        let mut pred_matched = vec![false; predicted.len()];

        // First pass: find exact matches and type mismatches
        for (pi, pred) in predicted.iter().enumerate() {
            let pred_offsets = match pred.location.text_offsets() {
                Some(o) => o,
                None => continue,
            };

            for (gi, g) in gold.iter().enumerate() {
                if gold_matched[gi] {
                    continue;
                }
                let gold_offsets = match g.location.text_offsets() {
                    Some(o) => o,
                    None => continue,
                };

                // Exact span match
                if pred_offsets == gold_offsets {
                    if pred.label == g.label {
                        matches.push(EvalMatch::Correct {
                            gold_id: g.id,
                            pred_id: pred.id,
                        });
                    } else {
                        matches.push(EvalMatch::TypeMismatch {
                            gold_id: g.id,
                            pred_id: pred.id,
                            gold_label: g.label.to_string(),
                            pred_label: pred.label.to_string(),
                        });
                    }
                    gold_matched[gi] = true;
                    pred_matched[pi] = true;
                    break;
                }
            }
        }

        // Second pass: find boundary errors (overlapping but not exact)
        for (pi, pred) in predicted.iter().enumerate() {
            if pred_matched[pi] {
                continue;
            }
            let pred_offsets = match pred.location.text_offsets() {
                Some(o) => o,
                None => continue,
            };

            for (gi, g) in gold.iter().enumerate() {
                if gold_matched[gi] {
                    continue;
                }
                let gold_offsets = match g.location.text_offsets() {
                    Some(o) => o,
                    None => continue,
                };

                // Check overlap
                if pred_offsets.0 < gold_offsets.1 && pred_offsets.1 > gold_offsets.0 {
                    let iou = pred.location.iou(&g.location).unwrap_or(0.0);
                    matches.push(EvalMatch::BoundaryError {
                        gold_id: g.id,
                        pred_id: pred.id,
                        iou,
                    });
                    gold_matched[gi] = true;
                    pred_matched[pi] = true;
                    break;
                }
            }
        }

        // Remaining unmatched predictions are spurious
        for (pi, pred) in predicted.iter().enumerate() {
            if !pred_matched[pi] {
                matches.push(EvalMatch::Spurious { pred_id: pred.id });
            }
        }

        // Remaining unmatched gold are missed
        for (gi, g) in gold.iter().enumerate() {
            if !gold_matched[gi] {
                matches.push(EvalMatch::Missed { gold_id: g.id });
            }
        }

        Self {
            text: text.to_string(),
            gold,
            predicted,
            matches,
        }
    }

    /// Count correct matches.
    #[must_use]
    pub fn correct_count(&self) -> usize {
        self.matches
            .iter()
            .filter(|m| matches!(m, EvalMatch::Correct { .. }))
            .count()
    }

    /// Count errors (type mismatch + boundary + spurious + missed).
    #[must_use]
    pub fn error_count(&self) -> usize {
        self.matches.len() - self.correct_count()
    }

    /// Calculate precision.
    #[must_use]
    pub fn precision(&self) -> f64 {
        if self.predicted.is_empty() {
            0.0
        } else {
            self.correct_count() as f64 / self.predicted.len() as f64
        }
    }

    /// Calculate recall.
    #[must_use]
    pub fn recall(&self) -> f64 {
        if self.gold.is_empty() {
            0.0
        } else {
            self.correct_count() as f64 / self.gold.len() as f64
        }
    }

    /// Calculate F1.
    #[must_use]
    pub fn f1(&self) -> f64 {
        let p = self.precision();
        let r = self.recall();
        if p + r > 0.0 {
            2.0 * p * r / (p + r)
        } else {
            0.0
        }
    }
}

/// Render an eval comparison as HTML.
///
/// Shows gold vs predicted side by side with error highlighting.
pub fn render_eval_html(cmp: &EvalComparison) -> String {
    render_eval_html_with_title(cmp, "eval comparison")
}

/// Render an eval comparison as HTML, with a custom title.
///
/// The title is used for both the page `<title>` and the top `<h1>`.
#[must_use]
pub fn render_eval_html_with_title(cmp: &EvalComparison, title: &str) -> String {
    let mut html = String::new();
    let title = html_escape(title);

    html.push_str(
        r#"<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="color-scheme" content="dark light">
"#,
    );
    html.push_str(&format!("<title>{}</title>", title));
    html.push_str(r#"
:root{
  color-scheme: light dark;
  --bg:#0a0a0a;
  --panel-bg:#0d0d0d;
  --text:#b0b0b0;
  --text-strong:#fff;
  --muted:#666;
  --border:#222;
  --border-strong:#333;
  --hover:#111;
  --input-bg:#080808;
  --active:#ddd;
  /* Eval entity colors (dark) */
  --gold-bg:#1a2e1a; --gold-br:#4a8a4a; --gold-tx:#88cc88;
  --pred-bg:#1a1a2e; --pred-br:#4a4a8a; --pred-tx:#8888cc;
  /* Match row borders */
  --m-ok:#4a8a4a;
  --m-type:#8a8a4a;
  --m-bound:#4a8a8a;
  --m-fp:#8a4a4a;
  --m-fn:#8a4a8a;
}
@media (prefers-color-scheme: light){
  :root{
    --bg:#ffffff;
    --panel-bg:#f7f7f7;
    --text:#222;
    --text-strong:#000;
    --muted:#555;
    --border:#d6d6d6;
    --border-strong:#c6c6c6;
    --hover:#f0f0f0;
    --input-bg:#ffffff;
    --active:#000;
    --gold-bg:#e9f7e9; --gold-br:#2f8a2f; --gold-tx:#1f5a1f;
    --pred-bg:#e9e9ff; --pred-br:#6c6cff; --pred-tx:#2b2b7a;
    --m-ok:#2f8a2f;
    --m-type:#8a7a2f;
    --m-bound:#2f7a8a;
    --m-fp:#8a2f2f;
    --m-fn:#6a2f8a;
  }
}
html[data-theme='dark']{
  --bg:#0a0a0a; --panel-bg:#0d0d0d; --text:#b0b0b0; --text-strong:#fff;
  --muted:#666; --border:#222; --border-strong:#333; --hover:#111; --input-bg:#080808; --active:#ddd;
  --gold-bg:#1a2e1a; --gold-br:#4a8a4a; --gold-tx:#88cc88;
  --pred-bg:#1a1a2e; --pred-br:#4a4a8a; --pred-tx:#8888cc;
  --m-ok:#4a8a4a; --m-type:#8a8a4a; --m-bound:#4a8a8a; --m-fp:#8a4a4a; --m-fn:#8a4a8a;
}
html[data-theme='light']{
  --bg:#ffffff; --panel-bg:#f7f7f7; --text:#222; --text-strong:#000;
  --muted:#555; --border:#d6d6d6; --border-strong:#c6c6c6; --hover:#f0f0f0; --input-bg:#ffffff; --active:#000;
  --gold-bg:#e9f7e9; --gold-br:#2f8a2f; --gold-tx:#1f5a1f;
  --pred-bg:#e9e9ff; --pred-br:#6c6cff; --pred-tx:#2b2b7a;
  --m-ok:#2f8a2f; --m-type:#8a7a2f; --m-bound:#2f7a8a; --m-fp:#8a2f2f; --m-fn:#6a2f8a;
}

<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font:12px/1.4 monospace;background:var(--bg);color:var(--text);padding:8px}
h1,h2{color:var(--text-strong);font-weight:normal;border-bottom:1px solid var(--border-strong);padding:4px 0;margin:16px 0 8px}
h1{font-size:14px}h2{font-size:12px}
table{width:100%;border-collapse:collapse;font-size:11px;margin:4px 0}
th,td{padding:4px 8px;text-align:left;border:1px solid var(--border)}
th{background:var(--hover);color:var(--muted);font-weight:normal;text-transform:uppercase;font-size:10px}
tr:hover{background:var(--hover)}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.panel{border:1px solid var(--border);background:var(--panel-bg);padding:8px}
.text-box{background:var(--input-bg);border:1px solid var(--border);padding:8px;white-space:pre-wrap;word-break:break-word;line-height:1.6}
.stats{display:flex;gap:24px;padding:8px 0;border-bottom:1px solid var(--border);margin-bottom:8px}
.stat{text-align:center}.stat-v{font-size:18px;color:var(--text-strong)}.stat-l{font-size:9px;color:var(--muted);text-transform:uppercase}
/* Entities */
.e{padding:1px 2px;border-bottom:2px solid}
.seg{cursor:pointer}
.e-gold{background:var(--gold-bg);border-color:var(--gold-br);color:var(--gold-tx)}
.e-pred{background:var(--pred-bg);border-color:var(--pred-br);color:var(--pred-tx)}
.e-active{outline:1px solid var(--active);outline-offset:1px}
/* Match types */
.correct{background:#1a2e1a;border-color:#4a8a4a}
.type-err{background:#2e2e1a;border-color:#8a8a4a}
.boundary{background:#1a2e2e;border-color:#4a8a8a}
.spurious{background:#2e1a1a;border-color:#8a4a4a}
.missed{background:#2e1a2e;border-color:#8a4a8a}
.match-row.correct{border-left:3px solid var(--m-ok)}
.match-row.type-err{border-left:3px solid var(--m-type)}
.match-row.boundary{border-left:3px solid var(--m-bound)}
.match-row.spurious{border-left:3px solid var(--m-fp)}
.match-row.missed{border-left:3px solid var(--m-fn)}
.match-row.active{outline:1px solid var(--muted)}
.sel{color:var(--muted);margin:6px 0 12px}
.metric{font-size:14px;color:var(--muted)}.metric b{color:var(--text-strong)}
</style>
</head>
<body>
"#);

    // Header (with theme toggle)
    html.push_str(&format!(
        "<div class=\"panel-h\" style=\"justify-content:space-between\"><h1>{}</h1><span class=\"toggle\" id=\"theme-toggle\" title=\"toggle theme (auto → dark → light)\">theme: auto</span></div>",
        title
    ));

    // Metrics bar
    html.push_str("<div class=\"stats\">");
    html.push_str(&format!(
        "<div class=\"stat\"><div class=\"stat-v\">{}</div><div class=\"stat-l\">gold</div></div>",
        cmp.gold.len()
    ));
    html.push_str(&format!(
        "<div class=\"stat\"><div class=\"stat-v\">{}</div><div class=\"stat-l\">predicted</div></div>",
        cmp.predicted.len()
    ));
    html.push_str(&format!(
        "<div class=\"stat\"><div class=\"stat-v\">{}</div><div class=\"stat-l\">correct</div></div>",
        cmp.correct_count()
    ));
    html.push_str(&format!(
        "<div class=\"stat\"><div class=\"stat-v\">{}</div><div class=\"stat-l\">errors</div></div>",
        cmp.error_count()
    ));
    html.push_str(&format!(
        "<div class=\"metric\">P=<b>{:.1}%</b> R=<b>{:.1}%</b> F1=<b>{:.1}%</b></div>",
        cmp.precision() * 100.0,
        cmp.recall() * 100.0,
        cmp.f1() * 100.0
    ));
    html.push_str("</div>");

    // Simple selection readout (helps debugging + browser-based verification)
    html.push_str("<div id=\"selection\" class=\"sel\">click a match row to select spans</div>");

    // Side-by-side text
    html.push_str("<div class=\"grid\">");

    // Gold panel
    html.push_str("<div class=\"panel\"><h2>gold (ground truth)</h2><div class=\"text-box\">");
    let gold_spans: Vec<EvalHtmlSpan> = cmp
        .gold
        .iter()
        .map(|s| {
            let (start, end) = s.location.text_offsets().unwrap_or((0, 0));
            EvalHtmlSpan {
                start,
                end,
                label: s.label.to_string(),
                class: "e-gold",
                id: format!("G{}", s.id),
            }
        })
        .collect();
    html.push_str(&annotate_text_spans(&cmp.text, &gold_spans));
    html.push_str("</div></div>");

    // Predicted panel
    html.push_str("<div class=\"panel\"><h2>predicted</h2><div class=\"text-box\">");
    let pred_spans: Vec<EvalHtmlSpan> = cmp
        .predicted
        .iter()
        .map(|s| {
            let (start, end) = s.location.text_offsets().unwrap_or((0, 0));
            EvalHtmlSpan {
                start,
                end,
                label: s.label.to_string(),
                class: "e-pred",
                id: format!("P{}", s.id),
            }
        })
        .collect();
    html.push_str(&annotate_text_spans(&cmp.text, &pred_spans));
    html.push_str("</div></div>");

    html.push_str("</div>");

    // Match table
    html.push_str("<h2>matches</h2><table>");
    html.push_str("<tr><th>type</th><th>gold</th><th>predicted</th><th>notes</th></tr>");

    for (mi, m) in cmp.matches.iter().enumerate() {
        let (class, mtype, gold_text, pred_text, notes, gid, pid) = match m {
            EvalMatch::Correct { gold_id, pred_id } => {
                let g = cmp.gold.iter().find(|s| s.id == *gold_id);
                let p = cmp.predicted.iter().find(|s| s.id == *pred_id);
                (
                    "correct",
                    "✓",
                    g.map(|s| format!("[{}] {}", s.label, s.surface()))
                        .unwrap_or_default(),
                    p.map(|s| format!("[{}] {}", s.label, s.surface()))
                        .unwrap_or_default(),
                    String::new(),
                    Some(format!("G{}", gold_id)),
                    Some(format!("P{}", pred_id)),
                )
            }
            EvalMatch::TypeMismatch {
                gold_id,
                pred_id,
                gold_label,
                pred_label,
            } => {
                let g = cmp.gold.iter().find(|s| s.id == *gold_id);
                let p = cmp.predicted.iter().find(|s| s.id == *pred_id);
                (
                    "type-err",
                    "type",
                    g.map(|s| format!("[{}] {}", s.label, s.surface()))
                        .unwrap_or_default(),
                    p.map(|s| format!("[{}] {}", s.label, s.surface()))
                        .unwrap_or_default(),
                    format!("{} → {}", gold_label, pred_label),
                    Some(format!("G{}", gold_id)),
                    Some(format!("P{}", pred_id)),
                )
            }
            EvalMatch::BoundaryError {
                gold_id,
                pred_id,
                iou,
            } => {
                let g = cmp.gold.iter().find(|s| s.id == *gold_id);
                let p = cmp.predicted.iter().find(|s| s.id == *pred_id);
                (
                    "boundary",
                    "bound",
                    g.map(|s| format!("[{}] \"{}\"", s.label, s.surface()))
                        .unwrap_or_default(),
                    p.map(|s| format!("[{}] \"{}\"", s.label, s.surface()))
                        .unwrap_or_default(),
                    format!("IoU={:.2}", iou),
                    Some(format!("G{}", gold_id)),
                    Some(format!("P{}", pred_id)),
                )
            }
            EvalMatch::Spurious { pred_id } => {
                let p = cmp.predicted.iter().find(|s| s.id == *pred_id);
                (
                    "spurious",
                    "FP",
                    String::new(),
                    p.map(|s| format!("[{}] {}", s.label, s.surface()))
                        .unwrap_or_default(),
                    "false positive".to_string(),
                    None,
                    Some(format!("P{}", pred_id)),
                )
            }
            EvalMatch::Missed { gold_id } => {
                let g = cmp.gold.iter().find(|s| s.id == *gold_id);
                (
                    "missed",
                    "FN",
                    g.map(|s| format!("[{}] {}", s.label, s.surface()))
                        .unwrap_or_default(),
                    String::new(),
                    "false negative".to_string(),
                    Some(format!("G{}", gold_id)),
                    None,
                )
            }
        };

        let mut data_attrs = String::new();
        if let Some(gid) = gid.as_deref() {
            data_attrs.push_str(&format!(" data-gid=\"{}\"", html_escape(gid)));
        }
        if let Some(pid) = pid.as_deref() {
            data_attrs.push_str(&format!(" data-pid=\"{}\"", html_escape(pid)));
        }

        html.push_str(&format!(
            "<tr id=\"M{mid}\" class=\"match-row {class}\"{attrs}><td><a class=\"match-link\" href=\"#M{mid}\">{mtype}</a></td><td>{gold}</td><td>{pred}</td><td>{notes}</td></tr>",
            mid = mi,
            class = class,
            attrs = data_attrs,
            mtype = html_escape(mtype),
            gold = html_escape(&gold_text),
            pred = html_escape(&pred_text),
            notes = html_escape(&notes)
        ));
    }
    html.push_str("</table>");

    html.push_str(
        r#"<script>
(() => {
  // Theme toggle: auto (prefers-color-scheme) → dark → light.
  const themeBtn = document.getElementById('theme-toggle');
  const themeKey = 'anno-theme';
  const applyTheme = (theme) => {
    const t = theme || 'auto';
    if (t === 'auto') {
      delete document.documentElement.dataset.theme;
    } else {
      document.documentElement.dataset.theme = t;
    }
    if (themeBtn) themeBtn.textContent = `theme: ${t}`;
  };
  const readTheme = () => {
    try { return localStorage.getItem(themeKey) || 'auto'; } catch (_) { return 'auto'; }
  };
  const writeTheme = (t) => {
    try { localStorage.setItem(themeKey, t); } catch (_) { /* ignore */ }
  };
  applyTheme(readTheme());
  if (themeBtn) {
    themeBtn.addEventListener('click', () => {
      const cur = readTheme();
      const next = cur === 'auto' ? 'dark' : (cur === 'dark' ? 'light' : 'auto');
      writeTheme(next);
      applyTheme(next);
    });
  }

  function clearActive() {
    document.querySelectorAll(".e-active").forEach((el) => el.classList.remove("e-active"));
    document.querySelectorAll("tr.match-row.active").forEach((el) => el.classList.remove("active"));
  }

  function findSpanEls(eid) {
    if (!eid) return [];
    // New segmented renderer: one span can be split across multiple elements.
    const els = Array.from(document.querySelectorAll(`span.e[data-eids~='${eid}']`));
    if (els.length) return els;
    // Back-compat: older HTML used a single element id.
    const single = document.getElementById(eid);
    return single ? [single] : [];
  }

  function activate(gid, pid, row) {
    clearActive();
    const gEls = findSpanEls(gid);
    const pEls = findSpanEls(pid);
    const sel = document.getElementById("selection");
    gEls.forEach((el) => el.classList.add("e-active"));
    pEls.forEach((el) => el.classList.add("e-active"));
    if (row) row.classList.add("active");
    if (sel) {
      const parts = [];
      if (gEls.length) {
        const lbl = gEls[0].dataset && gEls[0].dataset.label ? ` [${gEls[0].dataset.label}]` : "";
        parts.push(`gold ${gid}${lbl}`);
      }
      if (pEls.length) {
        const lbl = pEls[0].dataset && pEls[0].dataset.label ? ` [${pEls[0].dataset.label}]` : "";
        parts.push(`pred ${pid}${lbl}`);
      }
      sel.textContent = parts.length ? parts.join("  |  ") : "no selection";
    }
    if (row && row.id) {
      // Keep deep links stable without triggering navigation jump.
      // NOTE: single quotes avoid the Rust raw-string delimiter issue with quote+hash.
      history.replaceState(null, "", '#' + row.id);
    }
    const target = gEls[0] || pEls[0];
    if (target) target.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  document.querySelectorAll("tr.match-row[data-gid], tr.match-row[data-pid]").forEach((tr) => {
    tr.addEventListener("click", () => activate(tr.dataset.gid, tr.dataset.pid, tr));
  });

  document.querySelectorAll("a.match-link").forEach((a) => {
    a.addEventListener("click", (ev) => {
      ev.preventDefault();
      const tr = a.closest("tr.match-row");
      if (!tr) return;
      activate(tr.dataset.gid, tr.dataset.pid, tr);
    });
  });

  // Auto-select a match row if the URL has a deep link (e.g. #M12).
  const hash = (location.hash || "").slice(1);
  if (hash && hash.startsWith("M")) {
    const tr = document.getElementById(hash);
    if (tr && tr.classList && tr.classList.contains("match-row")) {
      activate(tr.dataset.gid, tr.dataset.pid, tr);
    }
  }
})();
</script>"#,
    );

    html.push_str("</body></html>");
    html
}

/// Annotate text with multiple labeled spans.
#[derive(Debug, Clone)]
struct EvalHtmlSpan {
    start: usize,
    end: usize,
    label: String,
    class: &'static str,
    id: String,
}

fn annotate_text_spans(text: &str, spans: &[EvalHtmlSpan]) -> String {
    let char_count = text.chars().count();
    if char_count == 0 || spans.is_empty() {
        return html_escape(text);
    }

    #[derive(Debug, Clone)]
    struct Meta {
        id: String,
        label: String,
        class: &'static str,
        len: usize,
    }
    #[derive(Debug, Clone)]
    struct Event {
        pos: usize,
        meta_idx: usize,
        delta: i32,
    }

    let mut metas: Vec<Meta> = Vec::with_capacity(spans.len());
    let mut events: Vec<Event> = Vec::new();
    let mut boundaries: Vec<usize> = vec![0, char_count];

    for s in spans {
        let start = s.start.min(char_count);
        let end = s.end.min(char_count);
        if start >= end {
            continue;
        }
        let meta_idx = metas.len();
        metas.push(Meta {
            id: s.id.clone(),
            label: s.label.to_string(),
            class: s.class,
            len: end - start,
        });
        boundaries.push(start);
        boundaries.push(end);
        events.push(Event {
            pos: start,
            meta_idx,
            delta: 1,
        });
        events.push(Event {
            pos: end,
            meta_idx,
            delta: -1,
        });
    }

    if metas.is_empty() {
        return html_escape(text);
    }

    boundaries.sort_unstable();
    boundaries.dedup();
    events.sort_by(|a, b| a.pos.cmp(&b.pos).then_with(|| a.delta.cmp(&b.delta)));

    let mut active_counts: Vec<u32> = vec![0; metas.len()];
    let mut active: Vec<usize> = Vec::new();
    let mut ev_idx = 0usize;
    let mut result = String::new();

    for bi in 0..boundaries.len().saturating_sub(1) {
        let pos = boundaries[bi];
        while ev_idx < events.len() && events[ev_idx].pos == pos {
            let e = &events[ev_idx];
            let idx = e.meta_idx;
            if e.delta < 0 {
                if active_counts[idx] > 0 {
                    active_counts[idx] -= 1;
                    if active_counts[idx] == 0 {
                        active.retain(|&x| x != idx);
                    }
                }
            } else {
                active_counts[idx] += 1;
                if active_counts[idx] == 1 {
                    active.push(idx);
                }
            }
            ev_idx += 1;
        }

        let next = boundaries[bi + 1];
        if next <= pos {
            continue;
        }

        let seg_text: String = text.chars().skip(pos).take(next - pos).collect();
        if active.is_empty() {
            result.push_str(&html_escape(&seg_text));
            continue;
        }

        let primary_idx = active
            .iter()
            .copied()
            .min_by_key(|i| metas[*i].len)
            .unwrap_or(active[0]);
        let primary = &metas[primary_idx];
        let mut eids: Vec<&str> = active.iter().map(|i| metas[*i].id.as_str()).collect();
        eids.sort_unstable();
        let data_eids = eids.join(" ");

        let title = format!(
            "eids=[{}] primary={} [{}..{})",
            data_eids, primary.id, pos, next
        );
        result.push_str(&format!(
            "<span class=\"e seg {class}\" data-eids=\"{eids}\" data-label=\"{label}\" data-start=\"{start}\" data-end=\"{end}\" title=\"{title}\">{text}</span>",
            class = primary.class,
            eids = html_escape(&data_eids),
            label = html_escape(&primary.label),
            start = pos,
            end = next,
            title = html_escape(&title),
            text = html_escape(&seg_text)
        ));
    }

    result
}

// =============================================================================
// URL/Text Input Processing
// =============================================================================

/// Options for processing arbitrary input.
#[derive(Debug, Clone, Default)]
pub struct ProcessOptions {
    /// Labels to extract (empty = all)
    pub labels: Vec<String>,
    /// Confidence threshold
    pub threshold: f32,
}

/// Result of processing input.
#[derive(Debug)]
pub struct ProcessResult {
    /// The document with signals
    pub document: GroundedDocument,
    /// Whether validation passed
    pub valid: bool,
    /// Any validation errors
    pub errors: Vec<SignalValidationError>,
}

impl ProcessResult {
    /// Render as HTML.
    #[must_use]
    pub fn to_html(&self) -> String {
        render_document_html(&self.document)
    }
}

// =============================================================================
// Corpus: Multi-Document Operations
// =============================================================================

/// A corpus of grounded documents for cross-document operations.
///
/// Enables inter-document coreference resolution and entity linking
/// across multiple documents.
#[derive(Debug, Clone)]
pub struct Corpus {
    documents: std::collections::HashMap<String, GroundedDocument>,
    identities: std::collections::HashMap<IdentityId, Identity>,
    next_identity_id: IdentityId,
}

impl Corpus {
    /// Create a new empty corpus.
    #[must_use]
    pub fn new() -> Self {
        Self {
            documents: std::collections::HashMap::new(),
            identities: std::collections::HashMap::new(),
            next_identity_id: IdentityId::ZERO,
        }
    }

    /// Get all identities in the corpus.
    #[must_use]
    pub fn identities(&self) -> &std::collections::HashMap<IdentityId, Identity> {
        &self.identities
    }

    /// Get an identity by ID.
    #[must_use]
    pub fn get_identity(&self, id: IdentityId) -> Option<&Identity> {
        self.identities.get(&id)
    }

    /// Add an identity to the corpus and return its ID.
    ///
    /// This method assigns the next available identity ID and inserts the identity.
    /// Used by coalescing operations to create cross-document identities.
    pub fn add_identity(&mut self, mut identity: Identity) -> IdentityId {
        let id = self.next_identity_id;
        identity.id = id;
        self.identities.insert(id, identity);
        self.next_identity_id += 1;
        id
    }

    /// Get the next identity ID that would be assigned.
    ///
    /// This is used by coalescing operations to reserve identity IDs.
    #[must_use]
    pub fn next_identity_id(&self) -> IdentityId {
        self.next_identity_id
    }

    /// Get all documents in the corpus.
    ///
    /// Returns an iterator over all documents.
    pub fn documents(&self) -> impl Iterator<Item = &GroundedDocument> {
        self.documents.values()
    }

    /// Get a document by ID.
    ///
    /// Returns `None` if the document doesn't exist.
    #[must_use]
    pub fn get_document(&self, doc_id: &str) -> Option<&GroundedDocument> {
        self.documents.get(doc_id)
    }

    /// Get a mutable reference to a document by ID.
    ///
    /// Returns `None` if the document doesn't exist.
    pub fn get_document_mut(&mut self, doc_id: &str) -> Option<&mut GroundedDocument> {
        self.documents.get_mut(doc_id)
    }

    /// Add a document to the corpus.
    ///
    /// If a document with the same ID already exists, it will be replaced.
    /// Returns the document ID.
    pub fn add_document(&mut self, document: GroundedDocument) -> String {
        let doc_id = document.id.clone();
        self.documents.insert(doc_id.clone(), document);
        doc_id
    }

    /// Link a track to a knowledge base entity.
    ///
    /// This is the entity linking (NED) operation. It creates or updates
    /// an identity with KB information.
    ///
    /// # Parameters
    ///
    /// * `track_ref` - Reference to the track to link
    /// * `kb_name` - Knowledge base name (e.g., "wikidata")
    /// * `kb_id` - Knowledge base entity ID (e.g., "Q7186")
    /// * `canonical_name` - Canonical name from KB
    ///
    /// # Returns
    ///
    /// The identity ID (new or existing), or an error if the track reference is invalid.
    ///
    /// # Errors
    ///
    /// Returns `Error::TrackRef` if:
    /// - The document ID doesn't exist in the corpus
    /// - The track ID doesn't exist in the document
    pub fn link_track_to_kb(
        &mut self,
        track_ref: &TrackRef,
        kb_name: impl Into<String>,
        kb_id: impl Into<String>,
        canonical_name: impl Into<String>,
    ) -> super::Result<IdentityId> {
        use super::error::Error;

        let doc = self.documents.get_mut(&track_ref.doc_id).ok_or_else(|| {
            Error::track_ref(format!(
                "Document '{}' not found in corpus",
                track_ref.doc_id
            ))
        })?;
        let track = doc.get_track(track_ref.track_id).ok_or_else(|| {
            Error::track_ref(format!(
                "Track {} not found in document '{}'",
                track_ref.track_id, track_ref.doc_id
            ))
        })?;

        let kb_name_str = kb_name.into();
        let kb_id_str = kb_id.into();
        let canonical_name_str = canonical_name.into();

        // Check if track already has an identity
        let identity_id = if let Some(existing_id) = track.identity_id {
            // Update existing identity with KB info if it exists in corpus
            if let Some(identity) = self.identities.get_mut(&existing_id) {
                identity.kb_id = Some(kb_id_str.clone());
                identity.kb_name = Some(kb_name_str.clone());
                identity.canonical_name = canonical_name_str.clone();

                // Update source
                identity.source = Some(match identity.source.take() {
                    Some(IdentitySource::CrossDocCoref { track_refs }) => IdentitySource::Hybrid {
                        track_refs,
                        kb_name: kb_name_str.clone(),
                        kb_id: kb_id_str.clone(),
                    },
                    _ => IdentitySource::KnowledgeBase {
                        kb_name: kb_name_str.clone(),
                        kb_id: kb_id_str.clone(),
                    },
                });

                existing_id
            } else {
                // Identity ID exists in document but not in corpus - this is inconsistent.
                // This can happen if:
                // 1. Document was added to corpus with pre-existing identities
                // 2. Identity was removed from corpus but document still references it
                //
                // Fix: Create new identity and update ALL references in the document
                // to ensure consistency between document and corpus state.
                let new_id = self.next_identity_id;
                self.next_identity_id += 1;

                let identity = Identity {
                    id: new_id,
                    canonical_name: canonical_name_str,
                    entity_type: track.entity_type.clone(),
                    kb_id: Some(kb_id_str.clone()),
                    kb_name: Some(kb_name_str.clone()),
                    description: None,
                    embedding: track.embedding.clone(),
                    aliases: Vec::new(),
                    confidence: track.cluster_confidence,
                    source: Some(IdentitySource::KnowledgeBase {
                        kb_name: kb_name_str,
                        kb_id: kb_id_str,
                    }),
                };

                self.identities.insert(new_id, identity);
                // Update the track's identity reference to point to the new identity
                // This ensures document and corpus are consistent
                doc.link_track_to_identity(track_ref.track_id, new_id);
                new_id
            }
        } else {
            // Create new identity
            let new_id = self.next_identity_id;
            self.next_identity_id += 1;

            let identity = Identity {
                id: new_id,
                canonical_name: canonical_name_str,
                entity_type: track.entity_type.clone(),
                kb_id: Some(kb_id_str.clone()),
                kb_name: Some(kb_name_str.clone()),
                description: None,
                embedding: track.embedding.clone(),
                aliases: Vec::new(),
                confidence: track.cluster_confidence,
                source: Some(IdentitySource::KnowledgeBase {
                    kb_name: kb_name_str,
                    kb_id: kb_id_str,
                }),
            };

            self.identities.insert(new_id, identity);
            doc.link_track_to_identity(track_ref.track_id, new_id);
            new_id
        };

        Ok(identity_id)
    }
}

impl Default for Corpus {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)] // unwrap() is acceptable in test code
    use super::*;
    use crate::EntityCategory;

    #[test]
    fn test_render_eval_html_has_interactive_hooks_and_is_unicode_safe() {
        // CJK example (multi-byte, no spaces)
        let text = "習近平在北京會見了普京。";

        let gold: Vec<Signal<Location>> = vec![
            Signal::new(SignalId::new(0), Location::text(0, 3), "習近平", "PER", 1.0),
            Signal::new(SignalId::new(1), Location::text(4, 6), "北京", "LOC", 1.0),
        ];

        // Intentionally introduce a type mismatch on 北京 to ensure a non-correct row exists.
        let predicted: Vec<Signal<Location>> = vec![
            Signal::new(SignalId::new(0), Location::text(0, 3), "習近平", "PER", 0.9),
            Signal::new(SignalId::new(1), Location::text(4, 6), "北京", "PER", 0.7),
        ];

        let cmp = EvalComparison::compare(text, gold, predicted);
        let html = render_eval_html_with_title(&cmp, "test");

        // Selection readout (useful for humans + enables browser-based verification)
        assert!(html.contains("id=\"selection\""));

        // Span IDs must be stable and distinct between gold/pred (segmented renderer uses data-eids)
        assert!(html.contains("data-eids=\"G0\""));
        assert!(html.contains("data-eids=\"P0\""));

        // Match rows must carry cross-links and be clickable
        assert!(html.contains("class=\"match-link\""));
        assert!(html.contains("href=\"#M0\""));
        assert!(html.contains("data-gid=\"G0\""));
        assert!(html.contains("data-pid=\"P0\""));

        // Ensure we didn't break Unicode rendering
        assert!(html.contains("北京"));
    }

    fn find_char_span(text: &str, needle: &str) -> Option<(usize, usize)> {
        let hay: Vec<char> = text.chars().collect();
        let pat: Vec<char> = needle.chars().collect();
        if pat.is_empty() || hay.len() < pat.len() {
            return None;
        }
        for i in 0..=(hay.len() - pat.len()) {
            if hay[i..(i + pat.len())] == pat[..] {
                return Some((i, i + pat.len()));
            }
        }
        None
    }

    #[test]
    fn test_annotate_text_html_supports_overlaps_discontinuous_and_unicode() {
        // Intentionally include multiple scripts and an overlap + discontinuous mention.
        let text = "Marie Curie met Cher in Paris. 習近平在北京會見了普京。 \
التقى محمد بن سلمان في الرياض. Путин встретился с Си Цзиньпином в Москве. \
प्रधान मंत्री शर्मा दिल्ली में मिले। severe pain ... in abdomen.";

        // Overlap: "Marie Curie" contains "Curie"
        let (m0s, m0e) = find_char_span(text, "Marie Curie").unwrap();
        let (m1s, m1e) = find_char_span(text, "Curie").unwrap();

        // Discontinuous: "pain" + "abdomen"
        let pain = find_char_span(text, "pain").unwrap();
        let abdomen = find_char_span(text, "abdomen").unwrap();

        let signals: Vec<Signal<Location>> = vec![
            Signal::new(
                SignalId::new(0),
                Location::text(m0s, m0e),
                "Marie Curie",
                "PER",
                0.9,
            ),
            Signal::new(
                SignalId::new(1),
                Location::text(m1s, m1e),
                "Curie",
                "PER",
                0.8,
            ),
            Signal::new(
                SignalId::new(2),
                Location::Discontinuous {
                    segments: vec![pain, abdomen],
                },
                "pain … abdomen",
                "SYMPTOM",
                0.7,
            ),
        ];

        let html = annotate_text_html(text, &signals, &std::collections::HashMap::new());

        // Overlap must be representable (segment(s) covered by both S0 and S1).
        assert!(html.contains("data-sids=\"S0 S1\"") || html.contains("data-sids=\"S1 S0\""));

        // Discontinuous mention should be present in two places (at least one segment contains S2).
        assert!(html.contains("data-sids=\"S2\""));

        // Unicode safety: the original text snippets should still appear.
        assert!(html.contains("北京"));
        assert!(html.contains("Москве"));
        assert!(html.contains("शर्मा"));
        assert!(html.contains("محمد"));
    }

    #[test]
    fn test_location_text_iou() {
        let l1 = Location::text(0, 10);
        let l2 = Location::text(5, 15);
        let iou = l1.iou(&l2).unwrap();
        // Intersection: [5, 10) = 5 chars
        // Union: [0, 15) = 15 chars
        // IoU = 5/15 = 0.333...
        assert!((iou - 0.333).abs() < 0.01);
    }

    #[test]
    fn test_signal_creation() {
        let signal: Signal<Location> =
            Signal::new(0, Location::text(0, 11), "Marie Curie", "Person", 0.95);
        assert_eq!(signal.surface, "Marie Curie");
        assert_eq!(signal.label, "Person".into());
        assert!((signal.confidence.value() - 0.95).abs() < 0.001);
        assert!(!signal.negated);
    }

    #[test]
    fn test_signal_with_linguistic_features() {
        let signal: Signal<Location> =
            Signal::new(0, Location::text(0, 10), "not a doctor", "Occupation", 0.8)
                .negated()
                .with_quantifier(Quantifier::Existential)
                .with_modality(Modality::Symbolic);

        assert!(signal.negated);
        assert_eq!(signal.quantifier, Some(Quantifier::Existential));
        assert_eq!(signal.modality, Modality::Symbolic);
    }

    #[test]
    fn test_track_formation() {
        let mut track = Track::new(0, "Marie Curie");
        track.add_signal(0, 0);
        track.add_signal(1, 1);
        track.add_signal(2, 2);

        assert_eq!(track.len(), 3);
        assert!(!track.is_singleton());
        assert!(!track.is_empty());
    }

    #[test]
    fn test_identity_creation() {
        let identity = Identity::from_kb(0, "Marie Curie", "wikidata", "Q7186")
            .with_type("Person")
            .with_embedding(vec![0.1, 0.2, 0.3]);

        assert_eq!(identity.canonical_name, "Marie Curie");
        assert_eq!(identity.kb_id, Some("Q7186".to_string()));
        assert_eq!(identity.kb_name, Some("wikidata".to_string()));
        assert!(identity.embedding.is_some());
    }

    #[test]
    fn test_grounded_document_hierarchy() {
        let mut doc = GroundedDocument::new(
            "doc1",
            "Marie Curie won the Nobel Prize. She was a physicist.",
        );

        // Add signals (Level 1)
        let s1 = doc.add_signal(Signal::new(
            0,
            Location::text(0, 12),
            "Marie Curie",
            "Person",
            0.95,
        ));
        let s2 = doc.add_signal(Signal::new(
            1,
            Location::text(38, 41),
            "She",
            "Person",
            0.88,
        ));
        let s3 = doc.add_signal(Signal::new(
            2,
            Location::text(17, 29),
            "Nobel Prize",
            "Award",
            0.92,
        ));

        // Form tracks (Level 2)
        let mut track1 = Track::new(0, "Marie Curie");
        track1.add_signal(s1, 0);
        track1.add_signal(s2, 1);
        let track1_id = doc.add_track(track1);

        let mut track2 = Track::new(1, "Nobel Prize");
        track2.add_signal(s3, 0);
        doc.add_track(track2);

        // Add identity (Level 3)
        let identity = Identity::from_kb(0, "Marie Curie", "wikidata", "Q7186");
        let identity_id = doc.add_identity(identity);
        doc.link_track_to_identity(track1_id, identity_id);

        // Verify hierarchy traversal
        assert_eq!(doc.signals().len(), 3);
        assert_eq!(doc.tracks().count(), 2);
        assert_eq!(doc.identities().count(), 1);

        // Signal → Track
        let track = doc.track_for_signal(s1).unwrap();
        assert_eq!(track.canonical_surface, "Marie Curie");
        assert_eq!(track.len(), 2);

        // Track → Identity
        let identity = doc.identity_for_track(track1_id).unwrap();
        assert_eq!(identity.kb_id, Some("Q7186".to_string()));

        // Signal → Identity (transitive)
        let identity = doc.identity_for_signal(s1).unwrap();
        assert_eq!(identity.canonical_name, "Marie Curie");
    }

    #[test]
    fn test_modality_variants() {
        assert_eq!(Modality::default(), Modality::Symbolic);
        assert_eq!(Location::text(0, 10).modality(), Modality::Symbolic);
    }

    #[test]
    fn test_location_from_span() {
        let span = Span::Text { start: 0, end: 10 };
        let location = Location::from(&span);
        assert_eq!(location.text_offsets(), Some((0, 10)));
    }

    #[test]
    fn test_entity_roundtrip() {
        use super::EntityType;

        let entities = vec![
            Entity::new("Marie Curie", EntityType::Person, 0, 12, 0.95),
            Entity::new(
                "Nobel Prize",
                EntityType::custom("Award", EntityCategory::Creative),
                17,
                29,
                0.92,
            ),
        ];

        let doc =
            GroundedDocument::from_entities("doc1", "Marie Curie won the Nobel Prize.", &entities);
        let converted = doc.to_entities();

        assert_eq!(converted.len(), 2);
        assert_eq!(converted[0].text, "Marie Curie");
        assert_eq!(converted[1].text, "Nobel Prize");
    }

    #[test]
    fn test_signal_confidence_threshold() {
        let signal: Signal<Location> = Signal::new(0, Location::text(0, 10), "test", "Type", 0.75);
        assert!(signal.is_confident(0.5));
        assert!(signal.is_confident(0.75));
        assert!(!signal.is_confident(0.8));
    }

    #[test]
    fn test_document_filtering() {
        let mut doc = GroundedDocument::new("doc1", "Test text");

        // Add signals with different confidences and labels
        doc.add_signal(Signal::new(0, Location::text(0, 4), "high", "Person", 0.95));
        doc.add_signal(Signal::new(1, Location::text(5, 8), "low", "Person", 0.3));
        doc.add_signal(Signal::new(
            2,
            Location::text(9, 12),
            "org",
            "Organization",
            0.8,
        ));

        // Filter by confidence
        let confident = doc.confident_signals(0.5);
        assert_eq!(confident.len(), 2);

        // Filter by label
        let persons = doc.signals_with_label("Person");
        assert_eq!(persons.len(), 2);

        let orgs = doc.signals_with_label("Organization");
        assert_eq!(orgs.len(), 1);
    }

    #[test]
    fn test_untracked_signals() {
        let mut doc = GroundedDocument::new("doc1", "Test");

        let s1 = doc.add_signal(Signal::new(0, Location::text(0, 4), "a", "T", 0.9));
        let s2 = doc.add_signal(Signal::new(1, Location::text(5, 8), "b", "T", 0.9));
        let _s3 = doc.add_signal(Signal::new(2, Location::text(9, 12), "c", "T", 0.9));

        // Only track s1 and s2
        let mut track = Track::new(0, "a");
        track.add_signal(s1, 0);
        track.add_signal(s2, 1);
        doc.add_track(track);

        // s3 should be untracked
        assert_eq!(doc.untracked_signal_count(), 1);
        let untracked = doc.untracked_signals();
        assert_eq!(untracked.len(), 1);
        assert_eq!(untracked[0].surface, "c");
    }

    #[test]
    fn test_linked_unlinked_tracks() {
        let mut doc = GroundedDocument::new("doc1", "Test");

        let s1 = doc.add_signal(Signal::new(0, Location::text(0, 4), "a", "T", 0.9));
        let s2 = doc.add_signal(Signal::new(1, Location::text(5, 8), "b", "T", 0.9));

        let mut track1 = Track::new(0, "a");
        track1.add_signal(s1, 0);
        let track1_id = doc.add_track(track1);

        let mut track2 = Track::new(1, "b");
        track2.add_signal(s2, 0);
        doc.add_track(track2);

        // Link only track1 to an identity
        let identity = Identity::new(0, "Entity A");
        let identity_id = doc.add_identity(identity);
        doc.link_track_to_identity(track1_id, identity_id);

        assert_eq!(doc.linked_tracks().count(), 1);
        assert_eq!(doc.unlinked_tracks().count(), 1);
    }

    #[test]
    fn test_iou_edge_cases() {
        // No overlap
        let l1 = Location::text(0, 5);
        let l2 = Location::text(10, 15);
        assert_eq!(l1.iou(&l2), Some(0.0));

        // Complete overlap (identical)
        let l3 = Location::text(0, 10);
        let l4 = Location::text(0, 10);
        assert_eq!(l3.iou(&l4), Some(1.0));

        // One contains the other
        let l5 = Location::text(0, 20);
        let l6 = Location::text(5, 15);
        let iou = l5.iou(&l6).unwrap();
        // Intersection: 10, Union: 20
        assert!((iou - 0.5).abs() < 0.001);
    }

    // Note: Tests that depend on anno::eval::coref types have been moved to anno crate
    // (test_coref_chain_conversion, test_from_coref_document, test_coref_roundtrip)

    #[test]
    fn test_document_stats() {
        let mut doc = GroundedDocument::new("doc1", "Test document with entities.");

        // Add signals with varying properties
        let s1 = doc.add_signal(Signal::new(0, Location::text(0, 4), "Test", "Type", 0.9));
        let mut negated = Signal::new(0, Location::text(5, 13), "document", "Type", 0.8);
        negated.negated = true;
        let s2 = doc.add_signal(negated);
        let _s3 = doc.add_signal(Signal::new(
            0,
            Location::text(19, 27),
            "entities",
            "Type",
            0.7,
        ));

        // Create one track with 2 signals
        let mut track = Track::new(0, "Test");
        track.add_signal(s1, 0);
        track.add_signal(s2, 1);
        doc.add_track(track);

        // Add identity for the track
        let identity = Identity::new(0, "Test Entity");
        let identity_id = doc.add_identity(identity);
        doc.link_track_to_identity(0, identity_id);

        let stats = doc.stats();

        assert_eq!(stats.signal_count, 3);
        assert_eq!(stats.track_count, 1);
        assert_eq!(stats.identity_count, 1);
        assert_eq!(stats.linked_track_count, 1);
        assert_eq!(stats.untracked_count, 1); // s3 is untracked
        assert_eq!(stats.negated_count, 1);
        assert!((stats.avg_confidence - 0.8).abs() < 0.01); // (0.9 + 0.8 + 0.7) / 3
        assert!((stats.avg_track_size - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_batch_operations() {
        let mut doc = GroundedDocument::new("doc1", "Test document.");

        // Batch add signals
        let signals = vec![
            Signal::new(0, Location::text(0, 4), "Test", "Type", 0.9),
            Signal::new(0, Location::text(5, 13), "document", "Type", 0.8),
        ];
        let ids = doc.add_signals(signals);

        assert_eq!(ids.len(), 2);
        assert_eq!(doc.signals().len(), 2);

        // Create track from signal IDs
        let track_id = doc.create_track_from_signals("Test", &ids);
        assert!(track_id.is_some());

        let track = doc.get_track(track_id.unwrap()).unwrap();
        assert_eq!(track.len(), 2);
        assert_eq!(track.canonical_surface, "Test");
    }

    #[test]
    fn test_merge_tracks() {
        let mut doc = GroundedDocument::new("doc1", "John Smith works at Acme. He is great.");

        // Add signals
        let s1 = doc.add_signal(Signal::new(
            0,
            Location::text(0, 10),
            "John Smith",
            "Person",
            0.9,
        ));
        let s2 = doc.add_signal(Signal::new(0, Location::text(26, 28), "He", "Person", 0.8));

        // Create two separate tracks
        let mut track1 = Track::new(0, "John Smith");
        track1.add_signal(s1, 0);
        let track1_id = doc.add_track(track1);

        let mut track2 = Track::new(0, "He");
        track2.add_signal(s2, 0);
        let track2_id = doc.add_track(track2);

        assert_eq!(doc.tracks().count(), 2);

        // Merge tracks
        let merged_id = doc.merge_tracks(&[track1_id, track2_id]);
        assert!(merged_id.is_some());

        // Should now have only 1 track with 2 signals
        assert_eq!(doc.tracks().count(), 1);
        let merged = doc.get_track(merged_id.unwrap()).unwrap();
        assert_eq!(merged.len(), 2);
        assert_eq!(merged.canonical_surface, "John Smith"); // From first track
    }

    #[test]
    fn test_find_overlapping_pairs() {
        let mut doc = GroundedDocument::new("doc1", "New York City is great.");

        // Add overlapping signals (nested entity)
        doc.add_signal(Signal::new(
            0,
            Location::text(0, 13),
            "New York City",
            "Location",
            0.9,
        ));
        doc.add_signal(Signal::new(
            0,
            Location::text(0, 8),
            "New York",
            "Location",
            0.85,
        ));
        doc.add_signal(Signal::new(0, Location::text(17, 22), "great", "Adj", 0.7)); // Not overlapping

        let pairs = doc.find_overlapping_signal_pairs();

        // Should find one overlapping pair (New York City & New York)
        assert_eq!(pairs.len(), 1);
    }

    #[test]
    fn test_signals_in_range() {
        let mut doc = GroundedDocument::new("doc1", "John went to Paris and Berlin last year.");

        doc.add_signal(Signal::new(0, Location::text(0, 4), "John", "Person", 0.9));
        doc.add_signal(Signal::new(
            0,
            Location::text(13, 18),
            "Paris",
            "Location",
            0.9,
        ));
        doc.add_signal(Signal::new(
            0,
            Location::text(23, 29),
            "Berlin",
            "Location",
            0.9,
        ));
        doc.add_signal(Signal::new(
            0,
            Location::text(30, 39),
            "last year",
            "Date",
            0.8,
        ));

        // Find signals in the "Paris and Berlin" section
        let in_range = doc.signals_in_range(10, 30);
        assert_eq!(in_range.len(), 2); // Paris and Berlin

        let surfaces: Vec<_> = in_range.iter().map(|s| &s.surface).collect();
        assert!(surfaces.contains(&&"Paris".to_string()));
        assert!(surfaces.contains(&&"Berlin".to_string()));
    }

    #[test]
    fn test_quantifier_variants() {
        // Ensure all quantifier variants work
        let quantifiers = [
            Quantifier::Universal,
            Quantifier::Existential,
            Quantifier::None,
            Quantifier::Definite,
            Quantifier::Bare,
            Quantifier::Approximate,
            Quantifier::MinBound,
            Quantifier::MaxBound,
        ];

        for q in quantifiers {
            let signal: Signal<Location> =
                Signal::new(0, Location::text(0, 5), "test", "Type", 0.9).with_quantifier(q);

            assert_eq!(signal.quantifier, Some(q));
        }
    }

    #[test]
    fn test_location_modality_derivation() {
        assert_eq!(Location::text(0, 10).modality(), Modality::Symbolic);
        assert_eq!(
            Location::Discontinuous {
                segments: vec![(0, 5), (10, 15)]
            }
            .modality(),
            Modality::Symbolic
        );
    }

    // Note: CrossDocCluster conversion test moved to anno crate
    // since CrossDocCluster is defined in anno/src/eval/cdcr.rs
}

// =============================================================================
// Property-Based Tests
// =============================================================================
//
// These tests verify invariants that should hold for ALL valid inputs,
// not just specific examples. They catch edge cases that unit tests miss.

#[cfg(test)]
mod proptests {
    #![allow(clippy::unwrap_used)] // unwrap() is acceptable in property tests
    use super::*;
    use proptest::prelude::*;

    // -------------------------------------------------------------------------
    // Strategies for generating test data
    // -------------------------------------------------------------------------

    /// Generate valid confidence values in [0, 1].
    fn confidence_strategy() -> impl Strategy<Value = f32> {
        0.0f32..=1.0
    }

    /// Generate signal labels.
    fn label_strategy() -> impl Strategy<Value = String> {
        prop_oneof![
            Just("Person".to_string()),
            Just("Organization".to_string()),
            Just("Location".to_string()),
            Just("Date".to_string()),
            "[A-Z][a-z]{2,10}".prop_map(|s| s),
        ]
    }

    /// Generate surface forms (entity text).
    fn surface_strategy() -> impl Strategy<Value = String> {
        "[A-Za-z ]{1,50}".prop_map(|s| s.trim().to_string())
    }

    // -------------------------------------------------------------------------
    // IoU Properties (Intersection over Union)
    // -------------------------------------------------------------------------

    proptest! {
        /// IoU is symmetric: iou(a, b) == iou(b, a)
        #[test]
        fn iou_symmetric(
            start1 in 0usize..1000,
            len1 in 1usize..500,
            start2 in 0usize..1000,
            len2 in 1usize..500,
        ) {
            let a = Location::text(start1, start1 + len1);
            let b = Location::text(start2, start2 + len2);

            let iou_ab = a.iou(&b);
            let iou_ba = b.iou(&a);

            prop_assert_eq!(iou_ab, iou_ba, "IoU must be symmetric");
        }

        /// IoU is bounded: 0 <= iou <= 1
        #[test]
        fn iou_bounded(
            start1 in 0usize..1000,
            len1 in 1usize..500,
            start2 in 0usize..1000,
            len2 in 1usize..500,
        ) {
            let a = Location::text(start1, start1 + len1);
            let b = Location::text(start2, start2 + len2);

            if let Some(iou) = a.iou(&b) {
                prop_assert!(iou >= 0.0, "IoU must be non-negative: got {}", iou);
                prop_assert!(iou <= 1.0, "IoU must be at most 1: got {}", iou);
            }
        }

        /// Self-IoU is 1: iou(a, a) == 1
        #[test]
        fn iou_self_identity(start in 0usize..1000, len in 1usize..500) {
            let loc = Location::text(start, start + len);
            let iou = loc.iou(&loc).unwrap();
            prop_assert!(
                (iou - 1.0).abs() < 1e-6,
                "Self-IoU must be 1.0, got {}",
                iou
            );
        }

        /// Non-overlapping locations have IoU = 0
        #[test]
        fn iou_non_overlapping_zero(
            start1 in 0usize..500,
            len1 in 1usize..100,
        ) {
            let end1 = start1 + len1;
            let start2 = end1 + 100; // Guaranteed gap
            let len2 = 50;

            let a = Location::text(start1, end1);
            let b = Location::text(start2, start2 + len2);

            let iou = a.iou(&b).expect("bbox iou should be defined");
            prop_assert!(
                iou.abs() < 1e-6,
                "Non-overlapping IoU must be 0, got {}",
                iou
            );
        }


    }

    // -------------------------------------------------------------------------
    // Signal Properties
    // -------------------------------------------------------------------------

    proptest! {
        /// Confidence is always clamped to [0, 1]
        #[test]
        fn signal_confidence_clamped(raw_conf in -10.0f32..10.0) {
            let signal: Signal<Location> = Signal::new(
                0,
                Location::text(0, 10),
                "test",
                "Type",
                raw_conf,
            );

            prop_assert!(signal.confidence.value() >= 0.0, "Confidence below 0: {}", signal.confidence);
            prop_assert!(signal.confidence.value() <= 1.0, "Confidence above 1: {}", signal.confidence);
        }

        /// Signal with valid inputs preserves surface and label
        #[test]
        fn signal_preserves_data(
            surface in surface_strategy(),
            label in label_strategy(),
            conf in confidence_strategy(),
            start in 0usize..1000,
            len in 1usize..100,
        ) {
            let signal: Signal<Location> = Signal::new(
                0,
                Location::text(start, start + len),
                &surface,
                label.as_str(),
                conf,
            );

            prop_assert_eq!(&signal.surface, &surface);
            let want = crate::TypeLabel::from(label.as_str());
            prop_assert_eq!(signal.label, want);
        }

        /// Negation is idempotent: negated().negated() still has negated=true
        /// (Note: our API doesn't have an "un-negate", so calling negated() twice
        /// just keeps it negated - this tests that it doesn't toggle)
        #[test]
        fn signal_negation_stable(conf in confidence_strategy()) {
            let signal: Signal<Location> = Signal::new(
                0,
                Location::text(0, 10),
                "test",
                "Type",
                conf,
            )
            .negated();

            prop_assert!(signal.negated, "Signal should be negated after .negated()");
        }

        /// Text locations have Symbolic modality
        #[test]
        fn text_location_is_symbolic(
            start in 0usize..1000,
            len in 1usize..100,
        ) {
            let loc = Location::text(start, start + len);
            prop_assert_eq!(
                loc.modality(),
                Modality::Symbolic,
                "Text locations must be Symbolic"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Track Properties
    // -------------------------------------------------------------------------

    proptest! {
        /// Track length increases with each added signal
        #[test]
        fn track_length_monotonic(signal_count in 1usize..20) {
            let mut track = Track::new(0, "test");

            for i in 0..signal_count {
                track.add_signal(i, i as u32);
                prop_assert_eq!(
                    track.len(),
                    i + 1,
                    "Track length should be {} after adding {} signals",
                    i + 1,
                    i + 1
                );
            }
        }

        /// Track is never empty after adding a signal
        #[test]
        fn track_not_empty_after_add(canonical in surface_strategy()) {
            let mut track = Track::new(0, &canonical);
            prop_assert!(track.is_empty(), "New track should be empty");

            track.add_signal(0, 0);
            prop_assert!(!track.is_empty(), "Track should not be empty after add");
        }

        /// Track positions are stored correctly
        #[test]
        fn track_positions_stored(signal_count in 1usize..10) {
            let mut track = Track::new(0, "test");

            for i in 0..signal_count {
                track.add_signal(i, i as u32);
            }

            for (idx, signal_ref) in track.signals.iter().enumerate() {
                prop_assert_eq!(
                    signal_ref.position as usize,
                    idx,
                    "Signal position mismatch at index {}",
                    idx
                );
            }
        }
    }

    // -------------------------------------------------------------------------
    // GroundedDocument Properties
    // -------------------------------------------------------------------------

    proptest! {
        /// Signal IDs are unique and monotonically increasing
        #[test]
        fn document_signal_ids_monotonic(signal_count in 1usize..20) {
            let mut doc = GroundedDocument::new("test", "test text");

            let mut prev_id: Option<SignalId> = None;
            for i in 0..signal_count {
                let id = doc.add_signal(Signal::new(
                    999, // Should be overwritten
                    Location::text(i * 10, i * 10 + 5),
                    format!("entity_{}", i),
                    "Type",
                    0.9,
                ));

                if let Some(prev) = prev_id {
                    prop_assert!(id > prev, "Signal IDs should be monotonically increasing");
                }
                prev_id = Some(id);
            }
        }

        /// Track membership is consistent: if signal is in track, track_for_signal returns that track
        #[test]
        fn document_track_membership_consistent(signal_count in 1usize..5) {
            let mut doc = GroundedDocument::new("test", "test text");

            // Add signals
            let mut signal_ids = Vec::new();
            for i in 0..signal_count {
                let id = doc.add_signal(Signal::new(
                    0,
                    Location::text(i * 10, i * 10 + 5),
                    format!("entity_{}", i),
                    "Type",
                    0.9,
                ));
                signal_ids.push(id);
            }

            // Create track with all signals
            let mut track = Track::new(0, "canonical");
            for (pos, &id) in signal_ids.iter().enumerate() {
                track.add_signal(id, pos as u32);
            }
            let track_id = doc.add_track(track);

            // Verify membership
            for &signal_id in &signal_ids {
                let found_track = doc.track_for_signal(signal_id);
                prop_assert!(found_track.is_some(), "Signal should be in a track");
                prop_assert_eq!(
                    found_track.unwrap().id,
                    track_id,
                    "Signal should be in the correct track"
                );
            }
        }

        /// Identity linking is transitive: signal → track → identity
        #[test]
        fn document_identity_transitivity(signal_count in 1usize..3) {
            let mut doc = GroundedDocument::new("test", "test text");

            // Add signals
            let mut signal_ids = Vec::new();
            for i in 0..signal_count {
                let id = doc.add_signal(Signal::new(
                    0,
                    Location::text(i * 10, i * 10 + 5),
                    format!("entity_{}", i),
                    "Type",
                    0.9,
                ));
                signal_ids.push(id);
            }

            // Create track and identity
            let mut track = Track::new(0, "canonical");
            for (pos, &id) in signal_ids.iter().enumerate() {
                track.add_signal(id, pos as u32);
            }
            let track_id = doc.add_track(track);

            let identity = Identity::from_kb(0, "Entity", "wikidata", "Q123");
            let identity_id = doc.add_identity(identity);
            doc.link_track_to_identity(track_id, identity_id);

            // Verify transitivity
            for &signal_id in &signal_ids {
                let identity = doc.identity_for_signal(signal_id);
                prop_assert!(identity.is_some(), "Should find identity through signal");
                prop_assert_eq!(
                    identity.unwrap().id,
                    identity_id,
                    "Should find correct identity"
                );
            }
        }

        /// Untracked signals are correctly identified
        #[test]
        fn document_untracked_signals(total in 2usize..10, tracked in 0usize..10) {
            let tracked = tracked.min(total - 1); // Ensure at least one untracked
            let mut doc = GroundedDocument::new("test", "test text");

            // Add all signals
            let mut signal_ids = Vec::new();
            for i in 0..total {
                let id = doc.add_signal(Signal::new(
                    0,
                    Location::text(i * 10, i * 10 + 5),
                    format!("entity_{}", i),
                    "Type",
                    0.9,
                ));
                signal_ids.push(id);
            }

            // Track only some signals
            let mut track = Track::new(0, "canonical");
            for (pos, &id) in signal_ids.iter().take(tracked).enumerate() {
                track.add_signal(id, pos as u32);
            }
            if tracked > 0 {
                doc.add_track(track);
            }

            // Verify counts
            prop_assert_eq!(
                doc.untracked_signal_count(),
                total - tracked,
                "Wrong untracked count"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Roundtrip / Conversion Properties
    // -------------------------------------------------------------------------

    proptest! {
        /// Entity → GroundedDocument → Entities preserves core data
        #[test]
        fn entity_roundtrip_preserves_text(
            text in surface_strategy(),
            start in 0usize..1000,
            len in 1usize..100,
            conf in 0.0f64..=1.0,
        ) {
            use super::EntityType;

            let end = start + len;
            let entity = super::Entity::new(&text, EntityType::Person, start, end, conf);

            let doc = GroundedDocument::from_entities("test", "x".repeat(end + 10), &[entity]);
            let converted = doc.to_entities();

            prop_assert_eq!(converted.len(), 1, "Should have exactly one entity");
            prop_assert_eq!(&converted[0].text, &text, "Text should be preserved");
            prop_assert_eq!(converted[0].start(), start, "Start should be preserved");
            prop_assert_eq!(converted[0].end(), end, "End should be preserved");
        }

        // Note: Property test that depends on anno::eval::coref types has been moved to anno crate
        // (coref_roundtrip_preserves_count)
    }

    // -------------------------------------------------------------------------
    // Modality Invariants
    // -------------------------------------------------------------------------

    // -------------------------------------------------------------------------
    // Location Overlap Properties
    // -------------------------------------------------------------------------

    proptest! {
        /// Overlap is symmetric: overlaps(a, b) == overlaps(b, a)
        #[test]
        fn overlap_symmetric(
            start1 in 0usize..1000,
            len1 in 1usize..100,
            start2 in 0usize..1000,
            len2 in 1usize..100,
        ) {
            let a = Location::text(start1, start1 + len1);
            let b = Location::text(start2, start2 + len2);

            prop_assert_eq!(
                a.overlaps(&b),
                b.overlaps(&a),
                "Overlap must be symmetric"
            );
        }

        /// A location always overlaps with itself
        #[test]
        fn overlap_reflexive(start in 0usize..1000, len in 1usize..100) {
            let loc = Location::text(start, start + len);
            prop_assert!(loc.overlaps(&loc), "Location must overlap with itself");
        }

        /// If IoU > 0, then overlaps is true
        #[test]
        fn iou_implies_overlap(
            start1 in 0usize..500,
            len1 in 1usize..100,
            start2 in 0usize..500,
            len2 in 1usize..100,
        ) {
            let a = Location::text(start1, start1 + len1);
            let b = Location::text(start2, start2 + len2);

            if let Some(iou) = a.iou(&b) {
                if iou > 0.0 {
                    prop_assert!(
                        a.overlaps(&b),
                        "IoU > 0 should imply overlap"
                    );
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // DocumentStats Properties
    // -------------------------------------------------------------------------

    proptest! {
        /// Stats signal count matches actual count
        #[test]
        fn stats_signal_count_accurate(signal_count in 0usize..20) {
            let mut doc = GroundedDocument::new("test", "test");
            for i in 0..signal_count {
                doc.add_signal(Signal::new(
                    0,
                    Location::text(i * 10, i * 10 + 5),
                    "entity",
                    "Type",
                    0.9,
                ));
            }

            let stats = doc.stats();
            prop_assert_eq!(stats.signal_count, signal_count);
        }

        /// Stats track count matches actual count
        #[test]
        fn stats_track_count_accurate(track_count in 0usize..10) {
            let mut doc = GroundedDocument::new("test", "test");
            for i in 0..track_count {
                let id = doc.add_signal(Signal::new(
                    0,
                    Location::text(i * 10, i * 10 + 5),
                    "entity",
                    "Type",
                    0.9,
                ));
                let mut track = Track::new(0, format!("track_{}", i));
                track.add_signal(id, 0);
                doc.add_track(track);
            }

            let stats = doc.stats();
            prop_assert_eq!(stats.track_count, track_count);
        }

        /// Avg confidence is in [0, 1]
        #[test]
        fn stats_avg_confidence_bounded(
            confidences in proptest::collection::vec(0.0f32..=1.0, 1..10)
        ) {
            let mut doc = GroundedDocument::new("test", "test");
            for (i, conf) in confidences.iter().enumerate() {
                doc.add_signal(Signal::new(
                    0,
                    Location::text(i * 10, i * 10 + 5),
                    "entity",
                    "Type",
                    *conf,
                ));
            }

            let stats = doc.stats();
            prop_assert!(stats.avg_confidence >= 0.0);
            prop_assert!(stats.avg_confidence <= 1.0);
        }
    }

    // -------------------------------------------------------------------------
    // Batch Operations Properties
    // -------------------------------------------------------------------------

    proptest! {
        /// add_signals returns correct number of IDs
        #[test]
        fn batch_add_returns_all_ids(count in 1usize..10) {
            let mut doc = GroundedDocument::new("test", "test");
            let signals: Vec<Signal<Location>> = (0..count)
                .map(|i| Signal::new(0, Location::text(i * 10, i * 10 + 5), "e", "T", 0.9))
                .collect();

            let ids = doc.add_signals(signals);
            prop_assert_eq!(ids.len(), count);
            prop_assert_eq!(doc.signals().len(), count);
        }

        /// create_track_from_signals creates valid track
        #[test]
        fn create_track_valid(signal_count in 1usize..5) {
            let mut doc = GroundedDocument::new("test", "test");
            let mut signal_ids = Vec::new();
            for i in 0..signal_count {
                let id = doc.add_signal(Signal::new(
                    0,
                    Location::text(i * 10, i * 10 + 5),
                    "entity",
                    "Type",
                    0.9,
                ));
                signal_ids.push(id);
            }

            let track_id = doc.create_track_from_signals("canonical", &signal_ids);
            prop_assert!(track_id.is_some());

            let track = doc.get_track(track_id.unwrap());
            prop_assert!(track.is_some());
            prop_assert_eq!(track.unwrap().len(), signal_count);
        }

        /// Empty signal list returns None for track creation
        #[test]
        fn create_track_empty_returns_none(_dummy in 0..1) {
            let mut doc = GroundedDocument::new("test", "test");
            let track_id = doc.create_track_from_signals("canonical", &[]);
            prop_assert!(track_id.is_none());
        }
    }

    // -------------------------------------------------------------------------
    // Filtering Properties
    // -------------------------------------------------------------------------

    proptest! {
        /// signals_in_range returns only signals within range
        #[test]
        fn signals_in_range_within_bounds(
            range_start in 0usize..100,
            range_len in 10usize..50,
        ) {
            let range_end = range_start + range_len;
            let mut doc = GroundedDocument::new("test", "x".repeat(200));

            // Add signals: some inside, some outside
            doc.add_signal(Signal::new(0, Location::text(range_start + 2, range_start + 5), "inside", "T", 0.9));
            doc.add_signal(Signal::new(0, Location::text(0, 5), "before", "T", 0.9));
            doc.add_signal(Signal::new(0, Location::text(190, 195), "after", "T", 0.9));

            let in_range = doc.signals_in_range(range_start, range_end);

            for signal in &in_range {
                if let Some((start, end)) = signal.location.text_offsets() {
                    prop_assert!(start >= range_start, "Signal start {} < range start {}", start, range_start);
                    prop_assert!(end <= range_end, "Signal end {} > range end {}", end, range_end);
                }
            }
        }

        /// overlapping_signals is symmetric: if A overlaps B, then B's overlaps includes A's location
        #[test]
        fn overlapping_signals_symmetric(
            start1 in 10usize..50,
            len1 in 5usize..20,
            start2 in 10usize..50,
            len2 in 5usize..20,
        ) {
            let mut doc = GroundedDocument::new("test", "x".repeat(100));

            let loc1 = Location::text(start1, start1 + len1);
            let loc2 = Location::text(start2, start2 + len2);

            doc.add_signal(Signal::new(0, loc1.clone(), "A", "T", 0.9));
            doc.add_signal(Signal::new(0, loc2.clone(), "B", "T", 0.9));

            let overlaps_loc1 = doc.overlapping_signals(&loc1);
            let overlaps_loc2 = doc.overlapping_signals(&loc2);

            // If loc1 overlaps loc2, both should find each other
            if loc1.overlaps(&loc2) {
                prop_assert!(overlaps_loc1.len() >= 2, "Should find both when overlapping");
                prop_assert!(overlaps_loc2.len() >= 2, "Should find both when overlapping");
            }
        }
    }

    // -------------------------------------------------------------------------
    // Invariant: Modality count consistency
    // -------------------------------------------------------------------------

    proptest! {
        /// Sum of modality counts equals total signal count
        #[test]
        fn modality_counts_sum_to_total(
            symbolic_count in 0usize..5,
            iconic_count in 0usize..5,
        ) {
            let mut doc = GroundedDocument::new("test", "test");

            // Add symbolic signals
            for i in 0..symbolic_count {
                let mut signal = Signal::new(
                    0,
                    Location::text(i * 10, i * 10 + 5),
                    "entity",
                    "Type",
                    0.9,
                );
                signal.modality = Modality::Symbolic;
                doc.add_signal(signal);
            }

            // Add iconic-modality signals (modality overridden on text locations)
            for i in 0..iconic_count {
                let mut signal = Signal::new(
                    0,
                    Location::text(1000 + i * 10, 1000 + i * 10 + 5),
                    "entity",
                    "Type",
                    0.9,
                );
                signal.modality = Modality::Iconic;
                doc.add_signal(signal);
            }

            let stats = doc.stats();
            prop_assert_eq!(
                stats.symbolic_count + stats.iconic_count + stats.hybrid_count,
                stats.signal_count,
                "Modality counts should sum to total"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Invariant: Signal-Text Offset Consistency
    // -------------------------------------------------------------------------

    proptest! {
        /// Signals created via from_text are always valid
        #[test]
        fn from_text_always_valid(
            text in "[a-zA-Z ]{20,100}",
            surface_start in 0usize..15,
            surface_len in 1usize..8,
        ) {
            let text_char_len = text.chars().count();
            let surface_end = (surface_start + surface_len).min(text_char_len);
            let surface_start = surface_start.min(surface_end.saturating_sub(1));

            if surface_start < surface_end && surface_end <= text_char_len {
                let surface: String = text.chars()
                    .skip(surface_start)
                    .take(surface_end - surface_start)
                    .collect();

                if !surface.is_empty() {
                    // from_text should find the surface and create a valid signal
                    if let Some(signal) = Signal::<Location>::from_text(&text, &surface, "Test", 0.9) {
                        // The created signal MUST be valid
                        prop_assert!(
                            signal.validate_against(&text).is_none(),
                            "Signal created via from_text must be valid"
                        );
                    }
                }
            }
        }

        /// Validated add never allows invalid signals
        #[test]
        fn validated_add_rejects_invalid(
            text in "[a-z]{10,50}",
            wrong_surface in "[A-Z]{3,10}",
        ) {
            let mut doc = GroundedDocument::new("test", &text);

            // Create a signal with offsets pointing to different text than surface
            let signal = Signal::new(
                0,
                Location::text(0, wrong_surface.chars().count().min(text.chars().count())),
                wrong_surface.clone(),
                "Test",
                0.9,
            );

            // If text doesn't actually contain wrong_surface at offset 0,
            // validated add should reject it
            let expected: String = text.chars().take(wrong_surface.chars().count()).collect();
            if expected != wrong_surface {
                let result = doc.add_signal_validated(signal);
                prop_assert!(result.is_err(), "Should reject signal with mismatched surface");
            }
        }

        /// Round-trip: add_signal_from_text creates retrievable signals
        #[test]
        fn round_trip_signal_from_text(
            prefix in "[a-z]{5,20}",
            entity in "[A-Z][a-z]{3,10}",
            suffix in "[a-z]{5,20}",
        ) {
            let text = format!("{} {} {}", prefix, entity, suffix);
            let mut doc = GroundedDocument::new("test", &text);

            let id = doc.add_signal_from_text(&entity, "Entity", 0.9);
            prop_assert!(id.is_some(), "Should find entity in text");

            let signal = doc.signals().iter().find(|s| s.id == id.unwrap());
            prop_assert!(signal.is_some(), "Should retrieve added signal");

            let signal = signal.unwrap();
            prop_assert_eq!(signal.surface(), entity.as_str(), "Surface should match");

            // Validation MUST pass
            prop_assert!(
                doc.is_valid(),
                "Document should be valid after from_text add"
            );
        }

        /// Multiple occurrences: nth variant finds correct occurrence
        #[test]
        fn nth_occurrence_finds_correct(
            entity in "[A-Z][a-z]{2,5}",
            sep in " [a-z]+ ",
        ) {
            // Create text with multiple occurrences
            let text = format!("{}{}{}{}{}", entity, sep, entity, sep, entity);
            let mut doc = GroundedDocument::new("test", &text);

            // Find each occurrence
            for n in 0..3 {
                let id = doc.add_signal_from_text_nth(&entity, "Entity", 0.9, n);
                prop_assert!(id.is_some(), "Should find occurrence {}", n);
            }

            // 4th occurrence shouldn't exist
            let id = doc.add_signal_from_text_nth(&entity, "Entity", 0.9, 3);
            prop_assert!(id.is_none(), "Should NOT find 4th occurrence");

            // All signals should be valid
            prop_assert!(doc.is_valid(), "All signals should be valid");

            // Check offsets are distinct
            let offsets: Vec<_> = doc.signals()
                .iter()
                .filter_map(|s| s.text_offsets())
                .collect();
            let unique: std::collections::HashSet<_> = offsets.iter().collect();
            prop_assert_eq!(offsets.len(), unique.len(), "Each occurrence should have distinct offset");
        }
    }

    // =========================================================================
    // TrackStats Tests
    // =========================================================================

    #[test]
    fn test_track_stats_basic() {
        let text = "John met Mary. He said hello. John left.";
        let mut doc = GroundedDocument::new("test", text);
        let text_len = text.chars().count();

        // Add signals for "John" at positions 0 and 30
        let s1 = doc.add_signal(Signal::new(0, Location::text(0, 4), "John", "Person", 0.95));
        let s2 = doc.add_signal(Signal::new(
            0,
            Location::text(30, 34),
            "John",
            "Person",
            0.90,
        ));

        // Create track linking both Johns
        let track_id = doc.add_track(Track::new(0, "John".to_string()));
        doc.add_signal_to_track(s1, track_id, 0);
        doc.add_signal_to_track(s2, track_id, 1);

        // Get track and compute stats
        let track = doc.get_track(track_id).unwrap();
        let stats = track.compute_stats(&doc, text_len);

        assert_eq!(stats.chain_length, 2, "Two mentions");
        assert_eq!(stats.variation_count, 1, "One unique surface form");
        assert!(stats.spread > 0, "Spread should be positive");
        assert!(stats.relative_spread > 0.0 && stats.relative_spread < 1.0);
        assert!((stats.min_confidence - 0.90).abs() < 0.01);
        assert!((stats.max_confidence - 0.95).abs() < 0.01);
        assert!((stats.mean_confidence - 0.925).abs() < 0.01);
    }

    #[test]
    fn test_track_stats_singleton() {
        let text = "Paris is beautiful.";
        let mut doc = GroundedDocument::new("test", text);
        let text_len = text.chars().count();

        let s1 = doc.add_signal(Signal::new(
            0,
            Location::text(0, 5),
            "Paris",
            "Location",
            0.88,
        ));
        let track_id = doc.add_track(Track::new(0, "Paris".to_string()));
        doc.add_signal_to_track(s1, track_id, 0);

        let track = doc.get_track(track_id).unwrap();
        let stats = track.compute_stats(&doc, text_len);

        assert_eq!(stats.chain_length, 1);
        assert_eq!(stats.spread, 0, "Singleton has zero spread");
        assert_eq!(stats.first_position, stats.last_position);
        assert!((stats.min_confidence - stats.max_confidence).abs() < 0.001);
    }
}
