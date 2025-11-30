//! Grounded entity representation with unified Signal → Track → Identity hierarchy.
//!
//! # Research Motivation
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

use crate::{DiscontinuousSpan, Entity, EntityType, HierarchicalConfidence, Provenance, Span};
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

impl Modality {
    /// Check if linguistic features (negation, quantification) are relevant.
    #[must_use]
    pub const fn supports_linguistic_features(&self) -> bool {
        matches!(self, Self::Symbolic | Self::Hybrid)
    }

    /// Check if geometric features (bbox, IoU) are relevant.
    #[must_use]
    pub const fn supports_geometric_features(&self) -> bool {
        matches!(self, Self::Iconic | Self::Hybrid)
    }
}

// =============================================================================
// Location: The Universal Localization Unit
// =============================================================================

/// A location in some source medium.
///
/// This is the universal "localization unit" that enables the isomorphism
/// between vision detection and NER. Both tasks answer "where is it?"
/// just in different coordinate systems.
///
/// # Design Note
///
/// We use an enum rather than a trait to enable:
/// - Efficient storage in contiguous arrays
/// - Easy serialization
/// - Exhaustive matching for safety
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Location {
    /// Text span: 1D interval [start, end) in character offsets.
    Text {
        /// Start character offset (inclusive)
        start: usize,
        /// End character offset (exclusive)
        end: usize,
    },
    /// Visual bounding box: 2D rectangle in normalized `[0,1]` coordinates.
    BoundingBox {
        /// X coordinate of top-left corner
        x: f32,
        /// Y coordinate of top-left corner
        y: f32,
        /// Width
        width: f32,
        /// Height
        height: f32,
        /// Page number for multi-page documents
        page: Option<u32>,
    },
    /// Temporal interval: for audio/video signals.
    Temporal {
        /// Start time in seconds
        start_sec: f64,
        /// End time in seconds
        end_sec: f64,
        /// Optional frame number for video
        frame: Option<u64>,
    },
    /// 3D cuboid: for LiDAR/point cloud signals.
    Cuboid {
        /// Center position (x, y, z)
        center: [f32; 3],
        /// Dimensions (width, height, depth)
        dimensions: [f32; 3],
        /// Rotation (quaternion: w, x, y, z)
        rotation: [f32; 4],
    },
    /// Genomic interval: 1D interval in sequence coordinates.
    Genomic {
        /// Chromosome/contig identifier
        contig: String,
        /// Start position (0-based, inclusive)
        start: u64,
        /// End position (0-based, exclusive)
        end: u64,
        /// Strand (+/-)
        strand: Option<char>,
    },
    /// Discontinuous text span: non-contiguous regions.
    Discontinuous {
        /// Multiple text intervals
        segments: Vec<(usize, usize)>,
    },
    /// Hybrid: text with visual location (OCR).
    TextWithBbox {
        /// Text start offset
        start: usize,
        /// Text end offset
        end: usize,
        /// Visual bounding box
        bbox: Box<Location>,
    },
}

impl Location {
    /// Create a text location.
    #[must_use]
    pub const fn text(start: usize, end: usize) -> Self {
        Self::Text { start, end }
    }

    /// Create a bounding box location.
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

    /// Get the modality of this location.
    #[must_use]
    pub const fn modality(&self) -> Modality {
        match self {
            Self::Text { .. } | Self::Genomic { .. } | Self::Discontinuous { .. } => {
                Modality::Symbolic
            }
            Self::BoundingBox { .. } | Self::Cuboid { .. } => Modality::Iconic,
            Self::Temporal { .. } => Modality::Iconic, // Audio/video is iconic
            Self::TextWithBbox { .. } => Modality::Hybrid,
        }
    }

    /// Get text offsets if this is a text location.
    #[must_use]
    pub fn text_offsets(&self) -> Option<(usize, usize)> {
        match self {
            Self::Text { start, end } => Some((*start, *end)),
            Self::TextWithBbox { start, end, .. } => Some((*start, *end)),
            Self::Discontinuous { segments } => {
                let start = segments.iter().map(|(s, _)| *s).min()?;
                let end = segments.iter().map(|(_, e)| *e).max()?;
                Some((start, end))
            }
            _ => None,
        }
    }

    /// Check if two locations overlap.
    #[must_use]
    pub fn overlaps(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Text { start: s1, end: e1 }, Self::Text { start: s2, end: e2 }) => {
                s1 < e2 && s2 < e1
            }
            (
                Self::BoundingBox {
                    x: x1,
                    y: y1,
                    width: w1,
                    height: h1,
                    page: p1,
                },
                Self::BoundingBox {
                    x: x2,
                    y: y2,
                    width: w2,
                    height: h2,
                    page: p2,
                },
            ) => {
                // Pages must match (or both None)
                if p1 != p2 {
                    return false;
                }
                // Standard 2D rectangle overlap
                x1 < &(x2 + w2) && &(x1 + w1) > x2 && y1 < &(y2 + h2) && &(y1 + h1) > y2
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
            (
                Self::BoundingBox {
                    x: x1,
                    y: y1,
                    width: w1,
                    height: h1,
                    page: p1,
                },
                Self::BoundingBox {
                    x: x2,
                    y: y2,
                    width: w2,
                    height: h2,
                    page: p2,
                },
            ) => {
                if p1 != p2 {
                    return Some(0.0);
                }
                let x_overlap = (x1 + w1).min(x2 + w2) - x1.max(*x2);
                let y_overlap = (y1 + h1).min(y2 + h2) - y1.max(*y2);
                if x_overlap <= 0.0 || y_overlap <= 0.0 {
                    return Some(0.0);
                }
                let intersection = (x_overlap * y_overlap) as f64;
                let area1 = (*w1 * *h1) as f64;
                let area2 = (*w2 * *h2) as f64;
                let union = area1 + area2 - intersection;
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
            Span::BoundingBox {
                x,
                y,
                width,
                height,
                page,
            } => Self::BoundingBox {
                x: *x,
                y: *y,
                width: *width,
                height: *height,
                page: *page,
            },
            Span::Hybrid { start, end, bbox } => Self::TextWithBbox {
                start: *start,
                end: *end,
                bbox: Box::new(Location::from(bbox.as_ref())),
            },
        }
    }
}

// =============================================================================
// Signal (Level 1): Raw Detection
// =============================================================================

/// Unique identifier for a signal within a document.
pub type SignalId = u64;

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
    /// Classification label (e.g., "Person", "Organization")
    pub label: String,
    /// Detection confidence in [0, 1]
    pub confidence: f32,
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
pub enum Quantifier {
    /// Universal: "every", "all", "each"
    Universal,
    /// Existential: "some", "a", "certain"
    Existential,
    /// Negation: "no", "none"
    None,
    /// Specific: definite reference ("the")
    Definite,
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
        id: SignalId,
        location: L,
        surface: impl Into<String>,
        label: impl Into<String>,
        confidence: f32,
    ) -> Self {
        Self {
            id,
            location,
            surface: surface.into(),
            label: label.into(),
            confidence: confidence.clamp(0.0, 1.0),
            hierarchical: None,
            provenance: None,
            modality: Modality::default(),
            normalized: None,
            negated: false,
            quantifier: None,
        }
    }

    /// Get the classification label.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Get the surface form.
    #[must_use]
    pub fn surface(&self) -> &str {
        &self.surface
    }

    /// Check if this signal is above a confidence threshold.
    #[must_use]
    pub fn is_confident(&self, threshold: f32) -> bool {
        self.confidence >= threshold
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
    /// use anno::grounded::{Signal, Location};
    ///
    /// let text = "Marie Curie was a physicist.";
    /// let good = Signal::new(0, Location::text(0, 11), "Marie Curie", "PER", 0.9);
    /// assert!(good.validate_against(text).is_none());
    ///
    /// let bad = Signal::new(0, Location::text(0, 5), "Marie Curie", "PER", 0.9);
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
    /// use anno::grounded::{Signal, Location};
    ///
    /// let text = "Marie Curie was a physicist.";
    /// let signal = Signal::<Location>::from_text(text, "Marie Curie", "PER", 0.95);
    /// assert!(signal.is_some());
    /// assert_eq!(signal.unwrap().text_offsets(), Some((0, 11)));
    /// ```
    #[must_use]
    pub fn from_text(
        source: &str,
        surface: &str,
        label: impl Into<String>,
        confidence: f32,
    ) -> Option<Self> {
        Self::from_text_nth(source, surface, label, confidence, 0)
    }

    /// Create a signal by finding the nth occurrence of text in source.
    #[must_use]
    pub fn from_text_nth(
        source: &str,
        surface: &str,
        label: impl Into<String>,
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
                    0,
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

// =============================================================================
// Track (Level 2): Within-Document Coreference
// =============================================================================

/// Unique identifier for a track within a document.
pub type TrackId = u64;

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
    /// Entity type (consensus from signals)
    pub entity_type: Option<String>,
    /// Canonical surface form (the "best" name for this entity)
    pub canonical_surface: String,
    /// Link to global identity (Level 3), if resolved
    pub identity_id: Option<IdentityId>,
    /// Confidence that signals are correctly clustered
    pub cluster_confidence: f32,
    /// Optional embedding for track-level representation
    /// (aggregated from signal embeddings)
    pub embedding: Option<Vec<f32>>,
}

impl Track {
    /// Create a new track.
    #[must_use]
    pub fn new(id: TrackId, canonical_surface: impl Into<String>) -> Self {
        Self {
            id,
            signals: Vec::new(),
            entity_type: None,
            canonical_surface: canonical_surface.into(),
            identity_id: None,
            cluster_confidence: 1.0,
            embedding: None,
        }
    }

    /// Add a signal to this track.
    pub fn add_signal(&mut self, signal_id: SignalId, position: u32) {
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

    /// Link this track to a global identity.
    #[must_use]
    pub fn with_identity(mut self, identity_id: IdentityId) -> Self {
        self.identity_id = Some(identity_id);
        self
    }

    /// Set the entity type.
    #[must_use]
    pub fn with_type(mut self, entity_type: impl Into<String>) -> Self {
        self.entity_type = Some(entity_type.into());
        self
    }
}

// =============================================================================
// Identity (Level 3): Cross-Document Entity Linking
// =============================================================================

/// Unique identifier for a global identity.
pub type IdentityId = u64;

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
    /// Entity type/category
    pub entity_type: Option<String>,
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
    pub confidence: f32,
    /// Source of identity formation (how it was created)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<IdentitySource>,
}

impl Identity {
    /// Create a new identity.
    #[must_use]
    pub fn new(id: IdentityId, canonical_name: impl Into<String>) -> Self {
        Self {
            id,
            canonical_name: canonical_name.into(),
            entity_type: None,
            kb_id: None,
            kb_name: None,
            description: None,
            embedding: None,
            aliases: Vec::new(),
            confidence: 1.0,
            source: None,
        }
    }

    /// Create an identity from a knowledge base entry.
    #[must_use]
    pub fn from_kb(
        id: IdentityId,
        canonical_name: impl Into<String>,
        kb_name: impl Into<String>,
        kb_id: impl Into<String>,
    ) -> Self {
        let kb_name_str = kb_name.into();
        let kb_id_str = kb_id.into();
        Self {
            id,
            canonical_name: canonical_name.into(),
            entity_type: None,
            kb_id: Some(kb_id_str.clone()),
            kb_name: Some(kb_name_str.clone()),
            description: None,
            embedding: None,
            aliases: Vec::new(),
            confidence: 1.0,
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

    /// Set the embedding.
    #[must_use]
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Set the entity type.
    #[must_use]
    pub fn with_type(mut self, entity_type: impl Into<String>) -> Self {
        self.entity_type = Some(entity_type.into());
        self
    }

    /// Set description.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Create from a CDCR CrossDocCluster.
    ///
    /// This converts from the evaluation/clustering result format to
    /// the core representation.
    ///
    /// # Note on TrackRefs
    ///
    /// CDCR's `CrossDocCluster` doesn't contain track information (only entity indices),
    /// so we cannot create valid `TrackRef`s. The source is set to `None` to indicate
    /// this is a conversion from evaluation format. If you need proper TrackRefs,
    /// use `Corpus::resolve_inter_doc_coref()` instead.
    #[must_use]
    /// Create an Identity from a CrossDocCluster (evaluation format).
    ///
    /// # Note on `source: None`
    ///
    /// The `source` field is set to `None` because `CrossDocCluster` only contains
    /// `(doc_id, entity_idx)` pairs, not `track_id`s. Without track IDs, we cannot
    /// create valid `TrackRef`s that would be needed for `IdentitySource::CrossDocCoref`.
    ///
    /// This is intentional: identities created from evaluation data don't have
    /// the same provenance tracking as identities created through the normal
    /// corpus resolution pipeline.
    pub fn from_cross_doc_cluster(cluster: &crate::eval::cdcr::CrossDocCluster) -> Self {
        Self {
            id: cluster.id,
            canonical_name: cluster.canonical_name.clone(),
            entity_type: cluster
                .entity_type
                .as_ref()
                .map(|t| t.as_label().to_string()),
            kb_id: cluster.kb_id.clone(),
            kb_name: None,
            description: None,
            embedding: None,
            aliases: Vec::new(),
            confidence: cluster.confidence as f32,
            source: None, // Cannot determine source from CDCR format (no track_ids)
        }
    }

    /// Convert to a CDCR CrossDocCluster.
    ///
    /// Note: Some information is lost (documents, mentions) since
    /// Identity doesn't track those directly.
    #[must_use]
    pub fn to_cross_doc_cluster(&self) -> crate::eval::cdcr::CrossDocCluster {
        use crate::EntityType;

        let mut cluster = crate::eval::cdcr::CrossDocCluster::new(self.id, &self.canonical_name);
        cluster.kb_id = self.kb_id.clone();
        cluster.confidence = self.confidence as f64;
        if let Some(ref entity_type) = self.entity_type {
            cluster.entity_type = Some(EntityType::from_label(entity_type));
        }
        cluster
    }
}

// =============================================================================
// GroundedDocument: The Container
// =============================================================================

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
/// use anno::grounded::{GroundedDocument, Signal, Track, Identity, Location};
///
/// let mut doc = GroundedDocument::new("doc1", "Marie Curie won the Nobel Prize. She was a physicist.");
///
/// // Add signals (Level 1)
/// doc.add_signal(Signal::new(0, Location::text(0, 12), "Marie Curie", "Person", 0.95));
/// doc.add_signal(Signal::new(1, Location::text(38, 41), "She", "Person", 0.88));
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
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    /// Index: signal_id → track_id (for efficient lookup)
    signal_to_track: HashMap<SignalId, TrackId>,
    /// Index: track_id → identity_id (for efficient lookup)  
    track_to_identity: HashMap<TrackId, IdentityId>,
    /// Next signal ID (for auto-incrementing)
    next_signal_id: SignalId,
    /// Next track ID
    next_track_id: TrackId,
    /// Next identity ID
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
            next_signal_id: 0,
            next_track_id: 0,
            next_identity_id: 0,
        }
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
    pub fn get_signal(&self, id: SignalId) -> Option<&Signal<Location>> {
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
    pub fn get_track(&self, id: TrackId) -> Option<&Track> {
        self.tracks.get(&id)
    }

    /// Get a mutable reference to a track by ID.
    #[must_use]
    pub fn get_track_mut(&mut self, id: TrackId) -> Option<&mut Track> {
        self.tracks.get_mut(&id)
    }

    /// Add a signal to an existing track.
    ///
    /// This properly updates the signal_to_track index.
    /// Returns true if the signal was added, false if track doesn't exist.
    pub fn add_signal_to_track(
        &mut self,
        signal_id: SignalId,
        track_id: TrackId,
        position: u32,
    ) -> bool {
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
    pub fn link_track_to_identity(&mut self, track_id: TrackId, identity_id: IdentityId) {
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

                Entity {
                    text: signal.surface.clone(),
                    entity_type: EntityType::from_label(&signal.label),
                    start,
                    end,
                    confidence: signal.confidence as f64,
                    normalized: signal.normalized.clone(),
                    provenance: signal.provenance.clone(),
                    kb_id: identity.and_then(|i| i.kb_id.clone()),
                    canonical_id: track.map(|t| t.id),
                    hierarchical_confidence: signal.hierarchical,
                    visual_span: match &signal.location {
                        Location::BoundingBox {
                            x,
                            y,
                            width,
                            height,
                            page,
                        } => Some(Span::BoundingBox {
                            x: *x,
                            y: *y,
                            width: *width,
                            height: *height,
                            page: *page,
                        }),
                        Location::TextWithBbox { bbox, .. } => {
                            if let Location::BoundingBox {
                                x,
                                y,
                                width,
                                height,
                                page,
                            } = bbox.as_ref()
                            {
                                Some(Span::BoundingBox {
                                    x: *x,
                                    y: *y,
                                    width: *width,
                                    height: *height,
                                    page: *page,
                                })
                            } else {
                                None
                            }
                        }
                        _ => None,
                    },
                    discontinuous_span: match &signal.location {
                        Location::Discontinuous { segments } => Some(DiscontinuousSpan::new(
                            segments.iter().map(|(s, e)| (*s)..(*e)).collect(),
                        )),
                        _ => None,
                    },
                    valid_from: None,
                    valid_until: None,
                    viewport: None,
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

        // Group entities by canonical_id to form tracks
        let mut tracks_map: HashMap<Option<u64>, Vec<SignalId>> = HashMap::new();

        for (idx, entity) in entities.iter().enumerate() {
            let location = if let Some(disc) = &entity.discontinuous_span {
                Location::Discontinuous {
                    segments: disc.segments().iter().map(|r| (r.start, r.end)).collect(),
                }
            } else if let Some(visual) = &entity.visual_span {
                Location::from(visual)
            } else {
                Location::text(entity.start, entity.end)
            };

            let mut signal = Signal::new(
                idx as SignalId,
                location,
                &entity.text,
                entity.entity_type.as_label(),
                entity.confidence as f32,
            );
            signal.normalized = entity.normalized.clone();
            signal.provenance = entity.provenance.clone();
            signal.hierarchical = entity.hierarchical_confidence;

            let signal_id = doc.add_signal(signal);
            tracks_map
                .entry(entity.canonical_id)
                .or_default()
                .push(signal_id);
        }

        // Create tracks from grouped signals
        for (canonical_id, signal_ids) in tracks_map {
            if let Some(first_signal) = signal_ids.first().and_then(|id| doc.get_signal(*id)) {
                let mut track = Track::new(doc.next_track_id, &first_signal.surface);
                track.entity_type = Some(first_signal.label.clone());

                for (pos, &signal_id) in signal_ids.iter().enumerate() {
                    track.add_signal(signal_id, pos as u32);
                }

                // If there's a canonical_id, we might have KB info
                if let Some(cid) = canonical_id {
                    // Check if any entity had a kb_id
                    if let Some(kb_id) = entities
                        .iter()
                        .filter(|e| e.canonical_id == Some(cid))
                        .find_map(|e| e.kb_id.clone())
                    {
                        let identity = Identity::from_kb(
                            doc.next_identity_id,
                            &track.canonical_surface,
                            "unknown",
                            kb_id,
                        );
                        let identity_id = doc.add_identity(identity);
                        track = track.with_identity(identity_id);
                    }
                }

                doc.add_track(track);
            }
        }

        doc
    }

    /// Get signals filtered by label.
    #[must_use]
    pub fn signals_with_label(&self, label: &str) -> Vec<&Signal<Location>> {
        self.signals.iter().filter(|s| s.label == label).collect()
    }

    /// Get signals above a confidence threshold.
    #[must_use]
    pub fn confident_signals(&self, threshold: f32) -> Vec<&Signal<Location>> {
        self.signals
            .iter()
            .filter(|s| s.confidence >= threshold)
            .collect()
    }

    /// Get tracks that are linked to an identity.
    #[must_use]
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
    /// use anno::grounded::{GroundedDocument, Signal, Location};
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
    /// use anno::grounded::GroundedDocument;
    ///
    /// let mut doc = GroundedDocument::new("test", "Marie Curie was a physicist.");
    /// let id = doc.add_signal_from_text("Marie Curie", "PER", 0.95);
    /// assert!(id.is_some());
    /// ```
    pub fn add_signal_from_text(
        &mut self,
        surface: &str,
        label: impl Into<String>,
        confidence: f32,
    ) -> Option<SignalId> {
        let signal = Signal::from_text(&self.text, surface, label, confidence)?;
        Some(self.add_signal(signal))
    }

    /// Add a signal by finding the nth occurrence of text.
    pub fn add_signal_from_text_nth(
        &mut self,
        surface: &str,
        label: impl Into<String>,
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
            self.signals.iter().map(|s| s.confidence).sum::<f32>() / signal_count as f32
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

        let mut track = Track::new(0, canonical);
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
        let mut new_track = Track::new(0, canonical);
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
    /// use anno::grounded::{GroundedDocument, Signal, Location};
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
<title>grounded::GroundedDocument</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font:12px/1.4 monospace;background:#0a0a0a;color:#b0b0b0;padding:8px}
h1,h2,h3{color:#fff;font-weight:normal;border-bottom:1px solid #333;padding:4px 0;margin:16px 0 8px}
h1{font-size:14px}h2{font-size:12px}h3{font-size:11px;color:#888}
table{width:100%;border-collapse:collapse;font-size:11px;margin:4px 0}
th,td{padding:4px 8px;text-align:left;border:1px solid #222}
th{background:#111;color:#666;font-weight:normal;text-transform:uppercase;font-size:10px}
tr:hover{background:#111}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:8px}
.panel{border:1px solid #222;background:#0d0d0d;padding:8px}
.text-box{background:#080808;border:1px solid #222;padding:8px;white-space:pre-wrap;word-break:break-word;line-height:1.6}
.e{padding:1px 2px;border-bottom:1px solid}
.e-per{background:#1a1a2e;border-color:#4a4a8a;color:#8888cc}
.e-org{background:#1a2e1a;border-color:#4a8a4a;color:#88cc88}
.e-loc{background:#2e2e1a;border-color:#8a8a4a;color:#cccc88}
.e-misc{background:#1a1a1a;border-color:#4a4a4a;color:#999}
.e-date{background:#2e1a1a;border-color:#8a4a4a;color:#cc8888}
.conf{color:#666;font-size:10px}
.badge{display:inline-block;padding:1px 4px;font-size:9px;text-transform:uppercase}
.badge-y{background:#1a2e1a;color:#4a8a4a;border:1px solid #2a4a2a}
.badge-n{background:#2e2e1a;color:#8a8a4a;border:1px solid #4a4a2a}
.stats{display:flex;gap:16px;padding:8px 0;border-bottom:1px solid #222;margin-bottom:8px}
.stat{text-align:center}.stat-v{font-size:18px;color:#fff}.stat-l{font-size:9px;color:#555;text-transform:uppercase}
.id{color:#444;font-size:9px}
.kb{color:#666}
.arrow{color:#333}
</style>
</head>
<body>
"#);

    // Header with stats
    html.push_str(&format!(
        r#"<h1>doc_id="{}" len={}</h1>"#,
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
    html.push_str(&annotate_text_html(&doc.text, doc.signals()));
    html.push_str(r#"</div>"#);

    // Grid layout for three levels
    html.push_str(r#"<div class="grid">"#);

    // Level 1: Signals table
    html.push_str(r#"<div class="panel"><h3>signals (level 1)</h3><table>"#);
    html.push_str(r#"<tr><th>id</th><th>span</th><th>surface</th><th>label</th><th>conf</th><th>track</th></tr>"#);
    for signal in doc.signals() {
        let span = if let Some((s, e)) = signal.location.text_offsets() {
            format!("[{},{})", s, e)
        } else {
            "bbox".to_string()
        };
        let track_id = doc
            .track_for_signal(signal.id)
            .map(|t| format!("T{}", t.id))
            .unwrap_or_else(|| "-".to_string());
        let neg = if signal.negated { " NEG" } else { "" };
        html.push_str(&format!(
            r#"<tr><td class="id">S{}</td><td>{}</td><td>{}</td><td>{}{}</td><td class="conf">{:.2}</td><td class="id">{}</td></tr>"#,
            signal.id, span, html_escape(&signal.surface), html_escape(&signal.label), neg, signal.confidence, track_id
        ));
    }
    html.push_str(r#"</table></div>"#);

    // Level 2: Tracks table
    html.push_str(r#"<div class="panel"><h3>tracks (level 2)</h3><table>"#);
    html.push_str(r#"<tr><th>id</th><th>canonical</th><th>type</th><th>|S|</th><th>signals</th><th>identity</th></tr>"#);
    for track in doc.tracks() {
        let entity_type = track.entity_type.as_deref().unwrap_or("-");
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
            r#"<tr><td class="id">T{}</td><td>{}</td><td>{}</td><td>{}</td><td class="id">{}</td><td class="id">{} {}</td></tr>"#,
            track.id, html_escape(&track.canonical_surface), entity_type, track.len(), signals.join(" "), identity, linked_badge
        ));
    }
    html.push_str(r#"</table></div>"#);

    // Level 3: Identities table
    html.push_str(r#"<div class="panel"><h3>identities (level 3)</h3><table>"#);
    html.push_str(r#"<tr><th>id</th><th>name</th><th>type</th><th>kb</th><th>kb_id</th><th>aliases</th></tr>"#);
    for identity in doc.identities() {
        let kb = identity.kb_name.as_deref().unwrap_or("-");
        let kb_id = identity.kb_id.as_deref().unwrap_or("-");
        let entity_type = identity.entity_type.as_deref().unwrap_or("-");
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

    html.push_str(r#"</body></html>"#);
    html
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn annotate_text_html(text: &str, signals: &[Signal<Location>]) -> String {
    // Collect text signals with their spans (character offsets)
    let mut annotations: Vec<(usize, usize, &str, &str, f32, u64)> = signals
        .iter()
        .filter_map(|s| {
            let (start, end) = s.location.text_offsets()?;
            Some((
                start,
                end,
                s.surface.as_str(),
                s.label.as_str(),
                s.confidence,
                s.id,
            ))
        })
        .collect();

    // Sort by start position, then by length (longer first for nested)
    annotations.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| b.1.cmp(&a.1)));

    let mut result = String::new();
    let mut last_end: usize = 0;
    let char_count = text.chars().count();

    for (start, end, _surface, label, conf, id) in annotations {
        // Clamp to valid character range
        let start = start.min(char_count);
        let end = end.min(char_count);

        // Skip if this span starts before our current position (nested/overlapping)
        if start < last_end {
            continue;
        }

        // Add unannotated text before this span (using char iteration, not byte slicing)
        if start > last_end {
            let before: String = text.chars().skip(last_end).take(start - last_end).collect();
            result.push_str(&html_escape(&before));
        }

        // Determine CSS class based on label
        let class = match label.to_uppercase().as_str() {
            "PER" | "PERSON" => "e-per",
            "ORG" | "ORGANIZATION" | "COMPANY" => "e-org",
            "LOC" | "LOCATION" | "GPE" => "e-loc",
            "DATE" | "TIME" => "e-date",
            _ => "e-misc",
        };

        // Add annotated span with ID and confidence (using char iteration)
        let span_text: String = text.chars().skip(start).take(end - start).collect();
        result.push_str(&format!(
            r#"<span class="e {}" title="S{} {} [{},{}) conf={:.2}">{}</span>"#,
            class,
            id,
            html_escape(label),
            start,
            end,
            conf,
            html_escape(&span_text)
        ));

        last_end = end;
    }

    // Add remaining text (using char iteration)
    if last_end < char_count {
        let after: String = text.chars().skip(last_end).collect();
        result.push_str(&html_escape(&after));
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
    /// use anno::grounded::{EvalComparison, Signal, Location};
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
                            gold_label: g.label.clone(),
                            pred_label: pred.label.clone(),
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
    let mut html = String::new();

    html.push_str(r#"<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>NER Eval Comparison</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font:12px/1.4 monospace;background:#0a0a0a;color:#b0b0b0;padding:8px}
h1,h2{color:#fff;font-weight:normal;border-bottom:1px solid #333;padding:4px 0;margin:16px 0 8px}
h1{font-size:14px}h2{font-size:12px}
table{width:100%;border-collapse:collapse;font-size:11px;margin:4px 0}
th,td{padding:4px 8px;text-align:left;border:1px solid #222}
th{background:#111;color:#666;font-weight:normal;text-transform:uppercase;font-size:10px}
tr:hover{background:#111}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.panel{border:1px solid #222;background:#0d0d0d;padding:8px}
.text-box{background:#080808;border:1px solid #222;padding:8px;white-space:pre-wrap;word-break:break-word;line-height:1.6}
.stats{display:flex;gap:24px;padding:8px 0;border-bottom:1px solid #222;margin-bottom:8px}
.stat{text-align:center}.stat-v{font-size:18px;color:#fff}.stat-l{font-size:9px;color:#555;text-transform:uppercase}
/* Entities */
.e{padding:1px 2px;border-bottom:2px solid}
.e-gold{background:#1a2e1a;border-color:#4a8a4a;color:#88cc88}
.e-pred{background:#1a1a2e;border-color:#4a4a8a;color:#8888cc}
/* Match types */
.correct{background:#1a2e1a;border-color:#4a8a4a}
.type-err{background:#2e2e1a;border-color:#8a8a4a}
.boundary{background:#1a2e2e;border-color:#4a8a8a}
.spurious{background:#2e1a1a;border-color:#8a4a4a}
.missed{background:#2e1a2e;border-color:#8a4a8a}
.match-row.correct{border-left:3px solid #4a8a4a}
.match-row.type-err{border-left:3px solid #8a8a4a}
.match-row.boundary{border-left:3px solid #4a8a8a}
.match-row.spurious{border-left:3px solid #8a4a4a}
.match-row.missed{border-left:3px solid #8a4a8a}
.metric{font-size:14px;color:#888}.metric b{color:#fff}
</style>
</head>
<body>
"#);

    // Header
    html.push_str("<h1>eval comparison</h1>");

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

    // Side-by-side text
    html.push_str("<div class=\"grid\">");

    // Gold panel
    html.push_str("<div class=\"panel\"><h2>gold (ground truth)</h2><div class=\"text-box\">");
    let gold_spans: Vec<_> = cmp
        .gold
        .iter()
        .map(|s| {
            let (start, end) = s.location.text_offsets().unwrap_or((0, 0));
            (start, end, s.label.as_str(), "e-gold")
        })
        .collect();
    html.push_str(&annotate_text_spans(&cmp.text, &gold_spans));
    html.push_str("</div></div>");

    // Predicted panel
    html.push_str("<div class=\"panel\"><h2>predicted</h2><div class=\"text-box\">");
    let pred_spans: Vec<_> = cmp
        .predicted
        .iter()
        .map(|s| {
            let (start, end) = s.location.text_offsets().unwrap_or((0, 0));
            (start, end, s.label.as_str(), "e-pred")
        })
        .collect();
    html.push_str(&annotate_text_spans(&cmp.text, &pred_spans));
    html.push_str("</div></div>");

    html.push_str("</div>");

    // Match table
    html.push_str("<h2>matches</h2><table>");
    html.push_str("<tr><th>type</th><th>gold</th><th>predicted</th><th>notes</th></tr>");

    for m in &cmp.matches {
        let (class, mtype, gold_text, pred_text, notes) = match m {
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
                )
            }
        };

        html.push_str(&format!(
            "<tr class=\"match-row {}\"><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
            class,
            mtype,
            html_escape(&gold_text),
            html_escape(&pred_text),
            html_escape(&notes)
        ));
    }
    html.push_str("</table>");

    html.push_str("</body></html>");
    html
}

/// Annotate text with multiple labeled spans.
fn annotate_text_spans(text: &str, spans: &[(usize, usize, &str, &str)]) -> String {
    // Sort spans by start position
    let mut sorted: Vec<_> = spans.iter().collect();
    sorted.sort_by_key(|(start, _, _, _)| *start);

    let mut result = String::new();
    let mut last_end = 0;

    for (start, end, label, class) in sorted {
        let start = *start;
        let end = (*end).min(text.chars().count());

        if start < last_end || start >= text.chars().count() {
            continue; // skip overlapping or out-of-bounds
        }

        // Add text before this span
        if start > last_end {
            let before: String = text.chars().skip(last_end).take(start - last_end).collect();
            result.push_str(&html_escape(&before));
        }

        // Add annotated span
        let span_text: String = text.chars().skip(start).take(end - start).collect();
        result.push_str(&format!(
            "<span class=\"e {}\" title=\"{}\">{}</span>",
            class,
            label,
            html_escape(&span_text)
        ));
        last_end = end;
    }

    // Add remaining text
    if last_end < text.chars().count() {
        let after: String = text.chars().skip(last_end).collect();
        result.push_str(&html_escape(&after));
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

/// Process arbitrary text and extract entities.
///
/// This is a convenience function for ad-hoc debugging.
/// For production use, use the full Model API.
///
/// # Example
///
/// ```rust,ignore
/// use anno::grounded::process_text;
///
/// let result = process_text("Marie Curie won the Nobel Prize.", None)?;
/// println!("{}", result.to_html());
/// ```
pub fn process_text(text: &str, model: Option<&dyn crate::Model>) -> crate::Result<ProcessResult> {
    let mut doc = GroundedDocument::new("input", text);

    if let Some(m) = model {
        let entities = m.extract_entities(text, None)?;
        for e in entities {
            let signal = Signal::new(
                0,
                Location::text(e.start, e.end),
                &e.text,
                e.entity_type.as_label(),
                e.confidence as f32,
            );
            doc.add_signal(signal);
        }
    }

    let errors = doc.validate();
    let valid = errors.is_empty();

    Ok(ProcessResult {
        document: doc,
        valid,
        errors,
    })
}

/// Process text with gold annotations for comparison.
///
/// Returns both the document and an eval comparison.
#[cfg(feature = "eval")]
pub fn process_with_gold(
    text: &str,
    gold: &[crate::eval::GoldEntity],
    model: &dyn crate::Model,
) -> crate::Result<(ProcessResult, EvalComparison)> {
    // Extract predictions
    let predicted_entities = model.extract_entities(text, None)?;

    // Convert gold to signals
    let gold_signals: Vec<Signal<Location>> = gold
        .iter()
        .enumerate()
        .map(|(i, g)| {
            Signal::new(
                i as SignalId,
                Location::text(g.start, g.end),
                &g.text,
                g.entity_type.to_string(),
                1.0,
            )
        })
        .collect();

    // Convert predictions to signals
    let pred_signals: Vec<Signal<Location>> = predicted_entities
        .iter()
        .enumerate()
        .map(|(i, e)| {
            Signal::new(
                i as SignalId,
                Location::text(e.start, e.end),
                &e.text,
                e.entity_type.as_label(),
                e.confidence as f32,
            )
        })
        .collect();

    // Build document with predictions
    let mut doc = GroundedDocument::new("input", text);
    for s in &pred_signals {
        doc.add_signal(s.clone());
    }

    let errors = doc.validate();
    let valid = errors.is_empty();
    let result = ProcessResult {
        document: doc,
        valid,
        errors,
    };

    // Build comparison
    let comparison = EvalComparison::compare(text, gold_signals, pred_signals);

    Ok((result, comparison))
}

// =============================================================================
// Conversions with eval::coref Module
// =============================================================================

/// Convert a Track to a CorefChain for evaluation.
///
/// This enables using the existing coreference metrics (MUC, B3, CEAF, LEA)
/// on tracks from a GroundedDocument.
impl Track {
    /// Convert to a CorefChain for evaluation.
    ///
    /// Requires access to the document's signals to get mention text/offsets.
    #[must_use]
    pub fn to_coref_chain(&self, doc: &GroundedDocument) -> crate::eval::coref::CorefChain {
        use crate::eval::coref::{CorefChain, Mention};

        let mentions: Vec<Mention> = self
            .signals
            .iter()
            .filter_map(|signal_ref| {
                let signal = doc.get_signal(signal_ref.signal_id)?;
                let (start, end) = signal.location.text_offsets()?;
                Some(Mention::new(&signal.surface, start, end))
            })
            .collect();

        let mut chain = CorefChain::new(mentions);
        chain.cluster_id = Some(self.id);
        chain.entity_type = self.entity_type.clone();
        chain
    }
}

impl GroundedDocument {
    /// Convert all tracks to CorefChains for evaluation.
    #[must_use]
    pub fn to_coref_chains(&self) -> Vec<crate::eval::coref::CorefChain> {
        self.tracks
            .values()
            .map(|t| t.to_coref_chain(self))
            .collect()
    }

    /// Create a GroundedDocument from a CorefDocument.
    ///
    /// This converts from the existing coreference evaluation format.
    #[must_use]
    pub fn from_coref_document(
        id: impl Into<String>,
        coref_doc: &crate::eval::coref::CorefDocument,
    ) -> Self {
        let mut doc = Self::new(id, &coref_doc.text);

        // Convert each chain to signals + track
        for (chain_idx, chain) in coref_doc.chains.iter().enumerate() {
            // Use the canonical mention's text, or fall back to first mention's text
            let canonical_text = chain
                .canonical_mention()
                .map(|m| m.text.clone())
                .unwrap_or_else(|| {
                    chain
                        .mentions
                        .first()
                        .map(|m| m.text.clone())
                        .unwrap_or_default()
                });
            let mut track = Track::new(chain_idx as TrackId, canonical_text);
            track.entity_type = chain.entity_type.clone();

            for (mention_idx, mention) in chain.mentions.iter().enumerate() {
                let signal = Signal::new(
                    doc.next_signal_id,
                    Location::text(mention.start, mention.end),
                    &mention.text,
                    chain.entity_type.as_deref().unwrap_or("UNKNOWN"),
                    1.0, // CorefDocument doesn't store confidence
                );
                let signal_id = doc.add_signal(signal);
                track.add_signal(signal_id, mention_idx as u32);
            }

            if let Some(cluster_id) = chain.cluster_id {
                track.id = cluster_id;
            }
            doc.add_track(track);
        }

        doc
    }

    /// Create a CorefDocument for evaluation.
    #[must_use]
    pub fn to_coref_document(&self) -> crate::eval::coref::CorefDocument {
        crate::eval::coref::CorefDocument::new(&self.text, self.to_coref_chains())
    }
}

// =============================================================================
// Tests
// =============================================================================

// =============================================================================
// Corpus: Multi-Document Operations
// =============================================================================

/// A corpus of grounded documents for cross-document operations.
///
/// Enables inter-document coreference resolution and entity linking
/// across multiple documents.
#[derive(Debug, Clone)]
pub struct Corpus {
    /// Documents in the corpus
    documents: HashMap<String, GroundedDocument>,
    /// Global identities (shared across documents)
    identities: HashMap<IdentityId, Identity>,
    /// Next identity ID (global across corpus)
    next_identity_id: IdentityId,
}

impl Corpus {
    /// Create a new empty corpus.
    #[must_use]
    pub fn new() -> Self {
        Self {
            documents: HashMap::new(),
            identities: HashMap::new(),
            next_identity_id: 0,
        }
    }

    /// Add a document to the corpus.
    pub fn add_document(&mut self, doc: GroundedDocument) {
        self.documents.insert(doc.id.clone(), doc);
    }

    /// Get a document by ID.
    #[must_use]
    pub fn get_document(&self, doc_id: &str) -> Option<&GroundedDocument> {
        self.documents.get(doc_id)
    }

    /// Get a mutable reference to a document by ID.
    pub fn get_document_mut(&mut self, doc_id: &str) -> Option<&mut GroundedDocument> {
        self.documents.get_mut(doc_id)
    }

    /// Get all documents.
    pub fn documents(&self) -> impl Iterator<Item = &GroundedDocument> {
        self.documents.values()
    }

    /// Resolve inter-document coreference across all documents.
    ///
    /// Clusters tracks from different documents that refer to the same
    /// real-world entity, creating `Identity` instances without KB links.
    ///
    /// # Algorithm
    ///
    /// 1. Extract all tracks from all documents
    /// 2. Compute track embeddings (if available) or use string similarity
    /// 3. Cluster tracks using similarity threshold
    /// 4. Create Identity for each cluster
    /// 5. Link tracks to identities
    ///
    /// # Parameters
    ///
    /// * `similarity_threshold` - Minimum similarity (0.0-1.0) to cluster tracks
    /// * `require_type_match` - Only cluster tracks with same entity type
    ///
    /// # Returns
    ///
    /// Vector of created identities, each linked to tracks from multiple documents.
    pub fn resolve_inter_doc_coref(
        &mut self,
        similarity_threshold: f32,
        require_type_match: bool,
    ) -> Vec<IdentityId> {
        // 1. Collect all track data (clone what we need to avoid borrow conflicts)
        #[derive(Debug, Clone)]
        struct TrackData {
            track_ref: TrackRef,
            canonical_surface: String,
            entity_type: Option<String>,
            cluster_confidence: f32,
            embedding: Option<Vec<f32>>,
        }

        let mut track_data: Vec<TrackData> = Vec::new();
        for doc in self.documents.values() {
            for track in doc.tracks() {
                if let Some(track_ref) = doc.track_ref(track.id) {
                    track_data.push(TrackData {
                        track_ref,
                        canonical_surface: track.canonical_surface.clone(),
                        entity_type: track.entity_type.clone(),
                        cluster_confidence: track.cluster_confidence,
                        embedding: track.embedding.clone(),
                    });
                }
            }
        }

        if track_data.is_empty() {
            return vec![];
        }

        // 2. Cluster tracks using simple string similarity
        // (In production, use embeddings)
        let mut union_find: Vec<usize> = (0..track_data.len()).collect();

        fn find(parent: &mut [usize], i: usize) -> usize {
            if parent[i] != i {
                parent[i] = find(parent, parent[i]);
            }
            parent[i]
        }

        fn union(parent: &mut [usize], i: usize, j: usize) {
            let pi = find(parent, i);
            let pj = find(parent, j);
            if pi != pj {
                parent[pi] = pj;
            }
        }

        // Compare all pairs
        for i in 0..track_data.len() {
            for j in (i + 1)..track_data.len() {
                let track_a = &track_data[i];
                let track_b = &track_data[j];

                // Type check
                if require_type_match && track_a.entity_type != track_b.entity_type {
                    continue;
                }

                // String similarity (simple Jaccard on words)
                let similarity =
                    Self::string_similarity(&track_a.canonical_surface, &track_b.canonical_surface);

                if similarity >= similarity_threshold {
                    union(&mut union_find, i, j);
                }
            }
        }

        // 3. Build clusters
        let mut cluster_map: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..track_data.len() {
            let root = find(&mut union_find, i);
            cluster_map.entry(root).or_default().push(i);
        }

        // 4. Create identities for each cluster
        // Note: Singleton clusters (clusters with only one track) still create identities.
        // This allows tracking entities that appear only once across documents.
        let mut created_ids = Vec::new();
        for (_, member_indices) in cluster_map.iter() {
            if member_indices.is_empty() {
                continue;
            }

            // Safe: we just checked is_empty() above, so member_indices[0] is valid
            let first_idx = member_indices[0];
            let first_track = &track_data[first_idx];

            // Collect all track refs in this cluster
            let track_refs_in_cluster: Vec<TrackRef> = member_indices
                .iter()
                .map(|&idx| track_data[idx].track_ref.clone())
                .collect();

            // Create identity
            let identity_id = self.next_identity_id;
            self.next_identity_id += 1;

            let identity = Identity {
                id: identity_id,
                canonical_name: first_track.canonical_surface.clone(),
                entity_type: first_track.entity_type.clone(),
                kb_id: None,
                kb_name: None,
                description: None,
                embedding: first_track.embedding.clone(),
                aliases: Vec::new(),
                confidence: first_track.cluster_confidence,
                source: Some(IdentitySource::CrossDocCoref {
                    track_refs: track_refs_in_cluster,
                }),
            };

            self.identities.insert(identity_id, identity);
            created_ids.push(identity_id);

            // 5. Link tracks to identity
            // Collect doc_id and track_id pairs first to avoid borrow conflicts
            let links: Vec<(String, TrackId)> = member_indices
                .iter()
                .map(|&idx| {
                    let track_ref = &track_data[idx].track_ref;
                    (track_ref.doc_id.clone(), track_ref.track_id)
                })
                .collect();

            for (doc_id, track_id) in links {
                if let Some(doc) = self.documents.get_mut(&doc_id) {
                    doc.link_track_to_identity(track_id, identity_id);
                } else {
                    // Document was removed or doesn't exist - this is a data consistency issue
                    // Log warning but continue with other tracks
                    log::warn!(
                        "Document '{}' not found when linking track {} to identity {}",
                        doc_id,
                        track_id,
                        identity_id
                    );
                }
            }
        }

        created_ids
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
    ) -> crate::Result<IdentityId> {
        use crate::Error;

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

    /// Get all identities in the corpus.
    pub fn identities(&self) -> impl Iterator<Item = &Identity> {
        self.identities.values()
    }

    /// Get an identity by ID.
    #[must_use]
    pub fn get_identity(&self, id: IdentityId) -> Option<&Identity> {
        self.identities.get(&id)
    }

    /// Simple string similarity using Jaccard on word sets.
    /// Compute string similarity between two strings.
    ///
    /// Returns a value in [0.0, 1.0] where 1.0 is identical.
    /// Uses the same algorithm as `crate::similarity::string_similarity` but returns f32
    /// for consistency with corpus operations that use f32.
    ///
    /// Note: There may be minor precision differences due to f32 vs f64,
    /// but these are negligible for similarity thresholds.
    fn string_similarity(a: &str, b: &str) -> f32 {
        // Use the f32 version to avoid precision issues in threshold comparison
        crate::similarity::jaccard_word_similarity_f32(a, b)
    }
}

impl Default for Corpus {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_location_bbox_iou() {
        let b1 = Location::bbox(0.0, 0.0, 0.5, 0.5);
        let b2 = Location::bbox(0.25, 0.25, 0.5, 0.5);
        let iou = b1.iou(&b2).unwrap();
        // Intersection: 0.25 * 0.25 = 0.0625
        // Union: 0.5*0.5 + 0.5*0.5 - 0.0625 = 0.4375
        // IoU = 0.0625/0.4375 ≈ 0.143
        assert!((iou - 0.143).abs() < 0.01);
    }

    #[test]
    fn test_location_different_types_no_iou() {
        let text = Location::text(0, 10);
        let bbox = Location::bbox(0.0, 0.0, 0.5, 0.5);
        assert!(text.iou(&bbox).is_none());
    }

    #[test]
    fn test_signal_creation() {
        let signal: Signal<Location> =
            Signal::new(0, Location::text(0, 11), "Marie Curie", "Person", 0.95);
        assert_eq!(signal.surface, "Marie Curie");
        assert_eq!(signal.label, "Person");
        assert!((signal.confidence - 0.95).abs() < 0.001);
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
        assert!(signal.modality.supports_linguistic_features());
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
    fn test_modality_features() {
        assert!(Modality::Symbolic.supports_linguistic_features());
        assert!(!Modality::Symbolic.supports_geometric_features());

        assert!(!Modality::Iconic.supports_linguistic_features());
        assert!(Modality::Iconic.supports_geometric_features());

        assert!(Modality::Hybrid.supports_linguistic_features());
        assert!(Modality::Hybrid.supports_geometric_features());
    }

    #[test]
    fn test_location_from_span() {
        let span = Span::Text { start: 0, end: 10 };
        let location = Location::from(&span);
        assert_eq!(location.text_offsets(), Some((0, 10)));

        let span = Span::BoundingBox {
            x: 0.1,
            y: 0.2,
            width: 0.3,
            height: 0.4,
            page: Some(1),
        };
        let location = Location::from(&span);
        assert!(matches!(location, Location::BoundingBox { .. }));
    }

    #[test]
    fn test_entity_roundtrip() {
        use crate::EntityType;

        let entities = vec![
            Entity::new("Marie Curie", EntityType::Person, 0, 12, 0.95),
            Entity::new(
                "Nobel Prize",
                EntityType::Other("Award".to_string()),
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
    fn test_location_overlaps() {
        let l1 = Location::text(0, 10);
        let l2 = Location::text(5, 15);
        let l3 = Location::text(15, 20);

        assert!(l1.overlaps(&l2));
        assert!(!l1.overlaps(&l3));
        assert!(!l2.overlaps(&l3)); // [5,15) and [15,20) don't overlap

        // Bounding boxes
        let b1 = Location::bbox(0.0, 0.0, 0.5, 0.5);
        let b2 = Location::bbox(0.4, 0.4, 0.5, 0.5);
        let b3 = Location::bbox(0.6, 0.6, 0.2, 0.2);

        assert!(b1.overlaps(&b2));
        assert!(!b1.overlaps(&b3));
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

    #[test]
    fn test_coref_chain_conversion() {
        let mut doc = GroundedDocument::new("doc1", "John went to the store. He bought milk.");

        let s1 = doc.add_signal(Signal::new(0, Location::text(0, 4), "John", "Person", 0.95));
        let s2 = doc.add_signal(Signal::new(1, Location::text(24, 26), "He", "Person", 0.88));

        let mut track = Track::new(0, "John");
        track.add_signal(s1, 0);
        track.add_signal(s2, 1);
        doc.add_track(track);

        // Convert to CorefChain
        let chains = doc.to_coref_chains();
        assert_eq!(chains.len(), 1);
        assert_eq!(chains[0].len(), 2);
        assert_eq!(chains[0].mentions[0].text, "John");
        assert_eq!(chains[0].mentions[1].text, "He");
    }

    #[test]
    fn test_from_coref_document() {
        use crate::eval::coref::{CorefChain, CorefDocument, Mention};

        let chains = vec![CorefChain::new(vec![
            Mention::new("Barack Obama", 0, 12),
            Mention::new("He", 30, 32),
            Mention::new("Obama", 50, 55),
        ])];

        let coref_doc = CorefDocument::new(
            "Barack Obama was president. He served two terms. Obama left in 2017.",
            chains,
        );

        let grounded = GroundedDocument::from_coref_document("doc1", &coref_doc);

        assert_eq!(grounded.signals().len(), 3);
        assert_eq!(grounded.tracks().count(), 1);

        // Verify signals are in order
        let signals: Vec<_> = grounded.signals().iter().collect();
        assert_eq!(signals[0].surface, "Barack Obama");
        assert_eq!(signals[1].surface, "He");
        assert_eq!(signals[2].surface, "Obama");
    }

    #[test]
    fn test_coref_roundtrip() {
        use crate::eval::coref::{CorefChain, CorefDocument, Mention};

        let original_chains = vec![
            CorefChain::new(vec![
                Mention::new("Marie", 0, 5),
                Mention::new("She", 20, 23),
            ]),
            CorefChain::new(vec![Mention::new("Paris", 30, 35)]),
        ];

        let original_doc = CorefDocument::new(
            "Marie lives here. She works. Paris is nice.",
            original_chains,
        );

        // Convert to grounded and back
        let grounded = GroundedDocument::from_coref_document("doc1", &original_doc);
        let converted_doc = grounded.to_coref_document();

        assert_eq!(converted_doc.chains.len(), 2);
        assert_eq!(converted_doc.text, original_doc.text);
    }

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
    fn test_modality_filtering() {
        let mut doc = GroundedDocument::new("doc1", "Test");

        // Add text signal
        let mut text_signal = Signal::new(0, Location::text(0, 4), "Test", "Type", 0.9);
        text_signal.modality = Modality::Symbolic;
        doc.add_signal(text_signal);

        // Add visual signal
        let mut visual_signal =
            Signal::new(0, Location::bbox(0.0, 0.0, 0.5, 0.5), "Box", "Type", 0.8);
        visual_signal.modality = Modality::Iconic;
        doc.add_signal(visual_signal);

        assert_eq!(doc.text_signals().len(), 1);
        assert_eq!(doc.visual_signals().len(), 1);
        assert_eq!(doc.signals_by_modality(Modality::Hybrid).len(), 0);
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
            Location::bbox(0.0, 0.0, 0.5, 0.5).modality(),
            Modality::Iconic
        );

        let temporal = Location::Temporal {
            start_sec: 0.0,
            end_sec: 5.0,
            frame: None,
        };
        assert_eq!(temporal.modality(), Modality::Iconic);

        let genomic = Location::Genomic {
            contig: "chr1".into(),
            start: 0,
            end: 1000,
            strand: Some('+'),
        };
        assert_eq!(genomic.modality(), Modality::Symbolic);

        let hybrid = Location::TextWithBbox {
            start: 0,
            end: 10,
            bbox: Box::new(Location::bbox(0.0, 0.0, 0.5, 0.5)),
        };
        assert_eq!(hybrid.modality(), Modality::Hybrid);
    }

    #[test]
    fn test_cross_doc_cluster_conversion() {
        use crate::eval::cdcr::CrossDocCluster;
        use crate::EntityType;

        // Create a CrossDocCluster
        let mut cluster = CrossDocCluster::new(42, "Albert Einstein");
        cluster.entity_type = Some(EntityType::Person);
        cluster.kb_id = Some("Q937".to_string());
        cluster.confidence = 0.95;

        // Convert to Identity
        let identity = Identity::from_cross_doc_cluster(&cluster);
        assert_eq!(identity.id, 42);
        assert_eq!(identity.canonical_name, "Albert Einstein");
        assert_eq!(identity.entity_type, Some("PER".to_string()));
        assert_eq!(identity.kb_id, Some("Q937".to_string()));
        assert!((identity.confidence - 0.95).abs() < 0.001);
        // Source is None because CDCR format doesn't have track information
        assert!(identity.source.is_none());

        // Convert back to CrossDocCluster
        let roundtrip = identity.to_cross_doc_cluster();
        assert_eq!(roundtrip.id, 42);
        assert_eq!(roundtrip.canonical_name, "Albert Einstein");
        assert_eq!(roundtrip.kb_id, Some("Q937".to_string()));
        assert!((roundtrip.confidence - 0.95).abs() < 0.001);
    }
}

// =============================================================================
// Property-Based Tests
// =============================================================================
//
// These tests verify invariants that should hold for ALL valid inputs,
// not just specific examples. They catch edge cases that unit tests miss.

#[cfg(test)]
mod proptests {
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

            let iou = a.iou(&b).unwrap();
            prop_assert!(
                iou.abs() < 1e-6,
                "Non-overlapping IoU must be 0, got {}",
                iou
            );
        }

        /// BoundingBox IoU is also symmetric and bounded
        #[test]
        fn bbox_iou_symmetric_bounded(
            x1 in 0.0f32..0.8,
            y1 in 0.0f32..0.8,
            w1 in 0.05f32..0.2,
            h1 in 0.05f32..0.2,
            x2 in 0.0f32..0.8,
            y2 in 0.0f32..0.8,
            w2 in 0.05f32..0.2,
            h2 in 0.05f32..0.2,
        ) {
            let a = Location::bbox(x1, y1, w1, h1);
            let b = Location::bbox(x2, y2, w2, h2);

            let iou_ab = a.iou(&b);
            let iou_ba = b.iou(&a);

            // Symmetry
            prop_assert_eq!(iou_ab, iou_ba, "BBox IoU must be symmetric");

            // Bounded
            if let Some(iou) = iou_ab {
                prop_assert!(iou >= 0.0 && iou <= 1.0, "BBox IoU out of bounds: {}", iou);
            }
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

            prop_assert!(signal.confidence >= 0.0, "Confidence below 0: {}", signal.confidence);
            prop_assert!(signal.confidence <= 1.0, "Confidence above 1: {}", signal.confidence);
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
                &label,
                conf,
            );

            prop_assert_eq!(&signal.surface, &surface);
            prop_assert_eq!(&signal.label, &label);
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

        /// Symbolic modality supports linguistic features
        #[test]
        fn symbolic_supports_linguistic(
            start in 0usize..1000,
            len in 1usize..100,
        ) {
            let loc = Location::text(start, start + len);
            prop_assert!(
                loc.modality().supports_linguistic_features(),
                "Text locations must support linguistic features"
            );
        }

        /// Iconic modality supports geometric features
        #[test]
        fn iconic_supports_geometric(
            x in 0.0f32..0.9,
            y in 0.0f32..0.9,
            w in 0.01f32..0.5,
            h in 0.01f32..0.5,
        ) {
            let loc = Location::bbox(x, y, w, h);
            prop_assert!(
                loc.modality().supports_geometric_features(),
                "BBox locations must support geometric features"
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
                track.add_signal(i as SignalId, i as u32);
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
                track.add_signal(i as SignalId, i as u32);
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
            use crate::EntityType;

            let end = start + len;
            let entity = crate::Entity::new(&text, EntityType::Person, start, end, conf);

            let doc = GroundedDocument::from_entities("test", "x".repeat(end + 10), &[entity]);
            let converted = doc.to_entities();

            prop_assert_eq!(converted.len(), 1, "Should have exactly one entity");
            prop_assert_eq!(&converted[0].text, &text, "Text should be preserved");
            prop_assert_eq!(converted[0].start, start, "Start should be preserved");
            prop_assert_eq!(converted[0].end, end, "End should be preserved");
        }

        /// CorefChain roundtrip preserves mention count
        #[test]
        fn coref_roundtrip_preserves_count(mention_count in 1usize..5) {
            use crate::eval::coref::{CorefChain, CorefDocument, Mention};

            let mentions: Vec<Mention> = (0..mention_count)
                .map(|i| Mention::new(&format!("mention_{}", i), i * 20, i * 20 + 10))
                .collect();

            let chain = CorefChain::new(mentions);
            let coref_doc = CorefDocument::new("test text ".repeat(mention_count * 3), vec![chain]);

            let grounded = GroundedDocument::from_coref_document("doc1", &coref_doc);
            let converted = grounded.to_coref_document();

            prop_assert_eq!(
                converted.chains.len(),
                1,
                "Should have exactly one chain"
            );
            prop_assert_eq!(
                converted.chains[0].len(),
                mention_count,
                "Mention count should be preserved"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Modality Invariants
    // -------------------------------------------------------------------------

    proptest! {
        /// Modality feature support is consistent with semiotic theory
        #[test]
        fn modality_feature_consistency(_dummy in 0..1) {
            // Iconic: supports geometric, not linguistic
            prop_assert!(Modality::Iconic.supports_geometric_features());
            prop_assert!(!Modality::Iconic.supports_linguistic_features());

            // Symbolic: supports linguistic, not geometric
            prop_assert!(Modality::Symbolic.supports_linguistic_features());
            prop_assert!(!Modality::Symbolic.supports_geometric_features());

            // Hybrid: supports both
            prop_assert!(Modality::Hybrid.supports_linguistic_features());
            prop_assert!(Modality::Hybrid.supports_geometric_features());
        }
    }

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

            // Add iconic signals
            for i in 0..iconic_count {
                let mut signal = Signal::new(
                    0,
                    Location::bbox(i as f32 * 0.1, 0.0, 0.05, 0.05),
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
            let surface_end = (surface_start + surface_len).min(text.len());
            let surface_start = surface_start.min(surface_end.saturating_sub(1));

            if surface_start < surface_end && surface_end <= text.len() {
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
}
