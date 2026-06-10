use super::Location;
use super::super::confidence::Confidence;
use super::super::entity::{Entity, HierarchicalConfidence, Provenance};
use super::super::types::{SignalId, TypeLabel};
use serde::{Deserialize, Serialize};

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
// Signal (Level 1): Raw Detection
// =============================================================================

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
    pub label: TypeLabel,
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
        label: impl Into<TypeLabel>,
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
    pub fn type_label(&self) -> TypeLabel {
        self.label.clone()
    }

    /// Get the surface form.
    #[must_use]
    pub fn surface(&self) -> &str {
        &self.surface
    }

    /// Check if this signal is above a confidence threshold.
    #[must_use]
    pub fn is_confident(&self, threshold: Confidence) -> bool {
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
    /// use anno::{Signal, Location};
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
    /// use anno::{Signal, Location};
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
        label: impl Into<TypeLabel>,
        confidence: f32,
    ) -> Option<Self> {
        Self::from_text_nth(source, surface, label, confidence, 0)
    }

    /// Create a signal by finding the nth occurrence of text in source.
    #[must_use]
    pub fn from_text_nth(
        source: &str,
        surface: &str,
        label: impl Into<TypeLabel>,
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

