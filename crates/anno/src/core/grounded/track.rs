use super::GroundedDocument;
use super::super::confidence::Confidence;
use super::super::types::{SignalId, TrackId, TypeLabel};
use super::identity::IdentityId;
use super::signal::SignalRef;
use serde::{Deserialize, Serialize};

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
    pub entity_type: Option<TypeLabel>,
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
        self.entity_type = Some(TypeLabel::from(s.as_str()));
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
    /// use anno::{Track, TypeLabel, EntityType};
    ///
    /// let track = Track::new(0, "Marie Curie")
    ///     .with_type_label(TypeLabel::Core(EntityType::Person));
    /// ```
    #[must_use]
    pub fn with_type_label(mut self, label: TypeLabel) -> Self {
        self.entity_type = Some(label);
        self
    }

    /// Get the entity type as a type-safe label.
    ///
    /// This converts the internal string representation to a `TypeLabel`,
    /// attempting to parse it as a core `EntityType` first.
    #[must_use]
    pub fn type_label(&self) -> Option<TypeLabel> {
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
            min_confidence: Confidence::new(min_conf as f64),
            max_confidence: Confidence::new(max_conf as f64),
            mean_confidence: Confidence::new(mean_conf as f64),
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
    pub min_confidence: Confidence,
    /// Maximum confidence across mentions.
    pub max_confidence: Confidence,
    /// Mean confidence across mentions.
    pub mean_confidence: Confidence,
    /// Whether this track has an embedding.
    pub has_embedding: bool,
}
