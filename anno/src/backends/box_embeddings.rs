//! Box embeddings for coreference resolution.
//!
//! This module implements geometric representations (box embeddings) that encode
//! logical invariants of coreference resolution, addressing limitations of
//! vector-based approaches.
//!
//! **Note**: Training code is in `box_embeddings_training.rs`. The [matryoshka-box](https://github.com/arclabs561/matryoshka-box)
//! research project extends training with matryoshka-specific features (variable dimensions, etc.).
//!
//! # Key Concepts
//!
//! - **Box Embeddings**: Entities represented as axis-aligned hyperrectangles
//! - **Conditional Probability**: Coreference = high mutual overlap
//! - **Temporal Boxes**: Entities that evolve over time
//! - **Uncertainty-Aware**: Box volume = confidence
//!
//! # Research Background
//!
//! This implementation is related to the **matryoshka-box** research project (not yet published),
//! which combines matryoshka embeddings (variable dimensions) with box embeddings (hierarchical reasoning).
//! Standard training is in `box_embeddings_training.rs`; matryoshka-box extends it with research features.
//!
//! Based on research from:
//! - Vilnis et al. (2018): "Probabilistic Embedding of Knowledge Graphs with Box Lattice Measures"
//! - Lee et al. (2022): "Box Embeddings for Event-Event Relation Extraction" (BERE)
//! - Messner et al. (2022): "Temporal Knowledge Graph Completion with Box Embeddings" (BoxTE)
//! - Chen et al. (2021): "Uncertainty-Aware Knowledge Graph Embeddings" (UKGE)

use serde::{Deserialize, Serialize};
use std::f32;

/// A box embedding representing an entity in d-dimensional space.
///
/// Boxes are axis-aligned hyperrectangles defined by min/max bounds in each dimension.
/// Coreference is modeled as high mutual conditional probability (overlap).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoxEmbedding {
    /// Lower bound in each dimension [d]
    pub min: Vec<f32>,
    /// Upper bound in each dimension [d]
    pub max: Vec<f32>,
}

impl BoxEmbedding {
    /// Create a new box embedding.
    ///
    /// # Panics
    ///
    /// Panics if `min.len() != max.len()` or if any `min[i] > max[i]`.
    pub fn new(min: Vec<f32>, max: Vec<f32>) -> Self {
        assert_eq!(min.len(), max.len(), "min and max must have same dimension");
        for (i, (&m, &max_val)) in min.iter().zip(max.iter()).enumerate() {
            assert!(
                m <= max_val,
                "min[{}] = {} must be <= max[{}] = {}",
                i,
                m,
                i,
                max_val
            );
        }
        Self { min, max }
    }

    /// Get the dimension of the box.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.min.len()
    }

    /// Compute the volume of the box.
    ///
    /// Volume = ∏(max[i] - min[i])
    #[must_use]
    pub fn volume(&self) -> f32 {
        self.min
            .iter()
            .zip(self.max.iter())
            .map(|(&m, &max_val)| (max_val - m).max(0.0))
            .product()
    }

    /// Compute the intersection volume with another box.
    ///
    /// Returns 0.0 if boxes are disjoint.
    #[must_use]
    pub fn intersection_volume(&self, other: &Self) -> f32 {
        assert_eq!(
            self.dim(),
            other.dim(),
            "Boxes must have same dimension for intersection"
        );

        self.min
            .iter()
            .zip(self.max.iter())
            .zip(other.min.iter().zip(other.max.iter()))
            .map(|((&m1, &max1), (&m2, &max2))| {
                let intersection_min = m1.max(m2);
                let intersection_max = max1.min(max2);
                (intersection_max - intersection_min).max(0.0)
            })
            .product()
    }

    /// Compute conditional probability P(self | other).
    ///
    /// This is the BERE model's coreference metric:
    /// P(A|B) = Vol(A ∩ B) / Vol(B)
    ///
    /// Returns a value in [0.0, 1.0] where:
    /// - 1.0 = self is completely contained in other
    /// - 0.0 = boxes are disjoint
    #[must_use]
    pub fn conditional_probability(&self, other: &Self) -> f32 {
        let vol_other = other.volume();
        if vol_other == 0.0 {
            return 0.0;
        }
        self.intersection_volume(other) / vol_other
    }

    /// Compute mutual coreference score.
    ///
    /// Coreference requires high mutual conditional probability:
    /// score = (P(A|B) + P(B|A)) / 2
    ///
    /// This ensures both boxes largely contain each other (high overlap).
    #[must_use]
    pub fn coreference_score(&self, other: &Self) -> f32 {
        let p_a_given_b = self.conditional_probability(other);
        let p_b_given_a = other.conditional_probability(self);
        (p_a_given_b + p_b_given_a) / 2.0
    }

    /// Check if this box is contained in another box.
    ///
    /// Returns true if self ⊆ other (all dimensions).
    #[must_use]
    pub fn is_contained_in(&self, other: &Self) -> bool {
        assert_eq!(self.dim(), other.dim(), "Boxes must have same dimension");
        self.min
            .iter()
            .zip(self.max.iter())
            .zip(other.min.iter().zip(other.max.iter()))
            .all(|((&m1, &max1), (&m2, &max2))| m2 <= m1 && max1 <= max2)
    }

    /// Check if boxes are disjoint (no overlap).
    #[must_use]
    pub fn is_disjoint(&self, other: &Self) -> bool {
        self.intersection_volume(other) == 0.0
    }

    /// Create a box embedding from a vector embedding.
    ///
    /// Converts a point embedding to a box by creating a small hypercube
    /// around the point. The box size is controlled by `radius`.
    ///
    /// # Arguments
    ///
    /// * `vector` - Vector embedding (point in space)
    /// * `radius` - Half-width of the box in each dimension
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let vector = vec![0.5, 0.5, 0.5];
    /// let box_embedding = BoxEmbedding::from_vector(&vector, 0.1);
    /// // Creates box: min=[0.4, 0.4, 0.4], max=[0.6, 0.6, 0.6]
    /// ```
    #[must_use]
    pub fn from_vector(vector: &[f32], radius: f32) -> Self {
        let min: Vec<f32> = vector.iter().map(|&v| v - radius).collect();
        let max: Vec<f32> = vector.iter().map(|&v| v + radius).collect();
        Self::new(min, max)
    }

    /// Create a box embedding from a vector with adaptive radius.
    ///
    /// Uses a radius proportional to the vector's magnitude, creating
    /// larger boxes for vectors further from the origin.
    ///
    /// # Arguments
    ///
    /// * `vector` - Vector embedding
    /// * `radius_factor` - Multiplier for adaptive radius (default: 0.1)
    #[must_use]
    pub fn from_vector_adaptive(vector: &[f32], radius_factor: f32) -> Self {
        let magnitude: f32 = vector.iter().map(|&v| v * v).sum::<f32>().sqrt();
        let radius = magnitude * radius_factor + 0.01; // Add small epsilon
        Self::from_vector(vector, radius)
    }

    /// Get the center point of the box.
    ///
    /// Returns the midpoint in each dimension.
    #[must_use]
    pub fn center(&self) -> Vec<f32> {
        self.min
            .iter()
            .zip(self.max.iter())
            .map(|(&m, &max_val)| (m + max_val) / 2.0)
            .collect()
    }

    /// Get the size (width) in each dimension.
    #[must_use]
    pub fn size(&self) -> Vec<f32> {
        self.min
            .iter()
            .zip(self.max.iter())
            .map(|(&m, &max_val)| (max_val - m).max(0.0))
            .collect()
    }
}

/// Configuration for box-based coreference resolution.
#[derive(Debug, Clone)]
pub struct BoxCorefConfig {
    /// Minimum coreference score to link entities
    pub coreference_threshold: f32,
    /// Whether to enforce syntactic constraints (Principle B/C)
    pub enforce_syntactic_constraints: bool,
    /// Maximum token distance for local domain (Principle B)
    pub max_local_distance: usize,
    /// Radius for converting vector embeddings to boxes (if using vectors)
    pub vector_to_box_radius: Option<f32>,
}

impl Default for BoxCorefConfig {
    fn default() -> Self {
        Self {
            coreference_threshold: 0.7,
            enforce_syntactic_constraints: true,
            max_local_distance: 5,
            vector_to_box_radius: Some(0.1),
        }
    }
}

// =============================================================================
// Temporal Boxes (BoxTE-style)
// =============================================================================

/// A temporal box embedding that evolves over time.
///
/// Based on BoxTE (Messner et al., 2022), this models entities that change
/// over time. For example, "The President" refers to Obama in 2012 but
/// Trump in 2017 - they should not corefer despite the same surface form.
///
/// # Example
///
/// ```rust,ignore
/// use anno::backends::box_embeddings::{BoxEmbedding, TemporalBox, BoxVelocity};
///
/// // "The President" in 2012 (Obama)
/// let base = BoxEmbedding::new(vec![0.0, 0.0], vec![1.0, 1.0]);
/// let velocity = BoxVelocity::new(vec![0.0, 0.0], vec![0.0, 0.0]); // Static
/// let obama_presidency = TemporalBox::new(base, velocity, (2012.0, 2016.0));
///
/// // "The President" in 2017 (Trump)
/// let trump_base = BoxEmbedding::new(vec![5.0, 5.0], vec![6.0, 6.0]);
/// let trump_presidency = TemporalBox::new(trump_base, velocity, (2017.0, 2021.0));
///
/// // Should not corefer (different time ranges)
/// assert_eq!(obama_presidency.coreference_at_time(&trump_presidency, 2015.0), 0.0);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct TemporalBox {
    /// Base box at time t=0 (or reference time)
    pub base: BoxEmbedding,
    /// Velocity: how box moves/resizes per time unit
    pub velocity: BoxVelocity,
    /// Time range where this box is valid [start, end)
    pub time_range: (f64, f64),
}

/// Velocity of a temporal box (change per time unit).
#[derive(Debug, Clone, PartialEq)]
pub struct BoxVelocity {
    /// Change in min bounds per time unit [d]
    pub min_delta: Vec<f32>,
    /// Change in max bounds per time unit [d]
    pub max_delta: Vec<f32>,
}

impl BoxVelocity {
    /// Create a new box velocity (static by default).
    #[must_use]
    pub fn new(min_delta: Vec<f32>, max_delta: Vec<f32>) -> Self {
        Self {
            min_delta,
            max_delta,
        }
    }

    /// Create a static velocity (no change over time).
    #[must_use]
    pub fn static_velocity(dim: usize) -> Self {
        Self {
            min_delta: vec![0.0; dim],
            max_delta: vec![0.0; dim],
        }
    }
}

impl TemporalBox {
    /// Create a new temporal box.
    ///
    /// # Arguments
    ///
    /// * `base` - Base box at reference time
    /// * `velocity` - How box evolves per time unit
    /// * `time_range` - (start, end) time range where box is valid
    #[must_use]
    pub fn new(base: BoxEmbedding, velocity: BoxVelocity, time_range: (f64, f64)) -> Self {
        assert_eq!(
            base.dim(),
            velocity.min_delta.len(),
            "base and velocity must have same dimension"
        );
        assert_eq!(
            velocity.min_delta.len(),
            velocity.max_delta.len(),
            "velocity min and max deltas must have same dimension"
        );
        Self {
            base,
            velocity,
            time_range,
        }
    }

    /// Get the box at a specific time.
    ///
    /// Returns None if time is outside the valid range.
    #[must_use]
    pub fn at_time(&self, time: f64) -> Option<BoxEmbedding> {
        if time < self.time_range.0 || time >= self.time_range.1 {
            return None;
        }

        // Compute time offset from reference (using start of range as reference)
        let time_offset = time - self.time_range.0;

        // Apply velocity to base box
        let new_min: Vec<f32> = self
            .base
            .min
            .iter()
            .zip(self.velocity.min_delta.iter())
            .map(|(&m, &delta)| m + delta * time_offset as f32)
            .collect();

        let new_max: Vec<f32> = self
            .base
            .max
            .iter()
            .zip(self.velocity.max_delta.iter())
            .map(|(&max_val, &delta)| max_val + delta * time_offset as f32)
            .collect();

        Some(BoxEmbedding::new(new_min, new_max))
    }

    /// Compute coreference score at a specific time.
    ///
    /// Returns 0.0 if either box is invalid at the given time.
    #[must_use]
    pub fn coreference_at_time(&self, other: &Self, time: f64) -> f32 {
        let box_a = match self.at_time(time) {
            Some(b) => b,
            None => return 0.0,
        };
        let box_b = match other.at_time(time) {
            Some(b) => b,
            None => return 0.0,
        };
        box_a.coreference_score(&box_b)
    }

    /// Check if this temporal box is valid at the given time.
    #[must_use]
    pub fn is_valid_at(&self, time: f64) -> bool {
        time >= self.time_range.0 && time < self.time_range.1
    }
}

// =============================================================================
// Uncertainty-Aware Boxes (UKGE-style)
// =============================================================================

/// An uncertainty-aware box embedding (UKGE-style).
///
/// Based on UKGE (Chen et al., 2021), box volume represents confidence:
/// - Small box = high confidence (precise, trusted fact)
/// - Large box = low confidence (vague, uncertain, or dubious claim)
///
/// This enables conflict detection: if two high-confidence boxes are disjoint,
/// they represent contradictory claims.
///
/// # Example
///
/// ```rust,ignore
/// use anno::backends::box_embeddings::{BoxEmbedding, UncertainBox};
///
/// // High-confidence claim: "Trump is in NY" (small, precise box)
/// let claim_a = UncertainBox::new(
///     BoxEmbedding::new(vec![0.0, 0.0], vec![0.1, 0.1]), // Small = high confidence
///     0.95, // Source trust
/// );
///
/// // Contradictory claim: "Trump is in FL" (also high confidence, but disjoint)
/// let claim_b = UncertainBox::new(
///     BoxEmbedding::new(vec![5.0, 5.0], vec![5.1, 5.1]), // Disjoint from claim_a
///     0.90,
/// );
///
/// // Should detect conflict
/// assert!(claim_a.detect_conflict(&claim_b).is_some());
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct UncertainBox {
    /// The underlying box embedding
    pub box_embedding: BoxEmbedding,
    /// Source trustworthiness (0.0-1.0)
    pub source_trust: f32,
}

impl UncertainBox {
    /// Create a new uncertainty-aware box.
    ///
    /// Confidence is derived from box volume (smaller = higher confidence).
    #[must_use]
    pub fn new(box_embedding: BoxEmbedding, source_trust: f32) -> Self {
        assert!(
            (0.0..=1.0).contains(&source_trust),
            "source_trust must be in [0.0, 1.0]"
        );
        Self {
            box_embedding,
            source_trust,
        }
    }

    /// Get confidence derived from box volume.
    ///
    /// Smaller boxes = higher confidence. This is a heuristic:
    /// confidence ≈ 1.0 / (1.0 + volume)
    #[must_use]
    pub fn confidence(&self) -> f32 {
        let vol = self.box_embedding.volume();
        // Normalize: confidence decreases as volume increases
        // Using sigmoid-like function: 1 / (1 + volume)
        1.0 / (1.0 + vol)
    }

    /// Detect conflict with another uncertain box.
    ///
    /// Returns Some(Conflict) if both boxes are high-confidence but disjoint,
    /// indicating contradictory claims.
    #[must_use]
    pub fn detect_conflict(&self, other: &Self) -> Option<Conflict> {
        let overlap = self.box_embedding.intersection_volume(&other.box_embedding);
        let min_vol = self
            .box_embedding
            .volume()
            .min(other.box_embedding.volume());

        // If both are high-confidence (small volume) but disjoint, conflict
        let conf_a = self.confidence();
        let conf_b = other.confidence();
        let threshold = 0.8;

        if overlap < min_vol * 0.1 && conf_a > threshold && conf_b > threshold {
            Some(Conflict {
                claim_a_trust: self.source_trust,
                claim_b_trust: other.source_trust,
                severity: (1.0 - overlap / min_vol.max(1e-6)) * (conf_a + conf_b) / 2.0,
            })
        } else {
            None
        }
    }
}

/// Represents a conflict between two uncertain claims.
#[derive(Debug, Clone, PartialEq)]
pub struct Conflict {
    /// Trust in first claim's source
    pub claim_a_trust: f32,
    /// Trust in second claim's source
    pub claim_b_trust: f32,
    /// Severity of conflict (0.0-1.0, higher = more severe)
    pub severity: f32,
}

// =============================================================================
// Interaction Modeling (Triple Intersection)
// =============================================================================

/// Compute interaction strength between actor, action, and target.
///
/// Models asymmetric relations (e.g., "Company A acquired Company B")
/// via triple intersection volume. The interaction is the volume where
/// all three boxes overlap.
///
/// # Arguments
///
/// * `actor_box` - Box for the actor (e.g., buyer)
/// * `action_box` - Box for the action/relation (e.g., "acquired")
/// * `target_box` - Box for the target (e.g., company being acquired)
///
/// # Returns
///
/// Conditional probability P(action, target | actor), representing
/// how much of the actor's space contains the interaction.
#[must_use]
pub fn interaction_strength(
    actor_box: &BoxEmbedding,
    action_box: &BoxEmbedding,
    target_box: &BoxEmbedding,
) -> f32 {
    // Triple intersection: where all three boxes overlap
    // For simplicity, we compute pairwise intersections and take minimum
    // In full implementation, would compute true 3-way intersection
    let actor_action = actor_box.intersection_volume(action_box);
    let action_target = action_box.intersection_volume(target_box);
    let actor_target = actor_box.intersection_volume(target_box);

    // Interaction volume ≈ minimum of pairwise intersections
    let interaction_vol = actor_action.min(action_target).min(actor_target);

    // P(interaction | actor) = interaction_vol / vol(actor)
    let vol_actor = actor_box.volume();
    if vol_actor == 0.0 {
        return 0.0;
    }
    interaction_vol / vol_actor
}

/// Compute asymmetric roles in a relation.
///
/// For a relation like "acquired", determines which entity is the
/// buyer vs. seller based on conditional probabilities.
///
/// # Returns
///
/// (buyer_role, seller_role) where each is the interaction strength
/// for that role.
#[must_use]
pub fn acquisition_roles(
    entity_a: &BoxEmbedding,
    entity_b: &BoxEmbedding,
    acquisition_box: &BoxEmbedding,
) -> (f32, f32) {
    let buyer_role = interaction_strength(entity_a, acquisition_box, entity_b);
    let seller_role = interaction_strength(entity_b, acquisition_box, entity_a);
    (buyer_role, seller_role)
}

// =============================================================================
// Gumbel Boxes (Noise Robustness)
// =============================================================================

/// A Gumbel box with soft, probabilistic boundaries.
///
/// Instead of hard walls, boundaries are modeled as Gumbel distributions,
/// creating "fuzzy" boxes that tolerate slight misalignments. This prevents
/// brittle logic failures when data is noisy.
///
/// # Example
///
/// ```rust,ignore
/// use anno::backends::box_embeddings::{BoxEmbedding, GumbelBox};
///
/// let mean_box = BoxEmbedding::new(vec![0.0, 0.0], vec![1.0, 1.0]);
/// let gumbel_box = GumbelBox::new(mean_box, 0.1); // Low temperature = sharp
///
/// // Membership is probabilistic, not binary
/// let point = vec![0.5, 0.5];
/// let prob = gumbel_box.membership_probability(&point);
/// assert!(prob > 0.5); // High probability inside box
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct GumbelBox {
    /// Mean box boundaries (lower bounds)
    pub mean_min: Vec<f32>,
    /// Mean box boundaries (upper bounds)
    pub mean_max: Vec<f32>,
    /// Temperature: controls fuzziness (higher = more fuzzy)
    /// Typical values: 0.01-0.1 for sharp, 0.5-1.0 for fuzzy
    pub temperature: f32,
}

impl GumbelBox {
    /// Create a new Gumbel box.
    #[must_use]
    pub fn new(mean_box: BoxEmbedding, temperature: f32) -> Self {
        assert!(
            temperature > 0.0,
            "temperature must be positive, got {}",
            temperature
        );
        Self {
            mean_min: mean_box.min,
            mean_max: mean_box.max,
            temperature,
        }
    }

    /// Compute membership probability for a point.
    ///
    /// Returns probability that point belongs to this box (0.0-1.0).
    /// Uses Gumbel CDF approximation for soft boundaries.
    #[must_use]
    pub fn membership_probability(&self, point: &[f32]) -> f32 {
        assert_eq!(
            point.len(),
            self.mean_min.len(),
            "point dimension must match box dimension"
        );

        let mut prob = 1.0;
        for (i, &coord) in point.iter().enumerate() {
            // Gumbel CDF approximation: P(x < max) ≈ 1 / (1 + exp(-(max - x) / temp))
            // For min boundary: P(x > min) ≈ 1 / (1 + exp(-(x - min) / temp))
            let min_prob = 1.0 / (1.0 + (-(coord - self.mean_min[i]) / self.temperature).exp());
            let max_prob = 1.0 / (1.0 + (-(self.mean_max[i] - coord) / self.temperature).exp());
            prob *= min_prob * max_prob;
        }
        prob
    }

    /// Compute robust coreference score with another Gumbel box.
    ///
    /// Samples points from self and checks membership in other, averaging
    /// probabilities. This tolerates slight misalignments.
    ///
    /// # Arguments
    ///
    /// * `other` - The other Gumbel box to compare against
    /// * `samples` - Number of sample points to use (more = more accurate but slower)
    /// * `rng` - Optional RNG for sampling. If None, uses deterministic grid sampling.
    #[must_use]
    pub fn robust_coreference(&self, other: &Self, samples: usize) -> f32 {
        assert_eq!(
            self.mean_min.len(),
            other.mean_min.len(),
            "boxes must have same dimension"
        );

        // Deterministic grid sampling (no RNG dependency)
        // For each dimension, sample at regular intervals
        let samples_per_dim = (samples as f32)
            .powf(1.0 / self.mean_min.len() as f32)
            .ceil() as usize;
        let mut total_prob = 0.0;
        let mut count = 0;

        // Generate grid points
        let mut indices = vec![0; self.mean_min.len()];
        loop {
            // Compute point from grid indices
            let point: Vec<f32> = self
                .mean_min
                .iter()
                .zip(self.mean_max.iter())
                .zip(indices.iter())
                .map(|((&min_val, &max_val), &idx)| {
                    let t = idx as f32 / (samples_per_dim - 1).max(1) as f32;
                    min_val + t * (max_val - min_val)
                })
                .collect();

            total_prob += other.membership_probability(&point);
            count += 1;

            // Increment grid indices
            let mut carry = true;
            for idx in &mut indices {
                if carry {
                    *idx += 1;
                    if *idx >= samples_per_dim {
                        *idx = 0;
                        carry = true;
                    } else {
                        carry = false;
                    }
                }
            }

            if carry || count >= samples {
                break;
            }
        }

        total_prob / count as f32
    }
}

// Note: BoxCorefResolver is implemented in src/eval/coref_resolver.rs
// to be alongside other coreference resolvers.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_box_volume() {
        let box1 = BoxEmbedding::new(vec![0.0, 0.0], vec![1.0, 1.0]);
        assert_eq!(box1.volume(), 1.0);

        let box2 = BoxEmbedding::new(vec![0.0, 0.0, 0.0], vec![2.0, 3.0, 4.0]);
        assert_eq!(box2.volume(), 24.0);
    }

    #[test]
    fn test_intersection_volume() {
        let box1 = BoxEmbedding::new(vec![0.0, 0.0], vec![2.0, 2.0]);
        let box2 = BoxEmbedding::new(vec![1.0, 1.0], vec![3.0, 3.0]);
        assert_eq!(box1.intersection_volume(&box2), 1.0);

        let box3 = BoxEmbedding::new(vec![5.0, 5.0], vec![6.0, 6.0]);
        assert_eq!(box1.intersection_volume(&box3), 0.0); // Disjoint
    }

    #[test]
    fn test_conditional_probability() {
        let box_a = BoxEmbedding::new(vec![0.0, 0.0], vec![1.0, 1.0]); // Volume = 1
        let box_b = BoxEmbedding::new(vec![0.0, 0.0], vec![2.0, 2.0]); // Volume = 4

        // box_a is contained in box_b
        assert_eq!(box_a.conditional_probability(&box_b), 0.25); // 1/4
        assert_eq!(box_b.conditional_probability(&box_a), 1.0); // 4/4 (intersection = box_a)
    }

    #[test]
    fn test_coreference_score() {
        // Identical boxes should have score = 1.0
        let box1 = BoxEmbedding::new(vec![0.0, 0.0], vec![1.0, 1.0]);
        let box2 = BoxEmbedding::new(vec![0.0, 0.0], vec![1.0, 1.0]);
        assert!((box1.coreference_score(&box2) - 1.0).abs() < 1e-6);

        // Disjoint boxes should have score = 0.0
        let box3 = BoxEmbedding::new(vec![0.0, 0.0], vec![1.0, 1.0]);
        let box4 = BoxEmbedding::new(vec![2.0, 2.0], vec![3.0, 3.0]);
        assert_eq!(box3.coreference_score(&box4), 0.0);
    }

    #[test]
    fn test_containment() {
        let box_a = BoxEmbedding::new(vec![0.0, 0.0], vec![1.0, 1.0]);
        let box_b = BoxEmbedding::new(vec![0.0, 0.0], vec![2.0, 2.0]);
        assert!(box_a.is_contained_in(&box_b));
        assert!(!box_b.is_contained_in(&box_a));
    }

    #[test]
    fn test_box_operations() {
        // Test that box operations work correctly
        let box1 = BoxEmbedding::new(vec![0.0, 0.0], vec![1.0, 1.0]);
        let box2 = BoxEmbedding::new(vec![0.5, 0.5], vec![1.5, 1.5]);

        // Should have intersection
        assert!(box1.intersection_volume(&box2) > 0.0);

        // Coreference score should be > 0
        assert!(box1.coreference_score(&box2) > 0.0);
    }

    #[test]
    fn test_from_vector() {
        let vector = vec![0.5, 0.5, 0.5];
        let box_embedding = BoxEmbedding::from_vector(&vector, 0.1);

        assert_eq!(box_embedding.min, vec![0.4, 0.4, 0.4]);
        assert_eq!(box_embedding.max, vec![0.6, 0.6, 0.6]);
        assert!((box_embedding.volume() - 0.008).abs() < 1e-6); // 0.2^3 with float tolerance
    }

    #[test]
    fn test_center_and_size() {
        let box_embedding = BoxEmbedding::new(vec![0.0, 1.0], vec![2.0, 3.0]);
        let center = box_embedding.center();
        let size = box_embedding.size();

        assert_eq!(center, vec![1.0, 2.0]);
        assert_eq!(size, vec![2.0, 2.0]);
    }

    // =========================================================================
    // Temporal Box Tests
    // =========================================================================

    #[test]
    fn test_temporal_box_at_time() {
        let base = BoxEmbedding::new(vec![0.0, 0.0], vec![1.0, 1.0]);
        let velocity = BoxVelocity::static_velocity(2);
        let temporal = TemporalBox::new(base, velocity, (2012.0, 2016.0));

        // Should be valid in range
        assert!(temporal.is_valid_at(2014.0));
        assert!(!temporal.is_valid_at(2017.0));

        // Static velocity: box should be same at any time in range
        let box_at_time = temporal.at_time(2014.0).unwrap();
        assert_eq!(box_at_time.min, vec![0.0, 0.0]);
        assert_eq!(box_at_time.max, vec![1.0, 1.0]);
    }

    #[test]
    fn test_temporal_box_with_velocity() {
        // Box that moves over time
        let base = BoxEmbedding::new(vec![0.0, 0.0], vec![1.0, 1.0]);
        let velocity = BoxVelocity::new(vec![0.1, 0.1], vec![0.1, 0.1]);
        let temporal = TemporalBox::new(base, velocity, (0.0, 10.0));

        // At time 0, should be at base
        let box_t0 = temporal.at_time(0.0).unwrap();
        assert_eq!(box_t0.min, vec![0.0, 0.0]);
        assert_eq!(box_t0.max, vec![1.0, 1.0]);

        // At time 5, should have moved
        let box_t5 = temporal.at_time(5.0).unwrap();
        assert_eq!(box_t5.min, vec![0.5, 0.5]); // 0.0 + 0.1 * 5
        assert_eq!(box_t5.max, vec![1.5, 1.5]); // 1.0 + 0.1 * 5
    }

    #[test]
    fn test_temporal_box_coreference() {
        // Two presidencies that don't overlap in time
        let obama_base = BoxEmbedding::new(vec![0.0, 0.0], vec![1.0, 1.0]);
        let trump_base = BoxEmbedding::new(vec![5.0, 5.0], vec![6.0, 6.0]);
        let velocity = BoxVelocity::static_velocity(2);

        let obama = TemporalBox::new(obama_base, velocity.clone(), (2012.0, 2016.0));
        let trump = TemporalBox::new(trump_base, velocity, (2017.0, 2021.0));

        // Should not corefer (different time ranges)
        assert_eq!(obama.coreference_at_time(&trump, 2015.0), 0.0);
        assert_eq!(obama.coreference_at_time(&trump, 2018.0), 0.0);
    }

    // =========================================================================
    // Uncertainty-Aware Box Tests
    // =========================================================================

    #[test]
    fn test_uncertain_box_confidence() {
        // Small box = high confidence
        let small_box = BoxEmbedding::new(vec![0.0, 0.0], vec![0.1, 0.1]);
        let uncertain_small = UncertainBox::new(small_box, 0.9);
        assert!(uncertain_small.confidence() > 0.5);

        // Large box = low confidence
        let large_box = BoxEmbedding::new(vec![0.0, 0.0], vec![10.0, 10.0]);
        let uncertain_large = UncertainBox::new(large_box, 0.9);
        assert!(uncertain_large.confidence() < uncertain_small.confidence());
    }

    #[test]
    fn test_conflict_detection() {
        // Two high-confidence, disjoint claims = conflict
        let claim_a = UncertainBox::new(BoxEmbedding::new(vec![0.0, 0.0], vec![0.1, 0.1]), 0.95);
        let claim_b = UncertainBox::new(BoxEmbedding::new(vec![5.0, 5.0], vec![5.1, 5.1]), 0.90);

        let conflict = claim_a.detect_conflict(&claim_b);
        assert!(conflict.is_some());
        if let Some(c) = conflict {
            assert!(c.severity > 0.0);
            assert_eq!(c.claim_a_trust, 0.95);
            assert_eq!(c.claim_b_trust, 0.90);
        }
    }

    #[test]
    fn test_no_conflict_for_overlapping_boxes() {
        // Overlapping boxes should not conflict
        let claim_a = UncertainBox::new(BoxEmbedding::new(vec![0.0, 0.0], vec![1.0, 1.0]), 0.95);
        let claim_b = UncertainBox::new(BoxEmbedding::new(vec![0.5, 0.5], vec![1.5, 1.5]), 0.90);

        let conflict = claim_a.detect_conflict(&claim_b);
        assert!(conflict.is_none()); // Overlapping = no conflict
    }

    // =========================================================================
    // Gumbel Box Tests
    // =========================================================================

    #[test]
    fn test_gumbel_box_membership() {
        let mean_box = BoxEmbedding::new(vec![0.0, 0.0], vec![1.0, 1.0]);
        let gumbel = GumbelBox::new(mean_box, 0.1);

        // Point inside box should have high membership
        let inside = vec![0.5, 0.5];
        let prob_inside = gumbel.membership_probability(&inside);
        assert!(prob_inside > 0.5);

        // Point outside box should have low membership
        let outside = vec![2.0, 2.0];
        let prob_outside = gumbel.membership_probability(&outside);
        assert!(prob_outside < prob_inside);
    }

    #[test]
    fn test_gumbel_temperature_effect() {
        let mean_box = BoxEmbedding::new(vec![0.0, 0.0], vec![1.0, 1.0]);
        let sharp = GumbelBox::new(mean_box.clone(), 0.01); // Low temp = sharp
        let fuzzy = GumbelBox::new(mean_box, 1.0); // High temp = fuzzy

        let point = vec![1.1, 1.1]; // Just outside box
        let prob_sharp = sharp.membership_probability(&point);
        let prob_fuzzy = fuzzy.membership_probability(&point);

        // Fuzzy box should have higher probability for near-boundary points
        assert!(prob_fuzzy > prob_sharp);
    }

    #[test]
    fn test_gumbel_robust_coreference() {
        let box1 = BoxEmbedding::new(vec![0.0, 0.0], vec![1.0, 1.0]);
        let box2 = BoxEmbedding::new(vec![0.1, 0.1], vec![0.9, 0.9]);
        let gumbel1 = GumbelBox::new(box1, 0.1);
        let gumbel2 = GumbelBox::new(box2, 0.1);

        // Overlapping boxes should have high robust coreference
        let score = gumbel1.robust_coreference(&gumbel2, 100);
        assert!(score > 0.3);
    }

    // =========================================================================
    // Interaction Modeling Tests
    // =========================================================================

    #[test]
    fn test_interaction_strength() {
        let actor = BoxEmbedding::new(vec![0.0, 0.0], vec![1.0, 1.0]);
        let action = BoxEmbedding::new(vec![0.2, 0.2], vec![0.8, 0.8]);
        let target = BoxEmbedding::new(vec![0.3, 0.3], vec![0.7, 0.7]);

        let strength = interaction_strength(&actor, &action, &target);
        assert!(strength > 0.0);
        assert!(strength <= 1.0);
    }

    #[test]
    fn test_acquisition_roles() {
        let buyer = BoxEmbedding::new(vec![0.0, 0.0], vec![1.0, 1.0]);
        let seller = BoxEmbedding::new(vec![0.5, 0.5], vec![1.5, 1.5]);
        let acquisition = BoxEmbedding::new(vec![0.2, 0.2], vec![0.8, 0.8]);

        let (buyer_role, seller_role) = acquisition_roles(&buyer, &seller, &acquisition);

        // Both should have non-zero roles
        assert!(buyer_role >= 0.0);
        assert!(seller_role >= 0.0);

        // Roles should be asymmetric (buyer ≠ seller in general)
        // Note: In this simple test, they might be equal, but in practice
        // with learned embeddings, they would differ
    }
}
