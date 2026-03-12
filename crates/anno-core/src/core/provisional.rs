//! Provisional types for experimental features.
//!
//! Types in this module are not yet stable and may change or be removed.
//! They exist to enable experimentation without polluting the core type system.
//!
//! # Currently Provisional
//!
//! - [`BoxEmbedding`]: Geometric box embeddings for coreference (research stage)
//! - [`ProvisionalIdentity`]: Identity wrapper with experimental fields
//!
//! # Why Provisional?
//!
//! Some features are valuable for research but not ready for the stable API:
//!
//! - **Representation may change**: Box embedding format is still being refined
//! - **Performance unproven**: Haven't benchmarked at scale
//! - **API surface unclear**: Don't know the right abstractions yet
//!
//! # Migration Path
//!
//! When a provisional type stabilizes:
//! 1. Move it to the appropriate module (`grounded`, `entity`, etc.)
//! 2. Add deprecation warning to the re-export here
//! 3. Remove after one major version

use serde::{Deserialize, Serialize};

/// Box embedding for geometric coreference resolution.
///
/// Uses axis-aligned hyperrectangles to encode logical invariants.
/// This is based on research showing that box embeddings can capture
/// containment relationships better than vector embeddings.
///
/// # Status: Experimental
///
/// This type may change significantly. The embedding format
/// (`Vec<f32>` min/max bounds) is functional but the API is not stable.
///
/// # References
///
/// - Vilnis et al., "Probabilistic Embedding of Knowledge Graphs with Box Lattice Measures"
/// - Dasgupta et al., "Improving Local Identifiability in Probabilistic Box Embeddings"
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct BoxEmbedding {
    /// Minimum corner of the box (lower bounds).
    pub min: Vec<f32>,
    /// Maximum corner of the box (upper bounds).
    pub max: Vec<f32>,
    /// Temperature parameter for softbox formulation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
}

impl BoxEmbedding {
    /// Create a new box embedding.
    #[must_use]
    pub fn new(min: Vec<f32>, max: Vec<f32>) -> Self {
        Self {
            min,
            max,
            temperature: None,
        }
    }

    /// Get the dimensionality of the box.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.min.len()
    }

    /// Check if this box is valid (min <= max in all dimensions).
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.min.len() == self.max.len() && self.min.iter().zip(&self.max).all(|(lo, hi)| lo <= hi)
    }

    /// Compute the volume of the box (product of side lengths).
    #[must_use]
    pub fn volume(&self) -> f32 {
        self.min
            .iter()
            .zip(&self.max)
            .map(|(lo, hi)| (hi - lo).max(0.0))
            .product()
    }

    /// Check if this box contains a point.
    #[must_use]
    pub fn contains_point(&self, point: &[f32]) -> bool {
        point.len() == self.min.len()
            && point
                .iter()
                .zip(&self.min)
                .zip(&self.max)
                .all(|((p, lo), hi)| p >= lo && p <= hi)
    }

    /// Check if this box contains another box.
    #[must_use]
    pub fn contains_box(&self, other: &BoxEmbedding) -> bool {
        self.min.len() == other.min.len()
            && self.min.iter().zip(&other.min).all(|(s, o)| s <= o)
            && self.max.iter().zip(&other.max).all(|(s, o)| s >= o)
    }

    /// Compute intersection volume with another box.
    #[must_use]
    pub fn intersection_volume(&self, other: &BoxEmbedding) -> f32 {
        if self.min.len() != other.min.len() {
            return 0.0;
        }

        self.min
            .iter()
            .zip(&self.max)
            .zip(other.min.iter().zip(&other.max))
            .map(|((lo1, hi1), (lo2, hi2))| {
                let lo = lo1.max(*lo2);
                let hi = hi1.min(*hi2);
                (hi - lo).max(0.0)
            })
            .product()
    }

    /// Convert from a JSON value (for backwards compatibility with existing data).
    pub fn from_json(value: &serde_json::Value) -> Option<Self> {
        serde_json::from_value(value.clone()).ok()
    }

    /// Convert to a JSON value (for backwards compatibility).
    #[must_use]
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
}

/// Identity extension with provisional fields.
///
/// Use this when you need box embeddings or other experimental features
/// without modifying the core `Identity` type.
///
/// # Example
///
/// ```rust
/// use anno_core::Identity;
/// use anno_core::core::provisional::{ProvisionalIdentity, BoxEmbedding};
///
/// let identity = Identity::new(0, "Marie Curie");
/// let provisional = ProvisionalIdentity::from_identity(identity)
///     .with_box_embedding(BoxEmbedding::new(vec![0.0; 64], vec![1.0; 64]));
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProvisionalIdentity {
    /// The base identity.
    pub base: super::grounded::Identity,
    /// Optional box embedding for geometric coreference.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub box_embedding: Option<BoxEmbedding>,
}

impl ProvisionalIdentity {
    /// Create a provisional identity from a base identity.
    #[must_use]
    pub fn from_identity(base: super::grounded::Identity) -> Self {
        Self {
            base,
            box_embedding: None,
        }
    }

    /// Add a box embedding.
    #[must_use]
    pub fn with_box_embedding(mut self, embedding: BoxEmbedding) -> Self {
        self.box_embedding = Some(embedding);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_box_embedding_basic() {
        let box_emb = BoxEmbedding::new(vec![0.0, 0.0], vec![1.0, 1.0]);
        assert!(box_emb.is_valid());
        assert_eq!(box_emb.dim(), 2);
        assert!((box_emb.volume() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_box_embedding_containment() {
        let outer = BoxEmbedding::new(vec![0.0, 0.0], vec![2.0, 2.0]);
        let inner = BoxEmbedding::new(vec![0.5, 0.5], vec![1.5, 1.5]);

        assert!(outer.contains_box(&inner));
        assert!(!inner.contains_box(&outer));
    }

    #[test]
    fn test_box_embedding_intersection() {
        let box1 = BoxEmbedding::new(vec![0.0, 0.0], vec![2.0, 2.0]);
        let box2 = BoxEmbedding::new(vec![1.0, 1.0], vec![3.0, 3.0]);

        let intersection = box1.intersection_volume(&box2);
        assert!((intersection - 1.0).abs() < 1e-6); // 1x1 overlap
    }

    #[test]
    fn test_box_embedding_serde() {
        let original = BoxEmbedding::new(vec![0.0, 1.0, 2.0], vec![1.0, 2.0, 3.0]);
        let json = serde_json::to_string(&original).unwrap();
        let parsed: BoxEmbedding = serde_json::from_str(&json).unwrap();
        assert_eq!(original, parsed);
    }

    #[test]
    fn test_box_embedding_invalid() {
        // min > max in one dimension
        let invalid = BoxEmbedding::new(vec![2.0, 0.0], vec![1.0, 1.0]);
        assert!(!invalid.is_valid());
        // Mismatched dimensions
        let mismatched = BoxEmbedding {
            min: vec![0.0],
            max: vec![1.0, 2.0],
            temperature: None,
        };
        assert!(!mismatched.is_valid());
    }

    #[test]
    fn test_box_embedding_contains_point() {
        let b = BoxEmbedding::new(vec![0.0, 0.0], vec![1.0, 1.0]);
        assert!(b.contains_point(&[0.5, 0.5]));
        assert!(b.contains_point(&[0.0, 0.0])); // boundary
        assert!(b.contains_point(&[1.0, 1.0])); // boundary
        assert!(!b.contains_point(&[1.5, 0.5])); // outside
        assert!(!b.contains_point(&[0.5])); // wrong dimension
    }

    #[test]
    fn test_box_embedding_zero_volume() {
        // Flat box (zero volume in one dimension)
        let flat = BoxEmbedding::new(vec![0.0, 0.0], vec![1.0, 0.0]);
        assert!((flat.volume() - 0.0).abs() < 1e-6);
        assert!(flat.is_valid()); // min == max is valid
    }

    #[test]
    fn test_box_embedding_no_intersection() {
        let b1 = BoxEmbedding::new(vec![0.0, 0.0], vec![1.0, 1.0]);
        let b2 = BoxEmbedding::new(vec![2.0, 2.0], vec![3.0, 3.0]);
        assert!((b1.intersection_volume(&b2) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_box_embedding_different_dim_intersection() {
        let b1 = BoxEmbedding::new(vec![0.0], vec![1.0]);
        let b2 = BoxEmbedding::new(vec![0.0, 0.0], vec![1.0, 1.0]);
        assert!((b1.intersection_volume(&b2) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_box_embedding_json_roundtrip() {
        let original = BoxEmbedding::new(vec![0.0, 1.0], vec![2.0, 3.0]);
        let json_val = original.to_json();
        let recovered = BoxEmbedding::from_json(&json_val).expect("should parse from JSON value");
        assert_eq!(original, recovered);
    }
}
