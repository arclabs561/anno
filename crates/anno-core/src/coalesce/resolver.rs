//! # Batch Entity Resolution with Union-Find
//!
//! This module provides the core batch entity resolution algorithm using the
//! **disjoint-set (union-find)** data structure with path compression and
//! union-by-rank.
//!
//! ## Algorithm Overview
//!
//! 1. **Collect** all tracks from all documents in the corpus
//! 2. **Compare** pairs: compute similarity (embedding cosine or string Jaccard)
//! 3. **Cluster** using union-find: if \( \text{sim}(t_i, t_j) \geq \theta \), merge
//! 4. **Create** identities: one per cluster, linked to constituent tracks
//!
//! ## Complexity Analysis
//!
//! Let \( n \) = number of tracks, \( m \) = number of merges performed.
//!
//! - **Pairwise comparison:** \( O(n^2) \) — the bottleneck for large corpora
//! - **Union-find operations:** \( O(m \cdot \alpha(n)) \) where \( \alpha \) is
//!   the inverse Ackermann function
//!
//! For all practical \( n \), \( \alpha(n) \leq 4 \), so union-find is effectively
//! \( O(m) \). The overall complexity is dominated by pairwise comparison.
//!
//! ## The Inverse Ackermann Function
//!
//! The Ackermann function \( A(m, n) \) grows faster than any primitive recursive
//! function. Its inverse \( \alpha(n) \) grows so slowly that:
//!
//! - \( \alpha(10^{80}) \leq 4 \) (more atoms than in the observable universe)
//!
//! Tarjan (1975) proved this bound is tight for union-find with path compression
//! and union-by-rank.
//!
//! ## Similarity Metrics
//!
//! Two metrics are provided:
//!
//! - **Embedding similarity** (cosine): \( \cos(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{||\mathbf{a}|| \cdot ||\mathbf{b}||} \)
//! - **String similarity** (Jaccard on words): \( J(A, B) = \frac{|A \cap B|}{|A \cup B|} \)
//!
//! If both tracks have embeddings, cosine is used; otherwise, Jaccard on word sets.
//!
//! ## Example
//!
//! ```
//! use anno_core::coalesce::Resolver;
//! use anno_core::Corpus;
//!
//! let resolver = Resolver::new()
//!     .with_threshold(0.7)
//!     .require_type_match(true);
//!
//! let mut corpus = Corpus::new();
//! // ... add documents with tracks ...
//!
//! let identities = resolver.resolve_inter_doc_coref(&mut corpus, None, None);
//! ```

use crate::core::{Corpus, Identity, IdentityId, IdentitySource, TrackId, TrackRef};
use std::collections::HashMap;

/// Coalescer for inter-document entity resolution.
#[derive(Debug, Clone)]
pub struct Resolver {
    similarity_threshold: f32,
    require_type_match: bool,
}

impl Resolver {
    /// Create a new resolver with default settings.
    pub fn new() -> Self {
        Self {
            similarity_threshold: 0.7,
            require_type_match: true,
        }
    }

    /// Create a new resolver with custom settings.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.similarity_threshold = threshold;
        self
    }

    /// Set whether to require entity type match for clustering.
    pub fn require_type_match(mut self, require: bool) -> Self {
        self.require_type_match = require;
        self
    }

    /// Coalesce inter-document entities across all documents in a corpus.
    ///
    /// This method clusters tracks from different documents that refer to the same
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
    /// * `corpus` - The corpus containing documents to resolve
    /// * `similarity_threshold` - Minimum similarity (0.0-1.0) to cluster tracks
    /// * `require_type_match` - Only cluster tracks with same entity type
    ///
    /// # Returns
    ///
    /// Vector of created identities, each linked to tracks from multiple documents.
    pub fn resolve_inter_doc_coref(
        &self,
        corpus: &mut Corpus,
        similarity_threshold: Option<f32>,
        require_type_match: Option<bool>,
    ) -> Vec<IdentityId> {
        let threshold = similarity_threshold.unwrap_or(self.similarity_threshold);
        let type_match = require_type_match.unwrap_or(self.require_type_match);

        // 1. Collect all track data (clone what we need to avoid borrow conflicts)
        #[derive(Debug, Clone)]
        struct TrackData {
            track_ref: TrackRef,
            canonical_surface: String,
            entity_type: Option<crate::TypeLabel>,
            cluster_confidence: crate::Confidence,
            embedding: Option<Vec<f32>>,
        }

        let mut track_data: Vec<TrackData> = Vec::new();
        // Collect document IDs first to avoid borrow checker issues
        let doc_ids: Vec<String> = corpus.documents().map(|d| d.id.clone()).collect();
        for doc_id in doc_ids {
            if let Some(doc) = corpus.get_document(&doc_id) {
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
        }

        if track_data.is_empty() {
            return vec![];
        }

        // 2. Cluster tracks using string similarity or embeddings
        // Uses embeddings if available (from track.embedding), otherwise falls back to string similarity
        let mut union_find: Vec<usize> = (0..track_data.len()).collect();
        let mut rank: Vec<usize> = vec![0; track_data.len()];

        fn find(parent: &mut [usize], i: usize) -> usize {
            if parent[i] != i {
                parent[i] = find(parent, parent[i]);
            }
            parent[i]
        }

        fn union(parent: &mut [usize], rank: &mut [usize], i: usize, j: usize) {
            let pi = find(parent, i);
            let pj = find(parent, j);
            if pi != pj {
                match rank[pi].cmp(&rank[pj]) {
                    std::cmp::Ordering::Less => parent[pi] = pj,
                    std::cmp::Ordering::Greater => parent[pj] = pi,
                    std::cmp::Ordering::Equal => {
                        parent[pi] = pj;
                        rank[pj] += 1;
                    }
                }
            }
        }

        // Compare all pairs
        for i in 0..track_data.len() {
            for j in (i + 1)..track_data.len() {
                let track_a = &track_data[i];
                let track_b = &track_data[j];

                // Type check
                if type_match && track_a.entity_type != track_b.entity_type {
                    continue;
                }

                // Compute similarity: prefer embeddings if BOTH available, fallback to string similarity
                // Edge case: If only one track has an embedding, we can't compare embeddings directly,
                // so we fall back to string similarity for consistency.
                let similarity =
                    if let (Some(emb_a), Some(emb_b)) = (&track_a.embedding, &track_b.embedding) {
                        // Both have embeddings: use cosine similarity
                        embedding_similarity(emb_a, emb_b)
                    } else {
                        // One or both missing embeddings: fallback to string similarity
                        // This handles: (Some, None), (None, Some), (None, None)
                        string_similarity(&track_a.canonical_surface, &track_b.canonical_surface)
                    };

                if similarity >= threshold {
                    union(&mut union_find, &mut rank, i, j);
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
            let identity = Identity {
                id: corpus.next_identity_id(), // Will be set by add_identity
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

            let identity_id = corpus.add_identity(identity);
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
                if let Some(doc) = corpus.get_document_mut(&doc_id) {
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
}

impl Default for Resolver {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute string similarity using Jaccard similarity on word sets.
///
/// Returns a value in [0.0, 1.0] where 1.0 is identical.
///
/// # Example
///
/// ```rust
/// use anno_core::coalesce::resolver::string_similarity;
///
/// let sim = string_similarity("Lynn Conway", "Lynn Conway");
/// assert_eq!(sim, 1.0);
///
/// let sim = string_similarity("Lynn Conway", "Conway");
/// assert!(sim > 0.0); // "Conway" shares one word with "Lynn Conway"
/// ```
pub fn string_similarity(a: &str, b: &str) -> f32 {
    // Normalize words: lowercase, strip possessives
    fn normalize_word(w: &str) -> String {
        let lower = w.to_lowercase();
        lower
            .trim_end_matches("'s")
            .trim_end_matches("'s")
            .trim_end_matches('\'')
            .to_string()
    }

    // Simple Jaccard similarity on normalized word sets
    let words_a: std::collections::HashSet<String> =
        a.split_whitespace().map(normalize_word).collect();
    let words_b: std::collections::HashSet<String> =
        b.split_whitespace().map(normalize_word).collect();

    if words_a.is_empty() && words_b.is_empty() {
        return 1.0;
    }
    if words_a.is_empty() || words_b.is_empty() {
        return 0.0;
    }

    let intersection = words_a.intersection(&words_b).count();
    let union = words_a.union(&words_b).count();

    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

/// Compute embedding similarity using cosine similarity.
///
/// Returns a value in [0.0, 1.0] where 1.0 is identical.
///
/// Formula: `cosine(a, b) = (a · b) / (||a|| × ||b||)`, normalized to [0, 1].
/// Measures angle between vectors, not magnitude, making it suitable for embeddings.
///
/// # Example
///
/// ```rust
/// use anno_core::coalesce::resolver::embedding_similarity;
///
/// let emb1 = vec![1.0, 0.0, 0.0];
/// let emb2 = vec![1.0, 0.0, 0.0];
/// let sim = embedding_similarity(&emb1, &emb2);
/// assert_eq!(sim, 1.0);
/// ```
pub fn embedding_similarity(emb_a: &[f32], emb_b: &[f32]) -> f32 {
    if emb_a.len() != emb_b.len() || emb_a.is_empty() {
        return 0.0;
    }

    // Cosine similarity
    let dot_product: f32 = emb_a.iter().zip(emb_b.iter()).map(|(a, b)| a * b).sum();
    let norm_a: f32 = emb_a.iter().map(|a| a * a).sum::<f32>().sqrt();
    let norm_b: f32 = emb_b.iter().map(|b| b * b).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    // Normalize to [0, 1] range (cosine similarity is [-1, 1])
    (dot_product / (norm_a * norm_b) + 1.0) / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_similarity_identical() {
        assert_eq!(string_similarity("hello world", "hello world"), 1.0);
    }

    #[test]
    fn test_string_similarity_partial() {
        let sim = string_similarity("hello world", "hello");
        assert!(sim > 0.0 && sim < 1.0);
        // "hello" is 1 word, "hello world" has 2 words
        // intersection = 1, union = 2, sim = 0.5
        assert!((sim - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_string_similarity_empty() {
        assert_eq!(string_similarity("", ""), 1.0);
        assert_eq!(string_similarity("hello", ""), 0.0);
        assert_eq!(string_similarity("", "hello"), 0.0);
    }

    #[test]
    fn test_string_similarity_symmetric() {
        let sim_ab = string_similarity("hello world", "world peace");
        let sim_ba = string_similarity("world peace", "hello world");
        assert_eq!(sim_ab, sim_ba);
    }

    #[test]
    fn test_embedding_similarity_identical() {
        let emb = vec![1.0, 0.0, 0.0];
        assert_eq!(embedding_similarity(&emb, &emb), 1.0);
    }

    #[test]
    fn test_embedding_similarity_orthogonal() {
        let emb1 = vec![1.0, 0.0];
        let emb2 = vec![0.0, 1.0];
        // Orthogonal vectors have cosine = 0, normalized to 0.5
        let sim = embedding_similarity(&emb1, &emb2);
        assert!((sim - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_embedding_similarity_opposite() {
        let emb1 = vec![1.0, 0.0];
        let emb2 = vec![-1.0, 0.0];
        // Opposite vectors have cosine = -1, normalized to 0.0
        let sim = embedding_similarity(&emb1, &emb2);
        assert!((sim - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_embedding_similarity_mismatched_length() {
        let emb1 = vec![1.0, 0.0];
        let emb2 = vec![1.0, 0.0, 0.0];
        assert_eq!(embedding_similarity(&emb1, &emb2), 0.0);
    }

    #[test]
    fn test_embedding_similarity_empty() {
        let emb1: Vec<f32> = vec![];
        let emb2: Vec<f32> = vec![];
        assert_eq!(embedding_similarity(&emb1, &emb2), 0.0);
    }

    #[test]
    fn test_embedding_similarity_zero_norm() {
        let emb1 = vec![0.0, 0.0];
        let emb2 = vec![1.0, 0.0];
        assert_eq!(embedding_similarity(&emb1, &emb2), 0.0);
    }

    #[test]
    fn test_resolver_builder() {
        let resolver = Resolver::new()
            .with_threshold(0.8)
            .require_type_match(false); // Default is true, test setting to false

        assert_eq!(resolver.similarity_threshold, 0.8);
        assert!(!resolver.require_type_match);
    }

    #[test]
    fn test_resolver_default() {
        let resolver = Resolver::default();
        assert_eq!(resolver.similarity_threshold, 0.7);
        assert!(resolver.require_type_match); // Default is true
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// String similarity is always in [0, 1]
        #[test]
        fn string_sim_bounded(a in ".*", b in ".*") {
            let sim = string_similarity(&a, &b);
            prop_assert!((0.0..=1.0).contains(&sim));
        }

        /// String similarity is symmetric
        #[test]
        fn string_sim_symmetric(a in "[a-z ]{0,30}", b in "[a-z ]{0,30}") {
            let sim_ab = string_similarity(&a, &b);
            let sim_ba = string_similarity(&b, &a);
            prop_assert!((sim_ab - sim_ba).abs() < 0.0001);
        }

        /// String similarity is reflexive (identical strings = 1.0)
        #[test]
        fn string_sim_reflexive(s in "[a-z]{1,20}") {
            let sim = string_similarity(&s, &s);
            prop_assert!((sim - 1.0).abs() < 0.0001);
        }

        /// Embedding similarity is bounded [0, 1] for our normalization
        #[test]
        fn embedding_sim_bounded(dim in 1usize..50, seed in any::<u64>()) {
            let mut rng = seed;
            let emb1: Vec<f32> = (0..dim).map(|_| {
                rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                (rng % 2000) as f32 / 1000.0 - 1.0
            }).collect();
            let emb2: Vec<f32> = (0..dim).map(|_| {
                rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                (rng % 2000) as f32 / 1000.0 - 1.0
            }).collect();

            let sim = embedding_similarity(&emb1, &emb2);
            prop_assert!((0.0..=1.0).contains(&sim),
                "Embedding similarity out of bounds: {}", sim);
        }

        /// Embedding similarity is symmetric
        #[test]
        fn embedding_sim_symmetric(dim in 1usize..20, seed in any::<u64>()) {
            let mut rng = seed;
            let emb1: Vec<f32> = (0..dim).map(|_| {
                rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                (rng % 100) as f32 / 100.0
            }).collect();
            let emb2: Vec<f32> = (0..dim).map(|_| {
                rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                (rng % 100) as f32 / 100.0
            }).collect();

            let sim_ab = embedding_similarity(&emb1, &emb2);
            let sim_ba = embedding_similarity(&emb2, &emb1);
            prop_assert!((sim_ab - sim_ba).abs() < 0.0001,
                "Embedding similarity not symmetric: {} vs {}", sim_ab, sim_ba);
        }
    }
}
