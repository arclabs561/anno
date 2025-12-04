//! Inter-document entity coalescing.

use anno_core::{Corpus, Identity, IdentityId, IdentitySource, TrackId, TrackRef};
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
            entity_type: Option<String>,
            cluster_confidence: f32,
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
        // TODO: Use embeddings if available, fallback to string similarity
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
                if type_match && track_a.entity_type != track_b.entity_type {
                    continue;
                }

                // Compute similarity: prefer embeddings if available, fallback to string similarity
                let similarity =
                    if let (Some(emb_a), Some(emb_b)) = (&track_a.embedding, &track_b.embedding) {
                        // Cosine similarity for embeddings
                        embedding_similarity(emb_a, emb_b)
                    } else {
                        // Fallback to string similarity
                        string_similarity(&track_a.canonical_surface, &track_b.canonical_surface)
                    };

                if similarity >= threshold {
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
            let identity = Identity {
                id: corpus.next_identity_id(), // Will be set by add_identity
                canonical_name: first_track.canonical_surface.clone(),
                entity_type: first_track.entity_type.clone(),
                kb_id: None,
                kb_name: None,
                description: None,
                embedding: first_track.embedding.clone(),
                box_embedding: None,
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
/// ```
/// use anno_coalesce::string_similarity;
///
/// let sim = string_similarity("Marie Curie", "Marie Curie");
/// assert_eq!(sim, 1.0);
///
/// let sim = string_similarity("Marie Curie", "Curie");
/// assert!(sim > 0.0); // "Curie" shares one word with "Marie Curie"
/// ```
pub fn string_similarity(a: &str, b: &str) -> f32 {
    // Simple Jaccard similarity on word sets
    let words_a: std::collections::HashSet<&str> = a.split_whitespace().collect();
    let words_b: std::collections::HashSet<&str> = b.split_whitespace().collect();

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
/// # Example
///
/// ```
/// use anno_coalesce::embedding_similarity;
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
