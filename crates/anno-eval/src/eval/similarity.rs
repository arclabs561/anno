//! Embedding-based entity similarity and retrieval.
//!
//! Provides infrastructure for:
//! - Computing entity embeddings from text
//! - Similarity search (find similar entities)
//! - Clustering entities by embedding distance
//! - Nearest neighbor retrieval for entity linking
//!
//! # Research Background
//!
//! Entity embeddings capture semantic similarity that surface matching misses:
//! - "NYC" and "New York City" are textually different but semantically identical
//! - Context-sensitive embeddings capture sense disambiguation
//!
//! # Architecture
//!
//! ```text
//! ┌───────────────┐     ┌──────────────┐     ┌─────────────┐
//! │ Entity Text   │────►│ Encoder      │────►│ Embedding   │
//! │ + Context     │     │ (BERT, etc)  │     │ Vector      │
//! └───────────────┘     └──────────────┘     └─────────────┘
//!                                                   │
//!                       ┌───────────────────────────┴────────┐
//!                       ▼                                    ▼
//!                ┌─────────────┐                      ┌─────────────┐
//!                │ FAISS Index │                      │ Similarity  │
//!                │ (ANN search)│                      │ Computation │
//!                └─────────────┘                      └─────────────┘
//! ```
//!
//! # Example
//!
//! ```rust
//! use anno::eval::similarity::{EntityEmbedding, SimilarityIndex};
//!
//! // Create embeddings
//! let e1 = EntityEmbedding::new("Q937", "Albert Einstein", vec![0.1, 0.2, 0.3]);
//! let e2 = EntityEmbedding::new("Q317521", "Elon Musk", vec![0.1, 0.25, 0.35]);
//!
//! // Build index
//! let index = SimilarityIndex::new(vec![e1.clone(), e2.clone()]);
//!
//! // Find similar
//! let similar = index.find_similar(&e1.embedding, 5);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Entity Embedding
// =============================================================================

/// An entity with its embedding vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityEmbedding {
    /// Entity ID (e.g., Wikidata QID)
    pub id: String,
    /// Entity label/name
    pub label: String,
    /// Embedding vector
    pub embedding: Vec<f32>,
    /// Optional entity type
    pub entity_type: Option<String>,
    /// Optional description
    pub description: Option<String>,
}

impl EntityEmbedding {
    /// Create a new entity embedding.
    pub fn new(id: &str, label: &str, embedding: Vec<f32>) -> Self {
        Self {
            id: id.to_string(),
            label: label.to_string(),
            embedding,
            entity_type: None,
            description: None,
        }
    }

    /// Set entity type.
    pub fn with_type(mut self, entity_type: &str) -> Self {
        self.entity_type = Some(entity_type.to_string());
        self
    }

    /// Compute cosine similarity with another embedding.
    #[must_use]
    pub fn cosine_similarity(&self, other: &[f32]) -> f32 {
        cosine_similarity(&self.embedding, other)
    }

    /// Compute L2 (Euclidean) distance with another embedding.
    #[must_use]
    pub fn l2_distance(&self, other: &[f32]) -> f32 {
        l2_distance(&self.embedding, other)
    }

    /// Compute dot product with another embedding.
    #[must_use]
    pub fn dot_product(&self, other: &[f32]) -> f32 {
        dot_product(&self.embedding, other)
    }

    /// Get embedding dimension.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.embedding.len()
    }
}

// =============================================================================
// Similarity Functions
// =============================================================================

/// Compute cosine similarity between two vectors.
#[must_use]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have same dimension");

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// Compute L2 (Euclidean) distance between two vectors.
#[must_use]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have same dimension");

    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Compute dot product of two vectors.
#[must_use]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have same dimension");

    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Normalize a vector to unit length.
#[must_use]
pub fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        v.to_vec()
    } else {
        v.iter().map(|x| x / norm).collect()
    }
}

/// Compute average of multiple vectors.
#[must_use]
pub fn average_embeddings(embeddings: &[Vec<f32>]) -> Vec<f32> {
    if embeddings.is_empty() {
        return Vec::new();
    }

    let dim = embeddings[0].len();
    let mut result = vec![0.0; dim];

    for emb in embeddings {
        for (i, v) in emb.iter().enumerate() {
            result[i] += v;
        }
    }

    let n = embeddings.len() as f32;
    for v in &mut result {
        *v /= n;
    }

    result
}

// =============================================================================
// Similarity Index
// =============================================================================

/// Distance metric for similarity search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DistanceMetric {
    /// Cosine similarity (higher = more similar)
    #[default]
    Cosine,
    /// L2 (Euclidean) distance (lower = more similar)
    L2,
    /// Dot product (higher = more similar)
    DotProduct,
}

/// Result of a similarity search.
#[derive(Debug, Clone)]
pub struct SimilarityResult {
    /// Entity ID
    pub id: String,
    /// Entity label
    pub label: String,
    /// Similarity/distance score
    pub score: f32,
    /// Rank (1-indexed)
    pub rank: usize,
}

/// Simple brute-force similarity index.
///
/// For large-scale use, integrate with FAISS or Annoy for ANN search.
#[derive(Debug, Clone, Default)]
pub struct SimilarityIndex {
    /// Stored embeddings
    embeddings: Vec<EntityEmbedding>,
    /// ID to index mapping
    id_to_idx: HashMap<String, usize>,
    /// Distance metric
    metric: DistanceMetric,
}

impl SimilarityIndex {
    /// Create a new index with given embeddings.
    pub fn new(embeddings: Vec<EntityEmbedding>) -> Self {
        let id_to_idx: HashMap<String, usize> = embeddings
            .iter()
            .enumerate()
            .map(|(i, e)| (e.id.clone(), i))
            .collect();

        Self {
            embeddings,
            id_to_idx,
            metric: DistanceMetric::default(),
        }
    }

    /// Set distance metric.
    pub fn with_metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Add an embedding to the index.
    pub fn add(&mut self, embedding: EntityEmbedding) {
        let idx = self.embeddings.len();
        self.id_to_idx.insert(embedding.id.clone(), idx);
        self.embeddings.push(embedding);
    }

    /// Find k most similar embeddings to query.
    pub fn find_similar(&self, query: &[f32], k: usize) -> Vec<SimilarityResult> {
        let mut scores: Vec<(usize, f32)> = self
            .embeddings
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let score = match self.metric {
                    DistanceMetric::Cosine => cosine_similarity(query, &e.embedding),
                    DistanceMetric::L2 => -l2_distance(query, &e.embedding), // Negate so higher is better
                    DistanceMetric::DotProduct => dot_product(query, &e.embedding),
                };
                (i, score)
            })
            .collect();

        // Sort by score descending
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scores
            .into_iter()
            .take(k)
            .enumerate()
            .map(|(rank, (idx, score))| {
                let e = &self.embeddings[idx];
                SimilarityResult {
                    id: e.id.clone(),
                    label: e.label.clone(),
                    score,
                    rank: rank + 1,
                }
            })
            .collect()
    }

    /// Find similar by entity ID.
    pub fn find_similar_to(&self, entity_id: &str, k: usize) -> Vec<SimilarityResult> {
        if let Some(&idx) = self.id_to_idx.get(entity_id) {
            let query = &self.embeddings[idx].embedding;
            // Exclude the query itself
            self.find_similar(query, k + 1)
                .into_iter()
                .filter(|r| r.id != entity_id)
                .take(k)
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get embedding by ID.
    pub fn get(&self, entity_id: &str) -> Option<&EntityEmbedding> {
        self.id_to_idx.get(entity_id).map(|&i| &self.embeddings[i])
    }

    /// Number of embeddings in the index.
    #[must_use]
    pub fn len(&self) -> usize {
        self.embeddings.len()
    }

    /// Is the index empty?
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }
}

// =============================================================================
// Mention-Entity Similarity
// =============================================================================

/// Context for computing mention-in-context embeddings.
#[derive(Debug, Clone)]
pub struct MentionContext {
    /// Mention text
    pub mention: String,
    /// Left context
    pub left_context: String,
    /// Right context
    pub right_context: String,
    /// Full sentence
    pub sentence: String,
}

impl MentionContext {
    /// Create a new mention context.
    pub fn new(mention: &str, left: &str, right: &str) -> Self {
        Self {
            mention: mention.to_string(),
            left_context: left.to_string(),
            right_context: right.to_string(),
            sentence: format!("{}{}{}", left, mention, right),
        }
    }

    /// Extract from text with character offsets.
    pub fn from_text(text: &str, start: usize, end: usize, context_window: usize) -> Self {
        let chars: Vec<char> = text.chars().collect();
        let text_len = chars.len();

        let ctx_start = start.saturating_sub(context_window);
        let ctx_end = (end + context_window).min(text_len);

        let left: String = chars[ctx_start..start].iter().collect();
        let mention: String = chars[start..end].iter().collect();
        let right: String = chars[end..ctx_end].iter().collect();

        Self {
            mention,
            left_context: left.clone(),
            right_context: right.clone(),
            sentence: format!(
                "{}{}{}",
                left,
                chars[start..end].iter().collect::<String>(),
                right
            ),
        }
    }
}

/// Pair-wise similarity between a mention and candidate entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MentionEntitySimilarity {
    /// Mention text
    pub mention: String,
    /// Entity ID
    pub entity_id: String,
    /// Entity label
    pub entity_label: String,
    /// Embedding similarity score
    pub embedding_sim: f32,
    /// String similarity score
    pub string_sim: f32,
    /// Combined score
    pub combined_score: f32,
}

/// Compute string similarity (simple Jaccard on characters).
#[must_use]
pub fn char_jaccard(a: &str, b: &str) -> f32 {
    use std::collections::HashSet;

    let chars_a: HashSet<char> = a.to_lowercase().chars().collect();
    let chars_b: HashSet<char> = b.to_lowercase().chars().collect();

    if chars_a.is_empty() && chars_b.is_empty() {
        return 1.0;
    }

    let intersection = chars_a.intersection(&chars_b).count();
    let union = chars_a.union(&chars_b).count();

    intersection as f32 / union as f32
}

// =============================================================================
// Entity Clustering
// =============================================================================

/// Simple hierarchical clustering result.
#[derive(Debug, Clone)]
pub struct EntityCluster {
    /// Cluster ID
    pub id: usize,
    /// Entity IDs in this cluster
    pub entity_ids: Vec<String>,
    /// Cluster centroid (average embedding)
    pub centroid: Vec<f32>,
    /// Intra-cluster similarity
    pub cohesion: f32,
}

/// Perform simple k-means-like clustering on embeddings.
pub fn cluster_embeddings(
    index: &SimilarityIndex,
    num_clusters: usize,
    max_iterations: usize,
) -> Vec<EntityCluster> {
    if index.is_empty() || num_clusters == 0 {
        return Vec::new();
    }

    let k = num_clusters.min(index.len());
    let dim = index.embeddings[0].dim();

    // Use `clump` as the single k-means implementation (k-means++ init + Lloyd).
    let points: Vec<&[f32]> = index
        .embeddings
        .iter()
        .map(|e| e.embedding.as_slice())
        .collect();
    let cfg = clump::KMeansConfig {
        k,
        max_iters: max_iterations.max(1),
        tol: 1e-4,
        seed: 42,
    };
    let res = match clump::kmeans(&points, &cfg) {
        Ok(r) => r,
        Err(_) => {
            // Conservative fallback: one cluster containing everything.
            let centroid = average_embeddings(
                &index
                    .embeddings
                    .iter()
                    .map(|e| e.embedding.clone())
                    .collect::<Vec<_>>(),
            );
            return vec![EntityCluster {
                id: 0,
                entity_ids: index.embeddings.iter().map(|e| e.id.clone()).collect(),
                centroid,
                cohesion: 0.0,
            }];
        }
    };
    let centroids = res.centroids;
    let assignments = res.assignments;

    // Build cluster objects
    let mut clusters: Vec<EntityCluster> = (0..k)
        .map(|c| EntityCluster {
            id: c,
            entity_ids: Vec::new(),
            centroid: centroids.get(c).cloned().unwrap_or_else(|| vec![0.0; dim]),
            cohesion: 0.0,
        })
        .collect();

    for (i, &cluster) in assignments.iter().enumerate() {
        if let Some(c) = clusters.get_mut(cluster) {
            c.entity_ids.push(index.embeddings[i].id.clone());
        }
    }

    // Compute cohesion
    for cluster in &mut clusters {
        if cluster.entity_ids.is_empty() {
            continue;
        }

        let sims: Vec<f32> = cluster
            .entity_ids
            .iter()
            .filter_map(|id| index.get(id))
            .map(|e| cosine_similarity(&e.embedding, &cluster.centroid))
            .collect();

        cluster.cohesion = if sims.is_empty() {
            0.0
        } else {
            sims.iter().sum::<f32>() / sims.len() as f32
        };
    }

    clusters
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 1e-6);
    }

    #[test]
    fn test_l2_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert!((l2_distance(&a, &b) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_entity_embedding() {
        let e = EntityEmbedding::new("Q937", "Einstein", vec![0.5, 0.5, 0.0]);
        assert_eq!(e.dim(), 3);
        assert!(e.cosine_similarity(&[0.5, 0.5, 0.0]) > 0.99);
    }

    #[test]
    fn test_similarity_index() {
        let embeddings = vec![
            EntityEmbedding::new("Q1", "Einstein", vec![1.0, 0.0, 0.0]),
            EntityEmbedding::new("Q2", "Curie", vec![0.9, 0.1, 0.0]),
            EntityEmbedding::new("Q3", "Google", vec![0.0, 1.0, 0.0]),
        ];

        let index = SimilarityIndex::new(embeddings);

        // Find similar to Einstein-like vector
        let results = index.find_similar(&[1.0, 0.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "Q1"); // Einstein is most similar
        assert_eq!(results[1].id, "Q2"); // Curie is second
    }

    #[test]
    fn test_find_similar_to() {
        let embeddings = vec![
            EntityEmbedding::new("Q1", "A", vec![1.0, 0.0]),
            EntityEmbedding::new("Q2", "B", vec![0.9, 0.1]),
            EntityEmbedding::new("Q3", "C", vec![0.0, 1.0]),
        ];

        let index = SimilarityIndex::new(embeddings);
        let results = index.find_similar_to("Q1", 2);

        // Should not include Q1 itself
        assert!(!results.iter().any(|r| r.id == "Q1"));
        assert_eq!(results[0].id, "Q2"); // B is most similar to A
    }

    #[test]
    fn test_mention_context() {
        let text = "Albert Einstein was a physicist.";
        let ctx = MentionContext::from_text(text, 0, 15, 10);
        assert_eq!(ctx.mention, "Albert Einstein");
    }

    #[test]
    fn test_char_jaccard() {
        assert!(char_jaccard("hello", "hello") > 0.99);
        assert!(char_jaccard("hello", "helo") > 0.8);
        assert!(char_jaccard("abc", "xyz") < 0.01);
    }

    #[test]
    fn test_clustering() {
        let embeddings = vec![
            EntityEmbedding::new("Q1", "A", vec![1.0, 0.0]),
            EntityEmbedding::new("Q2", "B", vec![0.9, 0.1]),
            EntityEmbedding::new("Q3", "C", vec![0.0, 1.0]),
            EntityEmbedding::new("Q4", "D", vec![0.1, 0.9]),
        ];

        let index = SimilarityIndex::new(embeddings);
        let clusters = cluster_embeddings(&index, 2, 10);

        assert_eq!(clusters.len(), 2);
        // Q1 and Q2 should be in one cluster, Q3 and Q4 in another
    }

    #[test]
    fn test_normalize() {
        let v = vec![3.0, 4.0];
        let n = normalize(&v);
        let len: f32 = n.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((len - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_average_embeddings() {
        let embeddings = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let avg = average_embeddings(&embeddings);
        assert!((avg[0] - 0.5).abs() < 1e-6);
        assert!((avg[1] - 0.5).abs() < 1e-6);
    }
}
