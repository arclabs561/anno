//! Coreference resolution utilities and data structures.

use crate::{Confidence, Entity};
use std::collections::HashMap;

// Coreference Resolution
// =============================================================================

/// A coreference cluster (mentions referring to same entity).
///
/// This is a different shape from `coref::resolve::CorefCluster` -- it stores
/// entity indices rather than mention texts, for use with embedding-based resolution.
#[derive(Debug, Clone)]
pub struct CoreferenceCluster {
    /// Cluster ID
    pub id: u64,
    /// Member entities (indices into entity list)
    pub members: Vec<usize>,
    /// Representative entity index (most informative mention)
    pub representative: usize,
    /// Canonical name (from representative)
    pub canonical_name: String,
}

/// Configuration for coreference resolution.
#[derive(Debug, Clone)]
pub struct CoreferenceConfig {
    /// Minimum cosine similarity to link mentions
    pub similarity_threshold: Confidence,
    /// Maximum token distance between coreferent mentions
    pub max_distance: Option<usize>,
    /// Whether to use exact string matching as a signal
    pub use_string_match: bool,
}

impl Default for CoreferenceConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: Confidence::new(0.85),
            max_distance: Some(500),
            use_string_match: true,
        }
    }
}

/// Resolve coreferences between entities using embedding similarity.
///
/// # Algorithm
///
/// 1. Compute pairwise cosine similarity between entity embeddings
/// 2. Link entities above threshold (with optional distance constraint)
/// 3. Build clusters via transitive closure
/// 4. Select representative (longest/most informative mention)
///
/// # Example
///
/// Input entities: ["Lynn Conway", "She", "The engineer", "Conway"]
/// Output clusters: [{0, 1, 2, 3}] with canonical_name = "Lynn Conway"
pub fn resolve_coreferences(
    entities: &[Entity],
    embeddings: &[f32], // [num_entities, hidden_dim]
    hidden_dim: usize,
    config: &CoreferenceConfig,
) -> Vec<CoreferenceCluster> {
    let n = entities.len();
    if n == 0 {
        return vec![];
    }

    // Union-find for clustering
    let mut parent: Vec<usize> = (0..n).collect();

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

    // Check all pairs
    for i in 0..n {
        for j in (i + 1)..n {
            // String match check (fast path)
            if config.use_string_match {
                let text_i = entities[i].text.to_lowercase();
                let text_j = entities[j].text.to_lowercase();
                if text_i == text_j || text_i.contains(&text_j) || text_j.contains(&text_i) {
                    // Same entity type required
                    if entities[i].entity_type == entities[j].entity_type {
                        union(&mut parent, i, j);
                        continue;
                    }
                }
            }

            // Distance check
            if let Some(max_dist) = config.max_distance {
                let dist = if entities[i].end() <= entities[j].start() {
                    entities[j].start() - entities[i].end()
                } else {
                    entities[i].start().saturating_sub(entities[j].end())
                };
                if dist > max_dist {
                    continue;
                }
            }

            // Embedding similarity
            if embeddings.len() >= (j + 1) * hidden_dim {
                let emb_i = &embeddings[i * hidden_dim..(i + 1) * hidden_dim];
                let emb_j = &embeddings[j * hidden_dim..(j + 1) * hidden_dim];

                let dot: f32 = emb_i.iter().zip(emb_j).map(|(a, b)| a * b).sum();
                let norm_i: f32 = emb_i.iter().map(|x| x * x).sum::<f32>().sqrt();
                let norm_j: f32 = emb_j.iter().map(|x| x * x).sum::<f32>().sqrt();
                let similarity = if norm_i > 0.0 && norm_j > 0.0 {
                    dot / (norm_i * norm_j)
                } else {
                    0.0
                };

                if similarity >= config.similarity_threshold.value() as f32 {
                    // Same entity type required
                    if entities[i].entity_type == entities[j].entity_type {
                        union(&mut parent, i, j);
                    }
                }
            }
        }
    }

    // Build clusters
    let mut cluster_members: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n {
        let root = find(&mut parent, i);
        cluster_members.entry(root).or_default().push(i);
    }

    // Convert to CoreferenceCluster
    let mut clusters = Vec::new();
    let mut cluster_id = 0u64;

    for (_root, members) in cluster_members {
        if members.len() > 1 {
            // Find representative (longest mention)
            let representative = *members
                .iter()
                .max_by_key(|&&i| entities[i].text.len())
                .unwrap_or(&members[0]);

            clusters.push(CoreferenceCluster {
                id: cluster_id,
                members,
                representative,
                canonical_name: entities[representative].text.clone(),
            });
            cluster_id += 1;
        }
    }

    clusters
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Entity, EntityType};

    // =========================================================================
    // CoreferenceConfig
    // =========================================================================

    #[test]
    fn test_coreference_config_default() {
        let config = CoreferenceConfig::default();
        assert!((config.similarity_threshold.value() - 0.85).abs() < f64::EPSILON);
        assert_eq!(config.max_distance, Some(500));
        assert!(config.use_string_match);
    }

    #[test]
    fn test_coreference_config_clone() {
        let config = CoreferenceConfig {
            similarity_threshold: Confidence::new(0.7),
            max_distance: None,
            use_string_match: false,
        };
        let cloned = config.clone();
        assert!((cloned.similarity_threshold.value() - 0.7).abs() < f64::EPSILON);
        assert!(cloned.max_distance.is_none());
        assert!(!cloned.use_string_match);
    }

    // =========================================================================
    // resolve_coreferences: embedding-based clustering
    // =========================================================================

    #[test]
    fn test_coreference_embedding_similarity_clusters() {
        // Two entities of the same type with identical embeddings should cluster
        let entities = vec![
            Entity::new("She", EntityType::Person, 0, 3, 0.9),
            Entity::new("Her", EntityType::Person, 10, 13, 0.9),
        ];

        let hidden_dim = 4;
        // Identical embeddings -> cosine similarity = 1.0, above any threshold
        let embeddings = vec![
            1.0, 0.0, 0.0, 0.0, // entity 0
            1.0, 0.0, 0.0, 0.0, // entity 1
        ];

        let config = CoreferenceConfig {
            similarity_threshold: Confidence::new(0.85),
            max_distance: Some(500),
            use_string_match: false, // disable string match to test embedding path
        };

        let clusters = resolve_coreferences(&entities, &embeddings, hidden_dim, &config);
        assert_eq!(clusters.len(), 1, "identical embeddings should cluster");
        assert_eq!(clusters[0].members.len(), 2);
    }

    #[test]
    fn test_coreference_embedding_below_threshold_no_cluster() {
        // Orthogonal embeddings -> similarity = 0, below threshold
        let entities = vec![
            Entity::new("Alice", EntityType::Person, 0, 5, 0.9),
            Entity::new("Bob", EntityType::Person, 10, 13, 0.9),
        ];

        let hidden_dim = 4;
        let embeddings = vec![
            1.0, 0.0, 0.0, 0.0, // entity 0
            0.0, 1.0, 0.0, 0.0, // entity 1
        ];

        let config = CoreferenceConfig {
            similarity_threshold: Confidence::new(0.85),
            max_distance: Some(500),
            use_string_match: false,
        };

        let clusters = resolve_coreferences(&entities, &embeddings, hidden_dim, &config);
        assert!(
            clusters.is_empty(),
            "orthogonal embeddings should not cluster"
        );
    }

    #[test]
    fn test_coreference_representative_is_longest_mention() {
        let entities = vec![
            Entity::new("Dr. Robert Johnson", EntityType::Person, 0, 18, 0.9),
            Entity::new("Johnson", EntityType::Person, 30, 37, 0.9),
            Entity::new("He", EntityType::Person, 50, 52, 0.9),
        ];

        // All identical embeddings
        let hidden_dim = 4;
        let embeddings = vec![
            1.0, 0.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, 0.0, //
        ];

        let config = CoreferenceConfig {
            similarity_threshold: Confidence::new(0.5),
            max_distance: Some(500),
            use_string_match: false,
        };

        let clusters = resolve_coreferences(&entities, &embeddings, hidden_dim, &config);
        assert_eq!(clusters.len(), 1);
        assert_eq!(
            clusters[0].canonical_name, "Dr. Robert Johnson",
            "representative should be the longest mention"
        );
    }

    #[test]
    fn test_coreference_no_distance_limit() {
        // With max_distance = None, even distant mentions can cluster
        let entities = vec![
            Entity::new("Alice", EntityType::Person, 0, 5, 0.9),
            Entity::new("Alice", EntityType::Person, 10000, 10005, 0.9),
        ];

        let embeddings = vec![0.0f32; 2 * 4];
        let config = CoreferenceConfig {
            similarity_threshold: Confidence::new(0.85),
            max_distance: None, // no distance limit
            use_string_match: true,
        };

        let clusters = resolve_coreferences(&entities, &embeddings, 4, &config);
        assert_eq!(
            clusters.len(),
            1,
            "no distance limit should allow clustering"
        );
    }

    #[test]
    fn test_coreference_two_separate_clusters() {
        // Four entities: two "Alice" (Person) and two "Acme" (Org)
        let entities = vec![
            Entity::new("Alice", EntityType::Person, 0, 5, 0.9),
            Entity::new("Alice", EntityType::Person, 20, 25, 0.9),
            Entity::new("Acme", EntityType::Organization, 40, 44, 0.9),
            Entity::new("Acme", EntityType::Organization, 60, 64, 0.9),
        ];

        let embeddings = vec![0.0f32; 4 * 768];
        let clusters =
            resolve_coreferences(&entities, &embeddings, 768, &CoreferenceConfig::default());

        assert_eq!(clusters.len(), 2, "should form two separate clusters");
        // Each cluster should have exactly 2 members
        for cluster in &clusters {
            assert_eq!(cluster.members.len(), 2);
        }
    }

    #[test]
    fn test_coreference_cluster_ids_are_sequential() {
        let entities = vec![
            Entity::new("A", EntityType::Person, 0, 1, 0.9),
            Entity::new("A", EntityType::Person, 5, 6, 0.9),
            Entity::new("B", EntityType::Organization, 10, 11, 0.9),
            Entity::new("B", EntityType::Organization, 15, 16, 0.9),
        ];

        let embeddings = vec![0.0f32; 4 * 4];
        let clusters =
            resolve_coreferences(&entities, &embeddings, 4, &CoreferenceConfig::default());

        let mut ids: Vec<u64> = clusters.iter().map(|c| c.id).collect();
        ids.sort();
        // IDs should be 0, 1, ...
        for (i, id) in ids.iter().enumerate() {
            assert_eq!(*id, i as u64, "cluster IDs should be sequential");
        }
    }

    // =========================================================================
    // CoreferenceCluster
    // =========================================================================

    #[test]
    fn test_coreference_cluster_debug_and_clone() {
        let cluster = CoreferenceCluster {
            id: 0,
            members: vec![0, 1, 2],
            representative: 0,
            canonical_name: "Test Entity".to_string(),
        };
        let cloned = cluster.clone();
        assert_eq!(cloned.id, 0);
        assert_eq!(cloned.members, vec![0, 1, 2]);
        assert_eq!(cloned.canonical_name, "Test Entity");

        let debug = format!("{:?}", cluster);
        assert!(debug.contains("Test Entity"));
    }
}
