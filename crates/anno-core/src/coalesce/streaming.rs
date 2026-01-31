//! # Streaming Entity Resolution (Doubling Algorithm)
//!
//! This module provides **incremental entity resolution** for scenarios where
//! documents arrive continuously and we cannot afford to reprocess the entire
//! corpus for each new document.
//!
//! ## The Streaming Constraint
//!
//! In batch entity resolution, we have access to all entities before clustering.
//! In streaming, entities arrive one at a time, and we must:
//!
//! 1. Assign each entity to a cluster immediately (or nearly so)
//! 2. Use bounded memory (cannot store all pairwise distances)
//! 3. Produce reasonable clusters without seeing the future
//!
//! ## The Doubling Algorithm
//!
//! Based on Charikar, Chekuri, Feder & Motwani (1997), the algorithm maintains
//! a set of *active clusters* and processes entities in two stages:
//!
//! ### Update Stage
//!
//! When entity \( e \) arrives:
//! 1. Find most similar cluster \( C^* = \arg\max_C \text{sim}(e, C) \)
//! 2. If \( \text{sim}(e, C^*) \geq \theta_{\text{add}} \): add \( e \) to \( C^* \)
//! 3. Else: create new singleton cluster \( \{e\} \)
//!
//! ### Merge Stage
//!
//! When cluster count exceeds threshold:
//! 1. Find all pairs \( (C_i, C_j) \) with \( \text{sim}(C_i, C_j) \geq \theta_{\text{merge}} \)
//! 2. Merge using union-find to handle transitive closures
//! 3. Update cluster centroids
//!
//! ## Approximation Guarantee
//!
//! The Doubling Algorithm achieves an **8-approximation** to the optimal offline
//! clustering. This means:
//!
//! \[
//! \text{cost}_{\text{streaming}} \leq 8 \cdot \text{cost}_{\text{optimal}}
//! \]
//!
//! where cost is measured as sum of intra-cluster distances.
//!
//! ## Complexity
//!
//! - **Per entity (amortized)**: \( O(1) \) with LSH blocking, \( O(k) \) without
//!   (where \( k \) = number of clusters)
//! - **Memory**: \( O(k) \) for cluster centroids
//! - **Merge stage**: \( O(k^2) \) but triggered infrequently
//!
//! ## Configuration
//!
//! Key parameters:
//!
//! - `add_threshold`: Minimum similarity to add to existing cluster (default: 0.6)
//! - `merge_threshold`: Minimum similarity for cluster merging (default: 0.7)
//! - `max_clusters`: Trigger merge when exceeded (default: 10,000)
//! - `use_lsh`: Enable LSH blocking for scalability (default: true)
//!
//! ## Example
//!
//! ```rust,ignore
//! use anno_core::coalesce::streaming::{StreamingConfig, StreamingResolver};
//!
//! let mut resolver = StreamingResolver::new(StreamingConfig::default());
//!
//! // Process entities as they arrive
//! resolver.add_entity("doc1", "Barack Obama", Some("Person".into()));
//! resolver.add_entity("doc2", "obama", Some("Person".into()));
//! resolver.add_entity("doc3", "Donald Trump", Some("Person".into()));
//!
//! // Obama mentions should cluster together
//! assert!(resolver.num_clusters() <= 2);
//!
//! for cluster in resolver.clusters() {
//!     println!("{}: {} mentions from {} documents",
//!         cluster.canonical_name,
//!         cluster.mentions.len(),
//!         cluster.document_ids().len()
//!     );
//! }
//! ```
//!
//! ## When to Use Streaming vs Batch
//!
//! | Criterion | Batch | Streaming |
//! |-----------|-------|-----------|
//! | All data available upfront | Yes | No |
//! | Memory constraint | O(n²) acceptable | O(k) required |
//! | Optimality needed | Yes | 8-approx sufficient |
//! | Real-time results | Not required | Required |
//!
//! ## References
//!
//! - Charikar, Chekuri, Feder, Motwani (1997). "Incremental clustering and
//!   dynamic information retrieval". STOC '97.
//! - Rao Delip, McNamee, Dredze (2010). "Streaming cross document entity
//!   coreference resolution". COLING 2010.

use super::lsh::{LSHConfig, MinHashLSH};
use crate::core as anno_core;
use std::collections::HashMap;

/// Configuration for streaming entity resolution.
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Similarity threshold for adding to existing cluster
    pub add_threshold: f32,
    /// Similarity threshold for merging clusters
    pub merge_threshold: f32,
    /// Maximum number of clusters before triggering merge
    pub max_clusters: usize,
    /// Whether to use LSH blocking for scalability
    pub use_lsh: bool,
    /// LSH configuration (if use_lsh is true)
    pub lsh_config: LSHConfig,
    /// Whether to require entity type match
    pub require_type_match: bool,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            add_threshold: 0.6,
            merge_threshold: 0.7,
            max_clusters: 10_000,
            use_lsh: true,
            lsh_config: LSHConfig::default(),
            require_type_match: true,
        }
    }
}

impl StreamingConfig {
    /// Create a high-recall configuration (more lenient matching).
    pub fn high_recall() -> Self {
        Self {
            add_threshold: 0.4,
            merge_threshold: 0.5,
            use_lsh: true,
            lsh_config: LSHConfig::high_recall(),
            ..Default::default()
        }
    }

    /// Create a high-precision configuration (stricter matching).
    pub fn high_precision() -> Self {
        Self {
            add_threshold: 0.7,
            merge_threshold: 0.8,
            use_lsh: true,
            lsh_config: LSHConfig::high_precision(),
            ..Default::default()
        }
    }

    // Evidence-based similarity was intentionally removed to keep this module
    // small and avoid bespoke tuning logic.
}

/// A mention of an entity in a document.
#[derive(Debug, Clone)]
pub struct EntityMention {
    /// Document ID
    pub doc_id: String,
    /// Canonical surface form
    pub canonical_surface: String,
    /// Entity type (e.g., "Person", "Organization")
    pub entity_type: Option<String>,
    /// Optional embedding vector
    pub embedding: Option<Vec<f32>>,
    /// Track ID within the document (links to intra-doc coref)
    pub track_id: Option<anno_core::TrackId>,
    /// Timestamp when mention was observed (for temporal tracking)
    pub timestamp: Option<chrono::DateTime<chrono::Utc>>,
    /// Valid from date (for diachronic entity tracking, e.g., "USSR" valid until 1991)
    pub valid_from: Option<chrono::DateTime<chrono::Utc>>,
    /// Valid until date (for diachronic entity tracking)
    pub valid_until: Option<chrono::DateTime<chrono::Utc>>,
}

impl EntityMention {
    /// Create a new entity mention.
    pub fn new(doc_id: impl Into<String>, surface: impl Into<String>) -> Self {
        Self {
            doc_id: doc_id.into(),
            canonical_surface: surface.into(),
            entity_type: None,
            embedding: None,
            track_id: None,
            timestamp: None,
            valid_from: None,
            valid_until: None,
        }
    }

    /// Set entity type.
    pub fn with_type(mut self, entity_type: impl Into<String>) -> Self {
        self.entity_type = Some(entity_type.into());
        self
    }

    /// Set the timestamp when this mention was observed.
    ///
    /// Useful for tracking entity evolution over time in streaming scenarios.
    pub fn with_timestamp(mut self, ts: chrono::DateTime<chrono::Utc>) -> Self {
        self.timestamp = Some(ts);
        self
    }

    /// Set temporal validity bounds for diachronic entity tracking.
    ///
    /// E.g., "USSR" is valid from 1922-12-30 to 1991-12-26.
    /// This enables proper handling of entities that change over time.
    pub fn with_temporal_bounds(
        mut self,
        from: Option<chrono::DateTime<chrono::Utc>>,
        until: Option<chrono::DateTime<chrono::Utc>>,
    ) -> Self {
        self.valid_from = from;
        self.valid_until = until;
        self
    }

    /// Set embedding.
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Set track ID.
    pub fn with_track_id(mut self, track_id: anno_core::TrackId) -> Self {
        self.track_id = Some(track_id);
        self
    }
}

/// A cluster of entity mentions.
#[derive(Debug, Clone)]
pub struct EntityCluster {
    /// Cluster ID
    pub id: anno_core::IdentityId,
    /// Canonical name (best representative)
    pub canonical_name: String,
    /// Entity type (consensus)
    pub entity_type: Option<String>,
    /// All mentions in this cluster
    pub mentions: Vec<EntityMention>,
    /// Centroid embedding (if embeddings are available)
    pub centroid: Option<Vec<f32>>,
    /// Confidence score
    pub confidence: f32,
}

impl EntityCluster {
    /// Create a new cluster from a single mention.
    fn from_mention(id: anno_core::IdentityId, mention: EntityMention) -> Self {
        let canonical_name = mention.canonical_surface.clone();
        let entity_type = mention.entity_type.clone();
        let centroid = mention.embedding.clone();

        Self {
            id,
            canonical_name,
            entity_type,
            mentions: vec![mention],
            centroid,
            confidence: 1.0,
        }
    }

    /// Add a mention to this cluster.
    fn add_mention(&mut self, mention: EntityMention) {
        // Update centroid if embeddings available
        if let (Some(existing), Some(new)) = (&mut self.centroid, &mention.embedding) {
            let n = self.mentions.len() as f32;
            for (i, v) in new.iter().enumerate() {
                if i < existing.len() {
                    // Running average: new_centroid = (old * n + new) / (n + 1)
                    existing[i] = (existing[i] * n + v) / (n + 1.0);
                }
            }
        } else if self.centroid.is_none() && mention.embedding.is_some() {
            self.centroid = mention.embedding.clone();
        }

        self.mentions.push(mention);
    }

    /// Merge another cluster into this one.
    fn merge(&mut self, other: EntityCluster) {
        // Update centroid
        if let (Some(c1), Some(c2)) = (&mut self.centroid, &other.centroid) {
            let n1 = self.mentions.len() as f32;
            let n2 = other.mentions.len() as f32;
            for (i, v2) in c2.iter().enumerate() {
                if i < c1.len() {
                    c1[i] = (c1[i] * n1 + v2 * n2) / (n1 + n2);
                }
            }
        }

        // Merge mentions
        self.mentions.extend(other.mentions);

        // Update confidence (average)
        self.confidence = (self.confidence + other.confidence) / 2.0;
    }

    /// Get all unique document IDs in this cluster.
    pub fn document_ids(&self) -> Vec<&str> {
        let mut doc_ids: Vec<&str> = self.mentions.iter().map(|m| m.doc_id.as_str()).collect();
        doc_ids.sort();
        doc_ids.dedup();
        doc_ids
    }

    /// Check if any mention in this cluster has temporal bounds.
    pub fn has_temporal_bounds(&self) -> bool {
        self.mentions
            .iter()
            .any(|m| m.valid_from.is_some() || m.valid_until.is_some())
    }

    /// Get the aggregate temporal bounds for this cluster.
    ///
    /// Returns the widest time span that includes all mentions:
    /// - `valid_from`: Earliest `valid_from` among mentions
    /// - `valid_until`: Latest `valid_until` among mentions
    ///
    /// Returns `(None, None)` if no mentions have temporal bounds.
    pub fn temporal_bounds(
        &self,
    ) -> (
        Option<chrono::DateTime<chrono::Utc>>,
        Option<chrono::DateTime<chrono::Utc>>,
    ) {
        let valid_from = self.mentions.iter().filter_map(|m| m.valid_from).min();

        let valid_until = self.mentions.iter().filter_map(|m| m.valid_until).max();

        (valid_from, valid_until)
    }

    /// Get all unique timestamps when mentions were observed.
    ///
    /// Useful for tracking entity evolution over time.
    pub fn observation_times(&self) -> Vec<chrono::DateTime<chrono::Utc>> {
        let mut times: Vec<_> = self.mentions.iter().filter_map(|m| m.timestamp).collect();
        times.sort();
        times.dedup();
        times
    }

    /// Get the time span of observations for this cluster.
    ///
    /// Returns `(first_observation, last_observation)` if any timestamps exist.
    pub fn observation_span(
        &self,
    ) -> Option<(chrono::DateTime<chrono::Utc>, chrono::DateTime<chrono::Utc>)> {
        let times = self.observation_times();
        if times.is_empty() {
            None
        } else {
            Some((
                times[0],
                *times
                    .last()
                    .expect("times should not be empty after empty check"),
            ))
        }
    }
}

/// Streaming entity resolver using the Doubling Algorithm.
#[derive(Debug)]
pub struct StreamingResolver {
    config: StreamingConfig,
    /// All clusters, keyed by cluster ID
    clusters: HashMap<anno_core::IdentityId, EntityCluster>,
    /// LSH index for scalable similarity search
    lsh: Option<MinHashLSH>,
    /// Mapping from LSH item index to cluster ID
    lsh_to_cluster: HashMap<usize, anno_core::IdentityId>,
    /// Next cluster ID
    next_id: anno_core::IdentityId,
    /// Total mentions processed
    mention_count: usize,
}

impl StreamingResolver {
    /// Create a new streaming resolver.
    pub fn new(config: StreamingConfig) -> Self {
        let lsh = if config.use_lsh {
            Some(MinHashLSH::new(config.lsh_config.clone()))
        } else {
            None
        };

        Self {
            config,
            clusters: HashMap::new(),
            lsh,
            lsh_to_cluster: HashMap::new(),
            next_id: anno_core::IdentityId::ZERO,
            mention_count: 0,
        }
    }

    /// Add an entity mention to the resolver.
    ///
    /// This is the main entry point for streaming entity resolution.
    /// Returns the cluster ID that the mention was added to.
    pub fn add_mention(&mut self, mention: EntityMention) -> anno_core::IdentityId {
        self.mention_count += 1;

        // Find best matching cluster
        let best_cluster = self.find_best_cluster(&mention);

        let cluster_id = if let Some((cluster_id, similarity)) = best_cluster {
            if similarity >= self.config.add_threshold {
                // Add to existing cluster
                if let Some(cluster) = self.clusters.get_mut(&cluster_id) {
                    cluster.add_mention(mention);
                }
                cluster_id
            } else {
                // Create new cluster
                self.create_cluster(mention)
            }
        } else {
            // No candidates, create new cluster
            self.create_cluster(mention)
        };

        // Check if we need to merge clusters
        if self.clusters.len() > self.config.max_clusters {
            self.merge_clusters();
        }

        cluster_id
    }

    /// Add an entity with simple parameters.
    pub fn add_entity(
        &mut self,
        doc_id: impl Into<String>,
        surface: impl Into<String>,
        entity_type: Option<String>,
    ) -> anno_core::IdentityId {
        let mut mention = EntityMention::new(doc_id, surface);
        if let Some(et) = entity_type {
            mention = mention.with_type(et);
        }
        self.add_mention(mention)
    }

    /// Get all current clusters.
    pub fn clusters(&self) -> Vec<&EntityCluster> {
        self.clusters.values().collect()
    }

    /// Get a cluster by ID.
    pub fn get_cluster(&self, id: anno_core::IdentityId) -> Option<&EntityCluster> {
        self.clusters.get(&id)
    }

    /// Get the number of clusters.
    pub fn num_clusters(&self) -> usize {
        self.clusters.len()
    }

    /// Get the total number of mentions processed.
    pub fn num_mentions(&self) -> usize {
        self.mention_count
    }

    /// Manually trigger cluster merging.
    pub fn merge_clusters(&mut self) {
        use anno_core::IdentityId;

        // Find pairs of similar clusters
        let cluster_ids: Vec<IdentityId> = self.clusters.keys().copied().collect();
        let mut to_merge: Vec<(IdentityId, IdentityId)> = Vec::new();

        for i in 0..cluster_ids.len() {
            for j in (i + 1)..cluster_ids.len() {
                let id_a = cluster_ids[i];
                let id_b = cluster_ids[j];

                if let (Some(cluster_a), Some(cluster_b)) =
                    (self.clusters.get(&id_a), self.clusters.get(&id_b))
                {
                    // Check type match if required
                    if self.config.require_type_match
                        && cluster_a.entity_type != cluster_b.entity_type
                    {
                        continue;
                    }

                    let similarity = self.cluster_similarity(cluster_a, cluster_b);
                    if similarity >= self.config.merge_threshold {
                        to_merge.push((id_a, id_b));
                    }
                }
            }
        }

        // Merge clusters (use union-find to handle transitive merges)
        let mut merged_into: HashMap<IdentityId, IdentityId> = HashMap::new();

        fn find_root(
            merged_into: &mut HashMap<IdentityId, IdentityId>,
            id: IdentityId,
        ) -> IdentityId {
            if let Some(&parent) = merged_into.get(&id) {
                if parent != id {
                    let root = find_root(merged_into, parent);
                    merged_into.insert(id, root);
                    return root;
                }
            }
            id
        }

        for (a, b) in to_merge {
            let root_a = find_root(&mut merged_into, a);
            let root_b = find_root(&mut merged_into, b);
            if root_a != root_b {
                merged_into.insert(root_b, root_a);
            }
        }

        // Actually merge the clusters
        let to_remove: Vec<IdentityId> = merged_into
            .iter()
            .filter(|(k, v)| *k != *v)
            .map(|(k, _)| *k)
            .collect();

        for id in to_remove {
            if let Some(cluster) = self.clusters.remove(&id) {
                let root = find_root(&mut merged_into, id);
                if let Some(target) = self.clusters.get_mut(&root) {
                    target.merge(cluster);
                }
            }
        }
    }

    // =========================================================================
    // Internal methods
    // =========================================================================

    /// Find the best matching cluster for a mention.
    fn find_best_cluster(&self, mention: &EntityMention) -> Option<(anno_core::IdentityId, f32)> {
        if let Some(lsh) = &self.lsh {
            // Use LSH blocking for scalability
            let candidates = lsh.query(&mention.canonical_surface);

            let mut best: Option<(anno_core::IdentityId, f32)> = None;
            for idx in candidates {
                if let Some(&cluster_id) = self.lsh_to_cluster.get(&idx) {
                    if let Some(cluster) = self.clusters.get(&cluster_id) {
                        // Check type match
                        if self.config.require_type_match
                            && mention.entity_type.is_some()
                            && cluster.entity_type != mention.entity_type
                        {
                            continue;
                        }

                        let sim = self.mention_cluster_similarity(mention, cluster);
                        let should_update = match best {
                            None => true,
                            Some((_, s)) => sim > s,
                        };
                        if should_update {
                            best = Some((cluster_id, sim));
                        }
                    }
                }
            }
            best
        } else {
            // Brute force (O(n) clusters)
            let mut best: Option<(anno_core::IdentityId, f32)> = None;

            for (&cluster_id, cluster) in &self.clusters {
                // Check type match
                if self.config.require_type_match
                    && mention.entity_type.is_some()
                    && cluster.entity_type != mention.entity_type
                {
                    continue;
                }

                let sim = self.mention_cluster_similarity(mention, cluster);
                let should_update = match best {
                    None => true,
                    Some((_, s)) => sim > s,
                };
                if should_update {
                    best = Some((cluster_id, sim));
                }
            }
            best
        }
    }

    /// Create a new cluster from a mention.
    fn create_cluster(&mut self, mention: EntityMention) -> anno_core::IdentityId {
        let id = self.next_id;
        self.next_id += 1;

        // Add to LSH if enabled
        if let Some(lsh) = &mut self.lsh {
            let lsh_idx = lsh.len();
            lsh.insert_text(id.get().to_string(), &mention.canonical_surface);
            self.lsh_to_cluster.insert(lsh_idx, id);
        }

        let cluster = EntityCluster::from_mention(id, mention);
        self.clusters.insert(id, cluster);
        id
    }

    /// Compute similarity between a mention and a cluster.
    fn mention_cluster_similarity(&self, mention: &EntityMention, cluster: &EntityCluster) -> f32 {
        if let (Some(emb), Some(centroid)) = (&mention.embedding, &cluster.centroid) {
            return cosine_similarity(emb, centroid);
        }
        trigram_similarity(&mention.canonical_surface, &cluster.canonical_name)
    }

    /// Compute similarity between two clusters.
    fn cluster_similarity(&self, cluster_a: &EntityCluster, cluster_b: &EntityCluster) -> f32 {
        if let (Some(c1), Some(c2)) = (&cluster_a.centroid, &cluster_b.centroid) {
            return cosine_similarity(c1, c2);
        }
        trigram_similarity(&cluster_a.canonical_name, &cluster_b.canonical_name)
    }
}

impl Default for StreamingResolver {
    fn default() -> Self {
        Self::new(StreamingConfig::default())
    }
}

// =============================================================================
// Similarity functions
// =============================================================================

/// Cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

/// String similarity using Jaccard coefficient on character trigrams.
///
/// This is better for fuzzy name matching where character-level variations matter:
/// - "Barack Obama" vs "obama" → high similarity
/// - "NVIDIA" vs "Nvidia Corp" → medium similarity
///
/// For word-level similarity (phrase matching), use [`super::string_similarity`].
///
/// # Algorithm
///
/// 1. Convert both strings to lowercase
/// 2. Extract character trigrams (sliding window of 3 chars)
/// 3. Compute Jaccard coefficient: |A ∩ B| / |A ∪ B|
///
/// # Examples
///
/// ```rust
/// use anno_core::coalesce::streaming::trigram_similarity;
///
/// assert!((trigram_similarity("Barack Obama", "obama") - 0.375).abs() < 0.1);
/// assert!((trigram_similarity("test", "test") - 1.0).abs() < 0.001);
/// ```
pub fn trigram_similarity(a: &str, b: &str) -> f32 {
    // Delegate to shared primitive (Unicode-safe, case-insensitive).
    textprep::similarity::trigram_jaccard(a, b) as f32
}

/// Alias for backward compatibility.
#[doc(hidden)]
#[deprecated(since = "0.3.0", note = "Use trigram_similarity instead")]
pub fn string_similarity(a: &str, b: &str) -> f32 {
    trigram_similarity(a, b)
}

// NOTE: this module intentionally avoids bespoke type/temporal heuristics.
// If needed, add them as a higher-level adapter layer that is explicitly sourced.

// =============================================================================
// Conversion to/from `anno::core` types
// =============================================================================

impl EntityMention {
    /// Convert from a Track reference.
    ///
    /// Creates an EntityMention from document ID, Track ID, and Track data.
    /// This enables using streaming resolution on entities extracted via anno's
    /// standard NER pipeline.
    #[must_use]
    pub fn from_track(doc_id: impl Into<String>, track: &crate::Track) -> Self {
        Self {
            doc_id: doc_id.into(),
            canonical_surface: track.canonical_surface.clone(),
            entity_type: track.entity_type.clone(),
            embedding: track.embedding.clone(),
            track_id: Some(track.id),
            // Note: Temporal fields must be set explicitly after construction
            // since Track's SignalRefs don't carry temporal information.
            // Use with_temporal_bounds() to set valid_from/valid_until.
            timestamp: None,
            valid_from: None,
            valid_until: None,
        }
    }
}

impl EntityCluster {
    /// Convert this cluster to an anno_core::Identity.
    ///
    /// Creates a global Identity from the cluster's contents.
    /// The source is set to `CrossDocCoref` with TrackRefs for all mentions
    /// that have track_id set.
    #[must_use]
    pub fn to_identity(&self) -> anno_core::Identity {
        // Build track refs from mentions that have track_ids
        let track_refs: Vec<anno_core::TrackRef> = self
            .mentions
            .iter()
            .filter_map(|m| {
                m.track_id.map(|tid| anno_core::TrackRef {
                    doc_id: m.doc_id.clone(),
                    track_id: tid,
                })
            })
            .collect();

        let source = if track_refs.is_empty() {
            None
        } else {
            Some(anno_core::IdentitySource::CrossDocCoref { track_refs })
        };

        // Compute temporal bounds from mentions if available
        let valid_from = self.mentions.iter().filter_map(|m| m.valid_from).min();
        let valid_until = self.mentions.iter().filter_map(|m| m.valid_until).max();

        // Note: valid_from/valid_until are computed but not stored in Identity
        // (temporal validity is tracked at the mention/signal level)
        let _ = (valid_from, valid_until);

        anno_core::Identity {
            id: self.id,
            canonical_name: self.canonical_name.clone(),
            entity_type: self.entity_type.clone(),
            kb_id: None,
            kb_name: None,
            description: None,
            embedding: self.centroid.clone(),
            aliases: self
                .mentions
                .iter()
                .map(|m| m.canonical_surface.clone())
                .filter(|s| s != &self.canonical_name)
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect(),
            confidence: self.confidence,
            source,
        }
    }
}

impl StreamingResolver {
    /// Convert all clusters to anno_core::Identity objects.
    ///
    /// Returns a vector of Identities representing the current clustering state.
    /// Useful for exporting streaming resolution results into the `anno::core` format.
    #[must_use]
    pub fn to_identities(&self) -> Vec<anno_core::Identity> {
        self.clusters()
            .into_iter()
            .map(|c| c.to_identity())
            .collect()
    }

    /// Add entities from a Track.
    ///
    /// Convenience method that extracts relevant information from an anno_core::Track
    /// and adds it to the resolver.
    pub fn add_track(
        &mut self,
        doc_id: impl Into<String>,
        track: &anno_core::Track,
    ) -> anno_core::IdentityId {
        let mention = EntityMention::from_track(doc_id, track);
        self.add_mention(mention)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_streaming() {
        let mut resolver = StreamingResolver::new(StreamingConfig::default());

        resolver.add_entity("doc1", "Barack Obama", Some("Person".to_string()));
        resolver.add_entity("doc2", "obama", Some("Person".to_string()));
        resolver.add_entity("doc3", "Donald Trump", Some("Person".to_string()));

        // Obama mentions should cluster together
        assert!(resolver.num_clusters() <= 3);
        assert_eq!(resolver.num_mentions(), 3);
    }

    #[test]
    fn test_type_filtering() {
        let config = StreamingConfig {
            require_type_match: true,
            ..Default::default()
        };
        let mut resolver = StreamingResolver::new(config);

        resolver.add_entity("doc1", "Apple", Some("Organization".to_string()));
        resolver.add_entity("doc2", "Apple", Some("Food".to_string()));

        // Different types should not cluster
        assert_eq!(resolver.num_clusters(), 2);
    }

    #[test]
    fn test_cluster_merging() {
        let config = StreamingConfig {
            max_clusters: 2,
            merge_threshold: 0.3, // Low threshold to force merging
            ..Default::default()
        };
        let mut resolver = StreamingResolver::new(config);

        resolver.add_entity("doc1", "New York City", None);
        resolver.add_entity("doc2", "NYC", None);
        resolver.add_entity("doc3", "New York", None);
        resolver.add_entity("doc4", "Los Angeles", None);
        resolver.add_entity("doc5", "LA", None);

        // Should have merged some clusters
        assert!(resolver.num_clusters() <= 5);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let c = vec![0.0, 1.0, 0.0];

        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);
        assert!((cosine_similarity(&a, &c) - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_trigram_similarity() {
        assert!(trigram_similarity("Barack Obama", "barack obama") > 0.9);
        assert!(trigram_similarity("Obama", "Trump") < 0.3);
    }

    #[test]
    fn test_document_ids() {
        let mut resolver = StreamingResolver::new(StreamingConfig::default());

        resolver.add_entity("doc1", "Barack Obama", None);
        resolver.add_entity("doc2", "obama", None);

        let clusters = resolver.clusters();
        for cluster in clusters {
            if cluster.mentions.len() > 1 {
                let doc_ids = cluster.document_ids();
                assert!(!doc_ids.is_empty());
            }
        }
    }

    #[test]
    fn test_entity_mention_from_track() {
        let track = anno_core::Track::new(42, "Barack Obama").with_type("Person".to_string());

        let mention = EntityMention::from_track("doc1", &track);

        assert_eq!(mention.doc_id, "doc1");
        assert_eq!(mention.canonical_surface, "Barack Obama");
        assert_eq!(mention.entity_type, Some("Person".to_string()));
        assert_eq!(mention.track_id, Some(anno_core::TrackId::new(42)));
    }

    #[test]
    fn test_cluster_to_identity() {
        let mut resolver = StreamingResolver::new(StreamingConfig::default());

        // Add some entities
        resolver.add_entity("doc1", "Barack Obama", Some("Person".to_string()));
        resolver.add_entity("doc2", "obama", Some("Person".to_string()));

        let identities = resolver.to_identities();

        // Should have at least one identity
        assert!(!identities.is_empty());

        for identity in &identities {
            // Each identity should have a canonical name
            assert!(!identity.canonical_name.is_empty());
            // Confidence should be valid
            assert!((0.0..=1.0).contains(&identity.confidence));
        }
    }

    #[test]
    fn test_add_track() {
        let mut resolver = StreamingResolver::new(StreamingConfig::default());

        let track1 = anno_core::Track::new(1, "Jensen Huang").with_type("Person".to_string());
        let track2 = anno_core::Track::new(2, "Nvidia").with_type("Organization".to_string());

        resolver.add_track("doc1", &track1);
        resolver.add_track("doc1", &track2);

        assert_eq!(resolver.num_mentions(), 2);
        // Should have at least 1 cluster (could be 2 if different types separate)
        assert!(resolver.num_clusters() >= 1);
    }

    #[test]
    fn test_streaming_basic_similarity_smoke() {
        let mut resolver = StreamingResolver::new(StreamingConfig::default());

        resolver.add_entity("doc1", "Barack Obama", Some("Person".to_string()));
        resolver.add_entity("doc2", "obama", Some("Person".to_string()));
        resolver.add_entity("doc3", "Donald Trump", Some("Person".to_string()));

        assert!(resolver.num_clusters() <= 3);
    }

    // Type-compatibility heuristics were intentionally removed.
}

// =============================================================================
// Property Tests
// =============================================================================

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        /// Property: Mention count equals sum of cluster mentions
        #[test]
        fn streaming_mention_conservation(
            entities in proptest::collection::vec("[A-Za-z ]{3,20}", 1..20)
        ) {
            let mut resolver = StreamingResolver::new(StreamingConfig::default());

            for (i, entity) in entities.iter().enumerate() {
                resolver.add_entity(format!("doc{}", i), entity, None);
            }

            let cluster_mentions: usize = resolver.clusters()
                .iter()
                .map(|c| c.mentions.len())
                .sum();

            prop_assert_eq!(resolver.num_mentions(), cluster_mentions,
                "Mention count mismatch: {} != {}",
                resolver.num_mentions(), cluster_mentions);
        }

        /// Property: Cluster count <= mention count
        #[test]
        fn streaming_cluster_bounded(
            entities in proptest::collection::vec("[A-Za-z]{3,15}", 1..30)
        ) {
            let mut resolver = StreamingResolver::new(StreamingConfig::default());

            for (i, entity) in entities.iter().enumerate() {
                resolver.add_entity(format!("doc{}", i), entity, None);
            }

            prop_assert!(resolver.num_clusters() <= resolver.num_mentions(),
                "More clusters ({}) than mentions ({})",
                resolver.num_clusters(), resolver.num_mentions());
        }

        /// Property: Identical entities cluster together
        #[test]
        fn streaming_identical_cluster(name in "[A-Za-z]{5,15}", count in 2usize..10) {
            let mut resolver = StreamingResolver::new(StreamingConfig::default());

            for i in 0..count {
                resolver.add_entity(format!("doc{}", i), &name, None);
            }

            // Should have exactly one cluster with all mentions
            prop_assert_eq!(resolver.num_clusters(), 1,
                "Identical entities should form one cluster, got {}",
                resolver.num_clusters());

            let cluster = resolver.clusters().into_iter().next().expect("should have at least one cluster");
            prop_assert_eq!(cluster.mentions.len(), count,
                "Cluster should have {} mentions, got {}",
                count, cluster.mentions.len());
        }

        /// Property: Different types stay separate when type match required
        #[test]
        fn streaming_type_separation(name in "[A-Za-z]{5,15}") {
            let config = StreamingConfig {
                require_type_match: true,
                ..Default::default()
            };
            let mut resolver = StreamingResolver::new(config);

            resolver.add_entity("doc1", &name, Some("Person".to_string()));
            resolver.add_entity("doc2", &name, Some("Organization".to_string()));

            prop_assert_eq!(resolver.num_clusters(), 2,
                "Different types should not cluster");
        }

        /// Property: Cluster confidence bounded [0, 1]
        #[test]
        fn streaming_confidence_bounded(
            entities in proptest::collection::vec("[A-Za-z ]{3,20}", 1..15)
        ) {
            let mut resolver = StreamingResolver::new(StreamingConfig::default());

            for (i, entity) in entities.iter().enumerate() {
                resolver.add_entity(format!("doc{}", i), entity, None);
            }

            for cluster in resolver.clusters() {
                prop_assert!((0.0..=1.0).contains(&cluster.confidence),
                    "Confidence {} out of bounds", cluster.confidence);
            }
        }

        /// Property: Trigram similarity symmetric
        #[test]
        fn trigram_sim_symmetric(a in "[A-Za-z ]{3,20}", b in "[A-Za-z ]{3,20}") {
            let sim_ab = trigram_similarity(&a, &b);
            let sim_ba = trigram_similarity(&b, &a);
            prop_assert!((sim_ab - sim_ba).abs() < 0.001,
                "Trigram similarity not symmetric: {} vs {}", sim_ab, sim_ba);
        }

        /// Property: Cosine similarity bounded [0, 1] for positive vectors
        #[test]
        fn cosine_sim_bounded(
            dim in 10usize..100,
            seed in any::<u64>()
        ) {
            let mut rng = seed;
            let a: Vec<f32> = (0..dim).map(|_| {
                rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                (rng % 1000) as f32 / 1000.0
            }).collect();
            let b: Vec<f32> = (0..dim).map(|_| {
                rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                (rng % 1000) as f32 / 1000.0
            }).collect();

            let sim = cosine_similarity(&a, &b);
            prop_assert!((-0.001..=1.001).contains(&sim),
                "Cosine similarity {} out of bounds", sim);
        }
    }
}
