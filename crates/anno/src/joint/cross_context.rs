//! Cross-context extension for joint entity analysis.
//!
//! This module bridges the within-document JointModel with the
//! cross-context cluster merging from xCoRe (Martinelli et al., 2025).
//!
//! # Architecture
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────────┐
//! │ Within-Context (JointModel)                                        │
//! │   For each context c_i:                                            │
//! │   1. Extract mentions (NER)                                        │
//! │   2. Run belief propagation → typed entities + coref chains        │
//! │   3. Extract local clusters                                        │
//! ├────────────────────────────────────────────────────────────────────┤
//! │ Cross-Context (ClusterEncoder + MergeScorer)                       │
//! │   4. Encode local clusters → cluster embeddings                    │
//! │   5. Score pairwise merges across contexts                         │
//! │   6. Merge clusters using union-find                               │
//! └────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # References
//!
//! - Durrett & Klein (2014): "A Joint Model for Entity Analysis"
//! - Martinelli et al. (2025): "xCoRe: Cross-context Coreference Resolution"

use crate::eval::cluster_encoder::{
    ClusterEmbedding, ClusterEncoder, ClusterMention, CrossContextConfig as EncoderConfig,
    LocalCluster, MergeScorer, MergedCluster,
};
use crate::{Entity, Result};
use std::collections::HashMap;

use super::types::{JointConfig, JointModel, JointResult};

// =============================================================================
// Cross-Context Configuration
// =============================================================================

/// Configuration for cross-context joint analysis.
#[derive(Debug, Clone)]
pub struct CrossContextJointConfig {
    /// Within-context joint model config
    pub joint_config: JointConfig,
    /// Cross-context encoder config
    pub encoder_config: EncoderConfig,
    /// Whether to run entity linking across contexts
    pub cross_context_linking: bool,
    /// Whether to propagate types across merged clusters
    pub propagate_types: bool,
}

impl Default for CrossContextJointConfig {
    fn default() -> Self {
        Self {
            joint_config: JointConfig::default(),
            encoder_config: EncoderConfig::default(),
            cross_context_linking: true,
            propagate_types: true,
        }
    }
}

// =============================================================================
// Context Representation
// =============================================================================

/// A context for cross-context processing.
///
/// A context can be:
/// - A single short document
/// - A window of a long document
/// - One document in a cross-document corpus
#[derive(Debug, Clone)]
pub struct Context {
    /// Unique context identifier
    pub id: usize,
    /// Context text
    pub text: String,
    /// Pre-extracted entities (from NER)
    pub entities: Vec<Entity>,
    /// Optional document ID (for cross-document)
    pub doc_id: Option<String>,
    /// Window index if this is part of a windowed document
    pub window_idx: Option<usize>,
}

impl Context {
    /// Create a new context from text.
    pub fn new(id: usize, text: impl Into<String>) -> Self {
        Self {
            id,
            text: text.into(),
            entities: Vec::new(),
            doc_id: None,
            window_idx: None,
        }
    }

    /// Add pre-extracted entities.
    pub fn with_entities(mut self, entities: Vec<Entity>) -> Self {
        self.entities = entities;
        self
    }

    /// Mark as part of a document.
    pub fn with_doc_id(mut self, doc_id: impl Into<String>) -> Self {
        self.doc_id = Some(doc_id.into());
        self
    }

    /// Mark as a window of a longer document.
    pub fn with_window_idx(mut self, idx: usize) -> Self {
        self.window_idx = Some(idx);
        self
    }
}

// =============================================================================
// Cross-Context Result
// =============================================================================

/// Result of cross-context joint analysis.
#[derive(Debug, Clone)]
pub struct CrossContextResult {
    /// Per-context results from JointModel
    pub context_results: Vec<JointResult>,
    /// Merged cross-context clusters
    pub merged_clusters: Vec<MergedCluster>,
    /// Global entity index (unified across contexts)
    pub global_entities: Vec<GlobalEntity>,
    /// Cross-context coreference chains
    pub global_chains: Vec<GlobalCorefChain>,
}

/// An entity with global (cross-context) identity.
#[derive(Debug, Clone)]
pub struct GlobalEntity {
    /// Original context ID
    pub context_id: usize,
    /// Original entity in context
    pub entity: Entity,
    /// Global cluster ID (assigned after merging)
    pub global_cluster_id: Option<usize>,
    /// Canonical form from merged cluster
    pub canonical: Option<String>,
}

/// A coreference chain spanning multiple contexts.
#[derive(Debug, Clone)]
pub struct GlobalCorefChain {
    /// Unique chain ID
    pub id: usize,
    /// Mentions from all contexts: (context_id, mention)
    pub mentions: Vec<(usize, ClusterMention)>,
    /// Canonical representative
    pub canonical: Option<String>,
    /// Source contexts
    pub source_contexts: Vec<usize>,
}

// =============================================================================
// Cross-Context Joint Model
// =============================================================================

/// Joint model extended for cross-context coreference.
///
/// Combines Durrett & Klein (2014) within-context joint inference
/// with xCoRe (2025) cross-context cluster merging.
pub struct CrossContextJointModel<E: ClusterEncoder, S: MergeScorer> {
    /// Within-context joint model
    joint_model: JointModel,
    /// Cluster encoder for cross-context merging
    encoder: E,
    /// Merge scorer for cluster pairs
    scorer: S,
    /// Configuration
    config: CrossContextJointConfig,
}

impl<E: ClusterEncoder, S: MergeScorer> CrossContextJointModel<E, S> {
    /// Create a new cross-context joint model.
    pub fn new(encoder: E, scorer: S, config: CrossContextJointConfig) -> Result<Self> {
        let joint_model = JointModel::new(config.joint_config.clone())?;
        Ok(Self {
            joint_model,
            encoder,
            scorer,
            config,
        })
    }

    /// Analyze multiple contexts jointly.
    ///
    /// # Pipeline
    ///
    /// 1. For each context: run within-context joint inference
    /// 2. Extract local clusters from each context
    /// 3. Encode clusters using ClusterEncoder
    /// 4. Score pairwise merges using MergeScorer
    /// 5. Merge clusters using union-find
    /// 6. Build global entities and chains
    pub fn analyze(&self, contexts: &[Context]) -> Result<CrossContextResult> {
        // Step 1: Within-context joint analysis
        let context_results: Vec<JointResult> = contexts
            .iter()
            .map(|ctx| self.joint_model.analyze(&ctx.text, &ctx.entities))
            .collect::<Result<Vec<_>>>()?;

        // Step 2: Extract local clusters
        let local_clusters = self.extract_local_clusters(contexts, &context_results);

        // Step 3 & 4: Encode clusters and score merges
        let merged_clusters = self.merge_clusters(&local_clusters);

        // Step 5: Build global entities and chains
        let (global_entities, global_chains) =
            self.build_global_results(contexts, &context_results, &merged_clusters);

        Ok(CrossContextResult {
            context_results,
            merged_clusters,
            global_entities,
            global_chains,
        })
    }

    /// Extract local clusters from joint analysis results.
    fn extract_local_clusters(
        &self,
        contexts: &[Context],
        results: &[JointResult],
    ) -> HashMap<usize, Vec<LocalCluster>> {
        let mut all_clusters = HashMap::new();

        for (ctx_idx, (_ctx, result)) in contexts.iter().zip(results.iter()).enumerate() {
            let mut ctx_clusters = Vec::new();

            // Each coref chain becomes a local cluster
            for (chain_idx, chain) in result.chains.iter().enumerate() {
                let mut cluster = LocalCluster::new(chain_idx, ctx_idx);

                for mention in &chain.mentions {
                    cluster.add_mention(ClusterMention {
                        start: mention.start,
                        end: mention.end,
                        text: mention.text.clone(),
                        context_id: ctx_idx,
                    });
                }

                cluster.compute_canonical();
                ctx_clusters.push(cluster);
            }

            // Also add singletons (entities not in any chain)
            let chained_starts: std::collections::HashSet<usize> = result
                .chains
                .iter()
                .flat_map(|c| c.mentions.iter().map(|m| m.start))
                .collect();

            for entity in result.entities.iter() {
                if !chained_starts.contains(&entity.start) {
                    let mut cluster = LocalCluster::new(ctx_clusters.len(), ctx_idx);
                    cluster.add_mention(ClusterMention {
                        start: entity.start,
                        end: entity.end,
                        text: entity.text.clone(),
                        context_id: ctx_idx,
                    });
                    cluster.compute_canonical();
                    ctx_clusters.push(cluster);
                }
            }

            all_clusters.insert(ctx_idx, ctx_clusters);
        }

        all_clusters
    }

    /// Merge clusters across contexts.
    fn merge_clusters(
        &self,
        local_clusters: &HashMap<usize, Vec<LocalCluster>>,
    ) -> Vec<MergedCluster> {
        // Encode all clusters
        let mut all_embeddings: Vec<ClusterEmbedding> = Vec::new();
        for clusters in local_clusters.values() {
            for cluster in clusters {
                let embedding = self.encoder.encode_cluster(cluster, None);
                all_embeddings.push(embedding);
            }
        }

        if all_embeddings.is_empty() {
            return Vec::new();
        }

        // Score pairwise merges
        let mut merge_decisions: Vec<(usize, usize, f32)> = Vec::new();
        for (i, emb_a) in all_embeddings.iter().enumerate() {
            for (j, emb_b) in all_embeddings.iter().enumerate().skip(i + 1) {
                // Skip same-context comparisons
                if emb_a.context_id == emb_b.context_id {
                    continue;
                }

                let score = self.scorer.score(emb_a, emb_b);
                if score >= self.config.encoder_config.threshold {
                    merge_decisions.push((i, j, score));
                }
            }
        }

        // Sort by score (best first)
        merge_decisions.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        // Union-find for merging
        let n = all_embeddings.len();
        let mut parent: Vec<usize> = (0..n).collect();
        let mut rank: Vec<usize> = vec![0; n];

        fn find(parent: &mut [usize], i: usize) -> usize {
            if parent[i] != i {
                parent[i] = find(parent, parent[i]);
            }
            parent[i]
        }

        fn union(parent: &mut [usize], rank: &mut [usize], x: usize, y: usize) {
            let px = find(parent, x);
            let py = find(parent, y);
            if px == py {
                return;
            }
            match rank[px].cmp(&rank[py]) {
                std::cmp::Ordering::Less => parent[px] = py,
                std::cmp::Ordering::Greater => parent[py] = px,
                std::cmp::Ordering::Equal => {
                    parent[py] = px;
                    rank[px] += 1;
                }
            }
        }

        for (i, j, _score) in merge_decisions {
            union(&mut parent, &mut rank, i, j);
        }

        // Group by root
        let mut merged_map: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..n {
            let root = find(&mut parent, i);
            merged_map.entry(root).or_default().push(i);
        }

        // Build MergedCluster results
        merged_map
            .into_iter()
            .enumerate()
            .map(|(merged_id, (_, indices))| {
                let mut merged = MergedCluster {
                    id: merged_id,
                    source_clusters: Vec::new(),
                    mentions: Vec::new(),
                    canonical: None,
                };

                for idx in indices {
                    let emb = &all_embeddings[idx];
                    merged
                        .source_clusters
                        .push((emb.context_id, emb.cluster_id));

                    // Copy mentions from original cluster
                    if let Some(clusters) = local_clusters.get(&emb.context_id) {
                        if let Some(cluster) = clusters.iter().find(|c| c.id == emb.cluster_id) {
                            merged.mentions.extend(cluster.mentions.clone());
                            if merged.canonical.is_none() {
                                merged.canonical = cluster.canonical.clone();
                            }
                        }
                    }
                }

                merged
            })
            .collect()
    }

    /// Build global entities and chains from merged clusters.
    fn build_global_results(
        &self,
        _contexts: &[Context],
        results: &[JointResult],
        merged_clusters: &[MergedCluster],
    ) -> (Vec<GlobalEntity>, Vec<GlobalCorefChain>) {
        // Build mention-to-cluster mapping
        let mut mention_to_cluster: HashMap<(usize, usize, usize), usize> = HashMap::new();
        for (cluster_idx, cluster) in merged_clusters.iter().enumerate() {
            for mention in &cluster.mentions {
                mention_to_cluster.insert(
                    (mention.context_id, mention.start, mention.end),
                    cluster_idx,
                );
            }
        }

        // Build global entities
        let mut global_entities = Vec::new();
        for (ctx_idx, result) in results.iter().enumerate() {
            for entity in &result.entities {
                let global_cluster_id = mention_to_cluster
                    .get(&(ctx_idx, entity.start, entity.end))
                    .copied();
                let canonical = global_cluster_id
                    .and_then(|cid| merged_clusters.get(cid))
                    .and_then(|c| c.canonical.clone());

                global_entities.push(GlobalEntity {
                    context_id: ctx_idx,
                    entity: entity.clone(),
                    global_cluster_id,
                    canonical,
                });
            }
        }

        // Build global chains
        let global_chains: Vec<GlobalCorefChain> = merged_clusters
            .iter()
            .map(|cluster| {
                let source_contexts: Vec<usize> = cluster
                    .mentions
                    .iter()
                    .map(|m| m.context_id)
                    .collect::<std::collections::HashSet<_>>()
                    .into_iter()
                    .collect();

                GlobalCorefChain {
                    id: cluster.id,
                    mentions: cluster
                        .mentions
                        .iter()
                        .map(|m| (m.context_id, m.clone()))
                        .collect(),
                    canonical: cluster.canonical.clone(),
                    source_contexts,
                }
            })
            .collect();

        (global_entities, global_chains)
    }

    /// Get configuration.
    pub fn config(&self) -> &CrossContextJointConfig {
        &self.config
    }
}

// =============================================================================
// Window Splitter (for long documents)
// =============================================================================

/// Split a long document into overlapping windows.
pub struct WindowSplitter {
    /// Maximum tokens per window
    pub max_tokens: usize,
    /// Token overlap between adjacent windows
    pub overlap: usize,
}

impl Default for WindowSplitter {
    fn default() -> Self {
        Self {
            max_tokens: 4000,
            overlap: 256,
        }
    }
}

impl WindowSplitter {
    /// Split text into contexts (windows).
    pub fn split(&self, doc_id: &str, text: &str, entities: &[Entity]) -> Vec<Context> {
        // Simple word-based tokenization for splitting
        let words: Vec<&str> = text.split_whitespace().collect();

        if words.len() <= self.max_tokens {
            // Single context
            return vec![Context::new(0, text)
                .with_entities(entities.to_vec())
                .with_doc_id(doc_id)];
        }

        let mut contexts = Vec::new();
        let mut start_word = 0;
        let mut window_idx = 0;

        // Precompute word byte spans (start/end) by searching forward through the original text.
        // This is Unicode-safe because spans are computed in byte offsets and only at UTF-8
        // character boundaries (from `find(word)` where `word` is a substring of `text`).
        let mut word_byte_spans: Vec<(usize, usize)> = Vec::with_capacity(words.len());
        let mut search_byte = 0usize;
        for w in &words {
            if let Some(rel) = text.get(search_byte..).and_then(|s| s.find(w)) {
                let start_byte = search_byte + rel;
                let end_byte = start_byte + w.len();
                word_byte_spans.push((start_byte, end_byte));
                search_byte = end_byte;
            } else {
                // If we can't locate the word, fall back to an empty span at the current search position.
                word_byte_spans.push((search_byte, search_byte));
            }
        }

        while start_word < words.len() {
            let end_word = (start_word + self.max_tokens).min(words.len());

            // Compute byte window bounds from the word spans.
            let start_byte = word_byte_spans
                .get(start_word)
                .map(|(s, _)| *s)
                .unwrap_or(0);
            let end_byte = word_byte_spans
                .get(end_word.saturating_sub(1))
                .map(|(_, e)| *e)
                .unwrap_or(start_byte)
                .min(text.len());

            // Convert byte offsets to character offsets for entity filtering/adjustment.
            let span = crate::offset::TextSpan::from_bytes(text, start_byte, end_byte);
            let char_start = span.char_start;
            let char_end = span.char_end;

            let window_text = text.get(start_byte..end_byte).unwrap_or("");

            // Filter entities to this window
            let window_entities: Vec<Entity> = entities
                .iter()
                .filter(|e| e.start >= char_start && e.end <= char_end)
                .map(|e| {
                    let mut adjusted = e.clone();
                    adjusted.start -= char_start;
                    adjusted.end -= char_start;
                    adjusted
                })
                .collect();

            contexts.push(
                Context::new(window_idx, window_text)
                    .with_entities(window_entities)
                    .with_doc_id(doc_id)
                    .with_window_idx(window_idx),
            );

            // Move to next window with overlap
            start_word = end_word.saturating_sub(self.overlap);
            if start_word >= words.len() || end_word >= words.len() {
                break;
            }
            window_idx += 1;
        }

        contexts
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::cluster_encoder::{CosineMergeScorer, HeuristicClusterEncoder};
    use crate::EntityType;

    #[test]
    fn test_context_creation() {
        let ctx = Context::new(0, "Obama visited France.")
            .with_doc_id("doc1")
            .with_window_idx(0);

        assert_eq!(ctx.id, 0);
        assert_eq!(ctx.doc_id, Some("doc1".to_string()));
        assert_eq!(ctx.window_idx, Some(0));
    }

    #[test]
    fn test_window_splitter_short_doc() {
        let splitter = WindowSplitter::default();
        let text = "Short document.";
        let contexts = splitter.split("doc1", text, &[]);

        assert_eq!(contexts.len(), 1);
        assert_eq!(contexts[0].text, text);
    }

    #[test]
    fn test_window_splitter_long_doc() {
        let splitter = WindowSplitter {
            max_tokens: 10,
            overlap: 2,
        };

        // Create a document with more than 10 words
        let words: Vec<&str> = (0..25)
            .map(|i| if i % 2 == 0 { "word" } else { "text" })
            .collect();
        let text = words.join(" ");

        let contexts = splitter.split("doc1", &text, &[]);

        // Should have multiple windows
        assert!(contexts.len() > 1);

        // All windows should have doc_id and window_idx
        for (i, ctx) in contexts.iter().enumerate() {
            assert_eq!(ctx.doc_id, Some("doc1".to_string()));
            assert_eq!(ctx.window_idx, Some(i));
        }
    }

    #[test]
    fn test_window_splitter_unicode_entity_offsets_do_not_panic() {
        let splitter = WindowSplitter {
            max_tokens: 3,
            overlap: 1,
        };
        let text = "Dr. Müller visited 北京 yesterday";
        // Entity offsets are character offsets in anno.
        // "Müller" starts after "Dr. " (4 chars) and is 6 chars.
        let entity = Entity::new("Müller", EntityType::Person, 4, 10, 0.9);
        let contexts = splitter.split("doc1", text, &[entity]);
        assert!(!contexts.is_empty());

        // At least one context should contain the entity in-range and adjusted spans should be valid.
        let mut saw_entity = false;
        for ctx in &contexts {
            for e in &ctx.entities {
                saw_entity = true;
                let char_len = ctx.text.chars().count();
                assert!(e.start <= e.end);
                assert!(e.end <= char_len);
            }
        }
        assert!(
            saw_entity,
            "Expected entity to appear in at least one window"
        );
    }

    #[test]
    fn test_cross_context_model_creation() {
        let encoder = HeuristicClusterEncoder::new(64);
        let scorer = CosineMergeScorer::new(0.5);
        let config = CrossContextJointConfig::default();

        let model = CrossContextJointModel::new(encoder, scorer, config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_cross_context_analyze_empty() {
        let encoder = HeuristicClusterEncoder::new(64);
        let scorer = CosineMergeScorer::new(0.5);
        let config = CrossContextJointConfig::default();

        let model = CrossContextJointModel::new(encoder, scorer, config)
            .expect("model creation should succeed");
        let result = model.analyze(&[]);

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.context_results.is_empty());
        assert!(result.merged_clusters.is_empty());
    }

    #[test]
    fn test_cross_context_analyze_single_context() {
        let encoder = HeuristicClusterEncoder::new(64);
        let scorer = CosineMergeScorer::new(0.5);
        let config = CrossContextJointConfig::default();

        let model = CrossContextJointModel::new(encoder, scorer, config)
            .expect("model creation should succeed");

        let entities = vec![Entity::new("Obama", EntityType::Person, 0, 5, 0.9)];

        let contexts = vec![Context::new(0, "Obama visited France.").with_entities(entities)];

        let result = model.analyze(&contexts);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.context_results.len(), 1);
    }

    #[test]
    fn test_cross_context_analyze_multiple_contexts() {
        let encoder = HeuristicClusterEncoder::new(64);
        let scorer = CosineMergeScorer::new(0.3); // Lower threshold for test
        let config = CrossContextJointConfig::default();

        let model = CrossContextJointModel::new(encoder, scorer, config)
            .expect("model creation should succeed");

        // Two contexts mentioning Obama
        let contexts = vec![
            Context::new(0, "Barack Obama gave a speech.").with_entities(vec![Entity::new(
                "Barack Obama",
                EntityType::Person,
                0,
                12,
                0.9,
            )]),
            Context::new(1, "Obama met with leaders.").with_entities(vec![Entity::new(
                "Obama",
                EntityType::Person,
                0,
                5,
                0.9,
            )]),
        ];

        let result = model.analyze(&contexts);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.context_results.len(), 2);

        // Should have global entities from both contexts
        assert!(!result.global_entities.is_empty());
    }
}
