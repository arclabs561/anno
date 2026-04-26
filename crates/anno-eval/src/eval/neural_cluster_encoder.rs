//! Neural Cluster Encoder for Cross-Context Coreference.
//!
//! Implements the cluster encoding approach from xCoRe (Martinelli et al., 2025)
//! using anno's Candle infrastructure.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                    Neural Cluster Encoder                           │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │ 1. Encode each mention with TextEncoder (DeBERTa/ModernBERT)       │
//! │    └── m_i = Encoder(text_i) → [seq_len, hidden_dim]              │
//! │                                                                     │
//! │ 2. Extract mention representations via mean pooling                │
//! │    └── h_i = mean(m_i[start:end])                                  │
//! │                                                                     │
//! │ 3. Pool mentions with single-layer Transformer (xCoRe style)       │
//! │    └── hs(W_j) = TransformerLayer([h_1, ..., h_k])                 │
//! │                                                                     │
//! │ 4. Output cluster embedding for merge scoring                       │
//! │    └── cluster_emb = mean(hs(W_j)) or [CLS] pooling               │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # References
//!
//! - Martinelli et al. (2025): "xCoRe: Cross-context Coreference Resolution"
//! - Section 3.2.2: Cross-context Cluster Merging

use crate::eval::cluster_encoder::{
    ClusterEmbedding, ClusterEncoder, ClusterMention, LocalCluster, MergeScorer,
};
use std::collections::HashMap;

#[cfg(feature = "candle")]
use {
    anno::backends::encoder_candle::CandleTextEncoder,
    candle_core::{DType, Device, Module, Tensor, D},
    candle_nn::{layer_norm, linear, LayerNorm, Linear, VarBuilder},
};

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for neural cluster encoder.
#[derive(Debug, Clone)]
pub struct NeuralClusterConfig {
    /// Hidden dimension (should match TextEncoder)
    pub hidden_dim: usize,
    /// Number of attention heads for pooling transformer
    pub num_heads: usize,
    /// Dropout probability
    pub dropout: f32,
    /// Use learned \[CLS\] token for pooling
    pub use_cls_pooling: bool,
    /// Maximum mentions per cluster (truncate if exceeded)
    pub max_mentions: usize,
    /// Merge probability threshold
    pub merge_threshold: f32,
}

impl Default for NeuralClusterConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 768, // DeBERTa-base default
            num_heads: 12,
            dropout: 0.1,
            use_cls_pooling: false, // xCoRe uses mean pooling
            max_mentions: 50,
            merge_threshold: 0.5,
        }
    }
}

// =============================================================================
// Neural Cluster Encoder (Candle)
// =============================================================================

#[cfg(feature = "candle")]
/// Neural cluster encoder using Candle.
///
/// Encodes clusters using:
/// 1. Pre-trained text encoder for mention representations
/// 2. Single-layer Transformer for cluster pooling (xCoRe style)
pub struct CandleClusterEncoder<E: CandleTextEncoder> {
    /// Text encoder (DeBERTa, ModernBERT, etc.)
    encoder: E,
    /// Pooling transformer layer
    pooling_layer: ClusterPoolingLayer,
    /// Configuration
    config: NeuralClusterConfig,
    /// Device
    device: Device,
}

#[cfg(feature = "candle")]
impl<E: CandleTextEncoder> CandleClusterEncoder<E> {
    /// Create a new neural cluster encoder.
    pub fn new(encoder: E, config: NeuralClusterConfig) -> crate::Result<Self> {
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        let pooling_layer = ClusterPoolingLayer::new(&config, &device)?;

        Ok(Self {
            encoder,
            pooling_layer,
            config,
            device,
        })
    }

    /// Encode a cluster's mentions into a single embedding.
    fn encode_cluster_impl(&self, cluster: &LocalCluster) -> crate::Result<Vec<f32>> {
        if cluster.mentions.is_empty() {
            return Ok(vec![0.0; self.config.hidden_dim]);
        }

        // Truncate to max mentions
        let mentions: Vec<&ClusterMention> = cluster
            .mentions
            .iter()
            .take(self.config.max_mentions)
            .collect();

        // Encode each mention text
        let mut mention_embeddings = Vec::new();
        for mention in &mentions {
            let (embeddings, seq_len) = self.encoder.encode(&mention.text)?;

            // Mean pool over tokens
            let hidden_dim = self.config.hidden_dim;
            let mut pooled = vec![0.0f32; hidden_dim];
            if seq_len > 0 {
                for i in 0..seq_len {
                    for j in 0..hidden_dim {
                        pooled[j] += embeddings[i * hidden_dim + j];
                    }
                }
                for p in &mut pooled {
                    *p /= seq_len as f32;
                }
            }
            mention_embeddings.push(pooled);
        }

        // Stack into tensor [num_mentions, hidden_dim]
        let num_mentions = mention_embeddings.len();
        let flat: Vec<f32> = mention_embeddings.into_iter().flatten().collect();
        let tensor = Tensor::from_vec(flat, (num_mentions, self.config.hidden_dim), &self.device)
            .map_err(|e: candle_core::Error| crate::Error::Inference(e.to_string()))?;

        // Apply pooling transformer
        let pooled = self.pooling_layer.forward(&tensor)?;

        // Extract as Vec<f32>
        let result = pooled
            .to_vec1::<f32>()
            .map_err(|e: candle_core::Error| crate::Error::Inference(e.to_string()))?;

        Ok(result)
    }
}

#[cfg(feature = "candle")]
impl<E: CandleTextEncoder> ClusterEncoder for CandleClusterEncoder<E> {
    fn encode_cluster(
        &self,
        cluster: &LocalCluster,
        _hidden_states: Option<&[Vec<f32>]>,
    ) -> ClusterEmbedding {
        let embedding = self
            .encode_cluster_impl(cluster)
            .unwrap_or_else(|_| vec![0.0; self.config.hidden_dim]);

        ClusterEmbedding {
            cluster_id: cluster.id,
            context_id: cluster.context_id,
            embedding,
            mention_count: cluster.mentions.len(),
        }
    }

    fn embedding_dim(&self) -> usize {
        self.config.hidden_dim
    }
}

// =============================================================================
// Cluster Pooling Layer
// =============================================================================

#[cfg(feature = "candle")]
/// Single-layer Transformer for pooling mention representations.
///
/// From xCoRe (Section 3.2.2):
/// "We compute the representation for each cluster W_j using a single-layer
/// Transformer T to encode the hidden states of each of its mentions."
struct ClusterPoolingLayer {
    /// Query projection
    wq: Linear,
    /// Key projection
    wk: Linear,
    /// Value projection
    wv: Linear,
    /// Output projection
    wo: Linear,
    /// Layer norm
    ln: LayerNorm,
    /// Number of heads
    num_heads: usize,
    /// Head dimension
    head_dim: usize,
}

#[cfg(feature = "candle")]
impl ClusterPoolingLayer {
    fn new(config: &NeuralClusterConfig, device: &Device) -> crate::Result<Self> {
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);

        let hidden_dim = config.hidden_dim;
        let num_heads = config.num_heads;
        let head_dim = hidden_dim / num_heads;

        let wq = linear(hidden_dim, hidden_dim, vb.pp("wq"))
            .map_err(|e| crate::Error::Inference(format!("Linear wq: {}", e)))?;
        let wk = linear(hidden_dim, hidden_dim, vb.pp("wk"))
            .map_err(|e| crate::Error::Inference(format!("Linear wk: {}", e)))?;
        let wv = linear(hidden_dim, hidden_dim, vb.pp("wv"))
            .map_err(|e| crate::Error::Inference(format!("Linear wv: {}", e)))?;
        let wo = linear(hidden_dim, hidden_dim, vb.pp("wo"))
            .map_err(|e| crate::Error::Inference(format!("Linear wo: {}", e)))?;
        let ln = layer_norm(hidden_dim, 1e-5, vb.pp("ln"))
            .map_err(|e| crate::Error::Inference(format!("LayerNorm: {}", e)))?;

        Ok(Self {
            wq,
            wk,
            wv,
            wo,
            ln,
            num_heads,
            head_dim,
        })
    }

    /// Forward pass: pool mentions into single cluster representation.
    fn forward(&self, x: &Tensor) -> crate::Result<Tensor> {
        let (seq_len, hidden_dim) = x
            .dims2()
            .map_err(|e| crate::Error::Inference(format!("Dims: {}", e)))?;

        // Self-attention
        let q = self
            .wq
            .forward(x)
            .map_err(|e| crate::Error::Inference(format!("Q: {}", e)))?;
        let k = self
            .wk
            .forward(x)
            .map_err(|e| crate::Error::Inference(format!("K: {}", e)))?;
        let v = self
            .wv
            .forward(x)
            .map_err(|e| crate::Error::Inference(format!("V: {}", e)))?;

        // Reshape for multi-head attention: [seq, heads, head_dim]
        let q = q
            .reshape((seq_len, self.num_heads, self.head_dim))
            .map_err(|e| crate::Error::Inference(format!("Q reshape: {}", e)))?
            .transpose(0, 1)
            .map_err(|e| crate::Error::Inference(format!("Q transpose: {}", e)))?;
        let k = k
            .reshape((seq_len, self.num_heads, self.head_dim))
            .map_err(|e| crate::Error::Inference(format!("K reshape: {}", e)))?
            .transpose(0, 1)
            .map_err(|e| crate::Error::Inference(format!("K transpose: {}", e)))?;
        let v = v
            .reshape((seq_len, self.num_heads, self.head_dim))
            .map_err(|e| crate::Error::Inference(format!("V reshape: {}", e)))?
            .transpose(0, 1)
            .map_err(|e| crate::Error::Inference(format!("V transpose: {}", e)))?;

        // Attention scores
        let scale = (self.head_dim as f64).sqrt();
        let scores = q
            .matmul(
                &k.transpose(1, 2)
                    .map_err(|e| crate::Error::Inference(format!("K^T: {}", e)))?,
            )
            .map_err(|e| crate::Error::Inference(format!("QK^T: {}", e)))?
            .affine(1.0 / scale, 0.0)
            .map_err(|e| crate::Error::Inference(format!("Scale: {}", e)))?;

        let attn = candle_nn::ops::softmax(&scores, D::Minus1)
            .map_err(|e| crate::Error::Inference(format!("Softmax: {}", e)))?;

        // Apply attention to values
        let context = attn
            .matmul(&v)
            .map_err(|e| crate::Error::Inference(format!("Attn*V: {}", e)))?;

        // Reshape back: [heads, seq, head_dim] -> [seq, hidden]
        let context = context
            .transpose(0, 1)
            .map_err(|e| crate::Error::Inference(format!("Context transpose: {}", e)))?
            .reshape((seq_len, hidden_dim))
            .map_err(|e| crate::Error::Inference(format!("Context reshape: {}", e)))?;

        // Output projection + residual + layer norm
        let out = self
            .wo
            .forward(&context)
            .map_err(|e| crate::Error::Inference(format!("Wo: {}", e)))?;
        let out = (x + &out).map_err(|e| crate::Error::Inference(format!("Residual: {}", e)))?;
        let out = self
            .ln
            .forward(&out)
            .map_err(|e| crate::Error::Inference(format!("LayerNorm: {}", e)))?;

        // Mean pool over mentions to get single cluster representation
        out.mean(0)
            .map_err(|e| crate::Error::Inference(format!("Mean pool: {}", e)))
    }
}

// =============================================================================
// Neural Merge Scorer
// =============================================================================

#[cfg(feature = "candle")]
/// Neural merge scorer using learned bilinear scoring.
///
/// From xCoRe (Section 3.2.2):
/// "We calculate the pairwise coreference probability p_cm between clusters'
/// hidden representations using a linear classification layer."
pub struct NeuralMergeScorer {
    /// Bilinear weight matrix
    bilinear: Linear,
    /// Output classification layer
    classifier: Linear,
    /// Device
    device: Device,
}

#[cfg(feature = "candle")]
impl NeuralMergeScorer {
    /// Create a new neural merge scorer.
    pub fn new(hidden_dim: usize) -> crate::Result<Self> {
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        // Concatenate two cluster embeddings -> hidden_dim * 2
        let bilinear = linear(hidden_dim * 2, hidden_dim, vb.pp("bilinear"))
            .map_err(|e| crate::Error::Inference(format!("Bilinear: {}", e)))?;
        let classifier = linear(hidden_dim, 1, vb.pp("classifier"))
            .map_err(|e| crate::Error::Inference(format!("Classifier: {}", e)))?;

        Ok(Self {
            bilinear,
            classifier,
            device,
        })
    }

    /// Score a pair of cluster embeddings.
    fn score_impl(&self, emb_a: &[f32], emb_b: &[f32]) -> crate::Result<f32> {
        // Concatenate embeddings
        let concat: Vec<f32> = emb_a.iter().chain(emb_b.iter()).cloned().collect();
        let input = Tensor::from_vec(concat, (1, emb_a.len() + emb_b.len()), &self.device)
            .map_err(|e| crate::Error::Inference(format!("Input tensor: {}", e)))?;

        // Forward through network
        let hidden = self
            .bilinear
            .forward(&input)
            .map_err(|e| crate::Error::Inference(format!("Bilinear forward: {}", e)))?;
        let hidden = hidden
            .relu()
            .map_err(|e| crate::Error::Inference(format!("ReLU: {}", e)))?;
        let logit = self
            .classifier
            .forward(&hidden)
            .map_err(|e| crate::Error::Inference(format!("Classifier forward: {}", e)))?;

        // Sigmoid to get probability
        let prob = candle_nn::ops::sigmoid(&logit)
            .map_err(|e| crate::Error::Inference(format!("Sigmoid: {}", e)))?;

        let score = prob
            .to_vec2::<f32>()
            .map_err(|e| crate::Error::Inference(format!("To vec: {}", e)))?[0][0];

        Ok(score)
    }
}

#[cfg(feature = "candle")]
impl MergeScorer for NeuralMergeScorer {
    fn score(&self, embedding_a: &ClusterEmbedding, embedding_b: &ClusterEmbedding) -> f32 {
        self.score_impl(&embedding_a.embedding, &embedding_b.embedding)
            .unwrap_or(0.0)
    }
}

// =============================================================================
// CDCR Integration Adapter
// =============================================================================

/// Adapter to use ClusterEncoder with existing CDCR infrastructure.
///
/// Converts between `cdcr::Document` and `cluster_encoder::LocalCluster`.
pub struct CDCRAdapter;

impl CDCRAdapter {
    /// Convert CDCR documents to local clusters.
    pub fn documents_to_clusters(
        docs: &[crate::eval::cdcr::Document],
    ) -> HashMap<usize, Vec<LocalCluster>> {
        let mut all_clusters = HashMap::new();

        for (doc_idx, doc) in docs.iter().enumerate() {
            let mut clusters = Vec::new();

            // Each coref chain becomes a cluster
            for (chain_idx, chain) in doc.coref_chains.iter().enumerate() {
                let mut cluster = LocalCluster::new(chain_idx, doc_idx);

                for mention in &chain.mentions {
                    cluster.add_mention(ClusterMention {
                        start: mention.start,
                        end: mention.end,
                        text: mention.text.clone(),
                        context_id: doc_idx,
                    });
                }

                cluster.compute_canonical();
                clusters.push(cluster);
            }

            // Also add singletons (entities not in chains)
            let chained_starts: std::collections::HashSet<usize> = doc
                .coref_chains
                .iter()
                .flat_map(|c| c.mentions.iter().map(|m| m.start))
                .collect();

            for entity in &doc.entities {
                if !chained_starts.contains(&entity.start()) {
                    let mut cluster = LocalCluster::new(clusters.len(), doc_idx);
                    cluster.add_mention(ClusterMention {
                        start: entity.start(),
                        end: entity.end(),
                        text: entity.text.clone(),
                        context_id: doc_idx,
                    });
                    cluster.compute_canonical();
                    clusters.push(cluster);
                }
            }

            all_clusters.insert(doc_idx, clusters);
        }

        all_clusters
    }

    /// Convert merged clusters back to CrossDocClusters.
    pub fn clusters_to_crossdoc(
        merged: &[crate::eval::cluster_encoder::MergedCluster],
        docs: &[crate::eval::cdcr::Document],
    ) -> Vec<crate::eval::cdcr::CrossDocCluster> {
        merged
            .iter()
            .map(|m| {
                let mut cluster = crate::eval::cdcr::CrossDocCluster::new(
                    m.id as u64,
                    m.canonical.as_deref().unwrap_or(""),
                );

                for mention in &m.mentions {
                    if let Some(doc) = docs.get(mention.context_id) {
                        // Find entity index in document
                        let entity_idx = doc
                            .entities
                            .iter()
                            .position(|e| e.start() == mention.start && e.end() == mention.end)
                            .unwrap_or(0);
                        cluster.add_mention(&doc.id, entity_idx);
                    }
                }

                cluster
            })
            .collect()
    }
}

// =============================================================================
// Unified Cross-Context Resolver
// =============================================================================

/// Unified resolver for cross-context coreference using xCoRe approach.
///
/// Supports both:
/// - Long document: Split into windows → resolve across windows
/// - Cross-document: Multiple docs → resolve across documents
pub struct UnifiedCrossContextResolver<E: ClusterEncoder, S: MergeScorer> {
    encoder: E,
    scorer: S,
    config: CrossContextConfig,
}

/// Configuration for cross-context resolution.
#[derive(Debug, Clone)]
pub struct CrossContextConfig {
    /// Window size for long documents
    pub window_size: usize,
    /// Window overlap
    pub window_overlap: usize,
    /// Merge probability threshold
    pub merge_threshold: f32,
}

impl Default for CrossContextConfig {
    fn default() -> Self {
        Self {
            window_size: 4000,
            window_overlap: 256,
            merge_threshold: 0.5,
        }
    }
}

impl<E: ClusterEncoder, S: MergeScorer> UnifiedCrossContextResolver<E, S> {
    /// Create a new resolver.
    pub fn new(encoder: E, scorer: S, config: CrossContextConfig) -> Self {
        Self {
            encoder,
            scorer,
            config,
        }
    }

    /// Resolve coreference across CDCR documents.
    pub fn resolve_documents(
        &self,
        docs: &[crate::eval::cdcr::Document],
    ) -> Vec<crate::eval::cdcr::CrossDocCluster> {
        // Convert to local clusters
        let local_clusters = CDCRAdapter::documents_to_clusters(docs);

        // Encode and merge
        let merged = self.merge_clusters(&local_clusters);

        // Convert back to CrossDocCluster format
        CDCRAdapter::clusters_to_crossdoc(&merged, docs)
    }

    /// Internal merge implementation.
    fn merge_clusters(
        &self,
        local_clusters: &HashMap<usize, Vec<LocalCluster>>,
    ) -> Vec<crate::eval::cluster_encoder::MergedCluster> {
        // Encode all clusters
        let mut embeddings: Vec<ClusterEmbedding> = Vec::new();
        for clusters in local_clusters.values() {
            for cluster in clusters {
                let emb = self.encoder.encode_cluster(cluster, None);
                embeddings.push(emb);
            }
        }

        if embeddings.is_empty() {
            return Vec::new();
        }

        // Score pairwise merges (skip same-context)
        let mut merge_pairs: Vec<(usize, usize, f32)> = Vec::new();
        for (i, emb_a) in embeddings.iter().enumerate() {
            for (j, emb_b) in embeddings.iter().enumerate().skip(i + 1) {
                if emb_a.context_id == emb_b.context_id {
                    continue;
                }

                let score = self.scorer.score(emb_a, emb_b);
                if score >= self.config.merge_threshold {
                    merge_pairs.push((i, j, score));
                }
            }
        }

        // Sort by score descending
        merge_pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        // Union-Find merge
        let n = embeddings.len();
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

        for (i, j, _) in merge_pairs {
            union(&mut parent, &mut rank, i, j);
        }

        // Build merged clusters
        let mut cluster_map: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..n {
            let root = find(&mut parent, i);
            cluster_map.entry(root).or_default().push(i);
        }

        cluster_map
            .into_iter()
            .enumerate()
            .map(|(merged_id, (_root, indices))| {
                let mut merged = crate::eval::cluster_encoder::MergedCluster {
                    id: merged_id,
                    source_clusters: Vec::new(),
                    mentions: Vec::new(),
                    canonical: None,
                };

                for idx in indices {
                    let emb = &embeddings[idx];
                    merged
                        .source_clusters
                        .push((emb.context_id, emb.cluster_id));

                    // Find original cluster and copy mentions
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
}

// =============================================================================
// Incremental Coref Integration
// =============================================================================

/// Adapter to integrate IncrementalCorefResolver with cross-context system.
///
/// Connects the long-document incremental resolver with xCoRe's
/// cross-context cluster merging.
pub struct IncrementalCorefAdapter;

impl IncrementalCorefAdapter {
    /// Convert incremental resolver output to local clusters for merging.
    ///
    /// Each window is treated as a separate context.
    pub fn windows_to_clusters(windows: &[WindowOutput]) -> HashMap<usize, Vec<LocalCluster>> {
        let mut all_clusters = HashMap::new();

        for (window_idx, output) in windows.iter().enumerate() {
            let mut clusters = Vec::new();

            for (chain_idx, chain) in output.chains.iter().enumerate() {
                let mut cluster = LocalCluster::new(chain_idx, window_idx);

                for mention in &chain.mentions {
                    cluster.add_mention(ClusterMention {
                        start: mention.start,
                        end: mention.end,
                        text: mention.text.clone(),
                        context_id: window_idx,
                    });
                }

                cluster.compute_canonical();
                clusters.push(cluster);
            }

            all_clusters.insert(window_idx, clusters);
        }

        all_clusters
    }

    /// Build final chains from merged clusters.
    pub fn clusters_to_chains(
        merged: &[crate::eval::cluster_encoder::MergedCluster],
    ) -> Vec<crate::eval::coref::CorefChain> {
        use crate::eval::coref::{CorefChain, Mention, MentionType};

        merged
            .iter()
            .map(|m| {
                let mentions: Vec<Mention> = m
                    .mentions
                    .iter()
                    .map(|cm| Mention {
                        text: cm.text.clone(),
                        start: cm.start,
                        end: cm.end,
                        head_start: None,
                        head_end: None,
                        entity_type: None,
                        mention_type: Some(MentionType::Proper),
                    })
                    .collect();

                CorefChain::new(mentions)
            })
            .collect()
    }
}

/// Output from a single window of incremental processing.
#[derive(Debug, Clone)]
pub struct WindowOutput {
    /// Window index
    pub window_idx: usize,
    /// Character offset of window start
    pub start_offset: usize,
    /// Character offset of window end
    pub end_offset: usize,
    /// Chains extracted from this window
    pub chains: Vec<crate::eval::coref::CorefChain>,
}

impl WindowOutput {
    /// Create a new window output.
    pub fn new(
        window_idx: usize,
        start_offset: usize,
        end_offset: usize,
        chains: Vec<crate::eval::coref::CorefChain>,
    ) -> Self {
        Self {
            window_idx,
            start_offset,
            end_offset,
            chains,
        }
    }
}

/// Extended cross-context resolver that works with long documents.
impl<E: ClusterEncoder, S: MergeScorer> UnifiedCrossContextResolver<E, S> {
    /// Resolve coreference in a long document by splitting into windows.
    ///
    /// Uses the xCoRe approach:
    /// 1. Split document into overlapping windows
    /// 2. Extract clusters within each window (using incremental resolver)
    /// 3. Merge clusters across windows
    pub fn resolve_long_document_windows(
        &self,
        windows: &[WindowOutput],
    ) -> Vec<crate::eval::coref::CorefChain> {
        // Convert windows to local clusters
        let local_clusters = IncrementalCorefAdapter::windows_to_clusters(windows);

        // Merge across contexts
        let merged = self.merge_clusters(&local_clusters);

        // Convert back to chains
        IncrementalCorefAdapter::clusters_to_chains(&merged)
    }

    /// Get the configuration.
    pub fn config(&self) -> &CrossContextConfig {
        &self.config
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::cluster_encoder::{CosineMergeScorer, HeuristicClusterEncoder};

    #[test]
    fn test_cdcr_adapter_empty() {
        let docs: Vec<crate::eval::cdcr::Document> = vec![];
        let clusters = CDCRAdapter::documents_to_clusters(&docs);
        assert!(clusters.is_empty());
    }

    #[test]
    fn test_cdcr_adapter_single_doc() {
        use crate::eval::cdcr::Document;
        use anno::{Entity, EntityType};

        let doc = Document::new("doc1", "Obama visited France.").with_entities(vec![Entity::new(
            "Obama",
            EntityType::Person,
            0,
            5,
            0.9,
        )]);

        let clusters = CDCRAdapter::documents_to_clusters(&[doc]);
        assert_eq!(clusters.len(), 1);
        assert!(!clusters[&0].is_empty());
    }

    #[test]
    fn test_unified_resolver() {
        use crate::eval::cdcr::Document;
        use anno::{Entity, EntityType};

        let encoder = HeuristicClusterEncoder::new(64);
        let scorer = CosineMergeScorer::new();
        let config = CrossContextConfig::default();

        let resolver = UnifiedCrossContextResolver::new(encoder, scorer, config);

        let docs =
            vec![
                Document::new("doc1", "Barack Obama gave a speech.").with_entities(vec![
                    Entity::new("Barack Obama", EntityType::Person, 0, 12, 0.9),
                ]),
                Document::new("doc2", "Obama met with leaders.").with_entities(vec![Entity::new(
                    "Obama",
                    EntityType::Person,
                    0,
                    5,
                    0.9,
                )]),
            ];

        let result = resolver.resolve_documents(&docs);
        // Should have clusters from both docs
        assert!(!result.is_empty());
    }

    #[test]
    fn test_neural_config_default() {
        let config = NeuralClusterConfig::default();
        assert_eq!(config.hidden_dim, 768);
        assert_eq!(config.num_heads, 12);
        assert!(!config.use_cls_pooling);
    }

    #[test]
    fn test_incremental_adapter_empty() {
        let windows: Vec<WindowOutput> = vec![];
        let clusters = IncrementalCorefAdapter::windows_to_clusters(&windows);
        assert!(clusters.is_empty());
    }

    #[test]
    fn test_incremental_adapter_single_window() {
        use crate::eval::coref::{CorefChain, Mention};
        use anno::MentionType;

        fn new_mention(text: &str, start: usize, end: usize, mt: MentionType) -> Mention {
            Mention {
                text: text.to_string(),
                start,
                end,
                head_start: None,
                head_end: None,
                entity_type: None,
                mention_type: Some(mt),
            }
        }

        let chain = CorefChain::new(vec![
            new_mention("Obama", 0, 5, MentionType::Proper),
            new_mention("he", 20, 22, MentionType::Pronominal),
        ]);

        let window = WindowOutput::new(0, 0, 100, vec![chain]);
        let clusters = IncrementalCorefAdapter::windows_to_clusters(&[window]);

        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[&0].len(), 1);
        assert_eq!(clusters[&0][0].mentions.len(), 2);
    }

    #[test]
    fn test_incremental_adapter_multi_window() {
        use crate::eval::coref::{CorefChain, Mention};
        use anno::MentionType;

        fn new_mention(text: &str, start: usize, end: usize, mt: MentionType) -> Mention {
            Mention {
                text: text.to_string(),
                start,
                end,
                head_start: None,
                head_end: None,
                entity_type: None,
                mention_type: Some(mt),
            }
        }

        let window1 = WindowOutput::new(
            0,
            0,
            100,
            vec![CorefChain::new(vec![new_mention(
                "Obama",
                0,
                5,
                MentionType::Proper,
            )])],
        );

        let window2 = WindowOutput::new(
            1,
            80,
            180,
            vec![CorefChain::new(vec![new_mention(
                "the President",
                90,
                103,
                MentionType::Nominal,
            )])],
        );

        let clusters = IncrementalCorefAdapter::windows_to_clusters(&[window1, window2]);

        assert_eq!(clusters.len(), 2);
        assert_eq!(clusters[&0].len(), 1);
        assert_eq!(clusters[&1].len(), 1);
    }

    #[test]
    fn test_long_document_resolution() {
        use crate::eval::coref::{CorefChain, Mention};
        use anno::MentionType;

        fn new_mention(text: &str, start: usize, end: usize, mt: MentionType) -> Mention {
            Mention {
                text: text.to_string(),
                start,
                end,
                head_start: None,
                head_end: None,
                entity_type: None,
                mention_type: Some(mt),
            }
        }

        let encoder = HeuristicClusterEncoder::new(64);
        let scorer = CosineMergeScorer::new();
        let config = CrossContextConfig {
            merge_threshold: 0.3,
            ..Default::default()
        };

        let resolver = UnifiedCrossContextResolver::new(encoder, scorer, config);

        // Create overlapping windows with similar mentions
        let window1 = WindowOutput::new(
            0,
            0,
            1000,
            vec![CorefChain::new(vec![new_mention(
                "Barack Obama",
                0,
                12,
                MentionType::Proper,
            )])],
        );

        let window2 = WindowOutput::new(
            1,
            800,
            1800,
            vec![CorefChain::new(vec![new_mention(
                "Obama",
                900,
                905,
                MentionType::Proper,
            )])],
        );

        let chains = resolver.resolve_long_document_windows(&[window1, window2]);

        // Should produce some chains
        assert!(!chains.is_empty());
    }
}
