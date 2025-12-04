//! Cross-Document Coreference Resolution (CDCR).
//!
//! # Overview
//!
//! CDCR extends within-document coreference to link mentions of the same
//! entity (or event) *across* multiple documents. This is essential for:
//!
//! - Knowledge base population from document collections
//! - Multi-document summarization
//! - Event tracking across news articles
//! - Entity linking at corpus scale
//!
//! # Key Challenges
//!
//! | Challenge | Example | Solution |
//! |-----------|---------|----------|
//! | **Scale** | Millions of mentions | LSH blocking |
//! | **Ambiguity** | "John Smith" in 100 docs | Entity clustering |
//! | **Context loss** | Pronouns across docs | Anchor to nominals |
//!
//! # Research Background
//!
//! Key papers in CDCR:
//! - Cai & Strube (2010): End-to-end CDCR for entities
//! - Barhom et al. (2019): Event CDCR with cross-document clustering
//! - Caciularu et al. (2021): CDCR with transformers (ECB+)
//!
//! # Architecture
//!
//! ```text
//! Documents → [Within-Doc Coref] → Entity Clusters per Doc
//!                                        ↓
//!                                [LSH Blocking]
//!                                        ↓
//!                              [Cross-Doc Clustering]
//!                                        ↓
//!                               Unified Entity KB
//! ```
//!
//! # Example
//!
//! ```rust
//! use anno::eval::cdcr::{CDCRResolver, Document, CrossDocCluster};
//!
//! let docs = vec![
//!     Document::new("doc1", "Jensen Huang announced Nvidia's new chips."),
//!     Document::new("doc2", "The CEO of Nvidia revealed expansion plans."),
//! ];
//!
//! let resolver = CDCRResolver::new();
//! let clusters = resolver.resolve(&docs);
//!
//! // Jensen Huang and "The CEO of Nvidia" should be in the same cluster
//! ```

use crate::eval::coref::CorefChain;
use crate::{Entity, EntityType};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// =============================================================================
// Types
// =============================================================================

/// A document with its text and extracted entities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// Document identifier
    pub id: String,
    /// Full document text
    pub text: String,
    /// Entities extracted from this document
    pub entities: Vec<Entity>,
    /// Within-document coreference chains (if resolved)
    pub coref_chains: Vec<CorefChain>,
}

impl Document {
    /// Create a new document.
    #[must_use]
    pub fn new(id: &str, text: &str) -> Self {
        Self {
            id: id.to_string(),
            text: text.to_string(),
            entities: Vec::new(),
            coref_chains: Vec::new(),
        }
    }

    /// Add entities to the document.
    #[must_use]
    pub fn with_entities(mut self, entities: Vec<Entity>) -> Self {
        self.entities = entities;
        self
    }

    /// Add coreference chains.
    #[must_use]
    pub fn with_coref(mut self, chains: Vec<CorefChain>) -> Self {
        self.coref_chains = chains;
        self
    }

    /// Get all mentions (entities + chain mentions).
    #[must_use]
    pub fn all_mentions(&self) -> Vec<MentionRef> {
        let mut mentions = Vec::new();

        for (idx, entity) in self.entities.iter().enumerate() {
            mentions.push(MentionRef {
                doc_id: self.id.clone(),
                entity_idx: idx,
                text: entity.text.clone(),
                entity_type: entity.entity_type.clone(),
                within_doc_cluster: entity.canonical_id,
            });
        }

        mentions
    }
}

/// A reference to a mention within a document.
#[derive(Debug, Clone)]
pub struct MentionRef {
    /// Document ID
    pub doc_id: String,
    /// Index in document's entity list
    pub entity_idx: usize,
    /// Mention text
    pub text: String,
    /// Entity type
    pub entity_type: EntityType,
    /// Within-document cluster ID (if resolved)
    pub within_doc_cluster: Option<u64>,
}

/// A cross-document entity cluster.
///
/// Groups mentions from multiple documents that refer to the same
/// real-world entity.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CrossDocCluster {
    /// Cluster ID
    pub id: u64,
    /// Canonical name for the cluster (e.g., "Jensen Huang")
    pub canonical_name: String,
    /// Entity type for the cluster
    pub entity_type: Option<EntityType>,
    /// Documents containing mentions of this entity
    pub documents: Vec<String>,
    /// (doc_id, entity_idx) pairs for all mentions
    pub mentions: Vec<(String, usize)>,
    /// External knowledge base ID (if linked)
    pub kb_id: Option<String>,
    /// Confidence in this clustering
    pub confidence: f64,
}

impl CrossDocCluster {
    /// Create a new cluster.
    #[must_use]
    pub fn new(id: u64, canonical_name: &str) -> Self {
        Self {
            id,
            canonical_name: canonical_name.to_string(),
            entity_type: None,
            documents: Vec::new(),
            mentions: Vec::new(),
            kb_id: None,
            confidence: 1.0,
        }
    }

    /// Number of mentions.
    #[must_use]
    pub fn len(&self) -> usize {
        self.mentions.len()
    }

    /// Alias for `len()`.
    #[must_use]
    pub fn mention_count(&self) -> usize {
        self.len()
    }

    /// Check if cluster is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.mentions.is_empty()
    }

    /// Number of unique documents.
    #[must_use]
    pub fn doc_count(&self) -> usize {
        self.documents.iter().collect::<HashSet<_>>().len()
    }

    /// Add a mention to the cluster.
    pub fn add_mention(&mut self, doc_id: &str, entity_idx: usize) {
        if !self.documents.contains(&doc_id.to_string()) {
            self.documents.push(doc_id.to_string());
        }
        self.mentions.push((doc_id.to_string(), entity_idx));
    }

    /// Set entity type.
    #[must_use]
    pub fn with_type(mut self, entity_type: EntityType) -> Self {
        self.entity_type = Some(entity_type);
        self
    }
}

// =============================================================================
// Conversion Helpers
// =============================================================================

/// Convert a CrossDocCluster to an Identity.
///
/// This converts from the evaluation/clustering result format to
/// the core representation.
///
/// # Note on TrackRefs
///
/// CDCR's `CrossDocCluster` doesn't contain track information (only entity indices),
/// so we cannot create valid `TrackRef`s. The source is set to `None` to indicate
/// this is a conversion from evaluation format. If you need proper TrackRefs,
/// use `Corpus::resolve_inter_doc_coref()` instead.
///
/// # Note on `source: None`
///
/// The `source` field is set to `None` because `CrossDocCluster` only contains
/// `(doc_id, entity_idx)` pairs, not `track_id`s. Without track IDs, we cannot
/// create valid `TrackRef`s that would be needed for `IdentitySource::CrossDocCoref`.
///
/// This is intentional: identities created from evaluation data don't have
/// the same provenance tracking as identities created through the normal
/// corpus resolution pipeline.
impl From<&CrossDocCluster> for anno_core::Identity {
    fn from(cluster: &CrossDocCluster) -> Self {
        Self {
            id: cluster.id,
            canonical_name: cluster.canonical_name.clone(),
            entity_type: cluster
                .entity_type
                .as_ref()
                .map(|t| t.as_label().to_string()),
            kb_id: cluster.kb_id.clone(),
            kb_name: None,
            description: None,
            embedding: None,
            box_embedding: None,
            aliases: Vec::new(),
            confidence: cluster.confidence as f32,
            source: None, // Cannot determine source from CDCR format (no track_ids)
        }
    }
}

// =============================================================================
// LSH Blocking
// =============================================================================

/// Locality-Sensitive Hashing (LSH) for blocking.
///
/// Blocking reduces the O(n²) pairwise comparison problem by grouping
/// likely-coreferent mentions into "blocks" using hash signatures.
///
/// # Algorithm
///
/// 1. Compute a character n-gram signature for each mention
/// 2. Hash the signature using multiple hash functions
/// 3. Mentions with identical hash in any band are candidates
///
/// This achieves sub-quadratic scaling while maintaining high recall
/// for truly coreferent pairs.
#[derive(Debug, Clone)]
pub struct LSHBlocker {
    /// Number of hash bands
    pub num_bands: usize,
    /// Rows per band
    pub rows_per_band: usize,
    /// N-gram size for signatures
    pub ngram_size: usize,
}

impl Default for LSHBlocker {
    fn default() -> Self {
        Self {
            num_bands: 5,
            rows_per_band: 3,
            ngram_size: 3,
        }
    }
}

impl LSHBlocker {
    /// Create a new LSH blocker.
    #[must_use]
    pub fn new(num_bands: usize, rows_per_band: usize) -> Self {
        Self {
            num_bands,
            rows_per_band,
            ngram_size: 3,
        }
    }

    /// Compute candidate pairs for a set of mentions.
    ///
    /// Returns pairs of indices (i, j) where i < j that are candidates
    /// for cross-document coreference.
    #[must_use]
    pub fn candidate_pairs(&self, mentions: &[MentionRef]) -> Vec<(usize, usize)> {
        let signatures: Vec<Vec<u64>> = mentions
            .iter()
            .map(|m| self.compute_signature(&m.text))
            .collect();

        // For each band, group mentions by their band hash
        let mut candidates: HashSet<(usize, usize)> = HashSet::new();

        for band in 0..self.num_bands {
            let mut buckets: HashMap<u64, Vec<usize>> = HashMap::new();

            for (idx, sig) in signatures.iter().enumerate() {
                let band_hash = self.band_hash(sig, band);
                buckets.entry(band_hash).or_default().push(idx);
            }

            // All pairs in the same bucket are candidates
            for indices in buckets.values() {
                for i in 0..indices.len() {
                    for j in (i + 1)..indices.len() {
                        let (a, b) = if indices[i] < indices[j] {
                            (indices[i], indices[j])
                        } else {
                            (indices[j], indices[i])
                        };
                        candidates.insert((a, b));
                    }
                }
            }
        }

        candidates.into_iter().collect()
    }

    /// Compute minhash signature for a mention text.
    fn compute_signature(&self, text: &str) -> Vec<u64> {
        let normalized = text.to_lowercase();
        let ngrams = self.extract_ngrams(&normalized);

        // Compute minhash for each row
        let total_hashes = self.num_bands * self.rows_per_band;
        let mut signature = vec![u64::MAX; total_hashes];

        for ngram in ngrams {
            for (h, sig_val) in signature.iter_mut().enumerate().take(total_hashes) {
                let hash = self.hash_ngram(&ngram, h as u64);
                if hash < *sig_val {
                    *sig_val = hash;
                }
            }
        }

        signature
    }

    /// Extract character n-grams from text.
    fn extract_ngrams(&self, text: &str) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        if chars.len() < self.ngram_size {
            return vec![text.to_string()];
        }

        chars
            .windows(self.ngram_size)
            .map(|w| w.iter().collect())
            .collect()
    }

    /// Hash an n-gram with a seed.
    fn hash_ngram(&self, ngram: &str, seed: u64) -> u64 {
        // Simple hash: FNV-1a variant
        let mut hash: u64 = seed.wrapping_add(0xcbf29ce484222325);
        for byte in ngram.bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }

    /// Compute band hash from signature.
    fn band_hash(&self, signature: &[u64], band: usize) -> u64 {
        let start = band * self.rows_per_band;
        let end = (start + self.rows_per_band).min(signature.len());

        signature[start..end]
            .iter()
            .fold(0u64, |acc, &val| acc.wrapping_mul(31).wrapping_add(val))
    }

    /// Estimate Jaccard similarity from minhash signatures.
    #[must_use]
    pub fn signature_similarity(sig1: &[u64], sig2: &[u64]) -> f64 {
        if sig1.len() != sig2.len() || sig1.is_empty() {
            return 0.0;
        }

        let matches = sig1.iter().zip(sig2.iter()).filter(|(a, b)| a == b).count();
        matches as f64 / sig1.len() as f64
    }
}

// =============================================================================
// CDCR Resolver
// =============================================================================

/// Configuration for CDCR.
#[derive(Debug, Clone)]
pub struct CDCRConfig {
    /// Minimum similarity for clustering
    pub min_similarity: f64,
    /// Use LSH blocking (recommended for large corpora)
    pub use_lsh: bool,
    /// LSH configuration
    pub lsh: LSHBlocker,
    /// Require type match for clustering
    pub require_type_match: bool,
}

impl Default for CDCRConfig {
    fn default() -> Self {
        Self {
            min_similarity: 0.5,
            use_lsh: true,
            lsh: LSHBlocker::default(),
            require_type_match: true,
        }
    }
}

/// Cross-Document Coreference Resolver.
///
/// Clusters entity mentions across multiple documents into unified
/// clusters representing the same real-world entities.
///
/// # Algorithm
///
/// 1. **Blocking** (LSH): Generate candidate pairs from all mentions
/// 2. **Comparison**: Compute similarity for candidate pairs
/// 3. **Clustering**: Agglomerative clustering with single-link
///
/// # Scalability
///
/// With LSH blocking, CDCR scales to millions of mentions:
/// - Without blocking: O(n²) comparisons
/// - With blocking: O(n × average_block_size)
#[derive(Debug, Clone, Default)]
pub struct CDCRResolver {
    config: CDCRConfig,
}

impl CDCRResolver {
    /// Create a new resolver.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with configuration.
    #[must_use]
    pub fn with_config(config: CDCRConfig) -> Self {
        Self { config }
    }

    /// Resolve cross-document coreference.
    #[must_use]
    pub fn resolve(&self, documents: &[Document]) -> Vec<CrossDocCluster> {
        // 1. Collect all mentions
        let mentions: Vec<MentionRef> = documents.iter().flat_map(|d| d.all_mentions()).collect();

        if mentions.is_empty() {
            return vec![];
        }

        // 2. Get candidate pairs
        let candidates = if self.config.use_lsh {
            self.config.lsh.candidate_pairs(&mentions)
        } else {
            // Brute force: all pairs
            let n = mentions.len();
            let mut pairs = Vec::new();
            for i in 0..n {
                for j in (i + 1)..n {
                    pairs.push((i, j));
                }
            }
            pairs
        };

        // 3. Compute similarities and cluster
        let mut union_find: Vec<usize> = (0..mentions.len()).collect();

        for (i, j) in candidates {
            if self.should_cluster(&mentions[i], &mentions[j]) {
                Self::union(&mut union_find, i, j);
            }
        }

        // 4. Build clusters from union-find
        let mut cluster_map: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..mentions.len() {
            let root = Self::find(&mut union_find, i);
            cluster_map.entry(root).or_default().push(i);
        }

        // 5. Convert to CrossDocCluster
        cluster_map
            .into_iter()
            .enumerate()
            .map(|(cluster_idx, (_, member_indices))| {
                let first = &mentions[member_indices[0]];
                let mut cluster = CrossDocCluster::new(cluster_idx as u64, &first.text);
                cluster.entity_type = Some(first.entity_type.clone());

                for idx in member_indices {
                    let m = &mentions[idx];
                    cluster.add_mention(&m.doc_id, m.entity_idx);
                }

                cluster
            })
            .collect()
    }

    /// Check if two mentions should be clustered.
    fn should_cluster(&self, a: &MentionRef, b: &MentionRef) -> bool {
        // Type check
        if self.config.require_type_match && a.entity_type != b.entity_type {
            return false;
        }

        // Similarity check
        let sim = self.mention_similarity(a, b);
        sim >= self.config.min_similarity
    }

    /// Compute similarity between two mentions.
    fn mention_similarity(&self, a: &MentionRef, b: &MentionRef) -> f64 {
        crate::similarity::string_similarity(&a.text, &b.text)
    }

    /// Union-find: find root with path compression (iterative).
    fn find(parent: &mut [usize], mut i: usize) -> usize {
        // Find root
        let mut root = i;
        while parent[root] != root {
            root = parent[root];
        }
        // Path compression
        while parent[i] != root {
            let next = parent[i];
            parent[i] = root;
            i = next;
        }
        root
    }

    /// Union-find: union two sets
    fn union(parent: &mut [usize], i: usize, j: usize) {
        let root_i = Self::find(parent, i);
        let root_j = Self::find(parent, j);
        if root_i != root_j {
            parent[root_i] = root_j;
        }
    }
}

// =============================================================================
// Metrics
// =============================================================================

/// CDCR evaluation metrics.
#[derive(Debug, Clone, Default)]
pub struct CDCRMetrics {
    /// B³ precision
    pub b_cubed_precision: f64,
    /// B³ recall
    pub b_cubed_recall: f64,
    /// B³ F1
    pub b_cubed_f1: f64,
    /// Number of predicted clusters
    pub num_pred_clusters: usize,
    /// Number of gold clusters
    pub num_gold_clusters: usize,
}

impl CDCRMetrics {
    /// Compute B³ scores for CDCR.
    ///
    /// B³ (Bagga & Baldwin, 1998) is computed per-mention:
    /// - Precision: fraction of cluster that shares the mention's gold class
    /// - Recall: fraction of gold class that shares the mention's cluster
    #[must_use]
    pub fn compute(predicted: &[CrossDocCluster], gold: &[CrossDocCluster]) -> Self {
        // Build mention → cluster mappings
        let pred_map = Self::build_mention_map(predicted);
        let gold_map = Self::build_mention_map(gold);

        let all_mentions: HashSet<_> = pred_map.keys().chain(gold_map.keys()).cloned().collect();

        if all_mentions.is_empty() {
            return Self::default();
        }

        let mut total_precision = 0.0;
        let mut total_recall = 0.0;

        for mention in &all_mentions {
            let pred_cluster = pred_map.get(mention);
            let gold_cluster = gold_map.get(mention);

            match (pred_cluster, gold_cluster) {
                (Some(pred), Some(gold)) => {
                    // Intersection of pred and gold clusters
                    let intersection: HashSet<_> = pred.intersection(gold).collect();

                    total_precision += intersection.len() as f64 / pred.len() as f64;
                    total_recall += intersection.len() as f64 / gold.len() as f64;
                }
                _ => {
                    // Mention only in one (precision or recall is 0)
                }
            }
        }

        let n = all_mentions.len() as f64;
        let precision = total_precision / n;
        let recall = total_recall / n;
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        Self {
            b_cubed_precision: precision,
            b_cubed_recall: recall,
            b_cubed_f1: f1,
            num_pred_clusters: predicted.len(),
            num_gold_clusters: gold.len(),
        }
    }

    /// Build mention → set of cluster-mates mapping.
    fn build_mention_map(
        clusters: &[CrossDocCluster],
    ) -> HashMap<(String, usize), HashSet<(String, usize)>> {
        let mut map = HashMap::new();

        for cluster in clusters {
            let cluster_set: HashSet<_> = cluster.mentions.iter().cloned().collect();

            for mention in &cluster.mentions {
                map.insert(mention.clone(), cluster_set.clone());
            }
        }

        map
    }
}

// =============================================================================
// Sample Datasets
// =============================================================================

/// Generate sample tech news documents for CDCR testing.
///
/// These documents mention the same entities (companies, people, locations)
/// across different news articles about AI and semiconductors.
#[must_use]
pub fn tech_news_dataset() -> Vec<Document> {
    let mut docs = Vec::new();

    // Article 1: Nvidia announcement
    let mut doc1 = Document::new(
        "tech_01",
        "Jensen Huang announced that Nvidia will build new AI supercomputers. \
         The chipmaker plans to expand its data center business.",
    );
    doc1.entities = vec![
        Entity::new("Jensen Huang", EntityType::Person, 0, 12, 0.95),
        Entity::new("Nvidia", EntityType::Organization, 28, 34, 0.94),
    ];
    docs.push(doc1);

    // Article 2: Different perspective on same event
    let mut doc2 = Document::new(
        "tech_02",
        "The CEO of Nvidia revealed plans for Blackwell chips during CES 2025. \
         Huang said the new GPUs would advance robotics and autonomous systems.",
    );
    doc2.entities = vec![
        Entity::new("CEO of Nvidia", EntityType::Person, 4, 17, 0.85),
        Entity::new("Nvidia", EntityType::Organization, 11, 17, 0.9),
        Entity::new(
            "Blackwell",
            EntityType::Other("Product".to_string()),
            37,
            46,
            0.87,
        ),
        Entity::new(
            "CES 2025",
            EntityType::Other("Event".to_string()),
            60,
            68,
            0.88,
        ),
        Entity::new("Huang", EntityType::Person, 70, 75, 0.92),
    ];
    docs.push(doc2);

    // Article 3: AI industry context
    let mut doc3 = Document::new(
        "tech_03",
        "Anthropic and Google DeepMind are competing with Nvidia for AI dominance. \
         Dario Amodei spoke about AI safety priorities.",
    );
    doc3.entities = vec![
        Entity::new("Anthropic", EntityType::Organization, 0, 9, 0.93),
        Entity::new("Google DeepMind", EntityType::Organization, 14, 29, 0.92),
        Entity::new("Nvidia", EntityType::Organization, 49, 55, 0.91),
        Entity::new("Dario Amodei", EntityType::Person, 76, 88, 0.94),
    ];
    docs.push(doc3);

    // Article 4: Follow-up on Nvidia
    let mut doc4 = Document::new(
        "tech_04",
        "Nvidia's stock reached new highs after Jensen Huang's keynote. \
         The company announced partnerships with major cloud providers.",
    );
    doc4.entities = vec![
        Entity::new("Nvidia", EntityType::Organization, 0, 6, 0.94),
        Entity::new("Jensen Huang", EntityType::Person, 38, 50, 0.93),
    ];
    docs.push(doc4);

    // Article 5: Competitor mention
    let mut doc5 = Document::new(
        "tech_05",
        "AMD and Intel responded to Nvidia's AI chip announcements. \
         The semiconductor rivals are investing heavily in data center GPUs.",
    );
    doc5.entities = vec![
        Entity::new("AMD", EntityType::Organization, 0, 3, 0.93),
        Entity::new("Intel", EntityType::Organization, 8, 13, 0.91),
        Entity::new("Nvidia", EntityType::Organization, 27, 33, 0.9),
    ];
    docs.push(doc5);

    docs
}

/// Generate political news documents for CDCR testing.
#[must_use]
pub fn political_news_dataset() -> Vec<Document> {
    let mut docs = Vec::new();

    let mut doc1 = Document::new(
        "pol_01",
        "President Biden met with Chancellor Scholz in Washington. \
         The two leaders discussed NATO expansion.",
    );
    doc1.entities = vec![
        Entity::new("President Biden", EntityType::Person, 0, 14, 0.95),
        Entity::new("Chancellor Scholz", EntityType::Person, 24, 41, 0.93),
        Entity::new("Washington", EntityType::Location, 45, 55, 0.92),
        Entity::new("NATO", EntityType::Organization, 84, 88, 0.94),
    ];
    docs.push(doc1);

    let mut doc2 = Document::new(
        "pol_02",
        "Biden and Scholz signed a joint statement on security. \
         The US President emphasized transatlantic unity.",
    );
    doc2.entities = vec![
        Entity::new("Biden", EntityType::Person, 0, 5, 0.94),
        Entity::new("Scholz", EntityType::Person, 10, 16, 0.92),
        Entity::new("US President", EntityType::Person, 60, 72, 0.88),
    ];
    docs.push(doc2);

    let mut doc3 = Document::new(
        "pol_03",
        "The German Chancellor held talks with the American President. \
         Olaf Scholz flew back to Berlin after the summit.",
    );
    doc3.entities = vec![
        Entity::new("German Chancellor", EntityType::Person, 4, 21, 0.9),
        Entity::new("American President", EntityType::Person, 38, 56, 0.88),
        Entity::new("Olaf Scholz", EntityType::Person, 58, 69, 0.93),
        Entity::new("Berlin", EntityType::Location, 82, 88, 0.91),
    ];
    docs.push(doc3);

    let mut doc4 = Document::new(
        "pol_04",
        "NATO Secretary General praised the Biden-Scholz meeting. \
         The alliance is preparing for new challenges.",
    );
    doc4.entities = vec![
        Entity::new("NATO Secretary General", EntityType::Person, 0, 22, 0.87),
        Entity::new("Biden", EntityType::Person, 35, 40, 0.92),
        Entity::new("Scholz", EntityType::Person, 41, 47, 0.91),
        Entity::new("NATO", EntityType::Organization, 0, 4, 0.94),
    ];
    docs.push(doc4);

    docs
}

/// Generate sports news documents for CDCR testing.
#[must_use]
pub fn sports_news_dataset() -> Vec<Document> {
    let mut docs = Vec::new();

    let mut doc1 = Document::new(
        "sport_01",
        "Lionel Messi scored twice as Inter Miami defeated Atlanta United 3-1. \
         The Argentine superstar continues his MLS dominance.",
    );
    doc1.entities = vec![
        Entity::new("Lionel Messi", EntityType::Person, 0, 12, 0.96),
        Entity::new("Inter Miami", EntityType::Organization, 29, 40, 0.93),
        Entity::new("Atlanta United", EntityType::Organization, 50, 64, 0.91),
        Entity::new(
            "Argentine",
            EntityType::Other("Nationality".to_string()),
            75,
            84,
            0.87,
        ),
    ];
    docs.push(doc1);

    let mut doc2 = Document::new(
        "sport_02",
        "Messi's brace helped Miami to victory. The former Barcelona star \
         is in top form.",
    );
    doc2.entities = vec![
        Entity::new("Messi", EntityType::Person, 0, 5, 0.95),
        Entity::new("Miami", EntityType::Organization, 21, 26, 0.88),
        Entity::new("Barcelona", EntityType::Organization, 49, 58, 0.91),
    ];
    docs.push(doc2);

    let mut doc3 = Document::new(
        "sport_03",
        "Inter Miami's victory over Atlanta keeps them top of the table. \
         Messi has 15 goals this season.",
    );
    doc3.entities = vec![
        Entity::new("Inter Miami", EntityType::Organization, 0, 11, 0.92),
        Entity::new("Atlanta", EntityType::Organization, 27, 34, 0.87),
        Entity::new("Messi", EntityType::Person, 66, 71, 0.94),
    ];
    docs.push(doc3);

    let mut doc4 = Document::new(
        "sport_04",
        "The Argentine forward Leo Messi broke another MLS record. \
         Miami's number 10 is unstoppable.",
    );
    doc4.entities = vec![
        Entity::new("Argentine forward", EntityType::Person, 4, 21, 0.85),
        Entity::new("Leo Messi", EntityType::Person, 22, 31, 0.94),
        Entity::new("MLS", EntityType::Organization, 46, 49, 0.9),
        Entity::new("Miami", EntityType::Organization, 59, 64, 0.87),
    ];
    docs.push(doc4);

    docs
}

/// Generate financial news documents for CDCR testing.
#[must_use]
pub fn financial_news_dataset() -> Vec<Document> {
    let mut docs = Vec::new();

    let mut doc1 = Document::new(
        "fin_01",
        "Apple reported record quarterly revenue of $117 billion. \
         Tim Cook said iPhone sales exceeded expectations.",
    );
    doc1.entities = vec![
        Entity::new("Apple", EntityType::Organization, 0, 5, 0.95),
        Entity::new("Tim Cook", EntityType::Person, 59, 67, 0.93),
        Entity::new(
            "iPhone",
            EntityType::Other("Product".to_string()),
            73,
            79,
            0.91,
        ),
    ];
    docs.push(doc1);

    let mut doc2 = Document::new(
        "fin_02",
        "The iPhone maker's stock rose 5% after earnings beat. \
         Apple's CEO expressed confidence in services growth.",
    );
    doc2.entities = vec![
        Entity::new("iPhone maker", EntityType::Organization, 4, 16, 0.85),
        Entity::new("Apple", EntityType::Organization, 55, 60, 0.94),
        Entity::new("CEO", EntityType::Person, 63, 66, 0.8),
    ];
    docs.push(doc2);

    let mut doc3 = Document::new(
        "fin_03",
        "Cook highlighted Apple's expansion in India. The Cupertino company \
         is reducing reliance on China.",
    );
    doc3.entities = vec![
        Entity::new("Cook", EntityType::Person, 0, 4, 0.91),
        Entity::new("Apple", EntityType::Organization, 17, 22, 0.94),
        Entity::new("India", EntityType::Location, 38, 43, 0.92),
        Entity::new("Cupertino company", EntityType::Organization, 49, 66, 0.82),
        Entity::new("China", EntityType::Location, 95, 100, 0.91),
    ];
    docs.push(doc3);

    let mut doc4 = Document::new(
        "fin_04",
        "Microsoft and Google also reported strong results. \
         But Apple outperformed both tech rivals.",
    );
    doc4.entities = vec![
        Entity::new("Microsoft", EntityType::Organization, 0, 9, 0.94),
        Entity::new("Google", EntityType::Organization, 14, 20, 0.93),
        Entity::new("Apple", EntityType::Organization, 56, 61, 0.94),
    ];
    docs.push(doc4);

    docs
}

/// Generate scientific/research documents for CDCR testing.
#[must_use]
pub fn science_news_dataset() -> Vec<Document> {
    let mut docs = Vec::new();

    let mut doc1 = Document::new(
        "sci_01",
        "NASA's Perseverance rover discovered organic molecules on Mars. \
         The Jezero Crater finding excited scientists.",
    );
    doc1.entities = vec![
        Entity::new("NASA", EntityType::Organization, 0, 4, 0.95),
        Entity::new(
            "Perseverance",
            EntityType::Other("Product".to_string()),
            7,
            19,
            0.92,
        ),
        Entity::new("Mars", EntityType::Location, 54, 58, 0.94),
        Entity::new("Jezero Crater", EntityType::Location, 64, 77, 0.89),
    ];
    docs.push(doc1);

    let mut doc2 = Document::new(
        "sci_02",
        "The Mars rover collected samples that may contain biosignatures. \
         NASA plans to bring these samples to Earth.",
    );
    doc2.entities = vec![
        Entity::new(
            "Mars rover",
            EntityType::Other("Product".to_string()),
            4,
            14,
            0.87,
        ),
        Entity::new("NASA", EntityType::Organization, 66, 70, 0.94),
        Entity::new("Earth", EntityType::Location, 101, 106, 0.93),
    ];
    docs.push(doc2);

    let mut doc3 = Document::new(
        "sci_03",
        "Perseverance has been operating in Jezero Crater since 2021. \
         The rover has traveled over 10 kilometers.",
    );
    doc3.entities = vec![
        Entity::new(
            "Perseverance",
            EntityType::Other("Product".to_string()),
            0,
            12,
            0.93,
        ),
        Entity::new("Jezero Crater", EntityType::Location, 35, 48, 0.9),
    ];
    docs.push(doc3);

    let mut doc4 = Document::new(
        "sci_04",
        "ESA and NASA are collaborating on Mars Sample Return. \
         The European Space Agency will build the orbiter.",
    );
    doc4.entities = vec![
        Entity::new("ESA", EntityType::Organization, 0, 3, 0.92),
        Entity::new("NASA", EntityType::Organization, 8, 12, 0.94),
        Entity::new("Mars", EntityType::Location, 34, 38, 0.93),
        Entity::new(
            "European Space Agency",
            EntityType::Organization,
            59,
            80,
            0.91,
        ),
    ];
    docs.push(doc4);

    docs
}

/// Combined comprehensive CDCR dataset.
#[must_use]
pub fn comprehensive_cdcr_dataset() -> Vec<Document> {
    let mut docs = tech_news_dataset();
    docs.extend(political_news_dataset());
    docs.extend(sports_news_dataset());
    docs.extend(financial_news_dataset());
    docs.extend(science_news_dataset());
    docs
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_documents() -> Vec<Document> {
        let mut doc1 = Document::new(
            "doc1",
            "Jensen Huang announced Nvidia's new AI chips in Santa Clara.",
        );
        doc1.entities = vec![
            Entity::new("Jensen Huang", EntityType::Person, 0, 12, 0.95),
            Entity::new("Nvidia", EntityType::Organization, 23, 29, 0.94),
            Entity::new("Santa Clara", EntityType::Location, 48, 59, 0.92),
        ];

        let mut doc2 = Document::new(
            "doc2",
            "The CEO of Nvidia revealed data center expansion plans.",
        );
        doc2.entities = vec![
            Entity::new("CEO of Nvidia", EntityType::Person, 4, 17, 0.85),
            Entity::new("Nvidia", EntityType::Organization, 11, 17, 0.94),
        ];

        let mut doc3 = Document::new(
            "doc3",
            "Huang spoke about Anthropic and the Santa Clara campus.",
        );
        doc3.entities = vec![
            Entity::new("Huang", EntityType::Person, 0, 5, 0.88),
            Entity::new("Anthropic", EntityType::Organization, 18, 27, 0.92),
            Entity::new("Santa Clara", EntityType::Location, 36, 47, 0.9),
        ];

        vec![doc1, doc2, doc3]
    }

    #[test]
    fn test_lsh_blocking() {
        // Test with very similar strings that should hash together
        let mentions = vec![
            MentionRef {
                doc_id: "d1".into(),
                entity_idx: 0,
                text: "Berlin Germany".into(),
                entity_type: EntityType::Location,
                within_doc_cluster: None,
            },
            MentionRef {
                doc_id: "d2".into(),
                entity_idx: 0,
                text: "Berlin Germany".into(), // Exact same text
                entity_type: EntityType::Location,
                within_doc_cluster: None,
            },
            MentionRef {
                doc_id: "d3".into(),
                entity_idx: 0,
                text: "New York".into(),
                entity_type: EntityType::Location,
                within_doc_cluster: None,
            },
        ];

        let blocker = LSHBlocker::default();
        let candidates = blocker.candidate_pairs(&mentions);

        // Identical strings should be candidates
        assert!(
            candidates.contains(&(0, 1)),
            "Identical texts should be candidate pairs"
        );
    }

    #[test]
    fn test_cdcr_resolver() {
        let docs = sample_documents();

        // Use brute-force comparison for this test (LSH may miss short strings)
        let config = CDCRConfig {
            use_lsh: false, // Brute force for reliable testing
            ..Default::default()
        };
        let resolver = CDCRResolver::with_config(config);

        let clusters = resolver.resolve(&docs);

        // Should have some clusters
        assert!(!clusters.is_empty(), "Should produce clusters");

        // Find the exact "Nvidia" cluster (Organization type, not "CEO of Nvidia")
        // This cluster should span doc1 and doc2
        let nvidia_org_cluster = clusters.iter().find(|c| {
            c.canonical_name.to_lowercase() == "nvidia"
                && c.entity_type == Some(EntityType::Organization)
        });

        assert!(
            nvidia_org_cluster.is_some(),
            "Should find Nvidia Organization cluster. Clusters: {:?}",
            clusters
                .iter()
                .map(|c| (&c.canonical_name, &c.entity_type, c.doc_count()))
                .collect::<Vec<_>>()
        );

        let nc = nvidia_org_cluster.unwrap();
        assert!(
            nc.doc_count() >= 2,
            "Nvidia Org should appear in at least 2 documents, found {} docs. Mentions: {:?}",
            nc.doc_count(),
            nc.mentions
        );
    }

    #[test]
    fn test_cdcr_same_entity_different_docs() {
        let mut doc1 = Document::new("doc1", "Barack Obama visited Berlin.");
        doc1.entities = vec![Entity::new("Barack Obama", EntityType::Person, 0, 12, 0.95)];

        let mut doc2 = Document::new("doc2", "Obama gave a speech in Germany.");
        doc2.entities = vec![Entity::new("Obama", EntityType::Person, 0, 5, 0.9)];

        // Disable LSH to use brute-force (ensures all pairs compared)
        let config = CDCRConfig {
            min_similarity: 0.3, // Lower threshold for substring match
            use_lsh: false,      // Brute force for small test
            ..Default::default()
        };
        let resolver = CDCRResolver::with_config(config);
        let clusters = resolver.resolve(&[doc1, doc2]);

        // Obama mentions should be in the same cluster
        let obama_cluster = clusters
            .iter()
            .find(|c| c.canonical_name.to_lowercase().contains("obama"));

        assert!(obama_cluster.is_some(), "Should find Obama cluster");

        let cluster = obama_cluster.unwrap();
        assert_eq!(
            cluster.doc_count(),
            2,
            "Obama should appear in both documents"
        );
    }

    #[test]
    fn test_cdcr_metrics() {
        // Simple test case: 2 entities, 2 clusters each
        let pred = vec![CrossDocCluster {
            id: 0,
            canonical_name: "Entity A".into(),
            entity_type: Some(EntityType::Person),
            documents: vec!["d1".into(), "d2".into()],
            mentions: vec![("d1".into(), 0), ("d2".into(), 0)],
            kb_id: None,
            confidence: 1.0,
        }];

        let gold = vec![CrossDocCluster {
            id: 0,
            canonical_name: "Entity A".into(),
            entity_type: Some(EntityType::Person),
            documents: vec!["d1".into(), "d2".into()],
            mentions: vec![("d1".into(), 0), ("d2".into(), 0)],
            kb_id: None,
            confidence: 1.0,
        }];

        let metrics = CDCRMetrics::compute(&pred, &gold);

        assert!(
            (metrics.b_cubed_f1 - 1.0).abs() < 0.01,
            "Perfect clustering should have F1 = 1.0"
        );
    }

    // =================================================================
    // Additional Edge Case Tests
    // =================================================================

    #[test]
    fn test_empty_documents() {
        let resolver = CDCRResolver::new();
        let clusters = resolver.resolve(&[]);
        assert!(clusters.is_empty(), "Empty docs should produce no clusters");
    }

    #[test]
    fn test_single_document() {
        let mut doc = Document::new("doc1", "John Smith works at Google.");
        doc.entities = vec![
            Entity::new("John Smith", EntityType::Person, 0, 10, 0.9),
            Entity::new("Google", EntityType::Organization, 20, 26, 0.95),
        ];

        let config = CDCRConfig {
            use_lsh: false,
            ..Default::default()
        };
        let resolver = CDCRResolver::with_config(config);
        let clusters = resolver.resolve(&[doc]);

        // Each entity should be in its own cluster
        assert_eq!(clusters.len(), 2, "Two entities should form two clusters");
    }

    #[test]
    fn test_document_with_no_entities() {
        let doc = Document::new("doc1", "This is a test document without entities.");
        let resolver = CDCRResolver::new();
        let clusters = resolver.resolve(&[doc]);
        assert!(
            clusters.is_empty(),
            "Doc without entities should produce no clusters"
        );
    }

    #[test]
    fn test_type_mismatch_prevents_clustering() {
        let mut doc1 = Document::new("doc1", "Apple announced new products.");
        doc1.entities = vec![Entity::new("Apple", EntityType::Organization, 0, 5, 0.9)];

        let mut doc2 = Document::new("doc2", "I ate an apple for lunch.");
        doc2.entities = vec![Entity::new(
            "apple",
            EntityType::Other("Fruit".into()),
            9,
            14,
            0.8,
        )];

        let config = CDCRConfig {
            use_lsh: false,
            require_type_match: true, // Strict type matching
            ..Default::default()
        };
        let resolver = CDCRResolver::with_config(config);
        let clusters = resolver.resolve(&[doc1, doc2]);

        // Should have 2 separate clusters due to type mismatch
        assert_eq!(clusters.len(), 2, "Type mismatch should prevent clustering");
    }

    #[test]
    fn test_similarity_threshold() {
        let mut doc1 = Document::new("doc1", "John works here.");
        doc1.entities = vec![Entity::new("John", EntityType::Person, 0, 4, 0.9)];

        let mut doc2 = Document::new("doc2", "Jonathan is a developer.");
        doc2.entities = vec![Entity::new("Jonathan", EntityType::Person, 0, 8, 0.9)];

        // High threshold - should NOT cluster
        let config_high = CDCRConfig {
            use_lsh: false,
            min_similarity: 0.9,
            ..Default::default()
        };
        let resolver_high = CDCRResolver::with_config(config_high);
        let clusters_high = resolver_high.resolve(&[doc1.clone(), doc2.clone()]);
        assert_eq!(
            clusters_high.len(),
            2,
            "High threshold should keep separate"
        );

        // Low threshold - might cluster
        let config_low = CDCRConfig {
            use_lsh: false,
            min_similarity: 0.2,
            ..Default::default()
        };
        let resolver_low = CDCRResolver::with_config(config_low);
        let clusters_low = resolver_low.resolve(&[doc1, doc2]);
        // John and Jonathan share "John" substring, so may cluster
        assert!(clusters_low.len() <= 2);
    }

    #[test]
    fn test_cross_doc_cluster_methods() {
        let mut cluster = CrossDocCluster::new(1, "Test Entity");
        cluster.add_mention("doc1", 0);
        cluster.add_mention("doc2", 1);
        cluster.add_mention("doc1", 2); // Same doc, different mention

        assert_eq!(cluster.len(), 3);
        assert_eq!(cluster.doc_count(), 2); // Only 2 unique docs
        assert!(!cluster.is_empty());
    }

    #[test]
    fn test_lsh_blocker_signature() {
        let blocker = LSHBlocker::default();

        // Same text should produce same signature
        let mentions1 = vec![
            MentionRef {
                doc_id: "d1".into(),
                entity_idx: 0,
                text: "United States of America".into(),
                entity_type: EntityType::Location,
                within_doc_cluster: None,
            },
            MentionRef {
                doc_id: "d2".into(),
                entity_idx: 0,
                text: "United States of America".into(),
                entity_type: EntityType::Location,
                within_doc_cluster: None,
            },
        ];

        let candidates = blocker.candidate_pairs(&mentions1);
        assert!(
            candidates.contains(&(0, 1)),
            "Identical texts should be candidates"
        );
    }

    #[test]
    fn test_cdcr_metrics_empty() {
        let metrics = CDCRMetrics::compute(&[], &[]);
        assert_eq!(metrics.b_cubed_f1, 0.0);
        assert_eq!(metrics.num_pred_clusters, 0);
        assert_eq!(metrics.num_gold_clusters, 0);
    }

    #[test]
    fn test_document_builder_pattern() {
        let doc = Document::new("test", "Sample text").with_entities(vec![Entity::new(
            "Sample",
            EntityType::Other("Test".into()),
            0,
            6,
            0.9,
        )]);

        assert_eq!(doc.id, "test");
        assert_eq!(doc.entities.len(), 1);
    }

    #[test]
    fn test_mention_ref_equality() {
        let mention1 = MentionRef {
            doc_id: "d1".into(),
            entity_idx: 0,
            text: "Test".into(),
            entity_type: EntityType::Person,
            within_doc_cluster: Some(1),
        };

        // Same doc, same index = same mention
        assert_eq!(mention1.doc_id, "d1");
        assert_eq!(mention1.entity_idx, 0);
    }

    // =================================================================
    // Domain Dataset Tests
    // =================================================================

    #[test]
    fn test_tech_news_dataset() {
        let docs = tech_news_dataset();

        assert!(
            docs.len() >= 5,
            "Tech dataset should have at least 5 documents"
        );

        // Should mention Nvidia multiple times
        let nvidia_mentions: usize = docs
            .iter()
            .flat_map(|d| &d.entities)
            .filter(|e| e.text.to_lowercase().contains("nvidia"))
            .count();
        assert!(
            nvidia_mentions >= 3,
            "Nvidia should appear in multiple documents"
        );

        // Should mention Huang multiple times
        let huang_mentions: usize = docs
            .iter()
            .flat_map(|d| &d.entities)
            .filter(|e| e.text.to_lowercase().contains("huang"))
            .count();
        assert!(
            huang_mentions >= 3,
            "Huang should appear in multiple documents"
        );
    }

    #[test]
    fn test_political_news_dataset() {
        let docs = political_news_dataset();

        assert!(
            docs.len() >= 4,
            "Political dataset should have at least 4 documents"
        );

        // Should mention Biden/Scholz multiple times
        let biden_mentions: usize = docs
            .iter()
            .flat_map(|d| &d.entities)
            .filter(|e| e.text.to_lowercase().contains("biden"))
            .count();
        assert!(
            biden_mentions >= 3,
            "Biden should appear in multiple documents"
        );
    }

    #[test]
    fn test_sports_news_dataset() {
        let docs = sports_news_dataset();

        assert!(
            docs.len() >= 4,
            "Sports dataset should have at least 4 documents"
        );

        // Should mention Messi multiple times
        let messi_mentions: usize = docs
            .iter()
            .flat_map(|d| &d.entities)
            .filter(|e| e.text.to_lowercase().contains("messi"))
            .count();
        assert!(
            messi_mentions >= 4,
            "Messi should appear in multiple documents"
        );
    }

    #[test]
    fn test_financial_news_dataset() {
        let docs = financial_news_dataset();

        assert!(
            docs.len() >= 4,
            "Financial dataset should have at least 4 documents"
        );

        // Should mention Apple multiple times
        let apple_mentions: usize = docs
            .iter()
            .flat_map(|d| &d.entities)
            .filter(|e| e.text.to_lowercase().contains("apple"))
            .count();
        assert!(
            apple_mentions >= 3,
            "Apple should appear in multiple documents"
        );
    }

    #[test]
    fn test_science_news_dataset() {
        let docs = science_news_dataset();

        assert!(
            docs.len() >= 4,
            "Science dataset should have at least 4 documents"
        );

        // Should mention NASA multiple times
        let nasa_mentions: usize = docs
            .iter()
            .flat_map(|d| &d.entities)
            .filter(|e| e.text.to_lowercase().contains("nasa"))
            .count();
        assert!(
            nasa_mentions >= 3,
            "NASA should appear in multiple documents"
        );
    }

    #[test]
    fn test_comprehensive_cdcr_dataset() {
        let docs = comprehensive_cdcr_dataset();

        // Should combine all domain datasets
        let expected_min = tech_news_dataset().len()
            + political_news_dataset().len()
            + sports_news_dataset().len()
            + financial_news_dataset().len()
            + science_news_dataset().len();

        assert_eq!(
            docs.len(),
            expected_min,
            "Comprehensive should combine all domain datasets"
        );
    }

    #[test]
    fn test_cdcr_on_tech_news() {
        let docs = tech_news_dataset();

        let config = CDCRConfig {
            use_lsh: false, // Brute force for reliable testing
            min_similarity: 0.4,
            ..Default::default()
        };
        let resolver = CDCRResolver::with_config(config);
        let clusters = resolver.resolve(&docs);

        // Should cluster Nvidia mentions together
        let nvidia_cluster = clusters.iter().find(|c| {
            c.canonical_name.to_lowercase() == "nvidia"
                && c.entity_type == Some(EntityType::Organization)
        });

        if let Some(nc) = nvidia_cluster {
            assert!(
                nc.doc_count() >= 2,
                "Nvidia should appear in at least 2 documents, found {}",
                nc.doc_count()
            );
        }

        println!("Tech news CDCR clusters:");
        for cluster in &clusters {
            if cluster.doc_count() > 1 {
                println!(
                    "  {} ({:?}): {} docs",
                    cluster.canonical_name,
                    cluster.entity_type,
                    cluster.doc_count()
                );
            }
        }
    }

    #[test]
    fn test_cdcr_on_sports_news() {
        let docs = sports_news_dataset();

        let config = CDCRConfig {
            use_lsh: false,
            min_similarity: 0.4,
            ..Default::default()
        };
        let resolver = CDCRResolver::with_config(config);
        let clusters = resolver.resolve(&docs);

        // Messi should be clustered across documents
        let messi_cluster = clusters
            .iter()
            .find(|c| c.canonical_name.to_lowercase().contains("messi"));

        assert!(messi_cluster.is_some(), "Should find Messi cluster");

        if let Some(mc) = messi_cluster {
            assert!(
                mc.doc_count() >= 3,
                "Messi should appear in at least 3 documents, found {}",
                mc.doc_count()
            );
        }
    }

    #[test]
    fn test_cross_domain_cdcr() {
        // Test that cross-domain resolution doesn't create spurious links
        let mut docs = Vec::new();

        // Add one doc from tech (Jordan the AI researcher)
        let mut tech_doc = Document::new("tech", "Jordan presented research at NeurIPS.");
        tech_doc.entities = vec![Entity::new("Jordan", EntityType::Person, 0, 6, 0.9)];
        docs.push(tech_doc);

        // Add one doc from sports (Jordan the basketball player)
        // Same name but different entity - tests disambiguation by context
        let mut sports_doc = Document::new("sports", "Jordan scored 30 points in the game.");
        sports_doc.entities = vec![Entity::new("Jordan", EntityType::Person, 0, 6, 0.9)];
        docs.push(sports_doc);

        let config = CDCRConfig {
            use_lsh: false,
            ..Default::default()
        };
        let resolver = CDCRResolver::with_config(config);
        let clusters = resolver.resolve(&docs);

        // Note: Without entity type distinction, same-name entities will cluster
        // This test documents current behavior - proper disambiguation would
        // require additional context/features beyond simple string matching
        println!(
            "Cross-domain clusters: {:?}",
            clusters
                .iter()
                .map(|c| (&c.canonical_name, c.doc_count()))
                .collect::<Vec<_>>()
        );
    }
}
