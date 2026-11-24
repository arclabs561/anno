//! Cross-Context Coreference Evaluation Harness.
//!
//! Evaluation framework for xCoRe-style cross-context coreference resolution,
//! supporting both long-document and cross-document benchmarks.
//!
//! # Supported Benchmarks
//!
//! This harness is intended for ECB+, SciCo, LitBank, BookCoref, and similar
//! long-document / cross-document benchmarks.
//!
//! # Usage
//!
//! ```rust,ignore
//! use anno_eval::eval::cross_context_eval::{CrossContextBenchmark, evaluate_benchmark};
//! use anno::metrics::cluster_encoder::{HeuristicClusterEncoder, CosineMergeScorer};
//!
//! let encoder = HeuristicClusterEncoder::new(64);
//! let scorer = CosineMergeScorer::new();
//!
//! let results = evaluate_benchmark(
//!     CrossContextBenchmark::ECBPlus,
//!     &encoder,
//!     &scorer,
//!     None, // Use default config
//! )?;
//!
//! println!("CoNLL F1: {:.1}", results.conll_f1 * 100.0);
//! ```
//!
//! # References
//!
//! - Martinelli et al. (2025): "xCoRe: Cross-context Coreference Resolution"
//! - Cybulska & Vossen (2014): "ECB+ Event Coreference Bank"
//! - Cattan et al. (2021): "SciCo Hierarchical Cross-Document Coreference"
//! - Bamman et al. (2020): "LitBank"
//! - Martinelli et al. (2025): "BOOKCOREF: Coreference Resolution at Book Scale"
//! - Guo et al. (2023): "Animal Farm annotation"

use crate::eval::cdcr::{CrossDocCluster, Document};
use crate::eval::cluster_encoder::{ClusterEncoder, MergeScorer};
use crate::eval::coref::{offset_chains, CorefChain, Mention};
use crate::eval::coref_metrics::{conll_f1, CorefScores};
use crate::eval::neural_cluster_encoder::{
    CrossContextConfig, UnifiedCrossContextResolver, WindowOutput,
};
use crate::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

// =============================================================================
// Benchmark Definitions
// =============================================================================

/// Cross-context coreference benchmarks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CrossContextBenchmark {
    /// ECB+ - Cross-document entity/event coreference (news)
    ECBPlus,
    /// SciCo - Cross-document concept coreference (scientific papers)
    SciCo,
    /// LitBank - Long-document coreference (literary fiction)
    LitBank,
    /// BookCoref - Full-book coreference (book-scale)
    BookCoref,
    /// Animal Farm - Single long novel benchmark
    AnimalFarm,
}

impl CrossContextBenchmark {
    /// Get benchmark name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::ECBPlus => "ECB+",
            Self::SciCo => "SciCo",
            Self::LitBank => "LitBank",
            Self::BookCoref => "BookCoref",
            Self::AnimalFarm => "Animal Farm",
        }
    }

    /// Is this a cross-document benchmark?
    pub fn is_cross_document(&self) -> bool {
        matches!(self, Self::ECBPlus | Self::SciCo)
    }

    /// Is this a long-document benchmark?
    pub fn is_long_document(&self) -> bool {
        matches!(self, Self::LitBank | Self::BookCoref | Self::AnimalFarm)
    }

    /// Get recommended window size for this benchmark.
    pub fn recommended_window_size(&self) -> usize {
        match self {
            Self::ECBPlus => 512,     // Documents are short
            Self::SciCo => 512,       // Paper sections
            Self::LitBank => 2000,    // Limited to 2k tokens
            Self::BookCoref => 4000,  // Full books
            Self::AnimalFarm => 4000, // Single long novel
        }
    }

    /// State-of-the-art CoNLL F1 from xCoRe (Martinelli et al. 2025, Table 3).
    pub fn xcore_sota_f1(&self) -> f64 {
        match self {
            Self::ECBPlus => 40.3,
            Self::SciCo => 34.5,
            Self::LitBank => 78.2,
            Self::BookCoref => 65.0,
            Self::AnimalFarm => 70.0,
        }
    }

    /// Get all benchmarks.
    pub fn all() -> &'static [Self] {
        &[
            Self::ECBPlus,
            Self::SciCo,
            Self::LitBank,
            Self::BookCoref,
            Self::AnimalFarm,
        ]
    }

    /// Get cross-document benchmarks.
    pub fn cross_document() -> &'static [Self] {
        &[Self::ECBPlus, Self::SciCo]
    }

    /// Get long-document benchmarks.
    pub fn long_document() -> &'static [Self] {
        &[Self::LitBank, Self::BookCoref, Self::AnimalFarm]
    }
}

// =============================================================================
// Evaluation Configuration
// =============================================================================

/// Configuration for cross-context evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossContextEvalConfig {
    /// Window size for long-document processing
    pub window_size: usize,
    /// Window overlap
    pub window_overlap: usize,
    /// Minimum merge score on the scale produced by the chosen `MergeScorer`.
    pub merge_threshold: f32,
    /// Whether to use gold mentions (vs predicted)
    pub use_gold_mentions: bool,
    /// Whether to use gold within-context clusters
    pub use_gold_clusters: bool,
    /// Maximum documents per topic, in supplied order. Zero includes every document.
    pub max_docs_per_topic: usize,
    /// Reserved for sampling. Cross-document caps preserve the supplied order.
    pub seed: u64,
}

impl Default for CrossContextEvalConfig {
    fn default() -> Self {
        Self {
            window_size: 4000,
            window_overlap: 256,
            merge_threshold: 0.5,
            use_gold_mentions: false,
            use_gold_clusters: false,
            max_docs_per_topic: 0,
            seed: 42,
        }
    }
}

impl CrossContextEvalConfig {
    /// Create config for a specific benchmark.
    pub fn for_benchmark(benchmark: CrossContextBenchmark) -> Self {
        Self {
            window_size: benchmark.recommended_window_size(),
            ..Default::default()
        }
    }

    /// Config for oracle evaluation (gold mentions + gold clusters).
    pub fn oracle() -> Self {
        Self {
            use_gold_mentions: true,
            use_gold_clusters: true,
            ..Default::default()
        }
    }

    /// Config for predicted mentions, gold clusters.
    pub fn gold_clusters() -> Self {
        Self {
            use_gold_mentions: false,
            use_gold_clusters: true,
            ..Default::default()
        }
    }
}

// =============================================================================
// Evaluation Results
// =============================================================================

/// Results from cross-context evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossContextEvalResults {
    /// Benchmark name
    pub benchmark: String,
    /// Configuration used
    pub config: CrossContextEvalConfig,
    /// MUC scores
    pub muc: CorefScores,
    /// B³ scores
    pub b_cubed: CorefScores,
    /// CEAF-e scores
    pub ceaf_e: CorefScores,
    /// LEA scores
    pub lea: CorefScores,
    /// CoNLL F1 (average of MUC, B³, CEAF-e)
    pub conll_f1: f64,
    /// Number of documents/contexts evaluated
    pub num_contexts: usize,
    /// Number of gold clusters
    pub num_gold_clusters: usize,
    /// Number of predicted clusters
    pub num_pred_clusters: usize,
    /// Average cluster size
    pub avg_cluster_size: f64,
    /// Processing time in milliseconds
    pub time_ms: f64,
    /// Per-topic results (for cross-document)
    pub per_topic: Option<HashMap<String, TopicResults>>,
    /// Per-document results (for long-document)
    pub per_document: Option<HashMap<String, DocumentResults>>,
}

impl CrossContextEvalResults {
    /// Format as summary string.
    pub fn summary(&self) -> String {
        format!(
            "{}: CoNLL F1 = {:.1}% (MUC: {:.1}, B³: {:.1}, CEAF: {:.1})",
            self.benchmark,
            self.conll_f1 * 100.0,
            self.muc.f1 * 100.0,
            self.b_cubed.f1 * 100.0,
            self.ceaf_e.f1 * 100.0,
        )
    }
}

/// Per-topic results for cross-document evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicResults {
    /// Topic ID
    pub topic_id: String,
    /// Number of documents in topic
    pub num_documents: usize,
    /// CoNLL F1 for this topic
    pub conll_f1: f64,
    /// Number of gold clusters
    pub num_gold_clusters: usize,
    /// Number of predicted clusters
    pub num_pred_clusters: usize,
}

/// Per-document results for long-document evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentResults {
    /// Document ID
    pub doc_id: String,
    /// Document length in tokens
    pub num_tokens: usize,
    /// Number of windows
    pub num_windows: usize,
    /// CoNLL F1 for this document
    pub conll_f1: f64,
    /// Number of gold chains
    pub num_gold_chains: usize,
    /// Number of predicted chains
    pub num_pred_chains: usize,
}

// =============================================================================
// Example Data Structures
// =============================================================================

/// A topic containing multiple documents (for cross-document evaluation).
#[derive(Debug, Clone)]
pub struct Topic {
    /// Topic ID
    pub id: String,
    /// Documents in this topic
    pub documents: Vec<Document>,
    /// Gold cross-document clusters
    pub gold_clusters: Vec<CrossDocCluster>,
}

impl Topic {
    /// Create a new topic.
    pub fn new(id: &str) -> Self {
        Self {
            id: id.to_string(),
            documents: Vec::new(),
            gold_clusters: Vec::new(),
        }
    }

    /// Add a document.
    pub fn add_document(&mut self, doc: Document) {
        self.documents.push(doc);
    }

    /// Add a gold cluster.
    pub fn add_gold_cluster(&mut self, cluster: CrossDocCluster) {
        self.gold_clusters.push(cluster);
    }
    /// Restrict both sides of the evaluation to the first documents in input order.
    fn with_document_limit(&self, limit: usize) -> Result<std::borrow::Cow<'_, Self>> {
        if limit == 0 || self.documents.len() <= limit {
            return Ok(std::borrow::Cow::Borrowed(self));
        }
        let mut doc_indices = HashMap::new();
        for (index, doc) in self.documents.iter().enumerate() {
            if doc_indices.insert(doc.id.as_str(), index).is_some() {
                return Err(crate::Error::InvalidInput(format!(
                    "Duplicate document {} in topic {}",
                    doc.id, self.id
                )));
            }
        }
        let mut selected = Self::new(&self.id);
        selected.documents = self.documents[..limit].to_vec();
        for gold in &self.gold_clusters {
            let mut group = CrossDocCluster::new(gold.id, &gold.canonical_name);
            for (doc, entity) in &gold.mentions {
                let index = *doc_indices.get(doc.as_str()).ok_or_else(|| {
                    crate::Error::InvalidInput(format!(
                        "Cluster {} refers to unknown document {doc}",
                        gold.id
                    ))
                })?;
                if index < limit {
                    group.add_mention(doc, *entity);
                }
            }
            if !group.mentions.is_empty() {
                selected.add_gold_cluster(group);
            }
        }
        Ok(std::borrow::Cow::Owned(selected))
    }
}

/// A long document with gold annotations (for long-document evaluation).
#[derive(Debug, Clone)]
pub struct LongDocument {
    /// Document ID
    pub id: String,
    /// Full text
    pub text: String,
    /// Gold coreference chains
    pub gold_chains: Vec<CorefChain>,
    /// Optional: pre-computed windows
    pub windows: Option<Vec<WindowOutput>>,
}

impl LongDocument {
    /// Create a new long document.
    pub fn new(id: &str, text: &str, gold_chains: Vec<CorefChain>) -> Self {
        Self {
            id: id.to_string(),
            text: text.to_string(),
            gold_chains,
            windows: None,
        }
    }

    /// Get text length in characters.
    pub fn char_len(&self) -> usize {
        self.text.chars().count()
    }

    /// Estimate token count (rough: chars / 5).
    pub fn approx_tokens(&self) -> usize {
        self.text.len() / 5
    }
}

// =============================================================================
// Evaluation Functions
// =============================================================================

/// Counts of unordered, same-cluster mention pairs drawn from different documents.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CrossDocumentLinkCounts {
    /// Pairs in both partitions.
    pub correct: u64,
    /// Pairs in the reference partition (recall denominator).
    pub gold: u64,
    /// Pairs in the predicted partition (precision denominator).
    pub predicted: u64,
}

fn cross_document_pair_count<'a, K: Copy + Eq + Hash>(
    mentions: impl Iterator<Item = (K, &'a str)>,
) -> u64 {
    let mut totals = HashMap::<K, u64>::new();
    let mut within_document = HashMap::<(K, &str), u64>::new();
    let mut pairs = 0;
    for (cluster, doc) in mentions {
        let total = totals.entry(cluster).or_default();
        let local = within_document.entry((cluster, doc)).or_default();
        pairs += *total - *local;
        *total += 1;
        *local += 1;
    }
    pairs
}

fn cluster_members(clusters: &[CrossDocCluster]) -> impl Iterator<Item = (usize, (&str, usize))> {
    clusters.iter().enumerate().flat_map(|(cluster, c)| {
        c.mentions
            .iter()
            .map(move |(doc, idx)| (cluster, (doc.as_str(), *idx)))
    })
}

impl CrossDocumentLinkCounts {
    fn add_topic(&mut self, gold: &[CrossDocCluster], predicted: &[CrossDocCluster]) {
        let membership: HashMap<_, _> = cluster_members(gold)
            .map(|(group, key)| (key, group))
            .collect();
        self.gold +=
            cross_document_pair_count(membership.iter().map(|(&(doc, _), &group)| (group, doc)));
        self.predicted += cross_document_pair_count(
            cluster_members(predicted).map(|(group, (doc, _))| (group, doc)),
        );
        // Only mentions grouped together by both partitions can form a correct link.
        self.correct += cross_document_pair_count(
            cluster_members(predicted)
                .filter_map(|(pred, key)| membership.get(&key).map(|&gold| ((pred, gold), key.0))),
        );
    }
}

/// Evaluate cross-document coreference on a set of topics.
///
/// Resolves the spans supplied in each document with `UnifiedCrossContextResolver`.
/// With `use_gold_mentions`, the caller must supply exactly the reference mentions.
/// With `use_gold_clusters`, local seeds come from the document restriction of each
/// reference cluster. Otherwise, the resolver uses each document's supplied local chains.
pub fn evaluate_cross_document<E: ClusterEncoder + Clone, S: MergeScorer + Clone>(
    topics: &[Topic],
    encoder: E,
    scorer: S,
    config: &CrossContextEvalConfig,
) -> Result<CrossContextEvalResults> {
    evaluate_cross_document_with_links(topics, encoder, scorer, config).map(|(scores, _)| scores)
}

/// Evaluate the same partition metrics and report link support strictly between documents.
pub fn evaluate_cross_document_with_links<E: ClusterEncoder + Clone, S: MergeScorer + Clone>(
    topics: &[Topic],
    encoder: E,
    scorer: S,
    config: &CrossContextEvalConfig,
) -> Result<(CrossContextEvalResults, CrossDocumentLinkCounts)> {
    let start = std::time::Instant::now();
    let mut links = CrossDocumentLinkCounts::default();

    let resolver_config = CrossContextConfig {
        window_size: config.window_size,
        window_overlap: config.window_overlap,
        merge_threshold: config.merge_threshold,
    };

    let resolver = UnifiedCrossContextResolver::new(encoder, scorer, resolver_config);

    let mut all_gold_chains: Vec<CorefChain> = Vec::new();
    let mut all_pred_chains: Vec<CorefChain> = Vec::new();
    let mut per_topic = HashMap::new();
    let mut num_contexts = 0;
    let mut total_gold_clusters = 0;
    let mut total_pred_clusters = 0;

    // The metric functions identify mentions by offsets. Place each document in
    // a disjoint range, including documents with the same ID in different topics.
    let mut next_offset = 0;
    for source in topics {
        let topic = source.with_document_limit(config.max_docs_per_topic)?;
        num_contexts += topic.documents.len();
        if per_topic.contains_key(&topic.id) {
            return Err(crate::Error::InvalidInput(format!(
                "Duplicate topic ID {}",
                topic.id
            )));
        }
        let mut documents = HashMap::new();
        for doc in &topic.documents {
            if documents
                .insert(doc.id.as_str(), (doc, next_offset))
                .is_some()
            {
                return Err(crate::Error::InvalidInput(format!(
                    "Duplicate document {} in topic {}",
                    doc.id, topic.id
                )));
            }
            let len = doc.text.chars().count();
            let mut spans = HashSet::new();
            if doc
                .entities
                .iter()
                .any(|e| !spans.insert((e.start(), e.end())))
            {
                return Err(crate::Error::InvalidInput(format!(
                    "Ambiguous entity span in document {}",
                    doc.id
                )));
            }
            if doc
                .entities
                .iter()
                .any(|e| e.start() > e.end() || e.end() > len)
            {
                return Err(crate::Error::InvalidInput(format!(
                    "Invalid entity span in document {}",
                    doc.id
                )));
            }
            next_offset += len + 1;
        }
        let topic_gold_chains = cross_doc_clusters_to_chains(&topic.gold_clusters, &documents)?;
        total_gold_clusters += topic.gold_clusters.len();

        if config.use_gold_mentions
            && topic_gold_chains.iter().map(CorefChain::len).sum::<usize>()
                != topic
                    .documents
                    .iter()
                    .map(|doc| doc.entities.len())
                    .sum::<usize>()
        {
            return Err(crate::Error::InvalidInput(
                "Gold-mention evaluation requires each supplied entity to belong to a gold cluster"
                    .into(),
            ));
        }

        let pred_clusters = if config.use_gold_clusters {
            // Keep mention indices stable. Only the resolver's within-document seeds change.
            let mut local: HashMap<&str, Vec<CorefChain>> = HashMap::new();
            for group in &topic.gold_clusters {
                let mut by_doc: HashMap<&str, Vec<Mention>> = HashMap::new();
                for (doc_id, idx) in &group.mentions {
                    let entity = &documents[doc_id.as_str()].0.entities[*idx];
                    by_doc.entry(doc_id).or_default().push(Mention::new(
                        &entity.text,
                        entity.start(),
                        entity.end(),
                    ));
                }
                for (doc, mentions) in by_doc {
                    local
                        .entry(doc)
                        .or_default()
                        .push(CorefChain::new(mentions));
                }
            }
            let mut seeded = topic.documents.clone();
            for doc in &mut seeded {
                doc.coref_chains = local.remove(doc.id.as_str()).unwrap_or_default();
            }
            resolver.resolve_documents(&seeded)
        } else {
            resolver.resolve_documents(&topic.documents)
        };
        total_pred_clusters += pred_clusters.len();

        let topic_pred_chains = cross_doc_clusters_to_chains(&pred_clusters, &documents)?;
        links.add_topic(&topic.gold_clusters, &pred_clusters);

        // Compute per-topic metrics
        let topic_f1 = conll_f1(&topic_pred_chains, &topic_gold_chains);

        per_topic.insert(
            topic.id.clone(),
            TopicResults {
                topic_id: topic.id.clone(),
                num_documents: topic.documents.len(),
                conll_f1: topic_f1,
                num_gold_clusters: topic.gold_clusters.len(),
                num_pred_clusters: pred_clusters.len(),
            },
        );

        all_gold_chains.extend(topic_gold_chains);
        all_pred_chains.extend(topic_pred_chains);
    }

    // Compute aggregate metrics
    let (muc_p, muc_r, muc_f1) =
        crate::eval::coref_metrics::muc_score(&all_pred_chains, &all_gold_chains);
    let (b3_p, b3_r, b3_f1) =
        crate::eval::coref_metrics::b_cubed_score(&all_pred_chains, &all_gold_chains);
    let (ceaf_p, ceaf_r, ceaf_f1) =
        crate::eval::coref_metrics::ceaf_e_score(&all_pred_chains, &all_gold_chains);
    let (lea_p, lea_r, lea_f1) =
        crate::eval::coref_metrics::lea_score(&all_pred_chains, &all_gold_chains);
    let conll = (muc_f1 + b3_f1 + ceaf_f1) / 3.0;

    let total_mentions: usize = all_pred_chains.iter().map(|c| c.len()).sum();
    let avg_cluster_size = if !all_pred_chains.is_empty() {
        total_mentions as f64 / all_pred_chains.len() as f64
    } else {
        0.0
    };

    Ok((
        CrossContextEvalResults {
            benchmark: "Cross-Document".to_string(),
            config: config.clone(),
            muc: CorefScores::from_tuple((muc_p, muc_r, muc_f1)),
            b_cubed: CorefScores::from_tuple((b3_p, b3_r, b3_f1)),
            ceaf_e: CorefScores::from_tuple((ceaf_p, ceaf_r, ceaf_f1)),
            lea: CorefScores::from_tuple((lea_p, lea_r, lea_f1)),
            conll_f1: conll,
            num_contexts,
            num_gold_clusters: total_gold_clusters,
            num_pred_clusters: total_pred_clusters,
            avg_cluster_size,
            time_ms: start.elapsed().as_millis() as f64,
            per_topic: Some(per_topic),
            per_document: None,
        },
        links,
    ))
}

/// Evaluate long-document coreference.
///
/// Uses the `UnifiedCrossContextResolver` to merge clusters across windows.
pub fn evaluate_long_document<E: ClusterEncoder + Clone, S: MergeScorer + Clone>(
    documents: &[LongDocument],
    encoder: E,
    scorer: S,
    config: &CrossContextEvalConfig,
) -> Result<CrossContextEvalResults> {
    let start = std::time::Instant::now();

    let resolver_config = CrossContextConfig {
        window_size: config.window_size,
        window_overlap: config.window_overlap,
        merge_threshold: config.merge_threshold,
    };

    let resolver = UnifiedCrossContextResolver::new(encoder, scorer, resolver_config);

    let mut all_gold_chains: Vec<CorefChain> = Vec::new();
    let mut all_pred_chains: Vec<CorefChain> = Vec::new();
    let mut per_document = HashMap::new();

    let mut doc_base = 0usize;
    for doc in documents {
        let len = doc.char_len();
        let fallback;
        let windows = match doc.windows.as_deref() {
            Some(windows) if !windows.is_empty() => windows,
            _ => {
                // Without precomputed windows or gold mentions, this harness has no mention detector.
                let chains = if config.use_gold_mentions {
                    doc.gold_chains.clone()
                } else {
                    Vec::new()
                };
                fallback = [WindowOutput::new(0, 0, len, chains)];
                &fallback
            }
        };
        let pred_chains = resolver.resolve_long_document_windows(windows);
        if doc
            .gold_chains
            .iter()
            .chain(&pred_chains)
            .flat_map(|chain| &chain.mentions)
            .any(|mention| mention.start > mention.end || mention.end > len)
        {
            return Err(crate::Error::InvalidInput(format!(
                "Invalid mention span in document {}",
                doc.id
            )));
        }
        let doc_f1 = conll_f1(&pred_chains, &doc.gold_chains);

        per_document.insert(
            doc.id.clone(),
            DocumentResults {
                doc_id: doc.id.clone(),
                num_tokens: doc.approx_tokens(),
                num_windows: windows.len(),
                conll_f1: doc_f1,
                num_gold_chains: doc.gold_chains.len(),
                num_pred_chains: pred_chains.len(),
            },
        );

        // Mention IDs in the scorer are coordinate pairs. A match must stay within its document.
        let next_base = doc_base
            .checked_add(len)
            .and_then(|end| end.checked_add(1))
            .ok_or_else(|| {
                crate::Error::InvalidInput("Corpus character offsets overflow".into())
            })?;
        all_gold_chains.extend(offset_chains(doc.gold_chains.clone(), doc_base));
        all_pred_chains.extend(offset_chains(pred_chains, doc_base));
        doc_base = next_base;
    }

    // Compute aggregate metrics
    let (muc_p, muc_r, muc_f1) =
        crate::eval::coref_metrics::muc_score(&all_pred_chains, &all_gold_chains);
    let (b3_p, b3_r, b3_f1) =
        crate::eval::coref_metrics::b_cubed_score(&all_pred_chains, &all_gold_chains);
    let (ceaf_p, ceaf_r, ceaf_f1) =
        crate::eval::coref_metrics::ceaf_e_score(&all_pred_chains, &all_gold_chains);
    let (lea_p, lea_r, lea_f1) =
        crate::eval::coref_metrics::lea_score(&all_pred_chains, &all_gold_chains);
    let conll = (muc_f1 + b3_f1 + ceaf_f1) / 3.0;

    let total_mentions: usize = all_pred_chains.iter().map(|c| c.len()).sum();
    let avg_cluster_size = if !all_pred_chains.is_empty() {
        total_mentions as f64 / all_pred_chains.len() as f64
    } else {
        0.0
    };

    Ok(CrossContextEvalResults {
        benchmark: "Long-Document".to_string(),
        config: config.clone(),
        muc: CorefScores::from_tuple((muc_p, muc_r, muc_f1)),
        b_cubed: CorefScores::from_tuple((b3_p, b3_r, b3_f1)),
        ceaf_e: CorefScores::from_tuple((ceaf_p, ceaf_r, ceaf_f1)),
        lea: CorefScores::from_tuple((lea_p, lea_r, lea_f1)),
        conll_f1: conll,
        num_contexts: documents.len(),
        num_gold_clusters: all_gold_chains.len(),
        num_pred_clusters: all_pred_chains.len(),
        avg_cluster_size,
        time_ms: start.elapsed().as_millis() as f64,
        per_topic: None,
        per_document: Some(per_document),
    })
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Convert cross-document clusters to coreference chains.
fn cross_doc_clusters_to_chains(
    clusters: &[CrossDocCluster],
    documents: &HashMap<&str, (&Document, usize)>,
) -> Result<Vec<CorefChain>> {
    let mut chains = Vec::new();
    let mut assigned = HashSet::new();
    for cluster in clusters {
        let mut mentions = Vec::new();
        for (doc_id, idx) in &cluster.mentions {
            if !assigned.insert((doc_id, idx)) {
                return Err(crate::Error::InvalidInput(format!(
                    "Entity {idx} in {doc_id} is assigned more than once"
                )));
            }
            let (doc, base) = documents.get(doc_id.as_str()).ok_or_else(|| {
                crate::Error::InvalidInput(format!(
                    "Cluster {} refers to unknown document {doc_id}",
                    cluster.id
                ))
            })?;
            let entity = doc.entities.get(*idx).ok_or_else(|| {
                crate::Error::InvalidInput(format!(
                    "Cluster {} refers to missing entity {idx} in {doc_id}",
                    cluster.id
                ))
            })?;
            let mut mention =
                Mention::new(&entity.text, base + entity.start(), base + entity.end());
            mention.entity_type = Some(entity.entity_type.as_label().to_string());
            mentions.push(mention);
        }
        if !mentions.is_empty() {
            chains.push(CorefChain::new(mentions));
        }
    }
    Ok(chains)
}

// =============================================================================
// Stepwise Error Analysis (Table 5 from xCoRe paper)
// =============================================================================

/// Stepwise error analysis configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StepwiseAnalysis {
    /// Full pipeline (predicted mentions, predicted clusters)
    FullPipeline,
    /// Gold mentions, predicted clusters
    GoldMentions,
    /// Gold mentions, gold clusters (cluster merging only)
    GoldMentionsAndClusters,
}

impl StepwiseAnalysis {
    /// Get description.
    pub fn description(&self) -> &'static str {
        match self {
            Self::FullPipeline => "xCoRe (full pipeline)",
            Self::GoldMentions => "xCoRe (gold mentions)",
            Self::GoldMentionsAndClusters => "xCoRe (gold mentions & clusters)",
        }
    }
}

/// Run stepwise error analysis as in xCoRe Table 5.
///
/// This helps identify which pipeline stage is the bottleneck:
/// - Mention extraction
/// - Within-context clustering
/// - Cross-context cluster merging
pub fn stepwise_error_analysis<E: ClusterEncoder + Clone, S: MergeScorer + Clone>(
    benchmark: CrossContextBenchmark,
    _topics: &[Topic],          // Reserved for a detector-backed cross-doc pipeline
    documents: &[LongDocument], // For long-doc
    encoder: E,
    scorer: S,
) -> Result<HashMap<StepwiseAnalysis, CrossContextEvalResults>> {
    if benchmark.is_cross_document() {
        return Err(crate::Error::FeatureNotAvailable(
            "Cross-document stepwise analysis requires independently detected mentions; Topic supplies a single entity inventory. Use evaluate_cross_document to score supplied mentions.".into(),
        ));
    }
    let mut results = HashMap::new();

    for analysis in [
        StepwiseAnalysis::FullPipeline,
        StepwiseAnalysis::GoldMentions,
        StepwiseAnalysis::GoldMentionsAndClusters,
    ] {
        let config = match analysis {
            StepwiseAnalysis::FullPipeline => CrossContextEvalConfig::for_benchmark(benchmark),
            StepwiseAnalysis::GoldMentions => CrossContextEvalConfig::gold_clusters(),
            StepwiseAnalysis::GoldMentionsAndClusters => CrossContextEvalConfig::oracle(),
        };

        let eval_result =
            evaluate_long_document(documents, encoder.clone(), scorer.clone(), &config)?;

        results.insert(analysis, eval_result);
    }

    Ok(results)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::cluster_encoder::{CosineMergeScorer, HeuristicClusterEncoder};
    use anno::{Entity, EntityType};

    #[test]
    fn test_benchmark_properties() {
        assert!(CrossContextBenchmark::ECBPlus.is_cross_document());
        assert!(!CrossContextBenchmark::ECBPlus.is_long_document());

        assert!(!CrossContextBenchmark::LitBank.is_cross_document());
        assert!(CrossContextBenchmark::LitBank.is_long_document());

        assert_eq!(CrossContextBenchmark::all().len(), 5);
        assert_eq!(CrossContextBenchmark::cross_document().len(), 2);
        assert_eq!(CrossContextBenchmark::long_document().len(), 3);
    }

    #[test]
    fn test_benchmark_sota() {
        assert!((CrossContextBenchmark::ECBPlus.xcore_sota_f1() - 40.3).abs() < 0.1);
        assert!((CrossContextBenchmark::LitBank.xcore_sota_f1() - 78.2).abs() < 0.1);
    }

    #[test]
    fn test_eval_config_default() {
        let config = CrossContextEvalConfig::default();
        assert_eq!(config.window_size, 4000);
        assert!(!config.use_gold_mentions);
    }

    #[test]
    fn test_eval_config_for_benchmark() {
        let config = CrossContextEvalConfig::for_benchmark(CrossContextBenchmark::ECBPlus);
        assert_eq!(config.window_size, 512);
    }

    #[test]
    fn test_topic_creation() {
        let mut topic = Topic::new("topic_1");
        topic.add_document(Document::new("doc1", "Obama visited Paris."));
        topic.add_document(Document::new("doc2", "The president met leaders."));

        assert_eq!(topic.documents.len(), 2);
    }

    #[test]
    fn test_long_document_creation() {
        use anno::MentionType;

        fn new_mention(text: &str, start: usize, end: usize) -> Mention {
            Mention {
                text: text.to_string(),
                start,
                end,
                head_start: None,
                head_end: None,
                entity_type: None,
                mention_type: Some(MentionType::Proper),
            }
        }

        let chains = vec![CorefChain::new(vec![
            new_mention("Obama", 0, 5),
            new_mention("he", 100, 102),
        ])];

        let doc = LongDocument::new(
            "book1",
            "Obama went to Paris. ".repeat(100).as_str(),
            chains,
        );

        assert!(doc.approx_tokens() > 100);
        assert_eq!(doc.gold_chains.len(), 1);
    }

    #[test]
    fn cross_document_scores_keep_repeated_offsets_distinct_across_topics() {
        let mut linked = Topic::new("linked");
        let mut separate = Topic::new("separate");
        let mut pair = CrossDocCluster::new(0u64, "Zoë");
        for id in ["first", "second"] {
            let doc = Document::new(id, "Zoë").with_entities(vec![Entity::new(
                "Zoë",
                EntityType::Person,
                0,
                3,
                1.0,
            )]);
            linked.add_document(doc.clone());
            separate.add_document(doc);
            pair.add_mention(id, 0);
            let mut singleton = CrossDocCluster::new(separate.gold_clusters.len() as u64, "Zoë");
            singleton.add_mention(id, 0);
            separate.add_gold_cluster(singleton);
        }
        linked.add_gold_cluster(pair);
        let results = evaluate_cross_document(
            &[linked, separate],
            HeuristicClusterEncoder::new(64),
            CosineMergeScorer::new(),
            &CrossContextEvalConfig {
                merge_threshold: 2.0,
                ..Default::default()
            },
        )
        .unwrap();
        // The scorer predicted four singletons. Two mentions earn half recall
        // for the missed pair, and the two true singletons each earn full recall.
        assert_eq!(results.num_pred_clusters, 4);
        assert_eq!(results.b_cubed.precision, 1.0);
        assert_eq!(results.b_cubed.recall, 0.75);
        assert_eq!(results.muc.recall, 0.0);
    }

    #[test]
    fn gold_local_seeds_change_the_resolution_and_links_exclude_within_doc_pairs() {
        let mut topic = Topic::new("topic");
        topic.add_document(Document::new("a", "Nia she").with_entities(vec![
            Entity::new("Nia", EntityType::Person, 0, 3, 1.0),
            Entity::new("she", EntityType::Person, 4, 7, 1.0),
        ]));
        topic.add_document(Document::new("b", "Nia").with_entities(vec![Entity::new(
            "Nia",
            EntityType::Person,
            0,
            3,
            1.0,
        )]));
        let mut gold = CrossDocCluster::new(0u64, "");
        for (doc, index) in [("a", 0), ("a", 1), ("b", 0)] {
            gold.add_mention(doc, index);
        }
        topic.add_gold_cluster(gold);
        let run = |gold_local_seeds| {
            evaluate_cross_document_with_links(
                std::slice::from_ref(&topic),
                HeuristicClusterEncoder::new(64),
                CosineMergeScorer::new(),
                &CrossContextEvalConfig {
                    merge_threshold: 2.0,
                    use_gold_mentions: true,
                    use_gold_clusters: gold_local_seeds,
                    ..Default::default()
                },
            )
            .unwrap()
        };
        let (singletons, links) = run(false);
        let (gold_local, seeded_links) = run(true);
        assert_eq!(singletons.num_pred_clusters, 3);
        assert_eq!(gold_local.num_pred_clusters, 2);
        assert!(gold_local.b_cubed.recall > singletons.b_cubed.recall);
        assert_eq!(
            links,
            CrossDocumentLinkCounts {
                gold: 2,
                predicted: 0,
                correct: 0
            }
        );
        assert_eq!(seeded_links, links);

        let mut split = CrossDocCluster::new(1u64, "");
        split.add_mention("a", 0);
        split.add_mention("b", 0);
        split.add_mention("c", 0);
        let mut counts = CrossDocumentLinkCounts::default();
        counts.add_topic(&topic.gold_clusters, &[split]);
        assert_eq!(
            counts,
            CrossDocumentLinkCounts {
                gold: 2,
                predicted: 3,
                correct: 1
            }
        );
    }

    #[test]
    fn per_topic_cap_projects_gold_predictions_and_cross_document_support() {
        let mut topics = Vec::new();
        for name in ["first", "second"] {
            let mut topic = Topic::new(name);
            let mut gold = CrossDocCluster::new(1u64, "");
            for doc in ["z", "b", "a"] {
                topic.add_document(Document::new(doc, "Nia").with_entities(vec![Entity::new(
                    "Nia",
                    EntityType::Person,
                    0,
                    3,
                    1.0,
                )]));
            }
            gold.add_mention("z", 0);
            let (other, singleton) = if name == "first" {
                ("a", "b")
            } else {
                ("b", "a")
            };
            gold.add_mention(other, 0);
            topic.add_gold_cluster(gold);
            let mut local = CrossDocCluster::new(2u64, "");
            local.add_mention(singleton, 0);
            topic.add_gold_cluster(local);
            topics.push(topic);
        }
        let (scores, links) = evaluate_cross_document_with_links(
            &topics,
            HeuristicClusterEncoder::new(64),
            CosineMergeScorer::new(),
            &CrossContextEvalConfig {
                max_docs_per_topic: 2,
                ..CrossContextEvalConfig::oracle()
            },
        )
        .unwrap();
        assert_eq!(scores.num_contexts, 4);
        assert_eq!(scores.num_gold_clusters, 3);
        assert_eq!(scores.num_pred_clusters, 2);
        assert_eq!(
            links,
            CrossDocumentLinkCounts {
                correct: 1,
                gold: 1,
                predicted: 2
            }
        );
        let per_topic = scores.per_topic.unwrap();
        assert_eq!(per_topic["first"].num_gold_clusters, 2);
        assert_eq!(per_topic["second"].num_gold_clusters, 1);
        assert!(per_topic.values().all(|topic| topic.num_documents == 2));
    }

    #[test]
    fn detector_comparison_and_incomplete_gold_input_are_explicit() {
        let analysis = stepwise_error_analysis(
            CrossContextBenchmark::ECBPlus,
            &[],
            &[],
            HeuristicClusterEncoder::new(64),
            CosineMergeScorer::new(),
        );
        assert!(matches!(
            analysis,
            Err(crate::Error::FeatureNotAvailable(_))
        ));
        let mut topic = Topic::new("topic");
        topic.add_document(Document::new("a", "Nia").with_entities(vec![Entity::new(
            "Nia",
            EntityType::Person,
            0,
            3,
            1.0,
        )]));
        let score = evaluate_cross_document(
            &[topic],
            HeuristicClusterEncoder::new(64),
            CosineMergeScorer::new(),
            &CrossContextEvalConfig::oracle(),
        );
        assert!(matches!(score, Err(crate::Error::InvalidInput(_))));
    }

    #[test]
    fn test_evaluate_cross_document_empty() {
        let encoder = HeuristicClusterEncoder::new(64);
        let scorer = CosineMergeScorer::new();
        let config = CrossContextEvalConfig::default();

        let topics: Vec<Topic> = vec![];
        let results = evaluate_cross_document(&topics, encoder, scorer, &config).unwrap();

        assert_eq!(results.num_contexts, 0);
    }

    #[test]
    fn test_evaluate_cross_document_single_topic() {
        let encoder = HeuristicClusterEncoder::new(64);
        let scorer = CosineMergeScorer::new();
        let config = CrossContextEvalConfig::default();

        let mut topic = Topic::new("topic_1");
        topic.add_document(
            Document::new("doc1", "Obama visited France.").with_entities(vec![Entity::new(
                "Obama",
                EntityType::Person,
                0,
                5,
                0.9,
            )]),
        );
        topic.add_document(
            Document::new("doc2", "The president met Macron.").with_entities(vec![
                Entity::new("The president", EntityType::Person, 0, 13, 0.8),
                Entity::new("Macron", EntityType::Person, 18, 24, 0.9),
            ]),
        );

        let results = evaluate_cross_document(&[topic], encoder, scorer, &config).unwrap();

        assert_eq!(results.num_contexts, 2);
        assert!(results.per_topic.is_some());
    }

    #[test]
    fn test_stepwise_analysis_types() {
        assert_eq!(
            StepwiseAnalysis::FullPipeline.description(),
            "xCoRe (full pipeline)"
        );
        assert_eq!(
            StepwiseAnalysis::GoldMentions.description(),
            "xCoRe (gold mentions)"
        );
    }

    #[test]
    fn test_results_summary() {
        let results = CrossContextEvalResults {
            benchmark: "Test".to_string(),
            config: CrossContextEvalConfig::default(),
            muc: CorefScores::new(0.8, 0.7),
            b_cubed: CorefScores::new(0.75, 0.65),
            ceaf_e: CorefScores::new(0.7, 0.6),
            lea: CorefScores::new(0.72, 0.62),
            conll_f1: 0.70,
            num_contexts: 10,
            num_gold_clusters: 50,
            num_pred_clusters: 45,
            avg_cluster_size: 2.5,
            time_ms: 100.0,
            per_topic: None,
            per_document: None,
        };

        let summary = results.summary();
        assert!(summary.contains("70.0%"));
    }

    #[test]
    fn test_evaluate_cross_document_with_synthetic_data() {
        // 2 topics with 3 docs each, overlapping entity names across docs
        let encoder = HeuristicClusterEncoder::new(64);
        let scorer = CosineMergeScorer::new();
        let config = CrossContextEvalConfig::default();

        // Topic 1: Obama visits France
        let mut topic1 = Topic::new("politics");
        topic1.add_document(
            Document::new("doc1", "Obama visited France yesterday.").with_entities(vec![
                Entity::new("Obama", EntityType::Person, 0, 5, 0.9),
                Entity::new("France", EntityType::Location, 14, 20, 0.9),
            ]),
        );
        topic1.add_document(
            Document::new("doc2", "The president arrived in Paris.").with_entities(vec![
                Entity::new("The president", EntityType::Person, 0, 13, 0.8),
                Entity::new("Paris", EntityType::Location, 25, 30, 0.9),
            ]),
        );
        topic1.add_document(
            Document::new("doc3", "Barack Obama met Macron in France.").with_entities(vec![
                Entity::new("Barack Obama", EntityType::Person, 0, 12, 0.95),
                Entity::new("Macron", EntityType::Person, 17, 23, 0.9),
                Entity::new("France", EntityType::Location, 27, 33, 0.9),
            ]),
        );
        // Gold: Obama cluster across 3 docs, France/Paris cluster across 2 docs
        let mut obama_cluster = crate::eval::cdcr::CrossDocCluster::new(0u64, "Obama");
        obama_cluster.mentions = vec![
            ("doc1".to_string(), 0),
            ("doc2".to_string(), 0),
            ("doc3".to_string(), 0),
        ];
        let mut france_cluster = crate::eval::cdcr::CrossDocCluster::new(1u64, "France");
        france_cluster.mentions = vec![
            ("doc1".to_string(), 1),
            ("doc2".to_string(), 1),
            ("doc3".to_string(), 2),
        ];
        topic1.add_gold_cluster(obama_cluster);
        topic1.add_gold_cluster(france_cluster);

        // Topic 2: Tech companies
        let mut topic2 = Topic::new("tech");
        topic2.add_document(
            Document::new("doc4", "Apple released new products.").with_entities(vec![Entity::new(
                "Apple",
                EntityType::Organization,
                0,
                5,
                0.9,
            )]),
        );
        topic2.add_document(
            Document::new("doc5", "The company expanded in Asia.").with_entities(vec![
                Entity::new("The company", EntityType::Organization, 0, 11, 0.8),
                Entity::new("Asia", EntityType::Location, 24, 28, 0.9),
            ]),
        );
        topic2.add_document(
            Document::new("doc6", "Apple Inc announced quarterly results.").with_entities(vec![
                Entity::new("Apple Inc", EntityType::Organization, 0, 9, 0.9),
            ]),
        );
        let mut apple_cluster = crate::eval::cdcr::CrossDocCluster::new(0u64, "Apple");
        apple_cluster.mentions = vec![
            ("doc4".to_string(), 0),
            ("doc5".to_string(), 0),
            ("doc6".to_string(), 0),
        ];
        topic2.add_gold_cluster(apple_cluster);

        let results = evaluate_cross_document(&[topic1, topic2], encoder, scorer, &config).unwrap();

        // Basic sanity: metrics in valid ranges
        assert!(results.conll_f1 >= 0.0 && results.conll_f1 <= 1.0);
        assert!(results.muc.f1 >= 0.0 && results.muc.f1 <= 1.0);
        assert!(results.b_cubed.f1 >= 0.0 && results.b_cubed.f1 <= 1.0);
        assert!(results.ceaf_e.f1 >= 0.0 && results.ceaf_e.f1 <= 1.0);

        // Should have evaluated 6 documents across 2 topics
        assert_eq!(results.num_contexts, 6);
        assert!(results.per_topic.is_some());
        let per_topic = results.per_topic.as_ref().unwrap();
        assert_eq!(per_topic.len(), 2);
        assert!(per_topic.contains_key("politics"));
        assert!(per_topic.contains_key("tech"));

        // Gold clusters: 3 total (obama, france, apple)
        assert_eq!(results.num_gold_clusters, 3);

        // Predicted clusters should be non-zero (heuristic encoder finds something)
        assert!(results.num_pred_clusters > 0);
    }

    #[test]
    fn long_document_scores_keep_identical_local_offsets_in_separate_documents() {
        let chain = CorefChain::new(vec![Mention::new("A", 0, 1), Mention::new("A", 2, 3)]);
        let mut spurious = LongDocument::new("spurious", "A A", vec![]);
        spurious.windows = Some(vec![WindowOutput::new(0, 0, 3, vec![chain.clone()])]);
        let missed = LongDocument::new("missed", "A A", vec![chain]);
        let evaluate = |docs: &[LongDocument]| {
            evaluate_long_document(
                docs,
                HeuristicClusterEncoder::new(64),
                CosineMergeScorer::new(),
                &CrossContextEvalConfig {
                    use_gold_mentions: false,
                    ..Default::default()
                },
            )
            .unwrap()
        };
        for docs in [[spurious.clone(), missed.clone()], [missed, spurious]] {
            let results = evaluate(&docs);
            assert_eq!(results.num_gold_clusters, 1);
            assert_eq!(results.num_pred_clusters, 1);
            assert!(results
                .per_document
                .unwrap()
                .values()
                .all(|d| d.conll_f1 == 0.0));
            assert_eq!(
                [
                    results.muc.f1,
                    results.b_cubed.f1,
                    results.ceaf_e.f1,
                    results.lea.f1,
                    results.conll_f1
                ],
                [0.0; 5]
            );
        }
    }

    #[test]
    fn test_evaluate_long_document_with_gold_mentions() {
        let encoder = HeuristicClusterEncoder::new(64);
        let scorer = CosineMergeScorer::new();
        let config = CrossContextEvalConfig {
            use_gold_mentions: true,
            ..CrossContextEvalConfig::default()
        };

        let chains = vec![
            CorefChain::new(vec![
                Mention::new("Obama", 0, 5),
                Mention::new("he", 50, 52),
            ]),
            CorefChain::new(vec![
                Mention::new("France", 14, 20),
                Mention::new("the country", 60, 71),
            ]),
        ];

        let doc = LongDocument::new("long_doc", &"Obama visited France. ".repeat(10), chains);

        let results = evaluate_long_document(&[doc], encoder, scorer, &config).unwrap();
        assert_eq!(results.num_contexts, 1);
        // With gold mentions in single-window mode, should produce per-doc results
        assert!(results.per_document.is_some());
        let per_doc = results.per_document.as_ref().unwrap();
        assert_eq!(per_doc.len(), 1);
        assert!(per_doc.contains_key("long_doc"));
    }
}
