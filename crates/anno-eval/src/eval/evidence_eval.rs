//! Evidence-based clustering evaluation.
//!
//! This module provides evaluation metrics specifically designed for
//! evidence-based entity resolution and coreference clustering.
//!
//! ## Key Metrics
//!
//! | Metric | Description |
//! |--------|-------------|
//! | **Transitivity Score** | How well clusters respect transitivity |
//! | **Evidence Agreement** | Correlation between evidence and clustering |
//! | **Mediation Quality** | How well mediation strategies combine signals |
//! | **Source Importance** | Which evidence sources contribute most |
//!
//! ## Related Modules
//!
//! - `archive/geometric-2024-12/sheaf.rs`: Sheaf neural networks enforce transitivity at
//!   the gradient level (training time) rather than as post-hoc evaluation.
//!   This module evaluates transitivity; sheaf learns to satisfy it.
//!
//! - `archive/geometric-2024-12/mod.rs::transitivity_violations`: Utility function for detecting
//!   transitivity violations in a score matrix.
//!
//! ## References
//!
//! - Hypergraph Clustering: Chodrow et al. 2022
//! - Dempster-Shafer Theory: Shafer 1976, recent applications in NLP
//! - Correlation Clustering: Bansal et al. 2004

use std::collections::HashMap;

/// Results from evidence-based evaluation.
#[derive(Debug, Clone, Default)]
pub struct EvidenceEvalResults {
    /// Overall score (0-1, higher is better)
    pub overall_score: f64,
    
    /// Transitivity score: how consistent are same-cluster pairs?
    pub transitivity_score: f64,
    
    /// Evidence-cluster agreement: does high evidence predict same cluster?
    pub evidence_agreement: f64,
    
    /// Per-source contribution scores
    pub source_contributions: HashMap<String, f64>,
    
    /// Number of transitivity violations detected
    pub transitivity_violations: usize,
    
    /// Breakdown by cluster size
    pub scores_by_cluster_size: HashMap<String, f64>,
}

/// Configuration for evidence evaluation.
#[derive(Debug, Clone)]
pub struct EvidenceEvalConfig {
    /// Threshold for considering evidence "positive"
    pub positive_threshold: f64,
    /// Threshold for considering evidence "negative"
    pub negative_threshold: f64,
    /// Whether to compute per-source metrics
    pub compute_source_metrics: bool,
    /// Maximum cluster size to analyze individually
    pub max_individual_cluster_size: usize,
}

impl Default for EvidenceEvalConfig {
    fn default() -> Self {
        Self {
            positive_threshold: 0.6,
            negative_threshold: 0.4,
            compute_source_metrics: true,
            max_individual_cluster_size: 100,
        }
    }
}

/// A single mention for evaluation purposes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EvalMention {
    /// Document ID
    pub doc_id: String,
    /// Surface text
    pub text: String,
    /// Unique identifier
    pub id: String,
}

/// Pairwise evidence for evaluation.
#[derive(Debug, Clone)]
pub struct EvalPairEvidence {
    /// First mention ID
    pub id_a: String,
    /// Second mention ID
    pub id_b: String,
    /// Combined confidence score
    pub confidence: f64,
    /// Source-specific scores
    pub source_scores: HashMap<String, f64>,
}

/// A cluster of mentions.
#[derive(Debug, Clone)]
pub struct EvalCluster {
    /// Cluster ID
    pub id: usize,
    /// Mentions in this cluster
    pub mentions: Vec<EvalMention>,
}

/// Evaluator for evidence-based clustering.
pub struct EvidenceEvaluator {
    config: EvidenceEvalConfig,
}

impl EvidenceEvaluator {
    /// Create a new evaluator with default config.
    pub fn new() -> Self {
        Self {
            config: EvidenceEvalConfig::default(),
        }
    }
    
    /// Create with custom config.
    pub fn with_config(config: EvidenceEvalConfig) -> Self {
        Self { config }
    }
    
    /// Evaluate clustering given pairwise evidence.
    pub fn evaluate(
        &self,
        clusters: &[EvalCluster],
        evidence: &[EvalPairEvidence],
    ) -> EvidenceEvalResults {
        let mut results = EvidenceEvalResults::default();
        
        // Build mention-to-cluster mapping
        let mut mention_to_cluster: HashMap<&str, usize> = HashMap::new();
        for cluster in clusters {
            for mention in &cluster.mentions {
                mention_to_cluster.insert(&mention.id, cluster.id);
            }
        }
        
        // Calculate evidence-cluster agreement
        let (agreement, violations) = self.compute_agreement(&mention_to_cluster, evidence);
        results.evidence_agreement = agreement;
        results.transitivity_violations = violations;
        
        // Calculate transitivity score
        results.transitivity_score = self.compute_transitivity_score(clusters, evidence);
        
        // Compute per-source contributions
        if self.config.compute_source_metrics {
            results.source_contributions = self.compute_source_contributions(
                &mention_to_cluster,
                evidence,
            );
        }
        
        // Scores by cluster size
        results.scores_by_cluster_size = self.scores_by_size(clusters, evidence, &mention_to_cluster);
        
        // Overall score: weighted combination
        results.overall_score = 0.4 * results.transitivity_score
            + 0.4 * results.evidence_agreement
            + 0.2 * (1.0 - (violations as f64 / evidence.len().max(1) as f64).min(1.0));
        
        results
    }
    
    /// Compute how well evidence agrees with clustering decisions.
    fn compute_agreement(
        &self,
        mention_to_cluster: &HashMap<&str, usize>,
        evidence: &[EvalPairEvidence],
    ) -> (f64, usize) {
        let mut correct = 0;
        let mut total = 0;
        let mut violations = 0;
        
        for ev in evidence {
            let cluster_a = mention_to_cluster.get(ev.id_a.as_str());
            let cluster_b = mention_to_cluster.get(ev.id_b.as_str());
            
            if let (Some(&ca), Some(&cb)) = (cluster_a, cluster_b) {
                total += 1;
                let same_cluster = ca == cb;
                let evidence_positive = ev.confidence >= self.config.positive_threshold;
                let evidence_negative = ev.confidence <= self.config.negative_threshold;
                
                // Agreement: high evidence + same cluster, OR low evidence + different cluster
                if (evidence_positive && same_cluster) || (evidence_negative && !same_cluster) {
                    correct += 1;
                }
                
                // Violation: high evidence but different cluster, OR low evidence but same cluster
                if (evidence_positive && !same_cluster) || (evidence_negative && same_cluster) {
                    violations += 1;
                }
            }
        }
        
        let agreement = if total > 0 {
            correct as f64 / total as f64
        } else {
            1.0
        };
        
        (agreement, violations)
    }
    
    /// Compute transitivity score for clusters.
    fn compute_transitivity_score(
        &self,
        clusters: &[EvalCluster],
        evidence: &[EvalPairEvidence],
    ) -> f64 {
        // Build evidence lookup
        let mut evidence_map: HashMap<(&str, &str), f64> = HashMap::new();
        for ev in evidence {
            let key = if ev.id_a < ev.id_b {
                (ev.id_a.as_str(), ev.id_b.as_str())
            } else {
                (ev.id_b.as_str(), ev.id_a.as_str())
            };
            evidence_map.insert(key, ev.confidence);
        }
        
        let mut total_triangles = 0;
        let mut valid_triangles = 0;
        
        // Check triangles within each cluster
        for cluster in clusters {
            if cluster.mentions.len() < 3 {
                continue;
            }
            
            let ids: Vec<&str> = cluster.mentions.iter().map(|m| m.id.as_str()).collect();
            
            // Sample triangles if cluster is large
            let max_checks = self.config.max_individual_cluster_size.pow(3) / 6;
            let mut checks = 0;
            
            for i in 0..ids.len() {
                for j in (i + 1)..ids.len() {
                    for k in (j + 1)..ids.len() {
                        if checks >= max_checks {
                            break;
                        }
                        checks += 1;
                        total_triangles += 1;
                        
                        let (a, b, c) = (ids[i], ids[j], ids[k]);
                        let ab = evidence_map.get(&if a < b { (a, b) } else { (b, a) });
                        let bc = evidence_map.get(&if b < c { (b, c) } else { (c, b) });
                        let ac = evidence_map.get(&if a < c { (a, c) } else { (c, a) });
                        
                        // Check if transitivity holds
                        if let (Some(&sab), Some(&sbc), Some(&sac)) = (ab, bc, ac) {
                            // If A-B and B-C are high, A-C should also be high
                            if sab >= self.config.positive_threshold
                                && sbc >= self.config.positive_threshold
                                && sac >= self.config.positive_threshold
                            {
                                valid_triangles += 1;
                            } else if sab < self.config.positive_threshold
                                || sbc < self.config.positive_threshold
                            {
                                // Weak A-B or B-C, so A-C can be anything
                                valid_triangles += 1;
                            }
                        } else {
                            // Missing evidence, assume valid
                            valid_triangles += 1;
                        }
                    }
                }
            }
        }
        
        if total_triangles == 0 {
            1.0
        } else {
            valid_triangles as f64 / total_triangles as f64
        }
    }
    
    /// Compute contribution of each evidence source.
    fn compute_source_contributions(
        &self,
        mention_to_cluster: &HashMap<&str, usize>,
        evidence: &[EvalPairEvidence],
    ) -> HashMap<String, f64> {
        let mut source_correct: HashMap<String, usize> = HashMap::new();
        let mut source_total: HashMap<String, usize> = HashMap::new();
        
        for ev in evidence {
            let cluster_a = mention_to_cluster.get(ev.id_a.as_str());
            let cluster_b = mention_to_cluster.get(ev.id_b.as_str());
            
            if let (Some(&ca), Some(&cb)) = (cluster_a, cluster_b) {
                let same_cluster = ca == cb;
                
                for (source, &score) in &ev.source_scores {
                    *source_total.entry(source.clone()).or_insert(0) += 1;
                    
                    let source_positive = score >= self.config.positive_threshold;
                    let source_negative = score <= self.config.negative_threshold;
                    
                    if (source_positive && same_cluster) || (source_negative && !same_cluster) {
                        *source_correct.entry(source.clone()).or_insert(0) += 1;
                    }
                }
            }
        }
        
        source_total
            .iter()
            .map(|(source, &total)| {
                let correct = source_correct.get(source).copied().unwrap_or(0);
                (source.clone(), correct as f64 / total as f64)
            })
            .collect()
    }
    
    /// Compute scores broken down by cluster size.
    fn scores_by_size(
        &self,
        clusters: &[EvalCluster],
        evidence: &[EvalPairEvidence],
        mention_to_cluster: &HashMap<&str, usize>,
    ) -> HashMap<String, f64> {
        let mut results = HashMap::new();
        
        // Categorize clusters
        let mut small = 0;   // 1-2 mentions
        let mut medium = 0;  // 3-10 mentions
        let mut large = 0;   // 11+ mentions
        let mut small_total = 0;
        let mut medium_total = 0;
        let mut large_total = 0;
        
        // Build cluster size lookup
        let cluster_sizes: HashMap<usize, usize> = clusters
            .iter()
            .map(|c| (c.id, c.mentions.len()))
            .collect();
        
        for ev in evidence {
            let cluster_a = mention_to_cluster.get(ev.id_a.as_str());
            let cluster_b = mention_to_cluster.get(ev.id_b.as_str());
            
            if let (Some(&ca), Some(&cb)) = (cluster_a, cluster_b) {
                let same_cluster = ca == cb;
                let evidence_positive = ev.confidence >= self.config.positive_threshold;
                let evidence_negative = ev.confidence <= self.config.negative_threshold;
                let correct = (evidence_positive && same_cluster) 
                    || (evidence_negative && !same_cluster);
                
                // Determine size category
                let size = if same_cluster {
                    cluster_sizes.get(&ca).copied().unwrap_or(1)
                } else {
                    // For different clusters, use max
                    let sa = cluster_sizes.get(&ca).copied().unwrap_or(1);
                    let sb = cluster_sizes.get(&cb).copied().unwrap_or(1);
                    sa.max(sb)
                };
                
                if size <= 2 {
                    small_total += 1;
                    if correct { small += 1; }
                } else if size <= 10 {
                    medium_total += 1;
                    if correct { medium += 1; }
                } else {
                    large_total += 1;
                    if correct { large += 1; }
                }
            }
        }
        
        if small_total > 0 {
            results.insert("small_clusters".into(), small as f64 / small_total as f64);
        }
        if medium_total > 0 {
            results.insert("medium_clusters".into(), medium as f64 / medium_total as f64);
        }
        if large_total > 0 {
            results.insert("large_clusters".into(), large as f64 / large_total as f64);
        }
        
        results
    }
}

impl Default for EvidenceEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

/// Compare two clustering strategies.
pub fn compare_strategies(
    clusters_a: &[EvalCluster],
    clusters_b: &[EvalCluster],
    evidence: &[EvalPairEvidence],
) -> StrategyComparison {
    let evaluator = EvidenceEvaluator::new();
    
    let results_a = evaluator.evaluate(clusters_a, evidence);
    let results_b = evaluator.evaluate(clusters_b, evidence);
    
    StrategyComparison {
        strategy_a_score: results_a.overall_score,
        strategy_b_score: results_b.overall_score,
        transitivity_delta: results_a.transitivity_score - results_b.transitivity_score,
        agreement_delta: results_a.evidence_agreement - results_b.evidence_agreement,
        better_strategy: if results_a.overall_score >= results_b.overall_score {
            "A".to_string()
        } else {
            "B".to_string()
        },
    }
}

/// Results of comparing two clustering strategies.
#[derive(Debug, Clone)]
pub struct StrategyComparison {
    /// Overall score for strategy A
    pub strategy_a_score: f64,
    /// Overall score for strategy B
    pub strategy_b_score: f64,
    /// Difference in transitivity scores
    pub transitivity_delta: f64,
    /// Difference in evidence agreement
    pub agreement_delta: f64,
    /// Which strategy is better overall
    pub better_strategy: String,
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    fn make_mention(id: &str) -> EvalMention {
        EvalMention {
            doc_id: "doc1".into(),
            text: id.into(),
            id: id.into(),
        }
    }
    
    fn make_evidence(id_a: &str, id_b: &str, confidence: f64) -> EvalPairEvidence {
        EvalPairEvidence {
            id_a: id_a.into(),
            id_b: id_b.into(),
            confidence,
            source_scores: HashMap::new(),
        }
    }
    
    #[test]
    fn test_perfect_clustering() {
        // Perfect clustering: high evidence within clusters, low between
        let clusters = vec![
            EvalCluster {
                id: 0,
                mentions: vec![make_mention("a1"), make_mention("a2")],
            },
            EvalCluster {
                id: 1,
                mentions: vec![make_mention("b1"), make_mention("b2")],
            },
        ];
        
        let evidence = vec![
            make_evidence("a1", "a2", 0.9), // Same cluster, high conf
            make_evidence("b1", "b2", 0.85), // Same cluster, high conf
            make_evidence("a1", "b1", 0.1), // Different cluster, low conf
        ];
        
        let evaluator = EvidenceEvaluator::new();
        let results = evaluator.evaluate(&clusters, &evidence);
        
        assert!(results.evidence_agreement > 0.9);
        assert_eq!(results.transitivity_violations, 0);
    }
    
    #[test]
    fn test_bad_clustering() {
        // Bad clustering: merged mentions that shouldn't be
        let clusters = vec![EvalCluster {
            id: 0,
            mentions: vec![
                make_mention("a1"),
                make_mention("a2"),
                make_mention("b1"), // Shouldn't be here
            ],
        }];
        
        let evidence = vec![
            make_evidence("a1", "a2", 0.9), // Good
            make_evidence("a1", "b1", 0.1), // Bad - low conf but same cluster
        ];
        
        let evaluator = EvidenceEvaluator::new();
        let results = evaluator.evaluate(&clusters, &evidence);
        
        // Should have violations
        assert!(results.transitivity_violations > 0);
        assert!(results.evidence_agreement < 1.0);
    }
    
    #[test]
    fn test_source_contributions() {
        let clusters = vec![EvalCluster {
            id: 0,
            mentions: vec![make_mention("a"), make_mention("b")],
        }];
        
        let mut source_scores = HashMap::new();
        source_scores.insert("string_sim".into(), 0.9);
        source_scores.insert("embedding".into(), 0.85);
        
        let evidence = vec![EvalPairEvidence {
            id_a: "a".into(),
            id_b: "b".into(),
            confidence: 0.87,
            source_scores,
        }];
        
        let evaluator = EvidenceEvaluator::new();
        let results = evaluator.evaluate(&clusters, &evidence);
        
        // Should have source contributions
        assert!(results.source_contributions.contains_key("string_sim"));
        assert!(results.source_contributions.contains_key("embedding"));
    }
}

