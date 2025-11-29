//! Evaluation metrics for inter-document coreference resolution.
//!
//! Provides metrics specific to cross-document entity clustering,
//! complementing the standard coreference metrics in `coref_metrics.rs`.

use crate::grounded::{Identity, IdentityId, TrackRef};
use std::collections::{HashMap, HashSet};

/// Metrics for inter-document coreference resolution quality.
#[derive(Debug, Clone)]
pub struct InterDocCorefMetrics {
    /// Cluster purity: average fraction of tracks in each identity that are correct
    pub cluster_purity: f64,
    /// Cluster completeness: average fraction of correct tracks that are in the same identity
    pub cluster_completeness: f64,
    /// Number of predicted identities
    pub num_pred_identities: usize,
    /// Number of gold identities
    pub num_gold_identities: usize,
    /// Number of tracks correctly clustered
    pub num_correct: usize,
    /// Total number of tracks
    pub num_total: usize,
}

impl InterDocCorefMetrics {
    /// Compute metrics comparing predicted identities to gold standard.
    ///
    /// # Arguments
    ///
    /// * `predicted` - Predicted identities from corpus
    /// * `gold` - Gold standard identities (track_refs grouped by identity)
    ///
    /// # Returns
    ///
    /// Metrics with purity, completeness, and counts.
    #[must_use]
    pub fn compute(predicted: &[Identity], gold: &[Vec<TrackRef>]) -> Self {
        if predicted.is_empty() && gold.is_empty() {
            return Self::default();
        }

        // Build track_ref -> identity_id mapping for predicted
        let mut pred_map: HashMap<TrackRef, IdentityId> = HashMap::new();
        for identity in predicted {
            if let Some(crate::grounded::IdentitySource::CrossDocCoref { track_refs }) =
                &identity.source
            {
                for track_ref in track_refs {
                    pred_map.insert(track_ref.clone(), identity.id);
                }
            }
        }

        // Build track_ref -> gold cluster index mapping
        let mut gold_map: HashMap<TrackRef, usize> = HashMap::new();
        for (idx, cluster) in gold.iter().enumerate() {
            for track_ref in cluster {
                gold_map.insert(track_ref.clone(), idx);
            }
        }

        // Get all track refs
        let all_tracks: HashSet<_> = pred_map.keys().chain(gold_map.keys()).cloned().collect();
        let num_total = all_tracks.len();

        if num_total == 0 {
            return Self::default();
        }

        // Compute cluster purity and completeness
        let mut total_purity = 0.0;
        let mut total_completeness = 0.0;
        let mut num_correct = 0;

        // For each predicted identity, compute purity
        for identity in predicted {
            if let Some(crate::grounded::IdentitySource::CrossDocCoref { track_refs }) =
                &identity.source
            {
                if track_refs.is_empty() {
                    continue;
                }

                // Count how many tracks in this identity share the same gold cluster
                let mut gold_cluster_counts: HashMap<usize, usize> = HashMap::new();
                for track_ref in track_refs {
                    if let Some(&gold_cluster) = gold_map.get(track_ref) {
                        *gold_cluster_counts.entry(gold_cluster).or_insert(0) += 1;
                    }
                }

                // Purity: fraction of tracks that share the most common gold cluster
                let max_count = gold_cluster_counts.values().max().copied().unwrap_or(0);
                let purity = if track_refs.is_empty() {
                    0.0
                } else {
                    max_count as f64 / track_refs.len() as f64
                };
                total_purity += purity * track_refs.len() as f64;

                // Count correct links
                num_correct += max_count;
            }
        }

        // For each gold cluster, compute completeness
        for cluster in gold.iter() {
            if cluster.is_empty() {
                continue;
            }

            // Count how many tracks in this gold cluster share the same predicted identity
            let mut pred_identity_counts: HashMap<IdentityId, usize> = HashMap::new();
            for track_ref in cluster {
                if let Some(&pred_identity) = pred_map.get(track_ref) {
                    *pred_identity_counts.entry(pred_identity).or_insert(0) += 1;
                }
            }

            // Completeness: fraction of tracks that share the most common predicted identity
            let max_count = pred_identity_counts.values().max().copied().unwrap_or(0);
            let completeness = if cluster.is_empty() {
                0.0
            } else {
                max_count as f64 / cluster.len() as f64
            };
            total_completeness += completeness * cluster.len() as f64;
        }

        let cluster_purity = if num_total > 0 {
            total_purity / num_total as f64
        } else {
            0.0
        };

        let cluster_completeness = if num_total > 0 {
            total_completeness / num_total as f64
        } else {
            0.0
        };

        Self {
            cluster_purity,
            cluster_completeness,
            num_pred_identities: predicted.len(),
            num_gold_identities: gold.len(),
            num_correct,
            num_total,
        }
    }

    /// Compute F1 score from purity and completeness.
    #[must_use]
    pub fn f1(&self) -> f64 {
        if self.cluster_purity + self.cluster_completeness == 0.0 {
            0.0
        } else {
            2.0 * self.cluster_purity * self.cluster_completeness
                / (self.cluster_purity + self.cluster_completeness)
        }
    }
}

impl Default for InterDocCorefMetrics {
    fn default() -> Self {
        Self {
            cluster_purity: 0.0,
            cluster_completeness: 0.0,
            num_pred_identities: 0,
            num_gold_identities: 0,
            num_correct: 0,
            num_total: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grounded::{GroundedDocument, Location, Signal, Track};

    fn create_test_corpus() -> (crate::grounded::Corpus, Vec<Vec<TrackRef>>) {
        let mut corpus = crate::grounded::Corpus::new();

        // Document 1: "Apple" and "Microsoft"
        let mut doc1 = GroundedDocument::new("doc1", "Apple and Microsoft");
        let s1 = doc1.add_signal(Signal::new(0, Location::text(0, 5), "Apple", "Org", 0.9));
        let s2 = doc1.add_signal(Signal::new(
            1,
            Location::text(10, 19),
            "Microsoft",
            "Org",
            0.9,
        ));
        let mut track1 = Track::new(0, "Apple");
        track1.add_signal(s1, 0);
        let mut track2 = Track::new(1, "Microsoft");
        track2.add_signal(s2, 0);
        doc1.add_track(track1);
        doc1.add_track(track2);
        corpus.add_document(doc1);

        // Document 2: "Apple Inc"
        let mut doc2 = GroundedDocument::new("doc2", "Apple Inc");
        let s3 = doc2.add_signal(Signal::new(
            0,
            Location::text(0, 10),
            "Apple Inc",
            "Org",
            0.9,
        ));
        let mut track3 = Track::new(0, "Apple Inc");
        track3.add_signal(s3, 0);
        doc2.add_track(track3);
        corpus.add_document(doc2);

        // Document 3: "Microsoft Corp"
        let mut doc3 = GroundedDocument::new("doc3", "Microsoft Corp");
        let s4 = doc3.add_signal(Signal::new(
            0,
            Location::text(0, 13),
            "Microsoft Corp",
            "Org",
            0.9,
        ));
        let mut track4 = Track::new(0, "Microsoft Corp");
        track4.add_signal(s4, 0);
        doc3.add_track(track4);
        corpus.add_document(doc3);

        // Resolve inter-doc coref
        let _identity_ids = corpus.resolve_inter_doc_coref(0.3, true);

        // Gold standard: Apple tracks should cluster, Microsoft tracks should cluster
        let gold = vec![
            vec![
                TrackRef {
                    doc_id: "doc1".to_string(),
                    track_id: 0,
                },
                TrackRef {
                    doc_id: "doc2".to_string(),
                    track_id: 0,
                },
            ],
            vec![
                TrackRef {
                    doc_id: "doc1".to_string(),
                    track_id: 1,
                },
                TrackRef {
                    doc_id: "doc3".to_string(),
                    track_id: 0,
                },
            ],
        ];

        (corpus, gold)
    }

    #[test]
    fn test_inter_doc_coref_metrics_basic() {
        let (corpus, gold) = create_test_corpus();

        let identity_ids: Vec<_> = corpus
            .identities()
            .filter(|id| {
                matches!(
                    id.source,
                    Some(crate::grounded::IdentitySource::CrossDocCoref { .. })
                )
            })
            .map(|id| id.id)
            .collect();
        let predicted: Vec<_> = identity_ids
            .iter()
            .filter_map(|&id| corpus.get_identity(id))
            .cloned()
            .collect();

        let metrics = InterDocCorefMetrics::compute(&predicted, &gold);

        assert!(metrics.cluster_purity >= 0.0 && metrics.cluster_purity <= 1.0);
        assert!(metrics.cluster_completeness >= 0.0 && metrics.cluster_completeness <= 1.0);
        assert!(metrics.f1() >= 0.0 && metrics.f1() <= 1.0);
    }

    #[test]
    fn test_inter_doc_coref_metrics_empty() {
        let metrics = InterDocCorefMetrics::compute(&[], &[]);
        assert_eq!(metrics.cluster_purity, 0.0);
        assert_eq!(metrics.cluster_completeness, 0.0);
        assert_eq!(metrics.f1(), 0.0);
    }
}
