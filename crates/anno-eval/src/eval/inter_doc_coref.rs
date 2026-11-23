//! Evaluation metrics for inter-document coreference resolution.
//!
//! Provides metrics specific to cross-document entity clustering,
//! complementing the standard coreference metrics in `coref_metrics.rs`.
//!
//! Score a corpus snapshot against one gold partition:
//!
//! ```
//! use anno::{Corpus, TrackRef};
//! use anno_eval::eval::{InterDocCorefMetrics, TrackPartition};
//!
//! fn evaluate(corpus: &Corpus, gold: &[Vec<TrackRef>]) -> anno::Result<InterDocCorefMetrics> {
//!     let current = TrackPartition::from_corpus(corpus)?;
//!     let expected = TrackPartition::from_clusters(gold)?;
//!     Ok(InterDocCorefMetrics::from_partitions(&current, &expected))
//! }
//! ```

use anno::{Corpus, Identity, TrackRef};
use std::collections::{HashMap, HashSet};

/// One assignment of document-scoped tracks to clusters. The cluster labels are
/// local to this partition: prediction and gold labels need not match.
#[derive(Debug, Clone, Default)]
pub struct TrackPartition {
    assignments: HashMap<TrackRef, u64>,
}

impl TrackPartition {
    /// Read the current corpus assignments. Historical identity records do
    /// not add memberships to the scored population.
    pub fn from_corpus(corpus: &Corpus) -> anno::Result<Self> {
        Self::checked(
            corpus
                .current_coref_clusters()
                .into_iter()
                .flat_map(|(id, members)| members.into_iter().map(move |track| (track, id.get()))),
        )
    }

    /// Validate track groups such as a gold annotation or a single saved run.
    /// Repeated tracks, including duplicates within a group, are rejected.
    pub fn from_clusters(clusters: &[Vec<TrackRef>]) -> anno::Result<Self> {
        Self::checked(cluster_assignments(clusters))
    }

    /// Validate the membership of identities exported from a single run.
    /// Both coreference and KB-enriched coreference identities carry members.
    pub fn from_identities(identities: &[Identity]) -> anno::Result<Self> {
        let mut seen = HashSet::new();
        for identity in identities {
            if !seen.insert(identity.id) {
                return Err(anno::Error::invalid_input(format!(
                    "repeated identity ID {}",
                    identity.id
                )));
            }
        }
        Self::checked(identity_assignments(identities))
    }

    fn checked(assignments: impl IntoIterator<Item = (TrackRef, u64)>) -> anno::Result<Self> {
        use std::collections::hash_map::Entry;

        let mut partition = Self::default();
        for (track, cluster) in assignments {
            match partition.assignments.entry(track) {
                Entry::Vacant(entry) => {
                    entry.insert(cluster);
                }
                Entry::Occupied(entry) => {
                    let track = entry.key();
                    return Err(anno::Error::track_ref(format!(
                        "repeated track {} in document {:?} (clusters {} and {cluster})",
                        track.track_id,
                        track.doc_id,
                        entry.get()
                    )));
                }
            }
        }
        Ok(partition)
    }
}

fn identity_assignments(identities: &[Identity]) -> impl Iterator<Item = (TrackRef, u64)> + '_ {
    identities.iter().flat_map(|identity| {
        identity
            .source
            .iter()
            .flat_map(|source| source.track_refs())
            .cloned()
            .map(move |track| (track, identity.id.get()))
    })
}

fn cluster_assignments(clusters: &[Vec<TrackRef>]) -> impl Iterator<Item = (TrackRef, u64)> + '_ {
    clusters.iter().enumerate().flat_map(|(index, members)| {
        members
            .iter()
            .cloned()
            .map(move |track| (track, index as u64))
    })
}

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
    /// Score two validated partitions. Each track in their union contributes
    /// once to the denominator, including tracks missing on either side.
    #[must_use]
    pub fn from_partitions(predicted: &TrackPartition, gold: &TrackPartition) -> Self {
        let mut overlap = HashMap::new();
        let mut num_total = gold.assignments.len();
        for (track, &prediction) in &predicted.assignments {
            if let Some(&target) = gold.assignments.get(track) {
                *overlap.entry((prediction, target)).or_insert(0usize) += 1;
            } else {
                num_total += 1;
            }
        }
        if num_total == 0 {
            return Self::default();
        }

        // A predicted cluster's largest intersection contributes to purity.
        // The largest intersection for each gold cluster contributes to completeness.
        let mut best_prediction = HashMap::new();
        let mut best_gold = HashMap::new();
        for ((prediction, target), count) in overlap {
            let p = best_prediction.entry(prediction).or_insert(0usize);
            *p = (*p).max(count);
            let g = best_gold.entry(target).or_insert(0usize);
            *g = (*g).max(count);
        }
        let num_correct = best_prediction.values().sum::<usize>();
        Self {
            cluster_purity: num_correct as f64 / num_total as f64,
            cluster_completeness: best_gold.values().sum::<usize>() as f64 / num_total as f64,
            num_pred_identities: predicted.assignments.values().collect::<HashSet<_>>().len(),
            num_gold_identities: gold.assignments.values().collect::<HashSet<_>>().len(),
            num_correct,
            num_total,
        }
    }

    /// Score an identity export and gold clusters, rejecting ambiguous or repeated membership.
    pub fn try_compute(predicted: &[Identity], gold: &[Vec<TrackRef>]) -> anno::Result<Self> {
        Ok(Self::from_partitions(
            &TrackPartition::from_identities(predicted)?,
            &TrackPartition::from_clusters(gold)?,
        ))
    }

    /// Compatibility adapter for identity exports. For repeated tracks, the
    /// final assignment in input order wins on each side, as in the original
    /// reverse lookup. Only clusters with assigned tracks count toward the result.
    /// Use [`Self::try_compute`] to detect overlap or [`TrackPartition::from_corpus`]
    /// to select current assignments independently of historical record order.
    #[must_use]
    pub fn compute(predicted: &[Identity], gold: &[Vec<TrackRef>]) -> Self {
        Self::from_partitions(
            &TrackPartition {
                assignments: identity_assignments(predicted).collect(),
            },
            &TrackPartition {
                assignments: cluster_assignments(gold).collect(),
            },
        )
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
    use anno::{GroundedDocument, Location, Signal, Track, TrackId};

    fn create_test_corpus() -> (anno::Corpus, Vec<Vec<TrackRef>>) {
        let mut corpus = anno::Corpus::new();

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
        use anno::coalesce::Resolver;
        let resolver = Resolver::new().with_threshold(0.3).require_type_match(true);
        let _identity_ids = resolver.resolve_inter_doc_coref(&mut corpus, None, None);

        // Gold standard: Apple tracks should cluster, Microsoft tracks should cluster
        let gold = vec![
            vec![
                TrackRef {
                    doc_id: "doc1".to_string(),
                    track_id: TrackId::new(0),
                },
                TrackRef {
                    doc_id: "doc2".to_string(),
                    track_id: TrackId::new(0),
                },
            ],
            vec![
                TrackRef {
                    doc_id: "doc1".to_string(),
                    track_id: TrackId::new(1),
                },
                TrackRef {
                    doc_id: "doc3".to_string(),
                    track_id: TrackId::new(0),
                },
            ],
        ];

        (corpus, gold)
    }

    #[test]
    fn current_partition_survives_kb_enrichment() {
        let (mut corpus, gold) = create_test_corpus();
        let enriched_track = gold[0][0].clone();
        let expected = TrackPartition::from_clusters(&gold).unwrap();
        for enrichment_count in 0..=2 {
            if enrichment_count > 0 {
                corpus
                    .link_track_to_kb(&enriched_track, "test", "apple", "Apple")
                    .unwrap();
            }
            let metrics = InterDocCorefMetrics::from_partitions(
                &TrackPartition::from_corpus(&corpus).unwrap(),
                &expected,
            );
            assert_eq!((metrics.num_correct, metrics.num_total), (4, 4));
            assert_eq!(metrics.f1(), 1.0);
            let records: Vec<_> = corpus.identities().values().cloned().collect();
            assert_eq!(
                InterDocCorefMetrics::try_compute(&records, &gold)
                    .unwrap()
                    .f1(),
                1.0
            );
        }
    }

    #[test]
    fn track_merge_updates_live_and_exported_partitions() {
        let mut document = GroundedDocument::new("d", "Ada Ada");
        let mut inputs = Vec::new();
        for offset in [0, 4] {
            let signal = document.add_signal(Signal::new(
                0,
                Location::text(offset, offset + 3),
                "Ada",
                "PERSON",
                0.9,
            ));
            let mut track = Track::new(0, "Ada");
            track.add_signal(signal, 0);
            inputs.push(document.add_track(track));
        }
        let mut original = Corpus::new();
        original.add_document(document);
        let id =
            anno::coalesce::Resolver::new().resolve_inter_doc_coref(&mut original, None, None)[0];
        let initial = original.get_identity(id).unwrap().clone();
        assert_eq!(initial.minimum_pair_similarity, Some(anno::Confidence::ONE));
        assert!(original.merge_tracks("missing", &inputs).is_none());
        assert!(original.merge_tracks("d", &[]).is_none());
        assert!(original
            .merge_tracks("d", &[inputs[0], inputs[0]])
            .is_none());
        assert_eq!(original.merge_tracks("d", &inputs[..1]), Some(inputs[0]));
        assert_eq!(original.get_identity(id), Some(&initial));

        // An independently serializable document can hold the canonical copy
        // even when none of its own tracks are assigned to this identity.
        let mut mirror = GroundedDocument::new("mirror", "");
        assert_eq!(mirror.add_identity(initial), id);
        original.add_document(mirror);

        for direct_edit in [false, true] {
            let mut corpus = original.clone();
            let merged_id = if direct_edit {
                corpus.get_document_mut("d").unwrap().merge_tracks(&inputs)
            } else {
                corpus.merge_tracks("d", &inputs)
            }
            .unwrap();
            let merged = corpus
                .get_document("d")
                .unwrap()
                .track_ref(merged_id)
                .unwrap();
            let gold = TrackPartition::from_clusters(&[vec![merged.clone()]]).unwrap();
            for enrichment in 0..=2 {
                if enrichment > 0 {
                    assert_eq!(
                        corpus
                            .link_track_to_kb(&merged, "catalog", "ada", "Ada")
                            .unwrap(),
                        id
                    );
                } else if direct_edit {
                    continue;
                }
                let identity = corpus.get_identity(id).unwrap();
                assert_eq!(
                    identity.source.as_ref().unwrap().track_refs(),
                    std::slice::from_ref(&merged)
                );
                assert_eq!(identity.minimum_pair_similarity, None);
                assert_eq!(identity.confidence, anno::Confidence::ZERO);
                for doc in corpus.documents() {
                    assert_eq!(doc.get_identity(id), Some(identity));
                    assert!(doc.invariants_hold());
                }
                for predicted in [
                    TrackPartition::from_corpus(&corpus).unwrap(),
                    TrackPartition::from_identities(std::slice::from_ref(identity)).unwrap(),
                ] {
                    let metrics = InterDocCorefMetrics::from_partitions(&predicted, &gold);
                    assert_eq!((metrics.num_correct, metrics.num_total), (1, 1));
                    assert_eq!(metrics.f1(), 1.0);
                }
            }
        }
    }

    #[test]
    fn rejects_overlapping_partitions_and_bounds_legacy_scores() {
        let (corpus, gold) = create_test_corpus();
        let identity = corpus.identities().values().next().unwrap().clone();
        let members = identity.source.as_ref().unwrap().track_refs().to_vec();
        let mut historical_copy = identity.clone();
        historical_copy.id = anno::IdentityId::new(999);
        let mut disjoint_copy = identity.clone();
        disjoint_copy.source = None;
        assert!(TrackPartition::from_identities(&[identity.clone(), disjoint_copy]).is_err());
        let overlapping = [identity, historical_copy];
        assert!(TrackPartition::from_identities(&overlapping).is_err());
        assert!(TrackPartition::from_clusters(&[members.clone(), members.clone()]).is_err());
        assert!(TrackPartition::from_clusters(&[vec![members[0].clone(); 2]]).is_err());
        assert!(InterDocCorefMetrics::try_compute(&overlapping, &gold).is_err());

        let metrics = InterDocCorefMetrics::compute(&overlapping, std::slice::from_ref(&members));
        assert_eq!(
            (metrics.num_correct, metrics.num_total),
            (members.len(), members.len())
        );
        assert_eq!(metrics.num_pred_identities, 1);
        assert_eq!(metrics.f1(), 1.0);
    }

    #[test]
    fn purity_and_completeness_use_unique_tracks_including_unmatched() {
        let (_, gold) = create_test_corpus();
        // Split a gold cluster and omit the second gold cluster.
        let predicted = vec![vec![gold[0][0].clone()], vec![gold[0][1].clone()]];
        let metrics = InterDocCorefMetrics::from_partitions(
            &TrackPartition::from_clusters(&predicted).unwrap(),
            &TrackPartition::from_clusters(&gold).unwrap(),
        );
        assert_eq!(metrics.num_correct, 2);
        assert_eq!(metrics.num_total, 4);
        assert_eq!(metrics.cluster_purity, 0.5);
        assert_eq!(metrics.cluster_completeness, 0.25);
    }

    #[test]
    fn test_inter_doc_coref_metrics_empty() {
        let metrics = InterDocCorefMetrics::compute(&[], &[]);
        assert_eq!(metrics.cluster_purity, 0.0);
        assert_eq!(metrics.cluster_completeness, 0.0);
        assert_eq!(metrics.f1(), 0.0);
    }
}
