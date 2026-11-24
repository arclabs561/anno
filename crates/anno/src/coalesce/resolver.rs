//! # Batch Entity Resolution with Union-Find
//!
//! This module provides the core batch entity resolution algorithm using the
//! **disjoint-set (union-find)** data structure with path compression and
//! union-by-rank.
//!
//! ## Algorithm Overview
//!
//! 1. **Collect** all tracks from all documents in the corpus
//! 2. **Compare** pairs: compute similarity (embedding cosine or string Jaccard)
//! 3. **Cluster** using union-find: if \( \text{sim}(t_i, t_j) \geq \theta \), merge
//! 4. **Create** identities: one per cluster, linked to constituent tracks
//!
//! ## Complexity Analysis
//!
//! Let \( n \) = number of tracks, \( m \) = number of merges performed.
//!
//! - **Pairwise comparison:** \( O(n^2) \) — the bottleneck for large corpora
//! - **Union-find operations:** \( O(m \cdot \alpha(n)) \) where \( \alpha \) is
//!   the inverse Ackermann function
//!
//! For all practical \( n \), \( \alpha(n) \leq 4 \), so union-find is effectively
//! \( O(m) \). The overall complexity is dominated by pairwise comparison.
//!
//! ## The Inverse Ackermann Function
//!
//! The Ackermann function \( A(m, n) \) grows faster than any primitive recursive
//! function. Its inverse \( \alpha(n) \) grows so slowly that:
//!
//! - \( \alpha(10^{80}) \leq 4 \) (more atoms than in the observable universe)
//!
//! Tarjan (1975) proved this bound is tight for union-find with path compression
//! and union-by-rank.
//!
//! ## Similarity Metrics
//!
//! Two metrics are provided:
//!
//! - **Embedding similarity** (cosine): \( \cos(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{||\mathbf{a}|| \cdot ||\mathbf{b}||} \)
//! - **String similarity** (Jaccard on words): \( J(A, B) = \frac{|A \cap B|}{|A \cup B|} \)
//!
//! Compatible nonempty embeddings use cosine. Other pairs use Jaccard on word sets.
//!
//! ## Example
//!
//! ```
//! use anno::coalesce::Resolver;
//! use anno::Corpus;
//!
//! let resolver = Resolver::new()
//!     .with_threshold(0.7)
//!     .require_type_match(true);
//!
//! let mut corpus = Corpus::new();
//! // ... add documents with tracks ...
//!
//! let identities = resolver.resolve_inter_doc_coref(&mut corpus, None, None);
//! ```

use crate::core::grounded::CorefTrackType;
use crate::core::{Corpus, Identity, IdentityId, IdentitySource, TrackRef};
use std::collections::{BTreeMap, HashMap};

/// Coalescer for inter-document entity resolution.
#[derive(Debug, Clone)]
pub struct Resolver {
    similarity_threshold: f32,
    require_type_match: bool,
}

impl Resolver {
    /// Create a new resolver with default settings.
    pub fn new() -> Self {
        Self {
            similarity_threshold: 0.7,
            require_type_match: true,
        }
    }

    /// Create a new resolver with custom settings.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.similarity_threshold = threshold;
        self
    }

    /// Require equal track types for clustering. An explicit track type takes precedence.
    /// Otherwise all referenced signals must agree. Empty untyped tracks can match each
    /// other. Conflicting, blank, or missing signal labels leave a track unpaired.
    pub fn require_type_match(mut self, require: bool) -> Self {
        self.require_type_match = require;
        self
    }

    /// Coalesce inter-document entities across all documents in a corpus.
    ///
    /// Cluster by thresholded pairwise similarity and take the transitive closure.
    /// Tracks with KB links stay assigned to their existing identities. The resolver
    /// evaluates the remaining tracks again, reusing IDs for unchanged memberships.
    /// Newly similar tracks can join an unanchored cluster on subsequent runs.
    /// The first current member in document/track order supplies the name, type, and
    /// usable embedding of an identity this resolver created. Imported and KB-linked
    /// records retain their representative metadata.
    /// Returns the current coreference IDs, including KB-linked clusters from earlier
    /// runs, in deterministic order. A singleton has no measured pairwise support.
    pub fn resolve_inter_doc_coref(
        &self,
        corpus: &mut Corpus,
        similarity_threshold: Option<f32>,
        require_type_match: Option<bool>,
    ) -> Vec<IdentityId> {
        let threshold = similarity_threshold.unwrap_or(self.similarity_threshold);
        let type_match = require_type_match.unwrap_or(self.require_type_match);

        // 1. Collect all track data (clone what we need to avoid borrow conflicts)
        #[derive(Debug, Clone)]
        struct TrackData {
            track_ref: TrackRef,
            canonical_surface: String,
            entity_type: CorefTrackType,
            embedding: Option<Vec<f32>>,
        }

        fn similarity(a: &TrackData, b: &TrackData) -> f32 {
            match (&a.embedding, &b.embedding) {
                (Some(left), Some(right)) if !left.is_empty() && left.len() == right.len() => {
                    embedding_similarity(left, right)
                }
                _ => string_similarity(&a.canonical_surface, &b.canonical_surface),
            }
        }

        let mut track_data = Vec::new();
        for doc in corpus.documents() {
            for track in doc.tracks() {
                if doc
                    .identity_for_track(track.id)
                    .is_some_and(|identity| identity.kb_id.is_some())
                {
                    continue;
                }
                track_data.push(TrackData {
                    track_ref: TrackRef {
                        doc_id: doc.id().to_owned(),
                        track_id: track.id,
                    },
                    canonical_surface: track.canonical_surface.clone(),
                    entity_type: track.coref_type(doc),
                    embedding: track.coref_embedding().cloned(),
                });
            }
        }
        track_data.sort_by(|a, b| {
            (&a.track_ref.doc_id, a.track_ref.track_id)
                .cmp(&(&b.track_ref.doc_id, b.track_ref.track_id))
        });
        let existing_coref: Vec<_> = corpus
            .identities()
            .values()
            .filter_map(|identity| {
                identity
                    .source
                    .as_ref()
                    .filter(|source| !source.track_refs().is_empty())
                    .map(|_| identity.id)
            })
            .collect();
        let previous = corpus.current_coref_clusters();
        let by_members: HashMap<_, _> = previous
            .iter()
            .map(|(id, members)| (members.as_slice(), *id))
            .collect();

        // 2. Cluster tracks using string similarity or embeddings
        // Uses embeddings if available (from track.embedding), otherwise falls back to string similarity
        let mut union_find: Vec<usize> = (0..track_data.len()).collect();
        let mut rank: Vec<usize> = vec![0; track_data.len()];

        fn find(parent: &mut [usize], i: usize) -> usize {
            if parent[i] != i {
                parent[i] = find(parent, parent[i]);
            }
            parent[i]
        }

        fn union(parent: &mut [usize], rank: &mut [usize], i: usize, j: usize) {
            let pi = find(parent, i);
            let pj = find(parent, j);
            if pi != pj {
                match rank[pi].cmp(&rank[pj]) {
                    std::cmp::Ordering::Less => parent[pi] = pj,
                    std::cmp::Ordering::Greater => parent[pj] = pi,
                    std::cmp::Ordering::Equal => {
                        parent[pi] = pj;
                        rank[pj] += 1;
                    }
                }
            }
        }

        // Compare all pairs
        for i in 0..track_data.len() {
            for j in (i + 1)..track_data.len() {
                let track_a = &track_data[i];
                let track_b = &track_data[j];

                // Type check
                if type_match && !track_a.entity_type.matches(&track_b.entity_type) {
                    continue;
                }

                if similarity(track_a, track_b) >= threshold {
                    union(&mut union_find, &mut rank, i, j);
                }
            }
        }

        // 3. Build clusters
        let mut cluster_map: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for i in 0..track_data.len() {
            let root = find(&mut union_find, i);
            cluster_map.entry(root).or_default().push(i);
        }

        // 4. Create identities for each cluster
        // Note: Singleton clusters (clusters with only one track) still create identities.
        // This allows tracking entities that appear only once across documents.
        for members in cluster_map.values() {
            let first = &track_data[members[0]];
            let refs: Vec<_> = members
                .iter()
                .map(|&i| track_data[i].track_ref.clone())
                .collect();
            let minimum = members
                .iter()
                .enumerate()
                .flat_map(|(position, left)| {
                    let left = &track_data[*left];
                    let data = &track_data;
                    members[position + 1..]
                        .iter()
                        .map(move |right| similarity(left, &data[*right]))
                })
                .min_by(f32::total_cmp)
                .map(crate::Confidence::from);
            let score = minimum.unwrap_or(crate::Confidence::ZERO);
            if let Some(id) = by_members.get(refs.as_slice()) {
                let mut identity = corpus.get_identity(*id).expect("current identity").clone();
                identity.minimum_pair_similarity = minimum;
                identity.confidence = score;
                identity.source = Some(IdentitySource::CrossDocCoref { track_refs: refs });
                corpus.update_shared_identity(identity);
                continue;
            }

            let mut identity = Identity::new(IdentityId::ZERO, &first.canonical_surface);
            identity.entity_type = first.entity_type.clone().into_label();
            identity.embedding = first.embedding.clone();
            identity.minimum_pair_similarity = minimum;
            identity.confidence = score;
            identity.source = Some(IdentitySource::CrossDocCoref {
                track_refs: refs.clone(),
            });
            if corpus
                .add_resolved_identity_for_tracks(identity, &refs)
                .is_none()
            {
                log::warn!("Could not bind coalesced identity for tracks {:?}", refs);
            }
        }

        corpus.refresh_coref_membership(&existing_coref);
        corpus
            .current_coref_clusters()
            .into_iter()
            .map(|(id, _)| id)
            .collect()
    }
}

impl Default for Resolver {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute string similarity using Jaccard similarity on word sets.
///
/// Returns a value in [0.0, 1.0] where 1.0 is identical.
///
/// # Example
///
/// ```rust
/// use anno::coalesce::resolver::string_similarity;
///
/// let sim = string_similarity("Lynn Conway", "Lynn Conway");
/// assert_eq!(sim, 1.0);
///
/// let sim = string_similarity("Lynn Conway", "Conway");
/// assert!(sim > 0.0); // "Conway" shares one word with "Lynn Conway"
/// ```
pub fn string_similarity(a: &str, b: &str) -> f32 {
    // Normalize words: lowercase, strip possessives
    fn normalize_word(w: &str) -> String {
        let lower = w.to_lowercase();
        lower
            .trim_end_matches("'s")
            .trim_end_matches("'s")
            .trim_end_matches('\'')
            .to_string()
    }

    // Simple Jaccard similarity on normalized word sets
    let words_a: std::collections::HashSet<String> =
        a.split_whitespace().map(normalize_word).collect();
    let words_b: std::collections::HashSet<String> =
        b.split_whitespace().map(normalize_word).collect();

    if words_a.is_empty() && words_b.is_empty() {
        return 1.0;
    }
    if words_a.is_empty() || words_b.is_empty() {
        return 0.0;
    }

    let intersection = words_a.intersection(&words_b).count();
    let union = words_a.union(&words_b).count();

    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

/// Compute embedding similarity using cosine similarity.
///
/// Returns a value in [0.0, 1.0] where 1.0 is identical.
///
/// Formula: `cosine(a, b) = (a · b) / (||a|| × ||b||)`, normalized to [0, 1].
/// Measures angle between vectors, not magnitude, making it suitable for embeddings.
///
/// # Example
///
/// ```rust
/// use anno::coalesce::resolver::embedding_similarity;
///
/// let emb1 = vec![1.0, 0.0, 0.0];
/// let emb2 = vec![1.0, 0.0, 0.0];
/// let sim = embedding_similarity(&emb1, &emb2);
/// assert_eq!(sim, 1.0);
/// ```
pub fn embedding_similarity(emb_a: &[f32], emb_b: &[f32]) -> f32 {
    if emb_a.len() != emb_b.len() || emb_a.is_empty() {
        return 0.0;
    }

    // Accumulate in f64 so finite f32 coordinates can be squared safely.
    let mut dot = 0.0_f64;
    let mut square_a = 0.0_f64;
    let mut square_b = 0.0_f64;
    for (&a, &b) in emb_a.iter().zip(emb_b) {
        if !a.is_finite() || !b.is_finite() {
            return 0.0;
        }
        let (a, b) = (f64::from(a), f64::from(b));
        dot += a * b;
        square_a += a * a;
        square_b += b * b;
    }
    if square_a == 0.0 || square_b == 0.0 {
        return 0.0;
    }
    ((dot / (square_a.sqrt() * square_b.sqrt()) + 1.0) / 2.0).clamp(0.0, 1.0) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GroundedDocument, Location, Signal, Track, TrackId};

    fn document_with_linked_local_identity(
        doc_id: &str,
        text: &str,
    ) -> (GroundedDocument, TrackId, crate::SignalId) {
        let mut document = GroundedDocument::new(doc_id, text);
        let signal_id = document.add_signal(Signal::new(
            0,
            Location::text(0, text.chars().count()),
            text,
            "PERSON",
            0.9,
        ));
        let track_id = document.add_track(Track::new(0, text));
        assert!(document.add_signal_to_track(signal_id, track_id, 0));
        let local_identity_id = document.add_identity(Identity::new(0, "local identity"));
        document.link_track_to_identity(track_id, local_identity_id);
        (document, track_id, signal_id)
    }

    #[test]
    fn test_string_similarity_identical() {
        assert_eq!(string_similarity("hello world", "hello world"), 1.0);
    }

    #[test]
    fn test_string_similarity_partial() {
        let sim = string_similarity("hello world", "hello");
        // Jaccard over normalized word sets: {"hello", "world"} vs {"hello"}.
        // intersection = 1, union = 2, sim = 1 / 2 exactly.
        assert_eq!(sim, 0.5);
    }

    #[test]
    fn test_string_similarity_empty() {
        assert_eq!(string_similarity("", ""), 1.0);
        assert_eq!(string_similarity("hello", ""), 0.0);
        assert_eq!(string_similarity("", "hello"), 0.0);
    }

    #[test]
    fn test_string_similarity_symmetric() {
        let sim_ab = string_similarity("hello world", "world peace");
        let sim_ba = string_similarity("world peace", "hello world");
        assert_eq!(sim_ab, sim_ba);
    }

    #[test]
    fn test_embedding_similarity_identical() {
        let emb = vec![1.0, 0.0, 0.0];
        assert_eq!(embedding_similarity(&emb, &emb), 1.0);
    }

    #[test]
    fn test_embedding_similarity_orthogonal() {
        let emb1 = vec![1.0, 0.0];
        let emb2 = vec![0.0, 1.0];
        // Orthogonal vectors have cosine = 0, normalized to 0.5
        let sim = embedding_similarity(&emb1, &emb2);
        assert!((sim - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_embedding_similarity_opposite() {
        let emb1 = vec![1.0, 0.0];
        let emb2 = vec![-1.0, 0.0];
        // Opposite vectors have cosine = -1, normalized to 0.0
        let sim = embedding_similarity(&emb1, &emb2);
        assert!((sim - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_embedding_similarity_mismatched_length() {
        let emb1 = vec![1.0, 0.0];
        let emb2 = vec![1.0, 0.0, 0.0];
        assert_eq!(embedding_similarity(&emb1, &emb2), 0.0);
    }

    #[test]
    fn test_embedding_similarity_empty() {
        let emb1: Vec<f32> = vec![];
        let emb2: Vec<f32> = vec![];
        assert_eq!(embedding_similarity(&emb1, &emb2), 0.0);
    }

    #[test]
    fn test_embedding_similarity_zero_norm() {
        let emb1 = vec![0.0, 0.0];
        let emb2 = vec![1.0, 0.0];
        assert_eq!(embedding_similarity(&emb1, &emb2), 0.0);
    }

    #[test]
    fn test_resolver_builder() {
        let resolver = Resolver::new()
            .with_threshold(0.8)
            .require_type_match(false); // Default is true, test setting to false

        assert_eq!(resolver.similarity_threshold, 0.8);
        assert!(!resolver.require_type_match);
    }

    #[test]
    fn test_resolver_default() {
        let resolver = Resolver::default();
        assert_eq!(resolver.similarity_threshold, 0.7);
        assert!(resolver.require_type_match); // Default is true
    }

    #[test]
    fn corpus_kb_link_propagates_a_collision_safe_identity_to_document() {
        let (document, track_id, signal_id) =
            document_with_linked_local_identity("ada", "Ada Lovelace");
        let mut corpus = Corpus::new();
        corpus.add_document(document);
        let track_ref = TrackRef {
            doc_id: "ada".to_string(),
            track_id,
        };

        let identity_id = corpus
            .link_track_to_kb(&track_ref, "wikidata", "Q7259", "Ada Lovelace")
            .unwrap();
        assert_ne!(identity_id, IdentityId::ZERO);
        assert_eq!(
            corpus.get_identity(identity_id).unwrap().kb_id.as_deref(),
            Some("Q7259")
        );

        let document = corpus.get_document("ada").unwrap();
        assert_eq!(
            document.get_track(track_id).unwrap().identity_id,
            Some(identity_id)
        );
        assert_eq!(
            document.identity_for_track(track_id).unwrap().id,
            identity_id
        );
        assert_eq!(
            document.identity_for_signal(signal_id).unwrap().id,
            identity_id
        );
        assert!(document.invariants_hold());
    }

    #[test]
    fn kb_link_does_not_reuse_a_post_insertion_local_id_collision() {
        let mut corpus = Corpus::new();
        let mut first = GroundedDocument::new("first", "Ada Lovelace");
        let first_track = first.add_track(Track::new(0, "Ada Lovelace"));
        corpus.add_document(first);
        let first_ref = TrackRef {
            doc_id: "first".to_string(),
            track_id: first_track,
        };
        let first_identity = corpus
            .link_track_to_kb(&first_ref, "wikidata", "Q7259", "Ada Lovelace")
            .unwrap();

        corpus.add_document(GroundedDocument::new("second", "Grace Hopper"));
        let second_track = {
            let second = corpus.get_document_mut("second").unwrap();
            let track_id = second.add_track(Track::new(0, "Grace Hopper"));
            let colliding_local_id = second.add_identity(Identity::new(0, "local Grace"));
            assert_eq!(colliding_local_id, first_identity);
            second.link_track_to_identity(track_id, colliding_local_id);
            track_id
        };
        let second_ref = TrackRef {
            doc_id: "second".to_string(),
            track_id: second_track,
        };

        let second_identity = corpus
            .link_track_to_kb(&second_ref, "wikidata", "Q11641", "Grace Hopper")
            .unwrap();
        assert_ne!(second_identity, first_identity);
        assert_eq!(
            corpus.get_identity(first_identity).unwrap().canonical_name,
            "Ada Lovelace"
        );
        assert_eq!(
            corpus.get_identity(second_identity).unwrap().canonical_name,
            "Grace Hopper"
        );

        let first = corpus.get_document("first").unwrap();
        let second = corpus.get_document("second").unwrap();
        assert_eq!(
            first
                .identity_for_track(first_track)
                .unwrap()
                .canonical_name,
            "Ada Lovelace"
        );
        assert_eq!(
            second.get_track(second_track).unwrap().identity_id,
            Some(second_identity)
        );
        assert!(first.invariants_hold());
        assert!(second.invariants_hold());
    }

    #[test]
    fn coalescing_links_document_mirrors_without_overwriting_local_identities() {
        let (first, first_track, first_signal) =
            document_with_linked_local_identity("first", "Ada Lovelace");
        let (second, second_track, second_signal) =
            document_with_linked_local_identity("second", "Ada Lovelace");
        let mut corpus = Corpus::new();
        corpus.add_document(first);
        corpus.add_document(second);

        let identity_ids = Resolver::new().resolve_inter_doc_coref(&mut corpus, None, None);
        assert_eq!(identity_ids.len(), 1);
        let identity_id = identity_ids[0];
        assert_ne!(identity_id, IdentityId::ZERO);

        for (doc_id, track_id, signal_id) in [
            ("first", first_track, first_signal),
            ("second", second_track, second_signal),
        ] {
            let document = corpus.get_document(doc_id).unwrap();
            assert_eq!(
                document.get_track(track_id).unwrap().identity_id,
                Some(identity_id)
            );
            assert_eq!(
                document.identity_for_signal(signal_id).unwrap().id,
                identity_id
            );
            assert!(document.invariants_hold());
        }
    }

    fn add_track_doc(corpus: &mut Corpus, doc_id: &str, surface: &str) -> TrackRef {
        let mut doc = GroundedDocument::new(doc_id, surface);
        let track = doc.add_track(Track::new(TrackId::ZERO, surface));
        corpus.add_document(doc);
        corpus
            .get_document(doc_id)
            .unwrap()
            .track_ref(track)
            .unwrap()
    }

    fn observed_track(doc_id: &str, surface: &str, labels: &[&str]) -> GroundedDocument {
        let text = vec![surface; labels.len()].join(" ");
        let mut doc = GroundedDocument::new(doc_id, &text);
        if labels.is_empty() {
            doc.add_track(Track::new(0, surface));
        } else {
            let signal_ids: Vec<_> = labels
                .iter()
                .enumerate()
                .map(|(n, &label)| {
                    let start = n * (surface.chars().count() + 1);
                    doc.add_signal(Signal::new(
                        0,
                        Location::text(start, start + surface.chars().count()),
                        surface,
                        label,
                        1.0,
                    ))
                })
                .collect();
            doc.create_track_from_signals(surface, &signal_ids).unwrap();
        }
        doc
    }

    #[test]
    fn type_guard_uses_unanimous_signal_types_and_respects_explicit_track_types() {
        let mut corpus = Corpus::new();
        for (id, labels) in [
            ("person", &["PER", "PERSON"][..]),
            ("persisted", &["PER"]),
            ("org", &["ORG"]),
            ("override", &["PER"]),
            ("mixed", &["PER", "ORG"]),
            ("mixed2", &["ORG", "PER"]),
            ("blank", &["  "]),
            ("missing", &["PER"]),
            ("empty", &[]),
            ("empty2", &[]),
        ] {
            let mut doc = observed_track(id, "Morgan", labels);
            let track = doc.tracks_map_mut().get_mut(&TrackId::ZERO).unwrap();
            if id == "override" {
                track.entity_type = Some("ORG".into());
            } else if id == "missing" {
                track.add_signal(99, 1);
            } else if id == "persisted" {
                let mut wire = serde_json::to_value(&doc).unwrap();
                wire["tracks"]["0"]
                    .as_object_mut()
                    .unwrap()
                    .remove("entity_type");
                doc = serde_json::from_value(wire).unwrap();
            }
            corpus.add_document(doc);
        }
        let resolver = Resolver::new();
        let clusters = resolver.resolve_inter_doc_coref(&mut corpus, None, None);
        let identity = |doc_id: &str| {
            corpus
                .get_document(doc_id)
                .unwrap()
                .identity_for_track(TrackId::ZERO)
                .unwrap()
        };
        assert_eq!(identity("person").entity_type, Some("PER".into()));
        assert_eq!(identity("person").id, identity("persisted").id);
        assert_ne!(identity("person").id, identity("org").id);
        assert_eq!(identity("org").id, identity("override").id);
        assert_eq!(identity("empty").id, identity("empty2").id);
        assert_eq!(clusters.len(), 7);
        for id in ["mixed", "mixed2", "blank", "missing"] {
            let identity = identity(id);
            assert!(identity.entity_type.is_none());
            assert_eq!(identity.source.as_ref().unwrap().track_refs().len(), 1);
        }
        assert_eq!(
            resolver
                .resolve_inter_doc_coref(&mut corpus, None, Some(false))
                .len(),
            1
        );
    }

    #[test]
    fn withdrawal_and_retry_publish_the_current_representative() {
        let mut corpus = Corpus::new();
        let resolver = Resolver::new().require_type_match(false);
        for (doc_id, name, label, embedding) in [
            ("a", "Morgan Lee", "PER", vec![1., 0.]),
            ("z", "Lee", "ORG", vec![0.9, 0.2]),
        ] {
            let mut doc = observed_track(doc_id, name, &[label]);
            doc.tracks_map_mut()
                .get_mut(&TrackId::ZERO)
                .unwrap()
                .embedding = Some(embedding);
            corpus.add_document(doc);
        }
        let ids = resolver.resolve_inter_doc_coref(&mut corpus, None, None);
        assert_eq!(ids.len(), 1);
        assert_eq!(
            corpus.get_identity(ids[0]).unwrap().canonical_name,
            "Morgan Lee"
        );
        corpus.add_document(GroundedDocument::new("a", ""));
        let survivor = corpus.get_document("z").unwrap();
        let remaining = corpus.get_identity(ids[0]).unwrap();
        assert_eq!(remaining.canonical_name, "Lee");
        assert_eq!(remaining.entity_type, Some("ORG".into()));
        assert_eq!(remaining.embedding, Some(vec![0.9, 0.2]));
        assert_eq!(remaining.minimum_pair_similarity, None);
        assert_eq!(survivor.identity_for_track(TrackId::ZERO), Some(remaining));
        let survivor = corpus
            .get_document_mut("z")
            .unwrap()
            .tracks_map_mut()
            .get_mut(&TrackId::ZERO)
            .unwrap();
        survivor.canonical_surface = "Lee Group".into();
        survivor.entity_type = Some("GROUP".into());
        survivor.embedding = Some(vec![0., 0.]);
        assert_eq!(
            resolver.resolve_inter_doc_coref(&mut corpus, None, None),
            ids
        );
        let remaining = corpus.get_identity(ids[0]).unwrap();
        assert_eq!(remaining.canonical_name, "Lee Group");
        assert_eq!(remaining.entity_type, Some("GROUP".into()));
        assert_eq!(remaining.embedding, None);
        assert_eq!(
            corpus
                .get_document("z")
                .unwrap()
                .identity_for_track(TrackId::ZERO),
            Some(remaining)
        );
    }

    #[test]
    fn retries_and_withdrawals_preserve_curated_representatives() {
        let resolver = Resolver::new();
        for catalog_owned in [false, true] {
            let mut corpus = Corpus::new();
            let first = add_track_doc(&mut corpus, "a", "Morgan");
            let second = add_track_doc(&mut corpus, "b", "Morgan");
            let members = vec![first.clone(), second.clone()];
            let id = if catalog_owned {
                resolver.resolve_inter_doc_coref(&mut corpus, None, None);
                corpus
                    .link_track_to_kb(&first, "catalog", "42", "Curated Morgan")
                    .unwrap()
            } else {
                let mut curated = Identity::new(0, "Curated Morgan");
                curated.entity_type = Some("curated_type".into());
                curated.embedding = Some(vec![3., 2.]);
                curated.aliases.push("Editorial alias".into());
                curated.source = Some(IdentitySource::CrossDocCoref {
                    track_refs: members.clone(),
                });
                let id = corpus.add_identity(curated.clone());
                for member in &members {
                    let doc = corpus.get_document_mut(&member.doc_id).unwrap();
                    assert_eq!(doc.add_identity(curated.clone()), id);
                    doc.link_track_to_identity(member.track_id, id);
                }
                id
            };
            let before = corpus.get_identity(id).unwrap().clone();
            assert_eq!(
                resolver.resolve_inter_doc_coref(&mut corpus, None, None),
                vec![id]
            );
            corpus.add_document(GroundedDocument::new("a", ""));
            assert_eq!(
                resolver.resolve_inter_doc_coref(&mut corpus, None, None),
                vec![id]
            );
            let current = corpus.get_identity(id).unwrap();
            assert_eq!(current.source.as_ref().unwrap().track_refs(), &[second]);
            assert_eq!(current.canonical_name, before.canonical_name);
            assert_eq!(current.entity_type, before.entity_type);
            assert_eq!(current.embedding, before.embedding);
            assert_eq!(current.aliases, before.aliases);
            assert_eq!(current.qualified_kb_ref(), before.qualified_kb_ref());
            assert_eq!(
                corpus
                    .get_document("b")
                    .unwrap()
                    .identity_for_track(TrackId::ZERO),
                Some(current)
            );
        }
    }

    #[test]
    fn retries_reuse_current_identity_and_singleton_has_no_pair_score() {
        let mut corpus = Corpus::new();
        add_track_doc(&mut corpus, "a", "Ada");
        let resolver = Resolver::new();
        let first = resolver.resolve_inter_doc_coref(&mut corpus, None, None);
        assert_eq!(first.len(), 1);
        assert_eq!(
            resolver.resolve_inter_doc_coref(&mut corpus, None, None),
            first
        );
        assert_eq!(corpus.identities().len(), 1);
        assert_eq!(
            corpus
                .get_identity(first[0])
                .unwrap()
                .minimum_pair_similarity,
            None
        );

        add_track_doc(&mut corpus, "b", "Ada Byron");
        add_track_doc(&mut corpus, "c", "Byron");
        let merged = resolver.resolve_inter_doc_coref(&mut corpus, Some(0.5), None);
        assert_eq!(merged.len(), 1);
        assert_eq!(corpus.identities().len(), 1);
        let id = corpus.get_identity(merged[0]).unwrap();
        assert_eq!(id.minimum_pair_similarity, Some(crate::Confidence::ZERO));
        assert_eq!(id.source.as_ref().unwrap().track_refs().len(), 3);
        let split = resolver.resolve_inter_doc_coref(&mut corpus, Some(0.9), None);
        assert_eq!(split.len(), 3);
        assert!(corpus.get_identity(merged[0]).is_none());
        assert!(corpus.documents().all(|doc| doc.invariants_hold()));
    }

    #[test]
    fn resolving_preserves_anchors_and_repeated_kb_updates_preserve_membership() {
        let mut corpus = Corpus::new();
        let ada = add_track_doc(&mut corpus, "a", "Ada");
        let resolver = Resolver::new();
        let first = resolver.resolve_inter_doc_coref(&mut corpus, None, None)[0];
        for _ in 0..2 {
            assert_eq!(
                corpus.link_track_to_kb(&ada, "kb", "a", "Ada").unwrap(),
                first
            );
            let id = corpus.get_identity(first).unwrap();
            assert!(
                matches!(&id.source, Some(IdentitySource::Hybrid { track_refs, .. }) if track_refs == std::slice::from_ref(&ada))
            );
        }
        let grace = add_track_doc(&mut corpus, "b", "Grace");
        let grace_id = corpus.link_track_to_kb(&grace, "kb", "g", "Grace").unwrap();
        assert_eq!(
            resolver.resolve_inter_doc_coref(&mut corpus, Some(0.0), None),
            vec![first]
        );
        assert_eq!(corpus.get_identity(grace_id).unwrap().kb_id(), Some("g"));
        assert_eq!(corpus.current_coref_clusters(), vec![(first, vec![ada])]);
    }

    #[test]
    fn replacing_a_document_retires_old_track_handles_and_updates_provenance() {
        let mut corpus = Corpus::new();
        let first = add_track_doc(&mut corpus, "a", "Ada");
        let second = add_track_doc(&mut corpus, "b", "Ada");
        let id = Resolver::new().resolve_inter_doc_coref(&mut corpus, None, None)[0];
        assert_eq!(
            corpus.get_identity(id).unwrap().minimum_pair_similarity,
            Some(crate::Confidence::ONE)
        );
        let mut replacement = GroundedDocument::new("a", "Grace");
        replacement.add_track(Track::new(TrackId::ZERO, "Grace"));
        corpus.add_document(replacement);
        let doc = corpus.get_document("a").unwrap();
        assert!(doc.get_track(first.track_id).is_none());
        assert!(corpus.link_track_to_kb(&first, "kb", "ada", "Ada").is_err());
        assert_eq!(
            corpus.current_coref_clusters(),
            vec![(id, vec![second.clone()])]
        );
        let remaining = corpus.get_identity(id).unwrap();
        assert_eq!(remaining.source.as_ref().unwrap().track_refs(), &[second]);
        assert_eq!(remaining.minimum_pair_similarity, None);
        assert_eq!(
            Some(remaining),
            corpus.get_document("b").unwrap().get_identity(id)
        );
        corpus.add_document(GroundedDocument::new("a", ""));
        let new_id = corpus
            .get_document_mut("a")
            .unwrap()
            .add_track(Track::new(TrackId::ZERO, "New"));
        assert!(new_id > first.track_id);
        corpus.add_document(GroundedDocument::new("b", ""));
        assert!(corpus.get_identity(id).is_none());
    }

    #[test]
    fn replacement_track_floor_survives_local_identity_collision_and_empty_document() {
        let mut corpus = Corpus::new();
        let original = add_track_doc(&mut corpus, "same-id", "Ada");
        let corpus_id = corpus.add_identity(Identity::new(0, "corpus"));
        let mut replacement = GroundedDocument::new("same-id", "");
        let local_id = replacement.add_identity(Identity::new(0, "local"));
        assert_eq!(local_id, corpus_id);
        corpus.add_document(replacement);
        let doc = corpus.get_document_mut("same-id").unwrap();
        assert!(doc.get_identity(local_id).is_none());
        doc.rebuild_indexes();
        let fresh = doc.add_track(Track::new(0, "Grace"));
        assert!(fresh > original.track_id);
        assert!(doc.invariants_hold());
    }

    #[test]
    fn replacement_preserves_sparse_ids_and_prunes_imported_local_provenance() {
        let mut corpus = Corpus::new();
        let retired = add_track_doc(&mut corpus, "d", "Ada");
        let mut other = GroundedDocument::new("other", "Ada");
        let mut local = Identity::new(0, "local");
        local.source = Some(IdentitySource::CrossDocCoref {
            track_refs: vec![retired.clone()],
        });
        let local_id = other.add_identity(local);
        corpus.add_document(other);

        let mut replacement = GroundedDocument::new("d", "");
        replacement.add_track(Track::new(0, "new"));
        let sparse = TrackId::new(u64::MAX - 1);
        replacement
            .tracks_map_mut()
            .insert(sparse, Track::new(sparse, "sparse"));
        replacement.rebuild_indexes();
        corpus.add_document(replacement);
        let doc = corpus.get_document("d").unwrap();
        assert!(doc.get_track(retired.track_id).is_none());
        assert!(doc.get_track(1).is_some());
        assert!(doc.get_track(sparse).is_some());
        assert!(doc.invariants_hold());
        let other = corpus.get_document("other").unwrap();
        assert!(other.get_identity(local_id).unwrap().source.is_none());
    }

    #[test]
    fn corpus_merge_updates_local_owner_and_drops_retired_imported_references() {
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
        let refs: Vec<_> = inputs
            .iter()
            .map(|&id| document.track_ref(id).unwrap())
            .collect();
        let mut owner = Identity::new(0, "local Ada");
        owner.source = Some(IdentitySource::CrossDocCoref {
            track_refs: refs.clone(),
        });
        owner.minimum_pair_similarity = Some(crate::Confidence::ONE);
        let owner_id = document.add_identity(owner);
        for &id in &inputs {
            document.link_track_to_identity(id, owner_id);
        }
        let mut other = GroundedDocument::new("other", "");
        let mut imported = Identity::new(0, "imported Ada");
        imported.source = Some(IdentitySource::Hybrid {
            track_refs: refs,
            kb_name: "catalog".into(),
            kb_id: "ada".into(),
        });
        let imported_id = other.add_identity(imported);

        let mut corpus = Corpus::new();
        corpus.add_document(document);
        corpus.add_document(other);
        let merged = corpus.merge_tracks("d", &inputs).unwrap();
        let doc = corpus.get_document("d").unwrap();
        let owner = doc.identity_for_track(merged).unwrap();
        assert_eq!(owner.id, owner_id);
        assert_eq!(
            owner.source.as_ref().unwrap().track_refs(),
            &[doc.track_ref(merged).unwrap()]
        );
        assert_eq!(owner.minimum_pair_similarity, None);
        assert_eq!(
            corpus
                .get_document("other")
                .unwrap()
                .get_identity(imported_id)
                .unwrap()
                .source,
            Some(IdentitySource::KnowledgeBase {
                kb_name: "catalog".into(),
                kb_id: "ada".into()
            })
        );
        assert!(corpus.current_coref_clusters().is_empty());
    }

    #[test]
    fn importing_a_shared_hybrid_refreshes_live_and_persisted_refs() {
        let mut corpus = Corpus::new();
        let first = add_track_doc(&mut corpus, "a", "Ada");
        let id = Resolver::new().resolve_inter_doc_coref(&mut corpus, None, None)[0];
        corpus.link_track_to_kb(&first, "kb", "1", "Ada").unwrap();
        let shared = corpus.get_identity(id).unwrap().clone();
        let mut imported = GroundedDocument::new("b", "Ada");
        assert!(imported.insert_identity_with_id(shared));
        let second_id = imported.add_track(Track::new(0, "Ada").with_identity(id));
        let second = imported.track_ref(second_id).unwrap();
        corpus.add_document(imported);
        let identity = corpus.get_identity(id).unwrap();
        assert_eq!(
            identity.source.as_ref().unwrap().track_refs(),
            &[first, second]
        );
        assert_eq!(
            Some(identity),
            corpus.get_document("b").unwrap().get_identity(id)
        );
    }

    #[test]
    fn nan_embeddings_do_not_break_retries_or_kb_updates_to_shared_copies() {
        let mut corpus = Corpus::new();
        let track_ref = add_track_doc(&mut corpus, "a", "Ada");
        corpus
            .get_document_mut("a")
            .unwrap()
            .get_track_mut(track_ref.track_id)
            .unwrap()
            .embedding = Some(vec![f32::NAN]);
        let resolver = Resolver::new();
        let ids = resolver.resolve_inter_doc_coref(&mut corpus, None, None);
        assert_eq!(ids.len(), 1);
        assert_eq!(
            resolver.resolve_inter_doc_coref(&mut corpus, None, None),
            ids
        );
        assert!(corpus.get_identity(ids[0]).unwrap().embedding.is_none());

        let direct = add_track_doc(&mut corpus, "b", "Grace");
        corpus
            .get_document_mut("b")
            .unwrap()
            .get_track_mut(direct.track_id)
            .unwrap()
            .embedding = Some(vec![f32::NAN]);
        let linked = corpus
            .link_track_to_kb(&direct, "kb", "g", "Grace")
            .unwrap();
        assert_eq!(
            corpus.get_identity(linked).unwrap().confidence,
            crate::Confidence::ZERO
        );
        assert_eq!(
            corpus
                .link_track_to_kb(&direct, "kb", "g", "Grace")
                .unwrap(),
            linked
        );
        assert_eq!(corpus.identities().len(), 2);
        assert_eq!(embedding_similarity(&[f32::MAX], &[f32::MAX]), 1.0);
    }

    #[test]
    fn corpus_identity_preview_accounts_for_mutable_document_additions() {
        let mut corpus = Corpus::new();
        corpus.add_document(GroundedDocument::new("local", "Ada"));
        let local_id = corpus
            .get_document_mut("local")
            .unwrap()
            .add_identity(Identity::new(0, "local Ada"));

        let preview = corpus.next_identity_id();
        let allocated = corpus.add_identity(Identity::new(0, "corpus identity"));
        assert_ne!(preview, local_id);
        assert_eq!(preview, allocated);
    }

    #[test]
    fn adding_document_remaps_a_conflicting_local_identity() {
        let mut corpus = Corpus::new();
        let corpus_identity_id = corpus.add_identity(Identity::new(0, "corpus identity"));
        let (document, track_id, _) = document_with_linked_local_identity("local", "Ada");
        corpus.add_document(document);

        let document = corpus.get_document("local").unwrap();
        let local_identity_id = document.get_track(track_id).unwrap().identity_id.unwrap();
        assert_ne!(local_identity_id, corpus_identity_id);
        assert!(document.get_identity(local_identity_id).is_some());
        assert!(document.invariants_hold());
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// String similarity is always in [0, 1]
        #[test]
        fn string_sim_bounded(a in ".*", b in ".*") {
            let sim = string_similarity(&a, &b);
            prop_assert!((0.0..=1.0).contains(&sim));
        }

        /// String similarity is symmetric
        #[test]
        fn string_sim_symmetric(a in "[a-z ]{0,30}", b in "[a-z ]{0,30}") {
            let sim_ab = string_similarity(&a, &b);
            let sim_ba = string_similarity(&b, &a);
            prop_assert!((sim_ab - sim_ba).abs() < 0.0001);
        }

        /// String similarity is reflexive (identical strings = 1.0)
        #[test]
        fn string_sim_reflexive(s in "[a-z]{1,20}") {
            let sim = string_similarity(&s, &s);
            prop_assert!((sim - 1.0).abs() < 0.0001);
        }

        /// Embedding similarity is bounded [0, 1] for our normalization
        #[test]
        fn embedding_sim_bounded(dim in 1usize..50, seed in any::<u64>()) {
            let mut rng = seed;
            let emb1: Vec<f32> = (0..dim).map(|_| {
                rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                (rng % 2000) as f32 / 1000.0 - 1.0
            }).collect();
            let emb2: Vec<f32> = (0..dim).map(|_| {
                rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                (rng % 2000) as f32 / 1000.0 - 1.0
            }).collect();

            let sim = embedding_similarity(&emb1, &emb2);
            prop_assert!((0.0..=1.0).contains(&sim),
                "Embedding similarity out of bounds: {}", sim);
        }

        /// Embedding similarity is symmetric
        #[test]
        fn embedding_sim_symmetric(dim in 1usize..20, seed in any::<u64>()) {
            let mut rng = seed;
            let emb1: Vec<f32> = (0..dim).map(|_| {
                rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                (rng % 100) as f32 / 100.0
            }).collect();
            let emb2: Vec<f32> = (0..dim).map(|_| {
                rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                (rng % 100) as f32 / 100.0
            }).collect();

            let sim_ab = embedding_similarity(&emb1, &emb2);
            let sim_ba = embedding_similarity(&emb2, &emb1);
            prop_assert!((sim_ab - sim_ba).abs() < 0.0001,
                "Embedding similarity not symmetric: {} vs {}", sim_ab, sim_ba);
        }
    }
}
