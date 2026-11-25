// SPDX-License-Identifier: MIT OR Apache-2.0
use anno::coalesce::memberships::{MembershipDocument, MembershipEdition};
use anno::coalesce::{lookup_predecessors, PredecessorOptions, PredecessorScorer};
use anno::{GroundedDocument, Identity, Track, TrackId, TrackRef};

fn mask_track(mask: u8) -> (String, String) {
    let surface = ["Reed", "Vale", "North"]
        .into_iter()
        .enumerate()
        .filter(|(bit, _)| mask & (1 << bit) != 0)
        .map(|(_, word)| word)
        .collect::<Vec<_>>()
        .join(" ");
    (format!("doc-{mask}"), surface)
}

fn edition(sequence: u64, tracks: impl IntoIterator<Item = (String, String)>) -> MembershipEdition {
    let documents = tracks
        .into_iter()
        .map(|(id, surface)| {
            let mut document = GroundedDocument::new(id, &surface);
            // Identical vectors and catalog links, but conflicting track types across editions.
            // A scorer that honors any of these signals will disagree with the word-set oracle.
            let owner = document.add_identity(Identity::from_kb(0, "Same", "catalog", "same"));
            let kind = if sequence == 1 { "ORG" } else { "PER" };
            let track = Track::new(0, &surface)
                .with_type(kind)
                .with_identity(owner)
                .with_embedding(vec![1.0]);
            document.add_track(track);
            MembershipDocument {
                document,
                source_revision: None,
            }
        })
        .collect();
    MembershipEdition {
        stream: "names".into(),
        sequence,
        threshold: 1.0,
        require_type_match: true,
        documents,
    }
}

#[test]
fn shortlist_matches_finite_word_set_oracle_at_tie_and_binary32_boundaries() {
    let third = 1.0_f32 / 3.0;
    let cutoffs = [
        0.0,
        f32::from_bits(third.to_bits() - 1),
        third,
        f32::from_bits(third.to_bits() + 1),
        0.5,
        1.0,
    ];
    let priors = [
        edition(1, (0..8).map(mask_track)),
        edition(1, (0..8).rev().map(mask_track)),
    ];
    for query_mask in 0_u8..8 {
        let current = edition(2, [mask_track(query_mask)]);
        let query = TrackRef {
            doc_id: format!("doc-{query_mask}"),
            track_id: TrackId::ZERO,
        };
        for minimum_score in cutoffs {
            let eligible: Vec<_> = (0_u8..8)
                .filter_map(|mask| {
                    let overlap = (query_mask & mask).count_ones();
                    if overlap == 0 {
                        return None;
                    }
                    let union = (query_mask | mask).count_ones();
                    let score = overlap as f32 / union as f32;
                    (score >= minimum_score).then_some((mask, score))
                })
                .collect();
            for limit in 1..=8 {
                // Each candidate has an interval of possible ranks within its tie.
                // It fits precisely when its worst possible rank fits.
                let mut expected = Vec::new();
                let mut boundary_tie = None;
                for &(mask, score) in &eligible {
                    let better = eligible.iter().filter(|(_, s)| *s > score).count();
                    let tied = eligible.iter().filter(|(_, s)| *s == score).count();
                    if better + tied <= limit {
                        expected.push((format!("doc-{mask}"), score));
                    } else if better < limit {
                        boundary_tie = Some((score, tied));
                    }
                }
                expected.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                for prior in &priors {
                    let result = lookup_predecessors(
                        &current,
                        prior,
                        &query,
                        PredecessorOptions {
                            limit,
                            minimum_score,
                        },
                    )
                    .unwrap();
                    let actual: Vec<_> = result
                        .candidates
                        .iter()
                        .map(|c| (c.track.doc_id.clone(), c.score))
                        .collect();
                    assert_eq!(
                        actual, expected,
                        "query mask {query_mask}, cutoff {minimum_score}, limit {limit}"
                    );
                    assert_eq!(result.truncated, eligible.len() > expected.len());
                    assert_eq!(
                        result.withheld_tie.map(|tie| (tie.score, tie.count)),
                        boundary_tie
                    );
                }
            }
        }
    }
}

#[test]
fn word_jaccard_v1_preserves_the_advertised_normalization() {
    let current = edition(2, [("query".into(), "ÉVA'S\tÉva\u{a0}Ruiz".into())]);
    let prior = edition(
        1,
        [
            ("exact".into(), "ruIZ\u{a0}RUIZ\téva".into()),
            ("decomposed".into(), "E\u{301}va Ruiz".into()),
            ("unaccented".into(), "Eva".into()),
        ],
    );
    let query = TrackRef {
        doc_id: "query".into(),
        track_id: TrackId::ZERO,
    };
    let result = lookup_predecessors(
        &current,
        &prior,
        &query,
        PredecessorOptions {
            limit: 3,
            minimum_score: 0.0,
        },
    )
    .unwrap();
    assert_eq!(result.scorer, PredecessorScorer::WordJaccardV1);
    let scores: Vec<_> = result
        .candidates
        .iter()
        .map(|candidate| (candidate.track.doc_id.as_str(), candidate.score))
        .collect();
    // The two-word sets coincide in the first comparison. Only “ruiz” is
    // shared in the second; the third has no shared words.
    assert_eq!(scores, [("exact", 1.0), ("decomposed", 1.0 / 3.0)]);
}
