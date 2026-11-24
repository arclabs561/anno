// SPDX-License-Identifier: MIT OR Apache-2.0
//! Candidate blocking and deletion repair for nonempty pre-normalized ASCII names,
//! one type, no embeddings, threshold .5. Uses anno's real string scorer and compares
//! each live partition to a full native resolver rebuild. Counts threshold score calls.
use anno::coalesce::{resolver::string_similarity, Resolver};
use anno::{Corpus, GroundedDocument, Location, Signal, SignalId};
use std::collections::{BTreeMap, BTreeSet};

type Groups = Vec<Vec<usize>>;
type Postings = BTreeMap<String, Vec<usize>>;

fn absorb(groups: &mut Groups, incoming: usize, matches: &BTreeSet<usize>) {
    let mut merged = vec![incoming];
    groups.retain(|group| {
        if group.iter().any(|member| matches.contains(member)) {
            merged.extend(group);
            false
        } else {
            true
        }
    });
    merged.sort_unstable();
    groups.push(merged);
    groups.sort();
}

fn insert(
    groups: &mut Groups,
    postings: &mut Postings,
    names: &[String],
    incoming: usize,
) -> usize {
    let tokens: BTreeSet<_> = names[incoming].split_ascii_whitespace().collect();
    assert!(!tokens.is_empty());
    assert!(tokens.iter().all(|token| token
        .bytes()
        .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit())));
    let mut candidates = BTreeSet::new();
    for token in &tokens {
        if let Some(prior) = postings.get(*token) {
            candidates.extend(prior.iter().copied());
        }
    }
    let mut calls = 0;
    let mut matches = BTreeSet::new();
    for candidate in candidates {
        calls += 1;
        if string_similarity(&names[incoming], &names[candidate]) >= 0.5 {
            matches.insert(candidate);
        }
    }
    absorb(groups, incoming, &matches);
    for token in tokens {
        postings.entry(token.to_owned()).or_default().push(incoming);
    }
    calls
}

fn blocked(names: &[String], active: &[usize]) -> (Groups, Postings, usize) {
    let mut groups = Groups::new();
    let mut postings = Postings::new();
    let calls = active
        .iter()
        .map(|&i| insert(&mut groups, &mut postings, names, i))
        .sum();
    (groups, postings, calls)
}

fn native(names: &[String], active: &[usize]) -> (Groups, f64) {
    let mut corpus = Corpus::new();
    for &i in active {
        let surface = &names[i];
        let mut doc = GroundedDocument::new(format!("d{i:03}"), surface);
        let signal = Signal::new(0, Location::text(0, surface.len()), surface, "PER", 1.0);
        let signal_id = doc.add_signal_validated(signal).unwrap();
        doc.create_track_from_signals(surface, &[signal_id])
            .unwrap();
        corpus.add_document(doc);
    }
    Resolver::new()
        .with_threshold(0.5)
        .resolve_inter_doc_coref(&mut corpus, None, None);
    let mut groups: Groups = corpus
        .current_coref_clusters()
        .into_iter()
        .map(|(_, members)| {
            let mut ids: Vec<usize> = members
                .iter()
                .map(|m| m.doc_id[1..].parse().unwrap())
                .collect();
            ids.sort_unstable();
            ids
        })
        .collect();
    groups.sort();
    let weakest = corpus
        .get_document("d000")
        .unwrap()
        .identity_for_signal(SignalId::ZERO)
        .unwrap()
        .minimum_pair_similarity
        .unwrap()
        .value();
    (groups, weakest)
}

fn main() {
    let mut names: Vec<_> = (0..128)
        .map(|i| match i {
            0..=3 => "left00".to_owned(),
            4..=7 => "right00".to_owned(),
            _ => format!("cluster{:02}", i / 8),
        })
        .collect();
    let original: Vec<_> = (0..names.len()).collect();
    let (mut groups, mut postings, initial_calls) = blocked(&names, &original);
    assert_eq!(initial_calls, 432);
    assert_eq!(groups.len(), 17);
    let (bulk, initial_support) = native(&names, &original);
    assert_eq!(groups, bulk);
    assert_eq!(initial_support, 1.0);

    let bridge = names.len();
    names.push("left00 right00".into());
    let insert_calls = insert(&mut groups, &mut postings, &names, bridge);
    assert_eq!(insert_calls, 8);
    assert_eq!(groups.len(), 16);
    let (bulk, bridged_support) = native(&names, &(0..names.len()).collect::<Vec<_>>());
    assert_eq!(groups, bulk);
    // Unscored left/right pairs have Jaccard zero within the merged component.
    assert_eq!(bridged_support, 0.0);

    // Remove the bridge from token lookup and find its last published component.
    for token in names[bridge].split_ascii_whitespace() {
        postings.get_mut(token).unwrap().retain(|&id| id != bridge);
    }
    let old = groups.remove(groups.iter().position(|g| g.contains(&bridge)).unwrap());
    let survivors: Vec<_> = old.iter().copied().filter(|&i| i != bridge).collect();
    assert_eq!(survivors.len(), 8);
    let (truth, repaired_support) = native(&names, &original);
    assert_eq!(repaired_support, 1.0);
    let mut forget_only = groups.clone();
    forget_only.push(survivors.clone());
    forget_only.sort();
    assert_ne!(forget_only, truth); // A deletion-only union-find keeps the two sides fused.

    let (split, _, repair_calls) = blocked(&names, &survivors);
    assert_eq!(repair_calls, 12);
    assert_eq!(split.len(), 2);
    groups.extend(split);
    groups.sort();
    assert_eq!(groups, truth);
    println!("stage,active,clusters,actual_blocked_score_calls,full_threshold_pairs_formula,observed_native_group_zero_min");
    println!("bulk,128,17,{initial_calls},8128,{initial_support}");
    println!("insert bridge,129,16,{insert_calls},8256,{bridged_support}");
    println!("delete bridge,128,17,{repair_calls},8128,{repaired_support}");
}
