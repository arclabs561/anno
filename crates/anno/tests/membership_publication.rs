// SPDX-License-Identifier: MIT OR Apache-2.0
use anno::coalesce::memberships::{publish_memberships, MembershipDocument, MembershipEdition};
use anno::{Corpus, GroundedDocument, Location, Signal, TrackId, TrackRef};
use std::collections::BTreeMap;

fn document(id: &str, name: &str) -> GroundedDocument {
    let mut doc = GroundedDocument::new(id, name);
    let signal = doc
        .add_signal_validated(Signal::new(
            0,
            Location::text(0, name.chars().count()),
            name,
            "PER",
            1.0,
        ))
        .unwrap();
    doc.create_track_from_signals(name, &[signal]).unwrap();
    doc
}

#[test]
fn live_remapped_handles_publish_the_same_source_addresses_as_bulk() {
    let mut live = Corpus::new();
    let mut revisions = BTreeMap::new();
    let mut latest = BTreeMap::new();
    for (i, (id, name)) in [
        ("a", "Kei"),
        ("b", "Dale"),
        ("x", "Kei Dale"),
        ("x", "Dale"),
        ("x", "Kei Dale"),
        ("x", "Nae"),
    ]
    .into_iter()
    .enumerate()
    {
        let doc = document(id, name);
        latest.insert(id, doc.clone());
        live.add_document(doc);
        revisions.insert(id.to_owned(), i.to_string());
        let current =
            publish_memberships(&mut live, "test", i as u64, 0.5, true, &revisions).unwrap();
        let bulk = MembershipEdition {
            stream: "test".into(),
            sequence: i as u64,
            threshold: 0.5,
            require_type_match: true,
            documents: latest
                .iter()
                .rev()
                .map(|(id, doc)| MembershipDocument {
                    source_revision: revisions.get(*id).cloned(),
                    document: doc.clone(),
                })
                .collect(),
        }
        .snapshot()
        .unwrap();
        assert_eq!(current.checkpoint(), bulk.checkpoint());
        assert!(current.rows().eq(bulk.rows()));
    }
    assert_ne!(
        live.get_document("x").unwrap().tracks().next().unwrap().id,
        TrackId::ZERO
    );
}

#[test]
fn committed_hybrid_keeps_membership_without_fabricating_current_direct_scores() {
    let mut corpus = Corpus::new();
    for id in ["a", "b"] {
        corpus.add_document(document(id, "Kei"));
    }
    let revisions = BTreeMap::new();
    let before = publish_memberships(&mut corpus, "panel", 1, 0.5, true, &revisions).unwrap();
    for (_, row) in before.rows() {
        assert_eq!(row["direct"].as_array().unwrap().len(), 1);
        assert!(row["catalog_record"].is_null());
    }
    corpus
        .link_track_to_kb(
            &TrackRef {
                doc_id: "a".into(),
                track_id: TrackId::ZERO,
            },
            "folio",
            "8",
            "Kei",
        )
        .unwrap();
    let after = publish_memberships(&mut corpus, "panel", 2, 0.5, true, &revisions).unwrap();
    for ((_, old), (_, new)) in before.rows().zip(after.rows()) {
        assert!(!new["component"].is_null());
        assert_eq!(new["component"], old["component"]);
        assert_eq!(new["catalog_record"]["accession"], "8");
        assert!(new["direct"].as_array().unwrap().is_empty());
    }
}

#[test]
fn direct_score_identifies_the_measured_basis() {
    let mut docs = Vec::new();
    for (id, name) in [("a", "Kei"), ("b", "Dale")] {
        let mut doc = document(id, name);
        doc.get_track_mut(TrackId::ZERO).unwrap().embedding = Some(vec![1.0, 0.0]);
        docs.push(MembershipDocument {
            source_revision: None,
            document: doc,
        });
    }
    let current = MembershipEdition {
        stream: "panel".into(),
        sequence: 0,
        threshold: 0.9,
        require_type_match: true,
        documents: docs,
    }
    .snapshot()
    .unwrap();
    for (_, row) in current.rows() {
        assert_eq!(row["direct"][0]["basis"], "scaled_embedding_cosine");
        assert_eq!(row["direct"][0]["similarity"], 1.0);
    }
}

#[test]
fn a_credit_release_pins_access_alongside_unchanged_membership() {
    use anno::publication::{IndexEdition, IndexEditionScope, IndexSnapshot, PublicationError};
    use serde_json::json;

    let mut corpus = Corpus::new();
    corpus.add_document(document("register-page", "Iona Reed"));
    let evidence =
        publish_memberships(&mut corpus, "register", 1, 0.5, true, &BTreeMap::new()).unwrap();
    let (source_key, _) = evidence.rows().next().unwrap();
    let credit = |sequence, access: &str| {
        IndexSnapshot::new(
            "register/credits",
            sequence,
            BTreeMap::from([(
                vec!["credit-1".into()],
                json!({
                    "evidence": source_key, "work": "work-17", "predicate": "attributed_to",
                    "subject": ["people", "P7"], "access": access,
                }),
            )]),
        )
        .unwrap()
    };
    let restricted = credit(1, "restricted");
    let same_sequence_grant = credit(1, "public");
    assert!(matches!(
        same_sequence_grant.changes_since(&restricted),
        Err(PublicationError::ConflictingRevision)
    ));

    let public = credit(2, "public");
    let delta = public.changes_since(&restricted).unwrap();
    let [(_, old)] = delta.removed() else {
        panic!("expected one removal")
    };
    let [(_, new)] = delta.upserted() else {
        panic!("expected one upsert")
    };
    assert_eq!(old["access"], "restricted");
    assert_eq!(new["access"], "public");

    let release = |sequence, claims: &IndexSnapshot| {
        IndexEdition::new(
            IndexEditionScope {
                collection: "works".into(),
                source: "register".into(),
                audience: "staff".into(),
            },
            sequence,
            vec![evidence.checkpoint().clone(), claims.checkpoint().clone()],
        )
        .unwrap()
    };
    let before = release(1, &restricted);
    assert_ne!(before.members(), release(1, &same_sequence_grant).members());
    let after = release(2, &public);
    assert!(before.members().contains(evidence.checkpoint()));
    assert!(after.members().contains(evidence.checkpoint()));
    assert!(before.members().contains(restricted.checkpoint()));
    assert!(after.members().contains(public.checkpoint()));
    assert_ne!(before.members(), after.members());
}
