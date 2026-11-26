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

#[test]
fn explanations_distinguish_decisions_behind_identical_publications() {
    use anno::coalesce::resolver::{PairDecision, PairMeasure};

    let edition = |right_type: &str, right_embedding: Vec<f32>, threshold| {
        let mut documents = Vec::new();
        for (id, label, vector) in [
            ("left", "PER", vec![1.0, 0.0]),
            ("right", right_type, right_embedding),
        ] {
            let mut doc = document(id, "Kei");
            let track = doc.get_track_mut(TrackId::ZERO).unwrap();
            track.entity_type = Some(label.into());
            track.embedding = Some(vector);
            documents.push(MembershipDocument {
                source_revision: Some("rev-1".into()),
                document: doc,
            });
        }
        MembershipEdition {
            stream: "test".into(),
            sequence: 1,
            threshold,
            require_type_match: true,
            documents,
        }
    };
    let track = |doc: &str| TrackRef {
        doc_id: doc.into(),
        track_id: TrackId::ZERO,
    };
    let (left, right) = (track("left"), track("right"));

    let guarded = edition("ORG", vec![1.0, 0.0], 0.75)
        .explain_pair(&left, &right)
        .unwrap();
    let missed = edition("PER", vec![0.0, 1.0], 0.75)
        .explain_pair(&left, &right)
        .unwrap();
    assert_eq!(guarded.checkpoint, missed.checkpoint);
    assert_eq!(
        guarded.checkpoint,
        *edition("ORG", vec![1.0, 0.0], 0.75)
            .snapshot()
            .unwrap()
            .checkpoint()
    );
    assert_eq!(guarded.decision, PairDecision::TypeMismatch);
    use anno::coalesce::resolver::CorefTrackType;
    assert_eq!(
        guarded.left.entity_type,
        CorefTrackType::Known("PER".into())
    );
    assert_eq!(
        guarded.right.entity_type,
        CorefTrackType::Known("ORG".into())
    );
    assert_eq!(
        missed.decision,
        PairDecision::BelowThreshold {
            similarity: 0.5,
            basis: PairMeasure::ScaledEmbeddingCosine,
        }
    );
    assert!(!guarded.same_component && !missed.same_component);

    let accepted = edition("PER", vec![0.0, 1.0], 0.5)
        .explain_pair(&right, &left)
        .unwrap();
    assert!(accepted.same_component);
    assert_eq!(
        accepted.decision,
        PairDecision::Accepted {
            similarity: 0.5,
            basis: PairMeasure::ScaledEmbeddingCosine,
        }
    );
    assert_eq!(accepted.left.key.first().unwrap(), "right");
    assert_eq!(accepted.right.key.first().unwrap(), "left");
    assert_eq!(accepted.left.key[2..], ["rev-1", "0"]);
}

#[test]
fn explanations_separate_missing_evidence_catalog_bindings_and_transitive_membership() {
    use anno::coalesce::resolver::{PairDecision, PairMeasure};

    let edition = || {
        let mut catalog = document("catalog", "Kei");
        let identity = catalog.add_identity(anno::Identity::from_kb(0, "Kei", "folio", "8"));
        catalog.link_track_to_identity(TrackId::ZERO, identity);
        let mut blank = document("blank", "Kei");
        blank
            .get_track_mut(TrackId::ZERO)
            .unwrap()
            .canonical_surface
            .clear();
        MembershipEdition {
            stream: "test".into(),
            sequence: 1,
            threshold: 0.5,
            require_type_match: true,
            documents: [
                document("left", "Kei"),
                document("bridge", "Kei Dale"),
                document("right", "Dale"),
                catalog,
                blank,
            ]
            .into_iter()
            .map(|document| MembershipDocument {
                source_revision: None,
                document,
            })
            .collect(),
        }
    };
    let track = |doc: &str| TrackRef {
        doc_id: doc.into(),
        track_id: TrackId::ZERO,
    };
    let explain = |l, r| edition().explain_pair(&track(l), &track(r)).unwrap();

    let inferred = explain("left", "right");
    assert!(inferred.same_component);
    assert_eq!(
        inferred.decision,
        PairDecision::BelowThreshold {
            similarity: 0.0,
            basis: PairMeasure::WordJaccard
        }
    );
    let blank = explain("blank", "left");
    assert_eq!(blank.decision, PairDecision::NoComparableEvidence);
    assert!(!blank.same_component);
    let anchored = explain("left", "catalog");
    assert_eq!(
        anchored.decision,
        PairDecision::CatalogBound {
            left: false,
            right: true
        }
    );
    assert!(!anchored.same_component);
    assert!(edition()
        .explain_pair(&track("left"), &track("absent"))
        .is_err());
    assert!(edition()
        .explain_pair(&track("left"), &track("left"))
        .is_err());
}
