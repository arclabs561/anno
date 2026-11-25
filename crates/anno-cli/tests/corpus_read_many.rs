// SPDX-License-Identifier: MIT OR Apache-2.0
use anno::core::grounded::{DeliveryOwnership, GroundedDocument, Identity, Track};
use serde_json::{json, Value};
use std::{
    fs,
    path::Path,
    process::{Command, Output},
};
use tempfile::tempdir;

fn invoke(directory: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_anno"))
        .current_dir(directory)
        .args(["corpus", "--store", "local.sqlite"])
        .args(args)
        .output()
        .unwrap()
}

fn decode(output: Output) -> Value {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

fn assert_rejected(output: Output) {
    assert!(!output.status.success());
    assert!(output.stdout.is_empty(), "unexpected partial response");
}

fn deliver(directory: &Path, id: &str, revision: u64, text: &str) -> String {
    apply_document(
        directory,
        GroundedDocument::new(id, text),
        revision,
        DeliveryOwnership::Independent,
    )
}

fn apply_document(
    directory: &Path,
    document: GroundedDocument,
    revision: u64,
    ownership: DeliveryOwnership,
) -> String {
    let request = json!({"kind":"delivery","request":{
        "issuer":"archive","delivery_id":format!("{}-{revision}", document.id()),
        "source_revision":revision,"ownership":ownership
    }});
    fs::write(
        directory.join("request"),
        serde_json::to_vec(&request).unwrap(),
    )
    .unwrap();
    fs::write(
        directory.join("doc"),
        serde_json::to_vec(&document).unwrap(),
    )
    .unwrap();
    let reply = decode(invoke(
        directory,
        &["apply", "request", "--document", "doc"],
    ));
    assert_eq!(reply["status"], "applied");
    reply["receipt"]["document_id"].as_str().unwrap().to_owned()
}

#[test]
fn pinning_a_review_rejects_an_advanced_head_even_when_the_reviewed_documents_match() {
    let temp = tempdir().unwrap();
    let dir = temp.path();
    let [a, b] = ["a", "b"].map(|id| {
        let mut document = GroundedDocument::new(id, "Reed");
        let owner = document.add_identity(Identity::from_kb(0, "Reed", "library", "17"));
        let track = document.add_track(Track::new(0, "Reed"));
        document.link_track_to_identity(track, owner);
        apply_document(
            dir,
            document,
            1,
            DeliveryOwnership::Restore {
                origin: "master".into(),
            },
        )
    });
    let observed = decode(invoke(dir, &["read", &a]));
    let track = observed["tracks"]
        .as_object()
        .unwrap()
        .values()
        .next()
        .unwrap();
    let review = json!({"kind":"assignment","trigger":{"doc_id":a,"track_id":track["id"]},
        "kb_name":"library","kb_id":"23","canonical_name":"Reviewed"});
    fs::write(dir.join("review"), serde_json::to_vec(&review).unwrap()).unwrap();
    let preview = || decode(invoke(dir, &["preview", "review"]));
    let original = preview();
    let scope = &original["proposal"]["scope"];
    let documents: Vec<_> = scope["affected_tracks"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["doc_id"].as_str().unwrap())
        .collect();
    assert_eq!(documents.len(), 2);
    assert!(documents.contains(&a.as_str()) && documents.contains(&b.as_str()));
    let pinned = |head: &str| {
        let mut args = vec!["read-many", "--at-head", head];
        args.extend(documents.iter().copied());
        invoke(dir, &args)
    };
    let head = original["head"].as_str().unwrap();
    let inspected = decode(pinned(head));
    let mut current = vec!["read-many"];
    current.extend(documents.iter().copied());
    assert_eq!(inspected, decode(invoke(dir, &current)));
    assert_eq!(inspected["head"], head);

    deliver(dir, "unrelated", 1, "Another topic");
    let stale = pinned(head);
    assert!(String::from_utf8_lossy(&stale.stderr)
        .to_lowercase()
        .contains("corpus head changed"));
    assert_rejected(stale);
    let refreshed = preview();
    let new_head = refreshed["head"].as_str().unwrap();
    assert_ne!(new_head, head);
    assert_eq!(refreshed["proposal"]["scope"], *scope);
    assert_eq!(
        decode(pinned(new_head)),
        json!({"head":new_head,"documents":inspected["documents"]})
    );
}

#[test]
fn reads_documents_in_request_order_at_the_committed_head() {
    let temp = tempdir().unwrap();
    let dir = temp.path();
    let a = deliver(dir, "a", 1, "First edition");
    let b = deliver(dir, "b", 1, "Another document");
    let read = |id| decode(invoke(dir, &["read", id]));
    let batch = || decode(invoke(dir, &["read-many", &b, &a, &b]));
    let head = || decode(invoke(dir, &["preview", "request", "--document", "doc"]))["head"].clone();

    let before = batch();
    assert_eq!(
        before,
        json!({"head":head(),"documents":[read(&b),read(&a),read(&b)]})
    );
    assert_eq!(before["documents"][1]["text"], "First edition");
    assert_eq!(deliver(dir, "a", 2, "Corrected edition"), a);
    let after = batch();
    assert_ne!(before["head"], after["head"]);
    assert_eq!(
        after,
        json!({"head":head(),"documents":[read(&b),read(&a),read(&b)]})
    );
    assert_eq!(after["documents"][1]["text"], "Corrected edition");
    assert_rejected(invoke(dir, &["read-many", &b, "missing"]));
}

#[test]
fn preserves_the_single_read_json_numbers_for_binary32_embeddings() {
    let temp = tempdir().unwrap();
    let dir = temp.path();
    let mut document = GroundedDocument::new("vectors", "Reed");
    document.add_track(Track::new(0, "Reed").with_embedding(vec![0.7]));
    let id = apply_document(dir, document, 1, DeliveryOwnership::Independent);
    let single = decode(invoke(dir, &["read", &id]));
    let batch = decode(invoke(dir, &["read-many", &id]));
    let coordinate = single["tracks"]["0"]["embedding"][0].as_f64().unwrap();
    assert_eq!(coordinate, f64::from(0.7_f32));
    assert_eq!(batch["documents"][0], single);
}

#[test]
fn validates_the_receipt_of_an_unrequested_earlier_delivery() {
    let temp = tempdir().unwrap();
    let dir = temp.path();
    deliver(dir, "earlier", 1, "First document");
    let id = deliver(dir, "requested", 1, "Requested document");
    let batch = || invoke(dir, &["read-many", &id]);
    let pristine = decode(batch());
    let db = rusqlite::Connection::open(dir.join("local.sqlite")).unwrap();
    let receipt: String = db
        .query_row(
            "SELECT receipt FROM corpus_events WHERE position=1",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(
        db.execute(
            "UPDATE corpus_events SET receipt=(SELECT receipt FROM corpus_events WHERE position=2) WHERE position=1",
            [],
        )
        .unwrap(),
        1
    );
    assert_rejected(batch());
    db.execute(
        "UPDATE corpus_events SET receipt=? WHERE position=1",
        [receipt],
    )
    .unwrap();
    assert_eq!(decode(batch()), pristine);
}

#[test]
fn enforces_input_bounds_counting_repeated_ids_and_never_creates_a_missing_store() {
    let temp = tempdir().unwrap();
    let dir = temp.path();
    assert_rejected(invoke(dir, &["read-many", "missing"]));
    assert!(!dir.join("local.sqlite").exists());
    let id = deliver(dir, "a", 1, "Repeated document");
    assert_rejected(invoke(dir, &["read-many"]));
    let mut args = vec!["read-many"];
    args.extend([id.as_str(); 128]);
    let expected = decode(invoke(dir, &["read", &id]));
    assert_eq!(
        decode(invoke(dir, &args))["documents"],
        json!(vec![expected; 128])
    );
    args.push(&id);
    assert_rejected(invoke(dir, &args));
}
