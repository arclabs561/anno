// SPDX-License-Identifier: MIT OR Apache-2.0
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

fn decode(out: Output) -> Value {
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).unwrap()
}

fn file(directory: &Path, name: &str, value: &Value) {
    fs::write(directory.join(name), serde_json::to_vec(value).unwrap()).unwrap()
}

#[test]
fn concurrent_delivery_review_and_restart() {
    let temp = tempdir().unwrap();
    let dir = temp.path();
    let mut request = json!({"kind":"delivery","request":{"issuer":"archive","delivery_id":"msg-1",
        "source_revision":1, "ownership":{"mode":"independent"}}});
    file(dir, "request", &request);
    let doc = json!({"id":"program","text":"Nera","signals":[{"id":0,"location":{"Text":{"start":0,"end":4}},
        "surface":"Nera","label":"PER","modality":"Symbolic","confidence":1.0,"negated":false}],
        "tracks":{"0":{"id":0,"signals":[{"signal_id":0,"position":0}],"canonical_surface":"Nera","cluster_confidence":1.0}},
        "identities":{}});
    file(dir, "doc", &doc);
    let apply = ["apply", "request", "--document", "doc"];
    let replies = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..6)
            .map(|_| scope.spawn(|| decode(invoke(dir, &apply))))
            .collect();
        workers
            .into_iter()
            .map(|w| w.join().unwrap())
            .collect::<Vec<_>>()
    });
    assert_eq!(
        replies.iter().filter(|r| r["status"] == "applied").count(),
        1
    );
    assert_eq!(
        replies
            .iter()
            .filter(|r| r["status"] == "previously_applied")
            .count(),
        5
    );
    let receipt = &replies[0]["receipt"];
    assert!(replies.iter().all(|r| &r["receipt"] == receipt));
    let doc_id = receipt["document_id"].as_str().unwrap();
    let read = || decode(invoke(dir, &["read", doc_id]));
    let original = read();
    // The token for an already accepted delivery continues to identify its first receipt.
    let seen = decode(invoke(dir, &["preview", "request", "--document", "doc"]));
    let seen_approval = seen["approval"].as_str().unwrap();
    let track = original["tracks"]
        .as_object()
        .unwrap()
        .values()
        .next()
        .unwrap()["id"]
        .clone();
    file(
        dir,
        "review",
        &json!({"kind":"assignment","trigger":{"doc_id":doc_id,"track_id":track},
        "kb_name":"archive","kb_id":"23","canonical_name":"Nera"}),
    );
    let preview = decode(invoke(dir, &["preview", "review"]));
    let approval = preview["approval"].as_str().unwrap();
    assert_eq!(
        preview["proposal"]["scope"]["affected_tracks"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    assert!(!invoke(dir, &["apply", "review"]).status.success());
    assert_eq!(
        decode(invoke(dir, &["apply", "review", "--approval", approval]))["status"],
        "applied"
    );
    assert_eq!(
        decode(invoke(dir, &["apply", "review", "--approval", approval]))["status"],
        "previously_applied"
    );
    assert_eq!(
        decode(invoke(
            dir,
            &[
                "apply",
                "request",
                "--document",
                "doc",
                "--approval",
                seen_approval
            ]
        ))["receipt"],
        *receipt
    );
    let reviewed = read();
    let identity = reviewed["tracks"]
        .as_object()
        .unwrap()
        .values()
        .next()
        .unwrap()["identity_id"]
        .as_u64()
        .unwrap();
    assert_eq!(reviewed["identities"][identity.to_string()]["kb_id"], "23");
    assert_eq!(decode(invoke(dir, &apply))["receipt"], *receipt);
    assert_eq!(read(), reviewed);
    // A changed payload under the same token is rejected without mutating accepted state.
    let mut altered = doc.clone();
    altered["text"] = json!("Nera!");
    file(dir, "doc", &altered);
    assert!(!invoke(dir, &apply).status.success());
    assert_eq!(read(), reviewed);
    // A legitimate later edition that removes the binding requires review on the current head.
    request["request"]["delivery_id"] = json!("msg-2");
    request["request"]["source_revision"] = json!(2);
    file(dir, "request", &request);
    let next = decode(invoke(dir, &["preview", "request", "--document", "doc"]));
    assert_eq!(next["proposal"]["status"], "review_required");
    assert!(!invoke(dir, &apply).status.success());
    assert!(!invoke(
        dir,
        &[
            "apply",
            "request",
            "--document",
            "doc",
            "--approval",
            approval
        ]
    )
    .status
    .success());
    assert_eq!(read(), reviewed);
    let changed = decode(invoke(
        dir,
        &[
            "apply",
            "request",
            "--document",
            "doc",
            "--approval",
            next["approval"].as_str().unwrap(),
        ],
    ));
    assert_eq!(changed["status"], "applied");
    assert_eq!(read()["text"], "Nera!");
    // Reject both a changed approved scope and a journal truncated to a valid prefix.
    let db = rusqlite::Connection::open(dir.join("local.sqlite")).unwrap();
    let saved: String = db
        .query_row(
            "SELECT proposal FROM corpus_events WHERE position=2",
            [],
            |r| r.get(0),
        )
        .unwrap();
    db.execute(
        "UPDATE corpus_events SET proposal='{}' WHERE position=2",
        [],
    )
    .unwrap();
    assert!(!invoke(dir, &["read", doc_id]).status.success());
    db.execute(
        "UPDATE corpus_events SET proposal=? WHERE position=2",
        [saved],
    )
    .unwrap();
    assert_eq!(read()["text"], "Nera!");
    db.execute("DELETE FROM corpus_events WHERE position=3", [])
        .unwrap();
    assert!(!invoke(dir, &["read", doc_id]).status.success());
}
