use serde_json::{json, Value};
use std::io::Write;
use std::process::{Command, Stdio};

#[test]
#[ignore = "requires GLINER25_JOINT_MODEL_DIR and GLINER25_PYTHON with gliner2[local]==2.2.0"]
fn streamed_joint_graph_preserves_unicode_and_fails_the_invalid_row() {
    let mut schema = tempfile::NamedTempFile::new().unwrap();
    write!(schema, "{}", json!({
        "entities": ["person","organization"],
        "relations": {"works_for": {"head": "person", "tail": "organization", "unique_head":true}}
    })).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_anno"))
        .args(["joint", "--checkpoint"])
        .arg(std::env::var_os("GLINER25_JOINT_MODEL_DIR").unwrap())
        .arg("--python")
        .arg(std::env::var_os("GLINER25_PYTHON").unwrap())
        .arg("--schema")
        .arg(schema.path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut input = child.stdin.take().unwrap();
    writeln!(
        input,
        "{}",
        json!({"doc_id":"one", "text":"Éva works for Acme."})
    )
    .unwrap();
    writeln!(input, "{}", json!({"doc_id":"two", "text":123})).unwrap();
    drop(input);

    let out = child.wait_with_output().unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("document 2"));
    let graph: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(graph["doc_id"], "one");
    assert_eq!(graph["feasible"], true);
    let entities = graph["entities"].as_array().unwrap();
    let person = entities.iter().find(|e| e["text"] == "Éva").unwrap();
    let org = entities.iter().find(|e| e["text"] == "Acme").unwrap();
    assert_eq!(
        (person["start"].as_u64(), person["end"].as_u64()),
        (Some(0), Some(3))
    );
    assert_eq!(
        (org["start"].as_u64(), org["end"].as_u64()),
        (Some(14), Some(18))
    );
    assert!(graph["relations"]
        .as_array()
        .unwrap()
        .iter()
        .any(|e| e["type"] == "works_for" && e["head"] == person["id"] && e["tail"] == org["id"]));
}
