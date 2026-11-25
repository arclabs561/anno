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
        .args(["--window-words", "7", "--overlap-words", "3"])
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
    let distant = format!("Éva. {}rain. Acme.", "rain ".repeat(16));
    writeln!(input, "{}", json!({"doc_id":"far", "text":distant})).unwrap();
    writeln!(input, "{}", json!({"doc_id":"invalid", "text":123})).unwrap();
    drop(input);

    let out = child.wait_with_output().unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("document 3"));
    let graphs: Vec<Value> = serde_json::Deserializer::from_slice(&out.stdout)
        .into_iter()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(graphs.len(), 2);
    let far = &graphs[1];
    assert_eq!(far["doc_id"], "far");
    assert_eq!(far["feasible"], true);
    let far_entities = far["entities"].as_array().unwrap();
    let first = far_entities.iter().find(|e| e["text"] == "Éva").unwrap();
    let last = far_entities
        .iter()
        .find(|e| e["type"] == "organization" && e["text"] == "Acme")
        .unwrap();
    assert_eq!(
        (last["start"].as_u64(), last["end"].as_u64()),
        (Some(91), Some(95))
    );
    let windows = far["scored_windows"].as_array().unwrap();
    assert!(windows.len() > 1);
    assert!(!windows.iter().any(|w| w["start"].as_u64().unwrap()
        <= first["start"].as_u64().unwrap()
        && w["end"].as_u64().unwrap() >= last["end"].as_u64().unwrap()));
    assert!(far["relations"].as_array().unwrap().is_empty());
    let graph = &graphs[0];
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
