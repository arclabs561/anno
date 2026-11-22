use std::process::Command;

fn anno() -> Command {
    Command::new(env!("CARGO_BIN_EXE_anno"))
}

fn email_entity(entities: &[serde_json::Value]) -> &serde_json::Value {
    entities
        .iter()
        .find(|entity| entity["text"] == "alice@example.com")
        .expect("pattern extraction should emit the email")
}

#[test]
fn json_and_jsonl_preserve_emitted_entity_provenance_with_unicode_offsets() {
    let text = "🧪\r\nalice@example.com";
    let expected_start = text.chars().position(|character| character == 'a').unwrap();
    let expected_end = expected_start + "alice@example.com".chars().count();

    let json_output = anno()
        .args([
            "extract", "--model", "pattern", "--format", "json", "--text", text,
        ])
        .env("ANNO_NO_DOWNLOADS", "1")
        .output()
        .unwrap();
    assert!(
        json_output.status.success(),
        "JSON extraction failed: {}",
        String::from_utf8_lossy(&json_output.stderr)
    );

    let json: serde_json::Value = serde_json::from_slice(&json_output.stdout).unwrap();
    assert_eq!(json["provenance"]["model"], "pattern");
    assert_eq!(
        json["provenance"]["entity_sources"],
        serde_json::json!(["pattern"])
    );
    let entities = json["entities"].as_array().unwrap();
    let entity = email_entity(entities);
    assert_eq!(entity["start"], expected_start);
    assert_eq!(entity["end"], expected_end);
    let extracted: String = text
        .chars()
        .skip(expected_start)
        .take(expected_end - expected_start)
        .collect();
    assert_eq!(entity["text"], extracted);
    assert_eq!(entity["provenance"]["source"], "pattern");
    assert_eq!(entity["provenance"]["method"], "Pattern");
    assert_eq!(entity["provenance"]["pattern"], "EMAIL");

    let jsonl_output = anno()
        .args([
            "extract", "--model", "pattern", "--format", "jsonl", "--text", text,
        ])
        .env("ANNO_NO_DOWNLOADS", "1")
        .output()
        .unwrap();
    assert!(
        jsonl_output.status.success(),
        "JSONL extraction failed: {}",
        String::from_utf8_lossy(&jsonl_output.stderr)
    );

    let lines: Vec<serde_json::Value> = String::from_utf8(jsonl_output.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(lines[0]["provenance"]["model"], "pattern");
    let text_hash = format!("xxh3:{:016x}", xxhash_rust::xxh3::xxh3_64(text.as_bytes()));
    assert_eq!(json["provenance"]["text_hash"], text_hash);
    assert_eq!(lines[0]["provenance"]["text_hash"], text_hash);
    assert!(json["provenance"].get("source_url").is_none());
    assert_eq!(
        lines[0]["provenance"]["entity_sources"],
        serde_json::json!(["pattern"])
    );
    let entity = email_entity(&lines[1..]);
    assert_eq!(entity["start"], expected_start);
    assert_eq!(entity["end"], expected_end);
    assert_eq!(entity["provenance"]["source"], "pattern");
}

#[test]
fn provenance_does_not_invent_entity_sources_when_nothing_is_emitted() {
    let output = anno()
        .args([
            "extract",
            "--model",
            "pattern",
            "--format",
            "json",
            "--text",
            "plain words only",
        ])
        .env("ANNO_NO_DOWNLOADS", "1")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "empty extraction failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let json: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(json["provenance"]["model"], "pattern");
    assert_eq!(json["entities"], serde_json::json!([]));
    assert!(json["provenance"].get("entity_sources").is_none());
}

#[cfg(all(
    feature = "extractor-html2text",
    not(feature = "extractor-readability")
))]
#[test]
fn full_page_extraction_keeps_a_table_contact_intact() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("directory.html");
    let export = dir.path().join("document.json");
    let email = "management-team@example.invalid";
    std::fs::write(
        &input,
        format!(
            "<html><body><table>
        <tr><th>Department and contact</th><th>Primary contact for account inquiries</th>
        <th>Discretionary account control status</th><th>Current escalation instructions</th></tr>
        <tr><td>Café supervision</td><td>{email}</td><td>Current review scheduled weekly</td>
        <td>Contact the department supervisor</td></tr></table></body></html>"
        ),
    )
    .unwrap();
    let output = anno()
        .args([
            "extract", "--model", "pattern", "--format", "json", "--file",
        ])
        .arg(input)
        .args(["--export-format", "minimal", "--export"])
        .arg(&export)
        .env("ANNO_NO_DOWNLOADS", "1")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let doc: serde_json::Value = serde_json::from_slice(&std::fs::read(export).unwrap()).unwrap();
    let text = doc["text"].as_str().unwrap();
    let signal = doc["signals"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["surface"] == email)
        .unwrap();
    assert_eq!(signal["label"], "EMAIL");
    let prefix = text.split_once(email).unwrap().0;
    assert_eq!(signal["start"], prefix.chars().count());
    assert_eq!(signal["end"], prefix.chars().count() + email.len());
}
