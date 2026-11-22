use std::process::Command;

const INPUT: &str = "Café: case@example.invalid.";

fn redact(types: &str) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_anno"));
    cmd.args([
        "privacy", "--model", "pattern", "--action", "redact", "--quiet", "--types", types,
        "--text", INPUT,
    ]);
    cmd.env("ANNO_NO_DOWNLOADS", "1");
    cmd
}

#[test]
fn unsupported_types_fail_before_writing_output() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("redacted.txt");
    for types in ["EMAIL", "CONTACT,unknown"] {
        let result = redact(types).arg("--output").arg(&path).output().unwrap();
        assert!(!result.status.success());
        assert!(result.stdout.is_empty());
        let diagnostic = String::from_utf8(result.stderr).unwrap();
        assert!(diagnostic.contains("CONTACT"));
        assert!(!diagnostic.contains(INPUT));
        assert!(!path.exists());
    }
}

#[test]
fn supported_filters_redact_contacts_case_insensitively() {
    let result = redact("person,cOnTaCt").output().unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        String::from_utf8(result.stdout).unwrap().trim_end(),
        "Café: [CONTACT_1]."
    );
}

fn pseudonymize(
    output: &std::path::Path,
    map: &std::path::Path,
    text: &str,
) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_anno"))
        .args([
            "privacy",
            "--model",
            "pattern",
            "--action",
            "pseudonymize",
            "--quiet",
            "--text",
            text,
        ])
        .arg("--output")
        .arg(output)
        .arg("--export-map")
        .arg(map)
        .env("ANNO_NO_DOWNLOADS", "1")
        .output()
        .unwrap()
}

#[test]
fn map_cannot_replace_pseudonymized_output() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("document.txt");
    assert!(!pseudonymize(&path, &path, INPUT).status.success());
    assert!(!path.exists());
    std::fs::write(&path, "keep me").unwrap();
    let alias = dir.path().join("map.tsv");
    std::fs::hard_link(&path, &alias).unwrap();
    assert!(!pseudonymize(&path, &alias, INPUT).status.success());
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "keep me");
}

#[test]
fn mapping_recovers_distinct_contacts_and_repeated_mentions() {
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("out.txt");
    let map = dir.path().join("map.tsv");
    let input =
        "alice@example.invalid bob@example.invalid alice@example.invalid contact@example.com";
    let run = pseudonymize(&output, &map, input);
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    let mapping = std::fs::read_to_string(&map).unwrap();
    let reverse: std::collections::HashMap<_, _> = mapping
        .lines()
        .map(|row| {
            let (real, fake) = row.split_once('\t').unwrap();
            (fake, real)
        })
        .collect();
    assert_eq!(reverse.len(), 3);
    let doc = std::fs::read_to_string(&output).unwrap();
    let recovered: Vec<_> = doc.split_whitespace().map(|s| reverse[s]).collect();
    assert_eq!(recovered.join(" "), input);
}
