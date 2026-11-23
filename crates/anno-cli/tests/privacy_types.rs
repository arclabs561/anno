use std::process::Command;

const INPUT: &str = "Café: case@example.invalid.";

fn redact(types: &str, input: &str) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_anno"));
    cmd.args([
        "privacy", "--model", "pattern", "--action", "redact", "--quiet", "--types", types,
        "--text", input,
    ]);
    cmd.env("ANNO_NO_DOWNLOADS", "1");
    cmd
}

#[test]
fn filter_selects_email_or_id_within_the_same_mailbox() {
    let input = "✉ 123-45-6789@example.org END";
    for (kind, expected) in [
        ("CONTACT", "✉ [CONTACT_1] END"),
        ("ID_NUMBER", "✉ [ID_NUMBER_1]@example.org END"),
    ] {
        let output = redact(kind, input).output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            String::from_utf8(output.stdout).unwrap().trim_end(),
            expected
        );
    }
}

#[test]
fn unsupported_types_fail_before_writing_output() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("redacted.txt");
    for types in ["EMAIL", "CONTACT,unknown"] {
        let result = redact(types, INPUT)
            .arg("--output")
            .arg(&path)
            .output()
            .unwrap();
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
    let result = redact("person,cOnTaCt", INPUT).output().unwrap();
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

#[test]
fn map_cannot_replace_redirected_stdout() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("out.txt");
    std::fs::write(&path, "sentinel").unwrap();
    let out = std::fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap();
    let run = Command::new(env!("CARGO_BIN_EXE_anno"))
        .args([
            "privacy",
            "--model",
            "pattern",
            "--action",
            "pseudonymize",
            "--text",
            INPUT,
        ])
        .arg("--export-map")
        .arg(&path)
        .stdout(out)
        .env("ANNO_NO_DOWNLOADS", "1")
        .output()
        .unwrap();
    assert!(!run.status.success());
    assert_eq!(std::fs::read_to_string(path).unwrap(), "sentinel");
}

#[test]
fn map_cannot_replace_input_or_create_output_on_rejection() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("source.txt");
    let alias = dir.path().join("linked-map.tsv");
    let output = dir.path().join("out.txt");
    std::fs::write(&input, INPUT).unwrap();
    std::fs::hard_link(&input, &alias).unwrap();
    for map in [&input, &alias] {
        let run = Command::new(env!("CARGO_BIN_EXE_anno"))
            .args([
                "privacy",
                "--model",
                "pattern",
                "--action",
                "pseudonymize",
                "--input",
            ])
            .arg(&input)
            .arg("--export-map")
            .arg(map)
            .arg("--output")
            .arg(&output)
            .env("ANNO_NO_DOWNLOADS", "1")
            .output()
            .unwrap();
        assert!(!run.status.success());
        assert!(String::from_utf8_lossy(&run.stderr).contains("--input and --export-map"));
        assert_eq!(std::fs::read_to_string(&input).unwrap(), INPUT);
        assert!(!output.exists());
    }
}

#[cfg(target_os = "linux")]
#[test]
fn explicit_stdout_pipe_receives_pseudonymized_text() {
    let dir = tempfile::tempdir().unwrap();
    let map = dir.path().join("map.tsv");
    let run = pseudonymize(std::path::Path::new("/dev/stdout"), &map, INPUT);
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert_eq!(
        String::from_utf8(run.stdout).unwrap(),
        "Café: contact@example.com."
    );
    assert_eq!(
        std::fs::read_to_string(map).unwrap(),
        "case@example.invalid\tcontact@example.com"
    );
}
