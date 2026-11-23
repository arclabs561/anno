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
