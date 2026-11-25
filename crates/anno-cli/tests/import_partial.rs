use std::fs;
use std::path::Path;
use std::process::{Command, Output};

fn import(input: &Path, output: &Path, partial: bool) -> Output {
    import_as("jsonl", input, output, partial)
}

fn import_as(format: &str, input: &Path, output: &Path, partial: bool) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_anno"));
    cmd.args([
        "dataset", "import", "--format", format, "--quiet", "--input",
    ])
    .arg(input)
    .arg("--output")
    .arg(output);
    if partial {
        cmd.arg("--allow-partial");
    }
    cmd.output().unwrap()
}

#[test]
fn partial_directory_results_require_opt_in_and_replace_the_destination() {
    let root = tempfile::tempdir().unwrap();
    let input = root.path().join("input");
    fs::create_dir(&input).unwrap();
    let dest = root.path().join("imported.jsonl");
    let original = root.path().join("old-destination");
    fs::write(&dest, b"previous dataset\n").unwrap();
    fs::hard_link(&dest, &original).unwrap();
    fs::write(input.join("good.jsonl"), r#"{"text":"Zoë"}"#).unwrap();
    fs::write(input.join("bad.jsonl"), "{").unwrap();

    let strict = import(&input, &dest, false);
    assert!(!strict.status.success());
    assert!(String::from_utf8_lossy(&strict.stderr).contains("bad.jsonl"));
    assert_eq!(fs::read(&dest).unwrap(), b"previous dataset\n");

    let partial = import(&input, &dest, true);
    assert!(
        partial.status.success(),
        "{}",
        String::from_utf8_lossy(&partial.stderr)
    );
    assert!(String::from_utf8_lossy(&partial.stderr).contains("bad.jsonl"));
    let row: serde_json::Value = serde_json::from_slice(&fs::read(&dest).unwrap()).unwrap();
    assert_eq!(row["text"], "Zoë");
    assert_eq!(fs::read(&original).unwrap(), b"previous dataset\n");

    fs::remove_file(input.join("good.jsonl")).unwrap();
    let previous = fs::read(&dest).unwrap();
    assert!(!import(&input, &dest, true).status.success());
    assert_eq!(fs::read(&dest).unwrap(), previous);
}

#[cfg(unix)]
#[test]
fn unreadable_directory_inventory_keeps_the_previous_dataset() {
    let root = tempfile::tempdir().unwrap();
    let input = root.path().join("input");
    fs::create_dir(&input).unwrap();
    fs::write(input.join("good.jsonl"), r#"{"text":"Zoë"}"#).unwrap();
    std::os::unix::fs::symlink("missing", input.join("dangling.jsonl")).unwrap();
    let dest = root.path().join("previous.jsonl");
    fs::write(&dest, b"previous").unwrap();
    for partial in [false, true] {
        let result = import(&input, &dest, partial);
        assert!(!result.status.success());
        assert!(String::from_utf8_lossy(&result.stderr).contains("dangling.jsonl"));
        assert_eq!(fs::read(&dest).unwrap(), b"previous");
    }
}

#[test]
fn failed_output_commit_cleans_up_staging() {
    let root = tempfile::tempdir().unwrap();
    let input = root.path().join("good.jsonl");
    fs::write(&input, r#"{"text":"Alice"}"#).unwrap();
    let dest = root.path().join("occupied");
    fs::create_dir(&dest).unwrap();
    fs::write(dest.join("existing"), b"previous").unwrap();
    let result = import(&input, &dest, false);
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("occupied"));
    assert_eq!(fs::read(dest.join("existing")).unwrap(), b"previous");
    assert_eq!(fs::read_dir(root.path()).unwrap().count(), 2);
}

#[test]
fn source_annotations_and_companions_cannot_be_replaced() {
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("corpus");
    fs::create_dir(&source).unwrap();
    let annotation = source.join("doc.ann");
    let companion = source.join("doc.txt");
    fs::write(&annotation, b"T1\tPER 0 3\tAda\n").unwrap();
    for input in [&annotation, &source] {
        assert!(!import_as("brat", input, &companion, false).status.success());
        assert!(!companion.exists());
    }
    fs::write(&companion, b"Ada").unwrap();
    for input in [&annotation, &source] {
        for protected in [&annotation, &companion] {
            let hard_link = root.path().join("alias.jsonl");
            fs::hard_link(protected, &hard_link).unwrap();
            let aliases = vec![protected.clone(), hard_link.clone()];
            #[cfg(unix)]
            let aliases = {
                let symbolic = root.path().join("symbolic.jsonl");
                std::os::unix::fs::symlink(protected, &symbolic).unwrap();
                [aliases, vec![symbolic]].concat()
            };
            for alias in aliases {
                let result = import_as("brat", input, &alias, true);
                assert!(!result.status.success(), "{input:?} -> {alias:?}");
                assert!(String::from_utf8_lossy(&result.stderr).contains("aliases"));
                assert!(same_file::is_same_file(protected, &alias).unwrap());
                if &alias != protected {
                    fs::remove_file(alias).unwrap();
                }
            }
            assert_eq!(fs::read(&annotation).unwrap(), b"T1\tPER 0 3\tAda\n");
            assert_eq!(fs::read(&companion).unwrap(), b"Ada");
        }
    }

    let output = source.join("result.jsonl");
    for _ in 0..2 {
        assert!(import_as("brat", &source, &output, false).status.success());
    }
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&fs::read(output).unwrap()).unwrap()["text"],
        "Ada"
    );
}

#[test]
fn malformed_brat_mention_reports_its_line_and_preserves_destination() {
    let root = tempfile::tempdir().unwrap();
    let annotation = root.path().join("people.ann");
    let dest = root.path().join("reviewed.jsonl");
    fs::write(&dest, b"reviewed corpus\n").unwrap();
    for malformed in [
        "T2\tPER",
        "T2\tPER\tZoe",
        "T2\tPER 8 11",
        "T2\tPER wrong 11\tZoe",
    ] {
        fs::write(&annotation, format!("T1\tPER 0 3\tAda\n{malformed}\n")).unwrap();
        let result = import_as("brat", &annotation, &dest, false);
        assert!(!result.status.success(), "{malformed}");
        assert!(
            String::from_utf8_lossy(&result.stderr).contains("line 2"),
            "{malformed}"
        );
        assert_eq!(fs::read(&dest).unwrap(), b"reviewed corpus\n");
    }
}
