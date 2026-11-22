use std::fs;
use std::path::Path;
use std::process::{Command, Output};

fn import(input: &Path, output: &Path, partial: bool) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_anno"));
    cmd.args([
        "dataset", "import", "--format", "jsonl", "--quiet", "--input",
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
