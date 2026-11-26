use anno::{DiscontinuousSpan, Entity, EntityType, GroundedDocument};
use std::{fs, path::Path, process::Command};

fn run(dir: &Path, args: &[&str]) {
    let result = Command::new(env!("CARGO_BIN_EXE_anno"))
        .current_dir(dir)
        .args(args)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{args:?}: {}",
        String::from_utf8_lossy(&result.stderr)
    );
}

#[test]
fn pair_manifest_binds_both_files_across_renames_and_preserves_old_output_on_failure() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let text = "😀 Ada agreed.\nAda denied.";
    let entity = Entity::new("Ada", EntityType::Person, 14, 17, 1.0);
    let doc = GroundedDocument::from_entity_signals("letter", text, &[entity]);
    fs::write(dir.join("letter.json"), serde_json::to_vec(&doc).unwrap()).unwrap();
    run(
        dir,
        &[
            "export",
            "--input",
            "letter.json",
            "--output",
            ".",
            "--format",
            "brat",
            "--model",
            "pattern",
        ],
    );
    for ext in ["ann", "txt", "anno-pair"] {
        fs::rename(
            dir.join(format!("letter.{ext}")),
            dir.join(format!("copy.{ext}")),
        )
        .unwrap();
    }
    let import = |extra: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_anno"))
            .current_dir(dir)
            .args([
                "dataset",
                "import",
                "--input",
                "copy.ann",
                "--output",
                "rows.jsonl",
            ])
            .args(extra)
            .output()
            .unwrap()
    };
    assert!(import(&["--require-brat-manifest"]).status.success());
    let prior = fs::read(dir.join("rows.jsonl")).unwrap();
    let reject = |args: &[&str], message: &str| {
        let result = import(args);
        assert!(!result.status.success());
        assert!(String::from_utf8_lossy(&result.stderr).contains(message));
        assert_eq!(fs::read(dir.join("rows.jsonl")).unwrap(), prior);
    };
    fs::write(dir.join("copy.txt"), "😀 Ada denied.\nAda agreed.").unwrap();
    reject(&[], "source text digest differs");
    fs::write(dir.join("copy.txt"), text).unwrap();
    fs::write(dir.join("copy.ann"), "T1\tPER 2 5\tAda\n").unwrap();
    reject(&[], "annotation digest differs");
    fs::remove_file(dir.join("copy.anno-pair")).unwrap();
    reject(&["--require-brat-manifest"], "Missing");
    let legacy = import(&[]);
    assert!(legacy.status.success());
    assert!(String::from_utf8_lossy(&legacy.stderr).contains("revision unverified"));
}
