#![cfg(unix)]

use std::fs;
use std::os::unix::fs::{symlink, PermissionsExt};
use std::path::Path;
use std::process::{Command, Output};

fn run_export(path: &Path) -> Output {
    Command::new("sh")
        .args(["-c", "umask 022; exec \"$@\"", "privacy-map-test"])
        .arg(env!("CARGO_BIN_EXE_anno"))
        .args([
            "privacy",
            "--model",
            "pattern",
            "--action",
            "pseudonymize",
            "--text",
            "Contact alice@example.com",
            "--export-map",
        ])
        .arg(path)
        .env("ANNO_NO_DOWNLOADS", "1")
        .output()
        .unwrap()
}

fn export_map(path: &Path) {
    let output = run_export(path);
    assert!(
        output.status.success(),
        "export failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!String::from_utf8_lossy(&output.stdout).contains("alice@example.com"));
    assert!(fs::read_to_string(path)
        .unwrap()
        .contains("alice@example.com\t"));
    assert_eq!(
        fs::metadata(path).unwrap().permissions().mode() & 0o777,
        0o600
    );
}

#[test]
fn new_reidentification_map_uses_owner_read_write_mode() {
    let dir = tempfile::tempdir().unwrap();
    export_map(&dir.path().join("mapping.tsv"));
}

#[test]
fn overwriting_a_shared_map_restricts_permissions_and_replaces_contents() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mapping.tsv");
    fs::write(&path, "stale mapping\n".repeat(100)).unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
    export_map(&path);
    assert!(!fs::read_to_string(path).unwrap().contains("stale mapping"));
}

#[test]
fn export_rejects_symlink_without_modifying_target() {
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("original.txt");
    let link = dir.path().join("mapping.tsv");
    fs::write(&target, "preserve original").unwrap();
    fs::set_permissions(&target, fs::Permissions::from_mode(0o644)).unwrap();
    symlink(&target, &link).unwrap();

    let output = run_export(&link);
    assert!(!output.status.success());
    assert_eq!(fs::read_to_string(&target).unwrap(), "preserve original");
    assert_eq!(
        fs::metadata(&target).unwrap().permissions().mode() & 0o777,
        0o644
    );
    assert!(fs::symlink_metadata(&link)
        .unwrap()
        .file_type()
        .is_symlink());
}
