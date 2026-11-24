// SPDX-License-Identifier: MIT OR Apache-2.0
use std::{path::Path, process::Command};

#[test]
fn sqlite_cached_memberships_agree_with_fresh_cli_queries_after_bridge_edits() {
    let script =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../verification/memberships/replay.py");
    let interpreter = if cfg!(windows) { "python" } else { "python3" };
    let status = Command::new(interpreter)
        .arg(script)
        .arg(env!("CARGO_BIN_EXE_anno"))
        .status()
        .expect("run the standard-library SQLite reader");
    assert!(status.success());
}
