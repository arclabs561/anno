use std::fs;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

#[test]
fn watch_stops_after_the_last_allowed_file() {
    for available in [1, 2] {
        let dir = tempfile::tempdir().unwrap();
        for i in 0..available {
            fs::write(dir.path().join(format!("{i}.txt")), "Alice visited Paris.").unwrap();
        }

        let mut child = Command::new(env!("CARGO_BIN_EXE_anno"))
            .args(["watch", "--directory"])
            .arg(dir.path())
            .args(["--model", "pattern", "--initial", "--max-files", "1"])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();

        let deadline = Instant::now() + Duration::from_secs(30);
        while child.try_wait().unwrap().is_none() {
            if Instant::now() >= deadline {
                child.kill().unwrap();
                let output = child.wait_with_output().unwrap();
                panic!(
                    "watch kept polling with {available} input(s): {}",
                    String::from_utf8_lossy(&output.stderr)
                );
            }
            thread::sleep(Duration::from_millis(20));
        }

        let output = child.wait_with_output().unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(output.status.success(), "{stderr}");
        let record: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert!(record["entities"].is_array());
        assert!(stderr.contains("Reached max files limit (1)"), "{stderr}");
    }
}
