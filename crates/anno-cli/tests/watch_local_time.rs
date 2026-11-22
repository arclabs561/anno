use jiff::{tz::Offset, Timestamp};
use std::{fs, process::Command};

#[test]
fn watch_reports_local_time() {
    let dir = tempfile::tempdir().unwrap();
    // The second input lets the watch loop stop at its one-file limit.
    for name in ["first.txt", "second.txt"] {
        fs::write(dir.path().join(name), "Alice visited Paris.").unwrap();
    }
    let start = Timestamp::now().as_second();
    let output = Command::new(env!("CARGO_BIN_EXE_anno"))
        .args(["watch", "--directory"])
        .arg(dir.path())
        .args(["--model", "pattern", "--initial", "--max-files", "1"])
        .env("TZ", "Asia/Kathmandu")
        .output()
        .unwrap();
    let end = Timestamp::now().as_second();
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(output.status.success(), "{stderr}");
    let reported = stderr
        .lines()
        .find(|line| line.ends_with(" entities"))
        .and_then(|line| line.strip_prefix('[')?.split_once(']'))
        .map(|(clock, _)| clock)
        .unwrap_or_else(|| panic!("missing watch timestamp: {stderr}"));

    // Nepal's UTC+05:45 also catches a formatter that drops offset minutes.
    let offset = Offset::from_seconds(5 * 3600 + 45 * 60).unwrap();
    let matches_local_time = (start..=end).any(|second| {
        Timestamp::from_second(second)
            .unwrap()
            .to_zoned(offset.to_time_zone())
            .strftime("%T")
            .to_string()
            == reported
    });
    assert!(matches_local_time, "unexpected local time: {stderr}");
}
