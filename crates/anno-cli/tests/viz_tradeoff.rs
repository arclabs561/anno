//! Exercise file, pipeline and publication behavior through the shipped binary.

use std::fs;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Output, Stdio};

const STUDY: &str = r#"{"schema":"anno.tradeoff/v1","title":"Cohort <&>","unit":"documents","series":[{"label":"Held out","observations":2,"gold":1,"points":[{"label":"Fallback","correct":1,"unmatched":3}]}]}"#;

fn tradeoff(dir: &Path, args: &[&str], stdin: &[u8]) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_anno"))
        .current_dir(dir)
        .env("ANNO_NO_DOWNLOADS", "1")
        .args(["viz", "tradeoff"])
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(stdin).unwrap();
    child.wait_with_output().unwrap()
}

fn successful(output: Output) -> String {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

#[test]
fn render_through_stdio_or_files_and_choose_html() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path();
    let svg = successful(tradeoff(path, &[], STUDY.as_bytes()));
    assert!(svg.starts_with("<svg") && svg.contains("Cohort &lt;&amp;&gt;"));
    fs::write(path.join("study.json"), STUDY).unwrap();
    assert!(successful(tradeoff(path, &["study.json", "-o", "chart.svg"], &[])).is_empty());
    assert_eq!(fs::read_to_string(path.join("chart.svg")).unwrap(), svg);
    successful(tradeoff(path, &["study.json", "-o", "verified.json"], &[]));
    let checked: serde_json::Value =
        serde_json::from_slice(&fs::read(path.join("verified.json")).unwrap()).unwrap();
    assert_eq!(
        checked,
        serde_json::from_str::<serde_json::Value>(STUDY).unwrap()
    );
    assert_eq!(successful(tradeoff(path, &["verified.json"], &[])), svg);
    let html = successful(tradeoff(
        path,
        &["-", "-o", "-", "--format", "html"],
        STUDY.as_bytes(),
    ));
    assert!(html.contains("<table>") && html.contains("<td>50.000</td><td>150.000</td>"));
    assert!(successful(tradeoff(
        path,
        &["study.json", "--output", "chart.HTML"],
        &[]
    ))
    .is_empty());
    assert_eq!(fs::read_to_string(path.join("chart.HTML")).unwrap(), html);
    successful(tradeoff(
        path,
        &["study.json", "--output", "chart.HTML", "--format", "svg"],
        &[],
    ));
    assert_eq!(fs::read_to_string(path.join("chart.HTML")).unwrap(), svg);
}

#[test]
fn invalid_reports_leave_existing_input_and_output_intact() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path();
    fs::write(path.join("study.json"), STUDY).unwrap();
    fs::write(path.join("published.svg"), "previous chart").unwrap();
    for (input, diagnostic) in [
        ("{".to_owned(), "invalid operating-point JSON"),
        (
            STUDY.replace("v1", "v2"),
            "expected schema anno.tradeoff/v1",
        ),
        (
            STUDY.replace("\"correct\":1", "\"correct\":2"),
            "points[0].correct",
        ),
    ] {
        let output = tradeoff(path, &["-o", "published.svg"], input.as_bytes());
        assert!(!output.status.success() && output.stdout.is_empty());
        assert!(String::from_utf8_lossy(&output.stderr).contains(diagnostic));
        assert_eq!(
            fs::read_to_string(path.join("published.svg")).unwrap(),
            "previous chart"
        );
    }
    let same = tradeoff(path, &["study.json", "-o", "./study.json"], &[]);
    assert!(!same.status.success() && String::from_utf8_lossy(&same.stderr).contains("same file"));
    assert_eq!(fs::read_to_string(path.join("study.json")).unwrap(), STUDY);
}
