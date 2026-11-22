use serde_json::Value;
use std::{fs, path::Path, process::Command};

#[test]
fn checked_conll_spans_slice_the_exact_document_revision() {
    let root = tempfile::tempdir().unwrap();
    let conll = root.path().join("sample.conll");
    let original = root.path().join("revision.txt");
    let output = root.path().join("entities.jsonl");
    let tokens = "-DOCSTART- -X- O O\n\n🦊 O\nZoë B-PER\nLi I-PER\n\nZoë B-PER\nLi I-PER\n. O\n";
    let source = "\t🦊  Zoë\tLi\r\nZoë\u{a0}Li.\r\n";
    fs::write(&conll, tokens).unwrap();
    fs::write(&original, source).unwrap();
    let run = |input: &Path, dest: &Path, checked: bool| {
        let mut command = Command::new(env!("CARGO_BIN_EXE_anno"));
        command
            .args(["dataset", "import", "--format", "conll", "--input"])
            .arg(input)
            .arg("--output")
            .arg(dest);
        if checked {
            command.arg("--source-text").arg(&original);
        }
        command.output().unwrap()
    };
    let rows = || {
        fs::read_to_string(&output)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect::<Vec<_>>()
    };

    assert!(run(&conll, &output, false).status.success());
    let reconstructed = rows();
    assert_eq!(
        (
            reconstructed[0]["start"].as_u64(),
            reconstructed[1]["start"].as_u64()
        ),
        (Some(2), Some(9))
    );
    assert_eq!(reconstructed[0]["text"], "Zoë Li");

    let checked = run(&conll, &output, true);
    assert!(
        checked.status.success(),
        "{}",
        String::from_utf8_lossy(&checked.stderr)
    );
    let aligned = rows();
    assert_eq!(aligned.len(), 2);
    let chars: Vec<_> = source.chars().collect();
    for (row, (start, end)) in aligned.iter().zip([(4, 10), (12, 18)]) {
        assert_eq!(row["start"], start);
        assert_eq!(row["end"], end);
        assert_eq!(row["text"], chars[start..end].iter().collect::<String>());
    }

    fs::write(&original, source.replace('\u{a0}', "\r\n")).unwrap();
    assert!(run(&conll, &output, true).status.success());
    let multiline = rows();
    assert_eq!(multiline[1]["text"], "Zoë\r\nLi");
    assert_eq!(multiline[1]["end"], 19);

    let previous = fs::read(&output).unwrap();
    for (token_file, text) in [
        (tokens, source.replace("🦊", "🐈")),
        (
            tokens,
            source.trim_end_matches(['\r', '\n', '.']).to_owned(),
        ),
        (tokens.trim_end_matches(". O\n"), source.to_owned()),
        ("🦊\n", "🦊".to_owned()),
        ("🦊 NNP\n", "🦊".to_owned()),
    ] {
        fs::write(&conll, token_file).unwrap();
        fs::write(&original, &text).unwrap();
        assert!(
            !run(&conll, &output, true).status.success(),
            "accepted {token_file:?} against {text:?}"
        );
        assert_eq!(fs::read(&output).unwrap(), previous);
    }
    fs::write(&conll, tokens).unwrap();
    fs::write(&original, source).unwrap();
    let alias = root.path().join("alias.jsonl");
    fs::hard_link(&original, &alias).unwrap();
    for protected in [&original, &alias, &conll] {
        assert!(!run(&conll, protected, true).status.success());
    }
    #[cfg(unix)]
    {
        let symbolic = root.path().join("alias-symbolic.jsonl");
        std::os::unix::fs::symlink(&original, &symbolic).unwrap();
        assert!(!run(&conll, &symbolic, true).status.success());
        assert!(fs::symlink_metadata(symbolic).unwrap().is_symlink());
    }
    assert!(same_file::is_same_file(&original, &alias).unwrap());
    assert_eq!(fs::read_to_string(&original).unwrap(), source);
    assert_eq!(fs::read_to_string(&conll).unwrap(), tokens);
    assert!(!run(root.path(), &output, true).status.success());
    assert_eq!(fs::read(&output).unwrap(), previous);
}
