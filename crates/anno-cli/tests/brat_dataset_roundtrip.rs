use anno::{DiscontinuousSpan, Entity, EntityType, GroundedDocument};
use serde_json::{json, Value};
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

fn read_rows(path: &Path) -> Vec<Value> {
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|row| serde_json::from_str(row).unwrap())
        .collect()
}

#[test]
fn repeated_segmented_mentions_keep_their_ranges_in_each_revision() {
    let temp = tempfile::tempdir().unwrap();
    let mention = "🦊 e\u{301} met Zoë";
    let joined = "🦊 e\u{301} Zoë";

    for (revision, prefix) in [("v1", ""), ("v2", "改: ")] {
        let dir = temp.path().join(revision);
        fs::create_dir(&dir).unwrap();
        let text = format!("{prefix}{mention} | {mention}");
        let first = prefix.chars().count();
        let starts = [first, first + 15];
        let entities: Vec<_> = starts
            .iter()
            .map(|&s| {
                let mut entity = Entity::new(mention, EntityType::Person, s, s + 12, 0.75);
                entity.set_discontinuous_span(DiscontinuousSpan::new(vec![
                    s..s + 1,
                    s + 2..s + 4,
                    s + 9..s + 12,
                ]));
                entity
            })
            .collect();
        let doc = GroundedDocument::from_entity_signals("record-7", &text, &entities);
        fs::write(dir.join("record.json"), serde_json::to_vec(&doc).unwrap()).unwrap();
        run(
            &dir,
            &[
                "export",
                "--input",
                "record.json",
                "--output",
                "brat",
                "--format",
                "brat",
                "--model",
                "pattern",
                "--include-confidence",
            ],
        );
        assert_eq!(
            fs::read(dir.join("brat/record.txt")).unwrap(),
            text.as_bytes()
        );
        // Import from a common working directory so the source path names the revision.
        let ann = format!("{revision}/brat/record.ann");
        let rows = format!("{revision}/rows.jsonl");
        run(
            temp.path(),
            &[
                "dataset",
                "import",
                "--format",
                "brat",
                "--include-text",
                "--input",
                &ann,
                "--output",
                &rows,
            ],
        );
        let records = read_rows(&temp.path().join(&rows));
        assert_eq!(records.len(), 2);
        for (row, s) in records.iter().zip(starts) {
            assert_eq!(row["source"], ann);
            assert_eq!(row["type"], "PER");
            assert_eq!(row["confidence"], 0.75);
            assert_eq!(row["text"], joined);
            assert_eq!(
                (row["start"].as_u64(), row["end"].as_u64()),
                (Some(s as u64), Some((s + 12) as u64))
            );
            assert_eq!(
                row["segments"],
                json!([
                    {"start": s, "end": s + 1}, {"start": s + 2, "end": s + 4},
                    {"start": s + 9, "end": s + 12}
                ])
            );
        }
        let copy = format!("{revision}/copy.jsonl");
        run(
            temp.path(),
            &[
                "dataset", "import", "--format", "jsonl", "--input", &rows, "--output", &copy,
            ],
        );
        assert_eq!(read_rows(&temp.path().join(copy)), records);
    }
}

#[test]
fn import_identifies_the_second_speaker_by_revision_and_offset() {
    let tmp = tempfile::tempdir().unwrap();
    for (revision, text, expected_start) in [
        ("v1", "😀 Ada agreed.\nAda objected.", 14),
        ("v2", "Ada objected.\n😀 Ada agreed.", 0),
    ] {
        let doc = tmp.path().join(revision);
        fs::create_dir(&doc).unwrap();
        fs::write(doc.join("source.txt"), text).unwrap();
        let ann = format!("{revision}/source.ann");
        fs::write(
            tmp.path().join(&ann),
            format!("T1\tPER {expected_start} {}\tAda\n", expected_start + 3),
        )
        .unwrap();
        run(
            tmp.path(),
            &[
                "dataset",
                "import",
                "--format",
                "brat",
                "--input",
                &ann,
                "--output",
                "rows.jsonl",
                "--include-text",
            ],
        );
        let rows = read_rows(&tmp.path().join("rows.jsonl"));
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row["source"], ann);
        assert_eq!(row["text"], "Ada");
        let start = row["start"].as_u64().unwrap() as usize;
        assert_eq!(
            text.chars().skip(start).take(12).collect::<String>(),
            "Ada objected"
        );
    }
}
