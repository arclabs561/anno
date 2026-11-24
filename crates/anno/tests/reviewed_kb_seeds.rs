#[path = "support/reviewed_kb_fuzz.rs"]
mod harness;

use std::collections::BTreeMap;
use std::{fs, path::Path};

fn regressions() -> BTreeMap<&'static str, (&'static [u8], Option<usize>)> {
    BTreeMap::from([
        ("changed-and-restored", (b"pBA".as_slice(), None)),
        ("foreign-owner", (b"1pc", Some(2))),
        ("independent-same-key", (b"p2A", Some(2))),
        ("local-rebind", (b"p3A", None)),
        ("local-siblings", (b"3p!4p", Some(1))),
        ("unlinked-track", (b"5p", Some(1))),
        ("shared-namespace-change", (b"p1N", None)),
    ])
}

fn replay_case(name: &str, seed: &[u8]) {
    let outcome = harness::replay(seed);
    let family = name.split_once("--").map_or(name, |(family, _)| family);
    if let Some((witness, expected)) = regressions().get(family) {
        let operations: Vec<_> = seed
            .iter()
            .take(harness::MAX_INPUT_BYTES)
            .copied()
            .filter(|b| !b.is_ascii_whitespace())
            .collect();
        assert!(
            operations
                .windows(witness.len())
                .any(|window| window == *witness),
            "{name} lost its distinguishing operations"
        );
        assert_eq!(outcome, *expected, "{name}");
    }
}

#[test]
fn shared_seeds_replay_against_the_catalog_assignment_oracle() {
    let mut regressions = regressions();
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fuzz/seeds/reviewed_kb");
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        let seed = fs::read(&path).unwrap();
        eprintln!("Replaying {}", path.display());
        let name = path.file_name().unwrap().to_str().unwrap();
        replay_case(name, &seed);
        regressions.remove(name);
    }
    assert!(
        regressions.is_empty(),
        "missing shared seeds: {:?}",
        regressions.keys()
    );
}

fn hash(input: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(input)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn failure_signature(payload: Box<dyn std::any::Any + Send>) -> String {
    if payload.is::<harness::VisibleBindingMismatch>() {
        return hash(b"anno:reviewed-kb:visible-binding-mismatch:v1");
    }
    let detail = payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| payload.downcast_ref::<&str>().copied());
    match detail {
        Some(detail) => hash(detail.as_bytes()),
        None => std::panic::resume_unwind(payload),
    }
}

#[test]
fn visible_binding_receipt_is_stable_and_distinct_from_an_assertion() {
    let direct = failure_signature(Box::new(harness::VisibleBindingMismatch));
    let actual =
        std::panic::catch_unwind(|| std::panic::panic_any(harness::VisibleBindingMismatch))
            .unwrap_err();
    assert_eq!(direct, failure_signature(actual));
    assert_ne!(direct, failure_signature(Box::new("unexpected assertion")));
}

/// Machine-readable, single-case replay for the opt-in QA scheduler.
#[test]
#[ignore = "the scheduler supplies a reviewed seed and a new receipt path"]
fn replay_one_reviewed_seed() {
    let path =
        std::path::PathBuf::from(std::env::var_os("ANNO_REVIEWED_SEED").expect("missing seed"));
    let seed = fs::read(&path).unwrap();
    let name = path.file_name().unwrap().to_str().unwrap();
    assert!(regressions().contains_key(name), "unknown reviewed seed");
    let result = std::panic::catch_unwind(|| replay_case(name, &seed));
    let (outcome, signature) = match result {
        Ok(_) => ("pass", String::new()),
        Err(payload) => ("fail", failure_signature(payload)),
    };
    let receipt = serde_json::json!({
        "outcome": outcome, "signature": signature,
        "source_sha256": hash(&seed),
        "oracle_files": [
            hash(include_bytes!("reviewed_kb_seeds.rs")),
            hash(include_bytes!("support/reviewed_kb_fuzz.rs")),
        ],
    });
    let path = std::env::var_os("ANNO_REVIEWED_RECEIPT").expect("missing receipt path");
    serde_json::to_writer(fs::File::create_new(path).unwrap(), &receipt).unwrap();
}
