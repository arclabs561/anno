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

/// Machine-readable, single-case replay for the opt-in QA scheduler.
#[test]
#[ignore = "the scheduler supplies a reviewed seed and a new receipt path"]
fn replay_one_reviewed_seed() {
    use sha2::{Digest, Sha256};

    let path =
        std::path::PathBuf::from(std::env::var_os("ANNO_REVIEWED_SEED").expect("missing seed"));
    let seed = fs::read(&path).unwrap();
    let name = path.file_name().unwrap().to_str().unwrap();
    let hash = |input: &[u8]| -> String {
        Sha256::digest(input)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    };
    let result = std::panic::catch_unwind(|| replay_case(name, &seed));
    let (outcome, signature) = match result {
        Ok(_) => ("pass", String::new()),
        Err(payload) => {
            let detail = payload
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| payload.downcast_ref::<&str>().copied())
                .expect("oracle panic needs a string diagnostic");
            ("fail", hash(detail.as_bytes()))
        }
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
