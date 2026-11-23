#[path = "support/reviewed_kb_fuzz.rs"]
mod harness;

use std::collections::BTreeMap;
use std::{fs, path::Path};

#[test]
fn shared_seeds_replay_against_the_catalog_assignment_oracle() {
    // Each named regression keeps a distinguishing operation and expected final
    // receipt size. A rejected stale preview has no receipt. Extra shared seeds
    // also run through the oracle, which checks every intermediate operation.
    let mut regressions: BTreeMap<&str, (&[u8], Option<usize>)> = BTreeMap::from([
        ("changed-and-restored", (b"pBA".as_slice(), None)),
        ("foreign-owner", (b"1pc", Some(2))),
        ("independent-same-key", (b"p2A", Some(2))),
        ("local-rebind", (b"p3A", None)),
        ("local-siblings", (b"3p!4p", Some(1))),
        ("unlinked-track", (b"5p", Some(1))),
        ("shared-namespace-change", (b"p1N", None)),
    ]);
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fuzz/seeds/reviewed_kb");
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        let seed = fs::read(&path).unwrap();
        eprintln!("Replaying {}", path.display());
        let outcome = harness::replay(&seed);
        let name = path.file_name().unwrap().to_str().unwrap();
        if let Some((witness, expected)) = regressions.remove(name) {
            let operations: Vec<_> = seed
                .iter()
                .take(harness::MAX_INPUT_BYTES)
                .copied()
                .filter(|b| !b.is_ascii_whitespace())
                .collect();
            assert!(
                operations
                    .windows(witness.len())
                    .any(|window| window == witness),
                "{name} lost its distinguishing operations"
            );
            assert_eq!(outcome, expected, "{name}");
        }
    }
    assert!(
        regressions.is_empty(),
        "missing shared seeds: {:?}",
        regressions.keys()
    );
}
