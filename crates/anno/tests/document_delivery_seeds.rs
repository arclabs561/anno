// SPDX-License-Identifier: MIT OR Apache-2.0
#[path = "support/document_delivery.rs"]
mod harness;

use std::{collections::BTreeMap, fs, path::Path};

#[test]
fn shared_histories_replay_with_expected_receipts_and_reader_bindings() {
    let old = Some("17");
    let reviewed = Some("23");
    let mut expected = BTreeMap::from([
        (
            "reviewed-peers-and-forks",
            ("aaaaaaprx!", [reviewed, reviewed, old, old, old, old]),
        ),
        (
            "reviewed-source-edit",
            ("ap!qqarr", [None, None, old, None, None, None]),
        ),
        (
            "late-restored-review",
            ("ap!aqqarr", [reviewed, reviewed, None, None, None, None]),
        ),
        (
            "issuer-token-collision",
            ("aaxarr", [old, old, None, old, None, None]),
        ),
        (
            "historical-receipt",
            ("aassrxxp!r", [None, None, reviewed, None, None, None]),
        ),
    ]);
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fuzz/seeds/document_delivery");
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_str().unwrap();
        let (outcomes, keys) = harness::replay(&fs::read(&path).unwrap());
        assert_eq!(
            (outcomes.as_str(), keys),
            expected.remove(name).expect("missing seed expectation"),
            "{name}"
        );
    }
    assert!(
        expected.is_empty(),
        "missing seed files: {:?}",
        expected.keys()
    );
}

#[test]
fn both_delivery_orders_preserve_peer_ownership_and_stale_previews() {
    for (first, peer) in [(b'0', b'1'), (b'1', b'0')] {
        let history = [
            first, b'a', b'p', b'!', b'p', peer, b'a', b'!', first, b'r', peer, b'r',
        ];
        let (outcomes, keys) = harness::replay(&history);
        assert_eq!(outcomes, "ap!pa~rr");
        assert_eq!(keys, [Some("23"), Some("23"), None, None, None, None]);
    }
}

#[test]
#[ignore = "set ANNO_DELIVERY_INPUT to a local reproducer"]
fn replay_local_delivery_input() {
    let path = std::env::var_os("ANNO_DELIVERY_INPUT").expect("missing reproducer path");
    let _ = harness::replay(&fs::read(path).unwrap());
}
