//! Opt-in process high-water measurement for the CLI's atomic cluster publication.
//! Run on Linux with `cargo test -p anno-cli --no-default-features --features
//! extractor-html2text --bin anno publication_peak_rss -- --ignored --nocapture`.
//! Set `ANNO_COREF_RSS_SIGNAL_STRIDE` to seed one signal every N eight-byte blocks.
//! The original sparse fixture uses 128. A stride of 8 gives 16 times as many signals.

use anno::backends::coref::resolve::CorefCluster;
use anno::{GroundedDocument, Location, Signal, TypeLabel};
use std::{env, fs, num::NonZeroUsize, process::Command};

const CHILD: &str = "ANNO_COREF_RSS_CASE";

fn measure(case: &str, bytes: usize, encoding: &str, stride: NonZeroUsize) {
    // Both blocks occupy eight bytes. Only the second needs Unicode offset tables.
    let block = match encoding {
        "ascii" => "Ada Lea!",
        "unicode" => "Ada Léa",
        _ => panic!("unknown encoding"),
    };
    let blocks = bytes / block.len();
    let mut doc = GroundedDocument::new("rss", block.repeat(blocks));
    let chars = block.chars().count();
    for index in (0..blocks).step_by(stride.get()) {
        let start = index * chars;
        doc.add_signal(Signal::new(
            0u64,
            Location::text(start, start + 3),
            "Ada",
            TypeLabel::from("PER"),
            1.0,
        ));
    }
    let last = (blocks - 1) * chars;
    let cluster = CorefCluster {
        id: 1,
        canonical: "Ada".into(),
        mentions: vec!["Ada".into(), "Ada".into()],
        spans: vec![(0, 3), (last, last + 3)],
    };
    let mut later = cluster.clone();
    later.id = 2;
    if case == "invalid" {
        later.mentions[1] = "Eve".into();
    }
    let mut clusters = [cluster, later];
    if case == "singleton" {
        for cluster in &mut clusters {
            cluster.mentions.truncate(1);
            cluster.spans.truncate(1);
        }
    }
    let before = doc.signals().len();
    assert_eq!(before, blocks.div_ceil(stride.get()));
    let added = match case {
        "baseline" => 0,
        "empty" => super::append_coref_clusters(&mut doc, &clusters[..0]).unwrap(),
        "singleton" => {
            assert_eq!(
                super::append_coref_clusters(&mut doc, &clusters).unwrap(),
                0
            );
            0
        }
        "success" => {
            let added = super::append_coref_clusters(&mut doc, &clusters).unwrap();
            assert_eq!(added, 2);
            added
        }
        "invalid" => {
            assert!(super::append_coref_clusters(&mut doc, &clusters).is_err());
            0
        }
        _ => panic!("unknown case"),
    };
    assert_eq!(doc.signals().len(), before + added * 2);
    assert_eq!(doc.tracks().count(), added);
    std::hint::black_box((&doc, &clusters));
    let status = fs::read_to_string("/proc/self/status").unwrap();
    println!(
        "{}",
        status.lines().find(|l| l.starts_with("VmHWM:")).unwrap()
    );
}

#[test]
#[ignore = "opt-in Linux process peak RSS measurement"]
fn publication_peak_rss() {
    let stride = env::var("ANNO_COREF_RSS_SIGNAL_STRIDE")
        .unwrap_or_else(|_| "128".into())
        .parse::<NonZeroUsize>()
        .expect("ANNO_COREF_RSS_SIGNAL_STRIDE must be a positive integer");
    if let Ok(case) = env::var(CHILD) {
        let bytes = env::var("ANNO_COREF_RSS_BYTES").unwrap().parse().unwrap();
        measure(
            &case,
            bytes,
            &env::var("ANNO_COREF_RSS_ENCODING").unwrap(),
            stride,
        );
        return;
    }

    let binary = env::current_exe().unwrap();
    let thread = std::thread::current();
    let name = thread.name().unwrap();
    println!("encoding,source_bytes,signal_stride_blocks,existing_signals,case,median_peak_kib,raw_peak_kib");
    for encoding in ["ascii", "unicode"] {
        for bytes in [65_536usize, 1_048_576, 4_194_304] {
            let existing_signals = (bytes / 8).div_ceil(stride.get());
            for case in ["baseline", "empty", "singleton", "invalid", "success"] {
                let mut peaks = [0u64; 3];
                for peak in &mut peaks {
                    let output = Command::new(&binary)
                        .args([
                            "--ignored",
                            "--exact",
                            name,
                            "--nocapture",
                            "--test-threads=1",
                        ])
                        .env(CHILD, case)
                        .env("ANNO_COREF_RSS_BYTES", bytes.to_string())
                        .env("ANNO_COREF_RSS_ENCODING", encoding)
                        .output()
                        .unwrap();
                    assert!(output.status.success(), "{output:?}");
                    let stdout = String::from_utf8(output.stdout).unwrap();
                    let (_, value) = stdout.split_once("VmHWM:").expect("child peak RSS");
                    *peak = value.split_whitespace().next().unwrap().parse().unwrap();
                }
                let mut sorted = peaks;
                sorted.sort_unstable();
                println!(
                    "{encoding},{bytes},{stride},{existing_signals},{case},{},{}|{}|{}",
                    sorted[1], peaks[0], peaks[1], peaks[2]
                );
            }
        }
    }
}
