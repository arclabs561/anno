//! Regression tests for High-severity bugs from the Oct 2026 bug hunt.

use anno::backends::coref::mention_ranking::{MentionRankingConfig, MentionRankingCoref};
use anno::pii;

/// IBAN_PATTERN required 7 digits after the 4-char bank code and capped length
/// at 31, so valid compact IBANs from RO, KZ (letters in BBAN positions 9-15)
/// and LC (32), RU (33) were not recognised. All four pass mod-97.
#[test]
fn compact_ibans_with_alnum_bban_or_long_length_are_redacted() {
    let mut leaks = Vec::new();
    for iban in [
        "RO49AAAA1B31007593840000",
        "KZ86125KZT5004100100",
        "LC55HEMM000100010012001200023015",
        "RU0304452522540817810538091310419",
    ] {
        let text = format!("Pay to {iban} today");
        let out = pii::redact_patterns(&text);
        if out.contains(&iban[..8]) {
            leaks.push(out);
        }
    }
    assert!(leaks.is_empty(), "leaked: {leaks:?}");
}

fn spans_in_multiple_clusters(
    clusters: &[anno::backends::coref::mention_ranking::MentionCluster],
) -> Vec<(usize, usize, String)> {
    let mut seen = std::collections::HashMap::new();
    let mut dup = Vec::new();
    for c in clusters {
        for m in &c.mentions {
            if let Some(prev) = seen.insert((m.start, m.end), c.id) {
                if prev != c.id {
                    dup.push((m.start, m.end, m.text.clone()));
                }
            }
        }
    }
    dup
}

/// Easy-first clustering (the `book_scale()` default) only tracked whether a
/// mention already *has an antecedent*, not whether it is already in a
/// cluster as someone else's antecedent. When pair (he2 -> he1) is processed
/// before (he1 -> John), "he1" was put into a second cluster instead of the
/// two clusters being merged, so the output was not a partition.
#[test]
fn easy_first_clusters_are_a_partition() {
    let coref = MentionRankingCoref::with_config(MentionRankingConfig::book_scale());
    let text = "Yesterday John arrived. He said he was tired.";
    let clusters = coref.resolve(text).unwrap();
    let dup = spans_in_multiple_clusters(&clusters);
    assert!(
        dup.is_empty(),
        "mentions in more than one cluster: {dup:?}; clusters: {:?}",
        clusters
            .iter()
            .map(|c| c
                .mentions
                .iter()
                .map(|m| (m.start, m.text.clone()))
                .collect::<Vec<_>>())
            .collect::<Vec<_>>()
    );
}
