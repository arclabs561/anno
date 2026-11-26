// Run from the repository root: kani verification/mention_spans/check.rs
#[path = "../../crates/anno/src/core/coref/span.rs"]
mod span;

#[kani::proof]
fn overlap_iff_shared_character() {
    let (a_start, a_end, b_start, b_end): (usize, usize, usize, usize) = kani::any();
    // The later start is the first possible shared character.
    let first = a_start.max(b_start);
    let shared = (a_start..a_end).contains(&first) && (b_start..b_end).contains(&first);
    assert_eq!(span::overlaps(a_start, a_end, b_start, b_end), shared);
}
