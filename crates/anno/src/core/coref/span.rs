/// Whether two half-open character ranges share a position.
pub(crate) const fn overlaps(a_start: usize, a_end: usize, b_start: usize, b_end: usize) -> bool {
    a_start < a_end && b_start < b_end && a_start < b_end && b_start < a_end
}

// Evaluated by rustc on every build, including cargo check.
const _: () = {
    assert!(!overlaps(1, 1, 0, 2));
    assert!(!overlaps(2, 1, 0, 3));
    assert!(!overlaps(0, 1, 1, 2));
    assert!(overlaps(0, usize::MAX, usize::MAX - 1, usize::MAX));
};
