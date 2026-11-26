//! Merge step for spans already sorted by start. Touching spans stay separate.

pub(crate) const fn absorb(end: &mut usize, next_start: usize, next_end: usize) -> bool {
    if next_start >= *end {
        return false;
    }
    if next_end > *end {
        *end = next_end;
    }
    true
}

const _: () = {
    let mut end = 4;
    assert!(absorb(&mut end, 2, 6) && end == 6);
    assert!(absorb(&mut end, 3, 4) && end == 6);
    assert!(!absorb(&mut end, 6, 8) && end == 6);
};
