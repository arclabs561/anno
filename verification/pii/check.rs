// Run with Kani 0.68.0: kani verification/pii/check.rs
#[path = "../../crates/anno/src/pii/span_union.rs"]
mod span_union;

#[kani::proof]
fn merge_preserves_exact_coverage_and_boundaries() {
    let (start, end, next_start, next_end, position): (usize, usize, usize, usize, usize) =
        kani::any();
    kani::assume(start <= end && start <= next_start && next_start <= next_end);
    let before = (start..end).contains(&position) || (next_start..next_end).contains(&position);
    let mut merged_end = end;
    let absorbed = span_union::absorb(&mut merged_end, next_start, next_end);
    assert_eq!(absorbed, next_start < end);
    if absorbed {
        assert_eq!((start..merged_end).contains(&position), before);
    } else {
        assert_eq!(merged_end, end);
    }
}
