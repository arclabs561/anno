//! In-place compaction of runs sorted by start. See
//! `verification/pii/traversal.rs` for the pointwise coverage theorem.
#![allow(unexpected_cfgs)] // verus_only is set by the standalone verifier.

#[cfg(verus_only)]
use crate::{advance_prefix, coverage, indexed, ordered, prefix_coverage, retained};
#[cfg(verus_only)]
use vstd::prelude::*;

#[cfg_attr(verus_only, verus_verify)]
#[derive(Clone, Copy)]
pub(super) struct Run {
    pub start: usize,
    pub end: usize,
    /// Position of the retained entity in the sorted input.
    pub first: usize,
}

#[cfg_attr(verus_only, verus_spec(out =>
    requires ordered(spans@), indexed(spans@),
    ensures coverage(spans@) =~= coverage(out@), retained(spans@, spans.len() as int, out@),
))]
pub(super) fn union_sorted(mut spans: Vec<Run>) -> Vec<Run> {
    // The prefix contains the merged runs. The unread suffix stays intact.
    let mut used = 0;
    let mut i = 0;
    #[cfg(verus_only)]
    proof_decl! { let ghost input = spans@; }
    #[cfg_attr(verus_only, verus_spec(
       invariant
           used <= i, i <= input.len(), spans.len() == input.len(),
           forall |k:int| i <= k < input.len() ==> #[trigger] spans@[k] == input[k],
           prefix_coverage(input, i as int, spans@.take(used as int)),
       decreases spans.len() - i,
    ))]
    while i < spans.len() {
        let next = spans[i];
        #[cfg(verus_only)]
        proof_decl! { let ghost prior = spans@.take(used as int); }
        if used > 0 {
            let last = &mut spans[used - 1];
            if absorb(&mut last.end, next.start, next.end) {
                #[cfg(verus_only)]
                proof! {
                    let now = spans@.take(used as int);
                    assert(prior.drop_last() =~= now.drop_last());
                    advance_prefix(input, i as int, prior, now);
                }
                i += 1;
                continue;
            }
        }
        spans[used] = next;
        used += 1;
        #[cfg(verus_only)]
        proof! {
            let now = spans@.take(used as int);
            assert(now =~= prior.push(next));
            advance_prefix(input, i as int, prior, now);
        }
        i += 1;
    }
    spans.truncate(used);
    #[cfg(verus_only)]
    proof! { assert(input.take(i as int) =~= input); }
    spans
}

#[cfg_attr(verus_only, verus_spec(merged =>
    ensures
        merged == (next_start < *old(end)),
        *final(end) == if merged && next_end > *old(end) { next_end } else { *old(end) },
))]
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
