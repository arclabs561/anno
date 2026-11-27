use vstd::prelude::*;

verus! {

// End of the first run of overlapping spans, as a proof experiment.
// Prove exact membership for any queried position and any input length.
fn first_run(spans: &[(usize, usize)], pos: usize) -> (out: (usize, usize))
    requires
        spans.len() > 0,
        forall|j: int| 0 <= j < spans.len() ==> spans[0].0 <= #[trigger] spans[j].0,
    ensures
        1 <= out.0 <= spans.len(),
        (spans[0].0 <= pos < out.1) <==>
            exists|j: int| 0 <= j < out.0 && #[trigger] spans[j].0 <= pos < spans[j].1,
        out.0 < spans.len() ==> spans[out.0 as int].0 >= out.1,
{
    let mut i = 1usize;
    let mut end = spans[0].1;
    while i < spans.len()
        invariant
            1 <= i <= spans.len(),
            forall|j: int| 0 <= j < spans.len() ==> spans[0].0 <= #[trigger] spans[j].0,
            (spans[0].0 <= pos < end) <==>
                exists|j: int| 0 <= j < i && #[trigger] spans[j].0 <= pos < spans[j].1,
        decreases spans.len() - i,
    {
        let (start, next_end) = spans[i];
        if start >= end { return (i, end); }
        if next_end > end { end = next_end; }
        assert((spans[0].0 <= pos < end) <==>
            exists|j: int| 0 <= j <= i && #[trigger] spans[j].0 <= pos < spans[j].1);
        i += 1;
    }
    (i, end)
}
}
