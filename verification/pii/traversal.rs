// SPDX-License-Identifier: MIT OR Apache-2.0
// This harness compiles the production loop. Its preconditions are established
// by sorting the entities by start and enumerating the resulting sequence.
#![feature(proc_macro_hygiene)]

#[path = "../../crates/anno/src/pii/span_union.rs"]
mod span_union;

use span_union::Run;
use vstd::iset::ISet;
use vstd::prelude::*;

verus! {

// Calling the real function also makes verification fail if its Verus
// annotation disappears and the verifier can no longer use its contract.
fn check_production_traversal(input: Vec<Run>) -> (out: Vec<Run>)
    requires ordered(input@), indexed(input@),
    ensures
        coverage(input@) =~= coverage(out@),
        retained(input@, input.len() as int, out@),
{
    span_union::union_sorted(input)
}

spec fn ordered(input: Seq<Run>) -> bool {
    forall|i: int, j: int| #![trigger input[i], input[j]]
        0 <= i <= j < input.len() ==> input[i].start <= input[j].start
}

spec fn indexed(input: Seq<Run>) -> bool {
    forall|i: int| 0 <= i < input.len() ==> #[trigger] input[i].first == i
}

// Each emitted range retains an original source for its type and text.
// Source positions increase and the output intervals do not intersect.
spec fn retained(input: Seq<Run>, limit: int, out: Seq<Run>) -> bool {
    &&& forall|j: int| 0 <= j < out.len() ==> {
        let run = #[trigger] out[j];
        &&& j <= run.first < limit
        &&& input[run.first as int].start == run.start
        &&& input[run.first as int].end <= run.end
    }
    &&& forall|j: int, k: int| #![trigger out[j], out[k]]
        0 <= j < k < out.len() ==>
            out[j].first < out[k].first && out[j].end <= out[k].start
}

// The specification is a union of sets of positions. It does not merge ranges.
spec fn coverage(runs: Seq<Run>) -> ISet<int>
    decreases runs.len(),
{
    if runs.len() == 0 {
        ISet::empty()
    } else {
        coverage(runs.drop_last()).union(
            ISet::new(|p: int| runs.last().start <= p < runs.last().end)
        )
    }
}

spec fn prefix_coverage(input: Seq<Run>, i: int, out: Seq<Run>) -> bool {
    &&& 0 <= i <= input.len()
    &&& ordered(input) && indexed(input)
    &&& out.len() <= i
    &&& retained(input, i, out)
    &&& forall|k: int| i <= k < input.len() && out.len() > 0 ==>
        out.last().start <= #[trigger] input[k].start
    &&& coverage(out) =~= coverage(input.take(i))
}

// Extend the invariant after either appending a new run or absorbing an item
// into the last run. The caller proves which update the actual Rust code made.
proof fn advance_prefix(input: Seq<Run>, i: int, prior: Seq<Run>, out: Seq<Run>)
    requires
        prefix_coverage(input, i, prior),
        i < input.len(),
        if prior.len() > 0 && input[i].start < prior.last().end {
            &&& out.len() == prior.len()
            &&& out.drop_last() =~= prior.drop_last()
            &&& out.last().first == prior.last().first
            &&& out.last().start == prior.last().start
            &&& out.last().end == if input[i].end > prior.last().end {
                input[i].end
            } else {
                prior.last().end
            }
        } else {
            out =~= prior.push(input[i])
        },
    ensures prefix_coverage(input, i + 1, out),
{
    assert(input.take(i + 1).drop_last() =~= input.take(i));
    if prior.len() == 0 || input[i].start >= prior.last().end {
        assert(out.drop_last() =~= prior);
    }

    assert forall|j: int| 0 <= j < out.len() implies {
        let run = #[trigger] out[j];
        &&& j <= run.first < i + 1
        &&& input[run.first as int].start == run.start
        &&& input[run.first as int].end <= run.end
    } by {
        if j < out.len() - 1 {
            assert(out.drop_last()[j] == prior[j]);
        } else if prior.len() == out.len() {
            assert(prior.last() == prior[prior.len() - 1]);
        } else {
            assert(out[j] == input[i]);
        }
    }

    assert forall|j: int, k: int| #![trigger out[j], out[k]]
        0 <= j < k < out.len() implies
            out[j].first < out[k].first && out[j].end <= out[k].start by {
        assert(out.drop_last()[j] == prior[j]);
        if k < out.len() - 1 {
            assert(out.drop_last()[k] == prior[k]);
        } else if out.len() == prior.len() {
            assert(prior.last() == prior[k]);
        } else {
            assert(out[k] == input[i]);
            if j < k - 1 {
                assert(prior[j].end <= prior[k - 1].start);
            }
        }
    }

    reveal_with_fuel(coverage, 2);
    assert(coverage(out) =~= coverage(input.take(i + 1)));
}

}
