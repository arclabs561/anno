# Anno merge verification experiments

These experiments accompany the production checks in [Anno PR 406](https://github.com/arclabs561/anno/pull/406).

## Verus

Run `verus --crate-type lib first_run.rs` with Verus release 0.2026.11.22.1ce6a59 and its Rust 1.100.0 toolchain. The prototype proves exact character membership in the first merged run for any nonempty input length, assuming the first start is a lower bound for the other starts. Removing the endpoint extension fails the assertion in the loop. This prototype is separate from Anno's production loops.

## Aeneas

Charon nightly-2026.11.26 and Aeneas nightly-2026.11.26-69280e0 generated `SpanUnion.lean` from the production Rust helper in commit f3f88bf6. From Anno's repository root, with the Charon Rust nightly-2026-11-25 compiler components available, run:

```sh
charon rustc --preset=aeneas --sysroot default --start-from crate::absorb --dest /tmp -- --crate-type lib crates/anno/src/pii/span_union.rs
aeneas -backend lean -dest /tmp /tmp/span_union.llbc
```

The generated result contains both the return value and the updated endpoint from the Rust mutable borrow. This experiment verified extraction only. The release's Lean backend targets 4.36.0 and Mathlib, whereas Anno's existing standalone proof uses Lean 4.19.0. The PR keeps the Kani proof on production code as its CI check.
