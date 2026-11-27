# Anno merge verification experiments

These experiments accompany the production checks in [Anno PR 406](https://github.com/arclabs561/anno/pull/406).

## Production traversal

Run `bash verification/pii/check_traversal.sh` from the repository root. It verifies the same in-place traversal used by redaction and the entity transforms. Callers sort the input by start, keep the longest span first for ties, deduplicate equal bounds and assign each remaining entity its position. Given nondecreasing starts and those source indices, the proof covers every representable input length. The output covers exactly the input character positions, keeps output intervals disjoint and ordered, and identifies strictly increasing source positions from which the caller retains metadata. Empty ranges are allowed. [Production Rust](../../../crates/anno/src/pii/span_union.rs) · [Verus specification](../traversal.rs)

The public Rust tests separately exercise sorting, Unicode replacement and metadata retention. The runner uses Verus 0.2026.11.22.1ce6a59 with Rust 1.100.0. On Linux x86-64 it can fetch the pinned verifier and check its archive digest. An existing installation can be supplied through `VERUS_BIN`.

## Verus prototype

Run `verus --crate-type lib first_run.rs` with Verus release 0.2026.11.22.1ce6a59 and its Rust 1.100.0 toolchain. The prototype proves exact character membership in the first merged run for any nonempty input length, assuming the first start is a lower bound for the other starts. Removing the endpoint extension fails the assertion in the loop. This prototype preceded the shared production traversal.

## Aeneas

Charon nightly-2026.11.26 and Aeneas nightly-2026.11.26-69280e0 generated `SpanUnion.lean` from the production Rust helper in commit f3f88bf6. From Anno's repository root, with the Charon Rust nightly-2026-11-25 compiler components available, run:

```sh
charon rustc --preset=aeneas --sysroot default --start-from crate::absorb --dest /tmp -- --crate-type lib crates/anno/src/pii/span_union.rs
aeneas -backend lean -dest /tmp /tmp/span_union.llbc
```

The generated result contains both the return value and the updated endpoint from the Rust mutable borrow. This experiment verified extraction only. The release's Lean backend targets 4.36.0 and Mathlib, whereas Anno's existing standalone proof uses Lean 4.19.0. Kani checks the production merge step and Verus checks the complete traversal.
