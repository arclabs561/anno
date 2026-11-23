#![no_main]
use libfuzzer_sys::fuzz_target;

#[path = "../../crates/anno/tests/support/reviewed_kb_fuzz.rs"]
mod harness;

fuzz_target!(|bytes: &[u8]| {
    let _ = harness::replay(bytes);
});
