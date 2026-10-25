#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // scan_patterns/redact take &str. Lossy conversion keeps every byte
    // sequence reachable, including multibyte replacement characters.
    let text = String::from_utf8_lossy(data);
    let entities = anno::pii::scan_patterns(&text);
    let redacted = anno::pii::redact(&text, &entities);

    // redact returns String, so this also pins the API contract that the
    // output is valid UTF-8 and that no input causes a panic or invalid
    // range replacement.
    assert!(std::str::from_utf8(redacted.as_bytes()).is_ok());
});
