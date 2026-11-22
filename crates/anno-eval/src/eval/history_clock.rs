use std::borrow::Cow;

/// Unix seconds and subsecond nanoseconds. Leap seconds retain the legacy
/// representation with nanoseconds in [1_000_000_000, 2_000_000_000).
pub(super) type Key = (i64, u32);

/// Match the historical reader while delegating calendar and offset validation to time.
pub(super) fn parse(text: &str) -> Option<Key> {
    if !matches!(text.as_bytes().get(10), Some(b'T' | b't' | b' ')) {
        return None;
    }
    let mut normalized = Cow::Borrowed(text);
    // Chrono allows Unicode MINUS SIGN instead of '-' in the numeric offset.
    if let Some((prefix, offset)) = text.rsplit_once('−') {
        if prefix.len() >= 19 && offset.len() == 5 {
            normalized = Cow::Owned(format!("{prefix}-{offset}"));
        }
    }
    // Chrono accepts second 60 even when historical leap tables have no entry.
    let leap = normalized.as_bytes().get(17..19) == Some(b"60");
    if leap {
        normalized.to_mut().replace_range(17..19, "59");
    }
    let instant =
        time::OffsetDateTime::parse(&normalized, &time::format_description::well_known::Rfc3339)
            .ok()?;
    Some((
        instant.unix_timestamp(),
        instant.nanosecond() + if leap { 1_000_000_000 } else { 0 },
    ))
}
