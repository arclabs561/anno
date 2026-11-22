use jiff::{tz::Offset, Timestamp};

pub(super) fn now() -> String {
    format_timestamp(Timestamp::now())
}

fn format_timestamp(timestamp: Timestamp) -> String {
    // Existing histories use +00:00 and groups of three fractional digits.
    let nanos = timestamp.subsec_nanosecond();
    let precision = if nanos == 0 {
        0
    } else if nanos % 1_000_000 == 0 {
        3
    } else if nanos % 1_000 == 0 {
        6
    } else {
        9
    };
    format!("{:.precision$}", timestamp.display_with_offset(Offset::UTC))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_stored_timestamps() {
        for stored in [
            "1969-12-31T23:59:59.999999999+00:00",
            "1970-01-01T00:00:00+00:00",
            "2026-03-08T07:00:00.120+00:00",
            "2026-03-08T07:00:00.123400+00:00",
            "2026-03-08T07:00:00.123456780+00:00",
        ] {
            let timestamp = stored.parse().unwrap();
            assert_eq!(format_timestamp(timestamp), stored);
        }
    }
}
