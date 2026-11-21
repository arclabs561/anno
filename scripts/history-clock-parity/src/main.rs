use chrono::DateTime;
use jiff::Timestamp;
#[path = "../../../crates/anno-eval/src/eval/history_clock.rs"]
mod history_clock;
use history_clock::parse as candidate_key;
use std::io::{self, BufRead};

/// Chrono stores a leap second as the preceding Unix second and an extra
/// billion nanoseconds. Keeping the fields separate preserves its ordering.
type Key = (i64, u32);

fn chrono_key(text: &str) -> Option<Key> {
    let instant = DateTime::parse_from_rfc3339(text).ok()?;
    Some((instant.timestamp(), instant.timestamp_subsec_nanos()))
}

fn jiff_key(text: &str) -> Option<Key> {
    let nanos = text.parse::<Timestamp>().ok()?.as_nanosecond();
    Some((
        nanos.div_euclid(1_000_000_000) as i64,
        nanos.rem_euclid(1_000_000_000) as u32,
    ))
}

fn display(key: Option<Key>) -> String {
    match key {
        Some((second, nanos)) => format!("{second}/{nanos}"),
        None => "legacy".to_owned(),
    }
}

fn example() {
    let mut differences = 0;
    let mut total = 0;
    for row in include_str!("../cases.tsv").lines().skip(1) {
        let columns: Vec<_> = row.split('\t').collect();
        let [text, old, new] = columns.as_slice() else {
            panic!("expected three tab-separated fields: {row}");
        };
        assert_eq!(display(chrono_key(text)), *old, "chrono: {text}");
        assert_eq!(display(candidate_key(text)), *old, "candidate: {text}");
        assert_eq!(display(jiff_key(text)), *new, "jiff: {text}");
        if old != new {
            println!("{row}");
            differences += 1;
        }
        total += 1;
    }
    println!("{differences} of {total} fixtures differ");
}

fn main() -> io::Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.is_empty() {
        example();
    } else if args == ["--stdin"] {
        for line in io::stdin().lock().lines() {
            let line = line?;
            println!(
                "{line}\t{}\t{}\t{}",
                display(chrono_key(&line)),
                display(jiff_key(&line)),
                display(candidate_key(&line))
            );
        }
    } else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "usage: history-clock-parity [--stdin]",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parsers_match_recorded_corpus() {
        example();
    }

    #[test]
    fn candidate_matches_historical_keys() {
        for date in [
            "0000-01-01",
            "0001-12-31",
            "1969-12-31",
            "1900-02-29",
            "2000-02-29",
            "2016-12-31",
            "2026-11-20",
            "9999-12-31",
            "2026-00-01",
            "2026-12-32",
        ] {
            for sep in ["T", "t", " ", "_"] {
                for clock in [
                    "00:00:00", "12:30:60", "23:59:59", "23:59:60", "24:00:00", "00:60:00",
                    "00:00:61",
                ] {
                    for frac in ["", ".", ".0", ".123400", ".123456789123", ",12"] {
                        for offset in [
                            "Z",
                            "z",
                            "-00:00",
                            "+05:45",
                            "−05:00",
                            "+23:59",
                            "-23:59",
                            "+24:00",
                            "+00:60",
                            "+0530",
                            "+00:00[UTC]",
                            "",
                        ] {
                            let text = format!("{date}{sep}{clock}{frac}{offset}");
                            assert_eq!(chrono_key(&text), candidate_key(&text), "{text}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn reader_results_change_at_the_leap_second() {
        let rows = [
            "2016-12-31T23:59:59Z",
            "2016-12-31T23:59:60Z",
            "2017-01-01T00:00:00Z",
        ];
        for (parse, latest, at_leap) in [
            (chrono_key as fn(&str) -> Option<Key>, [2, 1], vec![1]),
            (candidate_key, [2, 1], vec![1]),
            (jiff_key, [2, 0], vec![0, 1]),
        ] {
            let mut order = [0, 1, 2];
            order.sort_by_cached_key(|&i| std::cmp::Reverse(parse(rows[i]).unwrap()));
            assert_eq!(order[..2], latest);
            let matches: Vec<_> = (0..rows.len())
                .filter(|&i| parse(rows[i]) == parse(rows[1]))
                .collect();
            assert_eq!(matches, at_leap);
        }
    }
}
