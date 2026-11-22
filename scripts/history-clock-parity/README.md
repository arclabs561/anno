# Reading older evaluation histories

A history can contain both `2016-12-31T23:59:59Z` and `2016-12-31T23:59:60Z`. A range containing only the leap second should return the second row. Chrono and the Anno reader give it its own ordering key. Parsing both rows directly as Jiff timestamps gives them the same key.

Run the example from the repository root.

```sh
cargo run --quiet --locked --manifest-path scripts/history-clock-parity/Cargo.toml
cargo test --locked --manifest-path scripts/history-clock-parity/Cargo.toml
```

To check an existing JSONL history, pipe one timestamp per line. Each output row contains the input, the Chrono key, the Jiff key, and the Anno key. A key has the form `unix_seconds/nanoseconds`. The word `legacy` identifies a string that a parser rejected.

```sh
jq -r '.timestamp' reports/eval-results.jsonl |
  cargo run --quiet --locked --manifest-path scripts/history-clock-parity/Cargo.toml -- --stdin
```

The tab-separated fixtures record observed results for both published parsers. A second test compares the production reader with Chrono on 20,160 combinations of dates, separators, times, fractions, and offsets. The comparison tool has its own workspace and lockfile.
