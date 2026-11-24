# Replay the catalog publication pilot

The required index suite runs the small historical withdrawal and three checks on committee publication order offline. Run those checks from the anno repository root with Cargo and Python 3.10 or newer:

```sh
pilot=verification/index_consumers/congress
python3 "$pilot/test_committee_pilot.py"
python3 "$pilot/check_catalog_history.py"
```

The larger replay uses pinned upstream data on demand. It needs Git, Python, PyYAML, and a clone of the congressional directory with its history available. From the repository root:

```sh
pilot=verification/index_consumers/congress
scratch=$(mktemp -d)
git clone --bare https://github.com/unitedstates/congress-legislators.git "$scratch/legislators"
python3 "$pilot/extract_roster.py" "$scratch/legislators" "$scratch/roster.jsonl"
python3 "$pilot/committee_observations.py" "$scratch/legislators" "$scratch/committee.jsonl"
python3 "$pilot/committee_pilot.py" "$scratch/committee.jsonl" "$scratch/roster.jsonl" \
  --time-search --pairs "$scratch/pairs.tsv" > "$scratch/policy.json"
cargo run --quiet --locked -p anno --no-default-features --example congress_name_scores < "$scratch/pairs.tsv"
python3 "$pilot/check_rollcalls.py" "$scratch/legislators" --output-dir "$scratch/house"
cargo build --release --locked -p anno-cli --no-default-features --features extractor-readability
python3 "$pilot/check_catalog_history.py" --anno target/release/anno --roll-calls "$scratch/house"
```

For a custom Cargo target directory, pass the resulting binary's path to `--anno`. The two Clerk files are fetched from their publisher, verified by SHA-256 and projected to printed member names. To run offline, pass a local XML cache as the second positional argument to `check_rollcalls.py`. The timing includes starting the CLI on ready grounded input, validating and projecting that input, transferring JSON, committing postings and a checkpoint in SQLite, and checking a lookup through a second connection. Each trial uses a fresh database. Time to obtain the source or construct the catalog assignments is measured separately from this local path in the historical replay.

| Material | Source, reuse and receipt |
|---|---|
| Congressional directory and committee editions | Community-maintained data dedicated under CC0 1.0. The scripts pin both ends of the first-parent interval, select only fields used in the experiment and retain the source revisions. The generated JSONL stays in the caller's working directory. [Source license](https://github.com/unitedstates/congress-legislators/blob/82b1eaccd2cad9e4172059507f5129fc513962af/LICENSE) · [Expected checksums and counts](committee-findings.md) |
| Historical correction fixture | The checked fixture contains attributed CC0 structured fields from Wikidata and the congressional project, their pinned versions, links to terms, and explicitly identified pilot decisions. [Source manifest](attribution-correction.json) · [Wikidata structured data terms](https://www.wikidata.org/wiki/Wikidata:Copyright) |
| Official House roll calls | Government source files remain at the Clerk. Only aggregates, receipts and independently authored projection code are distributed here. See the source-work and jurisdiction notes in the receipt. [Clerk sources and receipts](rollcall-receipt.md) |
| Independent collector | The optional harness invokes upstream code in a separate local checkout against independently verified input. Its results describe that program's fallback when a supplied ID is withheld. [Setup and results](collector-study.md) |

The project-authored harnesses use `MIT OR Apache-2.0`. The local tests need no model weights. [Design decision and limitations](../../../docs/decisions/catalog-publication-pilot.md)
