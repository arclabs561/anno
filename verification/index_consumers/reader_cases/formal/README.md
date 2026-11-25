# Two independent issuers correcting the same local assertion ID

The Rust example and SQLite reader both accept a historical content edition and a separately selected policy. This check asks what each reader should return when two issuers sharing a source both call their assertion `7`. Withdrawing one assertion must preserve a permitted claim from the other issuer, including when both named the same target.

Run from the repository root with Python 3, Cargo and Z3 on PATH.

```sh
python3 verification/index_consumers/reader_cases/formal/check_support.py
```

Supply `--anno PATH --rust-reader PATH` to run prebuilt binaries. The runner generates project-authored fixtures locally and uses the existing verification readers.

## What the solver and readers establish

A support belongs to a particular issuer and can be absent, name X, or name Y. Z3 rules out three violations for all values of this finite domain. Updates made by different issuers commute, removing one issuer's claim preserves the other's permitted target, and changing a hidden claim leaves public targets unchanged. A separate satisfiable query confirms a correction can introduce a genuine conflict.

| Deliberately broken rule | Satisfiable counterexample |
|---|---|
| Withdraw by local ID across issuers | Both issuers name X. Withdrawing A must leave B's X visible. |
| Reuse the historical permission | Only A named X at the historical edition. A is now hidden. Reusing the older permission reveals X. |
| Always show the newest factual edition | A corrected X to Y. Reading the historical edition with permission still in force must return X. |
| Select the first eligible target | A names X and B names Y. A form requiring exactly one target would present X as settled. |

The runner asks Z3 for the permitted target set and single-slot decision on every combination of two claims and two visibility bits. It then feeds both executable readers all 81 ordered pairs of factual states as two-edition histories. At both editions it tests four policy masks, both audiences, and three issuer scopes. Each of the 3,888 queries per reader must return exactly the solver's target set. A separate application decision selects a target only when the reader returns exactly one. Multiple targets remain available for review.

## Proof boundary

The SMT result covers this finite support model. The reader comparisons exercise the concrete Rust example and SQLite index over generated histories. The caller supplies the authenticated audience and applicable policy. Citation text, excerpt permissions, cache propagation, open-world identification, legitimate multi-author credit and longer event histories need their respective checks. The single-slot decision applies to a form that explicitly requires one target. An empty result leaves the underlying real-world question open.
