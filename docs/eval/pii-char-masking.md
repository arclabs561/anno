# PII character-masking evaluation

> **Artifact kind: `synthetic-fixture`.** This is **not a benchmark**. No neural/ML model backends were exercised; only deterministic pattern scanning and the heuristic NER layer were used. The fixture is small and hand-assembled, so the numbers describe this fixture, not production accuracy.

| Field | Value |
|---|---|
| Metric | `char-masking` |
| Command | `cargo run -p anno-eval --bin pii_char_eval -- --metric char-masking --fixture crates/anno-eval/fixtures/pii_masking_fixture.jsonl --out-json docs/eval/pii-char-masking.json --out-md docs/eval/pii-char-masking.md --bootstrap` |
| Command note | Reproduction command; host-local CARGO_TARGET_DIR cache placement is omitted. |
| Commit | `fd28a2fe96f9c9847b2c80a75467b17f5f3e7b94` |
| Generated (UTC) | 2026-10-25T17:14:59Z |
| Fixture | `crates/anno-eval/fixtures/pii_masking_fixture.jsonl` |
| Fixture SHA-256 | `55ec74ec71bbb4ab5d282226e1866dd01818356044dce78c473fa324828a06b0` |
| Documents | 8 |
| Neural/ML model backends | not exercised |
| Backend scope | deterministic pattern + heuristic layers only; no neural/ML backends |
| Deterministic paths | `pattern (regex PII scan)`, `heuristic NER (StackedNER)` |
| Public dataset | `ai4privacy/pii-masking-200k` |
| Public split | train (the registry declares only train; no dev/test split is declared) |

## Headline metrics

| Metric | Value |
|---|---:|
| Character precision | 98.61% |
| Character recall | 93.11% |
| Character F1 | 95.78% |
| Character F2 (beta=2) | 94.16% |
| False-positive rate (non-PII) | 1.36% |
| Strict span F1 (exact boundary) | 69.77% |

Counts: `TP=284` masked PII characters, `masked=288` predicted characters, `gold=305` PII characters, `non_pii=294` non-PII characters, `total=599` characters.

F2 is below F1 here because pooled precision exceeds pooled recall; F2 weights the recall shortfall more heavily. That is the intended bias for redaction: missing PII characters is worse than a small over-mask.

## Per-entity-type breakdown

| Type | Gold chars | Predicted chars | TP | Precision | Recall | F1 | F2 | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ADDRESS | 38 | 38 | 38 | 100.00% | 100.00% | 100.00% | 100.00% | 0.00% |
| DOB | 10 | 10 | 10 | 100.00% | 100.00% | 100.00% | 100.00% | 0.00% |
| EMAIL | 68 | 68 | 68 | 100.00% | 100.00% | 100.00% | 100.00% | 0.00% |
| ID_NUMBER | 81 | 62 | 62 | 100.00% | 76.54% | 86.71% | 80.31% | 0.00% |
| PER | 76 | 78 | 74 | 94.87% | 97.37% | 96.10% | 96.86% | 1.36% |
| PHONE | 32 | 32 | 32 | 100.00% | 100.00% | 100.00% | 100.00% | 0.00% |

## Documents

| Document | Source | Chars | Gold spans | Predicted spans | Recall | FPR |
|---|---|---:|---:|---:|---:|---:|
| repo-doc1 | repo:testdata/fixtures/cross_doc/doc1.txt | 73 | 1 | 1 | 100.00% | 0.00% |
| repo-doc2 | repo:testdata/fixtures/cross_doc/doc2.txt | 55 | 1 | 1 | 100.00% | 0.00% |
| repo-doc3 | repo:testdata/fixtures/cross_doc/doc3.txt | 84 | 2 | 2 | 100.00% | 0.00% |
| synthetic-name-contact | synthetic | 79 | 3 | 5 | 100.00% | 0.00% |
| synthetic-identifiers | synthetic | 71 | 3 | 4 | 100.00% | 0.00% |
| synthetic-utf8 | synthetic (multi-byte UTF-8) | 52 | 3 | 4 | 72.22% | 0.00% |
| synthetic-dob-address | synthetic | 75 | 2 | 2 | 100.00% | 0.00% |
| synthetic-mixed | synthetic | 110 | 4 | 5 | 83.58% | 9.30% |

## Metric definitions

- **Precision**: correctly masked characters / all masked characters
- **Recall**: correctly masked characters / all gold PII characters
- **F1**: harmonic mean of character precision and recall
- **F2**: 5PR / (4P + R); recall-weighted, beta = 2
- **FPR**: over-masked non-PII characters / all non-PII characters
- **Strict span F1**: exact-boundary span F1 (SemEval exact schema; type ignored)
- **Overlap resolution**: overlapping predicted spans are merged into the union of masked characters; the same union is applied to gold spans
- **Per-type breakdown**: each predicted span is attributed to the gold type with the largest character overlap; unattributed over-masks appear only in the overall FPR

## What this artifact does not establish

- It is not a result on `ai4privacy/pii-masking-200k` or any other public PII benchmark split. The dataset registry declares only the `train` split for `ai4privacy/pii-masking-200k`, so an outsider-credible artifact also needs a documented held-out carve-out.
- It does not exercise neural/ML backends (ONNX, Candle, GLiNER) or trained classical backends (CRF, HMM).
- The fixture is too small and too synthetic to support an accuracy claim about production redaction.
- Per-type rows are diagnostic, not additive: a predicted span is attributed to one gold type.
