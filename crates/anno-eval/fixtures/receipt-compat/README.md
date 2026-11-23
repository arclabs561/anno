# Receipt compatibility fixtures

Both the Rust evaluator tests and the fixed-panel Python tests read these files. The archived rows are the first result object in each source report, copied from its opening brace through its closing brace with the original whitespace. The source report stays read-only during replay. Each case in `cases.json` applies JSON Pointer replacements to a fresh in-memory copy.

| Fixture | Source |
| --- | --- |
| `v0.json` | The pre-provenance 0.12 serializer at [1e367551](https://github.com/arclabs561/anno/blob/1e3675514dc4ad6cd93249ca87dc6761312d1a18/crates/anno-eval/src/eval/task_evaluator.rs#L4516), replayed locally with its original `make_test_result(false, Some("Model failed to load"), None)` test input and `serde_json::to_vec_pretty`. Its missing provenance field identifies the historical format. |
| `v1.json` | First result in `eval-sanity-report.json` from [run 45052490350, artifact 14770904202](https://github.com/arclabs561/anno/actions/runs/45052490350/artifacts/14770904202). |
| `v2.json` | First result in `eval-sanity-report.json` from [run 45099223146, artifact 14785479396](https://github.com/arclabs561/anno/actions/runs/45099223146/artifacts/14785479396). |

The two archived WikiGold successes contain real scores and were recorded with `cached_only=false`. They exercise historical receipt classification. A fresh fixed panel separately requires cached-only provenance. Controlled variants retain the original metrics even when changed to a failure or skip, to check that those values cannot become measured scores.
