# Backends

This page is intentionally minimal: it avoids benchmark numbers and “working set” claims that drift.

## Choose by constraints

- **No ML deps**: build with `default-features = false` and use `pattern`, `heuristic`, or `stacked`.
- **Zero-shot custom types**: use `--model gliner --extract-types "TYPE1,TYPE2"` (requires `onnx`).
- **Pure Rust inference**: use Candle backends (requires `candle`).

## Source of truth (generated at runtime)

Use the CLI to see what’s available in *your build*:

```bash
anno backends
anno models list
anno models recommend
```

## Measuring performance (generated artifacts)

Run your own benchmark/eval and keep the results as artifacts, not prose:

```bash
anno eval --help

# `benchmark` is feature-gated (build `anno-cli` with `--features eval-advanced`)
anno benchmark --help
```

If these commands write reports, treat the output directory (commonly `reports/`) as the source of truth for results.

## See also

- [Contract](CONTRACT.md) — scope + guarantees

