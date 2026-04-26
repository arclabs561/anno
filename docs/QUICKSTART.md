# Quickstart

Practical starting point for using `anno` via the CLI and as a Rust library.

## Install

As a Rust library:

```toml
[dependencies]
anno = "0.8"
```

Full CLI (recommended):

```bash
cargo install --git https://github.com/arclabs561/anno \
  --package anno-cli --bin anno --features "onnx eval"
```

From a local clone:

```bash
cargo install --path crates/anno-cli --bin anno --features "onnx eval"
```

## CLI: extract entities

```bash
anno extract --text "Lynn Conway worked at IBM and Xerox PARC in California."
```

Pattern-only extraction (offline, no weights):

```bash
anno extract --model pattern --text "Email jobs@acme.com by 2024-01-15 for \$100."
```

Zero-shot custom entity types (requires `--features onnx`):

```bash
anno extract --model gliner --extract-types "DRUG,SYMPTOM" \
  --text "Aspirin can treat headaches and reduce fever."
```

Machine-readable output:

```bash
anno extract --format json --text "Lynn Conway worked at IBM and Xerox PARC in California."
```

## CLI: relation extraction

Extract typed `(head, relation, tail)` triples with a `RelationCapable` backend:

```bash
# tplinker: heuristic baseline, no extra deps
anno extract --extract-relations --model tplinker \
  --text "Steve Jobs founded Apple in 1976."

# gliner_multitask: zero-shot entity types + heuristic relations (requires onnx)
anno extract --extract-relations --model gliner_multitask \
  --relation-types "FOUNDED,WORKS_FOR" \
  --text "Steve Jobs founded Apple in 1976."
```

## CLI: coreference

Link pronouns to their referents:

```bash
anno debug --coref -t "Sophie Wilson designed the ARM processor. She revolutionized computing."
```

## CLI: batch processing

Process a directory of `.txt`/`.md` files with optional parallelism and result caching:

```bash
# Sequential, write per-document JSON to results/
anno batch --dir docs/ --output results/

# 4 parallel workers; cache results by text hash + model version
anno batch --dir docs/ --parallel 4 --cache --output results/

# Stream JSONL from stdin: {"id":"…","text":"…"} per line
cat corpus.jsonl | anno batch --stdin --parallel 4 --cache --output results/
```

Cache entries live in `{cache_dir}/results/` keyed by `xxh3(text) + model + version`.
Flush with `anno cache clear`.

## CLI: cross-document clustering

Cluster entities across multiple documents (requires `--features eval`):

```bash
anno cross-doc ./docs --threshold 0.6 --format tree
```

## Library usage

```rust
use anno::{Model, StackedNER};

let ner = StackedNER::default();
let entities = ner.extract_entities("Lynn Conway worked at IBM and Xerox PARC.", None)?;

for e in entities {
    println!("{} [{}..{}] {:?}", e.text, e.start(), e.end(), e.entity_type);
}
```

## Offsets

Offsets are **character offsets** (Unicode scalar values), not byte offsets. See `docs/CONTRACT.md`.

## Next

- `docs/CONTRACT.md` — scope, offset guarantees, feature gating
- `docs/BACKENDS.md` — backend selection and feature flags
