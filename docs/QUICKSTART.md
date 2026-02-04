# Quickstart

This is a short, practical starting point for using `anno` via the CLI and as a Rust library.

## Install

As a Rust library (from Git, recommended for now):

```toml
[dependencies]
anno = { git = "https://github.com/arclabs561/anno", rev = "<commit>" }
```

Crates.io publishing is intentionally paused for now; see `docs/PUBLISH_STATUS.md` and `docs/TAGS.md`.

CLI (from source):

```bash
git clone https://github.com/arclabs561/anno
cd anno

# Minimal build (no ML backends):
cargo build --release -p anno-cli --bin anno

# Recommended (ONNX ML backends + zero-shot):
cargo build --release -p anno-cli --bin anno --features "onnx eval-advanced"
```

## CLI: extract entities

```bash
anno extract --text "Lynn Conway worked at IBM and Xerox PARC in California."
```

Pattern-only extraction (emails, dates, money):

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

## CLI: coreference

Link pronouns to their referents:

```bash
anno debug --coref -t "Sophie Wilson designed the ARM processor. She revolutionized computing."
```

## CLI: cross-document clustering

Cluster entities across multiple documents (requires `--features eval-advanced`):

```bash
anno cross-doc ./docs --threshold 0.6 --format tree
```

## Library usage

```rust
use anno::{Model, StackedNER};

let ner = StackedNER::default();
let entities = ner.extract_entities("Lynn Conway worked at IBM and Xerox PARC.", None)?;

for e in entities {
    println!("{} [{}..{}] {:?}", e.text, e.start, e.end, e.entity_type);
}
```

## Offsets

Offsets are **character offsets** (Unicode scalar values), not byte offsets. See `docs/CONTRACT.md`.

## Next

- `docs/CONTRACT.md` — scope + guarantees
- `docs/BACKENDS.md` — backend selection and feature flags
