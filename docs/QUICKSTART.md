# Quickstart

Extract entities with the CLI or a reusable Rust model. The examples below
start with offline heuristics; model downloads and evaluation are optional.

## Install

As a Rust library:

```toml
[dependencies]
anno = "0.14"
```

Install the CLI:

```bash
cargo install anno-cli
```

Add `--features eval` if you need evaluation or cross-document clustering.
To build from a local clone with those features:

```bash
cargo install --path crates/anno-cli --bin anno --features "onnx eval"
```

## CLI: extract entities

```console
$ ANNO_NO_DOWNLOADS=1 anno extract --model heuristic --text "Lynn Conway worked at IBM and Xerox PARC in California."
PER:1 "Lynn Conway"
ORG:2 "IBM" "Xerox PARC"
LOC:1 "California"
```

Pattern-only extraction (offline, no weights):

```bash
anno extract --model pattern --text "Email jobs@acme.com by 2024-01-15 for \$100."
```

Extract entity types that you supply (requires the default `onnx` feature and
model weights):

```bash
anno extract --model gliner --extract-types "DRUG,SYMPTOM" \
  --text "Aspirin can treat headaches and reduce fever."
```

Machine-readable output:

```bash
anno extract --model heuristic --format json --text "Lynn Conway worked at IBM and Xerox PARC in California."
```

## CLI: relation extraction

Extract typed `(head, relation, tail)` triples with a `RelationCapable` backend:

```bash
# tplinker: heuristic baseline, no extra deps
anno extract --extract-relations --model tplinker \
  --text "Steve Jobs founded Apple in 1976."

# gliner_multitask: zero-shot entity types + heuristic relations (requires onnx)
anno extract --extract-relations --model gliner-multitask \
  --relation-types "FOUNDED,WORKS_FOR" \
  --text "Steve Jobs founded Apple in 1976."
```

## CLI: coreference

Link pronouns to their referents:

```bash
anno debug --coref -t "Sophie Wilson designed the ARM processor. She revolutionized computing."
```

## CLI: batch processing

Process a directory of `.txt` or `.md` files:

```bash
# Sequential, write per-document JSON to results/
anno batch --model heuristic --dir docs/ --output results/

# 4 parallel workers; reuse eligible cached results
anno batch --model heuristic --dir docs/ --parallel 4 --cache --output results/

# Stream JSONL from stdin: {"id":"…","text":"…"} per line
cat corpus.jsonl | anno batch --model heuristic --stdin --parallel 4 --cache --output results/
```

Only deterministic backends use the result cache: `pattern`, `heuristic`,
`minimal`, and `heuristic-fr`. Runs using model weights, remote services, or
coreference bypass it. Entries live under `{cache_dir}/results/v2/`; the cache
key includes the document, CLI build, and extraction settings.
`anno cache clear --confirm` deletes the result cache.

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

Offsets count characters (Unicode scalar values), not UTF-8 bytes. See the
[span and offset contract](CONTRACT.md).

## Next

- [Contract](CONTRACT.md): scope, offset guarantees, feature flags
- [Backends](BACKENDS.md): model selection and hardware requirements
