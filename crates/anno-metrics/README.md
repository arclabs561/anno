# anno-metrics

[![crates.io](https://img.shields.io/crates/v/anno-metrics.svg)](https://crates.io/crates/anno-metrics)
[![Documentation](https://docs.rs/anno-metrics/badge.svg)](https://docs.rs/anno-metrics)
[![CI](https://github.com/arclabs561/anno/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/anno/actions/workflows/ci.yml)

Shared evaluation and analysis primitives for [anno](https://crates.io/crates/anno-lib): coreference scoring metrics and cluster encoders.

Depends only on `anno-core` and `serde`, so it can be used by both `anno` (library) and `anno-eval` (evaluation harness) without creating dependency cycles.

## Install

```toml
[dependencies]
anno-metrics = "0.6"
```

## Coreference metrics

Implements the standard coreference scoring metrics used in CoNLL-2011/2012 and CRAC shared tasks:

| Metric | Unit of evaluation | Reference |
|--------|--------------------|-----------|
| MUC | Links between mentions within an entity | Vilain et al., 1995 |
| B3 | Per-mention precision/recall | Bagga & Baldwin, 1998 |
| CEAF-e | Entity-level alignment (Dice phi4) | Luo, 2005 |
| CEAF-m | Mention-level alignment (raw count phi3) | Luo, 2005 |
| LEA | Link-based, entity-importance weighted | Moosavi & Strube, 2016 |
| BLANC | Rand-index over mention pairs | Recasens & Hovy, 2010 |
| CoNLL | Average of MUC, B3, CEAF-e F1 | Pradhan et al., 2012 |

## Usage

```rust
use anno_core::{CorefChain, Mention};
use anno_metrics::coref_metrics::conll_f1;

let gold = vec![CorefChain::new(vec![
    Mention::new("John", 0, 4),
    Mention::new("he", 10, 12),
])];
let pred = gold.clone();

let f1 = conll_f1(&pred, &gold);
assert!((f1 - 1.0).abs() < 1e-9);
```

## Output example

A runnable demo is at `examples/scoring_demo.rs` (`cargo run -p anno-metrics --example scoring_demo`).
It constructs three gold entities, introduces a split error in one predicted cluster, and scores the result:

```text
Metric     P       R       F1
------     -----   -----   -----
MUC        1.000   0.750   0.857
B3         1.000   0.810   0.895
CEAF-e     0.700   0.933   0.800
------
CoNLL F1:  0.851
```

## Cluster encoder

Primitives for representing within-context clusters and scoring merges across contexts (document windows or separate documents), used in cross-document coreference pipelines. Key types:

| Type | Purpose |
|------|---------|
| `ClusterMention` | A single mention with text, span, and entity type |
| `LocalCluster` | A set of mentions within one context window |
| `ClusterEmbedding` | Dense vector representation of a cluster |
| `ClusterEncoder` (trait) | Encode a `LocalCluster` into a `ClusterEmbedding` |
| `HeuristicClusterEncoder` | Default encoder using head-mention heuristics |
| `MergedCluster` | Result of merging two `LocalCluster`s across contexts |
| `CrossContextResolver` | Resolver that pairs clusters across contexts via a `MergeScorer` |

## License

Dual-licensed under MIT or Apache-2.0.
