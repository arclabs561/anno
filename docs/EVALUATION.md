# Evaluation

This module provides tools for measuring NER and coreference system performance. NER evaluation is complicated by the question of what counts as a "match" — should the span boundaries be exact? Does the type need to match? This library implements the SemEval-2013 evaluation modes that let you answer these questions explicitly.

### Basic usage

```rust
use anno::eval::{ReportBuilder, TestCase, SimpleGoldEntity};
use anno::PatternNER;

let model = PatternNER::new();

// Evaluate on built-in synthetic data
let report = ReportBuilder::new("PatternNER").build(&model);

// Or provide your own test cases
let tests = vec![
    TestCase {
        text: "Meeting on March 15".into(),
        gold_entities: vec![
            SimpleGoldEntity {
                text: "March 15".into(),
                entity_type: "DATE".into(),
                start: 11,
                end: 19,
            },
        ],
    },
];

let report = ReportBuilder::new("PatternNER")
    .with_test_data(tests)
    .with_error_analysis(true)
    .build(&model);

println!("{}", report.summary());
```

### F1 score variants

NER evaluation typically reports F1, but there are different ways to aggregate:

| Metric | Aggregation |
|--------|-------------|
| Micro F1 | Pools all entities together |
| Macro F1 | Averages F1 per entity type (treats rare types equally) |
| Weighted F1 | Averages F1 per type, weighted by support |

Micro F1 is standard for overall performance. Macro F1 matters when you care about performance on rare entity types that would otherwise be swamped by common ones.

### Evaluation modes

The SemEval-2013 shared task defined four evaluation modes that control matching strictness:

| Mode | Span matching | Type matching |
|------|---------------|---------------|
| Strict | Exact boundaries | Must match |
| Exact | Exact boundaries | Ignored |
| Partial | Overlap sufficient | Must match |
| Type | Ignored | Must match |

**Strict** is the standard for CoNLL-style evaluation. Use **Partial** when your application can tolerate boundary errors (e.g., "John" vs "John Smith" both referring to the same person). Use **Type** when you only care whether the system found *some* person entity, not the exact span.

```rust
use anno::eval::modes::{EvalMode, MultiModeResults};

let results = MultiModeResults::compute(&predicted, &gold);
println!("Strict F1: {:.1}%", results.strict.f1 * 100.0);
println!("Partial F1: {:.1}%", results.partial.f1 * 100.0);
```

### Coreference metrics

Coreference evaluation measures how well a system links mentions to entities. The task is different from NER: given mentions like "John", "he", "the CEO", determine which refer to the same real-world entity.

| Metric | Focus |
|--------|-------|
| MUC | Link-based. Counts missing/extra links. Ignores singletons. |
| B³ | Mention-based. Each mention contributes to precision/recall. |
| CEAF | Aligns predicted and gold clusters optimally, then scores. |
| LEA | Link-based but entity-aware. Penalizes splitting entities. |
| BLANC | Rand index over coreference/non-coreference decisions. |
| CoNLL F1 | Average of MUC, B³, and CEAF-e. Standard for comparison. |

```rust
use anno::eval::{CorefChain, Mention, conll_f1};

let gold = vec![
    CorefChain::new(0, vec![
        Mention::new("John", 0, 4),
        Mention::new("he", 20, 22),
    ]),
];
let pred = gold.clone();

let (p, r, f1) = conll_f1(&gold, &pred);
```

### BIO sequence conversion

Many NER datasets use BIO (Begin-Inside-Outside) sequence labels rather than span annotations. This library converts between them:

```rust
use anno::eval::bio_adapter::{bio_to_entities, BioScheme};

let tokens = ["John", "Smith", "works", "at", "Apple"];
let tags = ["B-PER", "I-PER", "O", "O", "B-ORG"];

let entities = bio_to_entities(&tokens, &tags, BioScheme::IOB2)?;
// entities[0].text == "John Smith"
```

### Datasets

With `--features eval-advanced`, you can load standard NER datasets:

```rust
use anno::eval::loader::{DatasetLoader, DatasetId};

let loader = DatasetLoader::new()?;
let dataset = loader.load_or_download(DatasetId::WikiGold)?;
```

| Dataset | Domain | Entities | GLiNER Paper |
|---------|--------|----------|--------------|
| CoNLL-2003 | News | PER, LOC, ORG, MISC | Yes |
| WikiGold | Wikipedia | PER, LOC, ORG, MISC | No |
| WNUT-17 | Social media | Emerging/novel entities | No |
| MIT Movie | Film reviews | Actor, Director, Genre, etc. | Yes |
| MIT Restaurant | Restaurants | Cuisine, Dish, Location, etc. | Yes |
| BC5CDR | Biomedical | Disease, Chemical | Yes |
| FewNERD | Cross-domain | 66 fine-grained types | No |

#### GLiNER Paper Benchmark Datasets (New)

We now support the full GLiNER 20-dataset benchmark from arxiv:2311.08526:

| Category | Datasets | Description |
|----------|----------|-------------|
| Biomedical | GENIA, AnatEM, BC2GM, BC4CHEMD, BC5CDR, NCBI Disease | PubMed/MEDLINE abstracts |
| Social Media | TweetNER7, BroadTwitterCorpus | Twitter NER |
| Specialized | FabNER, MIT Movie, MIT Restaurant | Domain-specific |
| Multilingual | WikiANN, WikiNeural, PolyglotNER, MultiNERD | Cross-lingual evaluation |

```rust
// Load biomedical benchmark datasets
let loader = DatasetLoader::new()?;
for dataset_id in DatasetId::all_biomedical() {
    let dataset = loader.load_or_download(*dataset_id)?;
    println!("{}: {} entities", dataset_id.name(), dataset.entity_count());
}

// Load social media benchmark datasets
for dataset_id in DatasetId::all_social_media() {
    let dataset = loader.load_or_download(*dataset_id)?;
}
```

Different datasets use different entity schemas. `TypeMapper` normalizes them:

```rust
use anno::TypeMapper;

// MIT Movie uses "ACTOR", our model uses "PER"
let mapper = TypeMapper::mit_movie();
```

### Bias analysis

With `--features eval-bias`, you can test for demographic biases:

```rust
use anno::eval::GenderBiasEvaluator;

let evaluator = GenderBiasEvaluator::new();
let results = evaluator.evaluate(&model)?;
```

This runs WinoBias-style tests that check whether the model performs equally on sentences with male vs female pronouns.

### Calibration

With `--features eval-advanced`, you can check whether confidence scores are calibrated:

```rust
use anno::eval::CalibrationEvaluator;

let cal = CalibrationEvaluator::new();
let results = cal.evaluate(&model, &test_data)?;
```

A well-calibrated model's 80% confidence predictions should be correct ~80% of the time. Miscalibration means the scores are useful for ranking but not as probabilities.

### Feature flags

| Feature | Modules |
|---------|---------|
| `eval` | Core metrics, BIO adapter, coreference, datasets |
| `eval-bias` | Gender, demographic, temporal, length bias |
| `eval-advanced` | Calibration, robustness, active learning |
| `eval-full` | All of the above |

### Multilingual NER

The non-ML backends (`PatternNER`, `HeuristicNER`) have different multilingual capabilities:

**PatternNER (regex-based)** works well across languages for structured entities:

| Entity Type | Cross-lingual Support | Notes |
|-------------|----------------------|-------|
| ISO dates (2024-01-15) | Universal | Language-agnostic format |
| Emails | Universal | Format-based detection |
| URLs | Universal | Format-based detection |
| Currency symbols ($, €, £, ¥) | Universal | Symbol + Unicode digits |
| Arabic-Indic numerals | Supported | Rust regex `\d` is Unicode-aware |
| Japanese dates (年月日) | Supported | 2024年1月15日 format |
| Korean dates (년월일) | Supported | 2024년 1월 15일 format |
| German months | Supported | Januar, Februar, März, etc. |
| French months | Supported | janvier, février, mars, etc. |
| Spanish months | Supported | enero, febrero, marzo, etc. |
| Italian months | Supported | gennaio, febbraio, marzo, etc. |
| Portuguese months | Supported | janeiro, fevereiro, março, etc. |
| Dutch months | Supported | januari, februari, maart, etc. |
| Russian months | Supported | января, февраля, марта, etc. |

**HeuristicNER (heuristic-based)** is English-centric:

| Language Category | Expected Performance | Challenge |
|------------------|---------------------|-----------|
| English | ~60-70% F1 | Reference implementation |
| German, French, Spanish | ~40-50% F1 | Capitalization helps, context words English |
| Russian, Greek | ~30-40% F1 | Different script, caps help |
| Chinese, Japanese, Korean | ~0% F1 | No capitalization signal |
| Arabic, Hebrew | ~0% F1 | RTL + no capitalization |

For production multilingual NER, use the ML backends:
- **GLiNER/NuNER**: Zero-shot NER works across languages with multilingual embeddings
- **Fine-tuned BERT**: Language-specific models available for most major languages

Multilingual evaluation datasets (all loadable via `DatasetLoader`):
- **WikiANN (PAN-X)**: 282 languages, PER/LOC/ORG entities — `DatasetId::WikiANN`
- **MultiCoNER**: 12 languages, 33 fine-grained entity types — `DatasetId::MultiCoNER`
- **MultiCoNER v2**: 36 entity types including medical — `DatasetId::MultiCoNERv2`
- **MultiNERD**: 15 entity types, Wikipedia-derived — `DatasetId::MultiNERD`
- **WikiNeural**: 9 languages, silver NER data from Wikipedia (GLiNER benchmark) — `DatasetId::WikiNeural`
- **PolyglotNER**: 40 languages, Wikipedia + Freebase (GLiNER benchmark) — `DatasetId::PolyglotNER`
- **UniversalNER**: 13 languages, gold-standard cross-lingual annotations (NAACL 2024) — `DatasetId::UniversalNER`

```rust
use anno::eval::loader::{DatasetLoader, DatasetId};

let loader = DatasetLoader::new()?;

// Load WikiANN (English subset)
let wikiann = loader.load_or_download(DatasetId::WikiANN)?;

// Load MultiCoNER
let multiconer = loader.load_or_download(DatasetId::MultiCoNER)?;
```

See `tests/multilingual_ner_tests.rs` for comprehensive cross-lingual test coverage.

### A Note on Evaluation Standards

**The SemEval-2013 metrics implemented here are legacy standards** useful for comparing to published work, but they have known limitations documented in 2023-2024 research:

- **Benchmark noise**: 7-10% of labels in CoNLL-03/OntoNotes are incorrect
- **False errors**: 47% of "errors" on CoNLL-03 are actually correct predictions penalized by annotation mistakes
- **Single-score blindness**: F1 hides boundary vs type errors, rare entity performance, error severity

See [`docs/EVALUATION_CRITIQUE.md`](EVALUATION_CRITIQUE.md) for:
- Research citations on evaluation limitations
- What `anno` does differently (dataset quality metrics, error taxonomy)
- Recommended modern evaluation workflow

For 2024+ evaluation, prefer:
- `anno::eval::dataset_quality` — unseen entity ratio, ambiguity metrics
- `anno::eval::error_analysis` — fine-grained error taxonomy
- Synthetic datasets with verified annotations (no benchmark noise)

### References

**Legacy Standards (for comparison to published work)**
- [SemEval-2013 Task 9](https://aclanthology.org/S13-2056/) — Evaluation modes
- [CoNLL-2012](https://aclanthology.org/W12-4501/) — Coreference metrics
- [WinoBias](https://arxiv.org/abs/1804.06876) — Gender bias evaluation

**Modern Critiques (recommended reading)**
- [CleanCoNLL](https://arxiv.org/abs/2310.16225) — 7% of CoNLL-03 labels corrected (EMNLP 2023)
- [OntoNotes Errors](https://arxiv.org/abs/2406.19172) — 8% of entities corrected (2024)
- [TMR](https://arxiv.org/abs/2103.12312) — Tough Mentions Recall (2021)
- [Coref Measurement Modeling](https://arxiv.org/abs/2303.09092) — Generalization validity (ACL 2024)
