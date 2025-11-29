# Datasets

Comprehensive reference for all synthetic and real datasets available in `anno`.

## Quick Reference

| Category | Count | Use Case |
|----------|-------|----------|
| Synthetic (built-in) | 28 | Unit testing, pattern validation, fast iteration |
| Real (downloadable) | 20 | Benchmarking, model comparison, production eval |

## Important: Benchmark Quality Warning

Research (2023-2024) has revealed significant annotation errors in standard NER benchmarks:

| Dataset | Error Rate | Source |
|---------|-----------|--------|
| CoNLL-03 | **7.0%** of labels incorrect | CleanCoNLL (EMNLP 2023) |
| OntoNotes 5.0 | **~8%** of entities | Bernier-Colborne (2024) |
| WikiNER | **>10%** (semi-supervised) | WikiNER-fr-gold (2024) |

**Consequence**: On original CoNLL-03, **47% of "errors" scored by F1 were actually correct predictions** penalized by annotation mistakes. After correction, SOTA F1 jumped from 94% to 97.1%.

**Recommendation**: 
- Use **synthetic datasets** for development (verified annotations, no noise)
- Use **cleaned benchmarks** (CleanCoNLL) for true error analysis
- Use **original benchmarks** only for paper comparisons

See [`docs/EVALUATION_CRITIQUE.md`](EVALUATION_CRITIQUE.md) for full research citations.

---

## Synthetic Datasets

Built-in datasets for testing. No network required.

### Core Domains

| Dataset | Function | Entity Types | Examples | Difficulty |
|---------|----------|--------------|----------|------------|
| News | `news_dataset()` | PER, ORG, LOC, DATE | ~15 | Easy-Hard |
| Biomedical | `biomedical_dataset()` | PER, ORG, LOC, Custom(GENE/DISEASE) | ~12 | Medium-Hard |
| Financial | `financial_dataset()` | PER, ORG, LOC, MONEY, PERCENT | ~10 | Medium |
| Legal | `legal_dataset()` | PER, ORG, LOC, DATE | ~8 | Medium-Hard |
| Scientific | `scientific_dataset()` | PER, ORG, LOC | ~8 | Medium |
| Entertainment | `entertainment_dataset()` | PER, ORG, LOC, DATE | ~8 | Easy-Medium |
| Social Media | `social_media_dataset()` | PER, ORG, LOC, URL, EMAIL | ~10 | Medium-Hard |

### Industry-Specific

| Dataset | Function | Focus | Entity Types | Examples |
|---------|----------|-------|--------------|----------|
| Technology | `technology_dataset()` | AI/tech companies | PER, ORG, LOC, MONEY, QUANTITY | 6 |
| Healthcare | `healthcare_dataset()` | Medical/clinical | PER, ORG, LOC, DATE, QUANTITY | 5 |
| Manufacturing | `manufacturing_dataset()` | Semiconductor/industrial | PER, ORG, LOC, MONEY, DATE | 5 |
| Automotive | `automotive_dataset()` | EV/vehicles | PER, ORG, LOC, MONEY, PERCENT, QUANTITY | 5 |
| Energy | `energy_dataset()` | Energy/climate | ORG, LOC, MONEY, PERCENT, DATE | 4 |
| Aerospace | `aerospace_dataset()` | Defense/space | PER, ORG, LOC, MONEY, DATE, QUANTITY | 4 |

### Specialized

| Dataset | Function | Focus | Examples |
|---------|----------|-------|----------|
| Sports | `sports_dataset()` | Athletes, teams, venues | ~8 |
| Politics | `politics_dataset()` | Politicians, parties | ~6 |
| E-commerce | `ecommerce_dataset()` | Products, prices | ~5 |
| Travel | `travel_dataset()` | Airlines, airports | ~5 |
| Weather | `weather_dataset()` | Forecasts, locations | ~4 |
| Academic | `academic_dataset()` | Universities, researchers | ~5 |
| Food | `food_dataset()` | Restaurants, cuisine | ~5 |
| Real Estate | `real_estate_dataset()` | Properties, prices | ~5 |
| Cybersecurity | `cybersecurity_dataset()` | CVEs, vendors | ~5 |

### Multilingual & Diversity

| Dataset | Function | Languages/Focus | Examples |
|---------|----------|-----------------|----------|
| Multilingual | `multilingual_dataset()` | DE, FR, ES, JP, CN, AR | ~8 |
| Globally Diverse | `globally_diverse_dataset()` | African, Asian, LatAm names | ~7 |

### Utility/Testing

| Dataset | Function | Purpose | Examples |
|---------|----------|---------|----------|
| Adversarial | `adversarial_dataset()` | Edge cases, ambiguity | ~20 |
| Structured | `structured_dataset()` | Tables, lists | ~10 |
| Conversational | `conversational_dataset()` | Dialog, chat | ~8 |
| Historical | `historical_dataset()` | Archaic text | ~6 |
| Hard Domain | `hard_domain_examples()` | Challenging cross-domain | ~5 |

---

## Real Datasets

Downloadable benchmarks. Requires `eval-advanced` feature.

### NER Datasets

| Dataset | ID | Source | Size | Entity Types | Format |
|---------|------|--------|------|--------------|--------|
| WikiGold | `WikiGold` | Wikipedia | ~2k sent | PER, ORG, LOC, MISC | CoNLL |
| WNUT-17 | `Wnut17` | Twitter | ~5k tweets | PER, ORG, LOC, etc. | CoNLL |
| MIT Movie | `MitMovie` | Movie queries | ~10k | Actor, Director, etc. | BIO |
| MIT Restaurant | `MitRestaurant` | Restaurant queries | ~10k | Cuisine, Location, etc. | BIO |
| CoNLL-2003 Sample | `CoNLL2003Sample` | News | ~1k sent | PER, ORG, LOC, MISC | CoNLL |
| OntoNotes Sample | `OntoNotesSample` | Mixed | ~1k sent | 18 types | CoNLL |
| MultiNERD | `MultiNERD` | Wikipedia | ~50k | 15 types | JSONL |
| BC5CDR | `BC5CDR` | PubMed | ~1.5k docs | Chemical, Disease | XML |
| NCBI Disease | `NCBIDisease` | PubMed | ~800 docs | Disease | TXT |
| FewNERD | `FewNERD` | Wikipedia | ~5k | 66 fine-grained | TXT |
| CrossNER | `CrossNER` | Multi-domain | ~5k | Domain-specific | TXT |
| UniversalNER | `UniversalNERBench` | Mixed | ~10k | Universal schema | JSON |

### Multilingual NER

| Dataset | ID | Languages | Entity Types |
|---------|------|-----------|--------------|
| WikiANN | `WikiANN` | 282 languages | PER, LOC, ORG |
| MultiCoNER | `MultiCoNER` | 12 languages | 6 coarse + 33 fine |
| MultiCoNER v2 | `MultiCoNERv2` | 12 languages | 36 types |

### Relation Extraction

| Dataset | ID | Focus | Relations |
|---------|------|-------|-----------|
| DocRED | `DocRED` | Document-level | 96 types |
| ReTACRED | `ReTACRED` | Sentence-level | 40 types |

### Coreference

| Dataset | ID | Size | Focus |
|---------|------|------|-------|
| GAP | `GAP` | 4.5k | Gender-balanced pronouns |
| PreCo | `PreCo` | 12k docs | Reading comprehension |
| LitBank | `LitBank` | 100 docs | Literary text |

---

## Usage

### Synthetic Datasets

```rust
use anno::eval::synthetic::{
    all_datasets,           // All ~200 examples
    technology_dataset,     // Single domain
    Domain, Difficulty,
};

// All examples
let all = all_datasets();

// Filter by domain
let tech = technology_dataset();
```

### Real Datasets

```rust
use anno::eval::loader::{DatasetLoader, DatasetId};

let loader = DatasetLoader::new();

// Load (downloads if needed)
let wikigold = loader.load_or_download(DatasetId::WikiGold)?;

// Check what's cached
for (id, is_cached) in loader.status() {
    println!("{:?}: {}", id, if is_cached { "✓" } else { "✗" });
}
```

### Backend Evaluation

```rust
use anno::eval::backend_eval::{BackendEvaluator, EvalConfig};

let evaluator = BackendEvaluator::new();
let report = evaluator.run_comprehensive();

println!("{}", report.to_markdown());
```

### NER Metrics

#### Modern Evaluation (2024+)

Modern NER evaluation goes beyond simple F1:

```rust
use anno::eval::dataset_quality::{DatasetQualityAnalyzer, QualityReport};

// Dataset quality metrics (research-backed)
let analyzer = DatasetQualityAnalyzer::default();
let report = analyzer.analyze(&train_data, &test_data);

println!("Unseen entity ratio: {:.1}%", report.difficulty.unseen_entity_ratio * 100.0);
println!("Entity ambiguity: {:.1}%", report.difficulty.entity_ambiguity * 100.0);
println!("Entity-null rate: {:.1}%", report.validity.entity_null_rate * 100.0);
```

Key modern metrics:
- **Unseen Entity Ratio**: % of test entities not in training (generalization test)
- **Entity Ambiguity**: Same surface form with different labels
- **Entity-Null Rate**: Token density of entities
- **Cross-corpus evaluation**: Train on X, test on Y

#### Legacy Standards (MUC/SemEval-2013)

For backwards compatibility with published benchmarks:

```rust
use anno::eval::ner_metrics::{evaluate_ner, EvalSpan};

let results = evaluate_ner(&gold, &predicted);
println!("{}", results.to_markdown());
```

Four schemas (useful for comparing with older papers):
- **Strict**: Exact boundary AND exact type
- **Exact**: Exact boundary only  
- **Partial**: Partial overlap (0.5 credit)
- **Type**: Overlap + type match

Note: These 2013 standards don't capture semantic similarity, LLM failure modes,
or cross-corpus generalization. Use `dataset_quality` for modern evaluation.

---

## Dataset Statistics

Run `cargo run --example eval_basic --features eval` to see live statistics:

```
┌─────────────────┬───────────┬────────┬────────┐
│ Backend         │ Precision │ Recall │ F1     │
├─────────────────┼───────────┼────────┼────────┤
│ Pattern         │    88.2%  │  12.8% │  22.4% │
│ Statistical     │    63.4%  │  22.2% │  32.9% │
│ Stacked         │    70.7%  │  35.0% │  46.9% │
└─────────────────┴───────────┴────────┴────────┘
```

---

## Adding Custom Datasets

```rust
use anno::eval::dataset::{AnnotatedExample, Domain, Difficulty};
use anno::eval::datasets::GoldEntity;
use anno::EntityType;

let custom = vec![
    AnnotatedExample {
        text: "Custom Corp hired Jane Doe on March 1.".into(),
        entities: vec![
            GoldEntity::new("Custom Corp", EntityType::Organization, 0),
            GoldEntity::new("Jane Doe", EntityType::Person, 19),
            GoldEntity::new("March 1", EntityType::Date, 31),
        ],
        domain: Domain::News,
        difficulty: Difficulty::Easy,
    },
];
```

