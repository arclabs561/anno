# Hack Folder - Real Data Testing

This folder contains real-world data for testing the cross-document coreference CLI.

## Structure

```
hack/real_data/
├── news/          # Scraped news articles from various sources
├── datasets/      # Extracted text from HuggingFace datasets
└── commoncrawl/   # Common Crawl data (if available)
```

## News Sources

- **Firecrawl scrapes**: Latest AI/tech news from BBC, Reuters, VentureBeat
- **Wikipedia**: Nvidia, Jensen Huang, CUDA articles
- **Tech news**: AWS re:Invent, DeepSeek, OpenAI, Nvidia partnerships

## Datasets

Extracted from HuggingFace datasets:
- **WikiGold**: Wikipedia-based NER (PER, LOC, ORG, MISC)
- **CoNLL-2003**: News articles (classic NER benchmark)
- **WNUT-17**: Social media NER (emerging entities)

## Usage

```bash
# Test with news articles
./target/debug/anno cross-doc hack/real_data/news --format tree --threshold 0.3

# Test with dataset extracts
./target/debug/anno cross-doc hack/real_data/datasets/wikigold --format tree

# Test with combined data
./target/debug/anno cross-doc hack/real_data/combined --format summary
```

## Extraction Script

To extract more dataset texts:

```bash
cargo run --example extract_dataset_texts --features eval-advanced
```

This extracts the first 20 examples from WikiGold, CoNLL-2003, and WNUT-17.

## Test Results Summary

- **113 test files** from diverse sources (news, datasets, combined)
- **324 entities** extracted across all documents
- **158 clusters** found (24 cross-doc, 15.2%)
- **Performance**: ~0.1s for 113 documents
- **Entity types**: PER (49.4%), LOC (22.2%), ORG (16.5%)

See `TESTING_NOTES.md` for detailed findings and recommendations.

