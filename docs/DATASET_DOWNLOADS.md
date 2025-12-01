# Dataset Downloads Reference

## Overview

Datasets come from **multiple sources** (unlike models which are 100% HuggingFace):
- GitHub raw files
- MIT servers
- HuggingFace datasets-server API
- HuggingFace direct downloads
- BioFLAIR GitHub
- Google Research
- CrossRE GitHub
- LitBank GitHub
- **Synthetic data** (generated in code, no download)

## Dataset Sources by Type

### 1. GitHub Raw Files (11 datasets)

| Dataset | URL | Format | Size |
|---------|-----|--------|------|
| WikiGold | `juand-r/entity-recognition-datasets` | CoNLL | ~500KB |
| WNUT-17 | `leondz/emerging_entities_17` | CoNLL | ~200KB |
| CoNLL-2003 | `autoih/conll2003` | CoNLL | ~500KB |
| OntoNotes | `autoih/conll2003` | CoNLL | ~500KB |
| BC5CDR | `shreyashub/BioFLAIR` | BIO | ~2MB |
| NCBIDisease | `shreyashub/BioFLAIR` | BIO | ~500KB |
| DocRED (proxy) | `mainlp/CrossRE` | JSON | ~100KB |
| ReTACRED (proxy) | `mainlp/CrossRE` | JSON | ~100KB |
| NYT-FB (proxy) | `mainlp/CrossRE` | JSON | ~100KB |
| WEBNLG (proxy) | `mainlp/CrossRE` | JSON | ~100KB |
| Google-RE (proxy) | `mainlp/CrossRE` | JSON | ~100KB |
| BioRED (proxy) | `mainlp/CrossRE` | JSON | ~100KB |
| GAP | `google-research-datasets/gap-coreference` | TSV | ~500KB |
| LitBank | `dbamman/litbank` | BRAT | ~1MB |

### 2. MIT Servers (3 datasets)

| Dataset | URL | Format | Size |
|---------|-----|--------|------|
| MIT Movie | `groups.csail.mit.edu/sls/downloads/movie` | BIO | ~200KB |
| MIT Restaurant | `groups.csail.mit.edu/sls/downloads/restaurant` | BIO | ~200KB |
| UniversalNERBench | `groups.csail.mit.edu/sls/downloads/movie` | BIO | ~500KB |

### 3. HuggingFace Datasets-Server API (13 datasets)

**Note**: These use the HF datasets-server API with automatic pagination (1000 rows/page, downloads full dataset)

| Dataset | HF Dataset | Config | Split | Format |
|---------|------------|--------|-------|--------|
| GENIA | `chufangao/GENIA-NER` | default | test | JSON |
| AnatEM | `disi-unibo-nlp/AnatEM` | default | test | JSON |
| BC2GM | `disi-unibo-nlp/bc2gm` | default | test | JSON |
| BC4CHEMD | `disi-unibo-nlp/bc4chemd` | default | test | JSON |
| FabNER | `DFKI-SLT/fabner` | fabner | test | JSON |
| FewNERD | `DFKI-SLT/few-nerd` | supervised | test | JSON |
| CrossNER | `DFKI-SLT/cross_ner` | ai | test | JSON |
| WikiANN | `unimelb-nlp/wikiann` | en | test | JSON |
| MultiCoNER | `DFKI-SLT/few-nerd` (proxy) | supervised | test | JSON |
| MultiCoNERv2 | `DFKI-SLT/cross_ner` (proxy) | politics | test | JSON |
| WikiNeural | `Babelscape/wikineural` | default | test_en | JSON |
| PolyglotNER | `unimelb-nlp/wikiann` (proxy) | en | test | JSON |
| UniversalNER | `Babelscape/wikineural` (proxy) | default | test_en | JSON |

### 4. HuggingFace Direct Downloads (5 datasets)

| Dataset | HF Dataset | Path | Format | Size |
|---------|------------|------|--------|------|
| MultiNERD | `Babelscape/multinerd` | `test/test_en.jsonl` | JSONL | ~50MB |
| TweetNER7 | `tner/tweetner7` | `dataset/2020.dev.json` | JSON | ~10MB |
| BroadTwitterCorpus | `GateNLP/broad_twitter_corpus` | `test/a.conll` | CoNLL | ~5MB |
| CADEC | `KevinSpaghetti/cadec` | `data/test.jsonl` | JSONL | ~2MB |

## Complete Dataset List (35 datasets)

### NER Datasets (20+)
1. **WikiGold** - GitHub (juand-r)
2. **WNUT-17** - GitHub (leondz)
3. **MIT Movie** - MIT servers
4. **MIT Restaurant** - MIT servers
5. **CoNLL-2003** - GitHub (autoih)
6. **OntoNotes** - GitHub (autoih, proxy)
7. **MultiNERD** - HuggingFace direct
8. **BC5CDR** - GitHub (BioFLAIR)
9. **NCBIDisease** - GitHub (BioFLAIR)
10. **GENIA** - HF datasets-server API
11. **AnatEM** - HF datasets-server API
12. **BC2GM** - HF datasets-server API
13. **BC4CHEMD** - HF datasets-server API
14. **TweetNER7** - HuggingFace direct
15. **BroadTwitterCorpus** - HuggingFace direct
16. **FabNER** - HF datasets-server API
17. **FewNERD** - HF datasets-server API
18. **CrossNER** - HF datasets-server API
19. **UniversalNERBench** - MIT servers
20. **WikiANN** - HF datasets-server API
21. **MultiCoNER** - HF datasets-server API (proxy)
22. **MultiCoNERv2** - HF datasets-server API (proxy)
23. **WikiNeural** - HF datasets-server API
24. **PolyglotNER** - HF datasets-server API (proxy)
25. **UniversalNER** - HF datasets-server API (proxy)

### Relation Extraction (6)
26. **DocRED** - GitHub (CrossRE proxy)
27. **ReTACRED** - GitHub (CrossRE proxy)
28. **NYT-FB** - GitHub (CrossRE proxy, RELD source)
29. **WEBNLG** - GitHub (CrossRE proxy, RELD source)
30. **Google-RE** - GitHub (CrossRE proxy)
31. **BioRED** - GitHub (CrossRE proxy)

### Discontinuous NER (1)
28. **CADEC** - HuggingFace direct

### Coreference (3)
32. **GAP** - GitHub (Google Research)
33. **PreCo** - HuggingFace (coref-data/preco, fixed source)
34. **LitBank** - GitHub (dbamman)

### Synthetic Data (Generated, No Download)

Synthetic datasets are **generated in code** (no downloads):
- Located in `src/eval/dataset/synthetic/`
- Domains: news, scientific, financial, legal, biomedical, social_media, entertainment, specialized, relations, discontinuous, misc
- Generated on-demand during tests
- No caching needed (fast generation)

## Caching Strategy

### Dataset Cache Location
- **Path**: `~/.anno_cache/datasets` (or `.anno/datasets` if `dirs` crate unavailable)
- **CI Cache**: `~/.anno_cache` with key `anno-datasets-${{ runner.os }}-v1`

### Model Cache Location (Different!)
- **Path**: `~/.cache/huggingface`
- **CI Cache**: `~/.cache/huggingface` with key `hf-models-${{ runner.os }}-v2`

**Important**: Datasets and models use **separate cache directories**.

## Download Methods

### 1. Direct HTTP (GitHub, MIT, etc.)
- Uses `ureq` crate for HTTP requests
- Downloads raw files directly
- No authentication required

### 2. HuggingFace Datasets-Server API
- REST API: `https://datasets-server.huggingface.co/rows?dataset=...`
- Returns JSON responses
- Limited to 100 rows per request (to avoid timeouts)
- Parsed via `parse_hf_api_response()`

### 3. HuggingFace Direct Downloads
- Uses HuggingFace CDN: `https://huggingface.co/datasets/.../resolve/main/...`
- Downloads full files
- No authentication required (public datasets)

### 4. Synthetic Data
- Generated in Rust code
- No download, no cache needed
- Fast generation (~milliseconds)

## Integrity Verification

- **SHA256 checksums** for downloaded datasets
- Checksums stored in code (some datasets have `expected_checksum()`)
- Verification happens on download
- Cached files are trusted (checksum verified on first download)

## Total Download Size

**Real datasets (first download):**
- GitHub files: ~5MB
- MIT servers: ~1MB
- HF datasets-server: ~50MB (now with pagination, full datasets)
- HF direct: ~70MB
- **Total**: ~126MB (increased due to full dataset downloads via pagination)

**Synthetic data:**
- Generated on-demand: 0MB (no download)

## CI Caching

**Current setup:**
```yaml
- name: Cache datasets
  uses: actions/cache@v4
  with:
    path: ~/.anno_cache
    key: anno-datasets-${{ runner.os }}-v1
```

**Recommendation**: Add `restore-keys` for partial cache hits (similar to model cache):
```yaml
restore-keys: |
  anno-datasets-${{ runner.os }}-
```

## Dataset vs Model Caches

| Type | Cache Path | CI Key | Size |
|------|------------|--------|------|
| **Models** | `~/.cache/huggingface` | `hf-models-${{ runner.os }}-v2` | ~6GB |
| **Datasets** | `~/.anno_cache` | `anno-datasets-${{ runner.os }}-v1` | ~86MB |

**Total cache size**: ~6.1GB (if both features enabled)

## Sources Summary

**Models**: 100% HuggingFace Hub (19 models, ~6GB)

**Datasets**: Multiple sources:
- GitHub: 11 datasets (~5MB)
- MIT: 3 datasets (~1MB)
- HuggingFace API: 13 datasets (~10MB, limited)
- HuggingFace Direct: 4 datasets (~70MB)
- Synthetic: Generated in code (0MB)
- **Total**: ~86MB

## Recommendations

1. **Separate caches** - Models and datasets use different cache directories (good!)
2. **Dataset cache is small** - Only ~126MB vs ~6GB for models
3. **Synthetic data is fast** - No download needed
4. **Restore-keys added** - Dataset cache now has restore-keys for better cache hits
5. **Pagination enabled** - HF datasets-server now downloads full datasets (1000 rows/page)
6. **Retry logic** - Automatic retry with exponential backoff for failed downloads
7. **Checksum verification** - SHA256 checksums verify dataset integrity

