# Document Ingestion and Preparation Design

## Current State Analysis

### What Exists
- **Input Sources**: Text arg, file, stdin, positional args
- **URL Downloading**: Only for dataset downloads in `eval/loader.rs` (uses `ureq`)
- **Text Processing**: Basic `get_input_text()` helper
- **No Document Cleaning**: Raw text is passed directly to models
- **No URL Resolution**: No general-purpose URL → text extraction

### What's Missing

1. **Document Preparation Pipeline**
   - Text cleaning (whitespace normalization, encoding issues)
   - HTML/PDF extraction (if URLs provided)
   - Chunking for long documents
   - Language detection and normalization

2. **URL Resolution Connectors**
   - HTTP/HTTPS fetching
   - HTML content extraction
   - PDF text extraction
   - Error handling and retries
   - Caching of fetched content

3. **Document Cleaning**
   - Unicode normalization
   - Whitespace cleanup
   - Encoding detection/correction
   - Line break normalization

## Proposed Architecture

### 1. Document Source Abstraction

```rust
pub enum DocumentSource {
    Text(String),
    File(PathBuf),
    Url(String),
    Stdin,
}

pub struct DocumentInput {
    source: DocumentSource,
    id: String,
    metadata: HashMap<String, String>,
}
```

### 2. URL Resolution Connectors

```rust
pub trait UrlResolver {
    fn can_resolve(&self, url: &str) -> bool;
    fn resolve(&self, url: &str) -> Result<ResolvedContent>;
}

pub struct ResolvedContent {
    pub text: String,
    pub metadata: HashMap<String, String>,
    pub source_url: String,
}

// Implementations:
// - HttpResolver: Basic HTTP/HTTPS
// - HtmlExtractor: Extract text from HTML
// - PdfExtractor: Extract text from PDF (optional feature)
```

### 3. Document Preparation Pipeline

```rust
pub struct DocumentPreprocessor {
    pub clean_whitespace: bool,
    pub normalize_unicode: bool,
    pub detect_language: bool,
    pub chunk_size: Option<usize>,
}

impl DocumentPreprocessor {
    pub fn prepare(&self, text: &str) -> PreparedDocument {
        // 1. Clean whitespace
        // 2. Normalize Unicode
        // 3. Detect language
        // 4. Optionally chunk
    }
}
```

### 4. Integration Points

**CLI Input Flow**:
```
User Input → DocumentSource → UrlResolver (if URL) → DocumentPreprocessor → GroundedDocument
```

**Pipeline Command**:
```
URLs/Files → Resolve → Clean → Extract → Enhance → Cross-Doc
```

## Implementation Plan

### Phase 1: URL Resolution (High Priority)
- Add `ureq` dependency (already used in eval/loader)
- Create `src/ingest/url_resolver.rs`
- Implement HTTP resolver
- Add HTML text extraction (simple, no full HTML parser needed)
- Integrate into `get_input_text()`

### Phase 2: Document Cleaning (Medium Priority)
- Create `src/ingest/preprocessor.rs`
- Implement whitespace normalization
- Add Unicode normalization (NFC)
- Language detection (already exists in `lang.rs`)

### Phase 3: Advanced Extraction (Low Priority)
- PDF extraction (requires additional dependency)
- HTML structure preservation (for better entity linking)
- Chunking for long documents

## CLI Integration

### Enhanced Input Handling

```bash
# URL support
anno extract https://example.com/article --model gliner

# Multiple URLs
anno pipeline --urls https://site1.com https://site2.com --coref

# URL with cleaning
anno extract https://example.com --clean --normalize

# File with preprocessing
anno extract file.txt --clean --detect-lang
```

### New Flags

- `--url`: Accept URL as input
- `--urls`: Multiple URLs (for pipeline/batch)
- `--clean`: Enable text cleaning
- `--normalize`: Unicode normalization
- `--chunk-size N`: Chunk long documents

## Design Principles

1. **Progressive Enhancement**: Basic text → URL → Cleaning → Advanced extraction
2. **Feature Gating**: PDF/advanced HTML parsing behind features
3. **Caching**: Cache resolved URLs to avoid re-fetching
4. **Error Handling**: Graceful degradation (fallback to raw text if cleaning fails)
5. **Consistency**: Same input handling across all commands

