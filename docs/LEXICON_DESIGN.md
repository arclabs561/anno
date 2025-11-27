# Lexicon Integration Design

## Research Context

Modern NER research converges on a nuanced view of lexical resources:

| Approach | When Useful | When Harmful |
|----------|-------------|--------------|
| **Hard Gazetteer** (exact match) | Closed domains, known entities | Open domains, novel entities |
| **Soft Gazetteer** (embedding similarity) | Low-resource languages, rare types | Large training data available |
| **Gated Integration** (GEMNET) | Short texts, domain mismatch | Long documents, in-domain |
| **Zero-shot** (GLiNER) | Arbitrary entity types | When precision on known entities critical |

### Key Papers

1. **GEMNET** (NAACL 2021): Gated gazetteer + MoE achieves +49% F1 on short texts
2. **Soft Gazetteers** (ACL 2020): Embedding-based soft matching for low-resource
3. **Song et al. (2020)**: Wikidata-derived gazetteers as *input features*
4. **GLiNER** (2024): Zero-shot eliminates need for fixed gazetteers

### The Core Insight

Gazetteers should be **features, not authorities**. The model learns when to trust them:

```
┌─────────────────────────────────────────────────────────────────┐
│                     GEMNET Architecture                         │
│                                                                 │
│   Text ───┬──→ BiLSTM ──────────────────────┐                   │
│           │                                  │                  │
│           └──→ Gazetteer Lookup ──→ CGR ────┤──→ MoE Gate ──→ CRF
│                                             │                   │
│                                             │                   │
│   The gate learns: "Trust gazetteer when context is ambiguous"  │
└─────────────────────────────────────────────────────────────────┘
```

## anno's Current Design

### What We Have

```
TypeMapper     ──→ Label normalization (ACTOR → Person)
PatternNER     ──→ Format detection (dates, emails, money)
RuleBasedNER   ──→ Hardcoded gazetteers (DEPRECATED)
GLiNER/BERT    ──→ Neural extraction
```

### The Gap

1. No explicit **Lexicon** abstraction (distinct from TypeMapper)
2. No **gated integration** mechanism for lexical features
3. No **soft matching** (embedding-based gazetteer lookup)
4. `ExtractionMethod::Rule` conflates gazetteers with patterns

## Proposed Design

### Conceptual Separation

```
┌─────────────────────────────────────────────────────────────────┐
│                    Knowledge Sources                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   TypeMapper          Lexicon              SemanticRegistry     │
│   ───────────         ───────              ────────────────     │
│   Label → Type        String → Type        Embedding → Type     │
│   (normalization)     (exact match)        (soft match)         │
│                                                                 │
│   "ACTOR" → Person    "Apple" → ORG        cos(span, label)     │
│   "DISEASE" → Agent   "Paris" → LOC        > threshold          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Trait Hierarchy

```rust
/// String → EntityType lookup (exact match)
/// 
/// Use for: Domain-specific entity lists (stock tickers, drug names)
/// Don't use for: General NER (use neural models instead)
pub trait Lexicon: Send + Sync {
    /// Lookup an exact string, returning matching entity type and confidence.
    fn lookup(&self, text: &str) -> Option<(EntityType, f64)>;
    
    /// Check if lexicon contains exact match.
    fn contains(&self, text: &str) -> bool;
    
    /// Get all entries (for debugging/analysis).
    fn entries(&self) -> impl Iterator<Item = (&str, &EntityType)>;
    
    /// Lexicon source identifier (for provenance tracking).
    fn source(&self) -> &str;
}

/// Embedding-based soft matching (soft gazetteer)
/// 
/// Use for: Low-resource languages, rare entity types
pub trait SoftLexicon: Send + Sync {
    /// Get similarity score for a span against all lexicon entries.
    fn similarity(&self, span_embedding: &[f32]) -> Vec<(EntityType, f64)>;
    
    /// Embed a span for lookup.
    fn embed_span(&self, text: &str) -> Vec<f32>;
}

/// Conditional lexicon integration (GEMNET-style gating)
pub trait GatedLexicon: Model {
    /// Extract with conditional lexicon trust.
    /// 
    /// The model learns when to weight lexicon evidence vs. context.
    fn extract_with_lexicon(
        &self,
        text: &str,
        lexicon: &dyn Lexicon,
        gate_threshold: f64,
    ) -> Result<Vec<Entity>>;
}
```

### ExtractionMethod Refinement

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExtractionMethod {
    /// Regex pattern matching (dates, emails, money)
    Pattern,
    
    /// Neural model inference (BERT, GLiNER)
    Neural,
    
    /// Exact lexicon lookup (gazetteers)
    Lexicon,
    
    /// Soft lexicon matching (embedding similarity)
    SoftLexicon,
    
    /// Gated: neural + lexicon with learned weighting
    GatedEnsemble,
    
    /// Multiple methods agreed
    Consensus,
    
    /// Unknown/unspecified
    Unknown,
}
```

### Example Implementations

```rust
/// Wikidata-derived lexicon (Song et al. 2020)
pub struct WikidataLexicon {
    entities: HashMap<String, (EntityType, f64)>,
    aliases: HashMap<String, String>, // alias → canonical
}

impl Lexicon for WikidataLexicon {
    fn lookup(&self, text: &str) -> Option<(EntityType, f64)> {
        let canonical = self.aliases.get(text).map(|s| s.as_str()).unwrap_or(text);
        self.entities.get(canonical).cloned()
    }
    // ...
}

/// Embedding-based soft lexicon (Rijhwani et al. 2020)
pub struct EmbeddingLexicon {
    entries: Vec<(String, EntityType)>,
    embeddings: Vec<Vec<f32>>,  // Pre-computed entry embeddings
    encoder: Box<dyn TextEncoder>,
}

impl SoftLexicon for EmbeddingLexicon {
    fn similarity(&self, span_embedding: &[f32]) -> Vec<(EntityType, f64)> {
        self.embeddings.iter()
            .zip(&self.entries)
            .map(|(emb, (_, ty))| {
                let sim = cosine_similarity(span_embedding, emb);
                (ty.clone(), sim)
            })
            .filter(|(_, sim)| *sim > 0.5)
            .collect()
    }
    // ...
}
```

## Integration with Current API

### Backward Compatibility

```rust
// Current API unchanged
let ner = StackedNER::default();
let entities = ner.extract_entities(text, None)?;

// New: optional lexicon enhancement
let lexicon = WikidataLexicon::load("path/to/wikidata")?;
let entities = ner.extract_with_lexicon(text, &lexicon, 0.7)?;

// New: soft lexicon for low-resource
let soft = EmbeddingLexicon::new(encoder, entries)?;
let entities = ner.extract_with_soft_lexicon(text, &soft)?;
```

### When to Use What

```
┌─────────────────────────────────────────────────────────────────┐
│                    Decision Tree                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Is entity type FIXED and KNOWN?                               │
│   ├─ Yes: Is text SHORT (<50 tokens)?                           │
│   │       ├─ Yes: GatedLexicon (GEMNET approach)                │
│   │       └─ No:  Neural with optional lexicon features         │
│   └─ No:  ZeroShotNER (GLiNER) - no lexicon needed              │
│                                                                 │
│   Is it LOW-RESOURCE language?                                  │
│   ├─ Yes: SoftLexicon (embedding-based)                         │
│   └─ No:  Neural models have enough signal                      │
│                                                                 │
│   Is domain CLOSED (medical codes, tickers)?                    │
│   ├─ Yes: Hard Lexicon with high confidence                     │
│   └─ No:  Neural with optional lexicon as weak prior            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation Priority

### Phase 1: Foundation
- [ ] Add `Lexicon` trait
- [ ] Refine `ExtractionMethod` enum
- [ ] Update `Entity.provenance` to track lexicon source

### Phase 2: Hard Lexicon
- [ ] `WikidataLexicon` implementation
- [ ] Simple `HashMapLexicon` for custom lists
- [ ] `extract_with_lexicon()` method on `Model`

### Phase 3: Soft Lexicon (Research-aligned)
- [ ] `SoftLexicon` trait
- [ ] `EmbeddingLexicon` using text encoder
- [ ] Integration with `SemanticRegistry`

### Phase 4: Gated Integration (Advanced)
- [ ] `GatedLexicon` trait
- [ ] GEMNET-style MoE gating
- [ ] Training data for gate weights

## Why This Design

1. **Separates concerns**: TypeMapper (labels), Lexicon (entities), SemanticRegistry (embeddings)
2. **Research-aligned**: Follows GEMNET, Soft Gazetteers, GLiNER insights
3. **Backward compatible**: Existing code unchanged
4. **Flexible**: Hard, soft, and gated modes for different use cases
5. **Provenance-aware**: Tracks which source contributed each entity

## References

- Song et al. (2020). "Improving Neural NER with Gazetteers"
- Rijhwani et al. (2020). "Soft Gazetteers for Low-Resource NER"  
- Nie et al. (2021). "GEMNET: Effective Gated Gazetteer Representations"
- Zaratiana et al. (2024). "GLiNER: Generalist Model for NER"

