# LLM-Based NER Design

Research-driven design for integrating LLM backends into `anno`, based on:
- **CodeNER** (arXiv:2507.20423): Code-based prompting for NER
- **CMAS** (arXiv:2502.18702): Cooperative multi-agent zero-shot NER

## Architecture Overview

```text
┌─────────────────────────────────────────────────────────────────┐
│                        LLM NER Backend                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐   │
│  │  Code Prompt  │ →  │     LLM       │ →  │   Parser      │   │
│  │  Generator    │    │   (API/Local) │    │   (BIO→Ent)   │   │
│  └───────────────┘    └───────────────┘    └───────────────┘   │
│         ↑                    ↑                    ↓             │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐   │
│  │ Entity Schema │    │  Demonstr.    │    │    Output     │   │
│  │ (BIO labels)  │    │  Selector     │    │   Entities    │   │
│  └───────────────┘    └───────────────┘    └───────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## CodeNER: Code-Based Prompting

### Key Insight

LLMs trained on code understand:
- Structured scope boundaries (function definitions, blocks)
- Variable naming conventions
- Type annotations
- Sequential instruction execution

This translates to NER as:
- Entity boundaries = function scopes
- Entity types = type annotations
- BIO tags = sequential state machine

### Prompt Template

```python
def extract_entities(text: str) -> list[dict]:
    """
    Extract named entities from text using BIO tagging.
    
    BIO Schema:
    - B-{TYPE}: Beginning of entity of TYPE
    - I-{TYPE}: Inside (continuation) of entity
    - O: Outside any entity
    
    Entity Types:
    - PER: Person names
    - ORG: Organizations
    - LOC: Locations
    - DATE: Temporal expressions
    - MONEY: Monetary values
    """
    tokens = text.split()
    labels = []
    
    # Label each token
    for token in tokens:
        # Analyze token in context
        # Return appropriate BIO tag
        pass
    
    return labels

# Input:
text = "{input_text}"

# Expected output format:
# [{"text": "entity_text", "type": "TYPE", "start": N, "end": M}, ...]
```

### Implementation Strategy

1. **Prompt Construction**
   - Embed BIO schema as docstring
   - Include entity type definitions
   - Format input as function call
   - Request structured JSON output

2. **Output Parsing**
   - Parse JSON array of entities
   - Validate spans against input text
   - Handle confidence if provided

3. **Chain-of-Thought Extension**
   - Add reasoning step before labeling
   - "First, identify potential entities..."
   - Improves accuracy on ambiguous cases

## CMAS: Multi-Agent Approach

### Agent Roles

1. **Self-Annotator**: Initial entity labeling
2. **TRF Extractor**: Finds context features
3. **Demonstration Discriminator**: Scores example quality
4. **Overall Predictor**: Final ensemble decision

### Simplified Implementation

For practical deployment, combine into 2 LLM calls:

```text
Pass 1: Entity Recognition + Feature Extraction
  Input: text
  Output: preliminary entities + context features

Pass 2: Refinement with Self-Consistency
  Input: text + pass 1 output + selected demonstrations
  Output: final entities with confidence
```

### Demonstration Selection

```rust
pub struct DemonstrationSelector {
    examples: Vec<AnnotatedExample>,
    embeddings: Option<Vec<Vec<f32>>>,
}

impl DemonstrationSelector {
    /// Select demonstrations based on similarity and helpfulness
    pub fn select(&self, query: &str, k: usize) -> Vec<&AnnotatedExample> {
        // 1. Compute query embedding
        // 2. Find k-nearest neighbors
        // 3. Filter by helpfulness score (> threshold)
        // 4. Return selected demonstrations
    }
}
```

## API Design

### Trait Extension

```rust
/// LLM-based NER backend
pub trait LLMBackend: Model {
    /// Get the underlying LLM provider name
    fn llm_provider(&self) -> &str;
    
    /// Extract with custom prompt template
    fn extract_with_prompt(
        &self,
        text: &str,
        template: &PromptTemplate,
    ) -> Result<Vec<Entity>>;
    
    /// Extract with chain-of-thought reasoning
    fn extract_with_reasoning(
        &self,
        text: &str,
        entity_types: &[EntityType],
    ) -> Result<(Vec<Entity>, String)>;  // entities + reasoning
}

/// Prompt template for code-based NER
pub struct PromptTemplate {
    pub system: String,
    pub schema: BIOSchema,
    pub demonstrations: Vec<AnnotatedExample>,
    pub use_cot: bool,
}

/// BIO tagging schema
pub struct BIOSchema {
    pub entity_types: Vec<EntityType>,
    pub descriptions: HashMap<EntityType, String>,
}
```

### Backend Implementations

```rust
// OpenAI-compatible API
pub struct OpenAINER {
    client: OpenAIClient,
    model: String,
    template: PromptTemplate,
}

// Local LLM via llama.cpp
pub struct LlamaNER {
    model_path: PathBuf,
    template: PromptTemplate,
}

// Anthropic Claude
pub struct ClaudeNER {
    client: AnthropicClient,
    model: String,
    template: PromptTemplate,
}
```

## Evaluation Integration

### Metrics for LLM NER

1. **Standard NER metrics**: P/R/F1
2. **Latency**: Important for API-based
3. **Cost**: Tokens consumed
4. **Consistency**: Same input → same output?
5. **Reasoning quality**: If using CoT

### Calibration Considerations

LLM confidence ≠ Probability

Options:
- Ask for numeric confidence (often poorly calibrated)
- Use temperature sampling for uncertainty
- Ensemble multiple prompts

## Feature Flags

```toml
[features]
llm = []                    # Enable LLM backends
llm-openai = ["llm", "dep:reqwest", "dep:serde_json"]
llm-anthropic = ["llm", "dep:reqwest", "dep:serde_json"]
llm-local = ["llm", "dep:llama-cpp-rs"]
```

## References

1. CodeNER: Code Prompting for NER (arXiv:2507.20423)
   - Key: BIO schema as code, exploits code understanding
   - Result: Outperforms text prompts across 10 benchmarks

2. CMAS: Cooperative Multi-Agent Zero-Shot NER (arXiv:2502.18702)
   - Key: Multi-agent with TRF extractor + demonstration discriminator
   - Result: Significant improvements on 6 benchmarks

3. GPT-NER: Named Entity Recognition via Large Language Models
   - Key: In-context learning for NER
   - Baseline approach

4. PromptNER: Prompt Learning for NER
   - Key: Prompt engineering techniques
   - Comparison point

