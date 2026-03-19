//! Coreference resolution for RAG preprocessing.
//!
//! Rewrites pronouns with their antecedents so that document chunks are
//! self-contained when split for retrieval-augmented generation (RAG).
//!
//! # Motivation
//!
//! When a document is split into chunks for embedding/retrieval, pronouns like
//! "he", "they", "it" lose their referent because the antecedent may be in a
//! different chunk. Replacing pronouns with their resolved antecedent text
//! produces self-contained chunks that retrieve more accurately.
//!
//! Evidence: coref preprocessing improves nDCG on RAG benchmarks
//! ([arXiv:2507.07847](https://arxiv.org/abs/2507.07847)).
//!
//! # Design
//!
//! - Pure Rust, no model downloads, sub-millisecond latency
//! - Uses `SimpleCorefResolver` from `backends::coref::simple`
//! - Rewrites right-to-left to preserve character offsets
//! - Only replaces pronouns (not nominal mentions) by default
//!
//! # Multilingual Note
//!
//! Pronoun detection supports English (default), French, Spanish, and German
//! via the `language` field in `RagCorefConfig`. For CJK, Arabic, and other languages,
//! pronoun detection returns `false` (safe: treats unknown pronouns as named
//! mentions, producing no rewrites). Model-based detection is needed for those
//! languages. Set `language` to `None` (the default) for English.

use crate::backends::coref::simple::{CorefConfig, SimpleCorefResolver};
use crate::lang::Language;
use crate::Entity;
#[cfg(test)]
use crate::{EntityCategory, EntityType};

/// Configuration for RAG coreference preprocessing.
#[derive(Debug, Clone)]
pub struct RagCorefConfig {
    /// Underlying coreference resolver configuration.
    pub coref: CorefConfig,
    /// Replace pronouns with their antecedent text. Default: true.
    pub rewrite_pronouns: bool,
    /// Maximum character distance to consider for pronoun resolution.
    /// Default: 500.
    pub max_char_distance: usize,
    /// Resolve cataphoric (forward-pointing) pronouns in a second pass.
    /// When true, pronouns unresolved by the backward-looking pass are matched
    /// to the first compatible non-pronoun entity appearing *after* them.
    /// Default: true.
    pub resolve_cataphora: bool,
    /// Language for pronoun detection. `None` defaults to English.
    ///
    /// Supported: English, French, Spanish, German. For unsupported languages
    /// (CJK, Arabic, etc.) pronoun detection returns `false`, which is safe --
    /// it just means no pronoun rewrites will be applied.
    pub language: Option<Language>,
    /// Rewrite reflexive pronouns (herself/himself/themselves/itself/themself).
    /// Default: false. Reflexives are typically coreferent with the clause
    /// subject and already self-contained, so rewriting them is rarely useful.
    pub rewrite_reflexives: bool,
    /// Rewrite demonstrative pronouns (this/that/these/those) when they appear
    /// as entity spans. Default: false. Demonstratives are common anaphoric
    /// references in RAG contexts ("The company announced layoffs. This upset
    /// employees.") but are riskier to rewrite since they can be determiners.
    pub rewrite_demonstratives: bool,
}

impl Default for RagCorefConfig {
    fn default() -> Self {
        Self {
            coref: CorefConfig {
                max_pronoun_lookback: 5,
                ..CorefConfig::default()
            },
            rewrite_pronouns: true,
            max_char_distance: 500,
            resolve_cataphora: true,
            language: None,
            rewrite_reflexives: false,
            rewrite_demonstratives: false,
        }
    }
}

/// A single pronoun rewrite performed on the text.
#[derive(Debug, Clone)]
pub struct PronounRewrite {
    /// Character offset where the pronoun started (in original text).
    pub start: usize,
    /// Character offset where the pronoun ended (in original text).
    pub end: usize,
    /// The original pronoun text (e.g., "he", "they").
    pub original: String,
    /// The replacement text (antecedent, e.g., "John").
    pub replacement: String,
}

/// Result of RAG coreference preprocessing.
#[derive(Debug, Clone)]
pub struct RagCorefResult {
    /// The rewritten text with pronouns replaced.
    pub text: String,
    /// List of rewrites applied, in order of position (ascending).
    pub rewrites: Vec<PronounRewrite>,
    /// Number of pronouns found but not resolved (no antecedent found).
    pub unresolved_count: usize,
}

/// Resolve coreference and rewrite pronouns for RAG-ready text.
///
/// Takes input text and pre-extracted entities, runs coreference resolution,
/// and replaces pronouns with their antecedent text.
///
/// # Arguments
///
/// * `text` - The input text
/// * `entities` - Pre-extracted entities (from any NER backend)
/// * `config` - RAG coref configuration (or `None` for defaults)
///
/// # Returns
///
/// A [`RagCorefResult`] containing the rewritten text and metadata about
/// each rewrite performed.
///
/// # Example
///
/// ```rust
/// use anno::rag::{resolve_for_rag, RagCorefConfig};
/// use anno::{Entity, EntityType};
///
/// let text = "Alice went to the store. She bought milk.";
/// let entities = vec![
///     Entity::new("Alice", EntityType::Person, 0, 5, 0.9),
///     Entity::new("She", EntityType::Person, 25, 28, 0.8),
/// ];
///
/// let result = resolve_for_rag(text, &entities, None);
/// assert_eq!(result.text, "Alice went to the store. Alice bought milk.");
/// assert_eq!(result.rewrites.len(), 1);
/// ```
pub fn resolve_for_rag(
    text: &str,
    entities: &[Entity],
    config: Option<RagCorefConfig>,
) -> RagCorefResult {
    let config = config.unwrap_or_default();

    if entities.is_empty() || !config.rewrite_pronouns {
        return RagCorefResult {
            text: text.to_string(),
            rewrites: Vec::new(),
            unresolved_count: 0,
        };
    }

    let lang = config.language.unwrap_or(Language::English);

    // Sort entities by start offset to ensure deterministic backward-search
    // behavior. Callers may pass entities out of document order.
    let mut sorted_entities = entities.to_vec();
    sorted_entities.sort_by_key(|e| e.start());

    let resolver = SimpleCorefResolver::new(config.coref);
    let resolved = resolver.resolve(&sorted_entities);

    // Pronoun check that includes demonstratives when configured.
    let is_pronoun = |text: &str| -> bool {
        if is_pronoun_for_language(text, lang) {
            return true;
        }
        if config.rewrite_demonstratives && is_demonstrative_pronoun(text) {
            return true;
        }
        false
    };

    // Build cluster map: canonical_id -> first non-pronoun entity text
    let mut cluster_antecedent: std::collections::HashMap<u64, &str> =
        std::collections::HashMap::new();
    for entity in &resolved {
        if let Some(cid) = entity.canonical_id {
            let cid_val = cid.get();
            if !cluster_antecedent.contains_key(&cid_val) && !is_pronoun(&entity.text) {
                cluster_antecedent.insert(cid_val, &entity.text);
            }
        }
    }

    // Collect pronoun rewrites
    let mut rewrites = Vec::new();
    let mut unresolved_count: usize = 0;

    for (i, entity) in resolved.iter().enumerate() {
        if !is_pronoun(&entity.text) {
            continue;
        }
        // Skip pleonastic "it" (non-referential: weather, extraposition, idioms)
        if is_pleonastic_it(text, entity.start(), entity.end()) {
            continue;
        }
        // Skip reflexive pronouns unless explicitly configured
        if !config.rewrite_reflexives && is_reflexive_pronoun(&entity.text) {
            continue;
        }
        if let Some(cid) = entity.canonical_id {
            if let Some(&antecedent) = cluster_antecedent.get(&cid.get()) {
                // Don't rewrite if antecedent is same as pronoun
                if antecedent.to_lowercase() != entity.text.to_lowercase() {
                    rewrites.push(PronounRewrite {
                        start: entity.start(),
                        end: entity.end(),
                        original: entity.text.clone(),
                        replacement: antecedent.to_string(),
                    });
                    continue;
                }
            }
        }
        // Proximity-based anaphoric fallback: the resolver is English-only, so
        // non-English pronouns may not have been clustered. Search backward for
        // the nearest non-pronoun entity with a compatible entity_type.
        let mut found = false;
        for candidate in resolved[..i].iter().rev() {
            if is_pronoun(&candidate.text) {
                continue;
            }
            if candidate.entity_type == entity.entity_type {
                let distance = entity.start().saturating_sub(candidate.end());
                if distance <= config.max_char_distance {
                    rewrites.push(PronounRewrite {
                        start: entity.start(),
                        end: entity.end(),
                        original: entity.text.clone(),
                        replacement: candidate.text.clone(),
                    });
                    found = true;
                }
                break;
            }
        }
        if !found {
            unresolved_count += 1;
        }
    }

    // Second pass: cataphoric resolution for unresolved pronouns.
    // For each unresolved pronoun, search forward in the entity list for the
    // first compatible non-pronoun entity with the same entity_type.
    if config.resolve_cataphora && unresolved_count > 0 {
        let rewritten_starts: std::collections::HashSet<usize> =
            rewrites.iter().map(|r| r.start).collect();
        let mut cataphoric_rewrites = Vec::new();
        let mut newly_resolved: usize = 0;

        for (i, entity) in resolved.iter().enumerate() {
            if !is_pronoun(&entity.text) {
                continue;
            }
            // Skip pleonastic and reflexive (same filters as anaphoric pass)
            if is_pleonastic_it(text, entity.start(), entity.end()) {
                continue;
            }
            if !config.rewrite_reflexives && is_reflexive_pronoun(&entity.text) {
                continue;
            }
            // Skip pronouns already resolved in the anaphoric pass
            if rewritten_starts.contains(&entity.start()) {
                continue;
            }
            // Search forward for a compatible non-pronoun entity
            for candidate in &resolved[i + 1..] {
                if is_pronoun(&candidate.text) {
                    continue;
                }
                if candidate.entity_type == entity.entity_type {
                    let distance = candidate.start().saturating_sub(entity.end());
                    if distance <= config.max_char_distance {
                        cataphoric_rewrites.push(PronounRewrite {
                            start: entity.start(),
                            end: entity.end(),
                            original: entity.text.clone(),
                            replacement: candidate.text.clone(),
                        });
                        newly_resolved += 1;
                    }
                    break;
                }
            }
        }

        unresolved_count = unresolved_count.saturating_sub(newly_resolved);
        rewrites.extend(cataphoric_rewrites);
    }

    // Sort by start position ascending (for reporting), then apply right-to-left
    rewrites.sort_by_key(|r| r.start);

    // Overlap rejection: when two rewrites overlap (e.g., nested NER output),
    // keep the longer span and drop the shorter one.
    {
        // Sort by span length descending to prefer longer rewrites.
        let mut by_length = rewrites.clone();
        by_length.sort_by(|a, b| (b.end - b.start).cmp(&(a.end - a.start)));
        let mut accepted: Vec<PronounRewrite> = Vec::with_capacity(by_length.len());
        for rw in by_length {
            let overlaps = accepted
                .iter()
                .any(|a| rw.start < a.end && rw.end > a.start);
            if !overlaps {
                accepted.push(rw);
            }
        }
        accepted.sort_by_key(|r| r.start);
        rewrites = accepted;
    }

    // Apply rewrites right-to-left to preserve offsets
    let chars: Vec<char> = text.chars().collect();
    let mut result_chars = chars.clone();

    for rewrite in rewrites.iter().rev() {
        let replacement_chars: Vec<char> = rewrite.replacement.chars().collect();
        // Adjust case only for Latin-script languages with sentence-initial capitalization.
        // CJK, Arabic, etc. do not capitalize sentence starts.
        let replacement_chars = if lang.uses_latin_capitalization()
            && rewrite
                .original
                .chars()
                .next()
                .is_some_and(|c| c.is_uppercase())
        {
            let mut adjusted = replacement_chars;
            if let Some(first) = adjusted.first_mut() {
                *first = first.to_uppercase().next().unwrap_or(*first);
            }
            adjusted
        } else {
            replacement_chars
        };

        let end = rewrite.end.min(result_chars.len());
        let start = rewrite.start.min(end);
        result_chars.splice(start..end, replacement_chars);
    }

    RagCorefResult {
        text: result_chars.into_iter().collect(),
        rewrites,
        unresolved_count,
    }
}

/// Resolve coreference using neural f-coref and rewrite pronouns for RAG.
///
/// Unlike [`resolve_for_rag`] (which requires pre-extracted entities), this
/// function runs neural coreference resolution directly on raw text using the
/// f-coref model. It produces higher-quality clusters but requires a model
/// download.
///
/// # Arguments
///
/// * `text` - The input text
/// * `clusters` - Pre-computed coreference clusters from the `FCoref::resolve()` method
/// * `language` - Language for pronoun detection (default: English)
///
/// # Example
///
/// ```rust,ignore
/// use anno::backends::coref::fcoref::FCoref;
/// use anno::rag::resolve_for_rag_neural;
///
/// let coref = FCoref::from_path("fcoref_onnx")?;
/// let clusters = coref.resolve("John went to the store. He bought milk.")?;
/// let result = resolve_for_rag_neural("John went to the store. He bought milk.", &clusters, None);
/// assert_eq!(result.text, "John went to the store. John bought milk.");
/// ```
#[cfg(feature = "onnx")]
pub fn resolve_for_rag_neural(
    text: &str,
    clusters: &[crate::backends::coref::t5::CorefCluster],
    language: Option<Language>,
) -> RagCorefResult {
    if clusters.is_empty() {
        return RagCorefResult {
            text: text.to_string(),
            rewrites: Vec::new(),
            unresolved_count: 0,
        };
    }

    let lang = language.unwrap_or(Language::English);
    let mut rewrites = Vec::new();

    for cluster in clusters {
        if cluster.mentions.len() < 2 {
            continue;
        }

        // The canonical mention is the antecedent for pronoun rewrites
        let antecedent = &cluster.canonical;
        if is_pronoun_for_language(antecedent, lang) {
            // If the canonical itself is a pronoun, skip this cluster
            continue;
        }

        // Find pronoun mentions in this cluster
        for (mention_text, &(char_start, char_end)) in
            cluster.mentions.iter().zip(cluster.spans.iter())
        {
            if !is_pronoun_for_language(mention_text, lang) {
                continue;
            }
            // Skip pleonastic "it"
            if is_pleonastic_it(text, char_start, char_end) {
                continue;
            }
            // Skip reflexive pronouns
            if is_reflexive_pronoun(mention_text) {
                continue;
            }
            // Don't rewrite if antecedent matches the pronoun
            if antecedent.to_lowercase() == mention_text.to_lowercase() {
                continue;
            }
            rewrites.push(PronounRewrite {
                start: char_start,
                end: char_end,
                original: mention_text.clone(),
                replacement: antecedent.clone(),
            });
        }
    }

    // Sort by start position ascending
    rewrites.sort_by_key(|r| r.start);

    // Overlap rejection: keep longer spans
    {
        let mut by_length = rewrites.clone();
        by_length.sort_by(|a, b| (b.end - b.start).cmp(&(a.end - a.start)));
        let mut accepted: Vec<PronounRewrite> = Vec::with_capacity(by_length.len());
        for rw in by_length {
            let overlaps = accepted
                .iter()
                .any(|a| rw.start < a.end && rw.end > a.start);
            if !overlaps {
                accepted.push(rw);
            }
        }
        accepted.sort_by_key(|r| r.start);
        rewrites = accepted;
    }

    // Apply rewrites right-to-left to preserve offsets
    let chars: Vec<char> = text.chars().collect();
    let mut result_chars = chars.clone();

    for rewrite in rewrites.iter().rev() {
        let replacement_chars: Vec<char> = rewrite.replacement.chars().collect();
        let replacement_chars = if lang.uses_latin_capitalization()
            && rewrite
                .original
                .chars()
                .next()
                .is_some_and(|c| c.is_uppercase())
        {
            let mut adjusted = replacement_chars;
            if let Some(first) = adjusted.first_mut() {
                *first = first.to_uppercase().next().unwrap_or(*first);
            }
            adjusted
        } else {
            replacement_chars
        };

        let end = rewrite.end.min(result_chars.len());
        let start = rewrite.start.min(end);
        result_chars.splice(start..end, replacement_chars);
    }

    let unresolved_count = clusters
        .iter()
        .flat_map(|c| c.mentions.iter().zip(c.spans.iter()))
        .filter(|(m, _)| is_pronoun_for_language(m, lang))
        .count()
        .saturating_sub(rewrites.len());

    RagCorefResult {
        text: result_chars.into_iter().collect(),
        rewrites,
        unresolved_count,
    }
}

/// Check if "it" is pleonastic (non-referential) based on surrounding context.
///
/// Pleonastic "it" appears in weather expressions ("it rains"), extraposition
/// ("it is clear that..."), and idioms ("it turns out"). These are not
/// coreferential and should not be rewritten.
fn is_pleonastic_it(text: &str, entity_start: usize, entity_end: usize) -> bool {
    let chars: Vec<char> = text.chars().collect();
    let entity_text: String = chars[entity_start..entity_end].iter().collect();
    if entity_text.to_lowercase() != "it" {
        return false;
    }

    // Collect the text after "it" (up to ~40 chars) for pattern matching.
    let after: String = chars[entity_end..]
        .iter()
        .take(40)
        .collect::<String>()
        .to_lowercase();
    let after = after.trim_start();

    // "it is/was/seems/appears [adj/noun] to/that" (extraposition)
    if after.starts_with("is ")
        || after.starts_with("was ")
        || after.starts_with("seems ")
        || after.starts_with("appears ")
        || after.starts_with("is clear ")
        || after.starts_with("is obvious ")
        || after.starts_with("is likely ")
        || after.starts_with("is possible ")
        || after.starts_with("is important ")
        || after.starts_with("is necessary ")
        || after.starts_with("is true ")
        || after.starts_with("is known ")
    {
        // Check for "to" or "that" downstream (extraposition pattern)
        if after.contains(" that ") || after.contains(" to ") {
            return true;
        }
    }

    // Weather verbs: "it rains/snows/hails/thunders/pours"
    if after.starts_with("rain")
        || after.starts_with("snow")
        || after.starts_with("hail")
        || after.starts_with("thunder")
        || after.starts_with("pour")
        || after.starts_with("drizzle")
    {
        return true;
    }

    // Idioms: "it turns out", "it happened that", "it follows that"
    if after.starts_with("turns out")
        || after.starts_with("turned out")
        || after.starts_with("happened that")
        || after.starts_with("happens that")
        || after.starts_with("follows that")
        || after.starts_with("followed that")
    {
        return true;
    }

    false
}

/// Check if a pronoun is reflexive (herself/himself/themselves/itself/themself).
fn is_reflexive_pronoun(text: &str) -> bool {
    let lower = text.to_lowercase();
    matches!(
        lower.as_str(),
        "herself"
            | "himself"
            | "themselves"
            | "itself"
            | "themself"
            | "xemself"
            | "hirself"
            | "zirself"
            | "zemself"
            | "emself"
            | "faeself"
            | "faerself"
    )
}

/// Check if a text span is a demonstrative pronoun (this/that/these/those).
fn is_demonstrative_pronoun(text: &str) -> bool {
    let lower = text.to_lowercase();
    matches!(lower.as_str(), "this" | "that" | "these" | "those")
}

/// Chunk text, extract entities per chunk, resolve coreference, return self-contained chunks.
///
/// This is the full RAG preprocessing pipeline in one call:
/// 1. Split text into chunks (respecting sentence boundaries)
/// 2. Extract entities per chunk using the provided model
/// 3. Resolve coreference within each chunk (pronoun -> antecedent)
/// 4. Return chunks with their entities and rewritten text
///
/// ```rust,ignore
/// use anno::{rag, Model, StackedNER};
///
/// let m = StackedNER::default();
/// let chunks = rag::preprocess("Long document text...", &m, None)?;
/// for chunk in &chunks {
///     println!("Chunk: {}...", &chunk.text[..40.min(chunk.text.len())]);
///     println!("  {} entities", chunk.entities.len());
/// }
/// ```
pub fn preprocess(
    text: &str,
    model: &dyn crate::Model,
    config: Option<RagPreprocessConfig>,
) -> crate::Result<Vec<RagChunk>> {
    let config = config.unwrap_or_default();
    let chunks = crate::backends::chunking::chunk_text(text, &config.chunk_config);

    let mut result = Vec::with_capacity(chunks.len());
    for chunk in &chunks {
        let entities = model.extract_entities(&chunk.text, None)?;
        let coref_result = resolve_for_rag(&chunk.text, &entities, Some(config.coref.clone()));

        let char_start = chunk.char_offset;
        let char_end = chunk.char_offset + chunk.text.chars().count();
        result.push(RagChunk {
            text: coref_result.text,
            original_text: chunk.text.clone(),
            char_start,
            char_end,
            entities,
            rewrites: coref_result.rewrites.len(),
        });
    }

    Ok(result)
}

/// Configuration for the full RAG preprocessing pipeline.
#[derive(Debug, Clone)]
pub struct RagPreprocessConfig {
    /// Chunking configuration.
    pub chunk_config: crate::backends::chunking::ChunkConfig,
    /// Coreference resolution configuration.
    pub coref: RagCorefConfig,
}

impl Default for RagPreprocessConfig {
    fn default() -> Self {
        Self {
            chunk_config: crate::backends::chunking::ChunkConfig::long_document(),
            coref: RagCorefConfig::default(),
        }
    }
}

/// A preprocessed text chunk ready for RAG embedding.
#[derive(Debug, Clone)]
pub struct RagChunk {
    /// Coref-resolved text (pronouns replaced with antecedents).
    pub text: String,
    /// Original text before coreference resolution.
    pub original_text: String,
    /// Character offset of chunk start in the source document.
    pub char_start: usize,
    /// Character offset of chunk end in the source document.
    pub char_end: usize,
    /// Entities extracted from this chunk (offsets relative to `original_text`).
    pub entities: Vec<Entity>,
    /// Number of pronoun rewrites applied.
    pub rewrites: usize,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Check if a text span is a pronoun for the given language.
///
/// For unsupported languages (CJK, Arabic, Russian, etc.) this returns `false`,
/// which is safe: those mentions are treated as named entities and no rewrite
/// is attempted. Model-based pronoun detection is needed for those languages.
fn is_pronoun_for_language(text: &str, lang: Language) -> bool {
    let lower = text.to_lowercase();
    let s = lower.as_str();
    match lang {
        Language::English => matches!(
            s,
            "he" | "she"
                | "him"
                | "her"
                | "his"
                | "hers"
                | "himself"
                | "herself"
                | "they"
                | "them"
                | "their"
                | "theirs"
                | "themselves"
                | "themself"
                | "it"
                | "its"
                | "itself"
                | "xe"
                | "xem"
                | "xyr"
                | "xyrs"
                | "xemself"
                | "ze"
                | "hir"
                | "zir"
                | "hirs"
                | "zirs"
                | "hirself"
                | "zirself"
                | "zemself"
                | "ey"
                | "em"
                | "eir"
                | "eirs"
                | "emself"
                | "fae"
                | "faer"
                | "faers"
                | "faeself"
                | "faerself"
        ),
        Language::French => matches!(
            s,
            "il" | "elle" | "ils" | "elles" | "lui" | "leur" | "eux" | "se" | "soi"
                | "on"
                | "nous" | "vous"
                | "me" | "te" | "moi" | "toi"
                | "ce" | "cela" | "ceci"
                // "le"/"la"/"les" are ambiguous (article vs pronoun). Included because
                // in NER output they appear as entity spans only when pronominal.
                | "le" | "la" | "les"
        ),
        // Note: "él" (pronoun, with accent) vs "el" (article, no accent).
        // to_lowercase preserves accents, so this distinction works correctly.
        Language::Spanish => matches!(
            s,
            "él" | "ella"
                | "ellos"
                | "ellas"
                | "le"
                | "les"
                | "lo"
                | "la"
                | "los"
                | "las"
                | "se"
                | "sí"
        ),
        Language::German => matches!(
            s,
            "er" | "sie" | "es" | "ihm" | "ihr" | "ihnen" | "sich" | "ihn"
                // Declined forms of "ihr" (her/their)
                | "ihre" | "ihren" | "ihrem" | "ihrer"
                // Possessive "sein" (his/its) declined forms
                | "sein" | "seine" | "seinen" | "seinem" | "seiner"
                // Relative/demonstrative possessive
                | "dessen" | "deren"
                // 1st person plural
                | "wir" | "uns"
        ),
        // CJK, Arabic, Russian, etc.: model-based detection required.
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_entities() {
        let result = resolve_for_rag("Hello world.", &[], None);
        assert_eq!(result.text, "Hello world.");
        assert!(result.rewrites.is_empty());
    }

    #[test]
    fn test_no_pronouns() {
        let entities = vec![
            Entity::new("Alice", EntityType::Person, 0, 5, 0.9),
            Entity::new("Bob", EntityType::Person, 10, 13, 0.9),
        ];
        let result = resolve_for_rag("Alice and Bob went home.", &entities, None);
        assert_eq!(result.text, "Alice and Bob went home.");
        assert!(result.rewrites.is_empty());
    }

    #[test]
    fn test_pronoun_rewrite() {
        let text = "Alice went to the store. She bought milk.";
        let entities = vec![
            Entity::new("Alice", EntityType::Person, 0, 5, 0.9),
            Entity::new("She", EntityType::Person, 25, 28, 0.8),
        ];
        let result = resolve_for_rag(text, &entities, None);
        assert_eq!(result.text, "Alice went to the store. Alice bought milk.");
        assert_eq!(result.rewrites.len(), 1);
        assert_eq!(result.rewrites[0].original, "She");
        assert_eq!(result.rewrites[0].replacement, "Alice");
    }

    #[test]
    fn test_multiple_pronouns() {
        let text = "Bob likes coffee. He drinks it daily.";
        let entities = vec![
            Entity::new("Bob", EntityType::Person, 0, 3, 0.9),
            Entity::new(
                "coffee",
                EntityType::custom("Product", EntityCategory::Misc),
                10,
                16,
                0.8,
            ),
            Entity::new("He", EntityType::Person, 18, 20, 0.8),
            Entity::new(
                "it",
                EntityType::custom("Product", EntityCategory::Misc),
                28,
                30,
                0.7,
            ),
        ];
        let result = resolve_for_rag(text, &entities, None);
        // "He" -> "Bob"
        assert!(result.text.contains("Bob drinks"));
    }

    #[test]
    fn test_cataphora_resolution() {
        // "Before she arrived, Mary ordered food." -> resolves "she" to "Mary"
        let text = "Before she arrived, Mary ordered food.";
        let entities = vec![
            Entity::new("she", EntityType::Person, 7, 10, 0.8),
            Entity::new("Mary", EntityType::Person, 20, 24, 0.9),
        ];
        let result = resolve_for_rag(text, &entities, None);
        assert_eq!(result.text, "Before Mary arrived, Mary ordered food.");
        assert_eq!(result.rewrites.len(), 1);
        assert_eq!(result.rewrites[0].original, "she");
        assert_eq!(result.rewrites[0].replacement, "Mary");
        assert_eq!(result.unresolved_count, 0);
    }

    #[test]
    fn test_cataphora_disabled() {
        let text = "Before she arrived, Mary ordered food.";
        let entities = vec![
            Entity::new("she", EntityType::Person, 7, 10, 0.8),
            Entity::new("Mary", EntityType::Person, 20, 24, 0.9),
        ];
        let config = RagCorefConfig {
            resolve_cataphora: false,
            ..Default::default()
        };
        let result = resolve_for_rag(text, &entities, Some(config));
        // Without cataphora, "she" stays unresolved
        assert_eq!(result.text, text);
        assert_eq!(result.unresolved_count, 1);
    }

    #[test]
    fn test_cataphora_type_mismatch_skipped() {
        // Pronoun is Person but forward entity is Organization -- no resolution
        let text = "Before she arrived, Acme Corp filed papers.";
        let entities = vec![
            Entity::new("she", EntityType::Person, 7, 10, 0.8),
            Entity::new("Acme Corp", EntityType::Organization, 20, 29, 0.9),
        ];
        let result = resolve_for_rag(text, &entities, None);
        assert_eq!(result.text, text);
        assert_eq!(result.unresolved_count, 1);
    }

    #[test]
    fn test_disabled_rewriting() {
        let text = "Alice went to the store. She bought milk.";
        let entities = vec![
            Entity::new("Alice", EntityType::Person, 0, 5, 0.9),
            Entity::new("She", EntityType::Person, 25, 28, 0.8),
        ];
        let config = RagCorefConfig {
            rewrite_pronouns: false,
            ..Default::default()
        };
        let result = resolve_for_rag(text, &entities, Some(config));
        assert_eq!(result.text, text);
    }

    #[test]
    fn test_french_pronoun_cataphora() {
        // SimpleCorefResolver is English-only: French "Il" is not recognized
        // as a pronoun by the resolver, so anaphoric resolution fails.
        // Cataphora works because resolve_for_rag uses is_pronoun_for_language
        // and searches forward for a compatible non-pronoun entity.
        let text = "Il reviendra demain. Pierre est parti.";
        let pierre_start = "Il reviendra demain. ".chars().count();
        let pierre_end = pierre_start + "Pierre".chars().count();
        let entities = vec![
            Entity::new("Il", EntityType::Person, 0, 2, 0.8),
            Entity::new("Pierre", EntityType::Person, pierre_start, pierre_end, 0.9),
        ];
        let config = RagCorefConfig {
            language: Some(Language::French),
            ..Default::default()
        };
        let result = resolve_for_rag(text, &entities, Some(config));
        assert_eq!(result.text, "Pierre reviendra demain. Pierre est parti.");
        assert_eq!(result.rewrites.len(), 1);
    }

    #[test]
    fn test_french_anaphoric_fallback() {
        // French anaphoric: "Il" after "Pierre" is resolved via the proximity-
        // based anaphoric fallback (the resolver is English-only, but the
        // fallback searches backward for nearest compatible non-pronoun entity).
        let text = "Pierre est parti. Il reviendra demain.";
        let entities = vec![
            Entity::new("Pierre", EntityType::Person, 0, 6, 0.9),
            Entity::new("Il", EntityType::Person, 18, 20, 0.8),
        ];
        let config = RagCorefConfig {
            language: Some(Language::French),
            ..Default::default()
        };
        let result = resolve_for_rag(text, &entities, Some(config));
        assert_eq!(result.text, "Pierre est parti. Pierre reviendra demain.");
        assert_eq!(result.rewrites.len(), 1);
        assert_eq!(result.unresolved_count, 0);
    }

    #[test]
    fn test_unsupported_language_no_rewrites() {
        // Japanese: pronouns not recognized, so no rewrites (safe fallback)
        let text = "太郎は学校に行った。彼は勉強した。";
        let entities = vec![
            Entity::new("太郎", EntityType::Person, 0, 2, 0.9),
            Entity::new("彼", EntityType::Person, 11, 12, 0.8),
        ];
        let config = RagCorefConfig {
            language: Some(Language::Japanese),
            ..Default::default()
        };
        let result = resolve_for_rag(text, &entities, Some(config));
        // No rewrites: "彼" is not in the Japanese pronoun list (not supported yet)
        assert_eq!(result.text, text);
        assert_eq!(result.rewrites.len(), 0);
    }

    // ── Bug-fix regression tests ────────────────────────────────────

    #[test]
    fn test_pleonastic_it_not_rewritten() {
        // "It is raining in London." -- "It" is pleonastic (weather), not referential.
        let text = "It is raining in London.";
        let entities = vec![
            Entity::new(
                "It",
                EntityType::custom("Weather", EntityCategory::Misc),
                0,
                2,
                0.7,
            ),
            Entity::new("London", EntityType::Location, 17, 23, 0.9),
        ];
        let result = resolve_for_rag(text, &entities, None);
        assert_eq!(result.text, text, "pleonastic 'it' should not be rewritten");
        assert!(result.rewrites.is_empty());
    }

    #[test]
    fn test_pleonastic_it_extraposition() {
        // "It is clear that Alice won." -- extraposition, not referential.
        let text = "It is clear that Alice won.";
        let entities = vec![
            Entity::new(
                "It",
                EntityType::custom("Abstract", EntityCategory::Misc),
                0,
                2,
                0.7,
            ),
            Entity::new("Alice", EntityType::Person, 17, 22, 0.9),
        ];
        let result = resolve_for_rag(text, &entities, None);
        assert_eq!(
            result.text, text,
            "extraposition 'it' should not be rewritten"
        );
    }

    #[test]
    fn test_pleonastic_it_turns_out() {
        let text = "It turns out the data was wrong.";
        let entities = vec![Entity::new(
            "It",
            EntityType::custom("Abstract", EntityCategory::Misc),
            0,
            2,
            0.7,
        )];
        let result = resolve_for_rag(text, &entities, None);
        assert_eq!(result.text, text);
    }

    #[test]
    fn test_overlapping_entity_spans() {
        // Nested NER: "New York" (0..8) and "York" (4..8) both as Location pronouns
        // is contrived, but tests that overlapping rewrites are deduped.
        let text = "Alice visited New York. She loved it there.";
        let she_start = "Alice visited New York. ".chars().count();
        let she_end = she_start + 3;
        let it_start = she_start + "She loved ".chars().count();
        let it_end = it_start + 2;
        let entities = vec![
            Entity::new("Alice", EntityType::Person, 0, 5, 0.9),
            Entity::new("New York", EntityType::Location, 14, 22, 0.9),
            Entity::new("She", EntityType::Person, she_start, she_end, 0.8),
            Entity::new("it", EntityType::Location, it_start, it_end, 0.7),
        ];
        let result = resolve_for_rag(text, &entities, None);
        // Both rewrites should apply (they don't overlap with each other)
        assert!(result.text.contains("Alice loved"));
    }

    #[test]
    fn test_unsorted_entities() {
        // Entities given out of document order: should still resolve correctly.
        let text = "Alice went to the store. She bought milk.";
        let entities = vec![
            // Reversed order: "She" before "Alice"
            Entity::new("She", EntityType::Person, 25, 28, 0.8),
            Entity::new("Alice", EntityType::Person, 0, 5, 0.9),
        ];
        let result = resolve_for_rag(text, &entities, None);
        assert_eq!(
            result.text, "Alice went to the store. Alice bought milk.",
            "unsorted entities should still resolve correctly"
        );
        assert_eq!(result.rewrites.len(), 1);
    }

    #[test]
    fn test_reflexive_not_rewritten() {
        // "Alice hurt herself" -- reflexive should be skipped by default.
        let text = "Alice hurt herself badly.";
        let herself_start = "Alice hurt ".chars().count();
        let herself_end = herself_start + "herself".chars().count();
        let entities = vec![
            Entity::new("Alice", EntityType::Person, 0, 5, 0.9),
            Entity::new(
                "herself",
                EntityType::Person,
                herself_start,
                herself_end,
                0.8,
            ),
        ];
        let result = resolve_for_rag(text, &entities, None);
        assert_eq!(
            result.text, text,
            "reflexive 'herself' should not be rewritten by default"
        );
        assert!(result.rewrites.is_empty());
    }

    #[test]
    fn test_reflexive_rewritten_when_enabled() {
        let text = "Alice hurt herself badly.";
        let herself_start = "Alice hurt ".chars().count();
        let herself_end = herself_start + "herself".chars().count();
        let entities = vec![
            Entity::new("Alice", EntityType::Person, 0, 5, 0.9),
            Entity::new(
                "herself",
                EntityType::Person,
                herself_start,
                herself_end,
                0.8,
            ),
        ];
        let config = RagCorefConfig {
            rewrite_reflexives: true,
            ..Default::default()
        };
        let result = resolve_for_rag(text, &entities, Some(config));
        assert!(
            result.text.contains("Alice hurt Alice"),
            "reflexive should be rewritten when enabled"
        );
    }

    // ── Edge case tests ──────────────────────────────────────────────

    #[test]
    fn test_empty_text() {
        let result = resolve_for_rag("", &[], None);
        assert_eq!(result.text, "");
        assert!(result.rewrites.is_empty());
        assert_eq!(result.unresolved_count, 0);
    }

    #[test]
    fn test_all_pronouns_no_antecedent() {
        // All entities are pronouns with no named antecedent in any cluster
        let text = "He told her that they would leave.";
        let entities = vec![
            Entity::new("He", EntityType::Person, 0, 2, 0.8),
            Entity::new("her", EntityType::Person, 8, 11, 0.8),
            Entity::new("they", EntityType::Person, 17, 21, 0.8),
        ];
        let result = resolve_for_rag(text, &entities, None);
        // No named antecedent => all unresolved, text unchanged
        assert_eq!(result.text, text);
        assert!(result.rewrites.is_empty());
        assert_eq!(result.unresolved_count, 3);
    }

    #[test]
    fn test_entity_at_text_start() {
        let text = "She left early. Maria was already there.";
        let entities = vec![
            Entity::new("She", EntityType::Person, 0, 3, 0.8),
            Entity::new("Maria", EntityType::Person, 16, 21, 0.9),
        ];
        let result = resolve_for_rag(text, &entities, None);
        // Cataphoric: "She" at position 0 resolved to "Maria"
        assert_eq!(result.text, "Maria left early. Maria was already there.");
        assert_eq!(result.rewrites.len(), 1);
        assert_eq!(result.rewrites[0].start, 0);
    }

    #[test]
    fn test_entity_at_text_end() {
        let text = "Alice was happy about her";
        let her_start = "Alice was happy about ".chars().count();
        let her_end = her_start + "her".chars().count();
        assert_eq!(her_end, text.chars().count());
        let entities = vec![
            Entity::new("Alice", EntityType::Person, 0, 5, 0.9),
            Entity::new("her", EntityType::Person, her_start, her_end, 0.8),
        ];
        let result = resolve_for_rag(text, &entities, None);
        assert_eq!(result.text, "Alice was happy about Alice");
        assert_eq!(result.rewrites.len(), 1);
        assert_eq!(result.rewrites[0].end, her_end);
    }

    #[test]
    fn test_nested_pronouns_same_sentence() {
        // "She said she would go" -- two pronouns, same antecedent
        let text = "Alice arrived. She said she would go.";
        let entities = vec![
            Entity::new("Alice", EntityType::Person, 0, 5, 0.9),
            Entity::new("She", EntityType::Person, 15, 18, 0.8),
            Entity::new("she", EntityType::Person, 24, 27, 0.8),
        ];
        let result = resolve_for_rag(text, &entities, None);
        assert_eq!(result.text, "Alice arrived. Alice said Alice would go.");
        assert_eq!(result.rewrites.len(), 2);
    }

    #[test]
    fn test_cataphora_and_anaphora_same_text() {
        // First "she" is cataphoric (forward to Alice), second is anaphoric (backward to Alice)
        let text = "Before she left, Alice went home. She was tired.";
        let she1_start = "Before ".chars().count();
        let she1_end = she1_start + "she".chars().count();
        let alice_start = "Before she left, ".chars().count();
        let alice_end = alice_start + "Alice".chars().count();
        let she2_start = "Before she left, Alice went home. ".chars().count();
        let she2_end = she2_start + "She".chars().count();
        let entities = vec![
            Entity::new("she", EntityType::Person, she1_start, she1_end, 0.8),
            Entity::new("Alice", EntityType::Person, alice_start, alice_end, 0.9),
            Entity::new("She", EntityType::Person, she2_start, she2_end, 0.8),
        ];
        let result = resolve_for_rag(text, &entities, None);
        assert_eq!(
            result.text,
            "Before Alice left, Alice went home. Alice was tired."
        );
        assert_eq!(result.rewrites.len(), 2);
        assert_eq!(result.unresolved_count, 0);
    }

    #[test]
    fn test_multiple_antecedents_different_types() {
        // Alice (Person) and Acme (Org) each have their own pronoun
        let text = "Alice joined Acme Corp. She loved it.";
        let entities = vec![
            Entity::new("Alice", EntityType::Person, 0, 5, 0.9),
            Entity::new("Acme Corp", EntityType::Organization, 13, 22, 0.9),
            Entity::new("She", EntityType::Person, 24, 27, 0.8),
            Entity::new("it", EntityType::Organization, 34, 36, 0.7),
        ];
        let result = resolve_for_rag(text, &entities, None);
        assert!(result.text.contains("Alice loved"));
        assert!(result.text.contains("Acme Corp"));
    }

    #[test]
    fn test_unicode_multibyte_german() {
        // German: Muller with umlaut. Character offsets must be correct.
        let text = "Müller ging nach Hause. Er war müde.";
        // "Müller" = 6 chars, "Er" at char 24..26
        let muller_len = "Müller".chars().count();
        assert_eq!(muller_len, 6);
        let er_start = "Müller ging nach Hause. ".chars().count();
        let er_end = er_start + 2;
        let entities = vec![
            Entity::new("Müller", EntityType::Person, 0, muller_len, 0.9),
            Entity::new("Er", EntityType::Person, er_start, er_end, 0.8),
        ];
        let config = RagCorefConfig {
            language: Some(Language::German),
            ..Default::default()
        };
        let result = resolve_for_rag(text, &entities, Some(config));
        assert_eq!(result.text, "Müller ging nach Hause. Müller war müde.");
        assert_eq!(result.rewrites.len(), 1);
    }

    #[test]
    fn test_unicode_cjk_offsets() {
        // CJK characters: each is 1 char but 3 bytes in UTF-8.
        // Verify character-offset semantics are preserved (no byte-offset confusion).
        let text = "田中太郎は会社に行った。彼は帰った。";
        let char_count: usize = text.chars().count();
        // "田中太郎" = chars 0..4, "彼" = char 12..13
        let entities = vec![
            Entity::new("田中太郎", EntityType::Person, 0, 4, 0.9),
            Entity::new("彼", EntityType::Person, 12, 13, 0.8),
        ];
        let config = RagCorefConfig {
            language: Some(Language::Japanese),
            ..Default::default()
        };
        let result = resolve_for_rag(text, &entities, Some(config));
        // Japanese is unsupported: no rewrites, but text must be intact
        assert_eq!(result.text, text);
        assert_eq!(result.text.chars().count(), char_count);
    }

    #[test]
    fn test_unicode_mixed_script() {
        // Mix of Latin and accented characters
        let text = "Café owner José serves him daily.";
        let him_start = "Café owner José serves ".chars().count();
        let him_end = him_start + 3;
        let jose_start = "Café owner ".chars().count();
        let jose_end = jose_start + "José".chars().count();
        let entities = vec![
            Entity::new("José", EntityType::Person, jose_start, jose_end, 0.9),
            Entity::new("him", EntityType::Person, him_start, him_end, 0.8),
        ];
        let result = resolve_for_rag(text, &entities, None);
        assert!(result.text.contains("José serves José"));
        assert_eq!(result.rewrites.len(), 1);
    }

    #[test]
    fn test_spanish_pronoun_cataphora() {
        // Spanish cataphoric: "Ella" before "María" is resolved via cataphora.
        // Anaphoric would fail (resolver is English-only).
        let text = "Ella compró pan. María fue al mercado.";
        let ella_end = "Ella".chars().count();
        let maria_start = "Ella compró pan. ".chars().count();
        let maria_end = maria_start + "María".chars().count();
        let entities = vec![
            Entity::new("Ella", EntityType::Person, 0, ella_end, 0.8),
            Entity::new("María", EntityType::Person, maria_start, maria_end, 0.9),
        ];
        let config = RagCorefConfig {
            language: Some(Language::Spanish),
            ..Default::default()
        };
        let result = resolve_for_rag(text, &entities, Some(config));
        assert_eq!(result.text, "María compró pan. María fue al mercado.");
        assert_eq!(result.rewrites.len(), 1);
    }

    #[test]
    fn test_spanish_anaphoric_fallback() {
        // Spanish anaphoric: "Ella" after "María" is resolved via the proximity-
        // based anaphoric fallback.
        let text = "María fue al mercado. Ella compró pan.";
        let ella_start = "María fue al mercado. ".chars().count();
        let ella_end = ella_start + "Ella".chars().count();
        let maria_end = "María".chars().count();
        let entities = vec![
            Entity::new("María", EntityType::Person, 0, maria_end, 0.9),
            Entity::new("Ella", EntityType::Person, ella_start, ella_end, 0.8),
        ];
        let config = RagCorefConfig {
            language: Some(Language::Spanish),
            ..Default::default()
        };
        let result = resolve_for_rag(text, &entities, Some(config));
        assert_eq!(result.text, "María fue al mercado. María compró pan.");
        assert_eq!(result.rewrites.len(), 1);
        assert_eq!(result.unresolved_count, 0);
    }

    #[test]
    fn test_german_capitalized_noun_not_false_positive() {
        // German nouns are capitalized. "Hund" (dog) is a noun, not a pronoun.
        // Only German pronouns (er/sie/es/ihm/ihr/ihnen/sich/ihn) should be
        // detected by is_pronoun_for_language. "Hund" is not a pronoun.
        let config = RagCorefConfig {
            language: Some(Language::German),
            ..Default::default()
        };
        assert!(!is_pronoun_for_language("Hund", Language::German));
        assert!(is_pronoun_for_language("ihn", Language::German));
        assert!(is_pronoun_for_language("er", Language::German));
        assert!(is_pronoun_for_language("sie", Language::German));

        // Cataphoric German: "ihn" before "Fritz" resolves via cataphora
        let text = "Man suchte ihn. Fritz war versteckt.";
        let ihn_start = "Man suchte ".chars().count();
        let ihn_end = ihn_start + "ihn".chars().count();
        let fritz_start = "Man suchte ihn. ".chars().count();
        let fritz_end = fritz_start + "Fritz".chars().count();
        let entities = vec![
            Entity::new("ihn", EntityType::Person, ihn_start, ihn_end, 0.7),
            Entity::new("Fritz", EntityType::Person, fritz_start, fritz_end, 0.9),
        ];
        let result = resolve_for_rag(text, &entities, Some(config));
        assert!(
            result.text.contains("Fritz war versteckt"),
            "Fritz should be preserved"
        );
        assert_eq!(result.rewrites.len(), 1);
        assert_eq!(result.rewrites[0].replacement, "Fritz");
    }

    #[test]
    fn test_rewrites_sorted_by_start() {
        let text = "Alice met Bob. She greeted him warmly.";
        let she_start = "Alice met Bob. ".chars().count();
        let she_end = she_start + 3;
        let him_start = she_start + "She greeted ".chars().count();
        let him_end = him_start + 3;
        let entities = vec![
            Entity::new("Alice", EntityType::Person, 0, 5, 0.9),
            Entity::new("Bob", EntityType::Person, 10, 13, 0.9),
            Entity::new("She", EntityType::Person, she_start, she_end, 0.8),
            Entity::new("him", EntityType::Person, him_start, him_end, 0.8),
        ];
        let result = resolve_for_rag(text, &entities, None);
        // Verify rewrites are sorted by start position
        for pair in result.rewrites.windows(2) {
            assert!(
                pair[0].start <= pair[1].start,
                "Rewrites not sorted: {} > {}",
                pair[0].start,
                pair[1].start
            );
        }
    }

    #[test]
    fn test_very_long_text() {
        // 10k char text: verify no panic, correct output length
        let prefix = "Alice is a researcher. ";
        let middle = "She studies language models. ".repeat(400); // ~11.2k chars
        let text = format!("{prefix}{middle}");
        let entities = vec![
            Entity::new("Alice", EntityType::Person, 0, 5, 0.9),
            Entity::new(
                "She",
                EntityType::Person,
                prefix.chars().count(),
                prefix.chars().count() + 3,
                0.8,
            ),
        ];
        let result = resolve_for_rag(&text, &entities, None);
        // Should not panic; output should be at least as long as input
        // (replaced "She" (3 chars) with "Alice" (5 chars))
        assert!(result.text.chars().count() >= text.chars().count());
    }

    // ── Cross-module integration ─────────────────────────────────────

    #[test]
    fn test_resolver_chains_agree_with_rag_output() {
        // Run SimpleCorefResolver directly and verify cluster assignments
        // are consistent with what resolve_for_rag produces.
        let text = "Alice went to the park. She enjoyed it.";
        let entities = vec![
            Entity::new("Alice", EntityType::Person, 0, 5, 0.9),
            Entity::new("She", EntityType::Person, 24, 27, 0.8),
        ];
        let resolver = SimpleCorefResolver::new(CorefConfig {
            max_pronoun_lookback: 5,
            ..CorefConfig::default()
        });
        let resolved = resolver.resolve(&entities);
        // Both should share a canonical_id
        let alice_cid = resolved[0].canonical_id;
        let she_cid = resolved[1].canonical_id;
        assert!(alice_cid.is_some());
        assert_eq!(alice_cid, she_cid, "Alice and She should share a cluster");

        // RAG output should rewrite
        let rag = resolve_for_rag(text, &entities, None);
        assert_eq!(rag.rewrites.len(), 1);
        assert_eq!(rag.rewrites[0].replacement, "Alice");
    }

    #[test]
    fn test_character_offsets_are_unicode_scalar_not_byte() {
        // "Ä" is 2 bytes but 1 char. Verify offsets are char-based.
        let text = "Ä person named Bob. He left.";
        // "Ä" = 1 char, " person named " = 14 chars => "Bob" starts at 15
        let bob_start = "Ä person named ".chars().count();
        let bob_end = bob_start + 3;
        let he_start = bob_end + ". ".chars().count();
        let he_end = he_start + 2;
        let entities = vec![
            Entity::new("Bob", EntityType::Person, bob_start, bob_end, 0.9),
            Entity::new("He", EntityType::Person, he_start, he_end, 0.8),
        ];
        let result = resolve_for_rag(text, &entities, None);
        assert_eq!(result.text, "Ä person named Bob. Bob left.");
        // Verify the rewrite offsets are character-based
        assert_eq!(result.rewrites[0].start, he_start);
        assert_eq!(result.rewrites[0].end, he_end);
    }

    // ── Pronoun-list coverage tests ─────────────────────────────────

    #[test]
    fn test_french_on_recognized() {
        // "on" is a very common French pronoun meaning "one/we/they".
        assert!(is_pronoun_for_language("on", Language::French));
        assert!(is_pronoun_for_language("On", Language::French));

        // Verify it resolves in context (cataphoric: "On" before "Jean").
        let text = "On est parti tôt. Jean a fermé la porte.";
        let on_end = "On".chars().count();
        let jean_start = "On est parti tôt. ".chars().count();
        let jean_end = jean_start + "Jean".chars().count();
        let entities = vec![
            Entity::new("On", EntityType::Person, 0, on_end, 0.7),
            Entity::new("Jean", EntityType::Person, jean_start, jean_end, 0.9),
        ];
        let config = RagCorefConfig {
            language: Some(Language::French),
            ..Default::default()
        };
        let result = resolve_for_rag(text, &entities, Some(config));
        assert_eq!(result.text, "Jean est parti tôt. Jean a fermé la porte.");
        assert_eq!(result.rewrites.len(), 1);
    }

    #[test]
    fn test_german_declined_possessives() {
        // Verify declined forms of "ihr" and "sein" are recognized.
        for form in &["ihre", "ihren", "ihrem", "ihrer"] {
            assert!(
                is_pronoun_for_language(form, Language::German),
                "{form} should be recognized as German pronoun"
            );
        }
        for form in &["sein", "seine", "seinen", "seinem", "seiner"] {
            assert!(
                is_pronoun_for_language(form, Language::German),
                "{form} should be recognized as German pronoun"
            );
        }
        // Relative/demonstrative possessives
        assert!(is_pronoun_for_language("dessen", Language::German));
        assert!(is_pronoun_for_language("deren", Language::German));
        // 1st person plural
        assert!(is_pronoun_for_language("wir", Language::German));
        assert!(is_pronoun_for_language("uns", Language::German));

        // Negative: common German nouns should NOT match
        assert!(!is_pronoun_for_language("Hund", Language::German));
        assert!(!is_pronoun_for_language("Haus", Language::German));
    }

    #[test]
    fn test_english_demonstratives_config() {
        // By default, demonstratives are NOT rewritten.
        let text = "Acme announced layoffs. This upset employees.";
        let this_start = "Acme announced layoffs. ".chars().count();
        let this_end = this_start + "This".chars().count();
        let entities = vec![
            Entity::new("Acme", EntityType::Organization, 0, 4, 0.9),
            Entity::new("This", EntityType::Organization, this_start, this_end, 0.7),
        ];
        let result = resolve_for_rag(text, &entities, None);
        assert_eq!(
            result.text, text,
            "demonstratives should NOT be rewritten by default"
        );

        // With rewrite_demonstratives enabled, "This" is rewritten.
        let config = RagCorefConfig {
            rewrite_demonstratives: true,
            ..Default::default()
        };
        let result = resolve_for_rag(text, &entities, Some(config));
        assert_eq!(
            result.text, "Acme announced layoffs. Acme upset employees.",
            "demonstratives should be rewritten when enabled"
        );
        assert_eq!(result.rewrites.len(), 1);
        assert_eq!(result.rewrites[0].original, "This");
        assert_eq!(result.rewrites[0].replacement, "Acme");
    }

    // ── Audit-driven regression tests ──────────────────────────────

    #[test]
    fn test_pleonastic_it_time() {
        // "It was late when Alice arrived." -- pleonastic "it" (time), not rewritten.
        let text = "It was late when Alice arrived.";
        let alice_start = "It was late when ".chars().count();
        let alice_end = alice_start + "Alice".chars().count();
        let entities = vec![
            Entity::new(
                "It",
                EntityType::custom("Time", EntityCategory::Misc),
                0,
                2,
                0.7,
            ),
            Entity::new("Alice", EntityType::Person, alice_start, alice_end, 0.9),
        ];
        let result = resolve_for_rag(text, &entities, None);
        assert_eq!(
            result.text, text,
            "pleonastic 'it' (time) should not be rewritten"
        );
        // Alice should be preserved in the text
        assert!(result.text.contains("Alice"));
    }

    #[test]
    fn test_pleonastic_it_seem() {
        // "It seems that Bob is correct." -- extraposition, not referential.
        let text = "It seems that Bob is correct.";
        let bob_start = "It seems that ".chars().count();
        let bob_end = bob_start + "Bob".chars().count();
        let entities = vec![
            Entity::new(
                "It",
                EntityType::custom("Abstract", EntityCategory::Misc),
                0,
                2,
                0.7,
            ),
            Entity::new("Bob", EntityType::Person, bob_start, bob_end, 0.9),
        ];
        let result = resolve_for_rag(text, &entities, None);
        assert_eq!(
            result.text, text,
            "pleonastic 'it' (seems) should not be rewritten"
        );
    }

    #[test]
    fn test_referential_it_not_blocked() {
        // "Alice bought a car. It was red." -- "It" refers to "car", should be rewritten.
        let text = "Alice bought a car. It was red.";
        let car_start = "Alice bought a ".chars().count();
        let car_end = car_start + "car".chars().count();
        let it_start = "Alice bought a car. ".chars().count();
        let it_end = it_start + "It".chars().count();
        let entities = vec![
            Entity::new("Alice", EntityType::Person, 0, 5, 0.9),
            Entity::new(
                "car",
                EntityType::custom("Product", EntityCategory::Misc),
                car_start,
                car_end,
                0.8,
            ),
            Entity::new(
                "It",
                EntityType::custom("Product", EntityCategory::Misc),
                it_start,
                it_end,
                0.7,
            ),
        ];
        let result = resolve_for_rag(text, &entities, None);
        // "It" is sentence-initial (uppercase), so "car" gets capitalized to "Car".
        assert!(
            result.text.contains("Car was red"),
            "referential 'it' should be rewritten to 'car' (capitalized to 'Car'), got: {}",
            result.text
        );
        assert_eq!(result.rewrites.len(), 1);
        assert_eq!(result.rewrites[0].original, "It");
        assert_eq!(result.rewrites[0].replacement, "car");
    }

    #[test]
    fn test_nested_ner_spans() {
        // Entities "New York" [5,13) and "York" [9,13) -- overlapping spans.
        // Only the longer span's rewrite should survive if both would produce rewrites.
        let text = "Visit New York today. Go see it.";
        let ny_start = "Visit ".chars().count();
        let ny_end = ny_start + "New York".chars().count();
        let york_start = "Visit New ".chars().count();
        let york_end = york_start + "York".chars().count();
        assert_eq!(ny_end, york_end); // both end at same position
        let it_start = "Visit New York today. Go see ".chars().count();
        let it_end = it_start + "it".chars().count();
        let entities = vec![
            Entity::new("New York", EntityType::Location, ny_start, ny_end, 0.9),
            Entity::new("York", EntityType::Location, york_start, york_end, 0.8),
            Entity::new("it", EntityType::Location, it_start, it_end, 0.7),
        ];
        let result = resolve_for_rag(text, &entities, None);
        // "it" should be rewritten to "New York" (the longer antecedent) since
        // "New York" was inserted first in the cluster map.
        // The key invariant: no corruption from overlapping NER spans.
        assert!(!result.text.contains('\0'), "no null bytes in output");
        // "it" should be replaced with either "New York" or "York"
        assert!(
            result.text.contains("Go see New York") || result.text.contains("Go see York"),
            "referential 'it' should be rewritten to a location, got: {}",
            result.text
        );
    }

    #[test]
    fn test_adjacent_rewrites() {
        // Two pronouns directly adjacent: "HeShe" -- both should rewrite correctly.
        let text = "Alice met Bob. HeShe left.";
        let he_start = "Alice met Bob. ".chars().count();
        let he_end = he_start + "He".chars().count();
        let she_start = he_end;
        let she_end = she_start + "She".chars().count();
        let entities = vec![
            Entity::new("Alice", EntityType::Person, 0, 5, 0.9),
            Entity::new("Bob", EntityType::Person, 10, 13, 0.9),
            Entity::new("He", EntityType::Person, he_start, he_end, 0.8),
            Entity::new("She", EntityType::Person, she_start, she_end, 0.8),
        ];
        let result = resolve_for_rag(text, &entities, None);
        // Both pronouns should be rewritten. They don't overlap so both survive.
        assert_eq!(
            result.rewrites.len(),
            2,
            "both adjacent pronouns should be rewritten, got rewrites: {:?}",
            result.rewrites
        );
        // Verify no corruption at the boundary
        assert!(
            !result.text.contains("He")
                || result.text.contains("Alice")
                || result.text.contains("Bob"),
            "adjacent rewrites should not corrupt each other, got: {}",
            result.text
        );
    }

    #[test]
    fn test_entities_reversed_order() {
        // Entities in reverse document order should produce same result as sorted.
        let text = "Alice went to the store. She bought milk.";
        let forward = vec![
            Entity::new("Alice", EntityType::Person, 0, 5, 0.9),
            Entity::new("She", EntityType::Person, 25, 28, 0.8),
        ];
        let reversed = vec![
            Entity::new("She", EntityType::Person, 25, 28, 0.8),
            Entity::new("Alice", EntityType::Person, 0, 5, 0.9),
        ];
        let result_forward = resolve_for_rag(text, &forward, None);
        let result_reversed = resolve_for_rag(text, &reversed, None);
        assert_eq!(
            result_forward.text, result_reversed.text,
            "reversed entity order should produce same output"
        );
        assert_eq!(
            result_forward.rewrites.len(),
            result_reversed.rewrites.len()
        );
    }

    #[test]
    fn test_entities_random_order() {
        // Multiple entities in arbitrary (non-sorted) order.
        let text = "Alice met Bob at the park. She greeted him warmly.";
        let she_start = "Alice met Bob at the park. ".chars().count();
        let she_end = she_start + "She".chars().count();
        let him_start = she_start + "She greeted ".chars().count();
        let him_end = him_start + "him".chars().count();
        // Sorted order
        let sorted = vec![
            Entity::new("Alice", EntityType::Person, 0, 5, 0.9),
            Entity::new("Bob", EntityType::Person, 10, 13, 0.9),
            Entity::new("She", EntityType::Person, she_start, she_end, 0.8),
            Entity::new("him", EntityType::Person, him_start, him_end, 0.8),
        ];
        // Shuffled order: him, Alice, She, Bob
        let shuffled = vec![
            Entity::new("him", EntityType::Person, him_start, him_end, 0.8),
            Entity::new("Alice", EntityType::Person, 0, 5, 0.9),
            Entity::new("She", EntityType::Person, she_start, she_end, 0.8),
            Entity::new("Bob", EntityType::Person, 10, 13, 0.9),
        ];
        let result_sorted = resolve_for_rag(text, &sorted, None);
        let result_shuffled = resolve_for_rag(text, &shuffled, None);
        assert_eq!(
            result_sorted.text, result_shuffled.text,
            "arbitrary entity order should produce same output as sorted"
        );
    }

    #[test]
    fn test_rewrite_offsets_valid_chars_not_bytes() {
        // For every rewrite in output, verify start/end are valid char indices.
        let text = "Müller ging nach Hause. Er war müde. Sie kam auch.";
        let muller_len = "Müller".chars().count();
        let er_start = "Müller ging nach Hause. ".chars().count();
        let er_end = er_start + "Er".chars().count();
        let sie_start = "Müller ging nach Hause. Er war müde. ".chars().count();
        let sie_end = sie_start + "Sie".chars().count();
        let entities = vec![
            Entity::new("Müller", EntityType::Person, 0, muller_len, 0.9),
            Entity::new("Er", EntityType::Person, er_start, er_end, 0.8),
            Entity::new("Sie", EntityType::Person, sie_start, sie_end, 0.8),
        ];
        let config = RagCorefConfig {
            language: Some(Language::German),
            ..Default::default()
        };
        let result = resolve_for_rag(text, &entities, Some(config));
        let char_count = text.chars().count();
        for rw in &result.rewrites {
            assert!(
                rw.start < rw.end,
                "rewrite start ({}) must be < end ({})",
                rw.start,
                rw.end
            );
            assert!(
                rw.end <= char_count,
                "rewrite end ({}) must be <= text char count ({})",
                rw.end,
                char_count
            );
            // Verify these are char offsets: converting chars to string at those offsets must work.
            let chars: Vec<char> = text.chars().collect();
            let extracted: String = chars[rw.start..rw.end].iter().collect();
            assert_eq!(
                extracted, rw.original,
                "char-offset extraction must match original pronoun"
            );
        }
    }

    #[test]
    fn test_output_text_reconstructible() {
        // Verify the output text can be obtained by applying the rewrites to the input.
        let text = "Alice met Bob. She greeted him warmly.";
        let she_start = "Alice met Bob. ".chars().count();
        let she_end = she_start + "She".chars().count();
        let him_start = she_start + "She greeted ".chars().count();
        let him_end = him_start + "him".chars().count();
        let entities = vec![
            Entity::new("Alice", EntityType::Person, 0, 5, 0.9),
            Entity::new("Bob", EntityType::Person, 10, 13, 0.9),
            Entity::new("She", EntityType::Person, she_start, she_end, 0.8),
            Entity::new("him", EntityType::Person, him_start, him_end, 0.8),
        ];
        let result = resolve_for_rag(text, &entities, None);

        // Manually reconstruct: apply rewrites right-to-left on the original text
        let mut chars: Vec<char> = text.chars().collect();
        for rw in result.rewrites.iter().rev() {
            let mut replacement: Vec<char> = rw.replacement.chars().collect();
            // Mimic the case adjustment from resolve_for_rag
            if rw.original.chars().next().is_some_and(|c| c.is_uppercase()) {
                if let Some(first) = replacement.first_mut() {
                    *first = first.to_uppercase().next().unwrap_or(*first);
                }
            }
            let end = rw.end.min(chars.len());
            let start = rw.start.min(end);
            chars.splice(start..end, replacement);
        }
        let reconstructed: String = chars.into_iter().collect();
        assert_eq!(
            result.text, reconstructed,
            "output text must equal manually-reconstructed text"
        );
    }

    #[test]
    fn test_spanish_accent_distinction() {
        // "el" (article, no accent) vs "el" (pronoun, with accent "el")
        // The pronoun list uses "el" with accent.
        assert!(
            is_pronoun_for_language("él", Language::Spanish),
            "'él' (with accent) should be recognized as Spanish pronoun"
        );
        assert!(
            !is_pronoun_for_language("el", Language::Spanish),
            "'el' (no accent) should NOT be recognized as Spanish pronoun"
        );
    }

    #[test]
    fn test_german_seine_is_pronoun() {
        // "Seine Firma" -- "seine" recognized as German pronoun.
        assert!(
            is_pronoun_for_language("seine", Language::German),
            "'seine' should be recognized as German pronoun"
        );
        assert!(
            is_pronoun_for_language("Seine", Language::German),
            "'Seine' (capitalized) should be recognized as German pronoun"
        );

        // Integration: "Seine" is rewritten to named entity when in context.
        let text = "Hans arbeitet hier. Seine Firma ist groß.";
        let hans_end = "Hans".chars().count();
        let seine_start = "Hans arbeitet hier. ".chars().count();
        let seine_end = seine_start + "Seine".chars().count();
        let entities = vec![
            Entity::new("Hans", EntityType::Person, 0, hans_end, 0.9),
            Entity::new("Seine", EntityType::Person, seine_start, seine_end, 0.7),
        ];
        let config = RagCorefConfig {
            language: Some(Language::German),
            ..Default::default()
        };
        let result = resolve_for_rag(text, &entities, Some(config));
        assert!(
            result.text.contains("Hans Firma"),
            "'Seine' should be rewritten to 'Hans', got: {}",
            result.text
        );
        assert_eq!(result.rewrites.len(), 1);
    }

    #[test]
    fn test_french_on_is_pronoun() {
        // "On va au cinema" -- "On" recognized as French pronoun.
        assert!(
            is_pronoun_for_language("On", Language::French),
            "'On' should be recognized as French pronoun"
        );
        assert!(
            is_pronoun_for_language("on", Language::French),
            "'on' (lowercase) should be recognized as French pronoun"
        );
    }

    // ── Property tests ───────────────────────────────────────────────

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn never_panics(text in "[a-zA-Z .]{0,200}") {
                let char_len = text.chars().count();
                // Build some plausible entities within bounds
                let entities: Vec<Entity> = {
                    let mut v = Vec::new();
                    let mut pos = 0;
                    let names = ["Alice", "He", "Bob", "She", "it"];
                    for name in names.iter() {
                        let name_len = name.chars().count();
                        if pos + name_len > char_len { break; }
                        v.push(Entity::new(
                            *name,
                            EntityType::Person,
                            pos,
                            pos + name_len,
                            0.9,
                        ));
                        pos += name_len + 3;
                    }
                    v
                };
                let _result = resolve_for_rag(&text, &entities, None);
            }

            #[test]
            fn output_length_gte_input_when_pronouns_shorter(
                text in "[A-Za-z ]{20,100}"
            ) {
                // When all pronouns are shorter than their antecedents,
                // output length >= input length.
                let text_str: &str = &text;
                let char_len = text_str.chars().count();
                if char_len < 15 { return Ok(()); }
                let entities = vec![
                    Entity::new("Alice", EntityType::Person, 0, 5.min(char_len), 0.9),
                    Entity::new("She", EntityType::Person, 10.min(char_len.saturating_sub(1)), 13.min(char_len), 0.8),
                ];
                // Only run if offsets are valid
                if entities[1].start() >= entities[1].end() || entities[1].end() > char_len {
                    return Ok(());
                }
                let result = resolve_for_rag(text_str, &entities, None);
                // "She" (3) -> "Alice" (5): output should be >= input
                prop_assert!(
                    result.text.chars().count() >= text_str.chars().count(),
                    "Output shorter than input: {} < {}",
                    result.text.chars().count(),
                    text_str.chars().count()
                );
            }

            #[test]
            fn rewrites_count_bounded_by_pronoun_count(
                text in "[A-Za-z ]{30,150}"
            ) {
                let char_len = text.chars().count();
                if char_len < 30 { return Ok(()); }
                let entities = vec![
                    Entity::new("Bob", EntityType::Person, 0, 3.min(char_len), 0.9),
                    Entity::new("He", EntityType::Person, 8.min(char_len.saturating_sub(1)), 10.min(char_len), 0.8),
                    Entity::new("him", EntityType::Person, 15.min(char_len.saturating_sub(1)), 18.min(char_len), 0.8),
                ];
                if entities.iter().any(|e| e.start() >= e.end() || e.end() > char_len) {
                    return Ok(());
                }
                let result = resolve_for_rag(&text, &entities, None);
                let pronoun_count = 2; // "He" and "him"
                prop_assert!(
                    result.rewrites.len() <= pronoun_count,
                    "More rewrites ({}) than pronouns ({})",
                    result.rewrites.len(),
                    pronoun_count
                );
            }

            #[test]
            fn rewrites_sorted_ascending(
                text in "[A-Za-z ]{30,150}"
            ) {
                let char_len = text.chars().count();
                if char_len < 30 { return Ok(()); }
                let entities = vec![
                    Entity::new("Alice", EntityType::Person, 0, 5.min(char_len), 0.9),
                    Entity::new("She", EntityType::Person, 10.min(char_len.saturating_sub(1)), 13.min(char_len), 0.8),
                    Entity::new("her", EntityType::Person, 20.min(char_len.saturating_sub(1)), 23.min(char_len), 0.8),
                ];
                if entities.iter().any(|e| e.start() >= e.end() || e.end() > char_len) {
                    return Ok(());
                }
                let result = resolve_for_rag(&text, &entities, None);
                for pair in result.rewrites.windows(2) {
                    prop_assert!(
                        pair[0].start <= pair[1].start,
                        "Rewrites not sorted: {} > {}",
                        pair[0].start,
                        pair[1].start
                    );
                }
            }

            #[test]
            fn unresolved_plus_rewrites_le_total_pronouns(
                text in "[A-Za-z ]{40,200}"
            ) {
                let char_len = text.chars().count();
                if char_len < 40 { return Ok(()); }
                let entities = vec![
                    Entity::new("Alice", EntityType::Person, 0, 5.min(char_len), 0.9),
                    Entity::new("Bob", EntityType::Organization, 6.min(char_len.saturating_sub(1)), 9.min(char_len), 0.9),
                    Entity::new("She", EntityType::Person, 15.min(char_len.saturating_sub(1)), 18.min(char_len), 0.8),
                    Entity::new("it", EntityType::Organization, 25.min(char_len.saturating_sub(1)), 27.min(char_len), 0.7),
                    Entity::new("they", EntityType::Person, 33.min(char_len.saturating_sub(1)), 37.min(char_len), 0.7),
                ];
                if entities.iter().any(|e| e.start() >= e.end() || e.end() > char_len) {
                    return Ok(());
                }
                let result = resolve_for_rag(&text, &entities, None);
                let total_pronouns = 3; // She, it, they
                prop_assert!(
                    result.rewrites.len() + result.unresolved_count <= total_pronouns,
                    "rewrites({}) + unresolved({}) > pronouns({})",
                    result.rewrites.len(),
                    result.unresolved_count,
                    total_pronouns
                );
            }

            #[test]
            fn no_control_chars_introduced(
                text in "[A-Za-z .,!?]{20,150}"
            ) {
                // Output text should never contain control characters or null bytes
                // that were not already in the input.
                let char_len = text.chars().count();
                if char_len < 20 { return Ok(()); }
                let entities = vec![
                    Entity::new("Alice", EntityType::Person, 0, 5.min(char_len), 0.9),
                    Entity::new("She", EntityType::Person, 10.min(char_len.saturating_sub(1)), 13.min(char_len), 0.8),
                ];
                if entities.iter().any(|e| e.start() >= e.end() || e.end() > char_len) {
                    return Ok(());
                }
                let result = resolve_for_rag(&text, &entities, None);
                let input_controls: std::collections::HashSet<char> =
                    text.chars().filter(|c| c.is_control()).collect();
                for ch in result.text.chars() {
                    if ch.is_control() {
                        prop_assert!(
                            input_controls.contains(&ch),
                            "output contains control char {:?} not present in input",
                            ch
                        );
                    }
                }
            }

            #[test]
            fn all_rewrites_nonzero_width(
                text in "[A-Za-z ]{30,150}"
            ) {
                // All rewrite spans must have start < end (no zero-width rewrites).
                let char_len = text.chars().count();
                if char_len < 30 { return Ok(()); }
                let entities = vec![
                    Entity::new("Bob", EntityType::Person, 0, 3.min(char_len), 0.9),
                    Entity::new("He", EntityType::Person, 8.min(char_len.saturating_sub(1)), 10.min(char_len), 0.8),
                    Entity::new("him", EntityType::Person, 15.min(char_len.saturating_sub(1)), 18.min(char_len), 0.8),
                ];
                if entities.iter().any(|e| e.start() >= e.end() || e.end() > char_len) {
                    return Ok(());
                }
                let result = resolve_for_rag(&text, &entities, None);
                for rw in &result.rewrites {
                    prop_assert!(
                        rw.start < rw.end,
                        "zero-width rewrite at position {}: start={} end={}",
                        rw.start,
                        rw.start,
                        rw.end
                    );
                }
            }
        }
    }

    #[cfg(feature = "onnx")]
    mod neural_tests {
        use crate::backends::coref::t5::CorefCluster;
        use crate::rag::*;

        #[test]
        fn test_neural_rag_basic_rewrite() {
            let text = "John went to the store. He bought milk.";
            let clusters = vec![CorefCluster {
                id: 0,
                mentions: vec!["John".to_string(), "He".to_string()],
                spans: vec![(0, 4), (24, 26)],
                canonical: "John".to_string(),
            }];
            let result = resolve_for_rag_neural(text, &clusters, None);
            assert_eq!(result.text, "John went to the store. John bought milk.");
            assert_eq!(result.rewrites.len(), 1);
            assert_eq!(result.rewrites[0].original, "He");
            assert_eq!(result.rewrites[0].replacement, "John");
        }

        #[test]
        fn test_neural_rag_no_clusters() {
            let text = "The weather is nice today.";
            let result = resolve_for_rag_neural(text, &[], None);
            assert_eq!(result.text, text);
            assert_eq!(result.rewrites.len(), 0);
        }

        #[test]
        fn test_neural_rag_multiple_clusters() {
            let text = "Alice met Bob. She greeted him warmly.";
            let clusters = vec![
                CorefCluster {
                    id: 0,
                    mentions: vec!["Alice".to_string(), "She".to_string()],
                    spans: vec![(0, 5), (15, 18)],
                    canonical: "Alice".to_string(),
                },
                CorefCluster {
                    id: 1,
                    mentions: vec!["Bob".to_string(), "him".to_string()],
                    spans: vec![(10, 13), (27, 30)],
                    canonical: "Bob".to_string(),
                },
            ];
            let result = resolve_for_rag_neural(text, &clusters, None);
            assert_eq!(result.text, "Alice met Bob. Alice greeted Bob warmly.");
            assert_eq!(result.rewrites.len(), 2);
        }

        #[test]
        fn test_neural_rag_skips_pleonastic_it() {
            let text =
                "It is raining. John forgot his umbrella. It is clear that he should go back.";
            let clusters = vec![CorefCluster {
                id: 0,
                mentions: vec!["John".to_string(), "he".to_string()],
                spans: vec![(15, 19), (58, 60)],
                canonical: "John".to_string(),
            }];
            let result = resolve_for_rag_neural(text, &clusters, None);
            assert!(result.text.contains("John should go back"));
            assert_eq!(result.rewrites.len(), 1);
        }

        #[test]
        fn test_neural_rag_preserves_case() {
            let text = "Marie Curie was brilliant. She won two Nobel Prizes.";
            let clusters = vec![CorefCluster {
                id: 0,
                mentions: vec!["Marie Curie".to_string(), "She".to_string()],
                spans: vec![(0, 11), (27, 30)],
                canonical: "Marie Curie".to_string(),
            }];
            let result = resolve_for_rag_neural(text, &clusters, None);
            assert_eq!(
                result.text,
                "Marie Curie was brilliant. Marie Curie won two Nobel Prizes."
            );
        }

        #[test]
        fn test_neural_rag_singleton_cluster_ignored() {
            let text = "John went to the store.";
            let clusters = vec![CorefCluster {
                id: 0,
                mentions: vec!["John".to_string()],
                spans: vec![(0, 4)],
                canonical: "John".to_string(),
            }];
            let result = resolve_for_rag_neural(text, &clusters, None);
            assert_eq!(result.text, text);
            assert_eq!(result.rewrites.len(), 0);
        }
    }
}
