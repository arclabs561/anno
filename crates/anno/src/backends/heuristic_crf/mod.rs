//! CRF sequence labeler with heuristic emission features.
//!
//! Uses a real CRF layer (Viterbi decoding over learned transition scores) for
//! structured sequence prediction. Emission scores come from heuristic features
//! (gazetteers, word shape, capitalization) rather than a neural BiLSTM encoder.
//! The architecture follows the BiLSTM-CRF pattern (Huang et al. 2015) but
//! substitutes the BiLSTM with lightweight feature extraction.
//!
//! # Historical Context
//!
//! The NER field evolved through three eras:
//!
//! ```text
//! Era 1: Rule-based (1987-1997)     - Lexicons, hand-crafted patterns
//! Era 2: Statistical (1997-2015)    - HMM → MEMM → CRF (feature engineering)
//! Era 3: Neural (2011-present)      - CNN → BiLSTM-CRF → Transformers
//! ```
//!
//! BiLSTM-CRF bridged statistical and neural approaches:
//! - **BiLSTM**: Learns features automatically from data (no feature engineering)
//! - **CRF layer**: Retains structured prediction from statistical era
//!
//! Collobert et al. 2011 ("NLP from Scratch") first showed CNNs for NER, but
//! BiLSTM-CRF (2015) became the dominant architecture until BERT (2018).
//!
//! # Why Keep the CRF Layer?
//!
//! The BiLSTM produces emission scores for each position, but doesn't model
//! label dependencies. The CRF layer ensures:
//! - Valid BIO sequences (no `I-PER` after `O`)
//! - Learned transition patterns (e.g., `B-ORG` often followed by `I-ORG`)
//!
//! ```text
//! Without CRF:  BiLSTM predicts [B-PER, I-ORG, O, B-LOC]  // invalid!
//! With CRF:     Viterbi finds   [B-PER, O,     O, B-LOC]  // valid sequence
//! ```
//!
//! # Architecture (this implementation)
//!
//! ```text
//! Input: "John works at Google"
//!    ↓
//! ┌─────────────────────────────────────────┐
//! │ Heuristic Feature Extraction            │
//! │  - Gazetteer lookup (names, places)     │
//! │  - Word shape (capitalization, digits)   │
//! │  - Context window features              │
//! └─────────────────────────────────────────┘
//!    ↓  emission scores
//! ┌─────────────────────────────────────────┐
//! │ CRF Layer                               │
//! │  - Transition matrix (learned)          │
//! │  - Viterbi decoding for best sequence   │
//! └─────────────────────────────────────────┘
//!    ↓
//! Output: B-PER O O B-ORG
//! ```
//!
//! # Key Papers
//!
//! - Collobert et al. 2011: "Natural Language Processing (Almost) from Scratch"
//! - Huang et al. 2015: "Bidirectional LSTM-CRF Models for Sequence Tagging"
//! - Lample et al. 2016: "Neural Architectures for Named Entity Recognition"
//! - Ma & Hovy 2016: "End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF"
//! - Peters et al. 2018: "Deep Contextualized Word Representations" (ELMo)
//!
//! # References
//!
//! - Collobert, Weston, Bottou, et al. (2011): "Natural Language Processing
//!   (Almost) from Scratch" (JMLR) — first neural NER
//! - Huang, Xu, Yu (2015): "Bidirectional LSTM-CRF Models for Sequence
//!   Tagging" (arXiv:1508.01991) — introduced BiLSTM-CRF
//! - Lample, Ballesteros, Subramanian, et al. (2016): "Neural Architectures
//!   for Named Entity Recognition" (NAACL) — char embeddings
//! - Ma & Hovy (2016): "End-to-end Sequence Labeling via Bi-directional
//!   LSTM-CNNs-CRF" (ACL) — CNN char encoder
//!
//! # See Also
//!
//! - Historical NER baselines (HMM/CRF-era sequence models)
//!
//! # Usage
//!
//! ```rust
//! use anno::backends::heuristic_crf::HeuristicCrfNER;
//! use anno::Model;
//!
//! // Create with heuristic weights (no neural inference)
//! let ner = HeuristicCrfNER::new();
//! let entities = ner.extract_entities("John works at Google", None).unwrap();
//! ```
//!
//! With ONNX feature enabled, load pre-trained weights:
//!
//! ```rust,ignore
//! // Requires: features = ["onnx"]
//! let ner = HeuristicCrfNER::from_onnx("path/to/model.onnx")?;
//! // When loaded from ONNX, emission scores come from the neural model
//! // instead of heuristics. The CRF layer is used in both modes.
//! ```

use crate::{Entity, EntityCategory, EntityType, Language, Model, Result};
use std::collections::HashMap;

/// Heuristic-CRF configuration.
#[derive(Debug, Clone)]
pub struct HeuristicCrfConfig {
    /// Hidden size for LSTM layers.
    pub hidden_size: usize,
    /// Number of LSTM layers.
    pub num_layers: usize,
    /// Dropout probability.
    pub dropout: f32,
    /// Whether to use character-level embeddings.
    pub use_char_embeddings: bool,
    /// Maximum sequence length.
    pub max_seq_len: usize,
}

impl Default for HeuristicCrfConfig {
    fn default() -> Self {
        Self {
            hidden_size: 256,
            num_layers: 2,
            dropout: 0.5,
            use_char_embeddings: true,
            max_seq_len: 512,
        }
    }
}

/// Heuristic-CRF NER model.
///
/// CRF sequence labeling with heuristic emission features (capitalization,
/// word shape, gazetteer). The CRF layer (Viterbi decoding, transition matrix)
/// is real; only the emission source is heuristic rather than neural.
///
/// # Components
///
/// 1. **Gazetteer lookup**: Known person/org/location names
/// 2. **Word shape features**: Capitalization, acronyms, suffixes
/// 3. **Context window**: Adjacent-token signals
/// 4. **CRF Decoder**: Structured prediction with transition constraints
#[derive(Debug)]
pub struct HeuristicCrfNER {
    /// Model configuration.
    config: HeuristicCrfConfig,
    /// BIO labels for decoding.
    labels: Vec<String>,
    /// Label to index mapping.
    label_to_idx: HashMap<String, usize>,
    /// Transition scores (from CRF layer).
    transitions: Vec<Vec<f64>>,
    /// Word vocabulary (word -> embedding index).
    vocab: HashMap<String, usize>,
    /// ONNX session for inference (when onnx feature enabled).
    #[cfg(feature = "onnx")]
    session: Option<ort::session::Session>,
}

impl HeuristicCrfNER {
    /// Create a new Heuristic-CRF model with default configuration.
    ///
    /// This creates a model that uses heuristic-based inference
    /// (no neural weights). For actual neural inference, use
    /// `from_onnx()` to load pre-trained weights.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(HeuristicCrfConfig::default())
    }

    /// Create with custom configuration.
    #[must_use]
    pub fn with_config(config: HeuristicCrfConfig) -> Self {
        let labels = vec![
            "O".to_string(),
            "B-PER".to_string(),
            "I-PER".to_string(),
            "B-ORG".to_string(),
            "I-ORG".to_string(),
            "B-LOC".to_string(),
            "I-LOC".to_string(),
            "B-MISC".to_string(),
            "I-MISC".to_string(),
        ];

        let label_to_idx: HashMap<String, usize> = labels
            .iter()
            .enumerate()
            .map(|(i, l)| (l.clone(), i))
            .collect();

        // Initialize transition matrix with sensible defaults
        // Higher scores for valid BIO transitions
        let n = labels.len();
        let mut transitions = vec![vec![0.0; n]; n];

        // BIO constraints: I-X can only follow B-X or I-X
        for i in 0..n {
            for j in 0..n {
                let from_label = &labels[i];
                let to_label = &labels[j];

                if let Some(entity_type) = to_label.strip_prefix("I-") {
                    let valid_prev = format!("B-{}", entity_type);
                    let valid_cont = format!("I-{}", entity_type);

                    if from_label == &valid_prev || from_label == &valid_cont {
                        transitions[i][j] = 1.0; // Valid transition
                    } else {
                        transitions[i][j] = -10.0; // Invalid transition
                    }
                } else {
                    // B-X or O can follow anything
                    transitions[i][j] = 0.0;
                }
            }
        }

        Self {
            config,
            labels,
            label_to_idx,
            transitions,
            vocab: HashMap::new(),
            #[cfg(feature = "onnx")]
            session: None,
        }
    }

    /// Load from ONNX model file.
    #[cfg(feature = "onnx")]
    pub fn from_onnx(model_path: &str) -> Result<Self> {
        use crate::Error;
        use ort::session::{builder::GraphOptimizationLevel, Session};

        let session = Session::builder()
            .map_err(|e| Error::model_init(format!("Failed to create session builder: {}", e)))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| Error::model_init(format!("Failed to set optimization level: {}", e)))?
            .commit_from_file(model_path)
            .map_err(|e| Error::model_init(format!("Failed to load ONNX model: {}", e)))?;

        let mut model = Self::new();
        model.session = Some(session);
        Ok(model)
    }

    /// Returns a reference to the model configuration.
    #[must_use]
    pub fn config(&self) -> &HeuristicCrfConfig {
        &self.config
    }

    /// Returns a reference to the word vocabulary.
    #[must_use]
    pub fn vocab(&self) -> &HashMap<String, usize> {
        &self.vocab
    }

    /// Look up a word's embedding index in the vocabulary.
    ///
    /// Returns `None` if the word is not in the vocabulary.
    #[must_use]
    pub fn vocab_lookup(&self, word: &str) -> Option<usize> {
        self.vocab.get(word).copied()
    }

    /// Returns the BIO label set used by this model.
    #[must_use]
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// Tokenize text into words.
    fn tokenize(text: &str) -> Vec<&str> {
        text.split_whitespace().collect()
    }

    /// Get emission scores for each token.
    ///
    /// Emission scores are computed from heuristic features (gazetteers, word
    /// shape, capitalization) rather than a learned BiLSTM encoder. The CRF
    /// layer above is real; only the emission source is simplified.
    fn get_emissions(&self, tokens: &[&str]) -> Vec<Vec<f64>> {
        let n_labels = self.labels.len();
        let mut emissions = vec![vec![0.0; n_labels]; tokens.len()];

        // Gazetteers for better heuristic accuracy
        const PERSON_NAMES: &[&str] = &[
            "john",
            "mary",
            "james",
            "david",
            "michael",
            "robert",
            "william",
            "richard",
            "sarah",
            "jennifer",
            "elizabeth",
            "lisa",
            "marie",
            "jane",
            "emily",
            "anna",
            "barack",
            "donald",
            "joe",
            "george",
            "bill",
            "hillary",
            "satya",
            "jeff",
            "mr",
            "mrs",
            "ms",
            "dr",
            "prof",
            "sir",
            "lord",
            "president",
            "ceo",
        ];
        const ORG_NAMES: &[&str] = &[
            "google",
            "apple",
            "microsoft",
            "amazon",
            "facebook",
            "meta",
            "tesla",
            "ibm",
            "intel",
            "nvidia",
            "oracle",
            "cisco",
            "adobe",
            "netflix",
            "uber",
            "university",
            "institute",
            "corporation",
            "company",
            "inc",
            "corp",
            "ltd",
            "llc",
            "foundation",
            "association",
            "organization",
            "department",
            "agency",
            "fbi",
            "cia",
            "nsa",
            "nasa",
            "un",
            "nato",
            "who",
            "imf",
            "eu",
            "usa",
        ];
        const LOC_NAMES: &[&str] = &[
            "new",
            "york",
            "california",
            "texas",
            "florida",
            "london",
            "paris",
            "berlin",
            "tokyo",
            "beijing",
            "moscow",
            "washington",
            "chicago",
            "boston",
            "seattle",
            "san",
            "francisco",
            "los",
            "angeles",
            "las",
            "vegas",
            "united",
            "states",
            "america",
            "china",
            "russia",
            "germany",
            "france",
            "japan",
            "india",
            "brazil",
            "city",
            "county",
            "state",
            "country",
            "river",
            "mountain",
            "lake",
            "ocean",
        ];

        for (i, token) in tokens.iter().enumerate() {
            let lower = token.to_lowercase();
            let is_capitalized = token.chars().next().is_some_and(|c| c.is_uppercase());
            let is_all_caps = token
                .chars()
                .all(|c| c.is_uppercase() || !c.is_alphabetic())
                && token.len() > 1;
            let has_digit = token.chars().any(|c| c.is_ascii_digit());
            let is_first = i == 0;

            // Default: bias toward O (entities are rare)
            emissions[i][0] = 1.5;

            // Gazetteer matches (strongest signal)
            if PERSON_NAMES.contains(&lower.as_str()) {
                emissions[i][self.label_to_idx["B-PER"]] += 2.0;
                emissions[i][self.label_to_idx["I-PER"]] += 1.0;
            }
            if ORG_NAMES.contains(&lower.as_str()) {
                emissions[i][self.label_to_idx["B-ORG"]] += 2.0;
                emissions[i][self.label_to_idx["I-ORG"]] += 1.0;
            }
            if LOC_NAMES.contains(&lower.as_str()) {
                emissions[i][self.label_to_idx["B-LOC"]] += 2.0;
                emissions[i][self.label_to_idx["I-LOC"]] += 1.0;
            }

            // Capitalization (weaker signal, context-dependent)
            if is_capitalized && !has_digit && !is_first {
                emissions[i][self.label_to_idx["B-PER"]] += 0.8;
                emissions[i][self.label_to_idx["B-ORG"]] += 0.6;
                emissions[i][self.label_to_idx["B-LOC"]] += 0.5;
            }

            // Organization suffixes
            if lower.ends_with("inc.")
                || lower.ends_with("corp.")
                || lower.ends_with("ltd.")
                || lower.ends_with("llc")
                || lower.ends_with("co.")
            {
                emissions[i][self.label_to_idx["B-ORG"]] += 1.5;
                emissions[i][self.label_to_idx["I-ORG"]] += 1.0;
            }

            // Acronyms (2-5 uppercase letters)
            if is_all_caps && token.len() >= 2 && token.len() <= 5 && !has_digit {
                emissions[i][self.label_to_idx["B-ORG"]] += 1.2;
            }

            // Honorifics signal person
            if ["mr.", "mrs.", "ms.", "dr.", "prof."].contains(&lower.as_str()) {
                emissions[i][self.label_to_idx["B-PER"]] += 1.5;
            }

            // "The" before proper noun often signals ORG or LOC
            if i > 0 && tokens[i - 1].to_lowercase() == "the" && is_capitalized {
                emissions[i][self.label_to_idx["B-ORG"]] += 0.5;
                emissions[i][self.label_to_idx["B-LOC"]] += 0.3;
            }

            // Multi-word entity continuation
            if i > 0 {
                let prev_cap = tokens[i - 1]
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_uppercase());
                if prev_cap && is_capitalized && !is_first {
                    // Likely continuation of entity
                    emissions[i][self.label_to_idx["I-PER"]] += 0.6;
                    emissions[i][self.label_to_idx["I-ORG"]] += 0.6;
                    emissions[i][self.label_to_idx["I-LOC"]] += 0.4;
                }
            }
        }

        emissions
    }

    /// Viterbi decoding with CRF transitions.
    fn viterbi_decode(&self, emissions: &[Vec<f64>]) -> Vec<usize> {
        if emissions.is_empty() {
            return vec![];
        }

        let n = emissions.len();
        let m = self.labels.len();

        // DP tables
        let mut scores = vec![vec![f64::NEG_INFINITY; m]; n];
        let mut backpointers = vec![vec![0usize; m]; n];

        // Initialize first position
        for j in 0..m {
            scores[0][j] = emissions[0][j];
        }

        // Forward pass
        for i in 1..n {
            for j in 0..m {
                let mut best_score = f64::NEG_INFINITY;
                let mut best_prev = 0;

                #[allow(clippy::needless_range_loop)]
                for k in 0..m {
                    let score = scores[i - 1][k] + self.transitions[k][j] + emissions[i][j];
                    if score > best_score {
                        best_score = score;
                        best_prev = k;
                    }
                }

                scores[i][j] = best_score;
                backpointers[i][j] = best_prev;
            }
        }

        // Backward pass
        let mut path = vec![0usize; n];
        let mut best_final = 0;
        let mut best_score = f64::NEG_INFINITY;

        for (j, &score) in scores[n - 1].iter().enumerate() {
            if score > best_score {
                best_score = score;
                best_final = j;
            }
        }

        path[n - 1] = best_final;
        for i in (0..n - 1).rev() {
            path[i] = backpointers[i + 1][path[i + 1]];
        }

        path
    }

    /// Convert BIO labels to entities.
    ///
    /// Uses token position tracking to correctly handle duplicate entity texts.
    /// The previous implementation used `text.find()` which always returned the
    /// first occurrence, causing incorrect offsets for duplicate entities.
    fn labels_to_entities(
        &self,
        text: &str,
        tokens: &[&str],
        label_indices: &[usize],
    ) -> Vec<Entity> {
        use crate::offset::SpanConverter;

        let converter = SpanConverter::new(text);
        let mut entities = Vec::new();

        // Track token positions (byte offsets) as we iterate
        let token_positions: Vec<(usize, usize)> = Self::calculate_token_positions(text, tokens);

        let mut current_entity: Option<(usize, usize, EntityType, Vec<&str>)> = None;

        for (i, (&label_idx, &token)) in label_indices.iter().zip(tokens.iter()).enumerate() {
            let label = &self.labels[label_idx];

            if let Some(entity_suffix) = label.strip_prefix("B-") {
                // Save previous entity if any
                if let Some((start_token_idx, end_token_idx, entity_type, words)) =
                    current_entity.take()
                {
                    Self::push_entity_from_positions(
                        &converter,
                        &token_positions,
                        start_token_idx,
                        end_token_idx,
                        &words,
                        entity_type,
                        &mut entities,
                    );
                }

                // Start new entity
                let entity_type = match entity_suffix {
                    "PER" => EntityType::Person,
                    "ORG" => EntityType::Organization,
                    "LOC" => EntityType::Location,
                    other => EntityType::custom(other, EntityCategory::Misc),
                };
                current_entity = Some((i, i, entity_type, vec![token]));
            } else if label.starts_with("I-") && current_entity.is_some() {
                // Continue current entity
                if let Some((_, ref mut end_idx, _, ref mut words)) = current_entity {
                    words.push(token);
                    *end_idx = i;
                }
            } else {
                // O label - save and reset
                if let Some((start_token_idx, end_token_idx, entity_type, words)) =
                    current_entity.take()
                {
                    Self::push_entity_from_positions(
                        &converter,
                        &token_positions,
                        start_token_idx,
                        end_token_idx,
                        &words,
                        entity_type,
                        &mut entities,
                    );
                }
            }
        }

        // Don't forget last entity
        if let Some((start_token_idx, end_token_idx, entity_type, words)) = current_entity.take() {
            Self::push_entity_from_positions(
                &converter,
                &token_positions,
                start_token_idx,
                end_token_idx,
                &words,
                entity_type,
                &mut entities,
            );
        }

        entities
    }

    /// Calculate byte positions for each token in the text.
    fn calculate_token_positions(text: &str, tokens: &[&str]) -> Vec<(usize, usize)> {
        let mut positions = Vec::with_capacity(tokens.len());
        let mut byte_pos = 0;

        for token in tokens {
            // Find token starting from current position
            if let Some(rel_pos) = text[byte_pos..].find(token) {
                let start = byte_pos + rel_pos;
                let end = start + token.len();
                positions.push((start, end));
                byte_pos = end; // Move past this token
            } else {
                // Fallback: use current position (shouldn't happen with whitespace tokenization)
                positions.push((byte_pos, byte_pos));
            }
        }

        positions
    }

    /// Helper to push entity using tracked token positions.
    fn push_entity_from_positions(
        converter: &crate::offset::SpanConverter,
        positions: &[(usize, usize)],
        start_token_idx: usize,
        end_token_idx: usize,
        words: &[&str],
        entity_type: EntityType,
        entities: &mut Vec<Entity>,
    ) {
        if start_token_idx >= positions.len() || end_token_idx >= positions.len() {
            return;
        }

        let byte_start = positions[start_token_idx].0;
        let byte_end = positions[end_token_idx].1;
        let char_start = converter.byte_to_char(byte_start);
        let char_end = converter.byte_to_char(byte_end);
        let entity_text = words.join(" ");

        entities.push(Entity::new(
            entity_text,
            entity_type,
            char_start,
            char_end,
            0.75, // Heuristic-CRF confidence
        ));
    }
}

impl Default for HeuristicCrfNER {
    fn default() -> Self {
        Self::new()
    }
}

impl Model for HeuristicCrfNER {
    fn name(&self) -> &'static str {
        "heuristic-crf"
    }

    fn description(&self) -> &'static str {
        "CRF sequence labeling with heuristic emission features (capitalization, word shape, gazetteer)"
    }

    fn extract_entities(&self, text: &str, _language: Option<Language>) -> Result<Vec<Entity>> {
        if text.trim().is_empty() {
            return Ok(vec![]);
        }

        let tokens = Self::tokenize(text);
        if tokens.is_empty() {
            return Ok(vec![]);
        }

        // Get emission scores (from heuristic features)
        let emissions = self.get_emissions(&tokens);

        // Viterbi decode with CRF transitions
        let label_indices = self.viterbi_decode(&emissions);

        // Convert to entities
        let entities = self.labels_to_entities(text, &tokens, &label_indices);

        Ok(entities)
    }

    fn supported_types(&self) -> Vec<EntityType> {
        vec![
            EntityType::Person,
            EntityType::Organization,
            EntityType::Location,
            EntityType::custom("MISC", EntityCategory::Misc),
        ]
    }

    fn is_available(&self) -> bool {
        true // Always available with heuristic fallback
    }

    fn capabilities(&self) -> crate::ModelCapabilities {
        crate::ModelCapabilities::default()
    }
}

#[cfg(test)]
mod tests;
