//! Conditional Random Field (CRF) NER backend.
//!
//! Implements classical statistical NER using CRF sequence labeling.
//! This provides a lightweight, interpretable baseline that requires no
//! external dependencies or GPU acceleration.
//!
//! # History
//!
//! CRF-based NER was the dominant approach from 2001-2015:
//! - Lafferty et al. 2001: Introduced CRFs for sequence labeling (ICML)
//! - McCallum & Li 2003: Applied CRFs to NER
//! - Stanford NER (2003-2014): CRF-based, still widely used
//! - State-of-art until neural methods (BiLSTM-CRF, 2015+)
//!
//! # Why CRF Beat Previous Methods
//!
//! CRFs solved the **label bias problem** that plagued MEMMs (Maximum Entropy
//! Markov Models, McCallum et al. 2000):
//!
//! ```text
//! Label Bias: In MEMMs, states with few successors effectively ignore
//! observations. Transition scores are conditional on current state,
//! so low-entropy states "absorb" probability mass regardless of input.
//!
//! HMM:   Generative model     P(x,y) = P(y) × P(x|y)
//! MEMM:  Local discriminative P(y_t|y_{t-1}, x)  ← label bias here
//! CRF:   Global discriminative P(y|x) = (1/Z) exp(∑ features × weights)
//!                                       ↑ normalizes over entire sequence
//! ```
//!
//! CRF models the conditional probability of the entire label sequence given
//! the observation sequence, using global normalization:
//!
//! ```text
//! P(y|x) = (1/Z(x)) × exp( ∑_t ∑_k λ_k × f_k(y_t, y_{t-1}, x, t) )
//!
//! where:
//!   - Z(x) is the partition function (normalizer)
//!   - f_k are feature functions
//!   - λ_k are learned weights
//! ```
//!
//! # References
//!
//! - Lafferty, McCallum, Pereira (2001): "Conditional Random Fields:
//!   Probabilistic Models for Segmenting and Labeling Sequence Data" (ICML)
//! - McCallum & Li (2003): "Early Results for Named Entity Recognition with
//!   Conditional Random Fields, Feature Induction and Web-Enhanced Lexicons"
//! - Finkel, Grenager, Manning (2005): "Incorporating Non-local Information
//!   into Information Extraction Systems by Gibbs Sampling" (ACL)
//!
//! # See Also
//!
//! - Historical NER baselines (HMM/CRF-era sequence models)
//!
//! # Feature Templates
//!
//! The CRF uses the following feature templates (matching `train_crf_weights.py`):
//!
//! ```text
//! - bias           : Always-on feature for label-specific bias
//! - word.lower     : Lowercased current word
//! - word.shape     : Word shape pattern (Xx, X, x, 0, etc.)
//! - word.isdigit   : Whether word is all digits
//! - word.istitle   : Whether word is titlecased
//! - word.isupper   : Whether word is all uppercase
//! - prefix{2,3}    : First 2-3 characters
//! - suffix{2,3}    : Last 2-3 characters
//! - -1:word.*      : Previous word features
//! - +1:word.*      : Next word features
//! - BOS/EOS        : Sentence boundary markers
//! ```
//!
//! # Performance
//!
//! Performance depends on weights, tokenization, and dataset; use the eval harness
//! for quantitative results.
//! | Heuristic | Lower | Hand-tuned, always available |
//! | Trained | Higher | From `train_crf_weights.py` |
//! | Neural | Highest | For comparison (GLiNER, BERT) |
//!
//! # Usage
//!
//! ```rust
//! use anno::CrfNER;
//! use anno::Model;
//!
//! // Use with default heuristic weights
//! let ner = CrfNER::new();
//! let entities = ner.extract_entities("John Smith works at Google", None)?;
//!
//! // Or load trained weights for better accuracy
//! // let ner = CrfNER::with_weights("crf_weights.json")?;
//! # Ok::<(), anno::Error>(())
//! ```
//!
//! # Training Weights
//!
//! To train weights on CoNLL-2003:
//!
//! ```bash
//! uv run scripts/train_crf_weights.py
//! ```
//!
//! This produces `crf_weights.json` which can be loaded with `CrfNER::with_weights()`.
//!
//! # Advantages Over Neural Methods
//!
//! - **Interpretable**: Features and weights are human-readable
//! - **Fast training**: CPU-only training; typically faster to iterate than neural training loops
//! - **No dependencies**: Pure Rust, no ONNX/Candle required
//! - **Deterministic**: Same input always produces same output
//! - **Small footprint**: Small weights file compared to ML model artifacts

use crate::{Entity, EntityType, Model, Result};
use std::collections::HashMap;

/// CRF-based NER model.
///
/// Uses hand-crafted features and sequence labeling for named entity recognition.
/// This is a pure-Rust implementation that doesn't require external libraries.
pub struct CrfNER {
    /// Feature weights learned during training (or loaded from file)
    weights: HashMap<String, f64>,
    /// Entity type gazetteer lists
    gazetteers: HashMap<EntityType, Vec<String>>,
    /// Label set (BIO tagging)
    labels: Vec<String>,
    /// Feature templates
    templates: Vec<FeatureTemplate>,
}

/// Feature template for CRF
#[derive(Debug, Clone)]
pub enum FeatureTemplate {
    /// Current word
    Word,
    /// Word at offset
    WordAt(i32),
    /// Word shape (Xx, XX, x, 0)
    Shape,
    /// Shape at offset
    ShapeAt(i32),
    /// Prefix of length n
    Prefix(usize),
    /// Suffix of length n
    Suffix(usize),
    /// Is in gazetteer for entity type
    InGazetteer(EntityType),
    /// Previous label
    PrevLabel,
    /// Bigram: current + previous label
    LabelBigram,
    /// Word + Label combination
    WordLabel,
}

impl Default for CrfNER {
    fn default() -> Self {
        Self::new()
    }
}

impl CrfNER {
    /// Create a new CRF NER model with default features.
    #[must_use]
    pub fn new() -> Self {
        // Default gazetteers (common names, locations, organizations)
        let mut gazetteers = HashMap::new();

        gazetteers.insert(
            EntityType::Person,
            vec![
                // Common first names
                "John",
                "Mary",
                "James",
                "Robert",
                "Michael",
                "David",
                "William",
                "Richard",
                "Joseph",
                "Thomas",
                "Elizabeth",
                "Jennifer",
                "Linda",
                "Barbara",
                "Susan",
                "Jessica",
                "Sarah",
                "Karen",
                "Nancy",
                "Margaret",
                // Titles that precede names
                "Dr",
                "Mr",
                "Mrs",
                "Ms",
                "Prof",
                "President",
                "CEO",
                "Senator",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
        );

        gazetteers.insert(
            EntityType::Location,
            vec![
                // Countries
                "USA",
                "UK",
                "France",
                "Germany",
                "China",
                "Japan",
                "India",
                "Brazil",
                "Canada",
                "Australia",
                "Russia",
                "Italy",
                "Spain",
                "Mexico",
                // US States
                "California",
                "Texas",
                "Florida",
                "New York",
                "Illinois",
                "Pennsylvania",
                // Major cities
                "London",
                "Paris",
                "Tokyo",
                "Beijing",
                "Moscow",
                "Berlin",
                "Rome",
                "Madrid",
                "Sydney",
                "Toronto",
                "Mumbai",
                "Shanghai",
                "Seoul",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
        );

        gazetteers.insert(
            EntityType::Organization,
            vec![
                // Companies
                "Google",
                "Apple",
                "Microsoft",
                "Amazon",
                "Facebook",
                "Tesla",
                "IBM",
                "Intel",
                "Oracle",
                "Cisco",
                "Samsung",
                "Sony",
                "Toyota",
                "Honda",
                // Suffixes
                "Inc",
                "Corp",
                "LLC",
                "Ltd",
                "Company",
                "Corporation",
                "Group",
                // Organizations
                "UN",
                "NATO",
                "WHO",
                "FBI",
                "CIA",
                "NASA",
                "EU",
                "OPEC",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
        );

        // Standard BIO labels
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

        // Default feature templates
        let templates = vec![
            FeatureTemplate::Word,
            FeatureTemplate::WordAt(-1),
            FeatureTemplate::WordAt(1),
            FeatureTemplate::Shape,
            FeatureTemplate::ShapeAt(-1),
            FeatureTemplate::ShapeAt(1),
            FeatureTemplate::Prefix(2),
            FeatureTemplate::Prefix(3),
            FeatureTemplate::Suffix(2),
            FeatureTemplate::Suffix(3),
            FeatureTemplate::InGazetteer(EntityType::Person),
            FeatureTemplate::InGazetteer(EntityType::Location),
            FeatureTemplate::InGazetteer(EntityType::Organization),
            FeatureTemplate::PrevLabel,
        ];

        // Initialize with heuristic weights (not trained)
        let weights = Self::default_weights();

        Self {
            weights,
            gazetteers,
            labels,
            templates,
        }
    }

    /// Load weights from a JSON file.
    ///
    /// # Example JSON format:
    /// ```json
    /// {
    ///     "gaz:PER:B-PER": 2.5,
    ///     "shape=Xx:B-PER": 1.5,
    ///     "trans:B-PER->I-PER": 1.0
    /// }
    /// ```
    pub fn load_weights(path: &str) -> Result<HashMap<String, f64>> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            crate::Error::invalid_input(format!("Failed to read weights file: {}", e))
        })?;
        let weights: HashMap<String, f64> = serde_json::from_str(&content).map_err(|e| {
            crate::Error::invalid_input(format!("Failed to parse weights JSON: {}", e))
        })?;
        Ok(weights)
    }

    /// Create CRF model with weights from a file.
    pub fn with_weights(path: &str) -> Result<Self> {
        let weights = Self::load_weights(path)?;
        let mut model = Self::new();
        model.weights = weights;
        Ok(model)
    }

    /// Create default heuristic weights for common features.
    ///
    /// These weights are hand-tuned heuristics, not learned from data.
    /// For better accuracy, train weights using scripts/train_crf_weights.py
    /// and load them with `CrfNER::with_weights("crf_weights.json")`.
    fn default_weights() -> HashMap<String, f64> {
        let mut w = HashMap::new();

        // Strong bias toward O (outside) by default - entities are rare
        w.insert("bias:O".to_string(), 3.0);

        // Extra strong bias for lowercase words
        w.insert("word.shape=x:O".to_string(), 2.5);
        w.insert("word.shape=x:B-PER".to_string(), -3.0);
        w.insert("word.shape=x:I-PER".to_string(), -2.0);

        // Gazetteer features are very strong signals for B- tags
        w.insert("gaz:PER:B-PER".to_string(), 4.0);
        w.insert("gaz:LOC:B-LOC".to_string(), 4.0);
        w.insert("gaz:ORG:B-ORG".to_string(), 4.0);

        // Capitalization patterns - only for B- tags
        w.insert("word.shape=Xx:B-PER".to_string(), 2.0);
        w.insert("word.shape=Xx:B-LOC".to_string(), 1.5);
        w.insert("word.shape=Xx:B-ORG".to_string(), 1.5);
        w.insert("word.shape=Xx:I-PER".to_string(), 1.0); // Continue if already in entity
        w.insert("word.shape=Xx:I-ORG".to_string(), 1.0);
        w.insert("word.shape=XX:B-ORG".to_string(), 2.5); // Acronyms like IBM, NASA

        // Lowercase words are unlikely to be entity starts
        w.insert("word.shape=x:B-PER".to_string(), -2.0);
        w.insert("word.shape=x:B-ORG".to_string(), -2.0);
        w.insert("word.shape=x:B-LOC".to_string(), -2.0);

        // Common words that are NOT entities - strongly bias toward O
        for word in [
            "the",
            "a",
            "an",
            "of",
            "in",
            "at",
            "to",
            "and",
            "or",
            "is",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "can",
            "won",
            "works",
            "worked",
            "working",
            "serves",
            "served",
            "announced",
            "said",
            "made",
            "that",
            "this",
            "which",
            "for",
            "with",
            "as",
            "by",
            "on",
            "from",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
        ] {
            w.insert(format!("word.lower={}:O", word), 5.0);
            w.insert(format!("word.lower={}:B-PER", word), -5.0);
            w.insert(format!("word.lower={}:B-ORG", word), -5.0);
            w.insert(format!("word.lower={}:B-LOC", word), -5.0);
            w.insert(format!("word.lower={}:I-PER", word), -4.0);
            w.insert(format!("word.lower={}:I-ORG", word), -4.0);
            w.insert(format!("word.lower={}:I-LOC", word), -4.0);
        }

        // Common suffixes for organizations
        w.insert("suffix3=inc:B-ORG".to_string(), 3.0);
        // Note: we intentionally do not add a suffix4 feature here; keep default weights aligned
        // with `scripts/train_crf_weights.py` (prefix/suffix lengths 2-3).
        w.insert("suffix3=ltd:B-ORG".to_string(), 3.0);
        w.insert("suffix3=llc:B-ORG".to_string(), 3.0);

        // Context words that suggest entities
        w.insert("-1:word.lower=dr:B-PER".to_string(), 2.5);
        w.insert("-1:word.lower=mr:B-PER".to_string(), 2.5);
        w.insert("-1:word.lower=mrs:B-PER".to_string(), 2.5);
        w.insert("-1:word.lower=ms:B-PER".to_string(), 2.5);
        w.insert("-1:word.lower=prof:B-PER".to_string(), 2.5);
        w.insert("-1:word.lower=president:B-PER".to_string(), 2.0);
        w.insert("-1:word.lower=ceo:B-PER".to_string(), 2.0);

        // Location context
        w.insert("-1:word.lower=in:B-LOC".to_string(), 1.5);
        w.insert("-1:word.lower=at:B-LOC".to_string(), 1.5);
        w.insert("-1:word.lower=from:B-LOC".to_string(), 1.5);
        w.insert("-1:word.lower=of:B-LOC".to_string(), 1.0);
        w.insert("-1:word.lower=of:B-ORG".to_string(), 1.0);

        // Transition features (BIO constraints) - very important
        // Valid transitions
        w.insert("trans:B-PER->I-PER".to_string(), 3.0);
        w.insert("trans:B-ORG->I-ORG".to_string(), 3.0);
        w.insert("trans:B-LOC->I-LOC".to_string(), 3.0);
        w.insert("trans:I-PER->I-PER".to_string(), 2.0);
        w.insert("trans:I-ORG->I-ORG".to_string(), 2.0);
        w.insert("trans:I-LOC->I-LOC".to_string(), 2.0);

        // End entity transitions
        w.insert("trans:B-PER->O".to_string(), 0.0);
        w.insert("trans:B-ORG->O".to_string(), 0.0);
        w.insert("trans:B-LOC->O".to_string(), 0.0);
        w.insert("trans:I-PER->O".to_string(), 0.0);
        w.insert("trans:I-ORG->O".to_string(), 0.0);
        w.insert("trans:I-LOC->O".to_string(), 0.0);

        // Invalid transitions (strongly penalize)
        w.insert("trans:O->I-PER".to_string(), -10.0);
        w.insert("trans:O->I-ORG".to_string(), -10.0);
        w.insert("trans:O->I-LOC".to_string(), -10.0);
        w.insert("trans:O->I-MISC".to_string(), -10.0);

        // Cross-type I- transitions are invalid
        w.insert("trans:B-PER->I-ORG".to_string(), -10.0);
        w.insert("trans:B-PER->I-LOC".to_string(), -10.0);
        w.insert("trans:B-ORG->I-PER".to_string(), -10.0);
        w.insert("trans:B-ORG->I-LOC".to_string(), -10.0);
        w.insert("trans:B-LOC->I-PER".to_string(), -10.0);
        w.insert("trans:B-LOC->I-ORG".to_string(), -10.0);

        w
    }

    /// Match Python `str.isdigit()` / `c.isdigit()` behavior used in
    /// `scripts/train_crf_weights.py`.
    ///
    /// Note: This is *Unicode-aware* (unlike `is_ascii_digit`).
    #[allow(clippy::is_digit_ascii_radix)]
    fn is_digit_py(c: char) -> bool {
        c.is_digit(10)
    }

    /// Compute word shape (e.g., "John" -> "Xxxx", "USA" -> "XXX")
    fn word_shape(word: &str) -> String {
        word.chars()
            .map(|c| {
                if c.is_uppercase() {
                    'X'
                } else if c.is_lowercase() {
                    'x'
                } else if Self::is_digit_py(c) {
                    '0'
                } else {
                    c
                }
            })
            .collect::<String>()
            // Compress repeated chars
            .chars()
            .fold(String::new(), |mut acc, c| {
                if !acc.ends_with(&c.to_string()) {
                    acc.push(c);
                }
                acc
            })
    }

    /// Extract features for a token at given position.
    ///
    /// Feature format matches `scripts/train_crf_weights.py` for compatibility
    /// with trained weights. The key insight is that features must match exactly
    /// between training and inference.
    ///
    /// # Feature Types
    ///
    /// - `bias` - Always present, allows label-specific bias
    /// - `word.lower={word}` - Lowercased word identity
    /// - `word.shape={shape}` - Word shape (Xx, X, x, 0)
    /// - `word.isdigit={bool}` - Whether all digits
    /// - `word.istitle={bool}` - Whether titlecase
    /// - `word.isupper={bool}` - Whether all uppercase
    /// - `prefix{n}={chars}` - First n characters
    /// - `suffix{n}={chars}` - Last n characters
    /// - `-1:word.lower={word}` - Previous word features
    /// - `+1:word.lower={word}` - Next word features
    /// - `BOS` / `EOS` - Beginning/end of sentence markers
    fn extract_features(&self, tokens: &[&str], pos: usize, _prev_label: &str) -> Vec<String> {
        let mut features = Vec::with_capacity(20);
        let word = tokens[pos];

        fn bool_py(v: bool) -> &'static str {
            if v {
                "True"
            } else {
                "False"
            }
        }

        // Bias feature (always present)
        features.push("bias".to_string());

        // Word identity features
        features.push(format!("word.lower={}", word.to_lowercase()));
        features.push(format!("word.shape={}", Self::word_shape(word)));
        features.push(format!(
            "word.isdigit={}",
            // Match Python `str.isdigit()` behavior used in `scripts/train_crf_weights.py`:
            // - empty string -> False
            // - Unicode-aware digits -> True
            bool_py(!word.is_empty() && word.chars().all(Self::is_digit_py))
        ));
        features.push(format!(
            "word.istitle={}",
            bool_py(
                word.chars().next().is_some_and(|c| c.is_uppercase())
                    && word.chars().skip(1).all(|c| c.is_lowercase())
            )
        ));
        features.push(format!(
            "word.isupper={}",
            bool_py(word.chars().all(|c| c.is_uppercase()))
        ));

        // Prefix/suffix features
        let chars: Vec<char> = word.chars().collect();
        if chars.len() >= 2 {
            let prefix2: String = chars[..2].iter().collect();
            let suffix2: String = chars[chars.len() - 2..].iter().collect();
            features.push(format!("prefix2={}", prefix2.to_lowercase()));
            features.push(format!("suffix2={}", suffix2.to_lowercase()));
        }
        if chars.len() >= 3 {
            let prefix3: String = chars[..3].iter().collect();
            let suffix3: String = chars[chars.len() - 3..].iter().collect();
            features.push(format!("prefix3={}", prefix3.to_lowercase()));
            features.push(format!("suffix3={}", suffix3.to_lowercase()));
        }

        // Context features (previous word)
        if pos > 0 {
            let prev_word = tokens[pos - 1];
            features.push(format!("-1:word.lower={}", prev_word.to_lowercase()));
            features.push(format!(
                "-1:word.istitle={}",
                bool_py(
                    prev_word.chars().next().is_some_and(|c| c.is_uppercase())
                        && prev_word.chars().skip(1).all(|c| c.is_lowercase())
                )
            ));
            features.push(format!(
                "-1:word.isupper={}",
                bool_py(prev_word.chars().all(|c| c.is_uppercase()))
            ));
            features.push(format!("-1:word.shape={}", Self::word_shape(prev_word)));
        } else {
            features.push("BOS".to_string());
        }

        // Context features (next word)
        if pos + 1 < tokens.len() {
            let next_word = tokens[pos + 1];
            features.push(format!("+1:word.lower={}", next_word.to_lowercase()));
            features.push(format!(
                "+1:word.istitle={}",
                bool_py(
                    next_word.chars().next().is_some_and(|c| c.is_uppercase())
                        && next_word.chars().skip(1).all(|c| c.is_lowercase())
                )
            ));
            features.push(format!(
                "+1:word.isupper={}",
                bool_py(next_word.chars().all(|c| c.is_uppercase()))
            ));
            features.push(format!("+1:word.shape={}", Self::word_shape(next_word)));
        } else {
            features.push("EOS".to_string());
        }

        // Gazetteer features (kept for backwards compatibility)
        for template in &self.templates {
            if let FeatureTemplate::InGazetteer(entity_type) = template {
                if let Some(gaz) = self.gazetteers.get(entity_type) {
                    if gaz.iter().any(|g| g.eq_ignore_ascii_case(word)) {
                        features.push(format!("gaz:{}", entity_type.as_label()));
                    }
                }
            }
        }

        features
    }

    /// Score a label for given features using learned weights.
    fn score_label(&self, features: &[String], label: &str) -> f64 {
        let mut score = 0.0;
        let debug = std::env::var("CRF_DEBUG").is_ok();

        if debug && label == "I-PER" {
            eprintln!("  Features for I-PER: {:?}", features);
        }

        for feat in features {
            let key = format!("{}:{}", feat, label);
            if let Some(&w) = self.weights.get(&key) {
                if debug && w.abs() > 0.1 {
                    eprintln!("  CRF: {} -> {:.2}", key, w);
                }
                score += w;
            }
            // Also check feature alone (type-independent)
            if let Some(&w) = self.weights.get(feat) {
                score += w * 0.5;
            }
        }
        // Bias towards O for unknown tokens (no features matched)
        if label == "O" {
            score += 0.5; // Small default bias toward O
        }
        score
    }

    /// Viterbi decoding to find best label sequence.
    fn viterbi_decode(&self, tokens: &[&str]) -> Vec<String> {
        if tokens.is_empty() {
            return vec![];
        }

        let n = tokens.len();
        let m = self.labels.len();

        // Dynamic programming tables
        let mut scores = vec![vec![f64::NEG_INFINITY; m]; n];
        let mut backpointers = vec![vec![0usize; m]; n];

        // Initialize first position
        let features = self.extract_features(tokens, 0, "O");
        for (j, label) in self.labels.iter().enumerate() {
            scores[0][j] = self.score_label(&features, label);
        }

        // Forward pass
        for i in 1..n {
            for (j, label) in self.labels.iter().enumerate() {
                let mut best_score = f64::NEG_INFINITY;
                let mut best_prev = 0;

                for (k, prev_label) in self.labels.iter().enumerate() {
                    let features = self.extract_features(tokens, i, prev_label);
                    let trans_key = format!("trans:{}->{}", prev_label, label);
                    let trans_score = self.weights.get(&trans_key).copied().unwrap_or(0.0);
                    let score = scores[i - 1][k] + self.score_label(&features, label) + trans_score;

                    if score > best_score {
                        best_score = score;
                        best_prev = k;
                    }
                }

                scores[i][j] = best_score;
                backpointers[i][j] = best_prev;
            }
        }

        // Backward pass to recover best path
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

        path.iter().map(|&j| self.labels[j].clone()).collect()
    }

    /// Convert BIO labels to entities.
    ///
    /// Note: Uses SpanConverter for correct byte-to-char offset conversion.
    /// Entity offsets are CHARACTER offsets, not byte offsets.
    ///
    /// Uses token position tracking to correctly handle duplicate entity texts.
    /// The previous implementation used `text.find()` which always returned the
    /// first occurrence, causing incorrect offsets for duplicate entities.
    fn labels_to_entities(&self, text: &str, tokens: &[&str], labels: &[String]) -> Vec<Entity> {
        use crate::offset::SpanConverter;

        let mut entities = Vec::new();

        // Build converter once for all byte-to-char conversions
        let converter = SpanConverter::new(text);

        // Track token positions (byte offsets) as we iterate
        let token_positions: Vec<(usize, usize)> = Self::calculate_token_positions(text, tokens);

        let mut current_entity: Option<(usize, usize, EntityType, Vec<&str>)> = None;

        for (i, (token, label)) in tokens.iter().zip(labels.iter()).enumerate() {
            if label.starts_with("B-") {
                // Save previous entity if any
                if let Some((start_idx, end_idx, entity_type, words)) = current_entity.take() {
                    Self::push_entity_from_positions(
                        &converter,
                        &token_positions,
                        start_idx,
                        end_idx,
                        &words,
                        entity_type,
                        &mut entities,
                    );
                }

                // Start new entity
                let entity_type = match label.as_str() {
                    "B-PER" => EntityType::Person,
                    "B-ORG" => EntityType::Organization,
                    "B-LOC" => EntityType::Location,
                    _ => EntityType::Other("MISC".to_string()),
                };
                current_entity = Some((i, i, entity_type, vec![token]));
            } else if label.starts_with("I-") {
                // Continue current entity
                if let Some((_, ref mut end_idx, _, ref mut words)) = current_entity {
                    words.push(token);
                    *end_idx = i;
                }
            } else {
                // O label - save and reset
                if let Some((start_idx, end_idx, entity_type, words)) = current_entity.take() {
                    Self::push_entity_from_positions(
                        &converter,
                        &token_positions,
                        start_idx,
                        end_idx,
                        &words,
                        entity_type,
                        &mut entities,
                    );
                }
            }
        }

        // Don't forget last entity
        if let Some((start_idx, end_idx, entity_type, words)) = current_entity.take() {
            Self::push_entity_from_positions(
                &converter,
                &token_positions,
                start_idx,
                end_idx,
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

    /// Helper to create entity with correct character offsets using token positions.
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
            &entity_text,
            entity_type,
            char_start,
            char_end,
            0.7, // CRF confidence is hard to calibrate
        ));
    }

    /// Simple whitespace tokenizer.
    fn tokenize(text: &str) -> Vec<&str> {
        text.split_whitespace().collect()
    }
}

impl Model for CrfNER {
    fn extract_entities(&self, text: &str, _language: Option<&str>) -> Result<Vec<Entity>> {
        if text.trim().is_empty() {
            return Ok(vec![]);
        }

        let tokens = Self::tokenize(text);
        if tokens.is_empty() {
            return Ok(vec![]);
        }

        let labels = self.viterbi_decode(&tokens);
        let entities = self.labels_to_entities(text, &tokens, &labels);

        Ok(entities)
    }

    fn supported_types(&self) -> Vec<EntityType> {
        vec![
            EntityType::Person,
            EntityType::Organization,
            EntityType::Location,
            EntityType::Other("MISC".to_string()),
        ]
    }

    fn is_available(&self) -> bool {
        true // Always available (no external dependencies)
    }

    fn name(&self) -> &'static str {
        "crf"
    }

    fn description(&self) -> &'static str {
        "CRF-based NER (classical statistical method)"
    }
}

impl crate::NamedEntityCapable for CrfNER {}

impl crate::BatchCapable for CrfNER {
    fn optimal_batch_size(&self) -> Option<usize> {
        Some(32) // CRF is fast, can handle batches
    }
}

impl crate::StreamingCapable for CrfNER {
    fn recommended_chunk_size(&self) -> usize {
        4096 // Smaller chunks since CRF is token-based
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crf_basic() {
        let ner = CrfNER::new();
        let entities = ner
            .extract_entities("John Smith works at Google in California", None)
            .unwrap();

        // With our default heuristic weights + gazetteers, we should usually get some entities.
        // (Trained weights will do better, but defaults should not be totally dead.)
        assert!(!entities.is_empty(), "Expected some entities, got none");
    }

    #[test]
    fn test_word_shape() {
        assert_eq!(CrfNER::word_shape("John"), "Xx");
        assert_eq!(CrfNER::word_shape("USA"), "X");
        assert_eq!(CrfNER::word_shape("hello"), "x");
        assert_eq!(CrfNER::word_shape("123"), "0");
        assert_eq!(CrfNER::word_shape("Hello123"), "Xx0");
    }

    #[test]
    fn test_tokenize() {
        let tokens = CrfNER::tokenize("Hello world");
        assert_eq!(tokens, vec!["Hello", "world"]);
    }

    #[test]
    fn test_empty_input() {
        let ner = CrfNER::new();
        let entities = ner.extract_entities("", None).unwrap();
        assert!(entities.is_empty());
    }

    #[test]
    fn test_gazetteer_lookup() {
        let ner = CrfNER::new();

        // Gazetteer should contain common entities
        assert!(ner.gazetteers[&EntityType::Person].contains(&"John".to_string()));
        assert!(ner.gazetteers[&EntityType::Location].contains(&"California".to_string()));
        assert!(ner.gazetteers[&EntityType::Organization].contains(&"Google".to_string()));
    }

    #[test]
    fn test_viterbi_returns_valid_labels() {
        let ner = CrfNER::new();
        let tokens = vec!["John", "works", "at", "Google"];
        let labels = ner.viterbi_decode(&tokens);

        assert_eq!(labels.len(), tokens.len());
        for label in &labels {
            assert!(ner.labels.contains(label));
        }
    }

    #[test]
    fn test_common_verbs_not_in_entities() {
        let ner = CrfNER::new();

        // Test that common verbs don't get tagged as part of entities
        let entities = ner
            .extract_entities("John Smith works at Apple", None)
            .unwrap();

        // Should find John Smith and Apple, but NOT "works"
        let entity_texts: Vec<&str> = entities.iter().map(|e| e.text.as_str()).collect();
        for entity_text in &entity_texts {
            assert!(
                !entity_text.contains("works"),
                "Entity '{}' should not contain 'works'",
                entity_text
            );
        }
    }

    #[test]
    fn test_weights_for_common_words() {
        let ner = CrfNER::new();

        // Check that weights exist for common stop words
        assert!(
            ner.weights.contains_key("word.lower=works:O"),
            "Missing weight for word.lower=works:O"
        );
        assert!(
            ner.weights.contains_key("word.lower=works:I-PER"),
            "Missing weight for word.lower=works:I-PER"
        );

        // Check that O weight is positive and I-* weight is negative
        let o_weight = *ner.weights.get("word.lower=works:O").unwrap();
        let i_per_weight = *ner.weights.get("word.lower=works:I-PER").unwrap();
        assert!(
            o_weight > 0.0,
            "O weight should be positive, got {}",
            o_weight
        );
        assert!(
            i_per_weight < 0.0,
            "I-PER weight should be negative, got {}",
            i_per_weight
        );
    }

    #[test]
    fn test_unicode_char_offsets() {
        // Test that entity offsets are character-based, not byte-based
        let ner = CrfNER::new();

        // "北京" is 2 chars, 6 bytes. "Beijing" is 7 chars, 7 bytes.
        // Text "北京 Beijing" is 10 chars, 14 bytes.
        let text = "北京 Beijing";
        assert_eq!(text.len(), 14, "Expected 14 bytes");
        assert_eq!(text.chars().count(), 10, "Expected 10 characters");

        let entities = ner.extract_entities(text, None).unwrap();

        // Regardless of what entities are found, check all offsets are valid char offsets
        let char_count = text.chars().count();
        for entity in &entities {
            assert!(
                entity.start <= entity.end,
                "Invalid span: start {} > end {}",
                entity.start,
                entity.end
            );
            assert!(
                entity.end <= char_count,
                "Entity end {} exceeds char count {} for text {:?}",
                entity.end,
                char_count,
                text
            );

            // Also verify we can extract the text at those offsets
            let extracted: String = text
                .chars()
                .skip(entity.start)
                .take(entity.end - entity.start)
                .collect();
            assert!(
                !extracted.is_empty() || entity.start == entity.end,
                "Empty extraction for entity at {}..{} in {:?}",
                entity.start,
                entity.end,
                text
            );
        }
    }

    #[test]
    fn test_multilingual_inputs_no_panic_and_valid_spans() {
        let ner = CrfNER::new();
        let texts = [
            // Latin
            "Marie Curie discovered radium in Paris.",
            // CJK
            "習近平在北京會見了普京。",
            // Arabic (RTL)
            "التقى محمد بن سلمان بالرئيس في الرياض",
            // Cyrillic
            "Путин встретился с Си Цзиньпином в Москве.",
            // Devanagari
            "प्रधान मंत्री शर्मा दिल्ली में मिले।",
        ];

        for text in texts {
            let entities = ner.extract_entities(text, None).unwrap();
            let char_count = text.chars().count();
            for e in entities {
                assert!(e.start <= e.end);
                assert!(e.end <= char_count);
                let _span: String = text.chars().skip(e.start).take(e.end - e.start).collect();
            }
        }
    }

    /// Test that duplicate entity texts get correct offsets.
    #[test]
    fn test_duplicate_entity_offsets() {
        // Test token position calculation directly
        let text = "Google bought Google for $1 billion.";
        let tokens: Vec<&str> = text.split_whitespace().collect();
        let positions = CrfNER::calculate_token_positions(text, &tokens);

        // First "Google" at byte 0-6
        assert_eq!(
            positions[0],
            (0, 6),
            "First 'Google' should be at bytes 0-6"
        );
        // Second "Google" at byte 14-20
        assert_eq!(
            positions[2],
            (14, 20),
            "Second 'Google' should be at bytes 14-20"
        );
    }

    /// Test token position calculation with Unicode.
    #[test]
    fn test_token_positions_unicode() {
        let text = "東京 Tokyo 東京";
        let tokens: Vec<&str> = text.split_whitespace().collect();
        let positions = CrfNER::calculate_token_positions(text, &tokens);

        // Each 東京 is 6 bytes (2 chars × 3 bytes each)
        assert_eq!(positions[0], (0, 6), "First '東京' at bytes 0-6");
        assert_eq!(positions[1], (7, 12), "Tokyo at bytes 7-12");
        assert_eq!(positions[2], (13, 19), "Second '東京' at bytes 13-19");
    }
}
