//! Conditional Random Field (CRF) NER backend.
//!
//! Implements classical statistical NER using CRF sequence labeling.
//! This provides a lightweight, interpretable baseline that requires no
//! external dependencies or GPU acceleration.
//!
//! # History
//!
//! CRF-based NER was a common baseline throughout the 2000s (pre-neural sequence labeling):
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
//! # Trained Parameters
//!
//! Bundled weights in `crf_weights.json` (28k features) are trained via `python-crfsuite`
//! on WikiANN EN (15k sentences, L1=0.1, L2=0.1, 100 iterations). Includes shape, affix,
//! casing, context features, and transition weights. Word identity features are limited
//! to a top-2000 vocab to keep the file shippable. Labels: PER, ORG, LOC.
//!
//! To retrain on a different dataset:
//! ```sh
//! uv run scripts/train_crf_weights.py --dataset <hf_dataset> --config <config>
//! ```
//!
//! Requires the `bundled-crf-weights` feature to use trained weights; otherwise
//! falls back to hand-tuned heuristic weights.
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
//! Nuance: CoNLL-2003’s English text is derived from Reuters/RCV1 and is commonly treated as
//! redistribution-restricted. The CoNLL site notes that, “because of copyright reasons we only
//! make available the annotations” and that you need separate access to the Reuters corpus to
//! build the full dataset: `http://www.clips.uantwerpen.be/conll2003/ner/`.
//!
//! Practical consequence: `anno` includes a training script, but it does not ship a CoNLL-trained
//! `crf_weights.json` out of the box.
//!
//! # Advantages Over Neural Methods
//!
//! - **Interpretable**: Features and weights are human-readable
//! - **Fast training**: CPU-only training; typically faster to iterate than neural training loops
//! - **No dependencies**: Pure Rust, no ONNX/Candle required
//! - **Deterministic**: Same input always produces same output
//! - **Small footprint**: Small weights file compared to ML model artifacts

use crate::{Entity, EntityCategory, EntityType, Language, Model, Result};
use std::collections::HashMap;
#[cfg(feature = "bundled-crf-weights")]
use std::sync::OnceLock;

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

mod algorithm;

/// Find sentence boundary character offsets in text.
///
/// A sentence boundary is `. `, `! `, or `? ` followed by an uppercase letter.
/// Returns the character offset of the punctuation mark (the split point).
fn sentence_boundary_offsets(text: &str) -> Vec<usize> {
    let chars: Vec<char> = text.chars().collect();
    let mut boundaries = Vec::new();
    for i in 0..chars.len().saturating_sub(2) {
        if matches!(chars[i], '.' | '!' | '?')
            && chars[i + 1].is_whitespace()
            && chars.get(i + 2).is_some_and(|c| c.is_uppercase())
        {
            boundaries.push(i);
        }
    }
    boundaries
}

/// Clip entities that cross sentence boundaries.
///
/// If an entity span contains a sentence boundary (`.` + whitespace + uppercase),
/// truncate the entity to end before the boundary. Removes entities that become
/// empty after clipping.
fn clip_entities_at_sentence_boundaries(text: &str, entities: &mut Vec<Entity>) {
    let boundaries = sentence_boundary_offsets(text);
    if boundaries.is_empty() {
        return;
    }

    entities.retain_mut(|e| {
        for &b in &boundaries {
            // Boundary is inside the entity span
            if b > e.start && b < e.end {
                // Truncate entity to end at the boundary
                e.end = b;
                // Rebuild entity text from the truncated span
                let new_text: String = text.chars().skip(e.start).take(e.end - e.start).collect();
                e.text = new_text.trim_end().to_string();
                if e.text.is_empty() || e.start >= e.end {
                    return false; // remove empty entity
                }
            }
        }
        true
    });
}

// CRF algorithm: feature extraction, Viterbi decoding, weight loading (see algorithm.rs).
impl Model for CrfNER {
    fn extract_entities(&self, text: &str, _language: Option<Language>) -> Result<Vec<Entity>> {
        if text.trim().is_empty() {
            return Ok(vec![]);
        }

        let tokens = Self::tokenize(text);
        if tokens.is_empty() {
            return Ok(vec![]);
        }

        let labels = self.viterbi_decode(&tokens);
        let mut entities = self.labels_to_entities(text, &tokens, &labels);

        // Post-process: truncate entities that cross sentence boundaries.
        // Sentence boundaries are detected as ". " followed by an uppercase letter,
        // or "! " / "? " followed by an uppercase letter.
        clip_entities_at_sentence_boundaries(text, &mut entities);

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
        true // Always available (no external dependencies)
    }

    fn name(&self) -> &'static str {
        "crf"
    }

    fn description(&self) -> &'static str {
        "CRF-based NER (classical statistical method)"
    }

    fn capabilities(&self) -> crate::ModelCapabilities {
        crate::ModelCapabilities {
            batch_capable: true,
            optimal_batch_size: Some(32),
            streaming_capable: true,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests;
