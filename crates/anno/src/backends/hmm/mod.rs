//! Hidden Markov Model (HMM) NER backend.
//!
//! Implements classical statistical NER using HMMs, the dominant approach
//! from the 1990s before CRFs became popular. This was the **first statistical
//! approach** to NER, replacing rule-based systems.
//!
//! # Historical Context
//!
//! NER first appeared at MUC-6 (1996), where Grishman & Sundheim defined
//! the task of identifying people, organizations, and locations. Early
//! systems were rule-based (lexicons, hand-crafted patterns). HMMs brought
//! statistical learning to NER:
//!
//! ```text
//! 1987  MUC-1: First IE conference (no formal NER task)
//! 1996  MUC-6: NER formally defined (PER, ORG, LOC)
//! 1997  Nymble (Bikel et al.): early HMM NER system
//! 1998  BBN IdentiFinder: HMM-based, MUC-7 benchmark
//! 2001  CRFs introduced (Lafferty et al.) — HMMs become a common comparison baseline
//! ```
//!
//! # Architecture
//!
//! ```text
//! Input: "John works at Google"
//!    ↓
//! ┌─────────────────────────────────────────────────────────┐
//! │ Hidden States (NER Tags)                                │
//! │                                                         │
//! │  B-PER ──> O ──> O ──> B-ORG                           │
//! │    │       │     │       │                              │
//! │    ↓       ↓     ↓       ↓                              │
//! │  John   works   at   Google                             │
//! │                                                         │
//! │ Observed Emissions                                      │
//! └─────────────────────────────────────────────────────────┘
//!
//! P(tags | words) ∝ P(tags) × P(words | tags)
//!                 = ∏ P(tag_i | tag_{i-1}) × P(word_i | tag_i)
//! ```
//!
//! # HMM Components
//!
//! 1. **States**: BIO tags (B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, O)
//! 2. **Observations**: Words in the text
//! 3. **Transition Probabilities**: P(tag_i | tag_{i-1})
//! 4. **Emission Probabilities**: P(word | tag)
//! 5. **Initial Probabilities**: P(tag | start)
//!
//! # Mathematical Formulation
//!
//! HMMs are **generative models** that model the joint probability:
//!
//! ```text
//! P(x, y) = P(y) × P(x | y)
//!         = P(y_1) × ∏_{t=2}^{T} P(y_t | y_{t-1})    // transitions
//!                 × ∏_{t=1}^{T} P(x_t | y_t)          // emissions
//! ```
//!
//! Decoding uses the **Viterbi algorithm** (dynamic programming) to find
//! the most likely state sequence in O(T × |S|²) time.
//!
//! # History
//!
//! - Rabiner (1989): "A Tutorial on Hidden Markov Models" (foundational)
//! - Bikel et al. 1997: "Nymble: A High-Performance Learning Name-finder"
//! - BBN IdentiFinder: One of the first HMM-based NER systems
//! - Often replaced by CRFs for NER in the 2000s, but still useful as a baseline/teaching model
//!
//! # Why HMMs Often Underperform CRFs (for NER)
//!
//! | Aspect | HMM | CRF |
//! |--------|-----|-----|
//! | Model Type | Generative | Discriminative |
//! | Features | Word identity only | Arbitrary features |
//! | Context | First-order Markov | Arbitrary windows |
//! | Label Bias | Inherent | Solved |
//! | Performance | task-dependent | task-dependent |
//!
//! HMMs are typically used with relatively limited emission features. CRFs can use arbitrary
//! feature functions (capitalization, prefixes/suffixes, gazetteers, etc.) while remaining a
//! globally normalized conditional model.
//!
//! # References
//!
//! - Rabiner (1989): "A Tutorial on Hidden Markov Models and Selected
//!   Applications in Speech Recognition" (Proceedings of IEEE)
//! - Bikel, Miller, Schwartz, Weischedel (1997): "Nymble: A High-Performance
//!   Learning Name-finder" (ANLP)
//! - Bikel, Schwartz, Weischedel (1999): "An Algorithm that Learns What's
//!   in a Name" (Machine Learning)
//!
//! # Trained Parameters
//!
//! Bundled parameters in `hmm_params.json` are trained on WikiANN EN (20k sentences)
//! via `scripts/train_hmm_params.py`. The training produces initial probabilities,
//! transition probabilities, and a compact emission backoff table (word features, no
//! word identity) with Laplace smoothing. Labels: PER, ORG, LOC (no MISC in WikiANN).
//!
//! To retrain on a different dataset:
//! ```sh
//! uv run scripts/train_hmm_params.py --dataset <hf_dataset> --config <config>
//! ```
//!
//! Requires the `bundled-hmm-params` feature to use trained parameters; otherwise
//! falls back to hand-tuned heuristic weights.
//!
//! # See Also
//!
//! - CRF-style sequence models (`backends/crf.rs`)

use crate::{Entity, EntityCategory, EntityType, Language, Model, Result};
use std::collections::HashMap;

#[cfg(feature = "bundled-hmm-params")]
use std::sync::OnceLock;

#[cfg(feature = "bundled-hmm-params")]
use serde_json as _;

#[derive(Debug, Clone)]
struct HmmParams {
    states: Vec<String>,
    initial: Vec<f64>,
    transitions: Vec<Vec<f64>>,
    backoff: serde_json::Value,
}

#[derive(Debug, Clone)]
struct HmmBackoff {
    /// len_bucket -> probs per state index (aligned with `states`)
    len: HashMap<String, Vec<f64>>,
    /// boolean feature -> P(feature_present | state) per state index
    bool_present: HashMap<String, Vec<f64>>,
    /// Stable list of boolean features to include absent probabilities.
    bool_keys: Vec<String>,
}

/// HMM configuration.
#[derive(Debug, Clone)]
pub struct HmmConfig {
    /// Smoothing parameter for unseen words.
    pub smoothing: f64,
    /// Use log probabilities for numerical stability.
    pub use_log_probs: bool,
    /// Optional penalty applied to non-O emissions when using bundled backoff.
    ///
    /// Values < 1.0 reduce spurious entities; values > 1.0 increase recall but may over-tag.
    pub non_o_emission_scale: f64,
    /// If true, prefer bundled priors/transitions (when available) instead of heuristic dynamics.
    pub use_bundled_dynamics: bool,
}

impl Default for HmmConfig {
    fn default() -> Self {
        Self {
            smoothing: 1e-10,
            use_log_probs: true,
            // Tuned to reduce spurious tagging when bundled params are enabled.
            non_o_emission_scale: 0.5,
            // When bundled params are available, use their dynamics by default.
            // This keeps the "trained" path genuinely end-to-end.
            use_bundled_dynamics: true,
        }
    }
}

/// Hidden Markov Model for NER.
///
/// This implements a first-order HMM (bigram) for sequence labeling.
/// Uses the Viterbi algorithm for decoding.
#[derive(Debug)]
pub struct HmmNER {
    /// Configuration.
    config: HmmConfig,
    /// State labels (BIO tags).
    states: Vec<String>,
    /// State to index mapping.
    state_to_idx: HashMap<String, usize>,
    /// Transition probabilities: P(state_j | state_i)
    /// transitions\[i\]\[j\] = P(j | i)
    transitions: Vec<Vec<f64>>,
    /// Initial state probabilities: P(state | start)
    initial: Vec<f64>,
    /// Emission probabilities: P(word | state)
    /// Key: (state_idx, word), Value: probability
    emissions: HashMap<(usize, String), f64>,
    /// Vocabulary for unknown word handling.
    #[allow(dead_code)] // Reserved for OOV handling
    vocab: HashMap<String, usize>,
    /// Optional bundled emission backoff (small, trained).
    backoff: Option<HmmBackoff>,
}

mod algorithm;
// HMM Viterbi, forward-backward, and emission scoring: see algorithm.rs.
impl Model for HmmNER {
    fn extract_entities(&self, text: &str, _language: Option<Language>) -> Result<Vec<Entity>> {
        if text.trim().is_empty() {
            return Ok(vec![]);
        }

        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return Ok(vec![]);
        }

        let label_indices = self.viterbi(&words);
        let entities = self.decode_entities(text, &words, &label_indices);

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
        true
    }

    fn capabilities(&self) -> crate::ModelCapabilities {
        // HMM has no specialized batch or streaming impl; all defaults are false.
        crate::ModelCapabilities::default()
    }
}

#[cfg(test)]
mod tests;
