//! Mention-Ranking Coreference Resolution.
//!
//! A simpler alternative to E2E-Coref that uses external mention detection
//! (from NER/parser) and ranks antecedent candidates.
//!
//! # Research Foundations
//!
//! This module is primarily a practical implementation of mention-ranking coreference:
//! score candidate antecedents for each mention, then cluster by transitive closure.
//!
//! Where this module cites papers, treat those citations as *context* for ideas that are
//! instantiated here (feature hooks, configuration defaults). If a comment cannot be
//! traced to a cited source, it should be removed rather than treated as authoritative.
//!
//! # Clinical heuristics (inspired by clinical-coref literature)
//!
//! This implementation includes optional heuristics commonly discussed in clinical-coref
//! settings (acronym expansion, “be-phrase” patterns, local context filtering). When
//! enabled, they should be validated on your target dataset; defaults aim to be conservative.
//!
//! ## "Be Phrase" Detection
//!
//! Identity patterns like "Resolution of X is Y" strongly indicate coreference.
//! From the paper: "if there is a 'be phrase' between two concepts of the same
//! type, they are probably saying 'something is something'."
//!
//! Enabled via [`MentionRankingConfig::enable_be_phrase_detection`].
//!
//! ## Acronym Matching
//!
//! Medical acronyms reliably link to their expansions:
//! - "MRSA" ↔ "Methicillin-resistant Staphylococcus aureus"
//! - "CHF" ↔ "Congestive Heart Failure"
//!
//! From the paper: "The first letters of each word in concepts that have two
//! or more words are taken and compared to whole words in other concepts."
//!
//! Enabled via [`MentionRankingConfig::enable_acronym_matching`].
//!
//! ## Context-Based Link Filtering
//!
//! Different dates/locations suggest different entities. From the paper:
//! "eliminate links that actually refer to two different entities based on
//! clues found in the sentences surrounding the mentions."
//!
//! Enabled via [`MentionRankingConfig::enable_context_filtering`].
//!
//! ## Synonym matching
//!
//! This module supports synonym-aware matching via pluggable sources (see code), but
//! avoids shipping large hardcoded domain synonym tables by default.
//!
//! ## Clinical Configuration
//!
//! Use [`MentionRankingConfig::clinical()`] for clinical/biomedical text:
//!
//! ```rust
//! use anno::backends::coref::mention_ranking::{MentionRankingConfig, MentionRankingCoref};
//!
//! let config = MentionRankingConfig::clinical();
//! let coref = MentionRankingCoref::with_config(config);
//!
//! let text = "The patient is John Smith. Pt was admitted with MRSA.";
//! let clusters = coref.resolve(text).unwrap();
//! ```
//!
//! # Long-document notes
//!
//! Long-document coreference is difficult. The long-doc literature is a good source of
//! evaluation benchmarks and error modes, but this implementation does not aim to
//! reproduce specific reported numbers in its doc comments.
//!
//! # Historical Context
//!
//! Coreference resolution approaches evolved through distinct paradigms:
//!
//! ```text
//! 1995-2010  Rule-based: Hobbs algorithm, centering theory
//! 1997       Kehler: Probabilistic coref with Dempster-Shafer (IE context)
//! 2010-2016  Mention-pair: Classify (m_i, m_j) independently
//! 2013-2017  Mention-ranking: Rank antecedents for each mention
//! 2017+      E2E-Coref: Joint mention detection + clustering
//! 2022       G2GT: Graph refinement with global decisions
//! 2024       Maverick: Efficient E2E with 500M params
//! ```
//!
//! Mention-ranking sits between mention-pair (too independent) and E2E
//! (too complex). It's still valuable for:
//! - Interpretable, feature-based debugging
//! - Fast inference without GPU
//! - Scenarios with good external mention detection
//!
//! ## Configuration-level uncertainty
//!
//! Some classic probabilistic formulations treat coreference as a distribution over
//! clusterings/configurations. This implementation is greedy and does not attempt to
//! represent full configuration uncertainty.
//!
//! ## Graph refinement (separate implementation)
//!
//! If you want iterative/global graph refinement, use the dedicated graph-coref backend
//! (separate module) rather than treating this mention-ranking implementation as equivalent.
//!
//! # Architecture
//!
//! ```text
//! Input: "John saw Mary. He waved."
//!    ↓
//! ┌─────────────────────────────────────────────────────────┐
//! │ 1. External Mention Detection                           │
//! │    Use NER/parser to find NPs, pronouns, named entities │
//! │    Mentions: [John, Mary, He]                          │
//! └─────────────────────────────────────────────────────────┘
//!    ↓
//! ┌─────────────────────────────────────────────────────────┐
//! │ 2. Mention Representation                               │
//! │    Extract features for each mention:                   │
//! │    - Surface form, head word                            │
//! │    - Type (pronoun, proper, nominal)                    │
//! │    - Gender, number, animacy                            │
//! └─────────────────────────────────────────────────────────┘
//!    ↓
//! ┌─────────────────────────────────────────────────────────┐
//! │ 3. Antecedent Ranking                                   │
//! │    For each mention, rank all previous mentions         │
//! │    Features: string match, distance, type compatibility │
//! │    Link to highest-scoring antecedent above threshold   │
//! └─────────────────────────────────────────────────────────┘
//!    ↓
//! ┌─────────────────────────────────────────────────────────┐
//! │ 4. Clustering                                           │
//! │    Group linked mentions into clusters via transitivity │
//! └─────────────────────────────────────────────────────────┘
//!    ↓
//! Output: {[John, He], [Mary]}
//! ```
//!
//! # Compared to other approaches
//!
//! Mention-ranking is typically simpler than end-to-end span models and can be faster to
//! debug and iterate on. For accuracy claims, rely on the evaluation harness and dataset
//! reports rather than prose numbers in docs.
//!
//! # References
//!
//! - NeuralCoref (HuggingFace): <https://github.com/huggingface/neuralcoref>
//! - Clark & Manning 2016: "Deep Reinforcement Learning for Mention-Ranking Coreference Models"
//! - Miculicich & Henderson 2022: "Graph Refinement for Coreference Resolution"
//!   [arXiv:2203.16574](https://arxiv.org/abs/2203.16574)
//!
//! # Salience Integration
//!
//! Entity salience (importance) can inform coreference decisions:
//! - Salient entities are mentioned more often (stronger evidence)
//! - Linking to salient antecedents is more likely correct
//! - Helps break ties between equally-scored candidates
//!
//! Use `with_salience` to provide pre-computed salience scores. Two approaches:
//!
//! **Option 1: TextRank/YAKE salience** (keyword-based)
//!
//! ```rust,ignore
//! use anno::salience::{EntityRanker, TextRankSalience};
//! use anno::backends::coref::mention_ranking::MentionRankingCoref;
//!
//! let ranker = TextRankSalience::default();
//! let ranked = ranker.rank(text, &entities);
//! let salience_scores: HashMap<String, f64> = ranked.into_iter()
//!     .map(|(e, score)| (e.text.to_lowercase(), score))
//!     .collect();
//!
//! let coref = MentionRankingCoref::new()
//!     .with_salience(salience_scores);
//! ```
//!
//! **Option 2: Chain-feature salience** (uses mention frequency, spread, type)
//!
//! ```rust,ignore
//! use anno::salience::features_to_salience_scores;
//! use anno::backends::coref::mention_ranking::MentionRankingCoref;
//!
//! let salience_scores = features_to_salience_scores(text, &entities);
//! let coref = MentionRankingCoref::new()
//!     .with_salience(salience_scores);
//! ```

#[allow(unused_imports)]
use crate::{CoreferenceResolver, Gender, MentionType};
#[allow(unused_imports)]
use crate::{Entity, Model, Result};
#[allow(unused_imports)]
use std::collections::{HashMap, HashSet};

pub mod types;
pub use types::*;

pub mod algorithm;
pub use algorithm::MentionRankingCoref;
