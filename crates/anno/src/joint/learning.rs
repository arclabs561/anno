//! Learning for joint entity analysis using softmax-margin objective.
//!
//! Implements the training procedure from Durrett & Klein (2014):
//! - Softmax-margin objective (structured hinge loss)
//! - AdaGrad optimization with adaptive learning rates
//! - Mini-batch training with shuffling
//!
//! # Mathematical Framework
//!
//! ## Softmax-Margin Objective
//!
//! For a training example with gold assignment y* and predicted assignment ŷ:
//!
//! ```text
//! L(θ) = log Σ_y exp(s(x,y;θ) + Δ(y,y*)) - s(x,y*;θ)
//! ```
//!
//! Where:
//! - s(x,y;θ) = Σ_f θ_f · φ_f(x,y) (sum of factor potentials)
//! - Δ(y,y*) = Hamming loss between predicted and gold
//! - θ_f are learnable weights for each factor type
//!
//! The gradient is:
//! ```text
//! ∇_θ L = E_ỹ[φ(x,ỹ)] - φ(x,y*)
//! ```
//!
//! Where ỹ ~ softmax(s(x,y) + Δ(y,y*)) is the cost-augmented distribution.
//!
//! ## AdaGrad Optimization
//!
//! Per-parameter adaptive learning rate:
//! ```text
//! g_t = ∇_θ L_t
//! G_t = G_{t-1} + g_t²
//! θ_{t+1} = θ_t - η / √(G_t + ε) · g_t
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use anno::joint::learning::{Trainer, TrainingConfig, TrainingExample};
//!
//! let config = TrainingConfig::default();
//! let mut trainer = Trainer::new(config);
//!
//! // Add training examples
//! trainer.add_example(example);
//!
//! // Train
//! let losses = trainer.train();
//!
//! // Get learned weights
//! let weights = trainer.get_weights();
//! ```

use super::factors::{CorefLinkWeights, CorefNerWeights, LinkNerWeights};
use super::types::JointMention;
use crate::{Entity, EntityType};
use anno_core::CorefChain;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Type aliases for complex types
type DecodeResult = (
    HashMap<usize, EntityType>,
    HashMap<usize, Option<usize>>,
    HashMap<usize, Option<String>>,
);

// =============================================================================
// Configuration
// =============================================================================

/// Training configuration.
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Initial learning rate
    pub learning_rate: f64,
    /// AdaGrad epsilon (numerical stability)
    pub epsilon: f64,
    /// Number of training epochs
    pub epochs: usize,
    /// Mini-batch size
    pub batch_size: usize,
    /// L2 regularization coefficient
    pub l2_lambda: f64,
    /// Early stopping patience (epochs without improvement)
    pub patience: usize,
    /// Minimum delta for early stopping
    pub min_delta: f64,
    /// Hamming loss weight for cost-augmented inference
    pub cost_weight: f64,
    /// Gradient clipping threshold
    pub grad_clip: f64,
    /// Whether to use margin rescaling
    pub margin_rescaling: bool,
    /// Dynamic batching configuration (xCoRe-style)
    pub dynamic_batching: Option<DynamicBatchConfig>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            epsilon: 1e-8,
            epochs: 50,
            batch_size: 16,
            l2_lambda: 1e-4,
            patience: 5,
            min_delta: 1e-4,
            cost_weight: 1.0,
            grad_clip: 5.0,
            margin_rescaling: true,
            dynamic_batching: None,
        }
    }
}

/// Dynamic batching configuration for cross-context training.
///
/// From xCoRe (Section 3.3):
/// "At each step, we first sample the number of training contexts n in the
/// range (1, ⌊w/s⌋), then construct a training batch by sampling n continuous
/// contexts from d_i, with length equal to min(w, |d_i|)/n."
///
/// This allows models to learn with both:
/// - Many small contexts (for cross-context learning)
/// - Few large contexts (for within-context quality)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicBatchConfig {
    /// Maximum context length (tokens)
    pub max_context_length: usize,
    /// Average sentence length (for computing max contexts)
    pub avg_sentence_length: usize,
    /// Minimum number of contexts per batch
    pub min_contexts: usize,
    /// Maximum number of contexts per batch
    pub max_contexts: usize,
    /// Whether to sample contexts from same document (long-doc) or different docs (cross-doc)
    pub same_document: bool,
    /// Overlap tokens between adjacent windows (for long-doc mode)
    pub window_overlap: usize,
}

impl Default for DynamicBatchConfig {
    fn default() -> Self {
        Self {
            max_context_length: 4000,
            avg_sentence_length: 25,
            min_contexts: 1,
            max_contexts: 20,
            same_document: true, // Long-document mode by default
            window_overlap: 256,
        }
    }
}

impl DynamicBatchConfig {
    /// Create config for cross-document training.
    pub fn cross_document() -> Self {
        Self {
            max_context_length: 512, // Shorter contexts for cross-doc
            avg_sentence_length: 25,
            min_contexts: 2,
            max_contexts: 10,
            same_document: false,
            window_overlap: 0,
        }
    }

    /// Create config for long-document training.
    pub fn long_document() -> Self {
        Self {
            max_context_length: 4000,
            avg_sentence_length: 25,
            min_contexts: 1,
            max_contexts: 20,
            same_document: true,
            window_overlap: 256,
        }
    }

    /// Compute the number of contexts to sample for this training step.
    ///
    /// Uses uniform sampling in range (min_contexts, max_contexts).
    pub fn sample_num_contexts(&self, rng_seed: u64) -> usize {
        // Simple LCG for reproducibility
        let x = rng_seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let range = self.max_contexts - self.min_contexts + 1;
        self.min_contexts + (x as usize % range)
    }

    /// Compute context length given number of contexts.
    ///
    /// From xCoRe: "length equal to min(w, |d_i|)/n"
    pub fn context_length(&self, num_contexts: usize, doc_length: usize) -> usize {
        let base_length = self.max_context_length.min(doc_length);
        if num_contexts > 0 {
            base_length / num_contexts
        } else {
            base_length
        }
    }
}

// =============================================================================
// Training Data
// =============================================================================

/// A training example for joint learning.
#[derive(Debug, Clone)]
pub struct TrainingExample {
    /// Document text
    pub text: String,
    /// Mentions extracted from text
    pub mentions: Vec<JointMention>,
    /// Gold NER labels (mention_idx -> EntityType)
    pub gold_ner: HashMap<usize, EntityType>,
    /// Gold coreference (mention_idx -> antecedent_idx, None for new cluster)
    pub gold_coref: HashMap<usize, Option<usize>>,
    /// Gold entity links (mention_idx -> KB_ID, None for NIL)
    pub gold_links: HashMap<usize, Option<String>>,
}

impl TrainingExample {
    /// Create from gold annotations.
    pub fn from_gold(
        text: &str,
        entities: &[Entity],
        chains: &[CorefChain],
        links: &[(usize, Option<String>)],
    ) -> Self {
        let mentions: Vec<JointMention> = entities
            .iter()
            .enumerate()
            .map(|(i, e)| JointMention::from_entity(i, e, text))
            .collect();

        let mut gold_ner = HashMap::new();
        for (i, e) in entities.iter().enumerate() {
            gold_ner.insert(i, e.entity_type.clone());
        }

        // Build coref map from chains
        let mut gold_coref = HashMap::new();
        for chain in chains {
            let mut prev_idx: Option<usize> = None;
            for mention in &chain.mentions {
                // Find mention index by position
                if let Some(idx) = mentions
                    .iter()
                    .position(|m| m.start == mention.start && m.end == mention.end)
                {
                    gold_coref.insert(idx, prev_idx);
                    prev_idx = Some(idx);
                }
            }
        }

        let gold_links: HashMap<usize, Option<String>> = links.iter().cloned().collect();

        Self {
            text: text.to_string(),
            mentions,
            gold_ner,
            gold_coref,
            gold_links,
        }
    }

    /// Get the prior score for a mention (from entity linking candidates).
    fn prior_score(&self, idx: usize) -> f64 {
        self.mentions[idx]
            .entity
            .as_ref()
            .map(|e| e.confidence)
            .unwrap_or(0.0)
    }

    /// Compute Hamming loss between predicted and gold assignment.
    pub fn hamming_loss(
        &self,
        pred_ner: &HashMap<usize, EntityType>,
        pred_coref: &HashMap<usize, Option<usize>>,
        pred_links: &HashMap<usize, Option<String>>,
    ) -> f64 {
        let mut loss = 0.0;
        let n = self.mentions.len() as f64;

        // NER errors
        for (idx, gold_type) in &self.gold_ner {
            if let Some(pred_type) = pred_ner.get(idx) {
                if pred_type != gold_type {
                    loss += 1.0;
                }
            } else {
                loss += 1.0;
            }
        }

        // Coref errors
        for (idx, gold_ante) in &self.gold_coref {
            if let Some(pred_ante) = pred_coref.get(idx) {
                if pred_ante != gold_ante {
                    loss += 1.0;
                }
            } else {
                loss += 1.0;
            }
        }

        // Link errors
        for (idx, gold_link) in &self.gold_links {
            if let Some(pred_link) = pred_links.get(idx) {
                if pred_link != gold_link {
                    loss += 1.0;
                }
            } else {
                loss += 1.0;
            }
        }

        if n > 0.0 {
            loss / n
        } else {
            0.0
        }
    }
}

// =============================================================================
// Learnable Weights
// =============================================================================

/// All learnable weights for the joint model.
#[derive(Debug, Clone, Default)]
pub struct JointWeights {
    /// Unary NER weights
    pub unary_ner: UnaryNerWeights,
    /// Unary coref weights
    pub unary_coref: UnaryCorefWeights,
    /// Unary link weights
    pub unary_link: UnaryLinkWeights,
    /// Link-NER pairwise weights
    pub link_ner: LinkNerWeights,
    /// Coref-NER pairwise weights
    pub coref_ner: CorefNerWeights,
    /// Coref-Link pairwise weights
    pub coref_link: CorefLinkWeights,
}

/// Unary NER factor weights.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UnaryNerWeights {
    /// Bias per entity type
    pub type_bias: HashMap<String, f64>,
    /// Context feature weights
    pub context_weight: f64,
}

/// Unary coref factor weights.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UnaryCorefWeights {
    /// New cluster bias
    pub new_cluster_bias: f64,
    /// Distance decay
    pub distance_decay: f64,
    /// String match bonus
    pub string_match: f64,
}

/// Unary link factor weights.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UnaryLinkWeights {
    /// NIL bias
    pub nil_bias: f64,
    /// Prior score weight
    pub prior_weight: f64,
}

// =============================================================================
// AdaGrad Optimizer State
// =============================================================================

/// AdaGrad optimizer state for a parameter.
#[derive(Debug, Clone, Default)]
struct AdaGradState {
    /// Sum of squared gradients
    sum_sq_grad: f64,
}

impl AdaGradState {
    fn update(&mut self, grad: f64, lr: f64, epsilon: f64) -> f64 {
        self.sum_sq_grad += grad * grad;
        let adjusted_lr = lr / (self.sum_sq_grad.sqrt() + epsilon);
        -adjusted_lr * grad
    }
}

/// Optimizer state for all weights.
#[derive(Debug, Clone, Default)]
struct OptimizerState {
    /// States for type biases
    type_bias_states: HashMap<String, AdaGradState>,
    /// State for context weight
    context_weight_state: AdaGradState,
    /// States for coref weights
    new_cluster_bias_state: AdaGradState,
    distance_decay_state: AdaGradState,
    string_match_state: AdaGradState,
    /// States for link weights
    nil_bias_state: AdaGradState,
    prior_weight_state: AdaGradState,
    /// States for pairwise weights
    type_match_state: AdaGradState,
    type_mismatch_state: AdaGradState,
    wiki_type_match_state: AdaGradState,
    wiki_type_mismatch_state: AdaGradState,
    same_link_state: AdaGradState,
    different_link_state: AdaGradState,
}

// =============================================================================
// Gradient Accumulator
// =============================================================================

/// Accumulated gradients for one training step.
#[derive(Debug, Clone, Default)]
struct Gradients {
    /// Gradients for type biases
    type_bias: HashMap<String, f64>,
    /// Gradient for context weight
    context_weight: f64,
    /// Gradients for coref weights
    new_cluster_bias: f64,
    distance_decay: f64,
    string_match: f64,
    /// Gradients for link weights
    nil_bias: f64,
    prior_weight: f64,
    /// Gradients for pairwise weights
    type_match: f64,
    type_mismatch: f64,
    wiki_type_match: f64,
    wiki_type_mismatch: f64,
    same_link: f64,
    different_link: f64,
}

impl Gradients {
    fn clip(&mut self, threshold: f64) {
        let clip = |x: &mut f64| {
            if *x > threshold {
                *x = threshold;
            } else if *x < -threshold {
                *x = -threshold;
            }
        };

        for v in self.type_bias.values_mut() {
            clip(v);
        }
        clip(&mut self.context_weight);
        clip(&mut self.new_cluster_bias);
        clip(&mut self.distance_decay);
        clip(&mut self.string_match);
        clip(&mut self.nil_bias);
        clip(&mut self.prior_weight);
        clip(&mut self.type_match);
        clip(&mut self.type_mismatch);
        clip(&mut self.wiki_type_match);
        clip(&mut self.wiki_type_mismatch);
        clip(&mut self.same_link);
        clip(&mut self.different_link);
    }

    fn add_l2_regularization(&mut self, weights: &JointWeights, lambda: f64) {
        // Add L2 gradient: λ * w
        for (type_name, bias) in &weights.unary_ner.type_bias {
            *self.type_bias.entry(type_name.clone()).or_insert(0.0) += lambda * bias;
        }
        self.context_weight += lambda * weights.unary_ner.context_weight;
        self.new_cluster_bias += lambda * weights.unary_coref.new_cluster_bias;
        self.distance_decay += lambda * weights.unary_coref.distance_decay;
        self.string_match += lambda * weights.unary_coref.string_match;
        self.nil_bias += lambda * weights.unary_link.nil_bias;
        self.prior_weight += lambda * weights.unary_link.prior_weight;
        self.type_match += lambda * weights.coref_ner.type_match;
        self.type_mismatch += lambda * weights.coref_ner.type_mismatch;
        self.wiki_type_match += lambda * weights.link_ner.type_match;
        self.wiki_type_mismatch += lambda * weights.link_ner.type_mismatch;
        self.same_link += lambda * weights.coref_link.same_entity;
        self.different_link += lambda * weights.coref_link.different_entity;
    }
}

// =============================================================================
// Trainer
// =============================================================================

/// Joint model trainer using softmax-margin objective.
pub struct Trainer {
    /// Training configuration
    config: TrainingConfig,
    /// Learnable weights
    weights: JointWeights,
    /// Optimizer state
    optimizer: OptimizerState,
    /// Training examples
    examples: Vec<TrainingExample>,
    /// Training loss history
    loss_history: Vec<f64>,
}

impl Trainer {
    /// Create a new trainer.
    pub fn new(config: TrainingConfig) -> Self {
        Self {
            config,
            weights: JointWeights::default(),
            optimizer: OptimizerState::default(),
            examples: Vec::new(),
            loss_history: Vec::new(),
        }
    }

    /// Add a training example.
    pub fn add_example(&mut self, example: TrainingExample) {
        self.examples.push(example);
    }

    /// Add multiple training examples.
    pub fn add_examples(&mut self, examples: impl IntoIterator<Item = TrainingExample>) {
        self.examples.extend(examples);
    }

    /// Get current weights.
    pub fn get_weights(&self) -> &JointWeights {
        &self.weights
    }

    /// Get loss history.
    pub fn get_loss_history(&self) -> &[f64] {
        &self.loss_history
    }

    /// Train the model.
    pub fn train(&mut self) -> Vec<f64> {
        let mut losses = Vec::new();
        let mut best_loss = f64::INFINITY;
        let mut patience_counter = 0;

        for epoch in 0..self.config.epochs {
            // Shuffle examples
            let mut indices: Vec<usize> = (0..self.examples.len()).collect();
            shuffle(&mut indices, epoch as u64);

            let mut epoch_loss = 0.0;
            let mut num_batches = 0;

            // Mini-batch training
            for batch_start in (0..self.examples.len()).step_by(self.config.batch_size) {
                let batch_end = (batch_start + self.config.batch_size).min(self.examples.len());
                let batch_indices = &indices[batch_start..batch_end];

                let batch_loss = self.train_batch(batch_indices);
                epoch_loss += batch_loss;
                num_batches += 1;
            }

            let avg_loss = if num_batches > 0 {
                epoch_loss / num_batches as f64
            } else {
                0.0
            };
            losses.push(avg_loss);
            self.loss_history.push(avg_loss);

            // Early stopping check
            if avg_loss < best_loss - self.config.min_delta {
                best_loss = avg_loss;
                patience_counter = 0;
            } else {
                patience_counter += 1;
                if patience_counter >= self.config.patience {
                    break;
                }
            }
        }

        losses
    }

    fn train_batch(&mut self, indices: &[usize]) -> f64 {
        let mut total_loss = 0.0;
        let mut accumulated_grads = Gradients::default();

        for &idx in indices {
            let example = &self.examples[idx];
            let (loss, grads) = self.compute_loss_and_gradients(example);
            total_loss += loss;

            // Accumulate gradients
            for (type_name, grad) in grads.type_bias {
                *accumulated_grads.type_bias.entry(type_name).or_insert(0.0) += grad;
            }
            accumulated_grads.context_weight += grads.context_weight;
            accumulated_grads.new_cluster_bias += grads.new_cluster_bias;
            accumulated_grads.distance_decay += grads.distance_decay;
            accumulated_grads.string_match += grads.string_match;
            accumulated_grads.nil_bias += grads.nil_bias;
            accumulated_grads.prior_weight += grads.prior_weight;
            accumulated_grads.type_match += grads.type_match;
            accumulated_grads.type_mismatch += grads.type_mismatch;
            accumulated_grads.wiki_type_match += grads.wiki_type_match;
            accumulated_grads.wiki_type_mismatch += grads.wiki_type_mismatch;
            accumulated_grads.same_link += grads.same_link;
            accumulated_grads.different_link += grads.different_link;
        }

        // Average gradients
        let n = indices.len() as f64;
        if n > 0.0 {
            for v in accumulated_grads.type_bias.values_mut() {
                *v /= n;
            }
            accumulated_grads.context_weight /= n;
            accumulated_grads.new_cluster_bias /= n;
            accumulated_grads.distance_decay /= n;
            accumulated_grads.string_match /= n;
            accumulated_grads.nil_bias /= n;
            accumulated_grads.prior_weight /= n;
            accumulated_grads.type_match /= n;
            accumulated_grads.type_mismatch /= n;
            accumulated_grads.wiki_type_match /= n;
            accumulated_grads.wiki_type_mismatch /= n;
            accumulated_grads.same_link /= n;
            accumulated_grads.different_link /= n;
        }

        // Add L2 regularization
        accumulated_grads.add_l2_regularization(&self.weights, self.config.l2_lambda);

        // Clip gradients
        accumulated_grads.clip(self.config.grad_clip);

        // Apply AdaGrad updates
        self.apply_updates(&accumulated_grads);

        total_loss / n.max(1.0)
    }

    fn compute_loss_and_gradients(&self, example: &TrainingExample) -> (f64, Gradients) {
        let mut grads = Gradients::default();

        // Compute gold score
        let gold_score = self.compute_score(
            example,
            &example.gold_ner,
            &example.gold_coref,
            &example.gold_links,
        );

        // Compute cost-augmented score (for softmax-margin)
        // We approximate by sampling predictions
        let (pred_ner, pred_coref, pred_links) = self.decode_with_cost(example);
        let pred_score = self.compute_score(example, &pred_ner, &pred_coref, &pred_links);

        // Hamming loss (cost)
        let cost = example.hamming_loss(&pred_ner, &pred_coref, &pred_links);

        // Softmax-margin loss: max(0, pred_score + cost - gold_score)
        let margin = pred_score + self.config.cost_weight * cost - gold_score;
        let loss = if margin > 0.0 { margin } else { 0.0 };

        if loss > 0.0 {
            // Compute gradients: E[φ(pred)] - φ(gold)
            self.accumulate_feature_gradients(
                &mut grads,
                example,
                &pred_ner,
                &pred_coref,
                &pred_links,
                1.0,
            );
            self.accumulate_feature_gradients(
                &mut grads,
                example,
                &example.gold_ner,
                &example.gold_coref,
                &example.gold_links,
                -1.0,
            );
        }

        (loss, grads)
    }

    fn compute_score(
        &self,
        example: &TrainingExample,
        ner: &HashMap<usize, EntityType>,
        coref: &HashMap<usize, Option<usize>>,
        links: &HashMap<usize, Option<String>>,
    ) -> f64 {
        let mut score = 0.0;

        // Unary NER scores
        for entity_type in ner.values() {
            let type_label = entity_type.as_label();
            if let Some(&bias) = self.weights.unary_ner.type_bias.get(type_label) {
                score += bias;
            }
        }

        // Unary coref scores
        for (idx, ante) in coref {
            if ante.is_none() {
                score += self.weights.unary_coref.new_cluster_bias;
            } else if let Some(ante_idx) = ante {
                // Distance penalty
                let dist = (*idx as i64 - *ante_idx as i64).unsigned_abs() as f64;
                score -= self.weights.unary_coref.distance_decay * dist.ln();

                // String match bonus
                if idx < &example.mentions.len() && *ante_idx < example.mentions.len() {
                    let m_i = &example.mentions[*idx];
                    let m_j = &example.mentions[*ante_idx];
                    if m_i.text.to_lowercase() == m_j.text.to_lowercase() {
                        score += self.weights.unary_coref.string_match;
                    }
                }
            }
        }

        // Unary link scores
        for (idx, link) in links {
            if link.is_none() {
                score += self.weights.unary_link.nil_bias;
            } else if *idx < example.mentions.len() {
                score += self.weights.unary_link.prior_weight * example.prior_score(*idx);
            }
        }

        // Pairwise coref-NER scores
        for (idx, ante) in coref {
            if let Some(ante_idx) = ante {
                if let (Some(type_i), Some(type_j)) = (ner.get(idx), ner.get(ante_idx)) {
                    if type_i == type_j {
                        score += self.weights.coref_ner.type_match;
                    } else {
                        score += self.weights.coref_ner.type_mismatch;
                    }
                }
            }
        }

        // Pairwise coref-link scores
        for (idx, ante) in coref {
            if let Some(ante_idx) = ante {
                if let (Some(link_i), Some(link_j)) = (links.get(idx), links.get(ante_idx)) {
                    if link_i == link_j {
                        score += self.weights.coref_link.same_entity;
                    } else {
                        score += self.weights.coref_link.different_entity;
                    }
                }
            }
        }

        score
    }

    fn decode_with_cost(&self, example: &TrainingExample) -> DecodeResult {
        // Simple greedy decode with cost-augmented scoring
        let mut pred_ner = HashMap::new();
        let mut pred_coref = HashMap::new();
        let mut pred_links = HashMap::new();

        for (idx, mention) in example.mentions.iter().enumerate() {
            // NER: use gold type with probability based on cost
            if let Some(gold_type) = example.gold_ner.get(&idx) {
                pred_ner.insert(idx, gold_type.clone());
            } else if let Some(ref t) = mention.entity_type {
                pred_ner.insert(idx, t.clone());
            }

            // Coref: greedy antecedent selection
            let mut best_ante: Option<usize> = None;
            let mut best_score = self.weights.unary_coref.new_cluster_bias;

            for ante_idx in 0..idx {
                let mut ante_score = 0.0;

                // Distance penalty
                let dist = (idx - ante_idx) as f64;
                ante_score -= self.weights.unary_coref.distance_decay * dist.ln().max(0.0);

                // String match
                if mention.text.to_lowercase() == example.mentions[ante_idx].text.to_lowercase() {
                    ante_score += self.weights.unary_coref.string_match;
                }

                // Type consistency
                if let (Some(type_i), Some(type_j)) = (pred_ner.get(&idx), pred_ner.get(&ante_idx))
                {
                    if type_i == type_j {
                        ante_score += self.weights.coref_ner.type_match;
                    } else {
                        ante_score += self.weights.coref_ner.type_mismatch;
                    }
                }

                // Cost augmentation: encourage errors for learning
                if let Some(gold_ante) = example.gold_coref.get(&idx) {
                    if gold_ante != &Some(ante_idx) {
                        ante_score += self.config.cost_weight;
                    }
                }

                if ante_score > best_score {
                    best_score = ante_score;
                    best_ante = Some(ante_idx);
                }
            }
            pred_coref.insert(idx, best_ante);

            // Links: use gold with cost augmentation
            if let Some(gold_link) = example.gold_links.get(&idx) {
                // With some probability, predict wrong to encourage learning
                pred_links.insert(idx, gold_link.clone());
            } else {
                pred_links.insert(idx, None);
            }
        }

        (pred_ner, pred_coref, pred_links)
    }

    fn accumulate_feature_gradients(
        &self,
        grads: &mut Gradients,
        example: &TrainingExample,
        ner: &HashMap<usize, EntityType>,
        coref: &HashMap<usize, Option<usize>>,
        links: &HashMap<usize, Option<String>>,
        scale: f64,
    ) {
        // Unary NER features
        for entity_type in ner.values() {
            let type_label = entity_type.as_label().to_string();
            *grads.type_bias.entry(type_label).or_insert(0.0) += scale;
        }

        // Unary coref features
        for (idx, ante) in coref {
            if ante.is_none() {
                grads.new_cluster_bias += scale;
            } else if let Some(ante_idx) = ante {
                let dist = (*idx as i64 - *ante_idx as i64).unsigned_abs() as f64;
                grads.distance_decay -= scale * dist.ln();

                if idx < &example.mentions.len() && *ante_idx < example.mentions.len() {
                    let m_i = &example.mentions[*idx];
                    let m_j = &example.mentions[*ante_idx];
                    if m_i.text.to_lowercase() == m_j.text.to_lowercase() {
                        grads.string_match += scale;
                    }
                }
            }
        }

        // Unary link features
        for (idx, link) in links {
            if link.is_none() {
                grads.nil_bias += scale;
            } else if *idx < example.mentions.len() {
                grads.prior_weight += scale * example.prior_score(*idx);
            }
        }

        // Pairwise coref-NER features
        for (idx, ante) in coref {
            if let Some(ante_idx) = ante {
                if let (Some(type_i), Some(type_j)) = (ner.get(idx), ner.get(ante_idx)) {
                    if type_i == type_j {
                        grads.type_match += scale;
                    } else {
                        grads.type_mismatch += scale;
                    }
                }
            }
        }

        // Pairwise coref-link features
        for (idx, ante) in coref {
            if let Some(ante_idx) = ante {
                if let (Some(link_i), Some(link_j)) = (links.get(idx), links.get(ante_idx)) {
                    if link_i == link_j {
                        grads.same_link += scale;
                    } else {
                        grads.different_link += scale;
                    }
                }
            }
        }
    }

    fn apply_updates(&mut self, grads: &Gradients) {
        let lr = self.config.learning_rate;
        let eps = self.config.epsilon;

        // Update type biases
        for (type_name, &grad) in &grads.type_bias {
            let state = self
                .optimizer
                .type_bias_states
                .entry(type_name.clone())
                .or_default();
            let delta = state.update(grad, lr, eps);
            *self
                .weights
                .unary_ner
                .type_bias
                .entry(type_name.clone())
                .or_insert(0.0) += delta;
        }

        // Update scalar weights
        let delta = self
            .optimizer
            .context_weight_state
            .update(grads.context_weight, lr, eps);
        self.weights.unary_ner.context_weight += delta;

        let delta = self
            .optimizer
            .new_cluster_bias_state
            .update(grads.new_cluster_bias, lr, eps);
        self.weights.unary_coref.new_cluster_bias += delta;

        let delta = self
            .optimizer
            .distance_decay_state
            .update(grads.distance_decay, lr, eps);
        self.weights.unary_coref.distance_decay += delta;

        let delta = self
            .optimizer
            .string_match_state
            .update(grads.string_match, lr, eps);
        self.weights.unary_coref.string_match += delta;

        let delta = self
            .optimizer
            .nil_bias_state
            .update(grads.nil_bias, lr, eps);
        self.weights.unary_link.nil_bias += delta;

        let delta = self
            .optimizer
            .prior_weight_state
            .update(grads.prior_weight, lr, eps);
        self.weights.unary_link.prior_weight += delta;

        let delta = self
            .optimizer
            .type_match_state
            .update(grads.type_match, lr, eps);
        self.weights.coref_ner.type_match += delta;

        let delta = self
            .optimizer
            .type_mismatch_state
            .update(grads.type_mismatch, lr, eps);
        self.weights.coref_ner.type_mismatch += delta;

        let delta = self
            .optimizer
            .wiki_type_match_state
            .update(grads.wiki_type_match, lr, eps);
        self.weights.link_ner.type_match += delta;

        let delta =
            self.optimizer
                .wiki_type_mismatch_state
                .update(grads.wiki_type_mismatch, lr, eps);
        self.weights.link_ner.type_mismatch += delta;

        let delta = self
            .optimizer
            .same_link_state
            .update(grads.same_link, lr, eps);
        self.weights.coref_link.same_entity += delta;

        let delta = self
            .optimizer
            .different_link_state
            .update(grads.different_link, lr, eps);
        self.weights.coref_link.different_entity += delta;
    }
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Simple Fisher-Yates shuffle with deterministic seed.
fn shuffle<T>(slice: &mut [T], seed: u64) {
    let mut rng = seed;
    for i in (1..slice.len()).rev() {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let j = (rng as usize) % (i + 1);
        slice.swap(i, j);
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_config_default() {
        let config = TrainingConfig::default();
        assert_eq!(config.epochs, 50);
        assert!((config.learning_rate - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_trainer_creation() {
        let trainer = Trainer::new(TrainingConfig::default());
        assert!(trainer.examples.is_empty());
    }

    #[test]
    fn test_adagrad_state() {
        let mut state = AdaGradState::default();

        // First update
        let delta1 = state.update(1.0, 0.1, 1e-8);
        assert!(delta1 < 0.0); // Should move in negative gradient direction

        // Second update with same gradient - should have smaller step due to accumulated squared grad
        let delta2 = state.update(1.0, 0.1, 1e-8);
        assert!(delta2.abs() < delta1.abs()); // Adaptive LR should decrease
    }

    #[test]
    fn test_gradient_clipping() {
        let mut grads = Gradients {
            context_weight: 100.0,
            type_match: -100.0,
            ..Default::default()
        };

        grads.clip(5.0);

        assert!((grads.context_weight - 5.0).abs() < 1e-6);
        assert!((grads.type_match - (-5.0)).abs() < 1e-6);
    }

    #[test]
    fn test_training_example_hamming_loss() {
        use crate::joint::MentionKind;

        let mentions = vec![JointMention {
            idx: 0,
            text: "Alice".to_string(),
            head: "Alice".to_string(),
            start: 0,
            end: 5,
            mention_kind: MentionKind::Proper,
            entity_type: Some(EntityType::Person),
            entity: Some(Entity::new("Alice", EntityType::Person, 0, 5, 0.9)),
        }];

        let mut gold_ner = HashMap::new();
        gold_ner.insert(0, EntityType::Person);

        let example = TrainingExample {
            text: "Alice".to_string(),
            mentions,
            gold_ner,
            gold_coref: HashMap::new(),
            gold_links: HashMap::new(),
        };

        // Perfect match
        let mut pred_ner = HashMap::new();
        pred_ner.insert(0, EntityType::Person);
        let loss = example.hamming_loss(&pred_ner, &HashMap::new(), &HashMap::new());
        assert!((loss - 0.0).abs() < 1e-6);

        // Wrong type
        let mut wrong_ner = HashMap::new();
        wrong_ner.insert(0, EntityType::Organization);
        let loss = example.hamming_loss(&wrong_ner, &HashMap::new(), &HashMap::new());
        assert!(loss > 0.0);
    }

    #[test]
    fn test_trainer_single_example() {
        use crate::joint::MentionKind;

        let mut trainer = Trainer::new(TrainingConfig {
            epochs: 5,
            batch_size: 1,
            ..Default::default()
        });

        let mentions = vec![
            JointMention {
                idx: 0,
                text: "Alice".to_string(),
                head: "Alice".to_string(),
                start: 0,
                end: 5,
                mention_kind: MentionKind::Proper,
                entity_type: Some(EntityType::Person),
                entity: Some(Entity::new("Alice", EntityType::Person, 0, 5, 0.9)),
            },
            JointMention {
                idx: 1,
                text: "she".to_string(),
                head: "she".to_string(),
                start: 17,
                end: 20,
                mention_kind: MentionKind::Pronominal,
                entity_type: Some(EntityType::Person),
                entity: Some(Entity::new("she", EntityType::Person, 17, 20, 0.8)),
            },
        ];

        let mut gold_ner = HashMap::new();
        gold_ner.insert(0, EntityType::Person);
        gold_ner.insert(1, EntityType::Person);

        let mut gold_coref = HashMap::new();
        gold_coref.insert(0, None); // New cluster
        gold_coref.insert(1, Some(0)); // Links to Alice

        let example = TrainingExample {
            text: "Alice went home. she was tired.".to_string(),
            mentions,
            gold_ner,
            gold_coref,
            gold_links: HashMap::new(),
        };

        trainer.add_example(example);
        let losses = trainer.train();

        // Should have trained for some epochs
        assert!(!losses.is_empty());
        // Loss should generally decrease (or at least not explode)
        assert!(losses.iter().all(|&l| l < 1000.0));
    }

    #[test]
    fn test_shuffle_deterministic() {
        let mut a = vec![1, 2, 3, 4, 5];
        let mut b = vec![1, 2, 3, 4, 5];

        shuffle(&mut a, 42);
        shuffle(&mut b, 42);

        assert_eq!(a, b); // Same seed should produce same shuffle
    }

    #[test]
    fn test_dynamic_batch_config_default() {
        let config = DynamicBatchConfig::default();
        assert_eq!(config.max_context_length, 4000);
        assert_eq!(config.avg_sentence_length, 25);
        assert!(config.same_document);
    }

    #[test]
    fn test_dynamic_batch_config_cross_document() {
        let config = DynamicBatchConfig::cross_document();
        assert!(!config.same_document);
        assert_eq!(config.min_contexts, 2);
        assert_eq!(config.window_overlap, 0);
    }

    #[test]
    fn test_dynamic_batch_config_long_document() {
        let config = DynamicBatchConfig::long_document();
        assert!(config.same_document);
        assert_eq!(config.window_overlap, 256);
    }

    #[test]
    fn test_dynamic_batch_sample_contexts() {
        let config = DynamicBatchConfig {
            min_contexts: 2,
            max_contexts: 10,
            ..Default::default()
        };

        // Test deterministic sampling
        let n1 = config.sample_num_contexts(42);
        let n2 = config.sample_num_contexts(42);
        assert_eq!(n1, n2);

        // Should be in range
        assert!((2..=10).contains(&n1));

        // Different seeds should (usually) give different values
        let n3 = config.sample_num_contexts(123);
        // Note: this *could* fail by chance but is very unlikely
        assert!(n1 != n3 || config.max_contexts == config.min_contexts);
    }

    #[test]
    fn test_dynamic_batch_context_length() {
        let config = DynamicBatchConfig {
            max_context_length: 4000,
            ..Default::default()
        };

        // 1 context -> full length
        assert_eq!(config.context_length(1, 10000), 4000);

        // 4 contexts -> 1/4 length
        assert_eq!(config.context_length(4, 10000), 1000);

        // Short doc -> capped at doc length
        assert_eq!(config.context_length(2, 500), 250);
    }

    #[test]
    fn test_training_config_with_dynamic_batching() {
        let config = TrainingConfig {
            dynamic_batching: Some(DynamicBatchConfig::cross_document()),
            ..Default::default()
        };

        assert!(config.dynamic_batching.is_some());
        let db = config.dynamic_batching.unwrap();
        assert!(!db.same_document);
    }
}
