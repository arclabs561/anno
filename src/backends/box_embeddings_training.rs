//! Training system for box embeddings in coreference resolution.
//!
//! This module implements the mathematical framework for learning box embeddings
//! from coreference annotations, based on research from BERE, BoxTE, and UKGE.
//!
//! **Note**: This is the canonical training implementation. The [matryoshka-box](https://github.com/arclabs561/matryoshka-box)
//! research project extends this with matryoshka-specific features (variable dimensions, etc.).
//! For standard box embedding training, use this module. For research extensions, see matryoshka-box.
//!
//! # Mathematical Foundation
//!
//! ## Objective Function
//!
//! We maximize the conditional probability for positive pairs (entities that corefer)
//! and minimize it for negative pairs (entities that don't corefer):
//!
//! ```
//! L = -Σ log P(box_i | box_j) for positive pairs (i,j)
//!     + λ * Σ max(0, margin - P(box_i | box_j)) for negative pairs (i,j)
//! ```
//!
//! Where:
//! - P(box_i | box_j) = Vol(box_i ∩ box_j) / Vol(box_j) (conditional probability)
//! - Positive pairs: entities in the same coreference chain
//! - Negative pairs: entities in different chains
//! - λ: negative sampling weight
//! - margin: minimum separation for negative pairs
//!
//! ## Gradient Computation
//!
//! We use **analytical gradients** (not finite differences) for efficiency:
//!
//! For a box with min = μ - exp(δ)/2, max = μ + exp(δ)/2:
//!
//! ∂Vol/∂μᵢ = 0 (volume doesn't depend on center position)
//! ∂Vol/∂δᵢ = Vol * 1 (since Vol = ∏exp(δᵢ))
//!
//! For intersection volume, gradients flow through each dimension's overlap.
//!
//! ## Box Constraints
//!
//! We enforce mᵢ ≤ Mᵢ for all dimensions using:
//! - Reparameterization: Mᵢ = mᵢ + exp(δᵢ) where δᵢ is the learned parameter

use crate::backends::box_embeddings::BoxEmbedding;
use crate::eval::coref::{CorefChain, CorefDocument};
use crate::Entity;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Trainable Box Embedding
// =============================================================================

/// A trainable box embedding with learnable parameters.
///
/// Uses reparameterization to ensure min ≤ max:
/// - min[i] = μᵢ - exp(δᵢ)/2 (learnable via μᵢ, δᵢ)
/// - max[i] = μᵢ + exp(δᵢ)/2 (learnable via μᵢ, δᵢ)
///
/// This ensures boxes are always valid (min ≤ max).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainableBox {
    /// Mean position in each dimension [d]
    pub mu: Vec<f32>,
    /// Log-width in each dimension [d] (width = exp(delta))
    pub delta: Vec<f32>,
    /// Dimension
    pub dim: usize,
}

impl TrainableBox {
    /// Create a new trainable box.
    ///
    /// # Arguments
    ///
    /// * `mu` - Mean position (center of box)
    /// * `delta` - Log-width (width = exp(delta))
    ///
    /// The box will have:
    /// - min[i] = mu[i] - exp(delta[i]) / 2
    /// - max[i] = mu[i] + exp(delta[i]) / 2
    #[must_use]
    pub fn new(mu: Vec<f32>, delta: Vec<f32>) -> Self {
        assert_eq!(
            mu.len(),
            delta.len(),
            "mu and delta must have same dimension"
        );
        let dim = mu.len();
        Self { mu, delta, dim }
    }

    /// Initialize from a vector embedding.
    ///
    /// Creates a small box around the vector with initial width `init_width`.
    #[must_use]
    pub fn from_vector(vector: &[f32], init_width: f32) -> Self {
        let mu = vector.to_vec();
        let delta: Vec<f32> = vec![init_width.ln(); mu.len()];
        Self::new(mu, delta)
    }

    /// Convert to a BoxEmbedding (for inference).
    #[must_use]
    pub fn to_box(&self) -> BoxEmbedding {
        let min: Vec<f32> = self
            .mu
            .iter()
            .zip(self.delta.iter())
            .map(|(&m, &d)| m - (d.exp() / 2.0))
            .collect();
        let max: Vec<f32> = self
            .mu
            .iter()
            .zip(self.delta.iter())
            .map(|(&m, &d)| m + (d.exp() / 2.0))
            .collect();
        BoxEmbedding::new(min, max)
    }
}

// =============================================================================
// Training Data Structures
// =============================================================================

/// A single training example (one document with coreference chains).
///
/// Each example contains:
/// - Entities: All mentions in the document
/// - Chains: Groups of entities that corefer
#[derive(Debug, Clone)]
pub struct TrainingExample {
    /// All entity mentions in the document
    pub entities: Vec<Entity>,
    /// Coreference chains (groups of entity IDs that refer to the same entity)
    pub chains: Vec<CorefChain>,
}

/// Convert a CorefDocument to a TrainingExample.
///
/// This extracts all mentions as entities and preserves the coreference chains.
impl From<&CorefDocument> for TrainingExample {
    fn from(doc: &CorefDocument) -> Self {
        // Collect all mentions as entities
        let mut entities = Vec::new();
        let mut mention_to_entity_id = HashMap::new();

        for chain in &doc.chains {
            for mention in &chain.mentions {
                // Use character offset as entity ID (unique per document)
                let entity_id = mention.start;

                // Convert mention entity_type (Option<String>) to EntityType
                // Default to Person if type is unknown
                let entity_type = mention
                    .entity_type
                    .as_ref()
                    .and_then(|s| match s.as_str() {
                        "PER" | "Person" | "person" => Some(crate::EntityType::Person),
                        "ORG" | "Organization" | "organization" => {
                            Some(crate::EntityType::Organization)
                        }
                        "LOC" | "Location" | "location" => Some(crate::EntityType::Location),
                        _ => None,
                    })
                    .unwrap_or(crate::EntityType::Person);

                // Create Entity from Mention
                let entity = Entity::new(
                    mention.text.clone(),
                    entity_type,
                    entity_id,
                    mention.end,
                    1.0,
                );

                entities.push(entity);
                mention_to_entity_id.insert((mention.start, mention.end), entity_id);
            }
        }

        // Rebuild chains with same mentions (they already have correct offsets)
        let chains = doc.chains.clone();

        Self { entities, chains }
    }
}

/// Convert multiple CorefDocuments to TrainingExamples.
pub fn coref_documents_to_training_examples(docs: &[CorefDocument]) -> Vec<TrainingExample> {
    docs.iter().map(TrainingExample::from).collect()
}

// =============================================================================
// Training Configuration
// =============================================================================

/// Training configuration for box embeddings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Learning rate
    pub learning_rate: f32,
    /// Weight for negative pairs
    pub negative_weight: f32,
    /// Margin for negative pairs
    pub margin: f32,
    /// L2 regularization weight
    pub regularization: f32,
    /// Number of training epochs
    pub epochs: usize,
    /// Batch size (for mini-batch training)
    pub batch_size: usize,
    /// Warmup epochs (linear increase from 0.1*lr to lr)
    pub warmup_epochs: usize,
    /// Use self-adversarial negative sampling
    pub use_self_adversarial: bool,
    /// Temperature for self-adversarial sampling
    pub adversarial_temperature: f32,
    /// Early stopping patience (stop if no improvement for N epochs)
    pub early_stopping_patience: Option<usize>,
    /// Minimum improvement for early stopping (relative)
    pub early_stopping_min_delta: f32,
    /// Multi-stage training: focus on positives first (epochs), then negatives
    pub positive_focus_epochs: Option<usize>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            negative_weight: 0.5,
            margin: 0.3,
            regularization: 0.0001,
            epochs: 100,
            batch_size: 32,
            warmup_epochs: 10,
            use_self_adversarial: true,
            adversarial_temperature: 1.0,
            early_stopping_patience: Some(10),
            early_stopping_min_delta: 0.001,
            positive_focus_epochs: None,
        }
    }
}

// =============================================================================
// AMSGrad Optimizer State
// =============================================================================

/// AMSGrad optimizer state for a single box.
#[derive(Debug, Clone)]
pub struct AMSGradState {
    /// First moment estimate (m)
    pub m: Vec<f32>,
    /// Second moment estimate (v)
    pub v: Vec<f32>,
    /// Max second moment estimate (v_hat)
    pub v_hat: Vec<f32>,
    /// Iteration counter
    pub t: usize,
    /// Learning rate
    pub lr: f32,
    /// Beta1 (momentum)
    pub beta1: f32,
    /// Beta2 (RMSprop)
    pub beta2: f32,
    /// Epsilon (numerical stability)
    pub epsilon: f32,
}

impl AMSGradState {
    /// Create new AMSGrad state.
    pub fn new(dim: usize, learning_rate: f32) -> Self {
        Self {
            m: vec![0.0; dim],
            v: vec![0.0; dim],
            v_hat: vec![0.0; dim],
            t: 0,
            lr: learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }

    /// Set learning rate.
    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

// =============================================================================
// Trainer
// =============================================================================

/// Trainer for box embeddings.
pub struct BoxEmbeddingTrainer {
    /// Training configuration
    config: TrainingConfig,
    /// Entity ID → TrainableBox mapping
    boxes: HashMap<usize, TrainableBox>,
    /// Entity ID → AMSGradState mapping
    optimizer_states: HashMap<usize, AMSGradState>,
    /// Embedding dimension
    dim: usize,
}

impl BoxEmbeddingTrainer {
    /// Create a new trainer.
    ///
    /// # Arguments
    ///
    /// * `config` - Training configuration
    /// * `dim` - Embedding dimension
    /// * `initial_embeddings` - Optional initial vector embeddings (entity_id → vector)
    pub fn new(
        config: TrainingConfig,
        dim: usize,
        initial_embeddings: Option<HashMap<usize, Vec<f32>>>,
    ) -> Self {
        let mut boxes = HashMap::new();
        let mut optimizer_states = HashMap::new();

        if let Some(embeddings) = initial_embeddings {
            // Initialize from vector embeddings
            for (entity_id, vector) in embeddings {
                assert_eq!(vector.len(), dim);
                let box_embedding = TrainableBox::from_vector(&vector, 0.1);
                boxes.insert(entity_id, box_embedding.clone());
                optimizer_states.insert(entity_id, AMSGradState::new(dim, config.learning_rate));
            }
        }

        Self {
            config,
            boxes,
            optimizer_states,
            dim,
        }
    }

    /// Initialize boxes from entities.
    ///
    /// Creates trainable boxes for all entities, either from provided
    /// vector embeddings or random initialization.
    ///
    /// **Key insight**: For positive pairs (entities that corefer), initialize
    /// boxes to overlap so gradients can flow from the start.
    ///
    /// # Arguments
    ///
    /// * `examples` - Training examples with entities and coreference chains
    /// * `initial_embeddings` - Optional pre-computed vector embeddings (entity_id → vector)
    ///   If provided, boxes are initialized around these vectors. If None, uses smart
    ///   random initialization with shared centers for coreferent entities.
    pub fn initialize_boxes(
        &mut self,
        examples: &[TrainingExample],
        initial_embeddings: Option<&HashMap<usize, Vec<f32>>>,
    ) {
        // Collect all unique entity IDs and build coreference groups
        let mut entity_ids = std::collections::HashSet::new();
        let mut coref_groups: Vec<Vec<usize>> = Vec::new();

        for example in examples {
            for entity in &example.entities {
                let entity_id = entity.start;
                entity_ids.insert(entity_id);
            }

            // Build groups of entities that corefer
            for chain in &example.chains {
                let group: Vec<usize> = chain.mentions.iter().map(|m| m.start).collect();
                if group.len() > 1 {
                    coref_groups.push(group);
                }
            }
        }

        // Initialize boxes
        for &entity_id in &entity_ids {
            // If vector embeddings provided, use them (better initialization)
            if let Some(embeddings) = initial_embeddings {
                if let Some(vector) = embeddings.get(&entity_id) {
                    // Normalize vector to unit length for better initialization
                    let norm: f32 = vector.iter().map(|&x| x * x).sum::<f32>().sqrt();
                    let normalized: Vec<f32> = if norm > 0.0 {
                        vector.iter().map(|&x| x / norm).collect()
                    } else {
                        vector.clone()
                    };

                    // Use larger initial width (0.2) when starting from vectors
                    // This ensures boxes can overlap even if vectors are slightly different
                    let box_embedding = TrainableBox::from_vector(&normalized, 0.2);
                    self.boxes.insert(entity_id, box_embedding.clone());
                    self.optimizer_states.insert(
                        entity_id,
                        AMSGradState::new(self.dim, self.config.learning_rate),
                    );
                    continue;
                }
            }

            // Check if this entity is in a coreference group
            let mut group_center: Option<Vec<f32>> = None;
            let mut in_coref_group = false;

            for group in &coref_groups {
                if group.contains(&entity_id) {
                    // Use a shared center for all entities in the group
                    if group_center.is_none() {
                        group_center = Some(
                            (0..self.dim)
                                .map(|_| (simple_random() - 0.5) * 0.3) // Smaller region for better overlap
                                .collect(),
                        );
                    }
                    in_coref_group = true;
                    break;
                }
            }

            // Initialize: if in coref group, use shared center; otherwise random
            let mu = if let Some(ref center) = group_center {
                // Add very small random offset to shared center (ensures overlap)
                center
                    .iter()
                    .map(|&c| c + (simple_random() - 0.5) * 0.05) // Very small offset
                    .collect()
            } else {
                // Random center, but spread out more to avoid accidental overlap
                (0..self.dim)
                    .map(|_| (simple_random() - 0.5) * 1.0)
                    .collect()
            };

            // Balanced initialization: coreferent entities should overlap significantly
            // Non-coreferent entities should be distinct and compact
            // Initialize with larger width for coreferent entities (to ensure overlap)
            let initial_width = if in_coref_group {
                1.1_f32 // Good width for coreferent entities (ensures overlap but allows learning)
            } else {
                0.18_f32 // Small width for non-coreferent (distinct but not too small)
            };
            let delta: Vec<f32> = vec![initial_width.ln(); self.dim];
            let box_embedding = TrainableBox::new(mu, delta);
            self.boxes.insert(entity_id, box_embedding.clone());
            self.optimizer_states.insert(
                entity_id,
                AMSGradState::new(self.dim, self.config.learning_rate),
            );
        }
    }

    /// Train on a single example.
    fn train_example(&mut self, example: &TrainingExample, epoch: usize) -> f32 {
        let mut total_loss = 0.0;
        let mut num_pairs = 0;

        // Update learning rate with warmup and decay
        let current_lr = get_learning_rate(
            epoch,
            self.config.epochs,
            self.config.learning_rate,
            self.config.warmup_epochs,
        );
        for state in self.optimizer_states.values_mut() {
            state.set_lr(current_lr);
        }

        // Build positive pairs (entities in same chain)
        let mut positive_pairs = Vec::new();
        for chain in &example.chains {
            let mentions: Vec<usize> = chain.mentions.iter().map(|m| m.start).collect();
            for i in 0..mentions.len() {
                for j in (i + 1)..mentions.len() {
                    positive_pairs.push((mentions[i], mentions[j]));
                }
            }
        }

        // Build negative pairs (entities in different chains)
        let mut negative_pairs = Vec::new();
        for i in 0..example.chains.len() {
            for j in (i + 1)..example.chains.len() {
                let chain_i: Vec<usize> =
                    example.chains[i].mentions.iter().map(|m| m.start).collect();
                let chain_j: Vec<usize> =
                    example.chains[j].mentions.iter().map(|m| m.start).collect();
                for &id_i in &chain_i {
                    for &id_j in &chain_j {
                        negative_pairs.push((id_i, id_j));
                    }
                }
            }
        }

        // Accumulate gradients for all pairs
        let mut gradients: HashMap<usize, (Vec<f32>, Vec<f32>)> = HashMap::new();

        // Process positive pairs
        for &(id_a, id_b) in &positive_pairs {
            // Clone boxes for gradient computation
            let box_a = self.boxes.get(&id_a).cloned();
            let box_b = self.boxes.get(&id_b).cloned();

            if let (Some(box_a_ref), Some(box_b_ref)) = (box_a.as_ref(), box_b.as_ref()) {
                let loss = compute_pair_loss(box_a_ref, box_b_ref, true, &self.config);
                total_loss += loss;
                num_pairs += 1;

                // Compute analytical gradients
                let (grad_mu_a, grad_delta_a, grad_mu_b, grad_delta_b) =
                    compute_analytical_gradients(box_a_ref, box_b_ref, true, &self.config);

                // Skip if gradients are invalid
                if grad_mu_a.iter().any(|&x| !x.is_finite())
                    || grad_delta_a.iter().any(|&x| !x.is_finite())
                    || grad_mu_b.iter().any(|&x| !x.is_finite())
                    || grad_delta_b.iter().any(|&x| !x.is_finite())
                {
                    continue;
                }

                // Accumulate gradients
                let entry_a = gradients
                    .entry(id_a)
                    .or_insert_with(|| (vec![0.0; self.dim], vec![0.0; self.dim]));
                for i in 0..self.dim {
                    entry_a.0[i] += grad_mu_a[i];
                    entry_a.1[i] += grad_delta_a[i];
                }

                let entry_b = gradients
                    .entry(id_b)
                    .or_insert_with(|| (vec![0.0; self.dim], vec![0.0; self.dim]));
                for i in 0..self.dim {
                    entry_b.0[i] += grad_mu_b[i];
                    entry_b.1[i] += grad_delta_b[i];
                }
            }
        }

        // Process negative pairs (with self-adversarial sampling if enabled)
        let negative_samples: Vec<(usize, usize)> =
            if self.config.use_self_adversarial && !negative_pairs.is_empty() {
                // Sample based on current predictions
                let num_samples = positive_pairs.len().min(negative_pairs.len());
                let sampled_indices = sample_self_adversarial_negatives(
                    &negative_pairs,
                    &self.boxes,
                    num_samples,
                    self.config.adversarial_temperature,
                );
                sampled_indices
                    .iter()
                    .map(|&idx| negative_pairs[idx])
                    .collect()
            } else {
                // Uniform sampling
                let num_samples = positive_pairs.len().min(negative_pairs.len());
                negative_pairs.into_iter().take(num_samples).collect()
            };

        for &(id_a, id_b) in &negative_samples {
            // Clone boxes for gradient computation
            let box_a = self.boxes.get(&id_a).cloned();
            let box_b = self.boxes.get(&id_b).cloned();

            if let (Some(box_a_ref), Some(box_b_ref)) = (box_a.as_ref(), box_b.as_ref()) {
                let loss = compute_pair_loss(box_a_ref, box_b_ref, false, &self.config);
                total_loss += loss;
                num_pairs += 1;

                // Compute analytical gradients
                let (grad_mu_a, grad_delta_a, grad_mu_b, grad_delta_b) =
                    compute_analytical_gradients(box_a_ref, box_b_ref, false, &self.config);

                // Skip if gradients are invalid
                if grad_mu_a.iter().any(|&x| !x.is_finite())
                    || grad_delta_a.iter().any(|&x| !x.is_finite())
                    || grad_mu_b.iter().any(|&x| !x.is_finite())
                    || grad_delta_b.iter().any(|&x| !x.is_finite())
                {
                    continue;
                }

                // Accumulate gradients
                let entry_a = gradients
                    .entry(id_a)
                    .or_insert_with(|| (vec![0.0; self.dim], vec![0.0; self.dim]));
                for i in 0..self.dim {
                    entry_a.0[i] += grad_mu_a[i];
                    entry_a.1[i] += grad_delta_a[i];
                }

                let entry_b = gradients
                    .entry(id_b)
                    .or_insert_with(|| (vec![0.0; self.dim], vec![0.0; self.dim]));
                for i in 0..self.dim {
                    entry_b.0[i] += grad_mu_b[i];
                    entry_b.1[i] += grad_delta_b[i];
                }
            }
        }

        // Apply accumulated gradients using AMSGrad
        for (entity_id, (grad_mu, grad_delta)) in gradients {
            if let (Some(box_mut), Some(state)) = (
                self.boxes.get_mut(&entity_id),
                self.optimizer_states.get_mut(&entity_id),
            ) {
                box_mut.update_amsgrad(&grad_mu, &grad_delta, state);
            }
        }

        if num_pairs > 0 {
            total_loss / num_pairs as f32
        } else {
            0.0
        }
    }

    /// Train on a dataset with mini-batching and early stopping.
    /// Uses adaptive negative weighting: starts with low weight to learn positives,
    /// then gradually increases to separate negatives.
    pub fn train(&mut self, examples: &[TrainingExample]) -> Vec<f32> {
        let mut losses = Vec::new();
        let mut best_loss = f32::INFINITY;
        let mut patience_counter = 0;

        // Track score gap for adaptive weighting
        let mut score_gap_history = Vec::new();

        for epoch in 0..self.config.epochs {
            // Multi-stage training: focus on positives first, then negatives
            let (avg_pos, avg_neg, _) = self.get_overlap_stats(examples);
            let current_gap = avg_pos - avg_neg;
            score_gap_history.push(current_gap);

            // Determine training stage
            let positive_focus_epochs = self
                .config
                .positive_focus_epochs
                .unwrap_or(self.config.epochs / 3);
            let is_positive_stage = epoch < positive_focus_epochs;

            // Calculate adaptive negative weight based on stage and performance
            let adaptive_negative_weight = if is_positive_stage {
                // Stage 1: Focus on positive learning - low negative weight but not zero
                // Still apply some negative gradients to prevent negative scores from growing too much
                // Gradually increase from 0.2 to 0.3 during positive stage
                let stage_progress = epoch as f32 / positive_focus_epochs as f32;
                self.config.negative_weight * (0.2 + stage_progress * 0.1)
            } else if avg_pos > 0.05 && avg_neg > 0.3 {
                // Stage 2: Positive learning is good but negatives are too high - aggressive separation
                // Increase negative weight more aggressively
                let progress = ((epoch - positive_focus_epochs) as f32
                    / (self.config.epochs - positive_focus_epochs) as f32)
                    .min(1.0);
                // Scale based on how bad negatives are - more aggressive
                let neg_penalty = (avg_neg / 0.4).min(1.0); // Penalty factor for high negatives (lower threshold)
                self.config.negative_weight * (0.7 + progress * 0.8 + neg_penalty * 0.4).min(2.0)
            // Up to 2.0x
            } else if avg_pos > 0.02 && current_gap > 0.0 {
                // Stage 2: Positive learning is good, can focus on separation
                // Gradually increase negative weight as gap improves
                let progress = ((epoch - positive_focus_epochs) as f32
                    / (self.config.epochs - positive_focus_epochs) as f32)
                    .min(1.0);
                self.config.negative_weight * (0.5 + progress * 0.5).min(1.0 + (current_gap / 0.1))
            } else if avg_pos < 0.01 {
                // Positive scores too low, reduce negative weight
                self.config.negative_weight * 0.3
            } else {
                // Default behavior - moderate weight, gradually increase
                let progress = ((epoch - positive_focus_epochs) as f32
                    / (self.config.epochs - positive_focus_epochs) as f32)
                    .min(1.0);
                self.config.negative_weight * (0.4 + progress * 0.4)
            };

            // Temporarily override negative weight for this epoch
            let original_negative_weight = self.config.negative_weight;
            self.config.negative_weight = adaptive_negative_weight;
            // Shuffle examples for better training (simple Fisher-Yates)
            let mut shuffled_indices: Vec<usize> = (0..examples.len()).collect();
            for i in (1..shuffled_indices.len()).rev() {
                let j = (simple_random() * (i + 1) as f32) as usize;
                shuffled_indices.swap(i, j);
            }

            let mut epoch_loss = 0.0;
            let mut num_batches = 0;

            // Mini-batch training
            for batch_start in (0..examples.len()).step_by(self.config.batch_size) {
                let batch_end = (batch_start + self.config.batch_size).min(examples.len());
                let batch_indices = &shuffled_indices[batch_start..batch_end];

                let mut batch_loss = 0.0;
                let mut batch_pairs = 0;

                // Process batch
                for &idx in batch_indices {
                    let example = &examples[idx];
                    let loss = self.train_example(example, epoch);
                    batch_loss += loss;
                    batch_pairs += 1;
                }

                if batch_pairs > 0 {
                    epoch_loss += batch_loss / batch_pairs as f32;
                    num_batches += 1;
                }
            }

            let avg_loss = if num_batches > 0 {
                epoch_loss / num_batches as f32
            } else {
                0.0
            };
            losses.push(avg_loss);

            let current_lr = get_learning_rate(
                epoch,
                self.config.epochs,
                self.config.learning_rate,
                self.config.warmup_epochs,
            );

            // Early stopping check
            let improved = avg_loss < best_loss - self.config.early_stopping_min_delta;
            if improved {
                best_loss = avg_loss;
                patience_counter = 0;
            } else {
                patience_counter += 1;
            }

            // Show overlap stats periodically
            if epoch % 10 == 0 || epoch == self.config.epochs - 1 || improved {
                let (avg_pos, avg_neg, overlap_rate) = self.get_overlap_stats(examples);
                let status = if improved { "✓" } else { " " };
                let patience_info = if let Some(patience) = self.config.early_stopping_patience {
                    format!(", patience={}/{}", patience_counter, patience)
                } else {
                    String::new()
                };
                let loss_reduction = if losses.len() > 1 {
                    format!(" ({:.1}%↓)", (1.0 - avg_loss / losses[0]) * 100.0)
                } else {
                    String::new()
                };
                let score_gap = avg_pos - avg_neg; // Positive should be higher than negative
                let positive_focus_epochs = self
                    .config
                    .positive_focus_epochs
                    .unwrap_or(self.config.epochs / 3);
                let stage = if epoch < positive_focus_epochs {
                    "P+"
                } else {
                    "S-"
                };
                println!("Epoch {}: loss = {:.4}{}, lr = {:.6}, best = {:.4} {} ({} batches{}, neg_w={:.2}, stage={})", 
                    epoch, avg_loss, loss_reduction, current_lr, best_loss, status, num_batches, patience_info, adaptive_negative_weight, stage);
                println!(
                    "  Overlap: {:.1}%, Pos: {:.4}, Neg: {:.4}, Gap: {:.4} {}",
                    overlap_rate * 100.0,
                    avg_pos,
                    avg_neg,
                    score_gap,
                    if score_gap > 0.0 { "✓" } else { "⚠" }
                );
            }

            // Restore original negative weight
            self.config.negative_weight = original_negative_weight;

            // Early stopping
            if let Some(patience) = self.config.early_stopping_patience {
                if patience_counter >= patience {
                    println!(
                        "Early stopping at epoch {} (no improvement for {} epochs)",
                        epoch, patience
                    );
                    break;
                }
            }
        }

        losses
    }

    /// Get trained boxes for inference.
    pub fn get_boxes(&self) -> HashMap<usize, BoxEmbedding> {
        self.boxes
            .iter()
            .map(|(id, trainable)| (*id, trainable.to_box()))
            .collect()
    }

    /// Get diagnostic statistics about box overlaps.
    ///
    /// Returns (avg_positive_score, avg_negative_score, overlap_rate)
    pub fn get_overlap_stats(&self, examples: &[TrainingExample]) -> (f32, f32, f32) {
        let mut positive_scores = Vec::new();
        let mut negative_scores = Vec::new();
        let mut overlapping_pairs = 0;
        let mut total_pairs = 0;

        for example in examples {
            // Positive pairs
            for chain in &example.chains {
                let mentions: Vec<usize> = chain.mentions.iter().map(|m| m.start).collect();
                for i in 0..mentions.len() {
                    for j in (i + 1)..mentions.len() {
                        if let (Some(box_a), Some(box_b)) =
                            (self.boxes.get(&mentions[i]), self.boxes.get(&mentions[j]))
                        {
                            let box_a_embed = box_a.to_box();
                            let box_b_embed = box_b.to_box();
                            let score = box_a_embed.coreference_score(&box_b_embed);
                            positive_scores.push(score);
                            if score > 0.01 {
                                overlapping_pairs += 1;
                            }
                            total_pairs += 1;
                        }
                    }
                }
            }

            // Negative pairs
            for i in 0..example.chains.len() {
                for j in (i + 1)..example.chains.len() {
                    let chain_i: Vec<usize> =
                        example.chains[i].mentions.iter().map(|m| m.start).collect();
                    let chain_j: Vec<usize> =
                        example.chains[j].mentions.iter().map(|m| m.start).collect();
                    for &id_i in &chain_i {
                        for &id_j in &chain_j {
                            if let (Some(box_a), Some(box_b)) =
                                (self.boxes.get(&id_i), self.boxes.get(&id_j))
                            {
                                let box_a_embed = box_a.to_box();
                                let box_b_embed = box_b.to_box();
                                let score = box_a_embed.coreference_score(&box_b_embed);
                                negative_scores.push(score);
                            }
                        }
                    }
                }
            }
        }

        let avg_positive = if !positive_scores.is_empty() {
            positive_scores.iter().sum::<f32>() / positive_scores.len() as f32
        } else {
            0.0
        };

        let avg_negative = if !negative_scores.is_empty() {
            negative_scores.iter().sum::<f32>() / negative_scores.len() as f32
        } else {
            0.0
        };

        let overlap_rate = if total_pairs > 0 {
            overlapping_pairs as f32 / total_pairs as f32
        } else {
            0.0
        };

        (avg_positive, avg_negative, overlap_rate)
    }

    /// Evaluate coreference accuracy on a test set.
    ///
    /// Returns (accuracy, precision, recall, f1) where:
    /// - Accuracy: fraction of pairs correctly classified
    /// - Precision: fraction of predicted positives that are correct
    /// - Recall: fraction of true positives that are predicted
    /// - F1: harmonic mean of precision and recall
    ///
    /// **Note**: This is a simple pair-wise evaluation. For standard coreference metrics
    /// (MUC, B³, CEAF, LEA, BLANC, CoNLL F1), use `evaluate_standard_metrics()` instead.
    pub fn evaluate(&self, examples: &[TrainingExample], threshold: f32) -> (f32, f32, f32, f32) {
        let mut true_positives = 0;
        let mut false_positives = 0;
        let mut false_negatives = 0;
        let mut total_pairs = 0;

        for example in examples {
            // Build positive pairs (should corefer)
            let mut positive_pairs = Vec::new();
            for chain in &example.chains {
                let mentions: Vec<usize> = chain.mentions.iter().map(|m| m.start).collect();
                for i in 0..mentions.len() {
                    for j in (i + 1)..mentions.len() {
                        positive_pairs.push((mentions[i], mentions[j]));
                    }
                }
            }

            // Build negative pairs (shouldn't corefer)
            let mut negative_pairs = Vec::new();
            for i in 0..example.chains.len() {
                for j in (i + 1)..example.chains.len() {
                    let chain_i: Vec<usize> =
                        example.chains[i].mentions.iter().map(|m| m.start).collect();
                    let chain_j: Vec<usize> =
                        example.chains[j].mentions.iter().map(|m| m.start).collect();
                    for &id_i in &chain_i {
                        for &id_j in &chain_j {
                            negative_pairs.push((id_i, id_j));
                        }
                    }
                }
            }

            // Evaluate positive pairs
            for &(id_a, id_b) in &positive_pairs {
                total_pairs += 1;
                if let (Some(box_a), Some(box_b)) = (self.boxes.get(&id_a), self.boxes.get(&id_b)) {
                    let box_a_embed = box_a.to_box();
                    let box_b_embed = box_b.to_box();
                    let score = box_a_embed.coreference_score(&box_b_embed);
                    if score >= threshold {
                        true_positives += 1;
                    } else {
                        false_negatives += 1;
                    }
                } else {
                    // Missing boxes - count as false negative (model can't predict)
                    false_negatives += 1;
                }
            }

            // Evaluate negative pairs
            for &(id_a, id_b) in &negative_pairs {
                total_pairs += 1;
                if let (Some(box_a), Some(box_b)) = (self.boxes.get(&id_a), self.boxes.get(&id_b)) {
                    let box_a_embed = box_a.to_box();
                    let box_b_embed = box_b.to_box();
                    let score = box_a_embed.coreference_score(&box_b_embed);
                    if score >= threshold {
                        false_positives += 1;
                    }
                    // If score < threshold, it's a true negative (correctly predicted as non-coreferent)
                }
                // If boxes are missing, we can't evaluate - don't count as error
            }
        }

        // Compute metrics
        let precision = if true_positives + false_positives > 0 {
            true_positives as f32 / (true_positives + false_positives) as f32
        } else {
            0.0
        };

        let recall = if true_positives + false_negatives > 0 {
            true_positives as f32 / (true_positives + false_negatives) as f32
        } else {
            0.0
        };

        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        let accuracy = if total_pairs > 0 {
            (true_positives + (total_pairs - true_positives - false_positives - false_negatives))
                as f32
                / total_pairs as f32
        } else {
            0.0
        };

        (accuracy, precision, recall, f1)
    }

    /// Save trained boxes to a file (JSON format).
    ///
    /// # Arguments
    ///
    /// * `path` - File path to save to
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// trainer.save_boxes("trained_boxes.json")?;
    /// ```
    pub fn save_boxes(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        use std::fs::File;
        use std::io::Write;

        let serialized = serde_json::to_string_pretty(&self.boxes)?;
        let mut file = File::create(path)?;
        file.write_all(serialized.as_bytes())?;
        Ok(())
    }

    /// Load trained boxes from a file (JSON format).
    ///
    /// # Arguments
    ///
    /// * `path` - File path to load from
    /// * `dim` - Expected embedding dimension
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let boxes = BoxEmbeddingTrainer::load_boxes("trained_boxes.json", 32)?;
    /// ```
    pub fn load_boxes(
        path: &str,
        dim: usize,
    ) -> Result<HashMap<usize, TrainableBox>, Box<dyn std::error::Error>> {
        use std::fs::File;
        use std::io::Read;

        let mut file = File::open(path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        let boxes: HashMap<usize, TrainableBox> = serde_json::from_str(&contents)?;

        // Verify dimensions
        for (id, box_embedding) in &boxes {
            if box_embedding.dim != dim {
                return Err(format!(
                    "Box for entity {} has dimension {}, expected {}",
                    id, box_embedding.dim, dim
                )
                .into());
            }
        }

        Ok(boxes)
    }

    /// Evaluate using standard coreference metrics (MUC, B³, CEAF, LEA, BLANC, CoNLL F1).
    ///
    /// This converts the trained boxes into coreference chains and evaluates using
    /// the standard metrics used in coreference research.
    ///
    /// # Arguments
    ///
    /// * `examples` - Test examples with gold coreference chains
    /// * `threshold` - Coreference threshold for box overlap
    ///
    /// # Returns
    ///
    /// `CorefEvaluation` with all standard metrics
    pub fn evaluate_standard_metrics(
        &self,
        examples: &[TrainingExample],
        threshold: f32,
    ) -> crate::eval::coref_metrics::CorefEvaluation {
        use crate::backends::box_embeddings::BoxCorefConfig;
        use crate::eval::coref_metrics::CorefEvaluation;
        use crate::eval::coref_resolver::BoxCorefResolver;

        let mut all_predicted_chains = Vec::new();
        let mut all_gold_chains = Vec::new();

        for example in examples {
            // Collect gold chains
            all_gold_chains.extend(example.chains.clone());

            // Get entities from example
            let entities = &example.entities;

            // Get boxes for entities (or create default boxes if missing)
            let mut boxes = Vec::new();
            for entity in entities {
                if let Some(trainable_box) = self.boxes.get(&entity.start) {
                    boxes.push(trainable_box.to_box());
                } else {
                    // Missing box - create a small default box
                    let center = vec![0.0; self.dim];
                    boxes.push(crate::backends::box_embeddings::BoxEmbedding::from_vector(
                        &center, 0.1,
                    ));
                }
            }

            // Resolve coreference using boxes
            let mut box_config = BoxCorefConfig::default();
            box_config.coreference_threshold = threshold;
            let resolver = BoxCorefResolver::new(box_config);
            let resolved_entities = resolver.resolve_with_boxes(entities, &boxes);

            // Convert resolved entities to chains
            let predicted_chains = crate::eval::coref::entities_to_chains(&resolved_entities);
            all_predicted_chains.extend(predicted_chains);
        }

        // Compute standard metrics
        CorefEvaluation::compute(&all_predicted_chains, &all_gold_chains)
    }
}

/// Split training examples into train/validation sets.
///
/// # Arguments
///
/// * `examples` - All training examples
/// * `val_ratio` - Fraction of examples to use for validation (0.0-1.0)
///
/// # Returns
///
/// (train_examples, val_examples)
pub fn split_train_val(
    examples: &[TrainingExample],
    val_ratio: f32,
) -> (Vec<TrainingExample>, Vec<TrainingExample>) {
    let val_size = (examples.len() as f32 * val_ratio) as usize;
    let mut shuffled: Vec<TrainingExample> = examples.to_vec();

    // Simple shuffle
    for i in (1..shuffled.len()).rev() {
        let j = (simple_random() * (i + 1) as f32) as usize;
        shuffled.swap(i, j);
    }

    let val_examples = shuffled.split_off(val_size);
    (shuffled, val_examples)
}

// =============================================================================
// Loss and Gradient Computation
// =============================================================================

/// Compute loss for a pair of boxes.
fn compute_pair_loss(
    box_a: &TrainableBox,
    box_b: &TrainableBox,
    is_positive: bool,
    config: &TrainingConfig,
) -> f32 {
    let box_a_embed = box_a.to_box();
    let box_b_embed = box_b.to_box();

    if is_positive {
        // Positive pair: maximize conditional probability
        let p_a_b = box_a_embed.conditional_probability(&box_b_embed);
        let p_b_a = box_b_embed.conditional_probability(&box_a_embed);

        // Clamp probabilities to avoid log(0)
        let p_a_b = p_a_b.max(1e-8);
        let p_b_a = p_b_a.max(1e-8);

        // Use symmetric score: min of both conditional probabilities
        // This ensures both boxes must overlap significantly
        let min_prob = p_a_b.min(p_b_a);
        let neg_log_prob = -min_prob.ln();

        // Also add penalty if boxes are too far apart (encourages movement)
        let vol_intersection = box_a_embed.intersection_volume(&box_b_embed);
        let distance_penalty = if vol_intersection < 1e-10 {
            // Boxes don't overlap - add distance penalty
            let center_a = box_a_embed.center();
            let center_b = box_b_embed.center();
            let dist: f32 = center_a
                .iter()
                .zip(center_b.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt();
            0.3 * dist // Stronger penalty to encourage boxes to move closer
        } else {
            // Even if overlapping, encourage more overlap
            // Penalize if overlap is too small
            let vol_a = box_a_embed.volume();
            let vol_b = box_b_embed.volume();
            let overlap_ratio = vol_intersection / vol_a.min(vol_b).max(1e-10);
            if overlap_ratio < 0.5 {
                // Encourage more overlap
                0.1 * (0.5 - overlap_ratio)
            } else {
                0.0
            }
        };

        // Regularization: penalize large volumes
        let vol_a = box_a_embed.volume();
        let vol_b = box_b_embed.volume();
        // Light regularization to prevent boxes from growing too large
        let reg = config.regularization * 1.0 * (vol_a + vol_b);

        (neg_log_prob + reg + distance_penalty).max(0.0)
    } else {
        // Negative pair: enforce separation
        // We want conditional probability to be LOW (boxes should be disjoint)
        let p_a_b = box_a_embed.conditional_probability(&box_b_embed);
        let p_b_a = box_b_embed.conditional_probability(&box_a_embed);

        // Use max of both conditional probabilities
        let max_prob = p_a_b.max(p_b_a);

        // Loss: penalize high conditional probability
        // Use hinge loss instead of quadratic for smoother gradients
        let margin_loss = if max_prob > config.margin {
            // Stronger quadratic penalty for exceeding margin
            let excess = max_prob - config.margin;
            excess.powi(2) * (1.0 + excess * 2.0) // Exponential scaling
        } else {
            0.0 // No loss if already below margin (good!)
        };

        // Add extra penalty for very high probabilities (exponential decay)
        // Note: This is currently not used in the loss calculation but kept for future use
        let _high_prob_penalty = if max_prob > 0.1 {
            (max_prob - 0.1).powi(2) * 0.5 // Extra penalty for very high probabilities
        } else {
            0.0
        };

        // Add extra penalty if boxes overlap significantly
        let vol_intersection = box_a_embed.intersection_volume(&box_b_embed);
        let vol_a = box_a_embed.volume();
        let vol_b = box_b_embed.volume();
        let overlap_penalty = if vol_intersection > 1e-10 {
            // Boxes overlap - add strong penalty (more aggressive for higher overlap)
            let overlap_ratio = vol_intersection / vol_a.min(vol_b).max(1e-10);
            // Exponential penalty for high overlap
            if overlap_ratio > 0.5 {
                4.0 * overlap_ratio * overlap_ratio // Stronger quadratic penalty
            } else if overlap_ratio > 0.3 {
                3.0 * overlap_ratio // Stronger linear penalty for moderate overlap
            } else {
                2.5 * overlap_ratio // Linear penalty for low overlap
            }
        } else {
            0.0
        };

        // Base loss: only penalize if probability is significant (above a threshold)
        // Don't penalize tiny probabilities - they're fine
        let base_loss = if max_prob > 0.01 {
            max_prob * 0.2 // Stronger penalty for significant probabilities
        } else {
            0.0 // No penalty for tiny probabilities
        };

        // Adaptive penalty: stronger for very high probabilities
        // Use exponential scaling for probabilities > 0.1, moderate for > 0.05
        let adaptive_penalty = if max_prob > 0.1 {
            // Exponential penalty for very high probabilities (stronger scaling)
            let prob_excess = max_prob - 0.1;
            prob_excess.powi(2) * (3.0 + prob_excess * 7.0) // Stronger exponential scaling
        } else if max_prob > 0.05 {
            // Moderate penalty for medium-high probabilities
            (max_prob - 0.05).powi(2) * 1.5 // Stronger
        } else if max_prob > 0.02 {
            // Light penalty for low-medium probabilities
            (max_prob - 0.02).powi(2) * 0.5
        } else {
            0.0
        };

        config.negative_weight * (margin_loss + overlap_penalty + base_loss + adaptive_penalty)
    }
}

/// Compute analytical gradients for a pair of boxes.
fn compute_analytical_gradients(
    box_a: &TrainableBox,
    box_b: &TrainableBox,
    is_positive: bool,
    config: &TrainingConfig,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let box_a_embed = box_a.to_box();
    let box_b_embed = box_b.to_box();
    let dim = box_a.dim;

    // Initialize gradients
    let mut grad_mu_a = vec![0.0; dim];
    let mut grad_delta_a = vec![0.0; dim];
    let mut grad_mu_b = vec![0.0; dim];
    let mut grad_delta_b = vec![0.0; dim];

    // Compute intersection volume and individual volumes
    let vol_a = box_a_embed.volume();
    let vol_b = box_b_embed.volume();
    let vol_intersection = box_a_embed.intersection_volume(&box_b_embed);

    if is_positive {
        // Positive pair: L = -log(P(A|B)) - log(P(B|A)) + reg
        // P(A|B) = Vol(A ∩ B) / Vol(B)
        // P(B|A) = Vol(A ∩ B) / Vol(A)

        let p_a_b = if vol_b > 0.0 {
            vol_intersection / vol_b
        } else {
            0.0
        };
        let p_b_a = if vol_a > 0.0 {
            vol_intersection / vol_a
        } else {
            0.0
        };

        // Clamp to avoid division by zero
        let p_a_b = p_a_b.max(1e-8);
        let p_b_a = p_b_a.max(1e-8);

        // Gradients through -log(P(A|B)) = -log(Vol_intersection) + log(Vol_B)
        // For positive pairs, we want to maximize overlap
        // If boxes don't overlap, we need gradients to move them together

        // Check if boxes overlap
        let vol_intersection = box_a_embed.intersection_volume(&box_b_embed);
        let has_overlap = vol_intersection > 1e-10;

        if !has_overlap {
            // Boxes don't overlap - add very strong gradient to move centers closer
            let center_a = box_a_embed.center();
            let center_b = box_b_embed.center();
            let center_dist = center_a
                .iter()
                .zip(center_b.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt();

            for i in 0..dim {
                let diff = center_b[i] - center_a[i];
                // Adaptive strength based on distance - stronger when far apart
                let distance_factor = (center_dist / dim as f32).clamp(0.5, 2.0);
                let attraction_strength = 4.0 * distance_factor; // Stronger when far apart

                grad_mu_a[i] += attraction_strength * diff;
                grad_mu_b[i] += -attraction_strength * diff;

                // Strong increase in box sizes to help them overlap
                grad_delta_a[i] += 0.5 * distance_factor; // Stronger growth when far apart
                grad_delta_b[i] += 0.5 * distance_factor;
            }
        }

        for i in 0..dim {
            // Gradient w.r.t. box A
            // ∂(-log(P(A|B)))/∂δ_A = -1/P(A|B) * ∂P(A|B)/∂δ_A
            // ∂P(A|B)/∂δ_A = (1/Vol_B) * ∂Vol_intersection/∂δ_A
            // ∂Vol_intersection/∂δ_A = Vol_intersection (if boxes overlap in dim i)

            let overlap_i = if box_a_embed.min[i] < box_b_embed.max[i]
                && box_b_embed.min[i] < box_a_embed.max[i]
            {
                // Boxes overlap in dimension i
                let min_overlap = box_a_embed.min[i].max(box_b_embed.min[i]);
                let max_overlap = box_a_embed.max[i].min(box_b_embed.max[i]);
                (max_overlap - min_overlap).max(0.0)
            } else {
                0.0
            };

            if overlap_i > 0.0 && vol_intersection > 0.0 {
                // Gradient through intersection volume
                // When boxes overlap, focus on improving overlap ratio (not just growing boxes)
                let overlap_ratio_a = vol_intersection / vol_a.max(1e-10);
                let overlap_ratio_b = vol_intersection / vol_b.max(1e-10);

                // If overlap ratio is low, encourage growth; if high, maintain
                // Adaptive growth based on current overlap - more aggressive
                if overlap_ratio_a < 0.15 {
                    // Extremely low overlap - encourage very strong growth
                    grad_delta_a[i] += 0.35;
                } else if overlap_ratio_a < 0.3 {
                    // Very low overlap - encourage extremely strong growth
                    grad_delta_a[i] += 0.3;
                } else if overlap_ratio_a < 0.5 {
                    // Low overlap - encourage very strong growth
                    grad_delta_a[i] += 0.2;
                } else if overlap_ratio_a < 0.7 {
                    // Moderate overlap - strong growth
                    grad_delta_a[i] += 0.1;
                } else if overlap_ratio_a < 0.85 {
                    // Good overlap - small growth
                    grad_delta_a[i] += 0.05;
                }
                // If overlap_ratio_a >= 0.85, don't grow (excellent)

                if overlap_ratio_b < 0.15 {
                    // Extremely low overlap - encourage very strong growth
                    grad_delta_b[i] += 0.35;
                } else if overlap_ratio_b < 0.3 {
                    // Very low overlap - encourage extremely strong growth
                    grad_delta_b[i] += 0.3;
                } else if overlap_ratio_b < 0.5 {
                    // Low overlap - encourage very strong growth
                    grad_delta_b[i] += 0.2;
                } else if overlap_ratio_b < 0.7 {
                    // Moderate overlap - strong growth
                    grad_delta_b[i] += 0.1;
                } else if overlap_ratio_b < 0.85 {
                    // Good overlap - small growth
                    grad_delta_b[i] += 0.05;
                }

                // Gradient through conditional probability (main signal) - adaptive strength
                // Stronger when overlap is low, gentler when overlap is good
                let gradient_strength = if overlap_ratio_a < 0.1 {
                    1.7 // Extremely strong when overlap is extremely low
                } else if overlap_ratio_a < 0.2 {
                    1.6 // Extremely strong when overlap is very low
                } else if overlap_ratio_a < 0.4 {
                    1.4 // Very strong when overlap is low
                } else if overlap_ratio_a < 0.6 {
                    1.1 // Strong when overlap is moderate
                } else {
                    0.6 // Gentle when overlap is good
                };

                let grad_vol_intersection_delta_a = vol_intersection * 0.5 * gradient_strength;
                let grad_p_a_b_delta_a = grad_vol_intersection_delta_a / vol_b.max(1e-8);
                grad_delta_a[i] += -grad_p_a_b_delta_a / p_a_b.max(1e-8) * gradient_strength;

                let grad_vol_intersection_delta_b = vol_intersection * 0.5 * gradient_strength;
                let grad_p_b_a_delta_b = grad_vol_intersection_delta_b / vol_a.max(1e-8);
                grad_delta_b[i] += -grad_p_b_a_delta_b / p_b_a.max(1e-8) * gradient_strength;
            } else {
                // Boxes don't overlap in this dimension - extremely strong growth to achieve overlap
                grad_delta_a[i] += 0.3; // Extremely strong growth for box A
                grad_delta_b[i] += 0.3; // Extremely strong growth for box B
            }

            // Regularization gradient: ∂(λ * Vol)/∂δ = λ * Vol
            // Light regularization to prevent boxes from growing too large
            // Apply to both positive and negative pairs (but lighter for positives)
            grad_delta_a[i] += config.regularization * 1.0 * vol_a; // Lighter regularization
            grad_delta_b[i] += config.regularization * 1.0 * vol_b;
        }
    } else {
        // Negative pair: L = max(0, margin - max(P(A|B), P(B|A))) * λ_neg + overlap_penalty
        let p_a_b = if vol_b > 0.0 {
            vol_intersection / vol_b
        } else {
            0.0
        };
        let p_b_a = if vol_a > 0.0 {
            vol_intersection / vol_a
        } else {
            0.0
        };
        let max_prob = p_a_b.max(p_b_a);

        // Always apply gradients for negative pairs (they should always be separated)
        // Don't check margin - always try to minimize conditional probability
        for i in 0..dim {
            // Check if boxes overlap in this dimension
            let overlap_i = if box_a_embed.min[i] < box_b_embed.max[i]
                && box_b_embed.min[i] < box_a_embed.max[i]
            {
                let min_overlap = box_a_embed.min[i].max(box_b_embed.min[i]);
                let max_overlap = box_a_embed.max[i].min(box_b_embed.max[i]);
                (max_overlap - min_overlap).max(0.0)
            } else {
                0.0
            };

            if overlap_i > 0.0 {
                // Boxes overlap - strong gradient to separate
                // Move centers apart
                let center_a = box_a_embed.center();
                let center_b = box_b_embed.center();
                let diff = center_b[i] - center_a[i];

                // Gradient to push boxes apart (adaptive strength based on overlap)
                // Stronger separation when overlap is high
                let overlap_factor =
                    (overlap_i / (box_a_embed.max[i] - box_a_embed.min[i]).max(1e-6)).min(1.0);
                let separation_strength = 1.5 + overlap_factor * 2.0; // 1.5 to 3.5 based on overlap (even stronger)
                if diff.abs() > 1e-6 {
                    grad_mu_a[i] += -config.negative_weight * separation_strength * diff;
                    grad_mu_b[i] += config.negative_weight * separation_strength * diff;
                } else {
                    // Boxes are on top of each other - push in random direction (very strong)
                    grad_mu_a[i] += -config.negative_weight * separation_strength * 2.5;
                    grad_mu_b[i] += config.negative_weight * separation_strength * 2.5;
                }

                // Shrink boxes to reduce overlap (adaptive strength based on overlap amount)
                // More aggressive shrinking for higher overlap
                let overlap_ratio_dim =
                    overlap_i / (box_a_embed.max[i] - box_a_embed.min[i]).max(1e-6);
                let shrink_strength = if overlap_ratio_dim > 0.7 {
                    0.7 // Extremely strong shrink for very high overlap
                } else if overlap_ratio_dim > 0.5 {
                    0.6 // Very strong shrink for high overlap
                } else if overlap_ratio_dim > 0.3 {
                    0.5 // Strong shrink for moderate overlap
                } else {
                    0.35 // Moderate shrink for low overlap
                };
                grad_delta_a[i] += -config.negative_weight * shrink_strength;
                grad_delta_b[i] += -config.negative_weight * shrink_strength;
            } else {
                // Boxes don't overlap - NO gradient (let positive pairs grow)
                // Don't shrink when already separated
            }

            // Gradient through overlap_penalty: ∂(2.0 * overlap_ratio)/∂δ
            // Overlap ratio = vol_intersection / min(vol_a, vol_b)
            // ONLY penalize if actually overlapping - stronger penalty for higher overlap
            if overlap_i > 0.0 && vol_intersection > 1e-10 {
                let min_vol = vol_a.min(vol_b);
                let overlap_ratio = vol_intersection / min_vol.max(1e-10);
                // Stronger penalty for higher overlap ratios (more aggressive)
                // Match the loss function: 2.5x/3.0x/4.0x based on overlap
                let penalty_strength = if overlap_ratio > 0.5 {
                    0.4 + overlap_ratio * 0.6 // 0.7 to 1.0 for very high overlap
                } else if overlap_ratio > 0.3 {
                    0.3 + overlap_ratio * 0.5 // 0.39 to 0.45 for moderate-high overlap
                } else {
                    0.2 + overlap_ratio * 0.4 // 0.2 to 0.32 for low overlap
                };
                let penalty_multiplier = if overlap_ratio > 0.5 {
                    4.0
                } else if overlap_ratio > 0.3 {
                    3.0
                } else {
                    2.5
                };
                grad_delta_a[i] +=
                    config.negative_weight * penalty_multiplier * overlap_ratio * penalty_strength;
                grad_delta_b[i] +=
                    config.negative_weight * penalty_multiplier * overlap_ratio * penalty_strength;
            }

            // Gradient through base_loss and margin_loss: minimize max_prob
            // ∂(0.2 * max_prob)/∂δ = 0.2 * ∂max_prob/∂δ
            // ∂((max_prob - margin)^2)/∂δ = 2 * (max_prob - margin) * ∂max_prob/∂δ
            // Always apply these gradients (not just when overlapping) to keep scores low
            if p_a_b >= p_b_a {
                // p_a_b is the max, minimize it
                if overlap_i > 0.0 && vol_intersection > 1e-10 {
                    let grad_vol_intersection_delta_a = vol_intersection * 0.4;
                    let grad_p_a_b_delta_a = grad_vol_intersection_delta_a / vol_b.max(1e-8);
                    // Positive gradient on delta = shrink box (reduce intersection)
                    grad_delta_a[i] += config.negative_weight * 0.2 * grad_p_a_b_delta_a;

                    // Also add margin loss gradient if active
                    if max_prob > config.margin {
                        let excess = max_prob - config.margin;
                        let margin_grad = 2.0 * excess * (1.0 + excess * 2.0) * grad_p_a_b_delta_a
                            + 2.0 * excess.powi(2) * 2.0 * grad_p_a_b_delta_a; // Exponential scaling
                        grad_delta_a[i] += config.negative_weight * margin_grad;
                    }

                    // Adaptive penalty gradient for very high probabilities (stronger for higher probs)
                    if max_prob > 0.1 {
                        // Exponential scaling: stronger penalty for very high probabilities
                        let prob_excess = max_prob - 0.1;
                        let adaptive_grad =
                            2.0 * prob_excess * grad_p_a_b_delta_a * (3.0 + prob_excess * 7.0); // Stronger
                        grad_delta_a[i] += config.negative_weight * adaptive_grad;
                    } else if max_prob > 0.05 {
                        // Moderate penalty gradient for medium-high probabilities
                        let prob_excess = max_prob - 0.05;
                        let adaptive_grad = 2.0 * prob_excess * grad_p_a_b_delta_a * 1.5; // Stronger
                        grad_delta_a[i] += config.negative_weight * adaptive_grad;
                    } else if max_prob > 0.02 {
                        // Light penalty gradient for low-medium probabilities
                        let prob_excess = max_prob - 0.02;
                        let adaptive_grad = 2.0 * prob_excess * grad_p_a_b_delta_a * 0.5;
                        grad_delta_a[i] += config.negative_weight * adaptive_grad;
                    }
                }
                // Don't add extra shrink when not overlapping - let positive pairs grow
            } else {
                // p_b_a is the max, minimize it
                if overlap_i > 0.0 && vol_intersection > 1e-10 {
                    let grad_vol_intersection_delta_b = vol_intersection * 0.4;
                    let grad_p_b_a_delta_b = grad_vol_intersection_delta_b / vol_a.max(1e-8);
                    // Positive gradient on delta = shrink box
                    grad_delta_b[i] += config.negative_weight * 0.25 * grad_p_b_a_delta_b; // Slightly stronger

                    // Also add margin loss gradient if active (stronger)
                    if max_prob > config.margin {
                        let excess = max_prob - config.margin;
                        let margin_grad = 2.0 * excess * (1.0 + excess * 2.0) * grad_p_b_a_delta_b
                            + 2.0 * excess.powi(2) * 2.0 * grad_p_b_a_delta_b; // Exponential scaling
                        grad_delta_b[i] += config.negative_weight * margin_grad;
                    }

                    // Adaptive penalty gradient for very high probabilities (stronger for higher probs)
                    if max_prob > 0.1 {
                        // Exponential scaling: stronger penalty for very high probabilities
                        let prob_excess = max_prob - 0.1;
                        let adaptive_grad =
                            2.0 * prob_excess * grad_p_b_a_delta_b * (2.0 + prob_excess * 5.0);
                        grad_delta_b[i] += config.negative_weight * adaptive_grad;
                    } else if max_prob > 0.05 {
                        // Moderate penalty gradient for medium-high probabilities
                        let prob_excess = max_prob - 0.05;
                        let adaptive_grad = 2.0 * prob_excess * grad_p_b_a_delta_b * 1.0;
                        grad_delta_b[i] += config.negative_weight * adaptive_grad;
                    }
                }
                // Don't add extra shrink when not overlapping - let positive pairs grow
            }
        }
    }

    // Clip gradients to prevent explosion
    for grad in &mut grad_mu_a {
        *grad = grad.clamp(-10.0_f32, 10.0_f32);
    }
    for grad in &mut grad_delta_a {
        *grad = grad.clamp(-10.0_f32, 10.0_f32);
    }
    for grad in &mut grad_mu_b {
        *grad = grad.clamp(-10.0_f32, 10.0_f32);
    }
    for grad in &mut grad_delta_b {
        *grad = grad.clamp(-10.0_f32, 10.0_f32);
    }

    (grad_mu_a, grad_delta_a, grad_mu_b, grad_delta_b)
}

/// Sample negative pairs using self-adversarial sampling.
fn sample_self_adversarial_negatives(
    negative_pairs: &[(usize, usize)],
    boxes: &HashMap<usize, TrainableBox>,
    num_samples: usize,
    temperature: f32,
) -> Vec<usize> {
    // Compute scores for all negative pairs
    let mut scores: Vec<(usize, f32)> = negative_pairs
        .iter()
        .enumerate()
        .filter_map(|(idx, &(id_a, id_b))| {
            if let (Some(box_a), Some(box_b)) = (boxes.get(&id_a), boxes.get(&id_b)) {
                let box_a_embed = box_a.to_box();
                let box_b_embed = box_b.to_box();
                let score = box_a_embed.coreference_score(&box_b_embed);
                Some((idx, score / temperature))
            } else {
                None
            }
        })
        .collect();

    // Sort by score (descending) - higher scores are "harder" negatives
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Sample top-k (hardest negatives)
    scores
        .into_iter()
        .take(num_samples)
        .map(|(idx, _)| idx)
        .collect()
}

/// Get learning rate with warmup and cosine decay.
fn get_learning_rate(epoch: usize, total_epochs: usize, base_lr: f32, warmup_epochs: usize) -> f32 {
    if epoch < warmup_epochs {
        // Linear warmup: 0.1 * lr → lr
        let warmup_lr = base_lr * 0.1;
        warmup_lr + (base_lr - warmup_lr) * (epoch as f32 / warmup_epochs as f32)
    } else {
        // Cosine decay: lr → 0.1 * lr
        let progress =
            (epoch - warmup_epochs) as f32 / (total_epochs - warmup_epochs).max(1) as f32;
        let min_lr = base_lr * 0.1;
        min_lr + (base_lr - min_lr) * (1.0 + (std::f32::consts::PI * progress).cos()) / 2.0
    }
}

// =============================================================================
// TrainableBox AMSGrad Update
// =============================================================================

impl TrainableBox {
    /// Update box parameters using AMSGrad optimizer.
    pub fn update_amsgrad(
        &mut self,
        grad_mu: &[f32],
        grad_delta: &[f32],
        state: &mut AMSGradState,
    ) {
        state.t += 1;
        let t = state.t as f32;

        // Update first moment (m)
        for i in 0..self.dim {
            state.m[i] = state.beta1 * state.m[i] + (1.0 - state.beta1) * grad_mu[i];
        }

        // Update second moment (v) and max (v_hat)
        for i in 0..self.dim {
            let v_new = state.beta2 * state.v[i] + (1.0 - state.beta2) * grad_mu[i] * grad_mu[i];
            state.v[i] = v_new;
            state.v_hat[i] = state.v_hat[i].max(v_new);
        }

        // Bias correction for first moment
        let m_hat: Vec<f32> = state
            .m
            .iter()
            .map(|&m| m / (1.0 - state.beta1.powf(t)))
            .collect();

        // Update mu
        for i in 0..self.dim {
            let update = state.lr * m_hat[i] / (state.v_hat[i].sqrt() + state.epsilon);
            self.mu[i] -= update;

            // Ensure finite
            if !self.mu[i].is_finite() {
                self.mu[i] = 0.0;
            }
        }

        // Similar for delta
        let mut m_delta = vec![0.0_f32; self.dim];
        let mut v_delta = vec![0.0_f32; self.dim];
        let mut v_hat_delta = vec![0.0_f32; self.dim];

        for i in 0..self.dim {
            m_delta[i] = state.beta1 * m_delta[i] + (1.0 - state.beta1) * grad_delta[i];
            let v_new: f32 =
                state.beta2 * v_delta[i] + (1.0 - state.beta2) * grad_delta[i] * grad_delta[i];
            v_delta[i] = v_new;
            v_hat_delta[i] = v_hat_delta[i].max(v_new);
        }

        let m_hat_delta: Vec<f32> = m_delta
            .iter()
            .map(|&m| m / (1.0 - state.beta1.powf(t)))
            .collect();

        for i in 0..self.dim {
            let update = state.lr * m_hat_delta[i] / (v_hat_delta[i].sqrt() + state.epsilon);
            self.delta[i] -= update;

            // Clamp delta to reasonable range (width between 0.01 and 10.0)
            self.delta[i] = self.delta[i].clamp(0.01_f32.ln(), 10.0_f32.ln());

            // Ensure finite
            if !self.delta[i].is_finite() {
                self.delta[i] = 0.5_f32.ln();
            }
        }
    }
}

// =============================================================================
// Simple Random Number Generator
// =============================================================================

/// Simple random number generator (for when rand feature is not available).
///
/// Thread-safe implementation using atomic counter to avoid unsafe static mut.
fn simple_random() -> f32 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static COUNTER: AtomicUsize = AtomicUsize::new(0);

    // Thread-safe counter increment
    let count = COUNTER.fetch_add(1, Ordering::Relaxed);

    let mut hasher = DefaultHasher::new();
    // Use duration since epoch, or fallback to count if time is unavailable
    let time_nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(count as u128);
    time_nanos.hash(&mut hasher);
    count.hash(&mut hasher);
    let hash = hasher.finish();
    (hash as f32) / (u64::MAX as f32)
}
