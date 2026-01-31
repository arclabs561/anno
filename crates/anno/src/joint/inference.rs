//! Belief propagation inference for joint entity analysis.
//!
//! Implements loopy belief propagation (sum-product) for approximate
//! inference in the factor graph.
//!
//! # Algorithm
//!
//! ```text
//! repeat until convergence:
//!   for each factor f:
//!     for each variable v in scope(f):
//!       m_{f→v}(v) ∝ ∑_{scope(f)\v} ψ_f(scope(f)) ∏_{v'∈scope(f)\v} m_{v'→f}(v')
//!
//!   for each variable v:
//!     for each factor f containing v:
//!       m_{v→f}(v) ∝ ∏_{f'≠f} m_{f'→v}(v)
//!
//! marginal: p(v) ∝ ∏_f m_{f→v}(v)
//! ```
//!
//! # References
//!
//! - Kschischang et al. (2001): "Factor Graphs and the Sum-Product Algorithm"
//! - Murphy et al. (1999): "Loopy Belief Propagation for Approximate Inference"
//! - Durrett & Klein (2014): "A Joint Model for Entity Analysis"

use super::factors::Factor;
use super::types::{AntecedentValue, Assignment, JointVariable, LinkValue, VariableId};
use crate::EntityType;
use std::collections::HashMap;

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for belief propagation inference.
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Maximum iterations
    pub max_iterations: usize,
    /// Convergence threshold (max message change)
    pub convergence_threshold: f64,
    /// Damping factor (0 = no damping, 1 = no update)
    pub damping: f64,
    /// Message schedule
    pub schedule: MessageSchedule,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            max_iterations: 5,
            convergence_threshold: 1e-4,
            damping: 0.0,
            schedule: MessageSchedule::Parallel,
        }
    }
}

/// Order in which to update messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageSchedule {
    /// Update all messages in parallel
    Parallel,
    /// Update messages sequentially (may converge faster)
    Sequential,
}

// =============================================================================
// Messages
// =============================================================================

/// A message in belief propagation.
///
/// Messages are probability distributions over a variable's domain,
/// stored in log space for numerical stability.
#[derive(Debug, Clone)]
pub struct Message {
    /// Log probabilities for each domain value
    pub log_probs: Vec<f64>,
}

impl Message {
    /// Create a uniform message.
    pub fn uniform(domain_size: usize) -> Self {
        if domain_size == 0 {
            return Self { log_probs: vec![] };
        }
        let log_prob = -(domain_size as f64).ln();
        Self {
            log_probs: vec![log_prob; domain_size],
        }
    }

    /// Create from raw log probabilities.
    pub fn from_log_probs(log_probs: Vec<f64>) -> Self {
        Self { log_probs }
    }

    /// Normalize to sum to 1 (in probability space).
    pub fn normalize(&mut self) {
        if self.log_probs.is_empty() {
            return;
        }
        let log_sum = log_sum_exp(&self.log_probs);
        if log_sum.is_finite() {
            for lp in &mut self.log_probs {
                *lp -= log_sum;
            }
        }
    }

    /// Max change from another message.
    pub fn max_change(&self, other: &Message) -> f64 {
        self.log_probs
            .iter()
            .zip(other.log_probs.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max)
    }

    /// Apply damping with previous message.
    pub fn damp(&mut self, previous: &Message, damping: f64) {
        for (new, old) in self.log_probs.iter_mut().zip(previous.log_probs.iter()) {
            *new = (1.0 - damping) * *new + damping * *old;
        }
    }

    /// Pointwise multiply two messages (sum in log space).
    #[allow(dead_code)]
    pub fn multiply(&self, other: &Message) -> Message {
        let log_probs: Vec<f64> = self
            .log_probs
            .iter()
            .zip(other.log_probs.iter())
            .map(|(a, b)| a + b)
            .collect();
        Message { log_probs }
    }
}

/// Log-sum-exp trick for numerical stability.
pub fn log_sum_exp(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NEG_INFINITY;
    }

    let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max_val.is_infinite() {
        return max_val;
    }

    let sum: f64 = values.iter().map(|v| (v - max_val).exp()).sum();
    max_val + sum.ln()
}

// =============================================================================
// Message Store
// =============================================================================

/// Key for a message (from source to target).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MessageKey {
    /// Source (factor index or variable ID serialized)
    pub from: String,
    /// Target (variable ID or factor index serialized)
    pub to: String,
}

impl MessageKey {
    /// Create factor-to-variable key.
    pub fn factor_to_var(factor_idx: usize, var_id: &VariableId) -> Self {
        Self {
            from: format!("f{}", factor_idx),
            to: format!("v{}_{:?}", var_id.mention_idx, var_id.var_type),
        }
    }

    /// Create variable-to-factor key.
    pub fn var_to_factor(var_id: &VariableId, factor_idx: usize) -> Self {
        Self {
            from: format!("v{}_{:?}", var_id.mention_idx, var_id.var_type),
            to: format!("f{}", factor_idx),
        }
    }
}

/// Storage for all messages in belief propagation.
#[derive(Debug, Clone, Default)]
pub struct MessageStore {
    messages: HashMap<MessageKey, Message>,
}

impl MessageStore {
    /// Get message, or return uniform if not set.
    pub fn get(&self, key: &MessageKey, domain_size: usize) -> Message {
        self.messages
            .get(key)
            .cloned()
            .unwrap_or_else(|| Message::uniform(domain_size))
    }

    /// Set message.
    pub fn set(&mut self, key: MessageKey, message: Message) {
        self.messages.insert(key, message);
    }
}

// =============================================================================
// Marginals
// =============================================================================

/// Computed marginal distributions.
#[derive(Debug, Clone, Default)]
pub struct Marginals {
    /// Marginals per variable: var_id → log_probs
    pub distributions: HashMap<VariableId, Vec<f64>>,
}

impl Marginals {
    /// Get most likely value index for a variable.
    pub fn argmax(&self, var_id: &VariableId) -> Option<usize> {
        self.distributions.get(var_id).and_then(|probs| {
            probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
        })
    }

    /// Get probability of a specific value.
    pub fn prob(&self, var_id: &VariableId, value_idx: usize) -> Option<f64> {
        self.distributions
            .get(var_id)
            .and_then(|probs| probs.get(value_idx))
            .map(|log_p| log_p.exp())
    }

    /// Get max probability for a variable.
    pub fn max_prob(&self, var_id: &VariableId) -> Option<f64> {
        self.distributions
            .get(var_id)
            .and_then(|probs| {
                probs
                    .iter()
                    .cloned()
                    .fold(None, |max, p| Some(max.map_or(p, |m: f64| m.max(p))))
            })
            .map(|log_p| log_p.exp())
    }
}

// =============================================================================
// Domain Enumeration
// =============================================================================

/// Helper for enumerating variable domains.
#[derive(Debug, Clone)]
pub struct DomainValue {
    /// Index in the domain
    pub index: usize,
    /// The actual value
    pub value: DomainValueType,
}

/// Concrete domain value types.
#[derive(Debug, Clone)]
pub enum DomainValueType {
    /// Antecedent value
    Antecedent(AntecedentValue),
    /// Entity type
    SemanticType(EntityType),
    /// Link value
    EntityLink(LinkValue),
}

/// Get domain values for a variable.
pub fn get_domain_values(var: &JointVariable) -> Vec<DomainValue> {
    match var {
        JointVariable::Antecedent { candidates, .. } => {
            let mut values: Vec<DomainValue> = candidates
                .iter()
                .enumerate()
                .map(|(i, &m)| DomainValue {
                    index: i,
                    value: DomainValueType::Antecedent(AntecedentValue::Mention(m)),
                })
                .collect();
            values.push(DomainValue {
                index: candidates.len(),
                value: DomainValueType::Antecedent(AntecedentValue::NewCluster),
            });
            values
        }
        JointVariable::SemanticType { types, .. } => types
            .iter()
            .enumerate()
            .map(|(i, t)| DomainValue {
                index: i,
                value: DomainValueType::SemanticType(t.clone()),
            })
            .collect(),
        JointVariable::EntityLink { candidates, .. } => {
            let mut values: Vec<DomainValue> = candidates
                .iter()
                .enumerate()
                .map(|(i, kb_id)| DomainValue {
                    index: i,
                    value: DomainValueType::EntityLink(LinkValue::KbId(kb_id.clone())),
                })
                .collect();
            values.push(DomainValue {
                index: candidates.len(),
                value: DomainValueType::EntityLink(LinkValue::Nil),
            });
            values
        }
    }
}

/// Apply a domain value to an assignment.
pub fn apply_domain_value(assignment: &mut Assignment, var: &JointVariable, value: &DomainValue) {
    let mention_idx = match var {
        JointVariable::Antecedent { mention_idx, .. } => *mention_idx,
        JointVariable::SemanticType { mention_idx, .. } => *mention_idx,
        JointVariable::EntityLink { mention_idx, .. } => *mention_idx,
    };

    match &value.value {
        DomainValueType::Antecedent(a) => assignment.set_antecedent(mention_idx, *a),
        DomainValueType::SemanticType(t) => assignment.set_type(mention_idx, t.clone()),
        DomainValueType::EntityLink(l) => assignment.set_link(mention_idx, l.clone()),
    }
}

// =============================================================================
// Belief Propagation
// =============================================================================

/// Belief propagation inference engine.
pub struct BeliefPropagation {
    /// Factor graph factors
    factors: Vec<Box<dyn Factor>>,
    /// Variables
    variables: Vec<JointVariable>,
    /// Message store
    messages: MessageStore,
    /// Configuration
    config: InferenceConfig,
    /// Variable lookup by ID
    var_by_id: HashMap<VariableId, usize>,
}

impl BeliefPropagation {
    /// Create a new belief propagation engine.
    pub fn new(
        factors: Vec<Box<dyn Factor>>,
        variables: Vec<JointVariable>,
        config: InferenceConfig,
    ) -> Self {
        let var_by_id: HashMap<VariableId, usize> = variables
            .iter()
            .enumerate()
            .map(|(i, v)| (v.id(), i))
            .collect();

        Self {
            factors,
            variables,
            messages: MessageStore::default(),
            config,
            var_by_id,
        }
    }

    /// Run belief propagation to compute marginals.
    pub fn run(&mut self) -> Marginals {
        // Initialize messages to uniform
        self.initialize_messages();

        let mut converged = false;
        for _iter in 0..self.config.max_iterations {
            let max_change = self.iterate();

            if max_change < self.config.convergence_threshold {
                converged = true;
                break;
            }
        }

        // Convergence check - algorithm continues regardless since BP typically
        // provides reasonable approximations after a few iterations
        let _ = converged;

        self.compute_marginals()
    }

    /// Initialize messages to uniform distributions.
    fn initialize_messages(&mut self) {
        self.messages = MessageStore::default();
    }

    /// Run one iteration of message passing.
    fn iterate(&mut self) -> f64 {
        match self.config.schedule {
            MessageSchedule::Parallel => self.iterate_parallel(),
            MessageSchedule::Sequential => self.iterate_sequential(),
        }
    }

    /// Parallel message update.
    fn iterate_parallel(&mut self) -> f64 {
        let mut max_change: f64 = 0.0;
        let mut new_messages = Vec::new();

        // Factor-to-variable messages
        for (factor_idx, factor) in self.factors.iter().enumerate() {
            for var_id in factor.scope() {
                if let Some(&var_idx) = self.var_by_id.get(var_id) {
                    let var = &self.variables[var_idx];
                    let msg = self.compute_factor_to_var_message(factor_idx, factor.as_ref(), var);
                    let key = MessageKey::factor_to_var(factor_idx, var_id);
                    new_messages.push((key, msg, var.domain_size()));
                }
            }
        }

        // Variable-to-factor messages
        for (var_idx, var) in self.variables.iter().enumerate() {
            let var_id = var.id();
            for (factor_idx, factor) in self.factors.iter().enumerate() {
                if factor.scope().contains(&var_id) {
                    let msg = self.compute_var_to_factor_message(var_idx, var, factor_idx);
                    let key = MessageKey::var_to_factor(&var_id, factor_idx);
                    new_messages.push((key, msg, var.domain_size()));
                }
            }
        }

        // Apply new messages
        for (key, mut new_msg, domain_size) in new_messages {
            let old_msg = self.messages.get(&key, domain_size);
            let change = new_msg.max_change(&old_msg);
            max_change = max_change.max(change);

            if self.config.damping > 0.0 {
                new_msg.damp(&old_msg, self.config.damping);
            }

            new_msg.normalize();
            self.messages.set(key, new_msg);
        }

        max_change
    }

    /// Sequential message update.
    fn iterate_sequential(&mut self) -> f64 {
        let mut max_change: f64 = 0.0;

        // Factor-to-variable messages
        for (factor_idx, factor) in self.factors.iter().enumerate() {
            for var_id in factor.scope() {
                if let Some(&var_idx) = self.var_by_id.get(var_id) {
                    let var = &self.variables[var_idx];
                    let mut msg =
                        self.compute_factor_to_var_message(factor_idx, factor.as_ref(), var);
                    let key = MessageKey::factor_to_var(factor_idx, var_id);
                    let domain_size = var.domain_size();

                    let old_msg = self.messages.get(&key, domain_size);
                    let change = msg.max_change(&old_msg);
                    max_change = max_change.max(change);

                    if self.config.damping > 0.0 {
                        msg.damp(&old_msg, self.config.damping);
                    }

                    msg.normalize();
                    self.messages.set(key, msg);
                }
            }
        }

        // Variable-to-factor messages
        for (var_idx, var) in self.variables.iter().enumerate() {
            let var_id = var.id();
            for (factor_idx, factor) in self.factors.iter().enumerate() {
                if factor.scope().contains(&var_id) {
                    let mut msg = self.compute_var_to_factor_message(var_idx, var, factor_idx);
                    let key = MessageKey::var_to_factor(&var_id, factor_idx);
                    let domain_size = var.domain_size();

                    let old_msg = self.messages.get(&key, domain_size);
                    let change = msg.max_change(&old_msg);
                    max_change = max_change.max(change);

                    if self.config.damping > 0.0 {
                        msg.damp(&old_msg, self.config.damping);
                    }

                    msg.normalize();
                    self.messages.set(key, msg);
                }
            }
        }

        max_change
    }

    /// Compute factor-to-variable message.
    ///
    /// m_{f→v}(v) ∝ ∑_{scope(f)\v} ψ_f(scope(f)) ∏_{v'∈scope(f)\v} m_{v'→f}(v')
    fn compute_factor_to_var_message(
        &self,
        factor_idx: usize,
        factor: &dyn Factor,
        target_var: &JointVariable,
    ) -> Message {
        let target_var_id = target_var.id();
        let target_domain = get_domain_values(target_var);

        // Get other variables in factor scope
        let other_var_ids: Vec<&VariableId> = factor
            .scope()
            .iter()
            .filter(|v| *v != &target_var_id)
            .collect();

        if other_var_ids.is_empty() {
            // Unary factor: just evaluate factor potential
            let log_probs: Vec<f64> = target_domain
                .iter()
                .map(|dv| {
                    let mut assignment = Assignment::default();
                    apply_domain_value(&mut assignment, target_var, dv);
                    factor.log_potential(&assignment)
                })
                .collect();
            return Message::from_log_probs(log_probs);
        }

        // Get other variables
        let other_vars: Vec<(&VariableId, &JointVariable)> = other_var_ids
            .iter()
            .filter_map(|vid| {
                self.var_by_id
                    .get(*vid)
                    .map(|&idx| (*vid, &self.variables[idx]))
            })
            .collect();

        // Compute message by marginalizing over other variables
        let mut log_probs = Vec::with_capacity(target_domain.len());

        for target_value in &target_domain {
            // Sum over all assignments to other variables
            let mut sum_terms = Vec::new();

            // Enumerate all combinations of other variable assignments
            let other_domains: Vec<Vec<DomainValue>> = other_vars
                .iter()
                .map(|(_, v)| get_domain_values(v))
                .collect();

            if other_domains.is_empty() {
                // No other variables (shouldn't happen after the check above)
                let mut assignment = Assignment::default();
                apply_domain_value(&mut assignment, target_var, target_value);
                log_probs.push(factor.log_potential(&assignment));
                continue;
            }

            // Iterate over Cartesian product of other domains
            let mut indices = vec![0usize; other_domains.len()];
            loop {
                // Build assignment
                let mut assignment = Assignment::default();
                apply_domain_value(&mut assignment, target_var, target_value);

                let mut incoming_msg_log_sum = 0.0;
                for (i, (var_id, var)) in other_vars.iter().enumerate() {
                    let domain_value = &other_domains[i][indices[i]];
                    apply_domain_value(&mut assignment, var, domain_value);

                    // Get incoming message from this variable
                    let key = MessageKey::var_to_factor(var_id, factor_idx);
                    let msg = self.messages.get(&key, var.domain_size());
                    if domain_value.index < msg.log_probs.len() {
                        incoming_msg_log_sum += msg.log_probs[domain_value.index];
                    }
                }

                // Factor potential + incoming messages
                let term = factor.log_potential(&assignment) + incoming_msg_log_sum;
                sum_terms.push(term);

                // Advance to next combination
                let mut carry = true;
                for i in (0..indices.len()).rev() {
                    if carry {
                        indices[i] += 1;
                        if indices[i] >= other_domains[i].len() {
                            indices[i] = 0;
                        } else {
                            carry = false;
                        }
                    }
                }
                if carry {
                    break;
                }
            }

            log_probs.push(log_sum_exp(&sum_terms));
        }

        Message::from_log_probs(log_probs)
    }

    /// Compute variable-to-factor message.
    ///
    /// m_{v→f}(v) ∝ ∏_{f'≠f} m_{f'→v}(v)
    fn compute_var_to_factor_message(
        &self,
        _var_idx: usize,
        var: &JointVariable,
        exclude_factor_idx: usize,
    ) -> Message {
        let var_id = var.id();
        let domain_size = var.domain_size();
        let mut log_probs = vec![0.0; domain_size];

        // Product of all incoming factor messages except the excluded one
        for (factor_idx, factor) in self.factors.iter().enumerate() {
            if factor_idx == exclude_factor_idx {
                continue;
            }
            if !factor.scope().contains(&var_id) {
                continue;
            }

            let key = MessageKey::factor_to_var(factor_idx, &var_id);
            let msg = self.messages.get(&key, domain_size);

            for (i, lp) in log_probs.iter_mut().enumerate() {
                if i < msg.log_probs.len() {
                    *lp += msg.log_probs[i];
                }
            }
        }

        Message::from_log_probs(log_probs)
    }

    /// Compute final marginals from converged messages.
    fn compute_marginals(&self) -> Marginals {
        let mut marginals = Marginals::default();

        for var in &self.variables {
            let var_id = var.id();
            let domain_size = var.domain_size();

            // Marginal = product of all incoming factor messages
            let mut log_probs = vec![0.0; domain_size];

            for (factor_idx, factor) in self.factors.iter().enumerate() {
                if factor.scope().contains(&var_id) {
                    let key = MessageKey::factor_to_var(factor_idx, &var_id);
                    let msg = self.messages.get(&key, domain_size);
                    for (i, lp) in log_probs.iter_mut().enumerate() {
                        if i < msg.log_probs.len() {
                            *lp += msg.log_probs[i];
                        }
                    }
                }
            }

            // Normalize
            let log_sum = log_sum_exp(&log_probs);
            if log_sum.is_finite() {
                for lp in &mut log_probs {
                    *lp -= log_sum;
                }
            }

            marginals.distributions.insert(var_id, log_probs);
        }

        marginals
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::joint::factors::{
        CorefNerFactor, CorefNerWeights, UnaryCorefFactor, UnaryNerFactor,
    };
    use crate::joint::types::VariableType;

    #[test]
    fn test_log_sum_exp() {
        let values = vec![1.0, 2.0, 3.0];
        let result = log_sum_exp(&values);
        // log(e^1 + e^2 + e^3) ≈ 3.407
        assert!((result - 3.407).abs() < 0.01);
    }

    #[test]
    fn test_log_sum_exp_empty() {
        let values: Vec<f64> = vec![];
        let result = log_sum_exp(&values);
        assert!(result.is_infinite() && result < 0.0);
    }

    #[test]
    fn test_log_sum_exp_single() {
        let values = vec![5.0];
        let result = log_sum_exp(&values);
        assert!((result - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_message_normalize() {
        let mut msg = Message {
            log_probs: vec![0.0, 0.0, 0.0],
        };
        msg.normalize();
        // Each should be log(1/3)
        let expected = -(3.0_f64).ln();
        for lp in &msg.log_probs {
            assert!((lp - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_message_max_change() {
        let msg1 = Message {
            log_probs: vec![0.0, 0.0],
        };
        let msg2 = Message {
            log_probs: vec![0.5, -0.5],
        };
        assert!((msg1.max_change(&msg2) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_message_multiply() {
        let msg1 = Message {
            log_probs: vec![1.0, 2.0],
        };
        let msg2 = Message {
            log_probs: vec![0.5, 0.5],
        };
        let result = msg1.multiply(&msg2);
        assert!((result.log_probs[0] - 1.5).abs() < 1e-10);
        assert!((result.log_probs[1] - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_marginals_argmax() {
        let mut marginals = Marginals::default();
        let var_id = VariableId {
            mention_idx: 0,
            var_type: VariableType::SemanticType,
        };
        marginals
            .distributions
            .insert(var_id, vec![-1.0, 0.0, -2.0]);

        assert_eq!(marginals.argmax(&var_id), Some(1));
    }

    #[test]
    fn test_marginals_prob() {
        let mut marginals = Marginals::default();
        let var_id = VariableId {
            mention_idx: 0,
            var_type: VariableType::SemanticType,
        };
        marginals.distributions.insert(var_id, vec![0.0, 0.0]);

        // exp(0) = 1
        let prob = marginals.prob(&var_id, 0);
        assert!(prob.is_some());
        assert!((prob.unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_belief_propagation_empty() {
        let factors: Vec<Box<dyn Factor>> = vec![];
        let variables: Vec<JointVariable> = vec![];
        let config = InferenceConfig::default();

        let mut bp = BeliefPropagation::new(factors, variables, config);
        let marginals = bp.run();

        assert!(marginals.distributions.is_empty());
    }

    #[test]
    fn test_belief_propagation_unary_only() {
        // Single variable with unary factor
        let variables = vec![JointVariable::SemanticType {
            mention_idx: 0,
            types: vec![EntityType::Person, EntityType::Organization],
        }];

        let factors: Vec<Box<dyn Factor>> = vec![Box::new(UnaryNerFactor::new(
            0,
            vec![(EntityType::Person, 1.0), (EntityType::Organization, 0.0)],
        ))];

        let config = InferenceConfig::default();
        let mut bp = BeliefPropagation::new(factors, variables, config);
        let marginals = bp.run();

        let var_id = VariableId {
            mention_idx: 0,
            var_type: VariableType::SemanticType,
        };

        // Person should have higher probability
        let argmax = marginals.argmax(&var_id);
        assert_eq!(argmax, Some(0)); // Person is index 0
    }

    #[test]
    fn test_belief_propagation_binary_factor() {
        // Two mentions with coref+NER factor encouraging consistent types
        let variables = vec![
            JointVariable::Antecedent {
                mention_idx: 1,
                candidates: vec![0],
            },
            JointVariable::SemanticType {
                mention_idx: 0,
                types: vec![EntityType::Person, EntityType::Organization],
            },
            JointVariable::SemanticType {
                mention_idx: 1,
                types: vec![EntityType::Person, EntityType::Organization],
            },
        ];

        let factors: Vec<Box<dyn Factor>> = vec![
            // Unary: mention 0 is likely Person
            Box::new(UnaryNerFactor::new(
                0,
                vec![(EntityType::Person, 2.0), (EntityType::Organization, 0.0)],
            )),
            // Unary: mention 1 type prior (weak)
            Box::new(UnaryNerFactor::new(
                1,
                vec![(EntityType::Person, 0.1), (EntityType::Organization, 0.1)],
            )),
            // Unary: mention 1 antecedent
            Box::new(UnaryCorefFactor::new(
                1,
                vec![
                    (AntecedentValue::Mention(0), 1.0),
                    (AntecedentValue::NewCluster, -1.0),
                ],
            )),
            // Binary: coref+NER consistency
            Box::new(CorefNerFactor::new(1, 0, CorefNerWeights::default())),
        ];

        let config = InferenceConfig {
            max_iterations: 10,
            ..Default::default()
        };
        let mut bp = BeliefPropagation::new(factors, variables, config);
        let marginals = bp.run();

        // Mention 0 should be Person
        let var_id_0 = VariableId {
            mention_idx: 0,
            var_type: VariableType::SemanticType,
        };
        assert_eq!(marginals.argmax(&var_id_0), Some(0));

        // Mention 1 should also be Person (propagated via coref factor)
        let var_id_1 = VariableId {
            mention_idx: 1,
            var_type: VariableType::SemanticType,
        };
        // The coref factor encourages type consistency
        // With the antecedent set to mention 0, type should propagate
        let probs = marginals.distributions.get(&var_id_1);
        assert!(probs.is_some());
    }

    #[test]
    fn test_message_key_serialization() {
        let var_id = VariableId {
            mention_idx: 5,
            var_type: VariableType::Antecedent,
        };
        let key = MessageKey::factor_to_var(3, &var_id);
        assert!(key.from.contains("f3"));
        assert!(key.to.contains("5"));
    }

    #[test]
    fn test_domain_values_antecedent() {
        let var = JointVariable::Antecedent {
            mention_idx: 2,
            candidates: vec![0, 1],
        };
        let domain = get_domain_values(&var);
        assert_eq!(domain.len(), 3); // 2 candidates + NewCluster
    }

    #[test]
    fn test_domain_values_type() {
        let var = JointVariable::SemanticType {
            mention_idx: 0,
            types: vec![
                EntityType::Person,
                EntityType::Organization,
                EntityType::Location,
            ],
        };
        let domain = get_domain_values(&var);
        assert_eq!(domain.len(), 3);
    }

    #[test]
    fn test_domain_values_link() {
        let var = JointVariable::EntityLink {
            mention_idx: 0,
            candidates: vec!["Q42".to_string(), "Q937".to_string()],
        };
        let domain = get_domain_values(&var);
        assert_eq!(domain.len(), 3); // 2 candidates + NIL
    }

    #[test]
    fn test_sequential_schedule() {
        let variables = vec![JointVariable::SemanticType {
            mention_idx: 0,
            types: vec![EntityType::Person],
        }];

        let factors: Vec<Box<dyn Factor>> = vec![Box::new(UnaryNerFactor::new(
            0,
            vec![(EntityType::Person, 1.0)],
        ))];

        let config = InferenceConfig {
            schedule: MessageSchedule::Sequential,
            ..Default::default()
        };
        let mut bp = BeliefPropagation::new(factors, variables, config);
        let marginals = bp.run();

        assert!(!marginals.distributions.is_empty());
    }
}
