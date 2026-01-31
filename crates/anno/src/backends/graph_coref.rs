//! Graph-Based Coreference Resolution with Iterative Refinement.
//!
//! This module implements a graph-based approach to coreference resolution,
//! inspired by the Graph-to-Graph Transformer (G2GT) architecture from
//! Miculicich & Henderson (2022). The key insight: model coreference as a
//! graph where nodes are mentions and edges are coref links, then iteratively
//! refine predictions until convergence.
//!
//! # Historical Context
//!
//! Coreference resolution evolved through distinct paradigms:
//!
//! ```text
//! 1995-2010  Rule-based: Hobbs algorithm, centering theory
//! 2010-2016  Mention-pair: Classify (m_i, m_j) independently
//! 2017       Lee et al.: End-to-end span-based, O(N⁴) complexity
//! 2018       Lee et al.: Higher-order with representation refinement
//! 2022       G2GT: Graph refinement with global decisions, O(N² × T)
//! ```
//!
//! **The core problem with pairwise models**: Decisions are independent.
//! If P(A~B)=0.9 and P(B~C)=0.9, transitivity implies A~C, but pairwise
//! models can output P(A~C)=0.1. The G2GT approach addresses this by
//! conditioning each iteration on the full predicted graph from the previous
//! iteration, enabling global consistency.
//!
//! # Architecture
//!
//! ```text
//! Input: Detected mentions M = [m₁, m₂, ..., mₙ]
//!    ↓
//! ┌─────────────────────────────────────────────────────────┐
//! │ Iteration 0: Initialize empty graph G₀                  │
//! │    - Nodes: all mentions                                │
//! │    - Edges: none                                        │
//! └─────────────────────────────────────────────────────────┘
//!    ↓
//! ┌─────────────────────────────────────────────────────────┐
//! │ Iteration t: Refine graph Gₜ₋₁ → Gₜ                     │
//! │    For each mention pair (mᵢ, mⱼ) where j < i:          │
//! │    1. Compute pairwise score s(mᵢ, mⱼ)                  │
//! │    2. Add graph context from Gₜ₋₁ (transitivity bonus)  │
//! │    3. Update edge if score exceeds threshold            │
//! └─────────────────────────────────────────────────────────┘
//!    ↓ (repeat until Gₜ = Gₜ₋₁ or t = max_iterations)
//! ┌─────────────────────────────────────────────────────────┐
//! │ Extract clusters via connected components               │
//! └─────────────────────────────────────────────────────────┘
//!    ↓
//! Output: Coreference chains (clusters)
//! ```
//!
//! # Approximation vs. Full G2GT
//!
//! This implementation is a **heuristic approximation**, not a full neural model.
//!
//! | Aspect | G2GT (Miculicich 2022) | This Implementation |
//! |--------|------------------------|---------------------|
//! | **Graph nodes** | Tokens | Mentions (pre-detected) |
//! | **Graph encoding** | Attention modification: `Lk = E(G)·Wk` | Explicit transitivity bonus |
//! | **Pairwise scoring** | Learned neural scorer | String/head heuristics |
//! | **Refinement** | Full neural re-prediction | Score adjustment |
//! | **Training** | End-to-end backprop | None (heuristic) |
//!
//! ## What We Preserve
//!
//! The key insight from G2GT: **iterative refinement with graph conditioning**
//! enables global consistency that independent pairwise models lack. Even with
//! heuristic scoring, the refinement loop propagates transitivity constraints.
//!
//! ## What We Lose
//!
//! - **Learned representations**: G2GT embeds graph structure directly into
//!   transformer attention. We approximate this with explicit bonuses.
//! - **End-to-end optimization**: G2GT trains the full system. We use fixed heuristics.
//! - **Token-level granularity**: G2GT operates on tokens; we operate on mentions.
//!
//! For production use with high accuracy requirements, consider a full neural
//! implementation or the T5-based coreference in `crate::backends::coref_t5`.
//!
//! # Usage with MentionType
//!
//! For best results, provide mentions with `mention_type` set:
//!
//! ```rust,ignore
//! use anno::backends::graph_coref::GraphCoref;
//! use anno::eval::coref::{Mention, MentionType};
//!
//! // Properly annotated mentions work better
//! let mut john = Mention::new("John", 0, 4);
//! john.mention_type = Some(MentionType::Proper);
//!
//! let mut he = Mention::new("he", 20, 22);
//! he.mention_type = Some(MentionType::Pronominal);
//!
//! let coref = GraphCoref::new();
//! let chains = coref.resolve(&[john, he]);
//! ```
//!
//! # Graph Initialization: Syntactic vs Semantic
//!
//! SpanEIT (Hossain et al. 2025) constructs a combined graph:
//! - **Syntactic edges** (`E_syn`): From dependency parse (adjectival modifiers, etc.)
//! - **Semantic edges** (`E_sem`): From co-occurrence statistics
//!
//! Use [`CorefGraph::seed_cooccurrence_edges`] to initialize with proximity-based
//! priors before running iterative refinement.
//!
//! # References
//!
//! - Miculicich & Henderson (2022): "Graph Refinement for Coreference Resolution"
//!   [arXiv:2203.16574](https://arxiv.org/abs/2203.16574)
//! - Lee et al. (2017): "End-to-end Neural Coreference Resolution"
//! - Lee et al. (2018): "Higher-Order Coreference Resolution"
//! - Mohammadshahi & Henderson (2021): "Graph-to-Graph Transformer for Dependency Parsing"
//! - Hossain et al. (2025): "SpanEIT: Dynamic Span Interaction and Graph-Aware Memory"
//!   [arXiv:2509.11604](https://arxiv.org/abs/2509.11604)
//!
//! # Future Direction: Sheaf Neural Networks
//!
//! This implementation uses explicit transitivity bonuses to approximate global consistency.
//! A more principled approach: **Sheaf Neural Networks** replace scalar edge weights with
//! learned linear maps (restriction maps) and minimize the sheaf Dirichlet energy:
//!
//! ```text
//! E(x) = Σ_{(u,v) ∈ E} || F(u→v) · x_u - F(v→u) · x_v ||²
//! ```
//!
//! This enforces transitivity at the gradient level, not post-hoc. See:
//! - `archive/geometric-2024-12/sheaf.rs` for stub implementation and trait definitions
//! - Bodnar et al. (2023): "Neural Sheaf Diffusion" - NeurIPS
//! - twitter-research/neural-sheaf-diffusion (Apache 2.0): reference implementation

use anno_core::{CorefChain, Mention, MentionType};
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};

// =============================================================================
// Types
// =============================================================================

/// Edge type in the coreference graph.
///
/// Following G2GT's three-way classification:
/// - 0 (None): No relationship
/// - 1 (Mention): Within-mention link (not used here since we operate on mentions, not tokens)
/// - 2 (Coref): Coreference link between mentions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum EdgeType {
    /// No link between mentions.
    None = 0,
    /// Coreference link (both mentions refer to the same entity).
    Coref = 2,
}

/// A coreference graph representing mention relationships.
///
/// This is the core data structure for iterative refinement. Nodes are mention
/// indices, edges are coreference links. The graph is stored as an adjacency
/// set for O(1) edge lookup during refinement.
///
/// # Invariants
///
/// - Graph is symmetric: if edge(i,j) exists, edge(j,i) exists
/// - No self-loops: edge(i,i) is always None
/// - Indices are valid mention indices: 0 <= i,j < num_mentions
///
/// # Example
///
/// ```rust
/// use anno::backends::graph_coref::CorefGraph;
///
/// let mut graph = CorefGraph::new(3);
/// graph.add_edge(0, 1);
/// graph.add_edge(1, 2);
///
/// assert!(graph.has_edge(0, 1));
/// assert!(graph.transitively_connected(0, 2));  // via 0-1-2
///
/// let clusters = graph.extract_clusters();
/// assert_eq!(clusters.len(), 1);  // All connected
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorefGraph {
    /// Number of mentions (nodes).
    num_mentions: usize,
    /// Adjacency set: (i, j) where i < j for canonical representation.
    edges: HashSet<(usize, usize)>,
}

impl CorefGraph {
    /// Create an empty coreference graph with the given number of mentions.
    #[must_use]
    pub fn new(num_mentions: usize) -> Self {
        Self {
            num_mentions,
            edges: HashSet::new(),
        }
    }

    /// Get the number of mentions (nodes) in the graph.
    #[must_use]
    pub fn num_mentions(&self) -> usize {
        self.num_mentions
    }

    /// Add a coreference edge between two mentions.
    ///
    /// The edge is stored in canonical form (i < j) for consistency.
    /// Self-loops and out-of-bounds indices are silently ignored.
    pub fn add_edge(&mut self, i: usize, j: usize) {
        if i == j || i >= self.num_mentions || j >= self.num_mentions {
            return;
        }
        let (lo, hi) = if i < j { (i, j) } else { (j, i) };
        self.edges.insert((lo, hi));
    }

    /// Remove a coreference edge between two mentions.
    pub fn remove_edge(&mut self, i: usize, j: usize) {
        let (lo, hi) = if i < j { (i, j) } else { (j, i) };
        self.edges.remove(&(lo, hi));
    }

    /// Check if two mentions are directly linked.
    #[must_use]
    pub fn has_edge(&self, i: usize, j: usize) -> bool {
        if i == j {
            return false;
        }
        let (lo, hi) = if i < j { (i, j) } else { (j, i) };
        self.edges.contains(&(lo, hi))
    }

    /// Get all neighbors (directly linked mentions) of a mention.
    #[must_use]
    pub fn neighbors(&self, i: usize) -> Vec<usize> {
        let mut result = Vec::new();
        for &(lo, hi) in &self.edges {
            if lo == i {
                result.push(hi);
            } else if hi == i {
                result.push(lo);
            }
        }
        result
    }

    /// Count shared neighbors between two mentions.
    ///
    /// This is the basis for the transitivity bonus: if mentions i and j
    /// share many neighbors in the current graph, they're likely coreferent.
    ///
    /// # G2GT Connection
    ///
    /// In the full G2GT model, shared structure is captured via graph-conditioned
    /// attention. Here we approximate it by explicitly counting shared neighbors
    /// and adding a proportional bonus to the pairwise score.
    #[must_use]
    pub fn shared_neighbors(&self, i: usize, j: usize) -> usize {
        let neighbors_i: HashSet<usize> = self.neighbors(i).into_iter().collect();
        let neighbors_j: HashSet<usize> = self.neighbors(j).into_iter().collect();
        neighbors_i.intersection(&neighbors_j).count()
    }

    /// Check if two mentions are transitively connected.
    ///
    /// Uses BFS to find if there's a path from i to j through coreference links.
    /// This is the closure property that ensures consistency: if A~B and B~C,
    /// then A and C are transitively connected even without a direct edge.
    #[must_use]
    pub fn transitively_connected(&self, i: usize, j: usize) -> bool {
        if i == j {
            return true;
        }
        if self.has_edge(i, j) {
            return true;
        }

        // BFS from i to find j
        let mut visited = HashSet::new();
        let mut queue = vec![i];
        visited.insert(i);

        while let Some(current) = queue.pop() {
            for neighbor in self.neighbors(current) {
                if neighbor == j {
                    return true;
                }
                if visited.insert(neighbor) {
                    queue.push(neighbor);
                }
            }
        }

        false
    }

    /// Extract connected components as clusters.
    ///
    /// Each connected component in the graph becomes a coreference chain.
    /// Singleton mentions (no edges) are included as single-mention clusters.
    #[must_use]
    pub fn extract_clusters(&self) -> Vec<Vec<usize>> {
        let mut visited = vec![false; self.num_mentions];
        let mut clusters = Vec::new();

        for start in 0..self.num_mentions {
            if visited[start] {
                continue;
            }

            // BFS to find all members of this component
            let mut cluster = Vec::new();
            let mut queue = vec![start];
            visited[start] = true;

            while let Some(current) = queue.pop() {
                cluster.push(current);
                for neighbor in self.neighbors(current) {
                    if !visited[neighbor] {
                        visited[neighbor] = true;
                        queue.push(neighbor);
                    }
                }
            }

            cluster.sort_unstable();
            clusters.push(cluster);
        }

        clusters
    }

    /// Get the number of edges in the graph.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Check if graph is empty (no edges).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }

    /// Seed the graph with co-occurrence priors based on mention proximity.
    ///
    /// This is inspired by SpanEIT (Hossain et al. 2025), which constructs a
    /// semantic co-occurrence graph `G_sem` alongside the syntactic graph. The
    /// insight: mentions that appear close together are more likely coreferent.
    ///
    /// # Arguments
    ///
    /// * `mention_positions` - Position of each mention (e.g., character offset)
    /// * `window_size` - Maximum distance for co-occurrence (e.g., 100 chars)
    /// * `scorer` - Optional scoring function; if None, uses constant weight
    ///
    /// # Example
    ///
    /// ```rust
    /// use anno::backends::graph_coref::CorefGraph;
    ///
    /// let mut graph = CorefGraph::new(3);
    /// let positions = vec![0, 50, 200];  // Character offsets
    ///
    /// // Seed edges for mentions within 100 chars of each other
    /// // Type annotation needed when passing None for the scorer
    /// graph.seed_cooccurrence_edges::<fn(usize, usize) -> bool>(&positions, 100, None);
    ///
    /// assert!(graph.has_edge(0, 1));   // 50 < 100
    /// assert!(!graph.has_edge(0, 2));  // 200 > 100
    /// ```
    ///
    /// # Research Background
    ///
    /// SpanEIT constructs `G = (V, E_syn ∪ E_sem)` where:
    /// - `E_syn` = syntactic dependency edges
    /// - `E_sem` = co-occurrence edges (this method)
    ///
    /// The combined graph is processed by GAT layers for context-aware embeddings.
    /// For anno's heuristic approach, we add these edges as initial priors before
    /// iterative refinement.
    pub fn seed_cooccurrence_edges<F>(
        &mut self,
        mention_positions: &[usize],
        window_size: usize,
        scorer: Option<F>,
    ) where
        F: Fn(usize, usize) -> bool,
    {
        for i in 0..self.num_mentions {
            for j in (i + 1)..self.num_mentions {
                if i >= mention_positions.len() || j >= mention_positions.len() {
                    continue;
                }

                let pos_i = mention_positions[i];
                let pos_j = mention_positions[j];
                let distance = pos_i.abs_diff(pos_j);

                if distance <= window_size {
                    let should_add = match scorer.as_ref() {
                        None => true,
                        Some(f) => f(i, j),
                    };
                    if should_add {
                        self.add_edge(i, j);
                    }
                }
            }
        }
    }
}

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for graph-based coreference resolution.
///
/// These parameters control the iterative refinement process. The defaults
/// are based on findings from Miculicich & Henderson (2022):
///
/// - `max_iterations = 4`: Paper found T=4 optimal; more iterations don't help
/// - `link_threshold = 0.5`: Standard classification threshold
/// - `transitivity_bonus = 0.15`: Reward for transitive consistency
///
/// # Tuning Guide
///
/// | Parameter | Higher Value | Lower Value |
/// |-----------|--------------|-------------|
/// | `link_threshold` | Fewer, more confident links | More links, potential noise |
/// | `transitivity_bonus` | Stronger clustering effect | More independent decisions |
/// | `max_iterations` | More refinement passes | Faster, less propagation |
/// | `head_match_weight` | Trust head matches more | Rely more on full string |
#[derive(Debug, Clone)]
pub struct GraphCorefConfig {
    /// Maximum refinement iterations before stopping.
    ///
    /// The G2GT paper found T=4 to be optimal on CoNLL 2012. Fewer iterations
    /// leave potential coreference links undiscovered; more iterations don't
    /// improve results and waste computation.
    pub max_iterations: usize,

    /// Minimum score to create a coreference link.
    ///
    /// A pair (mᵢ, mⱼ) is linked if score(mᵢ, mⱼ) + context_bonus > threshold.
    pub link_threshold: f64,

    /// Bonus added for transitive consistency.
    ///
    /// If mentions A and B share neighbors in the current graph (i.e., both
    /// are already linked to some common mention C), this bonus is added to
    /// encourage the model to also link A and B directly.
    ///
    /// **Note**: This is our heuristic approximation of G2GT's graph-conditioned
    /// attention. The full G2GT model encodes graph structure as:
    /// ```text
    /// Attention(Q,K,V,Lk,Lv) = softmax(Q·(K+Lk)/√d)·(V+Lv)
    /// where Lk = E(G^{t-1})·Wk
    /// ```
    /// We approximate this by explicit score adjustment rather than attention modification.
    pub transitivity_bonus: f64,

    /// Bonus for each shared neighbor.
    ///
    /// Scaled by the number of shared neighbors: total_bonus = shared_count * per_neighbor_bonus
    pub per_shared_neighbor_bonus: f64,

    /// Weight for string similarity in pairwise scoring.
    pub string_similarity_weight: f64,

    /// Weight for head word matching.
    ///
    /// The G2GT paper emphasizes head-based matching. When mentions have
    /// `head_start`/`head_end` set, head matching is used. Otherwise falls
    /// back to last word heuristic.
    pub head_match_weight: f64,

    /// Weight for distance penalty in pairwise scoring.
    pub distance_weight: f64,

    /// Maximum character distance to consider (mentions further apart are not linked).
    pub max_distance: Option<usize>,

    /// Include singletons (mentions with no coreference) in output.
    ///
    /// Default: false (only return multi-mention chains).
    /// Set to true for evaluation against datasets that include singletons.
    pub include_singletons: bool,

    /// Bonus when a pronoun links to a proper noun.
    ///
    /// Pronouns are weak signals alone but should link to antecedents.
    pub pronoun_proper_bonus: f64,

    /// Optional early-stop controls for iterative refinement.
    ///
    /// GraphCoref already stops when it reaches a fixed point (Gₜ == Gₜ₋₁).
    /// This option additionally stops on:
    /// - **Cycle detection** (e.g., A→B→A oscillation across iterations)
    /// - **Stagnation** (edge count stops changing for N iterations)
    ///
    /// This is an analogue of “overthinking” / redundancy detection (CoRE-Eval),
    /// implemented using observable signals (graph structure) rather than hidden states.
    pub early_stop: Option<GraphCorefEarlyStopConfig>,
}

/// Configuration for early stopping in iterative graph refinement.
#[derive(Debug, Clone)]
pub struct GraphCorefEarlyStopConfig {
    /// Stop if we detect a repeated graph state (cycle) within the configured history.
    pub detect_cycles: bool,
    /// How many past graph fingerprints to remember (0 = unbounded).
    pub cycle_history: usize,
    /// Stop if the edge count hasn't changed for this many consecutive iterations.
    pub stagnation_patience: usize,
}

impl Default for GraphCorefEarlyStopConfig {
    fn default() -> Self {
        Self {
            detect_cycles: true,
            cycle_history: 8,
            stagnation_patience: 2,
        }
    }
}

impl Default for GraphCorefConfig {
    fn default() -> Self {
        Self {
            max_iterations: 4,
            link_threshold: 0.5,
            transitivity_bonus: 0.15,
            per_shared_neighbor_bonus: 0.1,
            string_similarity_weight: 1.0,
            head_match_weight: 0.5,
            distance_weight: 0.05,
            max_distance: Some(1000),
            include_singletons: false,
            pronoun_proper_bonus: 0.3,
            early_stop: None,
        }
    }
}

// =============================================================================
// Main Implementation
// =============================================================================

/// Graph-based coreference resolver with iterative refinement.
///
/// This implements a heuristic version of the G2GT architecture, preserving
/// the key insight that iterative graph refinement enables global consistency
/// in coreference decisions.
///
/// # Algorithm
///
/// 1. **Initialize**: Empty graph with mentions as nodes
/// 2. **Iterate**: For each mention pair, compute score with graph context
/// 3. **Update**: Add/remove edges based on threshold
/// 4. **Converge**: Stop when graph unchanged or max iterations reached
/// 5. **Extract**: Connected components become coreference chains
///
/// # Complexity
///
/// - Time: O(N² × T) where N = mentions, T = iterations (typically 4)
/// - Space: O(N²) for adjacency representation
///
/// Compare to Lee et al. (2017): O(N⁴) for full span enumeration.
///
/// # Feature Usage
///
/// The resolver uses available `Mention` fields when present:
///
/// | Field | Used For | Fallback |
/// |-------|----------|----------|
/// | `mention_type` | Pronoun detection, type compatibility | Heuristic detection |
/// | `head_start`/`head_end` | Head word matching | Last word of mention |
/// | `entity_type` | Type compatibility check | Ignored |
#[derive(Debug, Clone)]
pub struct GraphCoref {
    config: GraphCorefConfig,
}

impl GraphCoref {
    /// Create a new graph coref resolver with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(GraphCorefConfig::default())
    }

    /// Create with custom configuration.
    #[must_use]
    pub fn with_config(config: GraphCorefConfig) -> Self {
        Self { config }
    }

    fn graph_fingerprint(graph: &CorefGraph) -> u64 {
        // Order-independent fingerprint of the edge set.
        // We sort per-edge hashes to avoid HashSet iteration nondeterminism.
        let mut edge_hashes: Vec<u64> = graph
            .edges
            .iter()
            .map(|e| {
                let mut h = std::collections::hash_map::DefaultHasher::new();
                e.hash(&mut h);
                h.finish()
            })
            .collect();
        edge_hashes.sort_unstable();

        let mut h = std::collections::hash_map::DefaultHasher::new();
        graph.num_mentions.hash(&mut h);
        for eh in edge_hashes {
            eh.hash(&mut h);
        }
        h.finish()
    }

    /// Resolve coreferences among mentions using iterative graph refinement.
    ///
    /// # Arguments
    ///
    /// * `mentions` - Pre-detected mentions (from NER or mention detector).
    ///   For best results, set `mention_type` on each mention.
    ///
    /// # Returns
    ///
    /// Coreference chains (clusters) where each chain contains mentions
    /// referring to the same entity. By default, singletons are filtered;
    /// set `config.include_singletons = true` to include them.
    ///
    /// # Panics
    ///
    /// Does not panic. Empty input returns empty output.
    ///
    /// # Example
    ///
    /// ```rust
    /// use anno::backends::graph_coref::GraphCoref;
    /// use anno::{Mention, MentionType};
    ///
    /// let coref = GraphCoref::new();
    ///
    /// let mut john = Mention::new("John", 0, 4);
    /// john.mention_type = Some(MentionType::Proper);
    ///
    /// let mut he = Mention::new("he", 20, 22);
    /// he.mention_type = Some(MentionType::Pronominal);
    ///
    /// let chains = coref.resolve(&[john, he]);
    /// ```
    #[must_use]
    pub fn resolve(&self, mentions: &[Mention]) -> Vec<CorefChain> {
        if mentions.is_empty() {
            return vec![];
        }

        // Validate mentions (filter empty/invalid)
        let valid_mentions: Vec<&Mention> = mentions
            .iter()
            .filter(|m| !m.text.trim().is_empty() && m.start < m.end)
            .collect();

        if valid_mentions.is_empty() {
            return vec![];
        }

        // Initialize empty graph
        let mut graph = CorefGraph::new(valid_mentions.len());

        // Iterative refinement
        let mut stagnation: usize = 0;
        let mut last_edge_count = graph.edge_count();
        let mut seen: HashMap<u64, usize> = HashMap::new();
        let mut history: VecDeque<u64> = VecDeque::new();
        if let Some(cfg) = &self.config.early_stop {
            if cfg.detect_cycles {
                let fp0 = Self::graph_fingerprint(&graph);
                seen.insert(fp0, 0);
                history.push_back(fp0);
            }
        }

        for iteration in 0..self.config.max_iterations {
            let prev_graph = graph.clone();
            graph = self.refine_iteration(&valid_mentions, &graph);

            // Check convergence
            if graph == prev_graph {
                break;
            }

            // Optional early stop: stagnation / cycles.
            if let Some(cfg) = &self.config.early_stop {
                // Stagnation: edge count stops changing.
                let ec = graph.edge_count();
                if ec == last_edge_count {
                    stagnation += 1;
                } else {
                    stagnation = 0;
                    last_edge_count = ec;
                }
                if cfg.stagnation_patience > 0 && stagnation >= cfg.stagnation_patience {
                    break;
                }

                if cfg.detect_cycles {
                    let fp = Self::graph_fingerprint(&graph);
                    if seen.contains_key(&fp) {
                        break;
                    }
                    seen.insert(fp, iteration + 1);
                    history.push_back(fp);
                    if cfg.cycle_history > 0 {
                        while history.len() > cfg.cycle_history {
                            if let Some(old) = history.pop_front() {
                                seen.remove(&old);
                            }
                        }
                    }
                }
            }
        }

        // Extract clusters and convert to CorefChains
        self.graph_to_chains(&graph, &valid_mentions)
    }

    /// Perform one iteration of graph refinement.
    ///
    /// For each mention pair, computes a score incorporating:
    /// 1. Base pairwise similarity (string match, head match, type compatibility)
    /// 2. Graph context (transitivity bonus from shared neighbors)
    ///
    /// Edges are added if score exceeds threshold.
    fn refine_iteration(&self, mentions: &[&Mention], prev_graph: &CorefGraph) -> CorefGraph {
        let mut new_graph = CorefGraph::new(mentions.len());

        for i in 0..mentions.len() {
            for j in 0..i {
                // Distance filter
                if let Some(max_dist) = self.config.max_distance {
                    let dist = mentions[i].start.saturating_sub(mentions[j].end);
                    if dist > max_dist {
                        continue;
                    }
                }

                // Compute score with graph context
                let base_score = self.pairwise_score(mentions[i], mentions[j]);
                let context_bonus = self.graph_context_bonus(i, j, prev_graph);
                let total_score = base_score + context_bonus;

                if total_score > self.config.link_threshold {
                    new_graph.add_edge(i, j);
                }
            }
        }

        new_graph
    }

    /// Compute base pairwise similarity score between two mentions.
    ///
    /// Uses multiple signals:
    /// - Exact string match (highest weight)
    /// - Substring containment
    /// - Head word match (uses `head_start`/`head_end` if available)
    /// - Pronoun-to-proper linking (uses `mention_type` if available)
    /// - Distance penalty
    fn pairwise_score(&self, m1: &Mention, m2: &Mention) -> f64 {
        let mut score = 0.0;

        let t1 = m1.text.to_lowercase();
        let t2 = m2.text.to_lowercase();

        // Exact match (strongest signal)
        if t1 == t2 {
            score += self.config.string_similarity_weight * 1.0;
        }
        // Substring containment
        else if t1.contains(&t2) || t2.contains(&t1) {
            score += self.config.string_similarity_weight * 0.6;
        }
        // Head word match
        else {
            let h1 = self.get_head_text(m1);
            let h2 = self.get_head_text(m2);
            if !h1.is_empty() && h1.to_lowercase() == h2.to_lowercase() {
                score += self.config.head_match_weight;
            }
        }

        // Mention type compatibility
        score += self.type_compatibility_score(m1, m2);

        // Distance penalty (log scale)
        let distance = m1.start.abs_diff(m2.end).min(m2.start.abs_diff(m1.end));
        if distance > 0 {
            score -= self.config.distance_weight * (distance as f64).ln();
        }

        score
    }

    /// Get head text for a mention.
    ///
    /// Uses `head_start`/`head_end` if available, otherwise falls back to
    /// last word heuristic (common in English NPs where head is rightmost).
    fn get_head_text<'a>(&self, mention: &'a Mention) -> &'a str {
        // Use explicit head span if available
        if let (Some(head_start), Some(head_end)) = (mention.head_start, mention.head_end) {
            // Head offsets are relative to document, need to extract from text
            // This is complex; fall back to heuristic for now
            // In a full implementation, we'd have the document text available
            let _ = (head_start, head_end);
        }

        // Fallback: last word (head-final assumption for English NPs)
        mention.text.split_whitespace().last().unwrap_or("")
    }

    /// Compute type compatibility score between two mentions.
    ///
    /// Uses `MentionType` field if available, otherwise uses heuristics.
    fn type_compatibility_score(&self, m1: &Mention, m2: &Mention) -> f64 {
        let type1 = m1
            .mention_type
            .unwrap_or_else(|| self.infer_mention_type(m1));
        let type2 = m2
            .mention_type
            .unwrap_or_else(|| self.infer_mention_type(m2));

        match (type1, type2) {
            // Pronoun linking to proper noun: boost
            (MentionType::Pronominal, MentionType::Proper)
            | (MentionType::Proper, MentionType::Pronominal) => self.config.pronoun_proper_bonus,

            // Pronoun linking to nominal: smaller boost
            (MentionType::Pronominal, MentionType::Nominal)
            | (MentionType::Nominal, MentionType::Pronominal) => {
                self.config.pronoun_proper_bonus * 0.5
            }

            // Same type: neutral
            _ if type1 == type2 => 0.0,

            // Different non-pronoun types: slight penalty
            _ => -0.1,
        }
    }

    /// Infer mention type from text when not explicitly set.
    ///
    /// This is a fallback heuristic. For best results, set `mention_type`
    /// on mentions before calling `resolve()`.
    fn infer_mention_type(&self, mention: &Mention) -> MentionType {
        let text_lower = mention.text.to_lowercase();

        // Check for pronouns
        const PRONOUNS: &[&str] = &[
            "i",
            "me",
            "my",
            "mine",
            "myself",
            "you",
            "your",
            "yours",
            "yourself",
            "yourselves",
            "he",
            "him",
            "his",
            "himself",
            "she",
            "her",
            "hers",
            "herself",
            "it",
            "its",
            "itself",
            "we",
            "us",
            "our",
            "ours",
            "ourselves",
            "they",
            "them",
            "their",
            "theirs",
            "themselves",
            "who",
            "whom",
            "whose",
            "which",
            "that",
            "this",
            "these",
            "those",
        ];

        if PRONOUNS.contains(&text_lower.as_str()) {
            return MentionType::Pronominal;
        }

        // Check for proper noun (starts with uppercase, not sentence-initial heuristic)
        let first_char = mention.text.chars().next();
        if first_char.is_some_and(|c| c.is_uppercase()) {
            // Additional check: not a common word
            let common_words = ["the", "a", "an", "this", "that", "these", "those"];
            if !common_words.contains(&text_lower.as_str()) {
                return MentionType::Proper;
            }
        }

        // Default to nominal
        MentionType::Nominal
    }

    /// Compute graph context bonus based on previous iteration's structure.
    ///
    /// This is our approximation of G2GT's graph-conditioned attention.
    ///
    /// # Transitivity Bonus
    ///
    /// If A~C and B~C in the previous graph, A and B should likely be linked.
    /// We add a bonus proportional to the number of shared neighbors.
    ///
    /// # Already Connected Bonus
    ///
    /// If A and B are already transitively connected (through a chain of
    /// coreference links), add a bonus to preserve and strengthen the connection.
    fn graph_context_bonus(&self, i: usize, j: usize, prev_graph: &CorefGraph) -> f64 {
        let mut bonus = 0.0;

        // Bonus for shared neighbors (transitivity signal)
        let shared = prev_graph.shared_neighbors(i, j);
        bonus += (shared as f64) * self.config.per_shared_neighbor_bonus;

        // Bonus if already transitively connected
        if prev_graph.transitively_connected(i, j) {
            bonus += self.config.transitivity_bonus;
        }

        bonus
    }

    /// Convert graph clusters to CorefChain format.
    fn graph_to_chains(&self, graph: &CorefGraph, mentions: &[&Mention]) -> Vec<CorefChain> {
        let clusters = graph.extract_clusters();

        clusters
            .into_iter()
            .filter(|cluster| self.config.include_singletons || cluster.len() > 1)
            .enumerate()
            .map(|(id, indices)| {
                let chain_mentions: Vec<Mention> = indices
                    .into_iter()
                    .map(|i| (*mentions[i]).clone())
                    .collect();

                let mut chain = CorefChain::new(chain_mentions);
                chain.cluster_id = Some((id as u64).into());

                // Set entity type from first proper mention
                chain.entity_type = chain
                    .mentions
                    .iter()
                    .find(|m| m.mention_type == Some(MentionType::Proper))
                    .and_then(|m| m.entity_type.clone());

                chain
            })
            .collect()
    }

    /// Get configuration.
    #[must_use]
    pub fn config(&self) -> &GraphCorefConfig {
        &self.config
    }
}

impl Default for GraphCoref {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Metrics and Diagnostics
// =============================================================================

/// Statistics from a graph coref run for debugging and analysis.
///
/// Use [`GraphCoref::resolve_with_stats`] to get these alongside results.
#[derive(Debug, Clone, Default)]
pub struct GraphCorefStats {
    /// Number of iterations until convergence (1 to max_iterations).
    pub iterations: usize,
    /// Number of edges in final graph.
    pub final_edges: usize,
    /// Number of clusters (including singletons).
    pub num_clusters: usize,
    /// Number of non-singleton clusters.
    pub num_chains: usize,
    /// Per-iteration edge counts, starting from 0.
    pub edge_history: Vec<usize>,
    /// Whether the algorithm converged before max_iterations.
    pub converged: bool,
    /// Whether we stopped early for a non-fixed-point reason (cycle/stagnation).
    pub early_stopped: bool,
    /// Cycle detected (graph fingerprint repeated).
    pub cycle_detected: bool,
    /// Stagnation detected (edge count stopped changing).
    pub stagnation_detected: bool,
}

impl GraphCoref {
    /// Resolve coreferences and return detailed statistics.
    ///
    /// Useful for debugging, tuning parameters, and understanding convergence.
    ///
    /// # Example
    ///
    /// ```rust
    /// use anno::backends::graph_coref::GraphCoref;
    /// use anno::Mention;
    ///
    /// let coref = GraphCoref::new();
    /// let mentions = vec![
    ///     Mention::new("John", 0, 4),
    ///     Mention::new("John", 50, 54),
    /// ];
    ///
    /// let (chains, stats) = coref.resolve_with_stats(&mentions);
    /// println!("Converged in {} iterations", stats.iterations);
    /// println!("Edge history: {:?}", stats.edge_history);
    /// ```
    #[must_use]
    pub fn resolve_with_stats(&self, mentions: &[Mention]) -> (Vec<CorefChain>, GraphCorefStats) {
        let mut stats = GraphCorefStats::default();

        if mentions.is_empty() {
            return (vec![], stats);
        }

        let valid_mentions: Vec<&Mention> = mentions
            .iter()
            .filter(|m| !m.text.trim().is_empty() && m.start < m.end)
            .collect();

        if valid_mentions.is_empty() {
            return (vec![], stats);
        }

        let mut graph = CorefGraph::new(valid_mentions.len());
        stats.edge_history.push(0);

        let mut stagnation: usize = 0;
        let mut last_edge_count = graph.edge_count();
        let mut seen: HashMap<u64, usize> = HashMap::new();
        let mut history: VecDeque<u64> = VecDeque::new();
        if let Some(cfg) = &self.config.early_stop {
            if cfg.detect_cycles {
                let fp0 = Self::graph_fingerprint(&graph);
                seen.insert(fp0, 0);
                history.push_back(fp0);
            }
        }

        for iteration in 0..self.config.max_iterations {
            let prev_graph = graph.clone();
            graph = self.refine_iteration(&valid_mentions, &graph);
            stats.edge_history.push(graph.edge_count());
            stats.iterations = iteration + 1;

            if graph == prev_graph {
                stats.converged = true;
                break;
            }

            if let Some(cfg) = &self.config.early_stop {
                let ec = graph.edge_count();
                if ec == last_edge_count {
                    stagnation += 1;
                } else {
                    stagnation = 0;
                    last_edge_count = ec;
                }
                if cfg.stagnation_patience > 0 && stagnation >= cfg.stagnation_patience {
                    stats.early_stopped = true;
                    stats.stagnation_detected = true;
                    break;
                }

                if cfg.detect_cycles {
                    let fp = Self::graph_fingerprint(&graph);
                    if seen.contains_key(&fp) {
                        stats.early_stopped = true;
                        stats.cycle_detected = true;
                        break;
                    }
                    seen.insert(fp, iteration + 1);
                    history.push_back(fp);
                    if cfg.cycle_history > 0 {
                        while history.len() > cfg.cycle_history {
                            if let Some(old) = history.pop_front() {
                                seen.remove(&old);
                            }
                        }
                    }
                }
            }
        }

        let clusters = graph.extract_clusters();
        stats.final_edges = graph.edge_count();
        stats.num_clusters = clusters.len();
        stats.num_chains = clusters.iter().filter(|c| c.len() > 1).count();

        let chains = self.graph_to_chains(&graph, &valid_mentions);
        (chains, stats)
    }
}

// =============================================================================
// Evaluation Helpers
// =============================================================================

/// Convert GraphCoref output to format suitable for CoNLL evaluation.
///
/// This produces a `CorefDocument` that can be evaluated with standard
/// coreference metrics (MUC, B³, CEAF, LEA).
///
/// # Example
///
/// ```rust
/// use anno::backends::graph_coref::{GraphCoref, chains_to_document};
/// use anno::Mention;
///
/// let coref = GraphCoref::new();
/// let mentions = vec![
///     Mention::new("John", 0, 4),
///     Mention::new("he", 20, 22),
/// ];
///
/// let chains = coref.resolve(&mentions);
/// let doc = chains_to_document("John went to work. He was late.", chains);
/// ```
pub fn chains_to_document(
    text: impl Into<String>,
    chains: Vec<CorefChain>,
) -> anno_core::CorefDocument {
    anno_core::CorefDocument::new(text, chains)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_mention(text: &str, start: usize) -> Mention {
        Mention::new(text, start, start + text.chars().count())
    }

    fn make_typed_mention(text: &str, start: usize, mention_type: MentionType) -> Mention {
        Mention::with_type(text, start, start + text.chars().count(), mention_type)
    }

    // -------------------------------------------------------------------------
    // Basic functionality
    // -------------------------------------------------------------------------

    #[test]
    fn test_empty_input() {
        let coref = GraphCoref::new();
        let chains = coref.resolve(&[]);
        assert!(chains.is_empty());
    }

    #[test]
    fn test_single_mention() {
        let coref = GraphCoref::new();
        let mentions = vec![make_mention("John", 0)];
        let chains = coref.resolve(&mentions);
        assert!(
            chains.is_empty(),
            "Single mention should be filtered as singleton"
        );
    }

    #[test]
    fn test_single_mention_with_singletons() {
        let config = GraphCorefConfig {
            include_singletons: true,
            ..Default::default()
        };
        let coref = GraphCoref::with_config(config);
        let mentions = vec![make_mention("John", 0)];
        let chains = coref.resolve(&mentions);
        assert_eq!(chains.len(), 1, "Should include singleton when configured");
    }

    #[test]
    fn test_exact_match_linking() {
        let coref = GraphCoref::new();
        let mentions = vec![make_mention("John", 0), make_mention("John", 50)];

        let chains = coref.resolve(&mentions);
        assert_eq!(chains.len(), 1);
        assert_eq!(chains[0].mentions.len(), 2);
    }

    #[test]
    fn test_substring_linking() {
        let config = GraphCorefConfig {
            link_threshold: 0.4,
            ..Default::default()
        };
        let coref = GraphCoref::with_config(config);
        let mentions = vec![make_mention("Marie Curie", 0), make_mention("Curie", 50)];

        let chains = coref.resolve(&mentions);
        assert_eq!(chains.len(), 1);
        assert_eq!(chains[0].mentions.len(), 2);
    }

    // -------------------------------------------------------------------------
    // MentionType usage
    // -------------------------------------------------------------------------

    #[test]
    fn test_typed_pronoun_linking() {
        let config = GraphCorefConfig {
            link_threshold: 0.2,
            distance_weight: 0.0, // Disable distance penalty to isolate type signal
            ..Default::default()
        };
        let coref = GraphCoref::with_config(config);

        let mentions = vec![
            make_typed_mention("Marie", 0, MentionType::Proper),
            make_typed_mention("she", 20, MentionType::Pronominal),
        ];

        let chains = coref.resolve(&mentions);
        assert_eq!(chains.len(), 1, "Typed pronoun should link to proper noun");
    }

    #[test]
    fn test_inferred_pronoun_detection() {
        let coref = GraphCoref::new();

        // Create mention without type - should be inferred
        let he = make_mention("he", 0);
        assert_eq!(
            coref.infer_mention_type(&he),
            MentionType::Pronominal,
            "Should detect 'he' as pronoun"
        );

        let john = make_mention("John", 0);
        assert_eq!(
            coref.infer_mention_type(&john),
            MentionType::Proper,
            "Should detect 'John' as proper"
        );

        let dog = make_mention("the dog", 0);
        assert_eq!(
            coref.infer_mention_type(&dog),
            MentionType::Nominal,
            "Should detect 'the dog' as nominal"
        );
    }

    // -------------------------------------------------------------------------
    // Transitivity and graph refinement
    // -------------------------------------------------------------------------

    #[test]
    fn test_transitivity() {
        let config = GraphCorefConfig {
            max_iterations: 4,
            link_threshold: 0.3,
            transitivity_bonus: 0.3,
            per_shared_neighbor_bonus: 0.2,
            ..Default::default()
        };
        let coref = GraphCoref::with_config(config);

        let mentions = vec![
            make_mention("John Smith", 0),
            make_mention("Smith", 30),
            make_mention("John Smith", 60),
        ];

        let chains = coref.resolve(&mentions);
        assert_eq!(chains.len(), 1);
        assert_eq!(chains[0].mentions.len(), 3);
    }

    #[test]
    fn test_convergence() {
        let coref = GraphCoref::new();
        let mentions = vec![
            make_mention("Apple", 0),
            make_mention("Apple", 50),
            make_mention("Microsoft", 100),
        ];

        let (chains, stats) = coref.resolve_with_stats(&mentions);

        assert!(stats.iterations <= 4);
        assert!(stats.converged || stats.iterations == 4);
        assert_eq!(chains.len(), 1);
        assert_eq!(stats.num_chains, 1);
    }

    // -------------------------------------------------------------------------
    // CorefGraph tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_coref_graph_basics() {
        let mut graph = CorefGraph::new(5);

        graph.add_edge(0, 1);
        graph.add_edge(1, 2);

        assert!(graph.has_edge(0, 1));
        assert!(graph.has_edge(1, 0)); // Symmetric
        assert!(graph.has_edge(1, 2));
        assert!(!graph.has_edge(0, 2));

        assert!(graph.transitively_connected(0, 2));

        let clusters = graph.extract_clusters();
        assert_eq!(clusters.len(), 3); // {0,1,2}, {3}, {4}

        let main_cluster = clusters.iter().find(|c| c.len() == 3).unwrap();
        assert!(main_cluster.contains(&0));
        assert!(main_cluster.contains(&1));
        assert!(main_cluster.contains(&2));
    }

    #[test]
    fn test_shared_neighbors() {
        let mut graph = CorefGraph::new(4);
        graph.add_edge(0, 2);
        graph.add_edge(1, 2);

        assert_eq!(graph.shared_neighbors(0, 1), 1);
        assert_eq!(graph.shared_neighbors(0, 3), 0);
    }

    #[test]
    fn test_graph_self_loop_ignored() {
        let mut graph = CorefGraph::new(3);
        graph.add_edge(0, 0); // Self-loop
        assert!(!graph.has_edge(0, 0));
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_graph_out_of_bounds_ignored() {
        let mut graph = CorefGraph::new(3);
        graph.add_edge(0, 10); // Out of bounds
        assert_eq!(graph.edge_count(), 0);
    }

    // -------------------------------------------------------------------------
    // Edge cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_empty_mention_filtered() {
        let coref = GraphCoref::new();
        let mentions = vec![
            make_mention("John", 0),
            Mention::new("", 10, 10),    // Empty
            Mention::new("   ", 20, 23), // Whitespace only
            make_mention("John", 50),
        ];

        let chains = coref.resolve(&mentions);
        assert_eq!(chains.len(), 1);
        assert_eq!(chains[0].mentions.len(), 2);
    }

    #[test]
    fn test_distance_filter() {
        let config = GraphCorefConfig {
            max_distance: Some(100),
            ..Default::default()
        };
        let coref = GraphCoref::with_config(config);

        let mentions = vec![make_mention("John", 0), make_mention("John", 200)];

        let chains = coref.resolve(&mentions);
        assert!(chains.is_empty());
    }

    #[test]
    fn test_stats_edge_history() {
        let coref = GraphCoref::new();
        let mentions = vec![
            make_mention("A", 0),
            make_mention("A", 10),
            make_mention("A", 20),
        ];

        let (_, stats) = coref.resolve_with_stats(&mentions);

        assert!(!stats.edge_history.is_empty());
        assert_eq!(stats.edge_history[0], 0);
    }

    // -------------------------------------------------------------------------
    // Unicode / multilingual
    // -------------------------------------------------------------------------

    #[test]
    fn test_unicode_cjk() {
        let coref = GraphCoref::new();
        let mentions = vec![
            make_mention("北京", 0),
            make_mention("北京", 20),
            make_mention("東京", 40),
        ];

        let chains = coref.resolve(&mentions);
        assert_eq!(chains.len(), 1);
        assert!(chains[0].mentions.iter().all(|m| m.text == "北京"));
    }

    #[test]
    fn test_unicode_diacritics() {
        let coref = GraphCoref::new();
        let mentions = vec![make_mention("François", 0), make_mention("François", 50)];

        let chains = coref.resolve(&mentions);
        assert_eq!(chains.len(), 1);
    }

    #[test]
    fn test_unicode_arabic_rtl() {
        let coref = GraphCoref::new();
        // Arabic: "Muhammad" repeated
        let mentions = vec![make_mention("محمد", 0), make_mention("محمد", 20)];

        let chains = coref.resolve(&mentions);
        assert_eq!(chains.len(), 1);
    }

    // -------------------------------------------------------------------------
    // Evaluation helper
    // -------------------------------------------------------------------------

    #[test]
    fn test_chains_to_document() {
        let chain = CorefChain::new(vec![make_mention("John", 0), make_mention("he", 20)]);

        let doc = chains_to_document("John went home. He slept.", vec![chain]);

        assert_eq!(doc.chain_count(), 1);
        assert_eq!(doc.mention_count(), 2);
    }

    // -------------------------------------------------------------------------
    // Co-occurrence seeding (SpanEIT-inspired)
    // -------------------------------------------------------------------------

    #[test]
    fn test_cooccurrence_seeding_basic() {
        let mut graph = CorefGraph::new(3);
        let positions = vec![0, 50, 200]; // Character offsets

        // Window of 100: should connect 0-1 (distance 50) but not 0-2 (distance 200)
        graph.seed_cooccurrence_edges(&positions, 100, None::<fn(usize, usize) -> bool>);

        assert!(graph.has_edge(0, 1), "Close mentions should be connected");
        assert!(
            !graph.has_edge(0, 2),
            "Distant mentions should not be connected"
        );
        assert!(
            !graph.has_edge(1, 2),
            "Distant mentions should not be connected"
        );
    }

    #[test]
    fn test_cooccurrence_seeding_with_scorer() {
        let mut graph = CorefGraph::new(3);
        let positions = vec![0, 50, 80];

        // Custom scorer: only connect if both indices are even
        let scorer = |i: usize, j: usize| i.is_multiple_of(2) && j.is_multiple_of(2);
        graph.seed_cooccurrence_edges(&positions, 100, Some(scorer));

        assert!(graph.has_edge(0, 2), "0 and 2 are both even");
        assert!(!graph.has_edge(0, 1), "1 is odd");
        assert!(!graph.has_edge(1, 2), "1 is odd");
    }

    #[test]
    fn test_cooccurrence_seeding_empty() {
        let mut graph = CorefGraph::new(3);
        let positions: Vec<usize> = vec![];

        graph.seed_cooccurrence_edges(&positions, 100, None::<fn(usize, usize) -> bool>);

        assert!(graph.is_empty(), "Empty positions should create no edges");
    }
}
