//! Leiden algorithm for hierarchical community detection.

use anno_core::GraphDocument;
use petgraph::graph::{Graph, NodeIndex};
use petgraph::Undirected;
use std::collections::HashMap;

/// Leiden algorithm implementation for community detection.
///
/// The Leiden algorithm is an improvement over the Louvain algorithm,
/// guaranteeing well-connected communities. This implementation provides
/// a basic version suitable for hierarchical clustering.
pub struct Leiden {
    resolution: f32,
    random_seed: Option<u64>,
}

impl Leiden {
    /// Create a new Leiden algorithm instance.
    pub fn new() -> Self {
        Self {
            resolution: 1.0,
            random_seed: None,
        }
    }

    /// Set the resolution parameter.
    ///
    /// Higher resolution values lead to more, smaller communities.
    /// Lower values lead to fewer, larger communities.
    pub fn with_resolution(mut self, resolution: f32) -> Self {
        self.resolution = resolution;
        self
    }

    /// Set a random seed for deterministic results.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }

    /// Run Leiden algorithm on a graph and return community assignments.
    ///
    /// Returns a map from node ID to community ID.
    ///
    /// # Algorithm
    ///
    /// The Leiden algorithm consists of three phases:
    /// 1. **Local moving**: Move nodes to neighboring communities that improve modularity
    /// 2. **Refinement**: Split communities to ensure well-connected sub-communities
    /// 3. **Aggregation**: Build a new graph where nodes are communities, repeat until convergence
    ///
    /// This implementation includes phases 1 and 2. Phase 3 (aggregation) is handled
    /// by the hierarchical wrapper.
    pub fn cluster(&self, graph: &GraphDocument) -> Result<HashMap<String, usize>, String> {
        if graph.nodes.is_empty() {
            return Ok(HashMap::new());
        }

        // Convert GraphDocument to petgraph::Graph
        let (petgraph, node_id_map) = graph_to_petgraph(graph)?;

        // Phase 1: Local moving - initialize each node as its own community
        let mut communities: HashMap<NodeIndex, usize> = HashMap::new();
        for (idx, node_idx) in node_id_map.values().enumerate() {
            communities.insert(*node_idx, idx);
        }

        // Local moving phase: iteratively move nodes to improve modularity
        let mut improved = true;
        let mut iterations = 0;
        let max_iterations = 100;

        while improved && iterations < max_iterations {
            improved = false;
            iterations += 1;

            // Visit nodes in random order (deterministic if seed is set)
            let mut node_indices: Vec<NodeIndex> = petgraph.node_indices().collect();
            if let Some(seed) = self.random_seed {
                // Simple deterministic shuffle using seed
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};
                let mut hasher = DefaultHasher::new();
                (seed + iterations as u64).hash(&mut hasher);
                let mut rng_state = hasher.finish();
                for i in 0..node_indices.len() {
                    rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                    let j = (rng_state as usize) % (node_indices.len() - i) + i;
                    node_indices.swap(i, j);
                }
            }

            // Try moving each node to a neighboring community
            for node_idx in node_indices {
                let current_community = *communities.get(&node_idx).unwrap_or(&0);
                let mut best_community = current_community;
                let mut best_delta_modularity = 0.0;

                // Calculate current modularity contribution
                let current_modularity =
                    modularity_with_resolution(&petgraph, &communities, self.resolution);

                // Check all neighboring communities
                for neighbor in petgraph.neighbors(node_idx) {
                    let neighbor_community = *communities.get(&neighbor).unwrap_or(&0);
                    if neighbor_community != current_community {
                        // Try moving node to neighbor's community
                        let mut test_communities = communities.clone();
                        test_communities.insert(node_idx, neighbor_community);
                        let test_modularity = modularity_with_resolution(
                            &petgraph,
                            &test_communities,
                            self.resolution,
                        );

                        let delta = test_modularity - current_modularity;
                        if delta > best_delta_modularity {
                            best_delta_modularity = delta;
                            best_community = neighbor_community;
                            improved = true;
                        }
                    }
                }

                // Also consider creating a new singleton community
                let new_community_id = communities.values().max().copied().unwrap_or(0) + 1;
                let mut test_communities = communities.clone();
                test_communities.insert(node_idx, new_community_id);
                let test_modularity =
                    modularity_with_resolution(&petgraph, &test_communities, self.resolution);
                let delta = test_modularity - current_modularity;
                if delta > best_delta_modularity {
                    best_community = new_community_id;
                    improved = true;
                }

                if best_community != current_community {
                    communities.insert(node_idx, best_community);
                }
            }
        }

        // Phase 2: Refinement - ensure communities are well-connected
        // (Simplified: in full implementation, this would split disconnected communities)
        let refined_communities = refine_communities(&petgraph, &communities)?;

        // Convert back to node ID -> community ID mapping
        let mut result = HashMap::new();
        for (node_id, node_idx) in node_id_map {
            if let Some(community_id) = refined_communities.get(&node_idx) {
                result.insert(node_id, *community_id);
            }
        }

        Ok(result)
    }
}

impl Default for Leiden {
    fn default() -> Self {
        Self::new()
    }
}

/// Modularity calculation for community detection.
///
/// Modularity measures the quality of a community partition:
/// Q = (1/2m) * Σ[A_ij - γ * (k_i * k_j / 2m)] * δ(c_i, c_j)
///
/// Where:
/// - m = total number of edges (sum of edge weights)
/// - A_ij = adjacency matrix (edge weight)
/// - k_i = degree of node i (sum of edge weights)
/// - c_i = community of node i
/// - γ = resolution parameter (higher = more communities)
/// - δ = Kronecker delta (1 if same community, 0 otherwise)
pub fn modularity_with_resolution<N>(
    graph: &Graph<N, f32, Undirected>,
    communities: &HashMap<NodeIndex, usize>,
    resolution: f32,
) -> f32 {
    // Calculate total edge weight (m)
    let mut m = 0.0;
    for edge in graph.edge_indices() {
        m += graph.edge_weight(edge).copied().unwrap_or(1.0);
    }

    if m == 0.0 {
        return 0.0;
    }

    // Calculate node degrees (sum of edge weights)
    let mut degrees: HashMap<NodeIndex, f32> = HashMap::new();
    for node_idx in graph.node_indices() {
        let mut degree = 0.0;
        for edge in graph.edges(node_idx) {
            degree += edge.weight();
        }
        degrees.insert(node_idx, degree);
    }

    // Calculate modularity
    let mut q = 0.0;
    for edge in graph.edge_indices() {
        // For undirected graphs, edge_endpoints always returns Some
        // But handle gracefully for safety
        let (a, b) = graph
            .edge_endpoints(edge)
            .expect("edge_endpoints should always return Some for undirected graph");
        let weight = graph.edge_weight(edge).copied().unwrap_or(1.0);

        if communities.get(&a) == communities.get(&b) {
            let deg_a = degrees.get(&a).copied().unwrap_or(0.0);
            let deg_b = degrees.get(&b).copied().unwrap_or(0.0);
            q += weight - resolution * (deg_a * deg_b) / (2.0 * m);
        }
    }

    q / (2.0 * m)
}

/// Legacy modularity function (without resolution parameter, defaults to 1.0).
pub fn modularity<N>(
    graph: &Graph<N, f32, Undirected>,
    communities: &HashMap<NodeIndex, usize>,
) -> f32 {
    modularity_with_resolution(graph, communities, 1.0)
}

/// Refine communities to ensure they are well-connected.
///
/// This is a simplified refinement phase. In the full Leiden algorithm,
/// this would split communities that are not well-connected into
/// smaller, well-connected sub-communities.
fn refine_communities(
    _graph: &Graph<String, f32, Undirected>,
    communities: &HashMap<NodeIndex, usize>,
) -> Result<HashMap<NodeIndex, usize>, String> {
    // For now, just return the communities as-is
    // Full implementation would:
    // 1. Identify communities that are not well-connected
    // 2. Split them into connected components
    // 3. Assign new community IDs to split components
    Ok(communities.clone())
}

/// Convert a GraphDocument to a petgraph::Graph.
///
/// Returns the graph and a mapping from node ID (String) to NodeIndex.
fn graph_to_petgraph(
    graph_doc: &GraphDocument,
) -> Result<(Graph<String, f32, Undirected>, HashMap<String, NodeIndex>), String> {
    let mut petgraph = Graph::<String, f32, Undirected>::new_undirected();
    let mut node_id_map = HashMap::new();

    // Add all nodes
    for node in &graph_doc.nodes {
        let node_idx = petgraph.add_node(node.id.clone());
        node_id_map.insert(node.id.clone(), node_idx);
    }

    // Add all edges
    for edge in &graph_doc.edges {
        let source_idx = node_id_map
            .get(&edge.source)
            .ok_or_else(|| format!("Source node '{}' not found", edge.source))?;
        let target_idx = node_id_map
            .get(&edge.target)
            .ok_or_else(|| format!("Target node '{}' not found", edge.target))?;

        // Use confidence as edge weight
        let weight = edge.confidence as f32;
        petgraph.add_edge(*source_idx, *target_idx, weight);
    }

    Ok((petgraph, node_id_map))
}
