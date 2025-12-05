//! # anno-strata
//!
//! Hierarchical clustering for graph construction.
//!
//! This crate provides algorithms for building hierarchical community structures
//! from graph documents using Leiden, RAPTOR, and other methods.
//!
//! **Extract. Coalesce. Stratify.**
//!
//! # Example
//!
//! ```rust,ignore
//! use anno_strata::HierarchicalLeiden;
//! use anno_core::GraphDocument;
//!
//! let clusterer = HierarchicalLeiden::new()
//!     .with_resolution(1.0)
//!     .with_levels(3);
//!
//! let graph = GraphDocument::new();
//! let clustered = clusterer.cluster(&graph)?;
//! ```

#![warn(missing_docs)]

pub mod leiden;

use anno_core::GraphDocument;

/// Hierarchical Leiden clustering.
///
/// This implements the Leiden algorithm for hierarchical community detection
/// in knowledge graphs, revealing strata of abstraction.
///
/// The algorithm is applied recursively at multiple resolutions to build
/// a hierarchical dendrogram of communities.
#[derive(Debug, Clone)]
pub struct HierarchicalLeiden {
    resolution: f32,
    levels: usize,
}

impl HierarchicalLeiden {
    /// Create a new hierarchical Leiden clusterer with default settings.
    pub fn new() -> Self {
        Self {
            resolution: 1.0,
            levels: 3,
        }
    }

    /// Set the resolution parameter for clustering.
    ///
    /// Higher resolution values lead to more, smaller communities.
    /// Lower values lead to fewer, larger communities.
    pub fn with_resolution(mut self, resolution: f32) -> Self {
        self.resolution = resolution;
        self
    }

    /// Set the number of hierarchical levels to compute.
    ///
    /// Each level represents a different granularity of community structure.
    pub fn with_levels(mut self, levels: usize) -> Self {
        self.levels = levels;
        self
    }

    /// Cluster a graph document using hierarchical Leiden algorithm.
    ///
    /// This implementation:
    /// 1. Builds a graph from the GraphDocument
    /// 2. Applies Leiden algorithm at multiple resolutions
    /// 3. Creates hierarchical community structure
    /// 4. Returns a new GraphDocument with community annotations
    ///
    /// # Errors
    ///
    /// Returns an error if the graph is empty or invalid.
    pub fn cluster(&self, graph: &GraphDocument) -> Result<GraphDocument, String> {
        use leiden::Leiden;

        // Apply Leiden algorithm at multiple resolutions for hierarchical structure
        let mut result = graph.clone();
        let mut all_communities = Vec::new();

        for level in 0..self.levels {
            let resolution = self.resolution * (2.0_f32.powi(level as i32));
            let leiden = Leiden::new().with_resolution(resolution);
            let communities = leiden.cluster(graph)?;
            all_communities.push((level, communities));
        }

        // Annotate nodes with community assignments at each level
        for (level, communities) in all_communities {
            for node in &mut result.nodes {
                if let Some(community_id) = communities.get(&node.id) {
                    node.properties
                        .insert(format!("level_{}_community", level), (*community_id).into());
                }
            }
        }

        Ok(result)
    }
}

impl Default for HierarchicalLeiden {
    fn default() -> Self {
        Self::new()
    }
}
