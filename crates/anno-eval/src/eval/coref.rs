//! Coreference resolution data structures.
//!
//! This module re-exports the canonical coref types from `anno::core`.
//! See [`anno::core::coref`] for the full documentation.

// Re-export all coref types from `anno::core::coref`.
pub use anno::core::coref::*;

/// Place document-local mention and head spans in corpus coordinates before pooled scoring.
pub(super) fn offset_chains(mut chains: Vec<CorefChain>, base: usize) -> Vec<CorefChain> {
    if base == 0 {
        return chains;
    }
    for chain in &mut chains {
        for m in &mut chain.mentions {
            m.start = m.start.saturating_add(base);
            m.end = m.end.saturating_add(base);
            if let Some(hs) = m.head_start.as_mut() {
                *hs = hs.saturating_add(base);
            }
            if let Some(he) = m.head_end.as_mut() {
                *he = he.saturating_add(base);
            }
        }
    }
    chains
}
