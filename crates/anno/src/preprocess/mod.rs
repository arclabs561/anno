//! Preprocessing utilities for text normalization and morphological analysis.
//!
//! This module provides preprocessing support for:
//! - Morphologically complex languages (polysynthetic, agglutinative)
//! - Text normalization and cleaning
//! - Script-specific handling (Cherokee syllabary, etc.)
//! - **Parenthetical analysis** - aliases, abbreviations, temporal bounds
//! - **Reference resolution** - URLs, citations, cross-references
//!
//! # Morphological Preprocessing
//!
//! For polysynthetic languages like Cherokee, Navajo, and Mohawk, standard
//! word-level tokenization fails because a single word can encode an entire
//! sentence. The `morphology` module provides segmentation strategies and
//! the `MorphologicalAnalyzer` trait for integrating external analyzers.
//!
//! # Parenthetical Analysis
//!
//! The `parenthetical` module extracts valuable entity information from
//! parenthetical text:
//!
//! ```rust
//! use anno::preprocess::parenthetical::{ParentheticalExtractor, ParentheticalType};
//!
//! let extractor = ParentheticalExtractor::new();
//! let results = extractor.extract("The World Health Organization (WHO) announced guidelines.");
//!
//! assert_eq!(results[0].content, "WHO");
//! assert_eq!(results[0].parenthetical_type, ParentheticalType::Abbreviation);
//! ```
//!
//! # Reference Resolution
//!
//! The [`reference`] module detects URLs, citations, and cross-references
//! that can be resolved to additional entity information:
//!
//! ```rust,ignore
//! use anno::preprocess::reference::{ReferenceExtractor, ReferenceType};
//!
//! let extractor = ReferenceExtractor::new();
//! let refs = extractor.extract("See https://en.wikipedia.org/wiki/Einstein");
//!
//! assert_eq!(refs[0].reference_type, ReferenceType::WikipediaUrl);
//! ```
//!
//! # Integration with Coalesce and Tier
//!
//! These modules integrate with the entity resolution pipeline:
//!
//! - **Coalesce**: Parenthetical aliases help link "WHO" ↔ "World Health Organization"
//! - **Tier**: Reference graphs create hierarchical entity relationships
//!
//! # Example
//!
//! ```rust
//! use anno::preprocess::morphology::{MorphologicalPreprocessor, SegmentationStrategy};
//!
//! // For Quechua with hyphenated morpheme boundaries
//! let preprocessor = MorphologicalPreprocessor::new()
//!     .with_strategy(SegmentationStrategy::RuleBased {
//!         boundary_chars: vec!['-', '='],
//!     });
//!
//! let result = preprocessor.segment("wasi-kuna-y-ki").unwrap();
//! assert_eq!(result.morphemes.len(), 4); // wasi, kuna, y, ki
//! ```

pub mod apposition;
pub mod morphology;
pub mod parenthetical;
pub mod reference;

// Re-export commonly used types
pub use morphology::{
    cherokee_syllable_inventory, navajo_prefix_inventory, quechua_boundary_chars, Morpheme,
    MorphemeType, MorphologicalAnalyzer, MorphologicalPreprocessor, ProdropConfig,
    SegmentationResult, SegmentationStrategy,
};

pub use parenthetical::{
    extract_aliases, AliasPair, Parenthetical, ParentheticalExtractor, ParentheticalType,
};

pub use reference::{
    ExtractedEntity, Reference, ReferenceExtractor, ReferenceGraph, ReferenceType,
    ResolvedReference,
};

pub use apposition::{extract_all_aliases, Apposition, AppositionExtractor, AppositionType};
