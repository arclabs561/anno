//! # anno-coalesce
//!
//! Inter-document entity coalescing and entity linking.
//!
//! This crate provides algorithms for clustering entities across multiple documents
//! and linking them to knowledge bases (Wikidata, DBpedia).
//!
//! **Extract. Coalesce. Stratify.**
//!
//! # Example
//!
//! ```
//! use anno_coalesce::Resolver;
//! use anno_core::Corpus;
//!
//! let mut resolver = Resolver::new();
//! let mut corpus = Corpus::new();
//! // ... add documents to corpus ...
//!
//! // Coalesce entities across documents
//! let identity_ids = resolver.resolve_inter_doc_coref(&mut corpus, None, None);
//! ```

#![warn(missing_docs)]

pub mod resolver;

pub use resolver::{embedding_similarity, string_similarity, Resolver};
