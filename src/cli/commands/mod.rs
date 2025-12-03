//! Command implementations for anno CLI
//!
//! Each command has its own module/file for better organization.

pub mod extract;
// TODO: Extract remaining commands
// pub mod debug;
// pub mod eval;
// pub mod validate;
// pub mod analyze;
// pub mod dataset;
// pub mod benchmark;
// pub mod info;
// pub mod models;
// pub mod crossdoc;
// pub mod enhance;
// pub mod pipeline;
// pub mod query;
// pub mod compare;
// pub mod cache;
// pub mod config;
// pub mod batch;

// Re-export argument types for parser
pub use extract::ExtractArgs;
// TODO: Re-export remaining command args as they are extracted
