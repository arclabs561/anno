//! Command implementations for anno CLI
//!
//! Each command has its own module/file for better organization.

pub mod analyze;
pub mod batch;
pub mod benchmark;
pub mod cache;
pub mod compare;
pub mod config;
pub mod crossdoc;
pub mod dataset;
pub mod debug;
pub mod enhance;
pub mod eval;
pub mod extract;
pub mod info;
pub mod models;
pub mod pipeline;
pub mod query;
pub mod validate;

// Re-export argument types for parser
pub use analyze::AnalyzeArgs;
pub use batch::BatchArgs;
pub use benchmark::BenchmarkArgs;
pub use cache::{CacheAction, CacheArgs};
pub use compare::CompareArgs;
pub use config::{ConfigAction, ConfigArgs};
pub use crossdoc::CrossDocArgs;
pub use dataset::DatasetArgs;
pub use debug::DebugArgs;
pub use enhance::EnhanceArgs;
pub use eval::EvalArgs;
pub use extract::ExtractArgs;
pub use models::ModelsArgs;
pub use pipeline::PipelineArgs;
pub use query::QueryArgs;
pub use validate::ValidateArgs;
