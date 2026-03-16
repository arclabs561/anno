//! Command implementations for anno CLI
//!
//! Each command has its own module/file for better organization.

#![allow(unused_imports)]

pub mod analyze;
pub mod batch;
pub mod cache;
pub mod compare;
pub mod config;
pub mod context;
pub mod dataset;
pub mod debug;
pub mod domain;
pub mod enhance;
pub mod eval;
pub mod explain;
pub mod export;
pub mod extract;
pub mod history;
pub mod import;
pub mod info;
pub mod models;
#[cfg(feature = "eval")]
pub mod muxer;
pub mod pipeline;
pub mod privacy;
pub mod query;
pub mod singleton;
pub mod validate;
pub mod watch;

// Heavy/optional commands.
#[cfg(feature = "eval")]
pub mod benchmark;
#[cfg(feature = "eval")]
pub mod crossdoc;

// Re-export argument types for parser
pub use analyze::AnalyzeArgs;
pub use batch::BatchArgs;
pub use cache::{CacheAction, CacheArgs};
pub use compare::CompareArgs;
pub use config::{ConfigAction, ConfigArgs};
pub use context::ContextArgs;
pub use dataset::DatasetArgs;
pub use debug::DebugArgs;
pub use domain::DomainArgs;
pub use enhance::EnhanceArgs;
pub use eval::EvalArgs;
pub use explain::ExplainArgs;
pub use export::{ExportArgs, ExportFormat};
pub use extract::ExtractArgs;
pub use history::HistoryArgs;
pub use import::ImportArgs;
pub use models::ModelsArgs;
pub use pipeline::PipelineArgs;
pub use privacy::PrivacyArgs;
pub use query::QueryArgs;
pub use singleton::SingletonArgs;
pub use validate::ValidateArgs;
pub use watch::WatchArgs;

#[cfg(feature = "eval")]
pub use benchmark::BenchmarkArgs;
#[cfg(feature = "eval")]
pub use crossdoc::CrossDocArgs;
#[cfg(feature = "eval")]
pub use muxer::MuxerArgs;
