// Single integration test binary to avoid per-file link overhead.
// Previously 14 separate binaries, each taking ~50s to link.

mod backend_properties;
mod enrich_findings;
mod fixture_regression;
mod identity_tests;
mod invariants;
mod property_tests;
mod qa_regression;
mod span_integrity;
mod unicode_stress;

#[cfg(feature = "analysis")]
mod rag_coref_e2e;

#[cfg(feature = "analysis")]
mod rag_e2e;

#[cfg(feature = "onnx")]
mod model_backends;

#[cfg(all(feature = "onnx", feature = "analysis"))]
mod model_integration;

#[cfg(feature = "llm")]
mod llm_integration;
