//! Shared HuggingFace model loading utilities.
//!
//! Centralizes the duplicated pattern of:
//! 1. Initializing the HF API (with optional token from `.env`)
//! 2. Downloading model files from a HuggingFace repo
//! 3. Creating ONNX Runtime sessions with standard configuration
//!
//! # Usage
//!
//! ```rust,ignore
//! use anno::backends::hf_loader::{hf_api, download_model_file, create_onnx_session, OnnxSessionConfig};
//!
//! let api = hf_api()?;
//! let repo = api.model("protectai/bert-base-NER-onnx".to_string());
//! let model_path = download_model_file(&repo, &["onnx/model.onnx", "model.onnx"])?;
//! let tokenizer_path = download_model_file(&repo, &["tokenizer.json"])?;
//! let session = create_onnx_session(&model_path, OnnxSessionConfig::default())?;
//! ```

use crate::{Error, Result};

/// Returns `true` when `ANNO_NO_DOWNLOADS` is set to a truthy value.
///
/// The flag blocks *new* network fetches; cached models still load via
/// [`download_model_file`]. Backends constructed from local paths bypass
/// this layer entirely.
pub fn no_downloads() -> bool {
    match std::env::var("ANNO_NO_DOWNLOADS") {
        Ok(v) => matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "y" | "on"
        ),
        Err(_) => false,
    }
}

/// Initialize the HuggingFace API client, loading `.env` and using `HF_TOKEN` if available.
///
/// This replaces the duplicated pattern:
/// ```rust,ignore
/// crate::env::load_dotenv();
/// let api = if let Some(token) = crate::env::hf_token() {
///     ApiBuilder::new().with_token(Some(token)).build()?
/// } else {
///     Api::new()?
/// };
/// ```
pub fn hf_api() -> Result<hf_hub::api::sync::Api> {
    use hf_hub::api::sync::{Api, ApiBuilder};

    crate::env::load_dotenv();

    if let Some(token) = crate::env::hf_token() {
        ApiBuilder::new()
            .with_token(Some(token))
            .build()
            .map_err(|e| Error::Retrieval(format!("HuggingFace API init with token: {}", e)))
    } else {
        Api::new().map_err(|e| Error::Retrieval(format!("HuggingFace API init: {}", e)))
    }
}

/// Download a file from a HuggingFace repo, trying multiple candidate paths in order.
///
/// Returns the local path to the downloaded file. Tries each candidate path in order
/// and returns the first successful download.
///
/// # Arguments
///
/// * `repo` - HuggingFace repo handle from `api.model()`
/// * `candidates` - File paths to try in order (e.g., `&["onnx/model.onnx", "model.onnx"]`)
///
/// # Errors
///
/// Returns `Error::Retrieval` if none of the candidates can be downloaded.
pub fn download_model_file(
    repo: &hf_hub::api::sync::ApiRepo,
    candidates: &[&str],
) -> Result<std::path::PathBuf> {
    if candidates.is_empty() {
        return Err(Error::Retrieval(
            "download_model_file: candidates must not be empty".to_string(),
        ));
    }

    if no_downloads() {
        for candidate in candidates {
            if let Some(path) = hf_hub::Cache::default()
                .repo(hf_hub::Repo::model(repo_id_of(repo)))
                .get(candidate)
            {
                return Ok(path);
            }
        }
        return Err(Error::Retrieval(format!(
            "ANNO_NO_DOWNLOADS is set and none of [{}] are present in the \
             HuggingFace cache. Pre-fetch the model (unset ANNO_NO_DOWNLOADS \
             and re-run once), or skip this backend.",
            candidates.join(", "),
        )));
    }

    let mut last_err = None;
    for candidate in candidates {
        match repo.get(candidate) {
            Ok(path) => return Ok(path),
            Err(e) => last_err = Some(e),
        }
    }

    Err(Error::Retrieval(format!(
        "Failed to download any of [{}]: {}",
        candidates.join(", "),
        last_err
            .map(|e| e.to_string())
            .unwrap_or_else(|| "unknown".to_string())
    )))
}

/// Best-effort repo-id extraction from an `ApiRepo` handle.
///
/// `hf_hub::api::sync::ApiRepo` doesn't expose its repo id directly, so we
/// re-derive it from its Debug output. Used only by the no-downloads path
/// to query the cache for the same repo.
fn repo_id_of(repo: &hf_hub::api::sync::ApiRepo) -> String {
    // Debug looks like: `ApiRepo { api: ..., repo: Repo { repo_id: "owner/name", ... } }`
    let dbg = format!("{:?}", repo);
    if let Some(start) = dbg.find("repo_id: \"") {
        let rest = &dbg[start + "repo_id: \"".len()..];
        if let Some(end) = rest.find('"') {
            return rest[..end].to_string();
        }
    }
    // Fallback: return empty so cache lookup misses, download_model_file errors.
    String::new()
}

/// Try to download a quantized ONNX model, falling back to FP32.
///
/// Tries quantized variants first (if `prefer_quantized` is true), then falls back
/// to the standard model path. Returns `(local_path, is_quantized)`.
///
/// # Arguments
///
/// * `repo` - HuggingFace repo handle
/// * `prefer_quantized` - Whether to try quantized variants first
pub fn download_onnx_model(
    repo: &hf_hub::api::sync::ApiRepo,
    prefer_quantized: bool,
) -> Result<(std::path::PathBuf, bool)> {
    if prefer_quantized {
        // Try quantized variants first
        let quantized_candidates = [
            "onnx/model_quantized.onnx",
            "model_quantized.onnx",
            "onnx/model_int8.onnx",
            "model_int8.onnx",
        ];
        for candidate in &quantized_candidates {
            if let Ok(path) = repo.get(candidate) {
                log::info!("[hf_loader] Using quantized model: {}", candidate);
                return Ok((path, true));
            }
        }
    }

    // Fall back to FP32
    let path = download_model_file(repo, &["onnx/model.onnx", "model.onnx"])?;
    if prefer_quantized {
        log::info!("[hf_loader] Using FP32 model (quantized not available)");
    }
    Ok((path, false))
}

/// Configuration for creating an ONNX Runtime session.
///
/// Marked `#[non_exhaustive]` to permit additional execution-provider
/// preferences in future versions without breaking struct-literal callers.
/// Construct via `OnnxSessionConfig::default()` and override the fields you
/// care about with `..Default::default()`.
#[cfg(feature = "onnx")]
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct OnnxSessionConfig {
    /// ONNX graph optimization level (1-3, default 3).
    pub optimization_level: u8,
    /// Number of intra-op threads (0 = auto/default).
    pub num_threads: usize,
    /// Whether to use CPU execution provider explicitly.
    pub use_cpu_provider: bool,
    /// Prefer Apple CoreML (Apple Neural Engine + GPU) when available.
    /// Effective only when the `onnx-coreml` feature is enabled at build
    /// time AND the host is macOS. CPU is added as a fallback so the
    /// session still loads if CoreML cannot handle the graph.
    ///
    /// Without the feature flag the field exists for API stability but
    /// the value is ignored, hence the `#[allow(dead_code)]`.
    #[cfg_attr(not(feature = "onnx-coreml"), allow(dead_code))]
    pub prefer_coreml: bool,
    /// Prefer NVIDIA CUDA when available.
    /// Effective only when the `onnx-cuda` feature is enabled at build
    /// time AND CUDA 12.x is present at link/runtime. CPU is added as a
    /// fallback so the session still loads if CUDA cannot initialise.
    ///
    /// **Silent CPU fallback is a known ort failure mode** when `cudart.so`
    /// is missing or the GPU is otherwise unavailable -- compile success
    /// does not prove runtime acceleration. Use `examples/onnx_gpu_smoke.rs`
    /// (or an equivalent throughput check) on a real GPU host to confirm.
    ///
    /// Without the feature flag the field exists for API stability but
    /// the value is ignored, hence the `#[allow(dead_code)]`.
    #[cfg_attr(not(feature = "onnx-cuda"), allow(dead_code))]
    pub prefer_cuda: bool,
}

#[cfg(feature = "onnx")]
impl Default for OnnxSessionConfig {
    fn default() -> Self {
        Self {
            optimization_level: 3,
            num_threads: 0,
            use_cpu_provider: true,
            prefer_coreml: false,
            prefer_cuda: false,
        }
    }
}

/// Create an ONNX Runtime session from a model file with the given configuration.
///
/// This replaces the duplicated pattern:
/// ```rust,ignore
/// Session::builder()?
///     .with_optimization_level(GraphOptimizationLevel::Level3)?
///     .with_execution_providers([CPUExecutionProvider::default().build()])?
///     .commit_from_file(&model_path)?
/// ```
#[cfg(feature = "onnx")]
pub fn create_onnx_session(
    model_path: &std::path::Path,
    config: OnnxSessionConfig,
) -> Result<ort::session::Session> {
    use ort::session::builder::GraphOptimizationLevel;
    use ort::session::Session;

    let opt_level = match config.optimization_level {
        1 => GraphOptimizationLevel::Level1,
        2 => GraphOptimizationLevel::Level2,
        _ => GraphOptimizationLevel::Level3,
    };

    let mut builder = Session::builder()
        .map_err(|e| Error::Retrieval(format!("ONNX session builder: {}", e)))?
        .with_optimization_level(opt_level)
        .map_err(|e| Error::Retrieval(format!("ONNX optimization level: {}", e)))?;

    // Build execution-provider list in priority order. ort tries each in
    // turn and falls back to the next if one can't load the graph. CPU is
    // always last so a session never fails to start because of an
    // accelerator-specific quirk.
    let mut providers: Vec<ort::execution_providers::ExecutionProviderDispatch> = Vec::new();
    #[cfg(feature = "onnx-cuda")]
    if config.prefer_cuda {
        use ort::execution_providers::CUDAExecutionProvider;
        providers.push(CUDAExecutionProvider::default().build());
    }
    #[cfg(feature = "onnx-coreml")]
    if config.prefer_coreml {
        use ort::execution_providers::CoreMLExecutionProvider;
        providers.push(CoreMLExecutionProvider::default().build());
    }
    if config.use_cpu_provider {
        use ort::execution_providers::CPUExecutionProvider;
        providers.push(CPUExecutionProvider::default().build());
    }
    if !providers.is_empty() {
        builder = builder
            .with_execution_providers(providers)
            .map_err(|e| Error::Retrieval(format!("ONNX execution providers: {}", e)))?;
    }

    if config.num_threads > 0 {
        builder = builder
            .with_intra_threads(config.num_threads)
            .map_err(|e| Error::Retrieval(format!("ONNX thread config: {}", e)))?;
    }

    builder
        .commit_from_file(model_path)
        .map_err(|e| Error::Retrieval(format!("ONNX model load: {}", e)))
}

/// Load a HuggingFace tokenizer from a file path.
pub fn load_tokenizer(path: &std::path::Path) -> Result<tokenizers::Tokenizer> {
    tokenizers::Tokenizer::from_file(path)
        .map_err(|e| Error::Retrieval(format!("Tokenizer load: {}", e)))
}
