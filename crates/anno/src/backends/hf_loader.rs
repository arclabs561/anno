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
    no_downloads_from(std::env::var("ANNO_NO_DOWNLOADS").ok().as_deref())
}

fn no_downloads_from(value: Option<&str>) -> bool {
    match value {
        Some(v) => matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "y" | "on"
        ),
        None => false,
    }
}

/// Initialize the HuggingFace API client, loading `.env` and using `HF_TOKEN` if available.
///
/// The client and offline cache both honor `HF_HOME` through `hf-hub`'s
/// environment-aware builders.
///
/// This replaces the duplicated pattern:
/// ```rust,ignore
/// crate::env::load_dotenv();
/// let builder = ApiBuilder::from_env();
/// let builder = if let Some(token) = crate::env::hf_token() {
///     builder.with_token(Some(token))
/// } else {
///     builder
/// };
/// let api = builder.build()?;
/// ```
pub fn hf_api() -> Result<hf_hub::api::sync::Api> {
    use hf_hub::api::sync::ApiBuilder;

    crate::env::load_dotenv();

    let builder = ApiBuilder::from_env();
    let builder = if let Some(token) = crate::env::hf_token() {
        builder.with_token(Some(token))
    } else {
        builder
    };
    builder
        .build()
        .map_err(|e| Error::Retrieval(format!("HuggingFace API init: {}", e)))
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
            if let Some(path) = hf_hub::Cache::from_env()
                .repo(cache_repo_identity(repo)?)
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

/// Recover repository type and revision from hf-hub's public URL contract.
/// Offline lookup uses the environment's cache, as does [`hf_api`]. Callers that
/// construct an API with a custom cache directory must use local-path loading
/// or set `HF_HOME` consistently; ApiRepo does not expose its cache directory.
fn cache_repo_identity(repo: &hf_hub::api::sync::ApiRepo) -> Result<hf_hub::Repo> {
    let url = repo.url("");
    let path = url
        .split_once("://")
        .and_then(|(_, rest)| rest.split_once('/'))
        .map(|(_, path)| path);
    let identity = path.and_then(|path| path.rsplit_once("/resolve/"));
    let Some((path, revision)) = identity else {
        return Err(Error::Retrieval(
            "Cannot identify offline HuggingFace repository".into(),
        ));
    };
    let (kind, id) = if let Some(id) = path.strip_prefix("datasets/") {
        (hf_hub::RepoType::Dataset, id)
    } else if let Some(id) = path.strip_prefix("spaces/") {
        (hf_hub::RepoType::Space, id)
    } else {
        (hf_hub::RepoType::Model, path)
    };
    let revision = revision.trim_end_matches('/').replace("%2F", "/");
    if id.is_empty() || revision.is_empty() {
        return Err(Error::Retrieval(
            "Cannot identify offline HuggingFace revision".into(),
        ));
    }
    Ok(hf_hub::Repo::with_revision(id.into(), kind, revision))
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
#[cfg(any(feature = "onnx", test))]
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
        if let Ok(path) = download_model_file(repo, &quantized_candidates) {
            log::info!("[hf_loader] Using quantized model");
            return Ok((path, true));
        }
    }

    // Fall back to FP32
    let path = download_model_file(repo, &["onnx/model.onnx", "model.onnx"])?;
    if prefer_quantized {
        log::info!("[hf_loader] Using FP32 model (quantized not available)");
    }
    Ok((path, false))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offline_quantized_lookup_preserves_revision_without_network() {
        // Isolate environment flags from tests running concurrently in this process.
        const CHILD: &str = "ANNO_TEST_OFFLINE_HF_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let home = tempfile::tempdir().unwrap();
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "backends::hf_loader::tests::offline_quantized_lookup_preserves_revision_without_network", "--nocapture"])
                .env(CHILD, "1").env("ANNO_NO_DOWNLOADS", "1")
                .env("HF_HOME", home.path()).status().unwrap();
            assert!(status.success());
            return;
        }
        // A live local listener makes any unintended network access observable.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let api = hf_hub::api::sync::ApiBuilder::from_env()
            .with_token(None)
            .with_progress(false)
            .with_retries(0)
            .with_endpoint(format!("http://{}", listener.local_addr().unwrap()))
            .build()
            .unwrap();
        let identity = hf_hub::Repo::with_revision(
            "anno-tests/offline".into(),
            hf_hub::RepoType::Model,
            "refs/pr/7".into(),
        );
        let repo = api.repo(identity.clone());
        // No server is needed: the guarded path must return before any connection.
        // The child timeout is enforced by nextest's ordinary test timeout.
        let error = download_onnx_model(&repo, true).unwrap_err();
        assert!(error.to_string().contains("ANNO_NO_DOWNLOADS"));
        assert!(listener.accept().is_err());
        let cache = hf_hub::Cache::from_env();
        let root = cache.path().join("models--anno-tests--offline");
        std::fs::create_dir_all(root.join("refs/refs/pr")).unwrap();
        std::fs::create_dir_all(root.join("snapshots/pinned")).unwrap();
        std::fs::write(root.join("refs/refs/pr/7"), "pinned").unwrap();
        let model = root.join("snapshots/pinned/model.onnx");
        std::fs::write(&model, b"cached model fixture").unwrap();
        assert_eq!(download_onnx_model(&repo, true).unwrap(), (model, false));
        assert!(listener.accept().is_err());
        assert_eq!(
            cache_repo_identity(&repo).unwrap().revision(),
            identity.revision()
        );
    }

    #[test]
    fn no_downloads_accepts_only_truthy_values() {
        for value in ["1", "true", "YES", " y ", "On"] {
            assert!(no_downloads_from(Some(value)), "{value}");
        }
        for value in ["0", "false", "", "maybe"] {
            assert!(!no_downloads_from(Some(value)), "{value}");
        }
        assert!(!no_downloads_from(None));
    }
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
#[cfg(feature = "onnx")]
pub fn load_tokenizer(path: &std::path::Path) -> Result<tokenizers::Tokenizer> {
    tokenizers::Tokenizer::from_file(path)
        .map_err(|e| Error::Retrieval(format!("Tokenizer load: {}", e)))
}
