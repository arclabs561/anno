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
        let identity = cache_repo_identity(repo)?;
        let cache = hf_hub::Cache::from_env().repo(identity.clone());
        for candidate in candidates {
            if let Some(path) = cached_model_file(&cache, identity.revision(), candidate) {
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

/// Python's Hub client stores immutable revisions directly as snapshots, without a
/// `refs/<commit>` file. hf-hub 0.5's `get` only resolves refs, so check the exact
/// snapshot for a full commit ID before resolving named revisions.
fn cached_model_file(
    cache: &hf_hub::CacheRepo,
    revision: &str,
    filename: &str,
) -> Option<std::path::PathBuf> {
    if revision.len() == 40 && revision.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        let path = cache.pointer_path(revision).join(filename);
        return path.is_file().then_some(path);
    }
    cache.get(filename)
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
    fn immutable_snapshot_loads_without_a_ref_file() {
        let home = tempfile::tempdir().unwrap();
        let revision = "4241d7c66b648e618c89c150bf4cf418d2f83159";
        let cache =
            hf_hub::Cache::new(home.path().to_path_buf()).repo(hf_hub::Repo::with_revision(
                "anno-tests/pinned".into(),
                hf_hub::RepoType::Model,
                revision.into(),
            ));
        let path = cache.pointer_path(revision).join("tokenizer.json");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, "{}").unwrap();
        assert!(cache.get("tokenizer.json").is_none());
        assert_eq!(
            cached_model_file(&cache, revision, "tokenizer.json"),
            Some(path)
        );
        assert!(cached_model_file(&cache, revision, "missing.json").is_none());
        assert!(cached_model_file(
            &cache,
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "tokenizer.json"
        )
        .is_none());
    }

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
    /// time AND the host is macOS. If requested, CoreML must register or
    /// session construction returns an error. CPU can execute graph nodes
    /// CoreML does not support after registration succeeds.
    ///
    /// Without the feature flag, requesting it returns an error that names
    /// the needed Cargo feature.
    #[cfg_attr(not(feature = "onnx-coreml"), allow(dead_code))]
    pub prefer_coreml: bool,
    /// Prefer NVIDIA CUDA when available.
    /// Effective only when the `onnx-cuda` feature is enabled at build
    /// time AND CUDA 12.x is present at link/runtime. If requested, CUDA
    /// must register or session construction returns an error.
    ///
    /// This rejects ONNX Runtime's silent fallback when CUDA registration
    /// fails. It does not prove every graph node runs on CUDA; CPU may execute
    /// nodes CUDA does not support.
    ///
    /// Without the feature flag, requesting it returns an error that names
    /// the needed Cargo feature.
    #[cfg_attr(not(feature = "onnx-cuda"), allow(dead_code))]
    pub prefer_cuda: bool,
}

/// An execution provider selectable through
/// [`create_onnx_session_with_provider`].
///
/// The function is additive: it avoids extending [`OnnxSessionConfig`], whose
/// public fields are also used by in-crate struct literals.
#[cfg(feature = "onnx")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum OnnxExecutionProvider {
    /// ONNX Runtime's CPU provider.
    Cpu,
    /// NVIDIA CUDA on Linux or Windows.
    Cuda,
    /// Apple CoreML on macOS.
    CoreMl,
    /// DirectML on Windows.
    DirectMl,
    /// ROCm on x86_64 Linux.
    Rocm,
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
    create_onnx_session_with_requested_provider(model_path, config, None)
}

/// Create a session with one explicitly selected execution provider.
///
/// If the selected accelerator cannot register, this returns an error rather
/// than accepting ONNX Runtime's default silent fallback to CPU. CPU remains
/// available for nodes unsupported by an accelerator that *does* register.
/// `provider` overrides `config.prefer_cuda` and `config.prefer_coreml`; those
/// legacy preferences are only honored by [`create_onnx_session`].
#[cfg(feature = "onnx")]
pub fn create_onnx_session_with_provider(
    model_path: &std::path::Path,
    mut config: OnnxSessionConfig,
    provider: OnnxExecutionProvider,
) -> Result<ort::session::Session> {
    // This API promises one explicit provider. Retaining a legacy preference
    // here could add a second EP or make an explicit CPU request fail because
    // CUDA/CoreML was not compiled into this build.
    config.prefer_cuda = false;
    config.prefer_coreml = false;
    create_onnx_session_with_requested_provider(model_path, config, Some(provider))
}

#[cfg(feature = "onnx")]
fn create_onnx_session_with_requested_provider(
    model_path: &std::path::Path,
    config: OnnxSessionConfig,
    requested_provider: Option<OnnxExecutionProvider>,
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

    // Build execution-provider list in priority order. A caller that requests
    // an accelerator needs an explicit error if it cannot register: ort's
    // default is otherwise to log and silently fall through to CPU. CPU is
    // still last so it can execute nodes unsupported by a registered EP.
    let mut providers: Vec<ort::execution_providers::ExecutionProviderDispatch> = Vec::new();
    match requested_provider {
        Some(OnnxExecutionProvider::Cuda) => {
            #[cfg(feature = "onnx-cuda")]
            {
                use ort::execution_providers::CUDAExecutionProvider;
                providers.push(CUDAExecutionProvider::default().build().error_on_failure());
            }
            #[cfg(not(feature = "onnx-cuda"))]
            return Err(Error::Retrieval(
                "CUDA requested, but anno was built without the `onnx-cuda` feature".into(),
            ));
        }
        Some(OnnxExecutionProvider::Rocm) => {
            #[cfg(feature = "onnx-rocm")]
            {
                use ort::ep::ROCm;
                providers.push(ROCm::default().build().error_on_failure());
            }
            #[cfg(not(feature = "onnx-rocm"))]
            return Err(Error::Retrieval(
                "ROCm requested, but anno was built without the `onnx-rocm` feature".into(),
            ));
        }
        Some(OnnxExecutionProvider::DirectMl) => {
            #[cfg(feature = "onnx-directml")]
            {
                use ort::ep::DirectML;
                // DirectML requires disabled memory patterns and sequential
                // execution. ort's provider registration does not set these.
                builder = builder
                    .with_memory_pattern(false)
                    .map_err(|e| Error::Retrieval(format!("ONNX DirectML memory pattern: {}", e)))?
                    .with_parallel_execution(false)
                    .map_err(|e| {
                        Error::Retrieval(format!("ONNX DirectML execution mode: {}", e))
                    })?;
                providers.push(DirectML::default().build().error_on_failure());
            }
            #[cfg(not(feature = "onnx-directml"))]
            return Err(Error::Retrieval(
                "DirectML requested, but anno was built without the `onnx-directml` feature".into(),
            ));
        }
        Some(OnnxExecutionProvider::CoreMl) => {
            #[cfg(feature = "onnx-coreml")]
            {
                use ort::execution_providers::CoreMLExecutionProvider;
                providers.push(
                    CoreMLExecutionProvider::default()
                        .build()
                        .error_on_failure(),
                );
            }
            #[cfg(not(feature = "onnx-coreml"))]
            return Err(Error::Retrieval(
                "CoreML requested, but anno was built without the `onnx-coreml` feature".into(),
            ));
        }
        Some(OnnxExecutionProvider::Cpu) | None => {}
    }

    #[cfg(not(feature = "onnx-cuda"))]
    if config.prefer_cuda && requested_provider != Some(OnnxExecutionProvider::Cuda) {
        return Err(Error::Retrieval(
            "CUDA requested, but anno was built without the `onnx-cuda` feature".into(),
        ));
    }
    #[cfg(feature = "onnx-cuda")]
    if config.prefer_cuda && requested_provider != Some(OnnxExecutionProvider::Cuda) {
        use ort::execution_providers::CUDAExecutionProvider;
        providers.push(CUDAExecutionProvider::default().build().error_on_failure());
    }
    #[cfg(not(feature = "onnx-coreml"))]
    if config.prefer_coreml && requested_provider != Some(OnnxExecutionProvider::CoreMl) {
        return Err(Error::Retrieval(
            "CoreML requested, but anno was built without the `onnx-coreml` feature".into(),
        ));
    }
    #[cfg(feature = "onnx-coreml")]
    if config.prefer_coreml && requested_provider != Some(OnnxExecutionProvider::CoreMl) {
        use ort::execution_providers::CoreMLExecutionProvider;
        providers.push(
            CoreMLExecutionProvider::default()
                .build()
                .error_on_failure(),
        );
    }
    if config.use_cpu_provider || requested_provider == Some(OnnxExecutionProvider::Cpu) {
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

#[cfg(all(test, feature = "onnx"))]
mod onnx_session_tests {
    use super::*;

    fn missing_model_path() -> std::path::PathBuf {
        tempfile::tempdir()
            .unwrap()
            .path()
            .join("missing-model.onnx")
    }

    #[cfg(not(feature = "onnx-cuda"))]
    #[test]
    fn explicit_cuda_without_feature_errors_before_model_load() {
        let error = create_onnx_session_with_provider(
            &missing_model_path(),
            OnnxSessionConfig::default(),
            OnnxExecutionProvider::Cuda,
        )
        .unwrap_err();

        assert!(error.to_string().contains("onnx-cuda"), "{error}");
        assert!(!error.to_string().contains("ONNX model load"), "{error}");
    }

    #[test]
    fn explicit_cpu_ignores_legacy_accelerator_preferences() {
        let config = OnnxSessionConfig {
            prefer_cuda: true,
            prefer_coreml: true,
            ..OnnxSessionConfig::default()
        };

        let error = create_onnx_session_with_provider(
            &missing_model_path(),
            config,
            OnnxExecutionProvider::Cpu,
        )
        .unwrap_err();

        assert!(error.to_string().contains("ONNX model load"), "{error}");
    }
}
