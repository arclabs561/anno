//! Shared HuggingFace model loading utilities.
//!
//! Resolves model artifacts from disk or the optional Hub client and builds ONNX sessions.

use crate::{Error, Result};

/// Returns `true` in a local-only build or when `ANNO_NO_DOWNLOADS` is truthy.
///
/// The flag blocks *new* network fetches; cached models still load via
/// [`download_model_file`]. Backends constructed from local paths bypass
/// this layer entirely.
pub fn no_downloads() -> bool {
    !cfg!(feature = "hub-downloads")
        || no_downloads_from(std::env::var("ANNO_NO_DOWNLOADS").ok().as_deref())
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

#[cfg(feature = "hub-downloads")]
mod transport;
#[cfg(feature = "hub-downloads")]
pub use transport::{download_model_file, download_model_file_at_revision};
#[cfg(feature = "hub-downloads")]
pub(crate) use transport::{hub_cache_dir, hub_model, model_source, ModelRepo};

#[cfg(not(feature = "hub-downloads"))]
mod local;
#[cfg(not(feature = "hub-downloads"))]
pub(crate) use local::{download_model_file, hub_cache_dir, hub_model, model_source, ModelRepo};

/// Directory of a model's local exports, named refs and snapshots.
pub(crate) fn model_cache_dir(cache: &std::path::Path, model: &str) -> std::path::PathBuf {
    cache.join(format!("models--{}", model.replace('/', "--")))
}

/// Probe the main snapshot without initializing a network client or async runtime.
pub(crate) fn cached_main_file(
    model_dir: &std::path::Path,
    name: &str,
) -> Option<std::path::PathBuf> {
    let revision = std::fs::read_to_string(model_dir.join("refs/main")).ok()?;
    let revision = revision.trim();
    if revision.is_empty()
        || !revision
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_')
    {
        return None;
    }
    let file = model_dir.join("snapshots").join(revision).join(name);
    file.is_file().then_some(file)
}

#[cfg(any(feature = "onnx-local", test))]
pub(crate) const QUANTIZED_ONNX_PATHS: &[&str] = &[
    "onnx/model_quantized.onnx",
    "model_quantized.onnx",
    "onnx/model_int8.onnx",
    "model_int8.onnx",
];
#[cfg(any(feature = "onnx-local", test))]
pub(crate) const FP32_ONNX_PATHS: &[&str] = &["onnx/model.onnx", "model.onnx"];

#[cfg(feature = "candle")]
pub(crate) const PYTORCH_WEIGHTS: &[&str] = &["pytorch_model.bin"];

#[cfg(feature = "candle")]
pub(crate) fn converted_pytorch_weights(path: &std::path::Path) -> std::path::PathBuf {
    path.with_file_name("model_converted.safetensors")
}

/// A native Candle model can also load a previously converted legacy checkpoint.
#[cfg(feature = "candle")]
pub(crate) fn candle_files_available(
    model: &str,
    config: &[&str],
    tokenizer: &[&str],
    weights: &[&str],
) -> bool {
    if !no_downloads() {
        return true;
    }
    let repo = model_cache_dir(&hub_cache_dir(), model);
    cached_candle_files(&repo, config, tokenizer, weights)
}

#[cfg(feature = "candle")]
fn cached_candle_files(
    repo: &std::path::Path,
    config: &[&str],
    tokenizer: &[&str],
    weights: &[&str],
) -> bool {
    let any_present = |names: &[&str]| {
        names
            .iter()
            .any(|name| cached_main_file(repo, name).is_some())
    };
    any_present(config)
        && any_present(tokenizer)
        && (any_present(weights)
            || PYTORCH_WEIGHTS.iter().any(|name| {
                cached_main_file(repo, name)
                    .is_some_and(|path| converted_pytorch_weights(&path).is_file())
            }))
}

/// Check exactly the graph variants the loader can use, without opening weights.
#[cfg(feature = "onnx-local")]
pub(crate) fn cached_hub_onnx_files(model: &str, quantized: bool, require_config: bool) -> bool {
    let repo = model_cache_dir(&hub_cache_dir(), model);
    let present = |name: &str| cached_main_file(&repo, name).is_some();
    present("tokenizer.json")
        && (!require_config || present("config.json"))
        && (FP32_ONNX_PATHS.iter().any(|name| present(name))
            || quantized && QUANTIZED_ONNX_PATHS.iter().any(|name| present(name)))
}

#[cfg(feature = "onnx-local")]
pub(crate) fn hub_onnx_files_available(model: &str, quantized: bool, require_config: bool) -> bool {
    !no_downloads() || cached_hub_onnx_files(model, quantized, require_config)
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
#[cfg(any(feature = "onnx-local", test))]
pub fn download_onnx_model(
    repo: &ModelRepo,
    prefer_quantized: bool,
) -> Result<(std::path::PathBuf, bool)> {
    if prefer_quantized {
        // Try quantized variants first
        if let Ok(path) = download_model_file(repo, QUANTIZED_ONNX_PATHS) {
            log::info!("[hf_loader] Using quantized model");
            return Ok((path, true));
        }
    }

    // Fall back to FP32
    let path = download_model_file(repo, FP32_ONNX_PATHS)?;
    if prefer_quantized {
        log::info!("[hf_loader] Using FP32 model (quantized not available)");
    }
    Ok((path, false))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "candle")]
    #[test]
    fn offline_candle_requires_preconverted_or_native_weights() {
        let home = tempfile::tempdir().unwrap();
        let repo = model_cache_dir(home.path(), "fixture/candle");
        std::fs::create_dir_all(repo.join("refs")).unwrap();
        std::fs::write(repo.join("refs/main"), "fixture").unwrap();
        let snapshot = repo.join("snapshots/fixture");
        std::fs::create_dir_all(&snapshot).unwrap();
        for name in ["config.json", "tokenizer.json", "pytorch_model.bin"] {
            std::fs::write(snapshot.join(name), []).unwrap();
        }
        let available = || {
            cached_candle_files(
                &repo,
                &["config.json"],
                &["tokenizer.json"],
                &["model.safetensors"],
            )
        };
        assert!(!available());
        std::fs::write(snapshot.join("model_converted.safetensors"), []).unwrap();
        assert!(available());
        std::fs::remove_file(snapshot.join("pytorch_model.bin")).unwrap();
        assert!(!available());
        std::fs::write(snapshot.join("model.safetensors"), []).unwrap();
        assert!(available());
    }

    #[cfg(feature = "hub-downloads")]
    #[test]
    fn immutable_snapshot_loads_without_a_ref_file() {
        let home = tempfile::tempdir().unwrap();
        let revision = "4241d7c66b648e618c89c150bf4cf418d2f83159";
        let api = hf_hub::HFClient::builder()
            .cache_dir(home.path())
            .build_sync()
            .unwrap();
        let repo = api.model("anno-tests", "pinned");
        let path = model_cache_dir(home.path(), "anno-tests/pinned")
            .join("snapshots")
            .join(revision)
            .join("tokenizer.json");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, "{}").unwrap();
        let cached = |rev: &str, file: &str| {
            repo.download_file()
                .filename(file)
                .revision(rev)
                .local_files_only(true)
                .send()
        };
        assert_eq!(cached(revision, "tokenizer.json").unwrap(), path);
        assert!(cached(revision, "missing.json").is_err());
        assert!(cached("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "tokenizer.json").is_err());
    }

    #[cfg(feature = "hub-downloads")]
    #[test]
    fn offline_candidate_lookup_preserves_revision_without_network() {
        // Isolate environment flags from tests running concurrently in this process.
        const CHILD: &str = "ANNO_TEST_OFFLINE_HF_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let home = tempfile::tempdir().unwrap();
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "backends::hf_loader::tests::offline_candidate_lookup_preserves_revision_without_network", "--nocapture"])
                .env(CHILD, "1").env("ANNO_NO_DOWNLOADS", "1")
                .env_remove("HF_HUB_CACHE").env_remove("HUGGINGFACE_HUB_CACHE")
                .env("HF_HOME", home.path()).status().unwrap();
            assert!(status.success());
            return;
        }
        // A live local listener makes any unintended network access observable.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let api = hf_hub::HFClient::builder()
            .endpoint(format!("http://{}", listener.local_addr().unwrap()))
            .build_sync()
            .unwrap();
        let repo = api.model("anno-tests", "offline");
        let lookup = || {
            download_model_file_at_revision(
                &repo,
                Some("refs/pr/7"),
                &["onnx/model_quantized.onnx", "model.onnx"],
            )
        };
        // No server is needed: the guarded path must return before any connection.
        // The child timeout is enforced by nextest's ordinary test timeout.
        let error = lookup().unwrap_err();
        assert!(error.to_string().contains("ANNO_NO_DOWNLOADS"));
        assert!(listener.accept().is_err());
        let root = model_cache_dir(&hub_cache_dir(), "anno-tests/offline");
        std::fs::create_dir_all(root.join("refs/refs/pr")).unwrap();
        std::fs::create_dir_all(root.join("snapshots/pinned")).unwrap();
        std::fs::write(root.join("refs/refs/pr/7"), "pinned").unwrap();
        // A directory at an earlier candidate cannot shadow a usable graph.
        std::fs::create_dir_all(root.join("snapshots/pinned/onnx/model_quantized.onnx")).unwrap();
        let model = root.join("snapshots/pinned/model.onnx");
        std::fs::write(&model, b"cached model fixture").unwrap();
        assert_eq!(lookup().unwrap(), model);
        assert!(listener.accept().is_err());
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
/// care about before constructing the session.
#[cfg(feature = "onnx-local")]
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct OnnxSessionConfig {
    /// ONNX graph optimization level (1-3, default 3).
    pub optimization_level: u8,
    /// Prefix for an ONNX Runtime execution profile. Call `Session::end_profiling`
    /// after inference to obtain the file containing node-to-provider assignments.
    pub profile_prefix: Option<std::path::PathBuf>,
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
#[cfg(feature = "onnx-local")]
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

#[cfg(feature = "onnx-local")]
impl Default for OnnxSessionConfig {
    fn default() -> Self {
        Self {
            optimization_level: 3,
            profile_prefix: None,
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
#[cfg(feature = "onnx-local")]
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
#[cfg(feature = "onnx-local")]
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

#[cfg(feature = "onnx-local")]
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

    if let Some(prefix) = &config.profile_prefix {
        builder = builder
            .with_profiling(prefix)
            .map_err(|e| Error::Retrieval(format!("ONNX profiling: {e}")))?;
    }

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
#[cfg(feature = "onnx-local")]
pub fn load_tokenizer(path: &std::path::Path) -> Result<tokenizers::Tokenizer> {
    tokenizers::Tokenizer::from_file(path)
        .map_err(|e| Error::Retrieval(format!("Tokenizer load: {}", e)))
}

#[cfg(all(test, feature = "onnx-local"))]
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
