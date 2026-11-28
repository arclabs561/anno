use super::no_downloads;
use crate::{Error, Result};

pub(crate) type ModelRepo = hf_hub::HFRepositorySync<hf_hub::RepoTypeModel>;

/// Initialize the Hugging Face client with the process environment's token and cache.
pub(crate) fn model_source() -> Result<hf_hub::HFClientSync> {
    let mut builder = hf_hub::HFClient::builder().cache_dir(hub_cache_dir());
    if let Some(token) = crate::env::hf_token() {
        builder = builder.token(token);
    }
    Ok(builder.build_sync()?)
}

pub(crate) fn hub_model(api: &hf_hub::HFClientSync, id: &str) -> ModelRepo {
    let (owner, name) = hf_hub::split_id(id);
    api.model(owner, name)
}

/// Preserve the native home directory on Windows when no Hub cache override is set.
pub(crate) fn hub_cache_dir() -> std::path::PathBuf {
    if cfg!(windows)
        && [
            "HF_HUB_CACHE",
            "HUGGINGFACE_HUB_CACHE",
            "HF_HOME",
            "XDG_CACHE_HOME",
            "HOME",
        ]
        .iter()
        .all(|key| std::env::var_os(key).is_none())
    {
        if let Some(home) = dirs::home_dir() {
            return home.join(".cache").join("huggingface").join("hub");
        }
    }
    hf_hub::resolve_cache_dir()
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
pub fn download_model_file<T: hf_hub::RepoType>(
    repo: &hf_hub::HFRepositorySync<T>,
    candidates: &[&str],
) -> Result<std::path::PathBuf> {
    download_model_file_at_revision(repo, None, candidates)
}

/// Resolve all candidates at the same revision, including in cache-only mode.
pub fn download_model_file_at_revision<T: hf_hub::RepoType>(
    repo: &hf_hub::HFRepositorySync<T>,
    revision: Option<&str>,
    candidates: &[&str],
) -> Result<std::path::PathBuf> {
    if candidates.is_empty() {
        return Err(Error::Retrieval(
            "download_model_file: candidates must not be empty".into(),
        ));
    }
    let offline = no_downloads();
    let mut last_error = String::new();
    for &filename in candidates {
        let fetch = |local_only| {
            repo.download_file()
                .filename(filename)
                .maybe_revision(revision.map(str::to_owned))
                .local_files_only(local_only)
                .send()
        };
        // Like the previous Hub client's get(), a cached named revision needs no HTTP request.
        let result = match fetch(true) {
            Err(_) if !offline => fetch(false),
            cached => cached,
        };
        match result {
            Ok(path) if path.is_file() => return Ok(path),
            Ok(path) => last_error = format!("{} is not a file", path.display()),
            Err(err) => last_error = err.to_string(),
        }
    }
    let mode = if offline {
        "cache lookup failed in offline mode (ANNO_NO_DOWNLOADS or HF_HUB_OFFLINE)"
    } else {
        "Hub download failed"
    };
    Err(Error::Retrieval(format!(
        "{mode} for [{}]: {last_error}",
        candidates.join(", ")
    )))
}
