//! Read installed model snapshots without linking the Hub transport.
use super::{cached_main_file, model_cache_dir};
use crate::{Error, Result};
use std::path::{Path, PathBuf};

pub(crate) struct ModelRepo {
    id: String,
    directory: PathBuf,
}

pub(crate) fn hub_cache_dir() -> PathBuf {
    for key in ["HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"] {
        if let Some(path) = std::env::var_os(key) {
            return PathBuf::from(path);
        }
    }
    if let Some(home) = std::env::var_os("HF_HOME") {
        return PathBuf::from(home).join("hub");
    }
    let cache = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            dirs::home_dir()
                .unwrap_or_else(std::env::temp_dir)
                .join(".cache")
        });
    cache.join("huggingface").join("hub")
}

pub(crate) fn model_source() -> Result<PathBuf> {
    Ok(hub_cache_dir())
}

pub(crate) fn hub_model(cache: &Path, id: &str) -> ModelRepo {
    ModelRepo {
        id: id.to_owned(),
        directory: model_cache_dir(cache, id),
    }
}

pub(crate) fn download_model_file(repo: &ModelRepo, candidates: &[&str]) -> Result<PathBuf> {
    candidates
        .iter()
        .find_map(|name| cached_main_file(&repo.directory, name))
        .ok_or_else(|| {
            Error::Retrieval(format!(
                "Could not find [{}] in {}. Run `hf download {}` with the same cache, supply a prepared model directory, or use the onnx feature for automatic downloads.",
                candidates.join(", "),
                repo.directory.display(),
                repo.id
            ))
        })
}
