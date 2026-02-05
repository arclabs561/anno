//! Environment variable utilities.
//!
//! Provides centralized handling of .env files and environment configuration.
//!
//! # Usage
//!
//! ```rust,ignore
//! use anno::env::load_dotenv;
//!
//! // Load .env file if present (won't override existing env vars)
//! load_dotenv();
//!
//! // Now HF_TOKEN etc. are available from std::env::var
//! if let Ok(token) = std::env::var("HF_TOKEN") {
//!     println!("HuggingFace token available");
//! }
//! ```
//!
//! # Environment Variables
//!
//! | Variable | Purpose |
//! |----------|---------|
//! | `HF_TOKEN` | HuggingFace API token for gated models |
//! | `HF_API_TOKEN` | Alias for `HF_TOKEN` (HuggingFace API token) |
//! | `OPENAI_API_KEY` | OpenAI API key for LLM backends |
//! | `ANTHROPIC_API_KEY` | Anthropic API key for Claude LLM backends |
//! | `OPENROUTER_API_KEY` | OpenRouter API key for LLM backends |
//! | `GEMINI_API_KEY` | Google Gemini API key |
//! | `ANNO_CACHE_DIR` | Custom cache directory for models/datasets |
//! | `ANNO_CI_SEED` | Fixed seed for reproducible CI testing |
//! | `ANNO_SAMPLE_STRATEGY` | Backend sampling strategy (random, ml-only, worst-first) |
//! | `ANNO_MATRIX_TASK` | Optional task override for matrix sampler harness (e.g. discontinuous-ner, events, ned) |
//! | `ANNO_MATRIX_REQUIRE_CACHED` | If true, matrix sampler harness/sweeps only use cached datasets (local default false; CI forced true) |
//! | `ANNO_MATRIX_COVERAGE_REPORT` | If set, write a JSON coverage report to this path during tests |
//! | `ANNO_MATRIX_DISTRIBUTION_REPORT` | If set, write a JSON selection-distribution report to this path during tests |
//! | `ANNO_MATRIX_DISTRIBUTION_ITERS` | Number of simulated selections to run for distribution report (default 200) |
//! | `ANNO_ML_IN_MATRIX` | Include ML-ish backends in CI matrix (1/true to enable) |
//! | `ANNO_HISTORY_FILE` | Override muxer history JSON path for matrix sampler harness |
//! | `ANNO_MUXER_WINDOW_CAP` | Muxer history window size (per arm) |
//! | `ANNO_MUXER_PER_DATASET` | Use dataset-scoped muxer history + selection (1/true recommended) |
//! | `ANNO_MUXER_DATASETS_PER_RUN` | Matrix sampler harness: datasets per run (default 2) |
//! | `ANNO_MUXER_EXPLORATION_C` | Muxer UCB exploration coefficient |
//! | `ANNO_MUXER_JUNK_WEIGHT` | Muxer soft-junk penalty weight |
//! | `ANNO_MUXER_HARD_JUNK_WEIGHT` | Muxer hard-junk penalty weight |
//! | `ANNO_MUXER_COST_WEIGHT` | Muxer mean-cost penalty weight |
//! | `ANNO_MUXER_LATENCY_WEIGHT` | Muxer mean-latency penalty weight |
//! | `ANNO_MUXER_MAX_MEAN_ELAPSED_MS` | Optional constraint for ml-only selection: exclude arms above this mean latency (ms) |
//! | `ANNO_MUXER_LATENCY_GUARDRAIL_ALLOW_FEWER` | If true (default), ml-only may return fewer than K arms instead of falling back to slow ones |
//! | `ANNO_MUXER_LATENCY_GUARDRAIL_REQUIRE_MEASUREMENT` | If true, untried arms (calls=0) are excluded under the latency guardrail |
//! | `ANNO_MUXER_PROFILE` | Presets for latency guardrail (`off`, `fast`, `fast-strict`, `regress`) |
//! | `ANNO_MUXER_JUNK_F1_NER` | Junk cutoff for NER F1 (0..1) (default 0.05) |
//! | `ANNO_MUXER_JUNK_F1_COREF` | Junk cutoff for coref CoNLL F1 (0..1) |
//! | `ANNO_MUXER_JUNK_F1_RELATION` | Junk cutoff for relation strict F1 (0..1) |
//! | `ANNO_MUXER_VERBOSE` | Print chosen slice + per-result outcomes in matrix sampler harness |
//! | `ANNO_MUXER_HISTORY_SALT` | Optional suffix to isolate muxer history files (useful when semantics change) |
//! | `ANNO_MUXER_DECISIONS_FILE` | Optional path to write selection decisions as JSONL |
//! | `ANNO_MUXER_DECISIONS_TOP` | Max candidate rows included per decision (JSONL; default 8) |
//! | `ANNO_MUXER_MLONLY_POLICY` | `ml-only`: choose `exp3ix` (default) or `mab` |
//! | `ANNO_MUXER_EXP3_HORIZON` | EXP3-IX: horizon parameter (default 1000) |
//! | `ANNO_MUXER_EXP3_DECAY` | EXP3-IX: exponential decay per update (0.01..=1.0; default 1.0) |
//! | `ANNO_WORST_EXPLORATION_C` | Worst-first exploration coefficient (default 0.8) |
//! | `ANNO_WORST_HARD_WEIGHT` | Worst-first weight for hard failures (default 1.0) |
//! | `ANNO_WORST_SOFT_WEIGHT` | Worst-first weight for soft junk (default 0.0) |
//! | `ANNO_RELATION_TPLINKER_ORACLE_ENTITIES` | If true (default), TPLinker relation eval uses gold entity spans as candidates (keeps placeholder baseline non-degenerate) |
//!
//! # Muxer presets (recommended)
//!
//! Prefer `ANNO_MUXER_PROFILE` over individual latency-guardrail env vars:
//!
//! - `ANNO_MUXER_PROFILE=fast`: cap mean latency around 2s in ml-only selection
//! - `ANNO_MUXER_PROFILE=fast-strict`: like `fast`, but excludes untried arms under the cap
//! - `ANNO_MUXER_PROFILE=regress`: disable latency guardrail (useful for worst-first regression hunting)
//!
//! You can still override the preset explicitly with:
//! - `ANNO_MUXER_MAX_MEAN_ELAPSED_MS`
//! - `ANNO_MUXER_LATENCY_GUARDRAIL_ALLOW_FEWER`
//! - `ANNO_MUXER_LATENCY_GUARDRAIL_REQUIRE_MEASUREMENT`

use once_cell::sync::OnceCell;
use std::path::Path;

/// Global flag to track if dotenv was loaded
static DOTENV_LOADED: OnceCell<bool> = OnceCell::new();

/// Load environment variables from .env file if present.
///
/// This function:
/// - Searches for .env in current directory and up to 2 parent directories
/// - Only sets variables that aren't already set (env vars take precedence)
/// - Is idempotent - safe to call multiple times
/// - Returns silently if no .env file is found
///
/// # Example
///
/// ```rust,ignore
/// anno::env::load_dotenv();
/// ```
pub fn load_dotenv() {
    // Only load once
    DOTENV_LOADED.get_or_init(|| {
        load_dotenv_impl();
        true
    });
}

/// Force reload of .env file (useful for testing)
#[doc(hidden)]
pub fn reload_dotenv() {
    load_dotenv_impl();
}

fn load_dotenv_impl() {
    // Try to find .env in current directory or parents
    let env_paths = [".env", "../.env", "../../.env", "../../../.env"];

    for path_str in env_paths {
        let path = Path::new(path_str);
        if let Ok(contents) = std::fs::read_to_string(path) {
            parse_dotenv(&contents);
            return;
        }
    }

    // Also check workspace root via Cargo manifest
    if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
        let workspace_env = Path::new(&manifest_dir).join("../.env");
        if let Ok(contents) = std::fs::read_to_string(&workspace_env) {
            parse_dotenv(&contents);
        }
    }

    // Token alias normalization (non-overriding):
    //
    // Some tools (transformers/huggingface_hub) look for `HF_TOKEN` or
    // `HUGGINGFACE_HUB_TOKEN`. If the user provided `HF_API_TOKEN` in `.env`,
    // mirror it into those conventional vars (but never override).
    if std::env::var("HF_TOKEN").is_err() {
        if let Ok(v) = std::env::var("HF_API_TOKEN") {
            std::env::set_var("HF_TOKEN", v.clone());
            if std::env::var("HUGGINGFACE_HUB_TOKEN").is_err() {
                std::env::set_var("HUGGINGFACE_HUB_TOKEN", v);
            }
        }
    }
}

fn parse_dotenv(contents: &str) {
    for line in contents.lines() {
        let mut line = line.trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Common .env style: `export KEY=value`
        if let Some(rest) = line.strip_prefix("export ") {
            line = rest.trim();
        }

        // Parse KEY=VALUE (handle quoted values)
        if let Some((key, value)) = line.split_once('=') {
            let key = key.trim();
            let value = value.trim();

            // Remove surrounding quotes if present
            let value = value
                .strip_prefix('"')
                .and_then(|v| v.strip_suffix('"'))
                .or_else(|| value.strip_prefix('\'').and_then(|v| v.strip_suffix('\'')))
                .unwrap_or(value);

            // Only set if not already set (env vars take precedence)
            if std::env::var(key).is_err() {
                std::env::set_var(key, value);
            }
        }
    }
}

/// Check if HuggingFace token is available.
#[must_use]
pub fn has_hf_token() -> bool {
    // Support common aliases so `.env` can use a more explicit name.
    std::env::var("HF_TOKEN").is_ok() || std::env::var("HF_API_TOKEN").is_ok()
}

/// Get HuggingFace token if available.
#[must_use]
pub fn hf_token() -> Option<String> {
    std::env::var("HF_TOKEN")
        .ok()
        .or_else(|| std::env::var("HF_API_TOKEN").ok())
}

/// Check if any LLM API key is available.
#[must_use]
pub fn has_llm_api_key() -> bool {
    fn nonempty(name: &str) -> bool {
        std::env::var(name)
            .ok()
            .is_some_and(|v| !v.trim().is_empty())
    }

    nonempty("OPENAI_API_KEY")
        || nonempty("ANTHROPIC_API_KEY")
        || nonempty("OPENROUTER_API_KEY")
        || nonempty("GEMINI_API_KEY")
}

/// Get the best available LLM API key and provider.
/// Returns (key, provider) tuple.
#[must_use]
pub fn llm_api_key() -> Option<(String, &'static str)> {
    let nonempty = |name: &str| -> Option<String> {
        std::env::var(name).ok().filter(|v| !v.trim().is_empty())
    };

    if let Some(key) = nonempty("OPENAI_API_KEY") {
        return Some((key, "openai"));
    }
    if let Some(key) = nonempty("ANTHROPIC_API_KEY") {
        return Some((key, "anthropic"));
    }
    if let Some(key) = nonempty("OPENROUTER_API_KEY") {
        return Some((key, "openrouter"));
    }
    if let Some(key) = nonempty("GEMINI_API_KEY") {
        return Some((key, "gemini"));
    }
    None
}

/// Get the cache directory for models and datasets.
#[must_use]
pub fn cache_dir() -> std::path::PathBuf {
    if let Ok(dir) = std::env::var("ANNO_CACHE_DIR") {
        return std::path::PathBuf::from(dir);
    }

    // When analysis/eval feature is enabled, use platform-specific cache directories
    // (this gate keeps minimal builds local by default)
    #[cfg(any(feature = "analysis", feature = "eval"))]
    {
        #[cfg(target_os = "macos")]
        {
            dirs::home_dir()
                .map(|h| h.join("Library/Caches/anno"))
                .unwrap_or_else(|| std::path::PathBuf::from(".anno-cache"))
        }

        #[cfg(target_os = "linux")]
        {
            dirs::cache_dir()
                .map(|c| c.join("anno"))
                .unwrap_or_else(|| std::path::PathBuf::from(".anno-cache"))
        }

        #[cfg(target_os = "windows")]
        {
            dirs::cache_dir()
                .map(|c| c.join("anno"))
                .unwrap_or_else(|| std::path::PathBuf::from(".anno-cache"))
        }

        #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
        {
            std::path::PathBuf::from(".anno-cache")
        }
    }

    // Fallback when analysis/eval feature is not enabled
    #[cfg(not(any(feature = "analysis", feature = "eval")))]
    {
        std::path::PathBuf::from(".anno-cache")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_dotenv() {
        let contents = r#"
# Comment
KEY1=value1
KEY2="quoted value"
KEY3='single quoted'
  SPACED_KEY = spaced_value
"#;
        // Use unique keys to avoid test pollution
        let test_prefix = format!("ANNO_TEST_{}", std::process::id());

        let test_contents = contents.replace("KEY", &test_prefix);
        parse_dotenv(&test_contents);

        // The test environment might have these set already, so just check parsing works
    }

    #[test]
    fn test_parse_dotenv_supports_export_prefix_and_sets_values() {
        let pid = std::process::id();
        let k1 = format!("ANNO_TEST_EXPORT_{}_K1", pid);
        let k2 = format!("ANNO_TEST_EXPORT_{}_K2", pid);

        // Ensure clean slate
        std::env::remove_var(&k1);
        std::env::remove_var(&k2);

        let contents = format!(
            r#"
export {k1}=value1
{k2}="quoted value"
"#
        );
        parse_dotenv(&contents);

        assert_eq!(std::env::var(&k1).as_deref(), Ok("value1"));
        assert_eq!(std::env::var(&k2).as_deref(), Ok("quoted value"));

        // Clean up
        std::env::remove_var(&k1);
        std::env::remove_var(&k2);
    }

    #[test]
    fn test_parse_dotenv_does_not_override_existing_env() {
        let pid = std::process::id();
        let key = format!("ANNO_TEST_NO_OVERRIDE_{}", pid);

        std::env::set_var(&key, "from_env");

        let contents = format!(r#"{key}=from_dotenv"#);
        parse_dotenv(&contents);

        assert_eq!(std::env::var(&key).as_deref(), Ok("from_env"));

        std::env::remove_var(&key);
    }

    #[test]
    fn test_load_dotenv_idempotent() {
        load_dotenv();
        load_dotenv();
        load_dotenv();
        // Should not panic or cause issues
    }

    #[test]
    fn test_cache_dir() {
        let dir = cache_dir();
        // Should return a valid path
        assert!(!dir.as_os_str().is_empty());
    }
}
