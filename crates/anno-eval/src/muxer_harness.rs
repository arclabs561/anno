//! Shared helpers for the muxer-backed eval matrix harness and CLI tooling.
//!
//! Goal: keep selection semantics (env parsing, guardrails, deterministic sampling) consistent
//! between:
//! - `crates/anno-eval/src/matrix_muxer_ci.rs` (the CI-friendly matrix harness)
//! - `crates/anno-cli/src/cli/commands/muxer.rs` (the `anno muxer` inspection tool)
//!
//! This module is only compiled when `eval-advanced` is enabled.

use muxer::MabConfig;
use muxer::Summary;

use std::fmt;

#[cfg(feature = "eval-advanced")]
use crate::eval::loader::DatasetId;

/// Pseudo-call prior budget for muxer smoothing.
///
/// When building summaries for selection, we can “top up” sparse slices (e.g. facet-scoped
/// histories) using a small pseudo-count prior derived from broader history.
///
/// This is intentionally coarse: it’s meant to reduce cold-start instability, not to be a
/// statistically rigorous hierarchical model.
pub fn prior_calls_from_env() -> u64 {
    env_usize("ANNO_MUXER_PRIOR_CALLS", 6) as u64
}

/// If true (default), allow borrowing priors from datasets matching the current slice facets
/// (language + domain) when available.
pub fn prior_by_facets_from_env() -> bool {
    env_bool("ANNO_MUXER_PRIOR_BY_FACETS", true)
}

/// If true (default), force at least some within-slice exploration even when priors are enabled.
///
/// Motivation: prior smoothing can otherwise “mask” that an arm has never been tried in the
/// current facet/dataset slice, preventing muxer’s built-in explore-first behavior.
pub fn novelty_from_env() -> bool {
    env_bool("ANNO_MUXER_NOVELTY", true)
}

/// If the dataset set has exactly one language and one domain, return them as a facet prior filter.
#[cfg(feature = "eval-advanced")]
pub fn facet_prior_filter(datasets: &[DatasetId]) -> Option<(&'static str, &'static str)> {
    if datasets.is_empty() {
        return None;
    }
    let mut langs: Vec<&'static str> = datasets.iter().map(|d| d.language()).collect();
    langs.sort();
    langs.dedup();
    if langs.len() != 1 {
        return None;
    }
    let mut doms: Vec<&'static str> = datasets.iter().map(|d| d.domain()).collect();
    doms.sort();
    doms.dedup();
    if doms.len() != 1 {
        return None;
    }
    Some((langs[0], doms[0]))
}

/// Apply a pseudo-count prior to a muxer `Summary` until it reaches `target_calls`.
///
/// This converts the prior's rates/means into approximate counts and adds them to `out`.
pub fn apply_prior_counts_to_summary(out: &mut Summary, prior: Summary, target_calls: u64) {
    if target_calls == 0 || out.calls >= target_calls {
        return;
    }
    if prior.calls == 0 {
        return;
    }
    let need = target_calls.saturating_sub(out.calls);
    if need == 0 {
        return;
    }

    let need_f = need as f64;
    let ok = (need_f * prior.ok_rate()).round() as u64;
    let http_429 = (need_f * prior.http_429_rate()).round() as u64;
    let junk = (need_f * prior.junk_rate()).round() as u64;
    let hard_junk = (need_f * prior.hard_junk_rate()).round() as u64;
    let cost_units = (need_f * prior.mean_cost_units()).round() as u64;
    let elapsed_ms_sum = (need_f * prior.mean_elapsed_ms()).round() as u64;

    out.calls = out.calls.saturating_add(need);
    out.ok = out.ok.saturating_add(ok.min(need));
    out.http_429 = out.http_429.saturating_add(http_429.min(need));
    out.junk = out.junk.saturating_add(junk.min(need));
    out.hard_junk = out.hard_junk.saturating_add(hard_junk.min(need));
    out.cost_units = out.cost_units.saturating_add(cost_units);
    out.elapsed_ms_sum = out.elapsed_ms_sum.saturating_add(elapsed_ms_sum);

    // Defensive invariant: counts must not exceed calls.
    out.ok = out.ok.min(out.calls);
    out.http_429 = out.http_429.min(out.calls);
    out.junk = out.junk.min(out.calls);
    out.hard_junk = out.hard_junk.min(out.calls);
}

#[cfg(test)]
mod prior_tests {
    use super::*;

    #[test]
    fn test_apply_prior_counts_to_summary_tops_up_calls() {
        let mut out = Summary::default();
        let prior = Summary {
            calls: 100,
            ok: 80,
            http_429: 0,
            junk: 10,
            hard_junk: 5,
            cost_units: 200,
            elapsed_ms_sum: 1000,
        };
        apply_prior_counts_to_summary(&mut out, prior, 10);
        assert_eq!(out.calls, 10);
        assert!(out.ok <= out.calls);
        assert!(out.junk <= out.calls);
        assert!(out.hard_junk <= out.calls);
    }
}

/// Latency guardrail settings applied *outside* muxer selection (anno-specific).
#[derive(Debug, Clone, Copy)]
pub struct LatencyGuardrail {
    /// If set, exclude arms whose windowed mean latency exceeds this value (ms).
    pub max_mean_ms: Option<f64>,
    /// If true, ml-only selection may return fewer than K arms instead of falling back.
    pub allow_fewer: bool,
    /// If true, arms with `calls == 0` are excluded under the latency guardrail.
    pub require_measured: bool,
}

/// Parse a boolean environment variable with a default.
pub fn env_bool(name: &str, default: bool) -> bool {
    match std::env::var(name).ok().map(|v| v.trim().to_lowercase()) {
        None => default,
        Some(v) if v.is_empty() => default,
        Some(v) if v == "1" || v == "true" || v == "yes" || v == "y" => true,
        Some(v) if v == "0" || v == "false" || v == "no" || v == "n" => false,
        _ => default,
    }
}

/// Parse a usize environment variable with a default.
pub fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(default)
}

/// Parse an f64 environment variable with a default.
pub fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .unwrap_or(default)
}

/// Parse an optional f64 environment variable.
pub fn env_f64_opt(name: &str) -> Option<f64> {
    std::env::var(name)
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
}

/// Resolve the effective latency guardrail settings from env/profile presets.
pub fn latency_guardrail_from_env() -> LatencyGuardrail {
    let profile = std::env::var("ANNO_MUXER_PROFILE")
        .ok()
        .unwrap_or_else(|| "off".to_string())
        .trim()
        .to_lowercase();

    let (profile_max_ms, profile_require_measured) = match profile.as_str() {
        "fast" => (Some(2000.0), false),
        "fast-strict" => (Some(2000.0), true),
        // Regression-hunting / worst-first: keep guardrails off by default.
        "regress" => (None, false),
        _ => (None, false),
    };

    let max_mean_ms = env_f64_opt("ANNO_MUXER_MAX_MEAN_ELAPSED_MS").or(profile_max_ms);
    let allow_fewer = env_bool("ANNO_MUXER_LATENCY_GUARDRAIL_ALLOW_FEWER", true);
    let require_measured = env_bool(
        "ANNO_MUXER_LATENCY_GUARDRAIL_REQUIRE_MEASUREMENT",
        profile_require_measured,
    );

    LatencyGuardrail {
        max_mean_ms,
        allow_fewer,
        require_measured,
    }
}

/// Resolve muxer `MabConfig` from env (shared defaults across harness + CLI).
pub fn mab_config_from_env() -> MabConfig {
    MabConfig {
        exploration_c: env_f64("ANNO_MUXER_EXPLORATION_C", 0.8),
        cost_weight: env_f64("ANNO_MUXER_COST_WEIGHT", 0.0).max(0.0),
        latency_weight: env_f64("ANNO_MUXER_LATENCY_WEIGHT", 0.0).max(0.0),
        junk_weight: env_f64("ANNO_MUXER_JUNK_WEIGHT", 0.8).max(0.0),
        hard_junk_weight: env_f64("ANNO_MUXER_HARD_JUNK_WEIGHT", 1.6).max(0.0),
        // Hard constraints (BwK-ish gating).
        max_junk_rate: env_f64_opt("ANNO_MUXER_MAX_JUNK_RATE"),
        max_hard_junk_rate: env_f64_opt("ANNO_MUXER_MAX_HARD_JUNK_RATE"),
        max_http_429_rate: env_f64_opt("ANNO_MUXER_MAX_HTTP_429_RATE"),
        max_mean_cost_units: env_f64_opt("ANNO_MUXER_MAX_MEAN_COST_UNITS"),
    }
}

/// Deterministic (not crypto) stable hash used for “random” sampling.
pub fn stable_hash64(seed: u64, s: &str) -> u64 {
    // Deterministic (not crypto): FNV-1a string hash, then SplitMix64 finalizer.
    //
    // Motivation: we use these hashes for sampling fairness (task/dataset/backends). Raw FNV
    // has visible structure; a SplitMix64-style mix gives a much more uniform distribution
    // without adding RNG dependencies.
    let mut h: u64 = 14695981039346656037u64;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(1099511628211u64);
    }
    splitmix64(seed ^ h)
}

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Deterministic “random subset” used by the harness and CLI.
pub fn pick_random_subset(seed: u64, items: &[String], k: usize) -> Vec<String> {
    // Deterministic sampling without external RNG dependencies.
    let mut scored: Vec<(u64, &String)> =
        items.iter().map(|s| (stable_hash64(seed, s), s)).collect();
    scored.sort_by_key(|(h, s)| (*h, (*s).as_str()));
    scored
        .into_iter()
        .take(k.min(items.len()))
        .map(|(_, s)| s.clone())
        .collect()
}

// =============================================================================
// Slice + facet typing (shared: harness + CLI)
// =============================================================================

/// A stable, filename-safe tag identifying the muxer “slice”.
///
/// This intentionally encodes “what is being averaged together” in history:
/// - task code (e.g. `ner`, `temporal`, `discourse-segmentation`)
/// - optionally dataset facets (e.g. `.lang=en.dom=news`)
///
/// We keep this as a dedicated type so we don't accidentally mix:
/// - task codes (inputs)
/// - history filenames (storage)
/// - human labels (display)
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SliceTag(String);

impl SliceTag {
    /// Parse a slice tag from user/config input.
    ///
    /// Contract: must be ASCII and filename-safe; we keep it strict because it
    /// influences local file paths.
    pub fn parse(raw: &str) -> Result<Self, String> {
        let t = raw.trim();
        if t.is_empty() {
            return Err("slice tag is empty".to_string());
        }
        let mut out = String::new();
        for ch in t.chars() {
            let ch = ch.to_ascii_lowercase();
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.' | '=') {
                out.push(ch);
            } else {
                return Err(format!("invalid character '{}' in slice tag", ch));
            }
            if out.len() >= 128 {
                break;
            }
        }
        if out.is_empty() {
            return Err("slice tag became empty after normalization".to_string());
        }
        Ok(Self(out))
    }

    /// Return the normalized, filename-safe tag string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for SliceTag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A coarse summary of a facet value set.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FacetSlug {
    /// No facet data (should be rare; defensive).
    Unknown,
    /// Multiple facet values present in the dataset set.
    Mixed,
    /// Exactly one facet value (canonical, lowercase, filename-safe).
    Single(String),
}

impl FacetSlug {
    /// Return the normalized facet slug as a string (`unknown` / `mixed` / `<value>`).
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::Unknown => "unknown",
            Self::Mixed => "mixed",
            Self::Single(s) => s.as_str(),
        }
    }
}

#[cfg(feature = "eval-advanced")]
fn facet_slug(values: &[&'static str]) -> FacetSlug {
    let mut uniq: Vec<&'static str> = values.to_vec();
    uniq.sort();
    uniq.dedup();
    if uniq.is_empty() {
        FacetSlug::Unknown
    } else if uniq.len() == 1 {
        // Keep consistent with SliceTag's expectations.
        let v = uniq[0].trim().to_ascii_lowercase();
        let safe = v
            .chars()
            .take(64)
            .map(|ch| {
                if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.' | '=') {
                    ch
                } else {
                    '_'
                }
            })
            .collect::<String>();
        if safe.is_empty() {
            FacetSlug::Unknown
        } else {
            FacetSlug::Single(safe)
        }
    } else {
        FacetSlug::Mixed
    }
}

/// Coarse dataset facet summary used for muxer history scoping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatasetFacetSummary {
    /// Language facet summary for the chosen dataset set.
    pub language: FacetSlug,
    /// Domain facet summary for the chosen dataset set.
    pub domain: FacetSlug,
}

/// Compute the effective muxer slice tag (optionally facet-scoped).
#[cfg(feature = "eval-advanced")]
pub fn muxer_slice_tag(
    task_code: &str,
    datasets: &[crate::eval::loader::DatasetId],
    slice_by_dataset_facets: bool,
) -> Result<SliceTag, String> {
    let base = SliceTag::parse(task_code)?;
    if !slice_by_dataset_facets {
        return Ok(base);
    }
    let langs: Vec<&'static str> = datasets.iter().map(|d| d.language()).collect();
    let domains: Vec<&'static str> = datasets.iter().map(|d| d.domain()).collect();
    let summary = DatasetFacetSummary {
        language: facet_slug(&langs),
        domain: facet_slug(&domains),
    };
    let tagged = format!(
        "{}.lang={}.dom={}",
        base.as_str(),
        summary.language.as_str(),
        summary.domain.as_str()
    );
    SliceTag::parse(&tagged)
}

#[cfg(all(test, feature = "eval-advanced"))]
mod tests {
    use super::*;
    use crate::eval::loader::DatasetId;

    #[test]
    fn test_slice_tag_parse_rejects_pathy_chars() {
        assert!(SliceTag::parse("ner").is_ok());
        assert!(SliceTag::parse("ner.lang=en.dom=news").is_ok());
        assert!(SliceTag::parse("ner/../../oops").is_err());
        assert!(SliceTag::parse("ner lang=en").is_err());
    }

    #[test]
    fn test_muxer_slice_tag_facet_scoping() {
        let d1 = DatasetId::Wnut17; // en + social_media
        let d2 = DatasetId::GermEvalDiscontinuous; // de + (some non-en domain)

        let base = muxer_slice_tag("ner", &[d1], false).expect("task-only slice");
        assert_eq!(base.as_str(), "ner");

        let scoped = muxer_slice_tag("ner", &[d1], true).expect("facet slice");
        assert!(scoped.as_str().starts_with("ner.lang="));

        let mixed = muxer_slice_tag("ner", &[d1, d2], true).expect("mixed facet slice");
        assert!(mixed.as_str().contains(".lang=mixed."));
    }
}
