//! Shared helpers for the muxer-backed eval matrix harness and CLI tooling.
//!
//! Goal: keep selection semantics (env parsing, guardrails, deterministic sampling) consistent
//! between:
//! - `crates/anno/src/matrix_muxer_ci.rs` (the CI-friendly matrix harness)
//! - `crates/anno/cli/commands/muxer.rs` (the `anno muxer` inspection tool)
//!
//! This module is only compiled when `eval-advanced` is enabled.

use muxer::MabConfig;

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
    // Deterministic (not crypto): FNV-1a 64-bit with seed mixing.
    let mut h: u64 = 14695981039346656037u64 ^ seed;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(1099511628211u64);
    }
    h
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
