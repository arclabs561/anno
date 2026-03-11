//! Shared helpers for the muxer-backed eval matrix sampler harness and CLI tooling.
//!
//! Goal: keep selection semantics (env parsing, guardrails, deterministic sampling) consistent
//! between:
//! - `crates/anno-eval/src/matrix_muxer_ci.rs` (the CI-friendly matrix sampler harness)
//! - `crates/anno-cli/src/cli/commands/muxer.rs` (the `anno sampler` inspection tool)
//!
//! This module is only compiled when `eval` is enabled.
//!
//! ## Monitored selection (drift / catKL / CUSUM)
//!
//! When any monitoring knob is enabled in `MabConfig` (a `max_*` guard or a non-zero `*_weight`),
//! the matrix sampler uses muxer’s *monitored* selection path. The key design choice is:
//!
//! - Base selection objectives can use *prior-smoothed summaries* (cold-start stabilization).
//! - Monitoring scores (drift / catKL / CUSUM) are computed from *raw windows*.
//!
//! This ensures priors cannot “mask” change detection.
//!
//! ### Backend-global monitoring (not per-dataset)
//!
//! Change monitors require a coherent time order of outcomes. Dataset-scoped histories aggregate
//! outcomes across datasets without a stable cross-dataset temporal ordering, so order-sensitive
//! monitors (CUSUM) become ill-defined. For that reason, `anno` computes monitoring windows at the
//! backend-global level (keys are backend names only) even when base summaries are dataset-scoped.

use muxer::{DriftConfig, DriftMetric, MabConfig};

// Prefer centralizing generic policy helpers in `muxer` itself so harnesses/CLIs don’t drift.
pub use muxer::{
    apply_prior_counts_to_summary,
    guardrail_filter_observed,
    guardrail_filter_observed_elapsed,
    novelty_pick_unseen,
    pick_control_arms,
    policy_fill_generic,
    policy_fill_k_observed_with_coverage,
    policy_plan_generic,
    select_k_without_replacement_by,
    split_control_budget,
    stable_hash64,
    suggested_window_cap,
    worst_first_pick_k,
    worst_first_pick_one,
    // 0.3.7: quality signal
    BanditPolicy,
    // 0.3.x: control-arm helpers + window sizing
    ControlConfig,
    CoverageConfig,
    LatencyGuardrailConfig,
    PipelineOrder,
    PolicyFill,
    PolicyPlan,
    WorstFirstConfig,
};

/// Compat shim: delegates to `policy_fill_k_observed_with_coverage` with default coverage.
pub fn policy_fill_k_observed_with<F, P>(
    seed: u64,
    arms: &[String],
    k: usize,
    novelty_enabled: bool,
    guard: LatencyGuardrailConfig,
    observed: F,
    pick_rest: P,
) -> PolicyFill
where
    F: FnMut(&str) -> (u64, u64),
    P: FnMut(&[String], usize) -> Vec<String>,
{
    policy_fill_k_observed_with_coverage(
        seed,
        arms,
        k,
        novelty_enabled,
        CoverageConfig::default(),
        guard,
        observed,
        pick_rest,
    )
}

/// Compat shim: delegates to `policy_plan_generic` with default coverage and NoveltyFirst order.
pub fn policy_plan_observed(
    seed: u64,
    arms: &[String],
    k: usize,
    novelty_enabled: bool,
    guard: LatencyGuardrailConfig,
    observed: impl FnMut(&str) -> (u64, u64),
) -> PolicyPlan {
    policy_plan_generic(
        seed,
        arms,
        k,
        novelty_enabled,
        CoverageConfig::default(),
        guard,
        PipelineOrder::NoveltyFirst,
        observed,
    )
}

/// Compat shim: deterministic random subset selection (was public in muxer < 0.3.12).
pub fn pick_random_subset(seed: u64, items: &[String], k: usize) -> Vec<String> {
    use muxer::stable_hash64;
    if k >= items.len() {
        return items.to_vec();
    }
    let mut scored: Vec<(u64, &String)> = items
        .iter()
        .map(|s| (stable_hash64(seed, s), s))
        .collect();
    scored.sort_by_key(|(h, _)| *h);
    scored.into_iter().take(k).map(|(_, s)| s.clone()).collect()
}

use std::fmt;

#[cfg(feature = "eval")]
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

/// Reserve a small “control” budget of picks to mitigate selection bias (default: 0/off).
///
/// If non-zero, callers should take `k_control` arms from a deterministic random subset and only
/// use muxer policies for the remaining \(k - k_control\) picks.
pub fn control_k_from_env() -> usize {
    env_usize("ANNO_MUXER_CONTROL_K", 0)
}

/// If the dataset set has exactly one language and one domain, return them as a facet prior filter.
#[cfg(feature = "eval")]
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

#[cfg(test)]
mod prior_tests {
    use super::*;
    use muxer::Summary;

    #[test]
    fn test_apply_prior_counts_to_summary_tops_up_calls() {
        let mut out = Summary::default();
        let prior = Summary {
            calls: 100,
            ok: 80,
            junk: 10,
            hard_junk: 5,
            cost_units: 200,
            elapsed_ms_sum: 1000,
            mean_quality_score: None,
        };
        apply_prior_counts_to_summary(&mut out, prior, 10);
        assert_eq!(out.calls, 10);
        assert!(out.ok <= out.calls);
        assert!(out.junk <= out.calls);
        assert!(out.hard_junk <= out.calls);
    }

    #[test]
    fn test_guardrail_filter_observed_elapsed_require_measured() {
        let guard = LatencyGuardrailConfig {
            max_mean_ms: Some(10.0),
            allow_fewer: true,
            require_measured: true,
        };
        let arms = vec!["a".to_string()];
        let (eligible, stop_early) =
            guardrail_filter_observed_elapsed(0, &arms, guard, |_b| (0, 0));
        assert!(eligible.is_empty());
        assert!(stop_early);
    }

    #[test]
    fn test_novelty_pick_unseen_deterministic() {
        let arms = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let pick1 = novelty_pick_unseen(123, &arms, 2, true, |b| if b == "b" { 1 } else { 0 });
        let pick2 = novelty_pick_unseen(123, &arms, 2, true, |b| if b == "b" { 1 } else { 0 });
        assert_eq!(pick1, pick2);
        assert!(!pick1.contains(&"b".to_string()));
    }

    #[test]
    fn test_policy_plan_observed_stop_early_keeps_prechosen() {
        let guard = LatencyGuardrailConfig {
            max_mean_ms: Some(1.0),
            allow_fewer: true,
            require_measured: true,
        };
        let arms = vec!["a".to_string(), "b".to_string()];
        // Only `a` is unseen -> it will be prechosen; remaining `b` is unmeasured and should be
        // filtered by guardrail, causing stop_early.
        let plan = policy_plan_observed(0, &arms, 1, true, guard, |_arm| (0, 0));
        assert_eq!(plan.prechosen, vec!["a".to_string()]);
        assert!(plan.eligible.is_empty());
        assert!(plan.stop_early);
    }

    #[test]
    fn test_policy_fill_falls_back_if_stop_early_would_pick_nothing() {
        let guard = LatencyGuardrailConfig {
            max_mean_ms: Some(1.0),
            allow_fewer: true,
            require_measured: true,
        };
        let arms = vec!["a".to_string()];
        let fill = policy_fill_k_observed_with(
            0,
            &arms,
            1,
            false,
            guard,
            |_b| (0, 0),
            |xs, _k| vec![xs[0].clone()],
        );
        assert!(fill.chosen.is_empty());
        assert!(!fill.fallback_used);
        assert!(fill.stopped_early);
    }

    #[test]
    fn test_policy_fill_novelty_then_stop_early_does_not_fallback() {
        // Difficult seam: novelty can create a non-empty chosen set, after which the observed
        // guardrail can stop-early. In that case we must *not* fall back and pick more.
        let guard = LatencyGuardrailConfig {
            max_mean_ms: Some(10.0),
            allow_fewer: true,
            require_measured: true,
        };
        let arms = vec!["unseen".to_string(), "slow".to_string(), "fast".to_string()];

        // unseen => prechosen; then guardrail filters remaining (measured but too slow) so it stops early.
        let fill = policy_fill_k_observed_with(
            123,
            &arms,
            2,
            true,
            guard,
            |b| {
                if b == "unseen" {
                    (0, 0)
                } else {
                    // Measured but too slow: mean_ms = 1000/10 = 100 > max_mean_ms
                    (10, 1000)
                }
            },
            |_eligible, _k| unreachable!("algorithm should not be called on stop-early"),
        );
        assert_eq!(fill.chosen, vec!["unseen".to_string()]);
        assert!(fill.stopped_early);
        assert!(!fill.fallback_used);
    }

    #[test]
    fn test_policy_fill_partial_novelty_invokes_algorithm_on_remaining_k() {
        // Difficult seam: novelty might pick < k; ensure the algorithm is invoked with eligible
        // excluding prechosen and with remaining_k only.
        let guard = LatencyGuardrailConfig {
            max_mean_ms: Some(100.0),
            allow_fewer: false,
            require_measured: false,
        };
        let arms = vec!["u1".to_string(), "m1".to_string(), "m2".to_string()];
        let mut saw_args: Option<(Vec<String>, usize)> = None;
        let fill = policy_fill_k_observed_with(
            0,
            &arms,
            2,
            true,
            guard,
            |b| {
                // Only u1 is unseen.
                if b == "u1" {
                    (0, 0)
                } else {
                    (10, 10)
                }
            },
            |eligible, k| {
                saw_args = Some((eligible.to_vec(), k));
                vec![eligible[0].clone()]
            },
        );
        assert_eq!(fill.chosen.len(), 2);
        assert!(fill.chosen.contains(&"u1".to_string()));
        let (eligible, k) = saw_args.expect("algorithm called");
        assert_eq!(k, 1);
        assert!(!eligible.contains(&"u1".to_string()));
    }

    #[test]
    fn test_policy_fill_fallback_used_when_guardrail_filters_all_and_no_picks_yet() {
        // When require_measured is set, we should stop early rather than falling back to
        // unmeasured arms.
        let guard = LatencyGuardrailConfig {
            max_mean_ms: Some(1.0),
            allow_fewer: true,
            require_measured: true,
        };
        let arms = vec!["a".to_string(), "b".to_string()];
        let fill = policy_fill_k_observed_with(
            999,
            &arms,
            1,
            false,
            guard,
            |_b| (0, 0),
            |eligible, _k| vec![eligible[0].clone()],
        );
        assert!(fill.chosen.is_empty());
        assert!(!fill.fallback_used);
        assert!(fill.stopped_early);
    }

    #[test]
    fn test_policy_fill_adversarial_invariants_small_fuzz() {
        // Adversarial “property-ish” test: exercise many small cases to catch drift.
        // We keep it deterministic and light (no heavy eval / IO).
        fn lcg(mut x: u64) -> u64 {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            x
        }

        for seed in 0u64..100 {
            let mut x = seed ^ 0xA11C_E5E1;
            let n = (lcg(x) % 9 + 2) as usize; // 2..=10 arms
            x = lcg(x);
            let k = ((lcg(x) % (n as u64)) + 1) as usize;
            x = lcg(x);

            let arms = (0..n).map(|i| format!("arm{i}")).collect::<Vec<_>>();
            let novelty = (x & 1) == 1;
            x = lcg(x);

            // Guardrail: sometimes active, sometimes off.
            let guard = LatencyGuardrailConfig {
                max_mean_ms: if (x & 1) == 1 { Some(10.0) } else { None },
                allow_fewer: true,
                require_measured: (x & 2) == 2,
            };
            x = lcg(x);

            // Observations: calls 0..=3, elapsed chosen so mean_ms is either 0, 5, or 50.
            // This ensures some arms are filtered when max_mean_ms=10.
            let mut obs: std::collections::BTreeMap<String, (u64, u64)> = Default::default();
            for a in &arms {
                x = lcg(x);
                let calls = x % 4;
                x = lcg(x);
                let mean_bucket = x % 3; // 0,1,2
                let mean_ms = match mean_bucket {
                    0 => 0,
                    1 => 5,
                    _ => 50,
                };
                let elapsed_sum = calls.saturating_mul(mean_ms);
                obs.insert(a.clone(), (calls, elapsed_sum));
            }

            let mut algo_calls = 0usize;
            let fill = policy_fill_k_observed_with(
                seed,
                &arms,
                k,
                novelty,
                guard,
                |b| *obs.get(b).unwrap(),
                |eligible, k_rem| {
                    algo_calls += 1;
                    // Adversarial-ish algorithm: may return duplicates or extra elements.
                    let mut out = pick_random_subset(seed ^ 0xBEEF, eligible, k_rem);
                    if !out.is_empty() {
                        out.push(out[0].clone()); // duplicate
                    }
                    out.push("not_in_set".to_string()); // invalid
                    out
                },
            );

            // Invariants: chosen is a subset, unique, and bounded by k.
            assert!(fill.chosen.len() <= k);
            let chosen_set: std::collections::BTreeSet<String> =
                fill.chosen.iter().cloned().collect();
            assert_eq!(chosen_set.len(), fill.chosen.len());
            for c in &fill.chosen {
                assert!(arms.contains(c));
            }

            // If we stopped early we must not have called the algorithm picker.
            if fill.stopped_early {
                assert_eq!(algo_calls, 0);
            }
        }
    }

    #[test]
    fn test_worst_first_pick_prefers_unseen() {
        let remaining = vec!["a".to_string(), "b".to_string()];
        let (pick, explore_first) = worst_first_pick_one(
            0,
            &remaining,
            WorstFirstConfig {
                exploration_c: 0.8,
                hard_weight: 1.0,
                soft_weight: 0.0,
            },
            |b| if b == "b" { 0 } else { 1 },
            |_b| (10, 0.0, 0.0),
        )
        .unwrap();
        assert!(explore_first);
        assert_eq!(pick, "b");
    }

    #[test]
    fn test_worst_first_pick_scores_worse_higher() {
        let remaining = vec!["a".to_string(), "b".to_string()];
        let (pick, explore_first) = worst_first_pick_one(
            0,
            &remaining,
            WorstFirstConfig {
                exploration_c: 0.0,
                hard_weight: 1.0,
                soft_weight: 0.0,
            },
            |_b| 1,
            |b| {
                if b == "a" {
                    (10, 0.1, 0.0)
                } else {
                    (10, 0.9, 0.0)
                }
            },
        )
        .unwrap();
        assert!(!explore_first);
        assert_eq!(pick, "b");
    }

    #[test]
    fn test_worst_first_pick_k_covers_unseen_first_without_duplicates() {
        let arms = (0..10).map(|i| format!("b{i}")).collect::<Vec<_>>();
        let unseen: std::collections::BTreeSet<String> =
            ["b1", "b3", "b7"].iter().map(|s| s.to_string()).collect();
        let picks = worst_first_pick_k(
            123,
            &arms,
            5,
            WorstFirstConfig {
                exploration_c: 0.8,
                hard_weight: 1.0,
                soft_weight: 0.0,
            },
            |b| if unseen.contains(b) { 0 } else { 5 },
            |_b| (10, 0.1, 0.0),
        );
        let chosen: Vec<String> = picks.iter().map(|p| p.0.clone()).collect();
        let chosen_set: std::collections::BTreeSet<String> = chosen.iter().cloned().collect();
        assert_eq!(chosen.len(), chosen_set.len());
        // The first |unseen| picks must all come from unseen.
        for (arm, explore) in picks.iter().take(unseen.len()) {
            assert!(unseen.contains(arm));
            assert!(*explore);
        }
    }

    #[test]
    fn test_worst_first_pick_is_deterministic_on_exact_ties() {
        // Seam: when all arms have identical stats, worst-first should be deterministic and use
        // only the seed+name tie-breaker.
        let remaining = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let seed = 42u64;
        let cfg = WorstFirstConfig {
            exploration_c: 0.0,
            hard_weight: 1.0,
            soft_weight: 1.0,
        };
        let (pick1, explore1) =
            worst_first_pick_one(seed, &remaining, cfg, |_b| 1, |_b| (10, 0.2, 0.2)).unwrap();
        let (pick2, explore2) =
            worst_first_pick_one(seed, &remaining, cfg, |_b| 1, |_b| (10, 0.2, 0.2)).unwrap();
        assert_eq!(pick1, pick2);
        assert_eq!(explore1, explore2);
        assert!(!explore1);

        // Tie-break contract: smallest stable hash under the worst-first salt.
        let expected = remaining
            .iter()
            .min_by_key(|b| stable_hash64(seed ^ 0x574F_5253, b))
            .unwrap()
            .clone();
        assert_eq!(pick1, expected);
    }

    #[test]
    fn test_worst_first_pick_exploration_favors_low_calls_when_rates_equal() {
        // When rates are equal, exploration should bias toward lower-sample arms.
        let remaining = vec!["low_calls".to_string(), "high_calls".to_string()];
        let cfg = WorstFirstConfig {
            exploration_c: 1.0,
            hard_weight: 1.0,
            soft_weight: 0.0,
        };
        let (pick, explore_first) = worst_first_pick_one(
            0,
            &remaining,
            cfg,
            |_b| 1,
            |b| {
                if b == "low_calls" {
                    (2, 0.2, 0.0)
                } else {
                    (200, 0.2, 0.0)
                }
            },
        )
        .unwrap();
        assert!(!explore_first);
        assert_eq!(pick, "low_calls");
    }

    #[test]
    fn test_select_k_without_replacement_by_dedups_and_preserves_order() {
        let items = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let out = select_k_without_replacement_by(0, &items, 3, |_seed, _rem, _k| {
            vec![
                "b".to_string(),
                "b".to_string(),
                "a".to_string(),
                "x".to_string(), // not in remaining -> ignored
            ]
        });
        assert_eq!(out, vec!["b".to_string(), "a".to_string()]);
    }

    #[test]
    fn test_select_k_without_replacement_by_breaks_on_no_progress() {
        // Adversarial: picker never returns a valid remaining item; driver must terminate.
        let items = vec!["a".to_string(), "b".to_string()];
        let out = select_k_without_replacement_by(0, &items, 2, |_seed, _rem, _k| {
            vec!["x".to_string(), "x".to_string()]
        });
        assert!(out.is_empty());
    }

    #[test]
    fn test_select_k_without_replacement_by_handles_partial_progress_then_stalls() {
        // Adversarial: first call makes progress, subsequent calls only repeat already-chosen items.
        let items = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let mut calls = 0usize;
        let out = select_k_without_replacement_by(0, &items, 3, |_seed, _rem, _k| {
            calls += 1;
            if calls == 1 {
                vec!["b".to_string()]
            } else {
                vec!["b".to_string(), "x".to_string()]
            }
        });
        assert_eq!(out, vec!["b".to_string()]);
    }

    // Tests for select_k_without_replacement_by_with_meta removed: function is no longer
    // public in muxer >= 0.3.12.
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

/// Coarse, domain-agnostic failure kind classification from an error string.
///
/// This is intended for *triage* and regression hunting, not for selection math. Keep it stable and
/// low-cardinality so history files remain readable and comparisons remain meaningful across time.
pub fn classify_failure_kind(err: &str) -> &'static str {
    let e = err.to_ascii_lowercase();

    // Rate limiting / quota / auth.
    if e.contains("429")
        || e.contains("rate limit")
        || e.contains("too many requests")
        || e.contains("quota")
        || e.contains("throttle")
    {
        return "rate_limit";
    }

    // Timeouts / cancellation.
    if e.contains("timeout")
        || e.contains("timed out")
        || e.contains("deadline")
        || e.contains("cancelled")
        || e.contains("canceled")
    {
        return "timeout";
    }

    // Dataset load/parse/IO.
    if e.contains("dataset")
        && (e.contains("load")
            || e.contains("open")
            || e.contains("read")
            || e.contains("parse")
            || e.contains("json")
            || e.contains("csv")
            || e.contains("conllu")
            || e.contains("format"))
    {
        return "dataset";
    }
    if e.contains("download") || e.contains("http") || e.contains("fetch") || e.contains("s3") {
        return "io";
    }

    // Metric/scoring issues.
    if e.contains("metric")
        || e.contains("score")
        || e.contains("confusion")
        || e.contains("nan")
        || e.contains("infinite")
    {
        return "metric";
    }

    // Backend/model/inference issues.
    if e.contains("backend")
        || e.contains("model")
        || e.contains("onnx")
        || e.contains("tensor")
        || e.contains("inference")
        || e.contains("tokenizer")
    {
        return "backend";
    }

    "unknown"
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

fn parse_simplex4_csv(raw: &str) -> Option<[f64; 4]> {
    let t = raw.trim();
    if t.is_empty() {
        return None;
    }
    let parts: Vec<&str> = t
        .split(',')
        .map(|x| x.trim())
        .filter(|x| !x.is_empty())
        .collect();
    if parts.len() != 4 {
        return None;
    }
    let mut xs = [0.0f64; 4];
    for (i, p) in parts.iter().enumerate() {
        let v = p.parse::<f64>().ok()?;
        if !v.is_finite() || v < 0.0 {
            return None;
        }
        xs[i] = v;
    }
    let sum = xs.iter().copied().sum::<f64>();
    if !sum.is_finite() || sum <= 0.0 {
        return None;
    }
    for x in xs.iter_mut() {
        *x /= sum;
    }
    Some(xs)
}

fn env_simplex4_opt(name: &str) -> Option<[f64; 4]> {
    let raw = std::env::var(name).ok()?;
    parse_simplex4_csv(&raw)
}

/// Resolve the effective latency guardrail settings from env/profile presets.
pub fn latency_guardrail_from_env() -> LatencyGuardrailConfig {
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

    LatencyGuardrailConfig {
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
        // Quality signal from continuous F1 scores (requires quality_score set on Outcome).
        // Default 0.0 = disabled; set > 0 to blend into the MAB objective.
        quality_weight: env_f64("ANNO_MUXER_QUALITY_WEIGHT", 0.0).max(0.0),
        // Hard constraints (BwK-ish gating).
        max_junk_rate: env_f64_opt("ANNO_MUXER_MAX_JUNK_RATE"),
        max_hard_junk_rate: env_f64_opt("ANNO_MUXER_MAX_HARD_JUNK_RATE"),
        max_mean_cost_units: env_f64_opt("ANNO_MUXER_MAX_MEAN_COST_UNITS"),

        // Drift monitoring (optional; monitored selection only).
        max_drift: env_f64_opt("ANNO_MUXER_MAX_DRIFT"),
        drift_metric: drift_metric_from_env("ANNO_MUXER_DRIFT_METRIC", DriftMetric::Hellinger),
        drift_weight: env_f64("ANNO_MUXER_DRIFT_WEIGHT", 0.0).max(0.0),

        // Change monitoring (optional; monitored selection only).
        //
        // catKL uses: S = n_recent * KL(q_recent || p0_baseline)
        max_catkl: env_f64_opt("ANNO_MUXER_MAX_CATKL"),
        catkl_alpha: env_f64("ANNO_MUXER_CATKL_ALPHA", 1e-3).max(0.0),
        catkl_min_baseline: env_usize("ANNO_MUXER_CATKL_MIN_BASELINE", 40) as u64,
        catkl_min_recent: env_usize("ANNO_MUXER_CATKL_MIN_RECENT", 20) as u64,
        catkl_weight: env_f64("ANNO_MUXER_CATKL_WEIGHT", 0.0).max(0.0),

        // CUSUM uses a categorical log-likelihood ratio stream over the recent window.
        max_cusum: env_f64_opt("ANNO_MUXER_MAX_CUSUM"),
        cusum_alpha: env_f64("ANNO_MUXER_CUSUM_ALPHA", 1e-3).max(0.0),
        cusum_min_baseline: env_usize("ANNO_MUXER_CUSUM_MIN_BASELINE", 40) as u64,
        cusum_min_recent: env_usize("ANNO_MUXER_CUSUM_MIN_RECENT", 20) as u64,
        // 4-vector over muxer’s outcome categories:
        // `[ok_clean, ok_soft_junk, ok_hard_junk, fail]` (will be normalized).
        cusum_alt_p: env_simplex4_opt("ANNO_MUXER_CUSUM_ALT_P"),
        cusum_weight: env_f64("ANNO_MUXER_CUSUM_WEIGHT", 0.0).max(0.0),

        ..MabConfig::default()
    }
}

/// Resolve muxer drift config from env (used by monitored selection).
pub fn drift_config_from_env(metric: DriftMetric) -> DriftConfig {
    DriftConfig {
        metric,
        tol: env_f64("ANNO_MUXER_DRIFT_TOL", 1e-12).abs().max(1e-15),
        min_baseline: env_usize("ANNO_MUXER_DRIFT_MIN_BASELINE", 20) as u64,
        min_recent: env_usize("ANNO_MUXER_DRIFT_MIN_RECENT", 10) as u64,
    }
}

fn drift_metric_from_env(name: &str, default: DriftMetric) -> DriftMetric {
    let Ok(v) = std::env::var(name) else {
        return default;
    };
    let t = v.trim().to_ascii_lowercase();
    if t.is_empty() {
        return default;
    }
    match t.as_str() {
        "hellinger" | "h" => DriftMetric::Hellinger,
        "rao" | "fr" | "fisher-rao" | "fisher_rao" => DriftMetric::Rao,
        "js" | "jensen-shannon" | "jensenshannon" | "jensen_shannon" => DriftMetric::JensenShannon,
        _ => default,
    }
}

#[cfg(test)]
mod mab_env_parse_tests {
    use super::*;

    #[test]
    fn env_simplex4_opt_parses_and_normalizes() {
        let p = parse_simplex4_csv("1, 1, 2, 0").expect("parsed");
        let s: f64 = p.iter().sum();
        assert!((s - 1.0).abs() < 1e-12, "sum={}", s);
        assert!(p.iter().all(|x| x.is_finite() && *x >= 0.0));
    }

    #[test]
    fn env_simplex4_opt_rejects_wrong_len_or_negative() {
        assert!(parse_simplex4_csv("1,2,3").is_none());
        assert!(parse_simplex4_csv("1, 1, -1, 1").is_none());
    }
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

#[cfg(feature = "eval")]
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
#[cfg(feature = "eval")]
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

// =============================================================================
// Contextual feature extraction (for LinUCB integration)
// =============================================================================
//
// Theoretical motivation: the objective manifold framework shows that in the
// non-contextual regime, estimation error and detection delay are proportional
// (D_eff = K-1, all objectives collapse to one parameter).  Adding context
// features creates the "contextual revival" -- the design measure gains spatial
// dimensions and objectives genuinely diverge, enabling the bandit to learn
// "biomedical text prefers BERT ONNX" without separate per-facet histories.
//
// Feature design: one-hot indicators for language/domain families plus a bias
// term.  All computable in microseconds from DatasetId metadata.  The bias
// ensures LinUCB learns a per-arm intercept even when all indicators are zero.

/// Fixed feature dimension for the contextual bandit.
pub const CONTEXT_DIM: usize = 8;

/// Build a context feature vector from dataset metadata.
///
/// Layout (dim=8):
/// ```text
/// [0] bias (always 1.0)
/// [1] lang_en (1.0 if language is "en")
/// [2] lang_multilingual (1.0 if language is "mul" or "mixed")
/// [3] domain_biomedical (1.0 if domain contains "biomedical" or "clinical")
/// [4] domain_news (1.0 if domain contains "news")
/// [5] domain_social (1.0 if domain contains "social")
/// [6] domain_science (1.0 if domain contains "science" or "technical")
/// [7] num_entity_types_normalized (|entity_types| / 20.0, clamped to [0,1])
/// ```
///
/// The feature vector is deterministic given the dataset set and is designed
/// so that LinUCB can learn generalizable patterns across slices (e.g.
/// "Cyrillic + biomedical => prefer ONNX") without combinatorial facet explosion.
#[cfg(feature = "eval")]
pub fn context_features(datasets: &[DatasetId]) -> [f64; CONTEXT_DIM] {
    let mut f = [0.0f64; CONTEXT_DIM];
    f[0] = 1.0; // bias

    if datasets.is_empty() {
        return f;
    }

    // Aggregate language signals across datasets in this batch.
    let mut n_en = 0u32;
    let mut n_mul = 0u32;
    let mut total_entity_types = 0u32;

    for d in datasets {
        let lang = d.language();
        if lang == "en" {
            n_en += 1;
        }
        if lang == "mul" || lang == "mixed" {
            n_mul += 1;
        }

        let dom = d.domain();
        if dom.contains("biomedical") || dom.contains("clinical") {
            f[3] += 1.0;
        }
        if dom.contains("news") {
            f[4] += 1.0;
        }
        if dom.contains("social") {
            f[5] += 1.0;
        }
        if dom.contains("scien") || dom.contains("technical") {
            f[6] += 1.0;
        }

        total_entity_types += d.entity_types().len() as u32;
    }

    let n = datasets.len() as f64;
    f[1] = (n_en as f64) / n;
    f[2] = (n_mul as f64) / n;
    // Normalize domain indicators to [0, 1] fractions.
    f[3] /= n;
    f[4] /= n;
    f[5] /= n;
    f[6] /= n;
    // Entity type richness (normalized).
    f[7] = ((total_entity_types as f64) / n / 20.0).min(1.0);

    f
}

#[cfg(all(test, feature = "eval"))]
mod tests {
    use super::*;
    use crate::eval::loader::DatasetId;

    #[test]
    fn test_context_features_smoke() {
        let f = context_features(&[DatasetId::Wnut17]);
        assert!((f[0] - 1.0).abs() < 1e-9, "bias should be 1.0");
        assert!(f[1] > 0.0, "Wnut17 is English");
        // Social media dataset
        assert!(f[5] > 0.0, "Wnut17 is social media: {:?}", f);
    }

    #[test]
    fn test_context_features_empty() {
        let f = context_features(&[]);
        assert!((f[0] - 1.0).abs() < 1e-9, "bias always 1.0");
        for x in f.iter().skip(1).take(CONTEXT_DIM - 1) {
            assert!(x.abs() < 1e-9, "empty datasets -> zero features");
        }
    }

    #[test]
    fn test_context_features_mixed_datasets() {
        let f = context_features(&[DatasetId::Wnut17, DatasetId::GENIA]);
        assert!((f[0] - 1.0).abs() < 1e-9);
        // One of two is English
        assert!(f[1] > 0.0 && f[1] <= 1.0);
        // GENIA is biomedical
        assert!(f[3] > 0.0, "GENIA should trigger biomedical: {:?}", f);
    }

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
