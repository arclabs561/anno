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

/// Deterministically pick up to `k` "unseen" arms for novelty/coverage.
///
/// An arm is "unseen" if `observed_calls(arm) == 0` (i.e., no observed evaluations in the current
/// slice). This intentionally ignores priors/smoothing.
///
/// This helper is domain-agnostic: callers provide the observation function.
pub fn novelty_pick_unseen<F>(
    seed: u64,
    arms: &[String],
    k: usize,
    novelty_enabled: bool,
    mut observed_calls: F,
) -> Vec<String>
where
    F: FnMut(&str) -> u64,
{
    if !novelty_enabled || k == 0 || arms.is_empty() {
        return Vec::new();
    }
    let mut unseen: Vec<String> = arms
        .iter()
        .filter(|b| observed_calls(b.as_str()) == 0)
        .cloned()
        .collect();
    if unseen.is_empty() {
        return Vec::new();
    }
    // Deterministic shuffle by seed.
    unseen.sort_by_key(|b| stable_hash64(seed ^ 0x4E4F_5645, b)); // "NOVE"
    unseen.truncate(k.min(unseen.len()));
    unseen
}

/// Planned policy result for a single selection step.
///
/// This is intentionally “muxer-side”: it only depends on observed stats, novelty, and guardrails,
/// not on task semantics.
#[derive(Debug, Clone)]
pub struct PolicyPlan {
    /// Arms to pre-pick for novelty/coverage (slice-unseen).
    pub prechosen: Vec<String>,
    /// Arms eligible for the algorithmic selector after applying observed guardrails.
    pub eligible: Vec<String>,
    /// If true, the observed guardrail filtered all remaining arms and requested stop-early.
    pub stop_early: bool,
}

/// Compute the muxer policy plan: novelty pre-picks + observed latency guardrail.
///
/// Returns:
/// - `prechosen`: up to `k` unseen arms (deterministic).
/// - `eligible`: remaining arms after removing `prechosen` and applying observed guardrails.
/// - `stop_early`: if true, the caller should not pick additional arms beyond `prechosen`.
pub fn policy_plan_observed<F>(
    seed: u64,
    arms: &[String],
    k: usize,
    novelty_enabled: bool,
    guard: LatencyGuardrail,
    mut observed: F,
) -> PolicyPlan
where
    F: FnMut(&str) -> (u64, u64),
{
    let prechosen = novelty_pick_unseen(seed, arms, k, novelty_enabled, |b| observed(b).0);

    let remaining: Vec<String> = arms
        .iter()
        .filter(|b| !prechosen.contains(*b))
        .cloned()
        .collect();

    let (eligible, stop_early) =
        guardrail_filter_observed_elapsed(seed ^ 0x504C_414E, &remaining, guard, observed); // "PLAN"

    PolicyPlan {
        prechosen,
        eligible,
        stop_early,
    }
}

/// Result of filling up to `k` picks using the shared policy pipeline.
#[derive(Debug, Clone)]
pub struct PolicyFill {
    /// Final chosen arms (order-preserving).
    pub chosen: Vec<String>,
    /// The policy plan computed for this step.
    pub plan: PolicyPlan,
    /// The eligible set that the algorithm selector actually saw.
    pub eligible_used: Vec<String>,
    /// True if the guardrail filtered all arms and we fell back to the unguarded remaining set.
    pub fallback_used: bool,
    /// True if the guardrail requested stop-early and we honored it.
    pub stopped_early: bool,
}

/// Shared muxer “policy pipeline”: novelty → observed guardrail → fill the remaining \(k\).
///
/// This helper centralizes the last bit of glue so CI harness and CLI cannot drift.
///
/// Semantics note:
/// - If the guardrail requests stop-early but we have **zero** picks so far, we **fall back**
///   to the unguarded remaining set so callers don't accidentally pick nothing.
pub fn policy_fill_k_observed_with<F, P>(
    seed: u64,
    arms: &[String],
    k: usize,
    novelty_enabled: bool,
    guard: LatencyGuardrail,
    mut observed: F,
    mut pick_rest: P,
) -> PolicyFill
where
    F: FnMut(&str) -> (u64, u64),
    P: FnMut(&[String], usize) -> Vec<String>,
{
    let plan = policy_plan_observed(seed, arms, k, novelty_enabled, guard, &mut observed);
    let mut chosen = plan.prechosen.clone();

    if chosen.len() >= k {
        return PolicyFill {
            chosen,
            plan,
            eligible_used: Vec::new(),
            fallback_used: false,
            stopped_early: false,
        };
    }

    // If guardrail says “stop early”, only honor it if we already picked something.
    if plan.stop_early && !chosen.is_empty() {
        return PolicyFill {
            chosen,
            eligible_used: Vec::new(),
            plan,
            fallback_used: false,
            stopped_early: true,
        };
    }

    // If the guardrail filtered everything:
    // - If we have picks already and allow_fewer, stop.
    // - Otherwise, fall back to the unguarded remaining set so we can still pick something.
    let mut eligible_used = plan.eligible.clone();
    let mut fallback_used = false;
    let mut stopped_early = false;
    if eligible_used.is_empty() {
        if guard.allow_fewer && !chosen.is_empty() {
            stopped_early = true;
            return PolicyFill {
                chosen,
                eligible_used,
                plan,
                fallback_used,
                stopped_early,
            };
        }
        eligible_used = arms
            .iter()
            .filter(|b| !chosen.contains(*b))
            .cloned()
            .collect();
        fallback_used = true;
    }

    let remaining_k = k.saturating_sub(chosen.len());
    if remaining_k > 0 && !eligible_used.is_empty() {
        // Defensive: the algorithm picker is expected to return <= remaining_k picks from
        // `eligible_used`, without duplicates. Enforce those invariants here so callers can't
        // accidentally violate the contract and create surprising output.
        let rest = pick_rest(&eligible_used, remaining_k);
        for b in rest {
            if chosen.len() >= k {
                break;
            }
            if !eligible_used.contains(&b) {
                continue;
            }
            if chosen.contains(&b) {
                continue;
            }
            chosen.push(b);
        }
    }

    PolicyFill {
        chosen,
        plan,
        eligible_used,
        fallback_used,
        stopped_early,
    }
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

/// Apply the latency guardrail using *observed* (non-smoothed) stats.
///
/// This is a muxer-side routing helper: it keeps guardrails meaningful when prior smoothing is
/// enabled by ensuring `require_measured` and `max_mean_ms` are based on real observations.
///
/// Returns `(eligible_arms, stop_early)`.
pub fn guardrail_filter_observed<F>(
    seed: u64,
    arms: &[String],
    guard: LatencyGuardrail,
    mut observed: F,
) -> (Vec<String>, bool)
where
    F: FnMut(&str) -> (u64, f64),
{
    let Some(max_ms) = guard.max_mean_ms else {
        return (arms.to_vec(), false);
    };
    let mut eligible: Vec<String> = arms
        .iter()
        .filter(|b| {
            let (calls, mean_ms) = observed(b.as_str());
            if guard.require_measured && calls == 0 {
                return false;
            }
            mean_ms <= max_ms
        })
        .cloned()
        .collect();
    eligible.sort_by_key(|b| stable_hash64(seed ^ 0x4755_4152, b)); // "GUAR"

    if eligible.is_empty() {
        if guard.allow_fewer {
            return (Vec::new(), true);
        }
        return (arms.to_vec(), false);
    }

    (eligible, false)
}

/// Like `guardrail_filter_observed`, but the observation function returns the raw elapsed-ms sum.
///
/// This centralizes mean-latency calculation so callers only provide raw counters.
pub fn guardrail_filter_observed_elapsed<F>(
    seed: u64,
    arms: &[String],
    guard: LatencyGuardrail,
    mut observed: F,
) -> (Vec<String>, bool)
where
    F: FnMut(&str) -> (u64, u64),
{
    guardrail_filter_observed(seed, arms, guard, |b| {
        let (calls, elapsed_ms_sum) = observed(b);
        let mean_ms = if calls == 0 {
            0.0
        } else {
            (elapsed_ms_sum as f64) / (calls as f64)
        };
        (calls, mean_ms)
    })
}

/// Pick one arm for "worst-first" regression hunting.
///
/// Policy:
/// - Prefer an *unseen* arm first (observed_calls == 0) to saturate coverage.
/// - Otherwise pick the arm with the highest "worse" score:
///   \(score = hard_weight * hard_junk + soft_weight * soft_junk + exploration_c * sqrt(ln(total_calls) / calls)\).
///
/// This helper is domain-agnostic: callers supply observed calls and summary rates/calls.
#[derive(Debug, Clone, Copy)]
pub struct WorstFirstConfig {
    /// Exploration coefficient for the worst-first score.
    pub exploration_c: f64,
    /// Weight applied to hard-junk (instability) rate.
    pub hard_weight: f64,
    /// Weight applied to soft-junk rate.
    pub soft_weight: f64,
}

/// Pick one arm for "worst-first" regression hunting.
pub fn worst_first_pick_one<FObs, FSum>(
    seed: u64,
    remaining: &[String],
    cfg: WorstFirstConfig,
    mut observed_calls: FObs,
    mut summary: FSum,
) -> Option<(String, bool)>
where
    FObs: FnMut(&str) -> u64,
    FSum: FnMut(&str) -> (u64, f64, f64), // (calls, hard_junk_rate, soft_junk_rate)
{
    if remaining.is_empty() {
        return None;
    }

    // Explore unseen arms first (stable order by deterministic hash).
    let mut unseen: Vec<String> = remaining
        .iter()
        .filter(|b| observed_calls(b.as_str()) == 0)
        .cloned()
        .collect();
    if !unseen.is_empty() {
        unseen.sort_by_key(|b| stable_hash64(seed ^ 0x574F_5253, b)); // "WORS"
        return Some((unseen[0].clone(), true));
    }

    // Otherwise score by "worse" objective + exploration.
    let mut total_calls_f: f64 = 0.0;
    let mut scored: Vec<(f64, String, u64, f64, f64)> = Vec::new();
    for b in remaining {
        let (calls_u64, hard, soft) = summary(b.as_str());
        let calls = (calls_u64 as f64).max(1.0);
        total_calls_f += calls;
        scored.push((0.0, b.clone(), calls_u64, hard, soft));
    }
    let total_calls_f = total_calls_f.max(1.0);

    for row in &mut scored {
        let calls = (row.2 as f64).max(1.0);
        let exploration = cfg.exploration_c * ((total_calls_f.ln() / calls).sqrt());
        let score = cfg.hard_weight * row.3 + cfg.soft_weight * row.4 + exploration;
        row.0 = score;
    }

    scored.sort_by(|a, b| {
        b.0.total_cmp(&a.0)
            .then_with(|| {
                stable_hash64(seed ^ 0x574F_5253, &a.1)
                    .cmp(&stable_hash64(seed ^ 0x574F_5253, &b.1))
            })
            .then_with(|| a.1.cmp(&b.1))
    });
    let pick = scored.first().map(|r| r.1.clone())?;
    Some((pick, false))
}

/// Pick up to `k` arms for worst-first regression hunting (without replacement).
///
/// Returns a list of `(arm, explore_first)` pairs in selection order.
pub fn worst_first_pick_k<FObs, FSum>(
    seed: u64,
    arms: &[String],
    k: usize,
    cfg: WorstFirstConfig,
    mut observed_calls: FObs,
    mut summary: FSum,
) -> Vec<(String, bool)>
where
    FObs: FnMut(&str) -> u64,
    FSum: FnMut(&str) -> (u64, f64, f64),
{
    select_k_without_replacement_by_with_meta(seed, arms, k, |seed_round, remaining, _k| {
        worst_first_pick_one(
            seed_round,
            remaining,
            cfg,
            |b| observed_calls(b),
            |b| summary(b),
        )
        .map(|(b, explore_first)| vec![(b, explore_first)])
        .unwrap_or_default()
    })
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

    #[test]
    fn test_guardrail_filter_observed_elapsed_require_measured() {
        let guard = LatencyGuardrail {
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
        let guard = LatencyGuardrail {
            max_mean_ms: Some(1.0),
            allow_fewer: true,
            require_measured: true,
        };
        let arms = vec!["a".to_string(), "b".to_string()];
        // Only `a` is unseen -> it will be prechosen; remaining `b` is unmeasured and should be
        // filtered by guardrail, causing stop_early.
        let plan = policy_plan_observed(0, &arms, 1, true, guard, |arm| {
            if arm == "a" {
                (0, 0)
            } else {
                (0, 0)
            }
        });
        assert_eq!(plan.prechosen, vec!["a".to_string()]);
        assert!(plan.eligible.is_empty());
        assert!(plan.stop_early);
    }

    #[test]
    fn test_policy_fill_falls_back_if_stop_early_would_pick_nothing() {
        let guard = LatencyGuardrail {
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
        assert_eq!(fill.chosen, vec!["a".to_string()]);
        assert!(fill.fallback_used);
        assert!(!fill.stopped_early);
    }

    #[test]
    fn test_policy_fill_novelty_then_stop_early_does_not_fallback() {
        // Difficult seam: novelty can create a non-empty chosen set, after which the observed
        // guardrail can stop-early. In that case we must *not* fall back and pick more.
        let guard = LatencyGuardrail {
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
        let guard = LatencyGuardrail {
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
        // Difficult seam: guardrail filters all arms (stop_early), but because we have no picks
        // yet the policy should fall back to remaining and still pick something.
        let guard = LatencyGuardrail {
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
        assert_eq!(fill.chosen.len(), 1);
        assert!(fill.fallback_used);
        assert!(!fill.stopped_early);
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
            let guard = LatencyGuardrail {
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
                let calls = (x % 4) as u64;
                x = lcg(x);
                let mean_bucket = (x % 3) as u64; // 0,1,2
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

    #[test]
    fn test_select_k_without_replacement_by_with_meta_preserves_meta_and_ignores_unknown() {
        let items = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let out = select_k_without_replacement_by_with_meta(0, &items, 3, |_seed, _rem, _k| {
            vec![
                ("b".to_string(), 11u32),
                ("x".to_string(), 99u32), // unknown -> ignored
                ("a".to_string(), 22u32),
                ("b".to_string(), 33u32), // duplicate -> ignored
            ]
        });
        assert_eq!(
            out,
            vec![("b".to_string(), 11u32), ("a".to_string(), 22u32)]
        );
    }

    #[test]
    fn test_select_k_without_replacement_by_with_meta_breaks_on_no_progress() {
        let items = vec!["a".to_string(), "b".to_string()];
        let out = select_k_without_replacement_by_with_meta(0, &items, 2, |_seed, _rem, _k| {
            vec![("x".to_string(), 0u8)]
        });
        assert!(out.is_empty());
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

/// Shared “select K without replacement” driver.
///
/// This is intended to reduce duplicated “remaining/chosen” plumbing across selection strategies.
/// Callers provide a picker that can return up to `k_remaining` candidates for the current
/// remaining set; the driver enforces de-duplication and ordering.
pub fn select_k_without_replacement_by_with_meta<F, M>(
    seed: u64,
    items: &[String],
    k: usize,
    mut pick: F,
) -> Vec<(String, M)>
where
    F: FnMut(u64, &[String], usize) -> Vec<(String, M)>,
{
    if k == 0 || items.is_empty() {
        return Vec::new();
    }
    let mut remaining: Vec<String> = items.to_vec();
    let mut out: Vec<(String, M)> = Vec::new();

    while !remaining.is_empty() && out.len() < k {
        let need = k - out.len();
        let batch = pick(seed ^ (out.len() as u64), &remaining, need);
        if batch.is_empty() {
            break;
        }
        let mut made_progress = false;
        for (b, meta) in batch {
            if out.len() >= k {
                break;
            }
            if !remaining.contains(&b) {
                continue;
            }
            if out.iter().any(|(x, _)| x == &b) {
                continue;
            }
            remaining.retain(|x| x != &b);
            out.push((b, meta));
            made_progress = true;
        }
        if !made_progress {
            break;
        }
    }
    out
}

/// Convenience wrapper over `select_k_without_replacement_by_with_meta` when no meta is needed.
pub fn select_k_without_replacement_by<F>(
    seed: u64,
    items: &[String],
    k: usize,
    mut pick: F,
) -> Vec<String>
where
    F: FnMut(u64, &[String], usize) -> Vec<String>,
{
    select_k_without_replacement_by_with_meta(seed, items, k, |s, rem, need| {
        pick(s, rem, need).into_iter().map(|b| (b, ())).collect()
    })
    .into_iter()
    .map(|(b, _)| b)
    .collect()
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
