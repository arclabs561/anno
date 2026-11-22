//! Schedule comparable outstanding failures and bounded checks for recurrent failures.
use super::*;
use crate::eval::history::EvalHistoryEntry;
use crate::eval::provenance::EvaluationScheduling;
use crate::eval::task_evaluator::TaskEvalResult;
use std::collections::{HashMap, VecDeque};
use std::io::{BufRead, BufReader};

#[derive(Clone, Debug, serde::Serialize)]
pub(super) struct RetryCell {
    pub task: Task,
    pub dataset: DatasetId,
    pub backend: String,
    observation_id: String,
    reason: Reason,
    #[serde(skip_serializing_if = "Option::is_none")]
    recovered_episodes: Option<usize>,
    /// Weighted failure frequency over the bounded window used for the draw.
    risk: f64,
    observations: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize)]
#[serde(rename_all = "snake_case")]
enum Reason {
    OutstandingFailure,
    RecurrentAfterRecovery,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Health {
    HardFailure,
    LowScore,
    Healthy,
    Unscored,
}

#[derive(Default)]
struct Recent {
    outcomes: VecDeque<(Health, String)>,
    pending: Option<(Health, String)>,
}

impl Recent {
    fn observe(&mut self, status: Health, id: String, cap: usize) {
        // A repeated receipt is still one attempt.
        if self.outcomes.iter().any(|(_, seen)| seen == &id) {
            return;
        }
        self.outcomes.push_back((status, id.clone()));
        if self.outcomes.len() > cap {
            self.outcomes.pop_front();
        }
        match status {
            Health::HardFailure | Health::LowScore => self.pending = Some((status, id)),
            Health::Healthy => self.pending = None,
            Health::Unscored if matches!(self.pending, Some((Health::HardFailure, _))) => {
                self.pending = None
            }
            Health::Unscored => {}
        }
    }

    fn risk(&self, config: WorstFirstConfig) -> f64 {
        let hard = self
            .outcomes
            .iter()
            .filter(|(v, _)| *v == Health::HardFailure)
            .count();
        let measured = self
            .outcomes
            .iter()
            .filter(|(v, _)| matches!(v, Health::Healthy | Health::LowScore))
            .count();
        let low = self
            .outcomes
            .iter()
            .filter(|(v, _)| *v == Health::LowScore)
            .count();
        config.hard_weight * hard as f64 / self.outcomes.len().max(1) as f64
            + config.soft_weight * low as f64 / measured.max(1) as f64
    }

    // Count distinct episodes closed by an observed recovery, not consecutive failures
    // in a single outage. For quality, an unmeasured row says nothing about recovery.
    fn recovered_risk(&self, kind: Health, weight: f64) -> Option<(f64, usize, &str)> {
        let mut last_failure = None;
        let mut evidence = None;
        let mut episodes = 0;
        let mut attempts = 0;
        for (status, id) in &self.outcomes {
            if kind == Health::LowScore && matches!(status, Health::HardFailure | Health::Unscored)
            {
                continue;
            }
            attempts += 1;
            if *status == kind {
                last_failure = Some(id.as_str());
            } else if let Some(failed) = last_failure.take() {
                episodes += 1;
                evidence = Some(failed);
            }
        }
        if episodes < 2 || last_failure.is_some() || weight <= 0.0 {
            return None;
        }
        // Four healthy pseudo-observations temper short histories. This is a routing
        // weight, not a calibrated probability of the next attempt failing.
        Some((
            weight * episodes as f64 / (attempts + 4) as f64,
            episodes,
            evidence?,
        ))
    }

    fn selection(
        &self,
        seed: u64,
        config: WorstFirstConfig,
    ) -> Option<(Reason, f64, Option<usize>, &str)> {
        if let Some((kind, id)) = &self.pending {
            let weight = if *kind == Health::HardFailure {
                config.hard_weight
            } else {
                config.soft_weight
            };
            if weight > 0.0 {
                return Some((
                    Reason::OutstandingFailure,
                    self.risk(config),
                    None,
                    id.as_str(),
                ));
            }
        }
        // Reserve three seed buckets out of four for ordinary worst-first exploration
        // when there are no eligible outstanding failures.
        if mh::stable_hash64(seed, "matrix-recovered-check") % 4 != 0 {
            return None;
        }
        [
            self.recovered_risk(Health::HardFailure, config.hard_weight),
            self.recovered_risk(Health::LowScore, config.soft_weight),
        ]
        .into_iter()
        .flatten()
        .max_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(risk, episodes, id)| (Reason::RecurrentAfterRecovery, risk, Some(episodes), id))
    }
}

// Exponential race: among independent draws the lowest time wins with probability
// proportional to its rate. A stable key makes the order independent of map iteration.
fn draw(seed: u64, key: &str, risk: f64) -> f64 {
    let bits = mh::stable_hash64(seed, key) >> 12;
    let uniform = (bits as f64 + 0.5) / ((1_u64 << 52) as f64);
    -uniform.ln() / risk
}

pub(super) fn next_cell(
    loader: &DatasetLoader,
    cached: bool,
    seed: u64,
    window_cap: usize,
    fixed_datasets: Option<&str>,
    fixed_backends: Option<&[String]>,
    pinned_backends: Option<&[String]>,
) -> Option<RetryCell> {
    let file = std::fs::File::open(eval_history_jsonl_path()).ok()?;
    let config = worst_first_config_from_env();
    let mut recent = HashMap::new();
    let max_examples = Some(max_examples_per_dataset());

    for line in BufReader::new(file).lines() {
        let line = line.ok()?;
        let Ok(entry) = serde_json::from_str::<EvalHistoryEntry>(&line) else {
            continue;
        };
        let (Some(observation_id), Some(cohort), Some(metadata)) = (
            entry.observation_id,
            entry.policy_cohort_key,
            entry.metadata,
        ) else {
            continue;
        };
        let Ok(result) = serde_json::from_str::<TaskEvalResult>(&metadata) else {
            continue;
        };
        if result.is_skipped()
            || (!result.success
                && matches!(
                    result.provenance.runtime.scheduling,
                    EvaluationScheduling::NotRun { .. }
                ))
            || Some(cohort)
                != policy_cohort_key(result.task, result.dataset, seed, max_examples, cached)
        {
            continue;
        }
        let status = if !result.success {
            Health::HardFailure
        } else {
            match result.primary_f1().filter(|v| v.is_finite()) {
                Some(score) if score < junk_f1_threshold(result.task) => Health::LowScore,
                Some(_) => Health::Healthy,
                None => Health::Unscored,
            }
        };
        recent
            .entry((result.task, result.dataset, result.backend))
            .or_insert_with(Recent::default)
            .observe(status, observation_id, window_cap.max(1));
    }
    let mut ranked = Vec::new();
    for ((task, dataset, backend), history) in recent {
        let Some((reason, risk, recovered_episodes, id)) = history.selection(seed, config) else {
            continue;
        };
        if !risk.is_finite() || risk <= 0.0 {
            continue;
        }
        let key = format!("{}@@{dataset:?}@@{backend}", task.code());
        let cell = RetryCell {
            task,
            dataset,
            backend,
            observation_id: id.to_owned(),
            reason,
            recovered_episodes,
            risk,
            observations: history.outcomes.len(),
        };
        ranked.push((reason, draw(seed, &key, risk), key, cell));
    }
    if ranked.is_empty() {
        return None;
    }
    ranked.sort_by(|(a_reason, a, a_key, _), (b_reason, b, b_key, _)| {
        a_reason
            .cmp(b_reason)
            .then_with(|| a.total_cmp(b))
            .then_with(|| a_key.cmp(b_key))
    });
    let tasks = candidate_tasks_for_run(loader, cached);
    for (_, _, _, cell) in ranked {
        let RetryCell {
            task,
            dataset,
            ref backend,
            ..
        } = cell;
        if !tasks.contains(&task)
            || !backend_candidates(SampleStrategy::WorstFirst, &[task]).contains(backend)
            || !TaskEvaluator::is_backend_compatible(backend, dataset)
            || fixed_backends
                .or(pinned_backends)
                .is_some_and(|bs| !bs.contains(backend))
        {
            continue;
        }
        if let Some(raw) = fixed_datasets {
            let fixed: Option<Vec<_>> = raw
                .split(',')
                .map(|s| s.trim().parse::<DatasetId>().ok())
                .collect();
            let fixed = fixed?;
            if !fixed.contains(&dataset)
                || !fixed
                    .iter()
                    .all(|ds| TaskEvaluator::is_backend_compatible(backend, *ds))
            {
                continue;
            }
        } else if !pinned_dataset_candidates(loader, &[task], cached).contains(&dataset) {
            continue;
        }
        if dataset_is_usable(loader, dataset, cached) {
            return Some(cell);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn more_frequent_failures_get_more_draws_and_equal_risks_remain_reachable() {
        let config = WorstFirstConfig {
            exploration_c: 0.0,
            hard_weight: 1.0,
            soft_weight: 0.0,
        };
        let history = |good: usize, bad: usize| {
            let mut window = Recent::default();
            for i in 0..good {
                window.observe(Health::Healthy, format!("good-{i}"), 30);
            }
            for i in 0..bad {
                window.observe(Health::HardFailure, format!("bad-{i}"), 30);
            }
            window
        };
        let rare = history(29, 1).risk(config);
        let frequent = history(2, 8).risk(config);
        assert_eq!(rare, 1.0 / 30.0);
        assert_eq!(frequent, 0.8);
        let count = |a, b| {
            (0..4096)
                .filter(|&seed| draw(seed, "A", a) < draw(seed, "B", b))
                .count()
        };
        let more = count(frequent, rare);
        assert!((3750..4050).contains(&more), "frequent first {more}/4096");
        assert_eq!(more, count(frequent, rare));
        assert!((1800..2300).contains(&count(rare, rare)));
    }

    #[test]
    fn rechecks_close_on_recovery_and_cap_retained_evidence() {
        let mut history = Recent::default();
        history.observe(Health::HardFailure, "fail".into(), 2);
        history.observe(Health::Unscored, "completed".into(), 2);
        assert!(history.pending.is_none());
        history.observe(Health::LowScore, "low".into(), 2);
        history.observe(Health::Unscored, "unmeasured".into(), 2);
        assert_eq!(
            history.pending.as_ref().map(|(_, id)| id.as_str()),
            Some("low")
        );
        history.observe(Health::Healthy, "resolved".into(), 2);
        assert!(history.pending.is_none());
        assert_eq!(history.outcomes.len(), 2);
    }

    #[test]
    fn recurrence_requires_separate_failures_and_observed_recoveries() {
        use Health::{HardFailure as Bad, Healthy as Ok, LowScore, Unscored};
        let make = |rows: &[(Health, &str)]| {
            let mut h = Recent::default();
            for &(status, id) in rows {
                h.observe(status, id.into(), 20);
            }
            h
        };
        for rows in [
            &[(Bad, "a"), (Ok, "b")][..],
            &[(Bad, "a"), (Bad, "b"), (Ok, "c")],
            &[(Bad, "a"), (Ok, "b"), (Bad, "a"), (Ok, "b")],
        ] {
            assert!(make(rows).recovered_risk(Bad, 1.0).is_none());
        }
        let h = make(&[(Bad, "a"), (Ok, "b"), (Bad, "c"), (Ok, "d")]);
        assert_eq!(h.recovered_risk(Bad, 1.0), Some((0.25, 2, "c")));
        assert!(h.recovered_risk(Bad, 0.0).is_none());
        // An unweighted low score can still prove recovery from an execution failure.
        let hard_only = WorstFirstConfig {
            exploration_c: 0.0,
            hard_weight: 1.0,
            soft_weight: 0.0,
        };
        let intermittent = make(&[(Bad, "a"), (LowScore, "b"), (Bad, "c"), (LowScore, "d")]);
        let sample = (0..100).find_map(|seed| intermittent.selection(seed, hard_only));
        assert_eq!(
            sample,
            Some((Reason::RecurrentAfterRecovery, 0.25, Some(2), "c"))
        );

        let unmeasured = make(&[(LowScore, "a"), (Unscored, "b"), (LowScore, "c"), (Ok, "d")]);
        assert!(unmeasured.recovered_risk(LowScore, 1.0).is_none());
    }

    #[test]
    fn prospective_recurrence_draws_keep_coverage_and_find_more_withheld_failures() {
        let cfg = WorstFirstConfig {
            exploration_c: 0.0,
            hard_weight: 1.0,
            soft_weight: 0.0,
        };
        let names = ["recurrent", "isolated", "healthy", "cold"];
        let mut histories = names.map(|_| Recent::default());
        for attempt in 0..20 {
            for (cell, history) in histories[..3].iter_mut().enumerate() {
                let failed = (cell == 0 && attempt % 2 == 0) || (cell == 1 && attempt == 0);
                history.observe(
                    if failed {
                        Health::HardFailure
                    } else {
                        Health::Healthy
                    },
                    attempt.to_string(),
                    20,
                );
            }
        }
        // Same held-out draws in three worlds: recurrence persists, disappears, or
        // moves to cell 1. The latter two are controls for past/future drift.
        let risks = [[55, 5, 5, 5], [5; 4], [5, 55, 5, 5]];
        let mut results = [(0i32, 0i32); 3];
        let mut targeted = 0;
        let mut visits = [0; 4];
        for seed in 0..4096 {
            // Separate deterministic streams for selection and the unseen next row.
            let future = names.map(|name| mh::stable_hash64(seed ^ 0x7B77, name) % 100);
            let uniform = (mh::stable_hash64(seed, "uniform cell") % 4) as usize;
            let selected = histories
                .iter()
                .enumerate()
                .filter_map(|(i, h)| {
                    let (reason, weight, episodes, _) = h.selection(seed, cfg)?;
                    assert_eq!(reason, Reason::RecurrentAfterRecovery);
                    assert_eq!(episodes, Some(10));
                    Some((draw(seed, names[i], weight), i))
                })
                .min_by(|a, b| a.0.total_cmp(&b.0))
                .map(|(_, i)| i);
            targeted += usize::from(selected.is_some());
            let selected = selected.unwrap_or(uniform);
            visits[selected] += 1;
            for (rates, (found, uniform_found)) in risks.iter().zip(&mut results) {
                *found += i32::from(future[selected] < rates[selected]);
                *uniform_found += i32::from(future[uniform] < rates[uniform]);
            }
        }
        eprintln!("withheld (targeted, uniform) by scenario: {results:?}, targeted={targeted}, visits={visits:?}");
        assert!((900..1150).contains(&targeted));
        assert!(visits.iter().all(|&n| n > 600));
        assert!(results[0].0 > results[0].1 + 250);
        assert!((results[1].0 - results[1].1).abs() < 40);
        assert!(results[2].0 < results[2].1);
    }
}
