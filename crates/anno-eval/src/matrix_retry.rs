//! Use the latest comparable observation for each cell to schedule regression rechecks.
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
    /// Weighted failure frequency over the bounded window used for the draw.
    risk: f64,
    observations: usize,
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
    outcomes: VecDeque<Health>,
    pending: Option<(Health, String)>,
}

impl Recent {
    fn observe(&mut self, status: Health, id: String, cap: usize) {
        self.outcomes.push_back(status);
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
            .filter(|&&v| v == Health::HardFailure)
            .count();
        let measured = self
            .outcomes
            .iter()
            .filter(|&&v| matches!(v, Health::Healthy | Health::LowScore))
            .count();
        let low = self
            .outcomes
            .iter()
            .filter(|&&v| v == Health::LowScore)
            .count();
        config.hard_weight * hard as f64 / self.outcomes.len().max(1) as f64
            + config.soft_weight * low as f64 / measured.max(1) as f64
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
        let risk = history.risk(config);
        let Some((kind, observation_id)) = history.pending else {
            continue;
        };
        let weight = if kind == Health::HardFailure {
            config.hard_weight
        } else {
            config.soft_weight
        };
        if weight <= 0.0 || !risk.is_finite() || risk <= 0.0 {
            continue;
        }
        let key = format!("{}@@{dataset:?}@@{backend}", task.code());
        let cell = RetryCell {
            task,
            dataset,
            backend,
            observation_id,
            risk,
            observations: history.outcomes.len(),
        };
        ranked.push((draw(seed, &key, risk), key, cell));
    }
    if ranked.is_empty() {
        return None;
    }
    ranked.sort_by(|(a, a_key, _), (b, b_key, _)| a.total_cmp(b).then_with(|| a_key.cmp(b_key)));
    let tasks = candidate_tasks_for_run(loader, cached);
    for (_, _, cell) in ranked {
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
            for _ in 0..good {
                window.observe(Health::Healthy, String::new(), 30);
            }
            for _ in 0..bad {
                window.observe(Health::HardFailure, String::new(), 30);
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
}
