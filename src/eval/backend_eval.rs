//! Comprehensive backend evaluation framework.
//!
//! Evaluates Pattern and Statistical backends against synthetic and benchmark datasets.
//!
//! # Usage
//!
//! ```rust,no_run
//! use anno::eval::backend_eval::{BackendEvaluator, EvalReport};
//!
//! let evaluator = BackendEvaluator::new();
//! let report = evaluator.run_comprehensive();
//! println!("{}", report.to_markdown());
//! ```

use crate::backends::{HeuristicNER, RegexNER, StackedNER};
use crate::eval::dataset::synthetic;
use crate::eval::dataset::{AnnotatedExample, Difficulty, Domain};
use crate::{Entity, EntityType, Model};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for backend evaluation.
#[derive(Debug, Clone)]
pub struct EvalConfig {
    /// Include Pattern backend in evaluation
    pub include_pattern: bool,
    /// Include Heuristic backend in evaluation
    pub include_heuristic: bool,
    /// Include Stacked backend in evaluation
    pub include_stacked: bool,
    /// Include GLiNER ONNX backend in evaluation (requires `onnx` feature)
    pub include_gliner: bool,
    /// Run per-domain metric breakdown
    pub per_domain: bool,
    /// Run per-difficulty metric breakdown
    pub per_difficulty: bool,
    /// Maximum examples to evaluate (0 = no limit)
    pub max_examples: usize,
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            include_pattern: true,
            include_heuristic: true,
            include_stacked: true,
            include_gliner: cfg!(feature = "onnx"), // Auto-enable if onnx feature present
            per_domain: true,
            per_difficulty: true,
            max_examples: 0,
        }
    }
}

// ============================================================================
// Results
// ============================================================================

/// Metrics for a single evaluation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalMetrics {
    /// Precision (TP / (TP + FP))
    pub precision: f64,
    /// Recall (TP / (TP + FN))
    pub recall: f64,
    /// F1 score (harmonic mean of precision and recall)
    pub f1: f64,
    /// True positive count
    pub true_positives: usize,
    /// False positive count
    pub false_positives: usize,
    /// False negative count
    pub false_negatives: usize,
    /// Evaluation duration in milliseconds
    pub duration_ms: u64,
    /// Number of examples evaluated
    pub examples_evaluated: usize,
}

impl EvalMetrics {
    /// Create new metrics from counts.
    pub fn from_counts(tp: usize, fp: usize, fn_: usize, duration: Duration, n: usize) -> Self {
        let precision = if tp + fp > 0 {
            tp as f64 / (tp + fp) as f64
        } else {
            0.0
        };
        let recall = if tp + fn_ > 0 {
            tp as f64 / (tp + fn_) as f64
        } else {
            0.0
        };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        Self {
            precision,
            recall,
            f1,
            true_positives: tp,
            false_positives: fp,
            false_negatives: fn_,
            duration_ms: duration.as_millis() as u64,
            examples_evaluated: n,
        }
    }
}

/// Results for a single backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendResults {
    /// Backend name
    pub name: String,
    /// Overall metrics
    pub overall: EvalMetrics,
    /// Metrics broken down by domain
    pub by_domain: HashMap<String, EvalMetrics>,
    /// Metrics broken down by difficulty
    pub by_difficulty: HashMap<String, EvalMetrics>,
    /// Metrics broken down by entity type
    pub by_entity_type: HashMap<String, EvalMetrics>,
}

/// Complete evaluation report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalReport {
    /// ISO 8601 timestamp when report was generated
    pub timestamp: String,
    /// Total examples evaluated
    pub total_examples: usize,
    /// Results for each backend
    pub backends: Vec<BackendResults>,
    /// Dataset statistics
    pub dataset_stats: DatasetStats,
}

/// Dataset statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetStats {
    /// Total number of examples
    pub total_examples: usize,
    /// Total number of entities across all examples
    pub total_entities: usize,
    /// Example count by domain
    pub by_domain: HashMap<String, usize>,
    /// Example count by difficulty
    pub by_difficulty: HashMap<String, usize>,
    /// Entity count by entity type
    pub by_entity_type: HashMap<String, usize>,
}

impl EvalReport {
    /// Generate markdown summary.
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();

        md.push_str("# NER Backend Evaluation Report\n\n");
        md.push_str(&format!("**Generated:** {}\n\n", self.timestamp));
        md.push_str(&format!("**Total Examples:** {}\n\n", self.total_examples));

        // Dataset stats
        md.push_str("## Dataset Statistics\n\n");
        md.push_str(&format!(
            "- Total entities: {}\n",
            self.dataset_stats.total_entities
        ));
        md.push_str(&format!(
            "- Domains: {}\n",
            self.dataset_stats.by_domain.len()
        ));
        md.push_str(&format!(
            "- Difficulties: {}\n\n",
            self.dataset_stats.by_difficulty.len()
        ));

        // Overall comparison
        md.push_str("## Overall Results\n\n");
        md.push_str("| Backend | Precision | Recall | F1 | TP | FP | FN | Time (ms) |\n");
        md.push_str("|---------|-----------|--------|----|----|----|----|------------|\n");

        for backend in &self.backends {
            md.push_str(&format!(
                "| {} | {:.1}% | {:.1}% | {:.1}% | {} | {} | {} | {} |\n",
                backend.name,
                backend.overall.precision * 100.0,
                backend.overall.recall * 100.0,
                backend.overall.f1 * 100.0,
                backend.overall.true_positives,
                backend.overall.false_positives,
                backend.overall.false_negatives,
                backend.overall.duration_ms,
            ));
        }
        md.push('\n');

        // Per-domain breakdown
        if !self.backends.is_empty() && !self.backends[0].by_domain.is_empty() {
            md.push_str("## Results by Domain\n\n");

            for backend in &self.backends {
                md.push_str(&format!("### {}\n\n", backend.name));
                md.push_str("| Domain | Precision | Recall | F1 |\n");
                md.push_str("|--------|-----------|--------|----|\n");

                let mut domains: Vec<_> = backend.by_domain.iter().collect();
                domains.sort_by(|a, b| a.0.cmp(b.0));

                for (domain, metrics) in domains {
                    md.push_str(&format!(
                        "| {} | {:.1}% | {:.1}% | {:.1}% |\n",
                        domain,
                        metrics.precision * 100.0,
                        metrics.recall * 100.0,
                        metrics.f1 * 100.0,
                    ));
                }
                md.push('\n');
            }
        }

        // Per-entity-type breakdown
        if !self.backends.is_empty() && !self.backends[0].by_entity_type.is_empty() {
            md.push_str("## Results by Entity Type\n\n");

            for backend in &self.backends {
                md.push_str(&format!("### {}\n\n", backend.name));
                md.push_str("| Entity Type | Precision | Recall | F1 |\n");
                md.push_str("|-------------|-----------|--------|----|\n");

                let mut types: Vec<_> = backend.by_entity_type.iter().collect();
                types.sort_by(|a, b| a.0.cmp(b.0));

                for (entity_type, metrics) in types {
                    md.push_str(&format!(
                        "| {} | {:.1}% | {:.1}% | {:.1}% |\n",
                        entity_type,
                        metrics.precision * 100.0,
                        metrics.recall * 100.0,
                        metrics.f1 * 100.0,
                    ));
                }
                md.push('\n');
            }
        }

        md
    }

    /// Generate HTML report.
    pub fn to_html(&self) -> String {
        let mut html = String::new();

        html.push_str(
            r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NER Backend Evaluation</title>
    <style>
        :root {
            --bg: #0d1117;
            --fg: #c9d1d9;
            --border: #30363d;
            --accent: #58a6ff;
            --green: #3fb950;
            --yellow: #d29922;
            --red: #f85149;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            color: var(--fg);
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }
        h1, h2, h3 { color: var(--accent); }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 1rem 0;
        }
        th, td {
            border: 1px solid var(--border);
            padding: 0.75rem;
            text-align: left;
        }
        th { background: #161b22; }
        tr:hover { background: #161b22; }
        .metric-good { color: var(--green); }
        .metric-ok { color: var(--yellow); }
        .metric-bad { color: var(--red); }
        .summary-card {
            background: #161b22;
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 1rem;
            margin: 1rem 0;
        }
    </style>
</head>
<body>
"#,
        );

        html.push_str("<h1>NER Backend Evaluation Report</h1>\n");
        html.push_str(&format!(
            "<p><strong>Generated:</strong> {}</p>\n",
            self.timestamp
        ));
        html.push_str(&format!(
            "<p><strong>Total Examples:</strong> {}</p>\n",
            self.total_examples
        ));

        // Summary cards
        html.push_str("<div style='display: flex; gap: 1rem; flex-wrap: wrap;'>\n");
        for backend in &self.backends {
            let f1_class = if backend.overall.f1 >= 0.8 {
                "metric-good"
            } else if backend.overall.f1 >= 0.5 {
                "metric-ok"
            } else {
                "metric-bad"
            };

            html.push_str(&format!(
                r#"<div class="summary-card">
    <h3>{}</h3>
    <p class="{}">F1: {:.1}%</p>
    <p>Precision: {:.1}% | Recall: {:.1}%</p>
    <p>Time: {}ms</p>
</div>
"#,
                backend.name,
                f1_class,
                backend.overall.f1 * 100.0,
                backend.overall.precision * 100.0,
                backend.overall.recall * 100.0,
                backend.overall.duration_ms,
            ));
        }
        html.push_str("</div>\n");

        // Overall comparison table
        html.push_str("<h2>Overall Results</h2>\n");
        html.push_str("<table>\n");
        html.push_str("<tr><th>Backend</th><th>Precision</th><th>Recall</th><th>F1</th><th>TP</th><th>FP</th><th>FN</th><th>Time</th></tr>\n");

        for backend in &self.backends {
            html.push_str(&format!(
                "<tr><td>{}</td><td>{:.1}%</td><td>{:.1}%</td><td>{:.1}%</td><td>{}</td><td>{}</td><td>{}</td><td>{}ms</td></tr>\n",
                backend.name,
                backend.overall.precision * 100.0,
                backend.overall.recall * 100.0,
                backend.overall.f1 * 100.0,
                backend.overall.true_positives,
                backend.overall.false_positives,
                backend.overall.false_negatives,
                backend.overall.duration_ms,
            ));
        }
        html.push_str("</table>\n");

        html.push_str("</body></html>");
        html
    }
}

// ============================================================================
// Evaluator
// ============================================================================

/// Backend evaluator.
pub struct BackendEvaluator {
    config: EvalConfig,
}

impl BackendEvaluator {
    /// Create new evaluator with default config.
    pub fn new() -> Self {
        Self {
            config: EvalConfig::default(),
        }
    }

    /// Create evaluator with custom config.
    pub fn with_config(config: EvalConfig) -> Self {
        Self { config }
    }

    /// Run comprehensive evaluation on all synthetic datasets.
    pub fn run_comprehensive(&self) -> EvalReport {
        let examples = synthetic::all_datasets();
        self.evaluate_on(&examples)
    }

    /// Run evaluation on specific domain.
    pub fn run_domain(&self, domain: Domain) -> EvalReport {
        let examples = synthetic::by_domain(domain);
        self.evaluate_on(&examples)
    }

    /// Run evaluation on specific difficulty.
    pub fn run_difficulty(&self, difficulty: Difficulty) -> EvalReport {
        let examples = synthetic::by_difficulty(difficulty);
        self.evaluate_on(&examples)
    }

    /// Run evaluation on technology datasets.
    pub fn run_technology(&self) -> EvalReport {
        let examples = synthetic::technology_dataset();
        self.evaluate_on(&examples)
    }

    /// Run evaluation on healthcare datasets.
    pub fn run_healthcare(&self) -> EvalReport {
        let examples = synthetic::healthcare_dataset();
        self.evaluate_on(&examples)
    }

    /// Run evaluation on custom examples.
    pub fn evaluate_on(&self, examples: &[AnnotatedExample]) -> EvalReport {
        let examples = if self.config.max_examples > 0 && examples.len() > self.config.max_examples
        {
            &examples[..self.config.max_examples]
        } else {
            examples
        };

        let dataset_stats = compute_dataset_stats(examples);
        let mut backends = Vec::new();

        if self.config.include_pattern {
            let results = self.evaluate_backend("Pattern", &RegexNER::new(), examples);
            backends.push(results);
        }

        if self.config.include_heuristic {
            let results = self.evaluate_backend("Heuristic", &HeuristicNER::new(), examples);
            backends.push(results);
        }

        if self.config.include_stacked {
            let results = self.evaluate_backend("Stacked", &StackedNER::default(), examples);
            backends.push(results);
        }

        #[cfg(feature = "onnx")]
        if self.config.include_gliner {
            use crate::{GLiNEROnnx, DEFAULT_GLINER_MODEL};
            match GLiNEROnnx::new(DEFAULT_GLINER_MODEL) {
                Ok(gliner) => {
                    let results = self.evaluate_backend("GLiNER", &gliner, examples);
                    backends.push(results);
                }
                Err(e) => {
                    log::warn!("Failed to load GLiNER for eval: {}", e);
                }
            }
        }

        EvalReport {
            timestamp: chrono::Utc::now()
                .format("%Y-%m-%d %H:%M:%S UTC")
                .to_string(),
            total_examples: examples.len(),
            backends,
            dataset_stats,
        }
    }

    fn evaluate_backend<M: Model>(
        &self,
        name: &str,
        model: &M,
        examples: &[AnnotatedExample],
    ) -> BackendResults {
        let start = Instant::now();

        let mut overall_tp = 0;
        let mut overall_fp = 0;
        let mut overall_fn = 0;

        let mut domain_metrics: HashMap<String, (usize, usize, usize)> = HashMap::new();
        let mut difficulty_metrics: HashMap<String, (usize, usize, usize)> = HashMap::new();
        let mut type_metrics: HashMap<String, (usize, usize, usize)> = HashMap::new();

        for example in examples {
            let predicted = model
                .extract_entities(&example.text, None)
                .unwrap_or_default();
            let gold = &example.entities;

            let (tp, fp, fn_) = compute_entity_matches(&predicted, gold);
            overall_tp += tp;
            overall_fp += fp;
            overall_fn += fn_;

            // Per-domain
            if self.config.per_domain {
                let domain_key = format!("{:?}", example.domain);
                let entry = domain_metrics.entry(domain_key).or_insert((0, 0, 0));
                entry.0 += tp;
                entry.1 += fp;
                entry.2 += fn_;
            }

            // Per-difficulty
            if self.config.per_difficulty {
                let diff_key = format!("{:?}", example.difficulty);
                let entry = difficulty_metrics.entry(diff_key).or_insert((0, 0, 0));
                entry.0 += tp;
                entry.1 += fp;
                entry.2 += fn_;
            }

            // Per entity type
            for gold_entity in gold {
                let type_key = entity_type_string(&gold_entity.entity_type);
                let entry = type_metrics.entry(type_key).or_insert((0, 0, 0));

                // Check if matched
                let matched = predicted.iter().any(|p| entities_match(p, gold_entity));
                if matched {
                    entry.0 += 1; // TP
                } else {
                    entry.2 += 1; // FN
                }
            }

            // Count FPs per type
            for pred_entity in &predicted {
                let matched = gold.iter().any(|g| entities_match(pred_entity, g));
                if !matched {
                    let type_key = entity_type_string(&pred_entity.entity_type);
                    let entry = type_metrics.entry(type_key).or_insert((0, 0, 0));
                    entry.1 += 1; // FP
                }
            }
        }

        let duration = start.elapsed();
        let overall =
            EvalMetrics::from_counts(overall_tp, overall_fp, overall_fn, duration, examples.len());

        let by_domain = domain_metrics
            .into_iter()
            .map(|(k, (tp, fp, fn_))| (k, EvalMetrics::from_counts(tp, fp, fn_, Duration::ZERO, 0)))
            .collect();

        let by_difficulty = difficulty_metrics
            .into_iter()
            .map(|(k, (tp, fp, fn_))| (k, EvalMetrics::from_counts(tp, fp, fn_, Duration::ZERO, 0)))
            .collect();

        let by_entity_type = type_metrics
            .into_iter()
            .map(|(k, (tp, fp, fn_))| (k, EvalMetrics::from_counts(tp, fp, fn_, Duration::ZERO, 0)))
            .collect();

        BackendResults {
            name: name.to_string(),
            overall,
            by_domain,
            by_difficulty,
            by_entity_type,
        }
    }
}

impl Default for BackendEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn compute_dataset_stats(examples: &[AnnotatedExample]) -> DatasetStats {
    let mut by_domain: HashMap<String, usize> = HashMap::new();
    let mut by_difficulty: HashMap<String, usize> = HashMap::new();
    let mut by_entity_type: HashMap<String, usize> = HashMap::new();
    let mut total_entities = 0;

    for example in examples {
        *by_domain
            .entry(format!("{:?}", example.domain))
            .or_insert(0) += 1;
        *by_difficulty
            .entry(format!("{:?}", example.difficulty))
            .or_insert(0) += 1;

        for entity in &example.entities {
            total_entities += 1;
            *by_entity_type
                .entry(entity_type_string(&entity.entity_type))
                .or_insert(0) += 1;
        }
    }

    DatasetStats {
        total_examples: examples.len(),
        total_entities,
        by_domain,
        by_difficulty,
        by_entity_type,
    }
}

fn compute_entity_matches(
    predicted: &[Entity],
    gold: &[crate::eval::datasets::GoldEntity],
) -> (usize, usize, usize) {
    let mut tp = 0;
    let mut matched_gold: Vec<bool> = vec![false; gold.len()];

    for pred in predicted {
        let mut found = false;
        for (i, gold_entity) in gold.iter().enumerate() {
            if !matched_gold[i] && entities_match(pred, gold_entity) {
                matched_gold[i] = true;
                found = true;
                break;
            }
        }
        if found {
            tp += 1;
        }
    }

    let fp = predicted.len() - tp;
    let fn_ = gold.len() - tp;

    (tp, fp, fn_)
}

fn entities_match(pred: &Entity, gold: &crate::eval::datasets::GoldEntity) -> bool {
    // Exact span match
    if pred.start == gold.start && pred.end == gold.end {
        return true;
    }

    // Text match with some tolerance
    let pred_text = pred.text.to_lowercase();
    let gold_text = gold.text.to_lowercase();

    if pred_text == gold_text {
        // Allow small position differences (off by 1-2 chars)
        let start_close = (pred.start as i64 - gold.start as i64).abs() <= 2;
        let end_close = (pred.end as i64 - gold.end as i64).abs() <= 2;
        return start_close && end_close;
    }

    false
}

fn entity_type_string(entity_type: &EntityType) -> String {
    match entity_type {
        EntityType::Person => "Person".to_string(),
        EntityType::Organization => "Organization".to_string(),
        EntityType::Location => "Location".to_string(),
        EntityType::Date => "Date".to_string(),
        EntityType::Time => "Time".to_string(),
        EntityType::Money => "Money".to_string(),
        EntityType::Percent => "Percent".to_string(),
        EntityType::Email => "Email".to_string(),
        EntityType::Phone => "Phone".to_string(),
        EntityType::Url => "Url".to_string(),
        EntityType::Quantity => "Quantity".to_string(),
        EntityType::Cardinal => "Cardinal".to_string(),
        EntityType::Ordinal => "Ordinal".to_string(),
        EntityType::Custom { name, .. } => format!("Custom({})", name),
        EntityType::Other(s) => format!("Other({})", s),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluator_runs() {
        let config = EvalConfig {
            max_examples: 50,
            ..Default::default()
        };
        let evaluator = BackendEvaluator::with_config(config);
        let report = evaluator.run_comprehensive();

        assert!(!report.backends.is_empty());
        assert!(report.total_examples > 0);
    }

    #[test]
    fn test_pattern_backend_eval() {
        let evaluator = BackendEvaluator::with_config(EvalConfig {
            include_pattern: true,
            include_heuristic: false,
            include_stacked: false,
            include_gliner: false, // Explicitly disable GLiNER
            max_examples: 20,
            ..Default::default()
        });

        let report = evaluator.run_comprehensive();
        assert_eq!(report.backends.len(), 1);
        assert_eq!(report.backends[0].name, "Pattern");
    }

    #[test]
    fn test_markdown_generation() {
        let evaluator = BackendEvaluator::with_config(EvalConfig {
            max_examples: 10,
            ..Default::default()
        });

        let report = evaluator.run_comprehensive();
        let md = report.to_markdown();

        assert!(md.contains("# NER Backend Evaluation Report"));
        assert!(md.contains("Pattern"));
    }

    #[test]
    fn test_html_generation() {
        let evaluator = BackendEvaluator::with_config(EvalConfig {
            max_examples: 10,
            ..Default::default()
        });

        let report = evaluator.run_comprehensive();
        let html = report.to_html();

        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("NER Backend Evaluation"));
    }

    #[test]
    fn test_technology_dataset_eval() {
        let evaluator = BackendEvaluator::with_config(EvalConfig {
            include_pattern: true,
            include_heuristic: false,
            include_stacked: false,
            include_gliner: false,
            ..Default::default()
        });

        let report = evaluator.run_technology();
        assert!(report.total_examples > 0);
    }

    #[test]
    fn test_metrics_calculation() {
        let metrics = EvalMetrics::from_counts(8, 2, 4, Duration::from_millis(100), 10);

        assert!((metrics.precision - 0.8).abs() < 0.01);
        assert!((metrics.recall - (8.0 / 12.0)).abs() < 0.01);
        assert!(metrics.f1 > 0.0);
    }
}
