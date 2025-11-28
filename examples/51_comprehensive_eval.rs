//! Comprehensive NER evaluation on real datasets.
//!
//! This benchmark:
//! 1. Downloads/loads real NER datasets (WikiGold, CoNLL-2003, WNUT-17, etc.)
//! 2. Evaluates all available backends
//! 3. Produces detailed per-dataset, per-type metrics
//! 4. Generates an HTML report for analysis
//!
//! Run with:
//!   cargo run --example comprehensive_eval --features network
//!
//! Or offline (uses cached datasets):
//!   cargo run --example comprehensive_eval

use anno::backends::{PatternNER, StatisticalNER, StackedNER};
use anno::eval::loader::{DatasetId, DatasetLoader, LoadedDataset};
use anno::eval::{evaluate_ner_model, TypeMetrics};
use anno::Model;
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::time::{Duration, Instant};

/// Results for a single backend on a single dataset.
#[derive(Debug)]
#[allow(dead_code)]
struct DatasetResult {
    dataset_name: String,
    dataset_sentences: usize,
    dataset_entities: usize,
    precision: f64,
    recall: f64,
    f1: f64,
    macro_f1: Option<f64>,
    per_type: HashMap<String, TypeMetrics>,
    latency_ms: f64,
    tokens_per_sec: f64,
}

/// Results for a single backend across all datasets.
#[derive(Debug)]
struct BackendResults {
    backend_name: String,
    backend_description: String,
    total_time: Duration,
    results: Vec<DatasetResult>,
    aggregate_f1: f64,
    aggregate_precision: f64,
    aggregate_recall: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Comprehensive NER Evaluation ===\n");

    // Initialize dataset loader
    let loader = DatasetLoader::new()?;
    println!("Cache directory: {:?}", loader.cache_dir());

    // Check what datasets are available
    println!("\nDataset availability:");
    for (id, cached) in loader.status() {
        let status = if cached { "[cached]" } else { "[not cached]" };
        println!("  {:20} {}", id.name(), status);
    }

    // Load available datasets
    let mut datasets: Vec<LoadedDataset> = Vec::new();

    for id in DatasetId::all() {
        // Try to load from cache first
        match loader.load(*id) {
            Ok(ds) => {
                let stats = ds.stats();
                println!(
                    "\nLoaded {}: {} sentences, {} entities",
                    id.name(),
                    stats.sentences,
                    stats.entities
                );
                datasets.push(ds);
            }
            Err(_) => {
                // Try to download if network feature is available
                #[cfg(feature = "network")]
                match loader.load_or_download(*id) {
                    Ok(ds) => {
                        let stats = ds.stats();
                        println!(
                            "\nDownloaded {}: {} sentences, {} entities",
                            id.name(),
                            stats.sentences,
                            stats.entities
                        );
                        datasets.push(ds);
                    }
                    Err(e) => {
                        println!("\nSkipping {} (download failed: {})", id.name(), e);
                    }
                }

                #[cfg(not(feature = "network"))]
                {
                    println!("\nSkipping {} (not cached, network feature disabled)", id.name());
                }
            }
        }
    }

    if datasets.is_empty() {
        println!("\nNo datasets available. Enable 'network' feature to download:");
        println!("  cargo run --example comprehensive_eval --features network");
        return Ok(());
    }

    println!("\n=== Running Evaluations ===\n");

    // Initialize backends
    #[allow(unused_mut)] // `mut` needed when onnx feature is enabled
    let mut backends: Vec<(&str, &str, Box<dyn Model>)> = vec![
        (
            "PatternNER",
            "Regex patterns for structured entities (DATE, MONEY, etc.)",
            Box::new(PatternNER::new()),
        ),
        (
            "StatisticalNER",
            "Heuristics for named entities (PER, ORG, LOC)",
            Box::new(StatisticalNER::new()),
        ),
        (
            "StackedNER",
            "Combined Pattern + Statistical extraction",
            Box::new(StackedNER::new()),
        ),
    ];

    // Add ONNX backends if feature enabled
    #[cfg(feature = "onnx")]
    {
        println!("Loading BERT ONNX model...");
        match anno::BertNEROnnx::new(anno::DEFAULT_BERT_ONNX_MODEL) {
            Ok(bert) => {
                println!("  BERT ONNX model loaded successfully");
                backends.push((
                    "BertNEROnnx",
                    "BERT NER via ONNX Runtime (~86% F1 on CoNLL-03)",
                    Box::new(bert),
                ));
            }
            Err(e) => {
                println!("  Failed to load BERT ONNX: {}", e);
            }
        }
    }

    let mut all_backend_results: Vec<BackendResults> = Vec::new();

    for (name, description, model) in &backends {
        println!("Evaluating {}...", name);
        let backend_start = Instant::now();
        let mut dataset_results: Vec<DatasetResult> = Vec::new();

        for dataset in &datasets {
            let test_cases = dataset.to_test_cases();
            let ds_start = Instant::now();

            match evaluate_ner_model(model.as_ref(), &test_cases) {
                Ok(results) => {
                    let ds_time = ds_start.elapsed();
                    dataset_results.push(DatasetResult {
                        dataset_name: dataset.id.name().to_string(),
                        dataset_sentences: dataset.len(),
                        dataset_entities: dataset.entity_count(),
                        precision: results.precision,
                        recall: results.recall,
                        f1: results.f1,
                        macro_f1: results.macro_f1,
                        per_type: results.per_type.clone(),
                        latency_ms: ds_time.as_secs_f64() * 1000.0,
                        tokens_per_sec: results.tokens_per_second,
                    });

                    println!(
                        "  {}: F1={:.1}% P={:.1}% R={:.1}% ({:.1}ms)",
                        dataset.id.name(),
                        results.f1 * 100.0,
                        results.precision * 100.0,
                        results.recall * 100.0,
                        ds_time.as_secs_f64() * 1000.0
                    );
                }
                Err(e) => {
                    println!("  {}: ERROR: {}", dataset.id.name(), e);
                }
            }
        }

        let total_time = backend_start.elapsed();

        // Compute aggregate metrics
        let (agg_f1, agg_p, agg_r) = if dataset_results.is_empty() {
            (0.0, 0.0, 0.0)
        } else {
            let sum_f1: f64 = dataset_results.iter().map(|r| r.f1).sum();
            let sum_p: f64 = dataset_results.iter().map(|r| r.precision).sum();
            let sum_r: f64 = dataset_results.iter().map(|r| r.recall).sum();
            let n = dataset_results.len() as f64;
            (sum_f1 / n, sum_p / n, sum_r / n)
        };

        all_backend_results.push(BackendResults {
            backend_name: name.to_string(),
            backend_description: description.to_string(),
            total_time,
            results: dataset_results,
            aggregate_f1: agg_f1,
            aggregate_precision: agg_p,
            aggregate_recall: agg_r,
        });

        println!(
            "  AGGREGATE: F1={:.1}% P={:.1}% R={:.1}% (total: {:.1}ms)\n",
            agg_f1 * 100.0,
            agg_p * 100.0,
            agg_r * 100.0,
            total_time.as_secs_f64() * 1000.0
        );
    }

    // Generate HTML report
    let html_path = Path::new("eval_results.html");
    generate_html_report(&all_backend_results, &datasets, html_path)?;
    println!("\nHTML report written to: {}", html_path.display());

    // Print summary table
    println!("\n=== Summary Table ===\n");
    println!(
        "{:<20} {:>10} {:>10} {:>10} {:>12}",
        "Backend", "F1", "Precision", "Recall", "Time"
    );
    println!("{}", "-".repeat(64));
    for br in &all_backend_results {
        println!(
            "{:<20} {:>9.1}% {:>9.1}% {:>9.1}% {:>10.1}ms",
            br.backend_name,
            br.aggregate_f1 * 100.0,
            br.aggregate_precision * 100.0,
            br.aggregate_recall * 100.0,
            br.total_time.as_secs_f64() * 1000.0
        );
    }

    // Per-type analysis
    println!("\n=== Per-Type Analysis (StackedNER) ===\n");
    if let Some(tiered) = all_backend_results.iter().find(|b| b.backend_name == "StackedNER") {
        let mut all_types: HashMap<String, (f64, f64, f64, usize)> = HashMap::new();

        for dr in &tiered.results {
            for (etype, metrics) in &dr.per_type {
                let entry = all_types.entry(etype.clone()).or_insert((0.0, 0.0, 0.0, 0));
                entry.0 += metrics.f1;
                entry.1 += metrics.precision;
                entry.2 += metrics.recall;
                entry.3 += 1;
            }
        }

        let mut sorted_types: Vec<_> = all_types.iter().collect();
        sorted_types.sort_by(|a, b| {
            let avg_f1_a = a.1 .0 / a.1 .3.max(1) as f64;
            let avg_f1_b = b.1 .0 / b.1 .3.max(1) as f64;
            avg_f1_b
                .partial_cmp(&avg_f1_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        println!(
            "{:<15} {:>10} {:>10} {:>10} {:>8}",
            "Type", "Avg F1", "Avg P", "Avg R", "Datasets"
        );
        println!("{}", "-".repeat(56));

        for (etype, (sum_f1, sum_p, sum_r, count)) in sorted_types {
            let n = *count.max(&1) as f64;
            println!(
                "{:<15} {:>9.1}% {:>9.1}% {:>9.1}% {:>8}",
                etype,
                (sum_f1 / n) * 100.0,
                (sum_p / n) * 100.0,
                (sum_r / n) * 100.0,
                count
            );
        }
    }

    // Recommendations
    println!("\n=== Recommendations ===\n");
    println!("1. PatternNER: Best for structured entities (DATE, MONEY, EMAIL, URL, PHONE)");
    println!("   - High precision, fast, no dependencies");
    println!();
    println!("2. StatisticalNER: Baseline for named entities (PER, ORG, LOC)");
    println!("   - Moderate accuracy (~60-70% F1), zero dependencies");
    println!();
    println!("3. StackedNER: Best zero-dependency option");
    println!("   - Combines pattern and statistical extraction");
    println!();
    println!("4. For production NER on named entities, enable ML backends:");
    println!("   cargo run --example comprehensive_eval --features onnx");
    println!("   cargo run --example comprehensive_eval --features candle");

    Ok(())
}

/// Generate an HTML report with interactive tables and charts.
fn generate_html_report(
    backend_results: &[BackendResults],
    datasets: &[LoadedDataset],
    path: &Path,
) -> std::io::Result<()> {
    let mut f = fs::File::create(path)?;

    // Generate timestamp
    let timestamp = chrono::Utc::now().to_rfc3339();

    // HTML header
    write!(
        f,
        r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>anno NER Evaluation Results</title>
    <style>
        :root {{
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-tertiary: #21262d;
            --text-primary: #c9d1d9;
            --text-secondary: #8b949e;
            --text-muted: #6e7681;
            --border-color: #30363d;
            --accent-green: #3fb950;
            --accent-yellow: #d29922;
            --accent-red: #f85149;
            --accent-blue: #58a6ff;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.5;
            padding: 2rem;
        }}
        h1 {{ color: var(--text-primary); margin-bottom: 0.5rem; font-size: 1.75rem; }}
        h2 {{ color: var(--text-secondary); margin: 2rem 0 1rem; font-size: 1.25rem; border-bottom: 1px solid var(--border-color); padding-bottom: 0.5rem; }}
        .timestamp {{ color: var(--text-muted); font-size: 0.875rem; margin-bottom: 2rem; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem; margin-bottom: 2rem; }}
        .card {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 1rem;
        }}
        .card h3 {{ color: var(--text-primary); font-size: 1rem; margin-bottom: 0.5rem; }}
        .card p {{ color: var(--text-secondary); font-size: 0.875rem; }}
        .metric {{ display: flex; justify-content: space-between; padding: 0.25rem 0; }}
        .metric-label {{ color: var(--text-secondary); }}
        .metric-value {{ font-weight: 600; }}
        .metric-good {{ color: var(--accent-green); }}
        .metric-ok {{ color: var(--accent-yellow); }}
        .metric-poor {{ color: var(--accent-red); }}
        table {{ width: 100%; border-collapse: collapse; background: var(--bg-secondary); border-radius: 6px; overflow: hidden; }}
        th, td {{ padding: 0.75rem 1rem; text-align: left; border-bottom: 1px solid var(--border-color); }}
        th {{ background: var(--bg-tertiary); color: var(--text-secondary); font-weight: 600; font-size: 0.875rem; text-transform: uppercase; letter-spacing: 0.05em; }}
        td {{ font-size: 0.875rem; }}
        tr:hover {{ background: var(--bg-tertiary); }}
        .bar {{ height: 8px; background: var(--bg-tertiary); border-radius: 4px; overflow: hidden; }}
        .bar-fill {{ height: 100%; transition: width 0.3s; }}
        .bar-green {{ background: var(--accent-green); }}
        .bar-yellow {{ background: var(--accent-yellow); }}
        .bar-red {{ background: var(--accent-red); }}
        .summary {{ display: flex; gap: 2rem; flex-wrap: wrap; margin: 2rem 0; }}
        .summary-stat {{ text-align: center; }}
        .summary-value {{ font-size: 2rem; font-weight: 700; color: var(--accent-blue); }}
        .summary-label {{ color: var(--text-muted); font-size: 0.875rem; }}
    </style>
</head>
<body>
    <h1>anno NER Evaluation Results</h1>
    <p class="timestamp">Generated: {}</p>
"#,
        timestamp
    )?;

    // Summary stats
    let total_sentences: usize = datasets.iter().map(|d| d.len()).sum();
    let total_entities: usize = datasets.iter().map(|d| d.entity_count()).sum();

    write!(
        f,
        r#"
    <div class="summary">
        <div class="summary-stat">
            <div class="summary-value">{}</div>
            <div class="summary-label">Datasets</div>
        </div>
        <div class="summary-stat">
            <div class="summary-value">{}</div>
            <div class="summary-label">Sentences</div>
        </div>
        <div class="summary-stat">
            <div class="summary-value">{}</div>
            <div class="summary-label">Entities</div>
        </div>
        <div class="summary-stat">
            <div class="summary-value">{}</div>
            <div class="summary-label">Backends</div>
        </div>
    </div>
"#,
        datasets.len(),
        total_sentences,
        total_entities,
        backend_results.len()
    )?;

    // Backend summary cards
    write!(f, "<h2>Backend Summary</h2>\n<div class=\"grid\">\n")?;

    for br in backend_results {
        let f1_class = if br.aggregate_f1 >= 0.8 {
            "metric-good"
        } else if br.aggregate_f1 >= 0.5 {
            "metric-ok"
        } else {
            "metric-poor"
        };

        write!(
            f,
            r#"
    <div class="card">
        <h3>{}</h3>
        <p>{}</p>
        <div class="metric">
            <span class="metric-label">F1 Score</span>
            <span class="metric-value {}">{:.1}%</span>
        </div>
        <div class="bar"><div class="bar-fill bar-{}" style="width: {:.0}%"></div></div>
        <div class="metric">
            <span class="metric-label">Precision</span>
            <span class="metric-value">{:.1}%</span>
        </div>
        <div class="metric">
            <span class="metric-label">Recall</span>
            <span class="metric-value">{:.1}%</span>
        </div>
        <div class="metric">
            <span class="metric-label">Total Time</span>
            <span class="metric-value">{:.1}ms</span>
        </div>
    </div>
"#,
            br.backend_name,
            br.backend_description,
            f1_class,
            br.aggregate_f1 * 100.0,
            if br.aggregate_f1 >= 0.8 {
                "green"
            } else if br.aggregate_f1 >= 0.5 {
                "yellow"
            } else {
                "red"
            },
            br.aggregate_f1 * 100.0,
            br.aggregate_precision * 100.0,
            br.aggregate_recall * 100.0,
            br.total_time.as_secs_f64() * 1000.0
        )?;
    }
    writeln!(f, "</div>")?;

    // Per-dataset results table
    write!(
        f,
        r#"
    <h2>Per-Dataset Results</h2>
    <table>
        <thead>
            <tr>
                <th>Dataset</th>
                <th>Backend</th>
                <th>F1</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>Time</th>
            </tr>
        </thead>
        <tbody>
"#
    )?;

    for br in backend_results {
        for dr in &br.results {
            let f1_class = if dr.f1 >= 0.8 {
                "metric-good"
            } else if dr.f1 >= 0.5 {
                "metric-ok"
            } else {
                "metric-poor"
            };

            write!(
                f,
                r#"
            <tr>
                <td>{}</td>
                <td>{}</td>
                <td class="{}">{:.1}%</td>
                <td>{:.1}%</td>
                <td>{:.1}%</td>
                <td>{:.1}ms</td>
            </tr>
"#,
                dr.dataset_name,
                br.backend_name,
                f1_class,
                dr.f1 * 100.0,
                dr.precision * 100.0,
                dr.recall * 100.0,
                dr.latency_ms
            )?;
        }
    }

    write!(f, "        </tbody>\n    </table>\n")?;

    // Dataset details
    write!(f, "<h2>Dataset Details</h2>\n<div class=\"grid\">\n")?;

    for ds in datasets {
        let stats = ds.stats();
        write!(
            f,
            r#"
    <div class="card">
        <h3>{}</h3>
        <div class="metric">
            <span class="metric-label">Sentences</span>
            <span class="metric-value">{}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Tokens</span>
            <span class="metric-value">{}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Entities</span>
            <span class="metric-value">{}</span>
        </div>
    </div>
"#,
            stats.name, stats.sentences, stats.tokens, stats.entities
        )?;
    }
    writeln!(f, "</div>")?;

    // Footer
    write!(
        f,
        r#"
    <h2>Notes</h2>
    <ul style="color: var(--text-secondary); margin-left: 1.5rem;">
        <li>PatternNER excels at structured entities (DATE, MONEY, EMAIL, URL, PHONE)</li>
        <li>StatisticalNER provides baseline named entity detection without ML</li>
        <li>For production use, enable ML backends: <code>--features onnx</code> or <code>--features candle</code></li>
        <li>All metrics use exact span matching (strict evaluation)</li>
    </ul>
</body>
</html>
"#
    )?;

    Ok(())
}

