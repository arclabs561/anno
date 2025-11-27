//! Full dataset evaluation for NER backends.
//!
//! Downloads real NER datasets and evaluates all available backends.
//!
//! Run with:
//!   cargo run --example full_dataset_eval --features "onnx,network"
//!
//! This evaluates:
//! - Zero-dep backends: PatternNER, StatisticalNER, StackedNER
//! - ML backends: BertNEROnnx (with onnx feature)
//! - Real datasets: WikiGold, WNUT-17, CoNLL-2003, MIT Movie/Restaurant

use anno::eval::{evaluate_ner_model_with_mapper, MetricWithVariance, NEREvaluationResults, GoldEntity};
use anno::eval::loader::{DatasetLoader, DatasetId};
use anno::{Model, PatternNER, StatisticalNER, StackedNER, TypeMapper};
use std::collections::HashMap;
use std::time::Instant;

/// Backend wrapper for uniform evaluation
struct Backend {
    name: &'static str,
    description: &'static str,
    model: Box<dyn Model>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Full NER Dataset Evaluation ===\n");
    println!("This downloads real NER datasets and evaluates all backends.\n");

    // Create backends
    #[allow(unused_mut)] // `mut` needed when onnx feature is enabled
    let mut backends: Vec<Backend> = vec![
        Backend {
            name: "PatternNER",
            description: "Regex patterns (DATE/MONEY/EMAIL/URL)",
            model: Box::new(PatternNER::new()),
        },
        Backend {
            name: "StatisticalNER",
            description: "Heuristic PER/ORG/LOC detection",
            model: Box::new(StatisticalNER::new()),
        },
        Backend {
            name: "StackedNER",
            description: "Combined Pattern + Statistical",
            model: Box::new(StackedNER::new()),
        },
    ];

    // Add ONNX backend if available
    #[cfg(feature = "onnx")]
    {
        match anno::BertNEROnnx::new(anno::DEFAULT_BERT_ONNX_MODEL) {
            Ok(bert) => {
                backends.push(Backend {
                    name: "BertNEROnnx",
                    description: "BERT NER (ONNX Runtime)",
                    model: Box::new(bert),
                });
                println!("BERT ONNX backend loaded successfully");
            }
            Err(e) => {
                eprintln!("Note: BERT ONNX not available: {}", e);
            }
        }
    }

    println!("\nBackends available: {}", backends.len());
    for b in &backends {
        println!("  - {}: {}", b.name, b.description);
    }

    // Dataset loader
    let loader = DatasetLoader::new()?;
    println!("\nCache directory: {:?}", loader.cache_dir());

    // Datasets to evaluate
    let datasets = [
        DatasetId::WikiGold,
        DatasetId::Wnut17,
        DatasetId::CoNLL2003Sample,
        DatasetId::MitMovie,
        DatasetId::MitRestaurant,
    ];

    println!("\n=== Downloading/Loading Datasets ===\n");

    let mut dataset_results: HashMap<DatasetId, Vec<(String, NEREvaluationResults)>> = HashMap::new();

    for dataset_id in &datasets {
        print!("Loading {:20}... ", dataset_id.name());
        std::io::Write::flush(&mut std::io::stdout())?;

        #[cfg(feature = "network")]
        let loaded = loader.load_or_download(*dataset_id);

        #[cfg(not(feature = "network"))]
        let loaded = loader.load(*dataset_id);

        match loaded {
            Ok(dataset) => {
                println!(
                    "OK ({} sentences, {} tokens)",
                    dataset.sentences.len(),
                    dataset.sentences.iter().map(|s| s.tokens.len()).sum::<usize>()
                );

                // Convert to test cases
                let test_cases: Vec<(String, Vec<GoldEntity>)> = dataset
                    .sentences
                    .iter()
                    .filter(|s| !s.tokens.is_empty())
                    .map(|s| {
                        let text = s.text();
                        let entities = s.entities();
                        (text, entities)
                    })
                    .collect();

                // Limit for faster evaluation (sample if too large)
                let max_samples = 2000;
                let test_cases: Vec<_> = if test_cases.len() > max_samples {
                    println!("  (sampling {} of {} examples)", max_samples, test_cases.len());
                    test_cases.into_iter().take(max_samples).collect()
                } else {
                    test_cases
                };

                if test_cases.is_empty() {
                    println!("  [WARN] No valid test cases");
                    continue;
                }

                // Get type mapper for domain-specific datasets
                let type_mapper: Option<TypeMapper> = dataset_id.type_mapper();
                if type_mapper.is_some() {
                    println!("  (using TypeMapper for domain-specific entity types)");
                }
                
                // Evaluate each backend
                let mut results = Vec::new();
                for backend in &backends {
                    let start = Instant::now();
                    let eval_result = evaluate_ner_model_with_mapper(
                        backend.model.as_ref(), 
                        &test_cases,
                        type_mapper.as_ref()
                    );
                    match eval_result {
                        Ok(eval_results) => {
                            let elapsed = start.elapsed();
                            results.push((backend.name.to_string(), eval_results.clone()));
                            println!(
                                "  {:<16} F1={:5.1}%  P={:5.1}%  R={:5.1}%  ({:.1}ms)",
                                backend.name,
                                eval_results.f1 * 100.0,
                                eval_results.precision * 100.0,
                                eval_results.recall * 100.0,
                                elapsed.as_secs_f64() * 1000.0
                            );
                        }
                        Err(e) => {
                            println!("  {:<16} ERROR: {}", backend.name, e);
                        }
                    }
                }
                dataset_results.insert(*dataset_id, results);
            }
            Err(e) => {
                println!("FAILED: {}", e);
            }
        }
    }

    // Summary with variance
    println!("\n=== Summary Across Datasets ===\n");
    
    for backend in &backends {
        let f1_scores: Vec<f64> = dataset_results
            .values()
            .filter_map(|results| {
                results.iter().find(|(name, _)| name == backend.name).map(|(_, r)| r.f1)
            })
            .collect();

        if !f1_scores.is_empty() {
            let variance = MetricWithVariance::from_samples(&f1_scores);
            println!(
                "{:<16} Avg F1: {}  Range: [{:.1}% - {:.1}%]",
                backend.name,
                variance,
                variance.min * 100.0,
                variance.max * 100.0
            );
        }
    }

    // Generate HTML report
    generate_html_report(&backends, &dataset_results)?;

    println!("\n=== Done ===");
    println!("HTML report written to: full_eval_results.html");

    Ok(())
}

fn generate_html_report(
    backends: &[Backend],
    results: &HashMap<DatasetId, Vec<(String, NEREvaluationResults)>>,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;

    let mut html = String::new();
    html.push_str(r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NER Full Dataset Evaluation</title>
    <style>
        :root {
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-tertiary: #21262d;
            --text-primary: #c9d1d9;
            --text-secondary: #8b949e;
            --border-color: #30363d;
            --accent-green: #3fb950;
            --accent-yellow: #d29922;
            --accent-red: #f85149;
            --accent-blue: #58a6ff;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 2rem;
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 { font-size: 1.75rem; margin-bottom: 0.5rem; }
        h2 { color: var(--text-secondary); margin: 2rem 0 1rem; font-size: 1.25rem; 
             border-bottom: 1px solid var(--border-color); padding-bottom: 0.5rem; }
        .timestamp { color: var(--text-secondary); font-size: 0.875rem; margin-bottom: 2rem; }
        table { width: 100%; border-collapse: collapse; background: var(--bg-secondary); 
                border-radius: 6px; overflow: hidden; margin: 1rem 0; }
        th, td { padding: 0.75rem 1rem; text-align: left; border-bottom: 1px solid var(--border-color); }
        th { background: var(--bg-tertiary); color: var(--text-secondary); font-weight: 600; 
             font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }
        td { font-size: 0.875rem; }
        tr:hover { background: var(--bg-tertiary); }
        .good { color: var(--accent-green); font-weight: 600; }
        .ok { color: var(--accent-yellow); }
        .poor { color: var(--accent-red); }
        .bar { height: 6px; background: var(--bg-tertiary); border-radius: 3px; margin-top: 4px; }
        .bar-fill { height: 100%; border-radius: 3px; }
        .bar-green { background: var(--accent-green); }
        .bar-yellow { background: var(--accent-yellow); }
        .bar-red { background: var(--accent-red); }
        .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 2rem 0; }
        .stat-card { background: var(--bg-secondary); border: 1px solid var(--border-color); 
                     border-radius: 6px; padding: 1rem; text-align: center; }
        .stat-value { font-size: 1.5rem; font-weight: 700; color: var(--accent-blue); }
        .stat-label { color: var(--text-secondary); font-size: 0.75rem; text-transform: uppercase; }
    </style>
</head>
<body>
    <h1>NER Full Dataset Evaluation</h1>
    <p class="timestamp">Generated: "#);

    html.push_str(&chrono::Utc::now().to_rfc3339());
    html.push_str("</p>\n");

    // Summary stats
    html.push_str(r#"
    <div class="summary">
        <div class="stat-card">
            <div class="stat-value">"#);
    html.push_str(&results.len().to_string());
    html.push_str(r#"</div>
            <div class="stat-label">Datasets</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">"#);
    html.push_str(&backends.len().to_string());
    html.push_str(r#"</div>
            <div class="stat-label">Backends</div>
        </div>
    </div>
"#);

    // Results table
    html.push_str("<h2>Results by Dataset</h2>\n<table>\n<thead><tr><th>Dataset</th>");
    for b in backends {
        html.push_str(&format!("<th>{}</th>", b.name));
    }
    html.push_str("</tr></thead>\n<tbody>\n");

    for (dataset_id, dataset_results) in results {
        html.push_str(&format!("<tr><td>{}</td>", dataset_id.name()));
        for backend in backends {
            if let Some((_, r)) = dataset_results.iter().find(|(name, _)| name == backend.name) {
                let class = if r.f1 > 0.5 { "good" } else if r.f1 > 0.2 { "ok" } else { "poor" };
                let bar_class = if r.f1 > 0.5 { "bar-green" } else if r.f1 > 0.2 { "bar-yellow" } else { "bar-red" };
                html.push_str(&format!(
                    r#"<td><span class="{}">{:.1}%</span><div class="bar"><div class="bar-fill {}" style="width: {}%"></div></div></td>"#,
                    class,
                    r.f1 * 100.0,
                    bar_class,
                    (r.f1 * 100.0).min(100.0)
                ));
            } else {
                html.push_str("<td>-</td>");
            }
        }
        html.push_str("</tr>\n");
    }

    html.push_str("</tbody></table>\n");

    // Backend descriptions
    html.push_str("<h2>Backend Descriptions</h2>\n<table>\n<thead><tr><th>Backend</th><th>Description</th></tr></thead>\n<tbody>\n");
    for b in backends {
        html.push_str(&format!("<tr><td>{}</td><td>{}</td></tr>\n", b.name, b.description));
    }
    html.push_str("</tbody></table>\n");

    html.push_str("</body></html>");

    let mut file = std::fs::File::create("full_eval_results.html")?;
    file.write_all(html.as_bytes())?;

    Ok(())
}

