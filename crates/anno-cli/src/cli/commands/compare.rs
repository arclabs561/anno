//! Compare command - Compare documents, models, or clusters

use clap::Parser;
use std::fs;

use super::super::parser::ModelBackend;
use anno::Entity;
use anno_core::GroundedDocument;

use serde::{Deserialize, Serialize};

/// Compare documents, models, or clusters
#[derive(Parser, Debug)]
pub struct CompareArgs {
    /// First input file
    #[arg(value_name = "FILE1")]
    pub file1: String,

    /// Second input file (or text for compare-models)
    #[arg(value_name = "FILE2")]
    pub file2: Option<String>,

    /// Compare models on the same input text (reads FILE1 from disk)
    #[arg(long)]
    pub models: bool,

    /// Models to compare (when --models is used)
    #[arg(long, value_delimiter = ',', value_name = "MODEL")]
    pub model_list: Vec<String>,

    /// Output format (diff, table, summary)
    #[arg(long, default_value = "diff")]
    pub format: String,

    /// Confidence delta threshold for treating an otherwise-identical entity as modified
    #[arg(long, default_value_t = 0.05)]
    pub confidence_epsilon: f64,

    /// Only emit changed entities (added/removed/modified); omit unchanged diffs
    #[arg(long)]
    pub changes_only: bool,

    /// Output file
    #[arg(short, long, value_name = "PATH")]
    pub output: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
struct ComparableEntity {
    text: String,
    entity_type: String,
    start: usize,
    end: usize,
    confidence: f64,
}

impl ComparableEntity {
    fn exact_key(&self) -> (String, String, usize, usize) {
        (
            self.text.clone(),
            self.entity_type.clone(),
            self.start,
            self.end,
        )
    }
    fn loose_key(&self) -> (String, String) {
        (self.text.clone(), self.entity_type.clone())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct DiffEntry {
    text: String,
    #[serde(rename = "type")]
    entity_type: String,
    change_type: String,
    start_a: Option<usize>,
    end_a: Option<usize>,
    confidence_a: Option<f64>,
    start_b: Option<usize>,
    end_b: Option<usize>,
    confidence_b: Option<f64>,
}

/// Execute the compare command.
pub fn run(args: CompareArgs) -> Result<(), String> {
    if args.models {
        // Compare models on same text
        let text = fs::read_to_string(&args.file1)
            .map_err(|e| format!("Failed to read {}: {}", args.file1, e))?;

        if args.model_list.is_empty() {
            return Err("--models requires --model-list with model names".to_string());
        }

        let mut results: Vec<(String, Vec<Entity>)> = Vec::new();

        for model_name in &args.model_list {
            let backend = match model_name.as_str() {
                "pattern" => ModelBackend::Pattern,
                "heuristic" => ModelBackend::Heuristic,
                "stacked" => ModelBackend::Stacked,
                #[cfg(feature = "onnx")]
                "gliner" => ModelBackend::Gliner,
                _ => {
                    return Err(format!("Unknown model: {}", model_name));
                }
            };

            let model = backend.create_model()?;
            let entities = model
                .extract_entities(&text, None)
                .map_err(|e| format!("Model {} failed: {}", model_name, e))?;
            results.push((model_name.clone(), entities));
        }

        // Output comparison
        match args.format.as_str() {
            "table" => {
                println!("\nModel Comparison:");
                println!("{:<15} {:<10}", "Model", "Entities");
                println!("{}", "-".repeat(25));
                for (name, entities) in &results {
                    println!("{:<15} {:<10}", name, entities.len());
                }
            }
            _ => {
                for (name, entities) in &results {
                    println!("\n{} ({} entities):", name, entities.len());
                    for e in entities {
                        println!("  - {} ({})", e.text, e.entity_type.as_label());
                    }
                }
            }
        }
    } else {
        // Compare two documents / extraction outputs
        let file2 = args
            .file2
            .ok_or("Second file required for document comparison")?;

        let json1 = fs::read_to_string(&args.file1)
            .map_err(|e| format!("Failed to read {}: {}", args.file1, e))?;
        let json2 =
            fs::read_to_string(&file2).map_err(|e| format!("Failed to read {}: {}", file2, e))?;

        let ents1 = parse_entities_from_any_json(&json1)
            .map_err(|e| format!("Failed to parse {}: {}", args.file1, e))?;
        let ents2 = parse_entities_from_any_json(&json2)
            .map_err(|e| format!("Failed to parse {}: {}", file2, e))?;

        let (diffs, counts, jaccard) =
            compare_entities(&ents1, &ents2, args.confidence_epsilon, args.changes_only);

        match args.format.as_str() {
            "json" => {
                let out = serde_json::json!({
                    "file1": args.file1,
                    "file2": file2,
                    "jaccard_similarity": jaccard,
                    "added": counts.added,
                    "removed": counts.removed,
                    "unchanged": counts.unchanged,
                    "modified": counts.modified,
                    "diffs": diffs,
                });
                println!("{}", serde_json::to_string_pretty(&out).unwrap_or_default());
            }
            "jsonl" => {
                let summary = serde_json::json!({
                    "_type": "summary",
                    "jaccard_similarity": jaccard,
                    "added": counts.added,
                    "removed": counts.removed,
                    "unchanged": counts.unchanged,
                    "modified": counts.modified,
                });
                println!("{}", summary);
                for d in diffs {
                    println!("{}", serde_json::to_string(&d).unwrap_or_default());
                }
            }
            "diff" | "summary" => {
                // Keep legacy human output for interactive usage.
                println!("\nComparison: {} vs {}", args.file1, file2);
                println!(
                    "added={} removed={} modified={} unchanged={} jaccard={:.3}",
                    counts.added, counts.removed, counts.modified, counts.unchanged, jaccard
                );
                for d in diffs {
                    println!("  {}: {} [{}]", d.change_type, d.text, d.entity_type);
                }
            }
            _ => {
                return Err(format!(
                    "Unknown format: {}. Use: diff, summary, json, jsonl",
                    args.format
                ))
            }
        }
    }

    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct Counts {
    added: u64,
    removed: u64,
    unchanged: u64,
    modified: u64,
}

fn parse_entities_from_any_json(content: &str) -> Result<Vec<ComparableEntity>, String> {
    let v: serde_json::Value =
        serde_json::from_str(content).map_err(|e| format!("invalid JSON: {}", e))?;

    // Case 1: extract output: { entities: [...] }
    if let Some(arr) = v.get("entities").and_then(|e| e.as_array()) {
        return Ok(arr
            .iter()
            .filter_map(|e| {
                Some(ComparableEntity {
                    text: e.get("text")?.as_str()?.to_string(),
                    entity_type: e.get("type")?.as_str()?.to_string(),
                    start: e.get("start")?.as_u64()? as usize,
                    end: e.get("end")?.as_u64()? as usize,
                    confidence: e.get("confidence")?.as_f64()?,
                })
            })
            .collect());
    }

    // Case 2: GroundedDocument JSON
    if let Ok(doc) = serde_json::from_value::<GroundedDocument>(v.clone()) {
        return Ok(doc
            .signals()
            .iter()
            .map(|s| {
                let (start, end) = s.text_offsets().unwrap_or((0, 0));
                ComparableEntity {
                    text: s.surface().to_string(),
                    entity_type: s.label().to_string(),
                    start,
                    end,
                    confidence: s.confidence as f64,
                }
            })
            .collect());
    }

    Err("unrecognized input format: expected {entities:[...]} or GroundedDocument".to_string())
}

fn compare_entities(
    a: &[ComparableEntity],
    b: &[ComparableEntity],
    eps: f64,
    changes_only: bool,
) -> (Vec<DiffEntry>, Counts, f64) {
    use std::collections::{HashMap, HashSet};

    let set_a: HashSet<(String, String, usize, usize)> = a.iter().map(|e| e.exact_key()).collect();
    let set_b: HashSet<(String, String, usize, usize)> = b.iter().map(|e| e.exact_key()).collect();
    let inter = set_a.intersection(&set_b).count() as f64;
    let union = set_a.union(&set_b).count() as f64;
    let jaccard = if union == 0.0 { 1.0 } else { inter / union };

    // Phase 1: exact matches by (text,type,start,end)
    let mut remaining_a: Vec<ComparableEntity> = Vec::new();
    let mut remaining_b: Vec<ComparableEntity> = Vec::new();

    let mut map_b_exact: HashMap<(String, String, usize, usize), Vec<ComparableEntity>> =
        HashMap::new();
    for e in b.iter().cloned() {
        map_b_exact.entry(e.exact_key()).or_default().push(e);
    }

    let mut diffs: Vec<DiffEntry> = Vec::new();
    let mut counts = Counts {
        added: 0,
        removed: 0,
        unchanged: 0,
        modified: 0,
    };

    for ea in a.iter().cloned() {
        let key = ea.exact_key();
        if let Some(list) = map_b_exact.get_mut(&key) {
            if let Some(eb) = list.pop() {
                let conf_delta = (ea.confidence - eb.confidence).abs();
                let change_type = if conf_delta <= eps {
                    counts.unchanged += 1;
                    "unchanged"
                } else {
                    counts.modified += 1;
                    "modified"
                };
                if !(changes_only && change_type == "unchanged") {
                    diffs.push(DiffEntry {
                        text: ea.text.clone(),
                        entity_type: ea.entity_type.clone(),
                        change_type: change_type.to_string(),
                        start_a: Some(ea.start),
                        end_a: Some(ea.end),
                        confidence_a: Some(ea.confidence),
                        start_b: Some(eb.start),
                        end_b: Some(eb.end),
                        confidence_b: Some(eb.confidence),
                    });
                }
                continue;
            }
        }
        remaining_a.push(ea);
    }

    for (_k, mut leftover) in map_b_exact {
        remaining_b.append(&mut leftover);
    }

    // Phase 2: loose matches by (text,type) for span shifts / approximate pairing
    let mut by_key_b: HashMap<(String, String), Vec<ComparableEntity>> = HashMap::new();
    for e in remaining_b {
        by_key_b.entry(e.loose_key()).or_default().push(e);
    }
    for list in by_key_b.values_mut() {
        list.sort_by_key(|e| (e.start, e.end));
    }

    for ea in remaining_a {
        let key = ea.loose_key();
        if let Some(candidates) = by_key_b.get_mut(&key) {
            if candidates.is_empty() {
                counts.removed += 1;
                diffs.push(DiffEntry {
                    text: ea.text,
                    entity_type: ea.entity_type,
                    change_type: "removed".to_string(),
                    start_a: Some(ea.start),
                    end_a: Some(ea.end),
                    confidence_a: Some(ea.confidence),
                    start_b: None,
                    end_b: None,
                    confidence_b: None,
                });
                continue;
            }

            // Greedy: choose closest span by absolute start distance.
            let mut best_i = 0usize;
            let mut best_d = (ea.start as i64 - candidates[0].start as i64).abs();
            for (i, eb) in candidates.iter().enumerate().skip(1) {
                let d = (ea.start as i64 - eb.start as i64).abs();
                if d < best_d {
                    best_d = d;
                    best_i = i;
                }
            }
            let eb = candidates.remove(best_i);
            counts.modified += 1;
            diffs.push(DiffEntry {
                text: ea.text.clone(),
                entity_type: ea.entity_type.clone(),
                change_type: "modified".to_string(),
                start_a: Some(ea.start),
                end_a: Some(ea.end),
                confidence_a: Some(ea.confidence),
                start_b: Some(eb.start),
                end_b: Some(eb.end),
                confidence_b: Some(eb.confidence),
            });
        } else {
            counts.removed += 1;
            diffs.push(DiffEntry {
                text: ea.text,
                entity_type: ea.entity_type,
                change_type: "removed".to_string(),
                start_a: Some(ea.start),
                end_a: Some(ea.end),
                confidence_a: Some(ea.confidence),
                start_b: None,
                end_b: None,
                confidence_b: None,
            });
        }
    }

    // Remaining B are added
    for (_k, candidates) in by_key_b {
        for eb in candidates {
            counts.added += 1;
            diffs.push(DiffEntry {
                text: eb.text,
                entity_type: eb.entity_type,
                change_type: "added".to_string(),
                start_a: None,
                end_a: None,
                confidence_a: None,
                start_b: Some(eb.start),
                end_b: Some(eb.end),
                confidence_b: Some(eb.confidence),
            });
        }
    }

    // Ensure deterministic ordering for stable outputs/tests.
    diffs.sort_by(|a, b| {
        a.text
            .cmp(&b.text)
            .then_with(|| a.entity_type.cmp(&b.entity_type))
            .then_with(|| a.change_type.cmp(&b.change_type))
            .then_with(|| {
                a.start_a
                    .unwrap_or(usize::MAX)
                    .cmp(&b.start_a.unwrap_or(usize::MAX))
            })
            .then_with(|| {
                a.start_b
                    .unwrap_or(usize::MAX)
                    .cmp(&b.start_b.unwrap_or(usize::MAX))
            })
    });

    if changes_only {
        diffs.retain(|d| d.change_type != "unchanged");
    }

    (diffs, counts, jaccard)
}
