//! Compare command - Compare documents, models, or clusters

use clap::Parser;
use std::fs;

use super::super::parser::ModelBackend;
use anno::Entity;
use anno::GroundedDocument;

use serde::{Deserialize, Serialize};

/// Compare documents, models, or clusters
#[derive(Parser, Debug)]
pub struct CompareArgs {
    /// First input file
    #[arg(value_name = "FILE1")]
    pub file1: String,

    /// Second saved document for document comparison
    #[arg(value_name = "FILE2")]
    pub file2: Option<String>,

    /// Compare models on the same input text (reads FILE1 from disk)
    #[arg(long, conflicts_with_all = ["file2", "changes_only", "confidence_epsilon"], requires = "model_list")]
    pub models: bool,

    /// Models to compare (when --models is used)
    #[arg(long, value_delimiter = ',', value_name = "MODEL", requires = "models")]
    pub model_list: Vec<ModelBackend>,

    /// Document diff, summary, JSON or JSONL; model comparison also supports table
    #[arg(long, default_value = "diff", value_parser = ["diff", "summary", "table", "json", "jsonl"])]
    pub format: String,

    /// Confidence delta threshold for treating an otherwise-identical entity as modified
    #[arg(long, default_value_t = 0.05, value_parser = super::super::utils::parse_confidence)]
    pub confidence_epsilon: f64,

    /// Only emit changed entities (added/removed/modified); omit unchanged diffs
    #[arg(long)]
    pub changes_only: bool,

    /// Output file
    #[arg(short, long, value_name = "PATH")]
    pub output: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
struct ComparableEntity {
    text: String,
    #[serde(rename = "type")]
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

        for backend in &args.model_list {
            let model = backend.create_model()?;
            let entities = model
                .extract_entities(&text, None)
                .map_err(|e| format!("Model {} failed: {}", backend.name(), e))?;
            results.push((backend.name().to_string(), entities));
        }

        let output = match args.format.as_str() {
            "json" | "jsonl" => {
                let rows: Vec<_> = results
                    .iter()
                    .map(
                        |(model, entities)| serde_json::json!({"model":model, "entities":entities}),
                    )
                    .collect();
                if args.format == "json" {
                    serde_json::to_string_pretty(&rows).map_err(|e| e.to_string())?
                } else {
                    rows.iter().map(|r| format!("{r}\n")).collect()
                }
            }
            "table" | "summary" => {
                let mut output = String::from("\nModel Comparison:\n");
                output.push_str(&format!("{:<15} {:<10}\n", "Model", "Entities"));
                output.push_str(&format!("{}\n", "-".repeat(25)));
                for (name, entities) in &results {
                    output.push_str(&format!("{:<15} {:<10}\n", name, entities.len()));
                }
                output
            }
            _ => {
                let mut output = String::new();
                for (name, entities) in &results {
                    output.push_str(&format!("\n{} ({} entities):\n", name, entities.len()));
                    for e in entities {
                        output.push_str(&format!(
                            "  - {} ({})\n",
                            e.text,
                            e.entity_type.as_label()
                        ));
                    }
                }
                output
            }
        };
        write_or_print_output(args.output.as_deref(), &output)?;
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

        let output = match args.format.as_str() {
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
                serde_json::to_string_pretty(&out)
                    .map_err(|e| format!("Failed to serialize comparison: {e}"))?
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
                let mut output = format!("{summary}\n");
                for d in diffs {
                    output.push_str(
                        &serde_json::to_string(&d)
                            .map_err(|e| format!("Failed to serialize comparison: {e}"))?,
                    );
                    output.push('\n');
                }
                output
            }
            "diff" | "summary" => {
                // Keep legacy human output for interactive usage.
                let mut output = format!("\nComparison: {} vs {}\n", args.file1, file2);
                output.push_str(&format!(
                    "added={} removed={} modified={} unchanged={} jaccard={:.3}\n",
                    counts.added, counts.removed, counts.modified, counts.unchanged, jaccard
                ));
                for d in diffs.iter().filter(|_| args.format == "diff") {
                    output.push_str(&format!(
                        "  {}: {} [{}]\n",
                        d.change_type, d.text, d.entity_type
                    ));
                }
                output
            }
            _ => {
                return Err(format!(
                    "Unknown format: {}. Use: diff, summary, json, jsonl",
                    args.format
                ))
            }
        };
        write_or_print_output(args.output.as_deref(), &output)?;
    }

    Ok(())
}

fn write_or_print_output(path: Option<&str>, output: &str) -> Result<(), String> {
    if let Some(path) = path {
        fs::write(path, output).map_err(|e| format!("Failed to write output: {e}"))
    } else {
        print!("{output}");
        Ok(())
    }
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
        return arr
            .iter()
            .enumerate()
            .map(|(i, row)| {
                let entity: ComparableEntity = serde_json::from_value(row.clone())
                    .map_err(|e| format!("entities[{i}]: {e}"))?;
                if entity.start > entity.end || !(0.0..=1.0).contains(&entity.confidence) {
                    return Err(format!(
                        "entities[{i}]: expected ordered offsets and confidence in 0..=1"
                    ));
                }
                Ok(entity)
            })
            .collect();
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
                    confidence: s.confidence.value(),
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

    let mut groups = std::collections::BTreeMap::new();
    for e in a {
        groups
            .entry(e.exact_key())
            .or_insert_with(|| (Vec::new(), Vec::new()))
            .0
            .push(e);
    }
    for e in b {
        groups
            .entry(e.exact_key())
            .or_insert_with(|| (Vec::new(), Vec::new()))
            .1
            .push(e);
    }

    let mut diffs: Vec<DiffEntry> = Vec::new();
    let mut counts = Counts {
        added: 0,
        removed: 0,
        unchanged: 0,
        modified: 0,
    };

    for (_, (mut left, mut right)) in groups {
        left.sort_by(|a, b| a.confidence.total_cmp(&b.confidence));
        right.sort_by(|a, b| a.confidence.total_cmp(&b.confidence));
        let mut left = left.into_iter().peekable();
        let mut right = right.into_iter().peekable();
        let mut unmatched_left = Vec::new();
        let mut unmatched_right = Vec::new();
        let mut pairs = Vec::new();
        // In sorted order, the smaller out-of-tolerance score cannot match
        // any later score on the opposite side.
        while let (Some(&ea), Some(&eb)) = (left.peek(), right.peek()) {
            if (ea.confidence - eb.confidence).abs() <= eps {
                pairs.push((ea, eb, "unchanged"));
                left.next();
                right.next();
            } else if ea.confidence < eb.confidence {
                unmatched_left.push(ea);
                left.next();
            } else {
                unmatched_right.push(eb);
                right.next();
            }
        }
        unmatched_left.extend(left);
        unmatched_right.extend(right);
        let common = unmatched_left.len().min(unmatched_right.len());
        pairs.extend(
            unmatched_left
                .drain(..common)
                .zip(unmatched_right.drain(..common))
                .map(|(ea, eb)| (ea, eb, "modified")),
        );
        remaining_a.extend(unmatched_left.into_iter().cloned());
        remaining_b.extend(unmatched_right.into_iter().cloned());
        for (ea, eb, change) in pairs {
            if change == "unchanged" {
                counts.unchanged += 1;
                if changes_only {
                    continue;
                }
            } else {
                counts.modified += 1;
            }
            diffs.push(DiffEntry {
                text: ea.text.clone(),
                entity_type: ea.entity_type.clone(),
                change_type: change.into(),
                start_a: Some(ea.start),
                end_a: Some(ea.end),
                confidence_a: Some(ea.confidence),
                start_b: Some(eb.start),
                end_b: Some(eb.end),
                confidence_b: Some(eb.confidence),
            });
        }
    }

    // Phase 2: loose matches by (text,type) for span shifts / approximate pairing
    let mut by_key_b: HashMap<(String, String), Vec<ComparableEntity>> = HashMap::new();
    for e in remaining_b {
        by_key_b.entry(e.loose_key()).or_default().push(e);
    }
    for list in by_key_b.values_mut() {
        list.sort_by(|a, b| {
            (a.start, a.end)
                .cmp(&(b.start, b.end))
                .then_with(|| a.confidence.total_cmp(&b.confidence))
        });
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
            let mut best_d = ea.start.abs_diff(candidates[0].start);
            for (i, eb) in candidates.iter().enumerate().skip(1) {
                let d = ea.start.abs_diff(eb.start);
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

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    /// Wrapper so we can test CompareArgs parsing in isolation.
    #[derive(Parser)]
    struct TestCli {
        #[command(flatten)]
        args: CompareArgs,
    }

    #[test]
    fn model_list_accepts_all_always_available_backends() {
        // These backends are always available (no feature gates).
        let cli = TestCli::try_parse_from([
            "test",
            "--models",
            "--model-list",
            "stacked,pattern,heuristic,crf,hmm,ensemble,tplinker",
            "input.txt",
        ])
        .expect("should parse valid model names");
        assert_eq!(cli.args.model_list.len(), 7);
        assert!(matches!(cli.args.model_list[0], ModelBackend::Stacked));
        assert!(matches!(cli.args.model_list[1], ModelBackend::Pattern));
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn model_list_accepts_onnx_backends() {
        let cli = TestCli::try_parse_from([
            "test",
            "--models",
            "--model-list",
            "stacked,bert-onnx,gliner",
            "input.txt",
        ])
        .expect("should parse bert-onnx and gliner with onnx feature");
        assert_eq!(cli.args.model_list.len(), 3);
        assert!(matches!(cli.args.model_list[1], ModelBackend::BertOnnx));
    }

    #[test]
    fn model_list_rejects_unknown_backend() {
        let result = TestCli::try_parse_from([
            "test",
            "--models",
            "--model-list",
            "nonexistent",
            "input.txt",
        ]);
        assert!(
            result.is_err(),
            "Unknown backend should be rejected by clap"
        );
    }

    #[test]
    fn compare_command_writes_json_output() {
        let dir = tempfile::tempdir().unwrap();
        let file1 = dir.path().join("first.json");
        let file2 = dir.path().join("second.json");
        let output = dir.path().join("comparison.json");
        fs::write(
            &file1,
            r#"{"entities":[{"text":"Alice","type":"PER","start":0,"end":5,"confidence":0.9}]}"#,
        )
        .unwrap();
        fs::write(&file2, r#"{"entities":[]}"#).unwrap();

        run(CompareArgs {
            file1: file1.to_string_lossy().into_owned(),
            file2: Some(file2.to_string_lossy().into_owned()),
            models: false,
            model_list: Vec::new(),
            format: "json".to_string(),
            confidence_epsilon: 0.05,
            changes_only: false,
            output: Some(output.to_string_lossy().into_owned()),
        })
        .unwrap();

        let output: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(output).unwrap()).unwrap();
        assert_eq!(output["removed"], 1);
        assert_eq!(output["diffs"][0]["text"], "Alice");
    }
}
