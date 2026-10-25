//! Deterministic character-level PII masking evaluation over a local fixture.
//!
//! This harness intentionally exercises only the dependency-free deterministic path:
//! pattern scanning plus the heuristic NER layer. Built without the `onnx` feature (as
//! in the committed command), `StackedNER::default()` falls back to `RegexNER` +
//! `HeuristicNER`; it does not load ONNX weights, download a model, or contact the
//! network.
//!
//! Run with:
//!
//! ```sh
//! cargo run -p anno-eval --bin pii_char_eval -- \
//!   --metric char-masking \
//!   --fixture crates/anno-eval/fixtures/pii_masking_fixture.jsonl \
//!   --out-json docs/eval/pii-char-masking.json \
//!   --out-md docs/eval/pii-char-masking.md \
//!   --bootstrap
//! ```
//!
//! The emitted JSON is a `synthetic-fixture` artifact, not a benchmark result. See the
//! generated markdown for the exact caveats.

use anno::{Entity, EntityType, Model, StackedNER};
use anno_eval::eval::metrics::{
    evaluate_char_masking_documents, CharMaskingDocument, CharMaskingMetrics,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

const ARTIFACT_KIND: &str = "synthetic-fixture";

#[derive(Debug, Deserialize)]
struct FixtureDocument {
    id: String,
    text: String,
    #[serde(default)]
    source: String,
    entities: Vec<FixtureEntity>,
}

#[derive(Debug, Deserialize)]
struct FixtureEntity {
    text: String,
    #[serde(rename = "type")]
    label: String,
    start: usize,
    end: usize,
}

#[derive(Debug)]
struct Args {
    fixture: PathBuf,
    out_json: PathBuf,
    out_md: PathBuf,
    metric: String,
    bootstrap: bool,
    commit_sha: Option<String>,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let mut args = Self {
            fixture: manifest.join("fixtures/pii_masking_fixture.jsonl"),
            out_json: manifest.join("../../docs/eval/pii-char-masking.json"),
            out_md: manifest.join("../../docs/eval/pii-char-masking.md"),
            metric: "char-masking".to_string(),
            bootstrap: false,
            commit_sha: None,
        };

        let mut iter = std::env::args().skip(1);
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--fixture" => {
                    args.fixture = PathBuf::from(next_value(&mut iter, "--fixture")?);
                }
                "--out-json" => {
                    args.out_json = PathBuf::from(next_value(&mut iter, "--out-json")?);
                }
                "--out-md" => {
                    args.out_md = PathBuf::from(next_value(&mut iter, "--out-md")?);
                }
                "--metric" => {
                    args.metric = next_value(&mut iter, "--metric")?;
                }
                "--commit-sha" => {
                    args.commit_sha = Some(next_value(&mut iter, "--commit-sha")?);
                }
                "--bootstrap" => args.bootstrap = true,
                "--help" | "-h" => {
                    println!(
                        "pii_char_eval --metric char-masking [--fixture PATH] \
                         [--out-json PATH] [--out-md PATH] [--bootstrap] [--commit-sha SHA]"
                    );
                    std::process::exit(0);
                }
                other => return Err(format!("unknown argument: {other}")),
            }
        }

        if args.metric != "char-masking" {
            return Err(format!(
                "unsupported metric '{}'; only 'char-masking' is available in this harness",
                args.metric
            ));
        }
        Ok(args)
    }
}

fn next_value(iter: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    iter.next()
        .ok_or_else(|| format!("{flag} requires a value"))
}

#[derive(Debug, Serialize)]
struct DatasetInfo {
    kind: &'static str,
    name: &'static str,
    fixture_path: String,
    fixture_sha256: String,
    documents: usize,
    public_dataset: &'static str,
    public_split: &'static str,
    note: &'static str,
}

#[derive(Debug, Serialize)]
struct MetricDefinitions {
    precision: &'static str,
    recall: &'static str,
    f1: &'static str,
    f2: &'static str,
    false_positive_rate: &'static str,
    strict_span_f1: &'static str,
    overlap_resolution: &'static str,
    per_entity_type: &'static str,
}

#[derive(Debug, Serialize)]
struct DocumentResult {
    id: String,
    source: String,
    text_chars: usize,
    gold_spans: Vec<Entity>,
    predicted_spans: Vec<Entity>,
    metrics: CharMaskingMetrics,
}

#[derive(Debug, Serialize)]
struct Artifact {
    artifact_kind: &'static str,
    metric: String,
    title: &'static str,
    description: &'static str,
    command: String,
    commit_sha: String,
    commit_sha_source: &'static str,
    working_tree_dirty_at_generation: Option<bool>,
    generated_at_utc: String,
    dataset: DatasetInfo,
    model_backends_exercised: bool,
    backend_scope: &'static str,
    deterministic_paths: Vec<&'static str>,
    metric_definitions: MetricDefinitions,
    results: CharMaskingMetrics,
    documents: Vec<DocumentResult>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("pii_char_eval: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = Args::parse()?;

    let fixture_bytes = fs::read(&args.fixture)
        .map_err(|e| format!("failed to read {}: {e}", args.fixture.display()))?;
    let fixture_sha256 = hex_sha256(&fixture_bytes);
    let fixture_text =
        String::from_utf8(fixture_bytes).map_err(|e| format!("fixture is not valid UTF-8: {e}"))?;

    let model = StackedNER::default();
    let mut documents = Vec::new();
    let mut evaluation_documents = Vec::new();

    for (line_number, line) in fixture_text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let fixture: FixtureDocument = serde_json::from_str(line)
            .map_err(|e| format!("invalid fixture JSON at line {}: {e}", line_number + 1))?;

        let gold: Vec<Entity> = fixture
            .entities
            .iter()
            .map(|entity| {
                Entity::new(
                    entity.text.clone(),
                    EntityType::from_label(&entity.label),
                    entity.start,
                    entity.end,
                    1.0,
                )
            })
            .collect();

        // Deterministic path only: NER entities classified as PII, plus regex patterns.
        let ner_entities = model
            .extract_entities(&fixture.text, None)
            .map_err(|e| format!("heuristic extraction failed for {}: {e}", fixture.id))?;
        let mut predicted: Vec<Entity> = ner_entities
            .iter()
            .filter_map(anno::pii::classify_entity)
            .map(|pii| {
                Entity::new(
                    pii.text,
                    EntityType::from_label(&pii.pii_type),
                    pii.start,
                    pii.end,
                    1.0,
                )
            })
            .collect();
        predicted.extend(
            anno::pii::scan_patterns(&fixture.text)
                .into_iter()
                .map(|pii| {
                    Entity::new(
                        pii.text,
                        EntityType::from_label(&pii.pii_type),
                        pii.start,
                        pii.end,
                        1.0,
                    )
                }),
        );

        let text_chars = fixture.text.chars().count();
        let metrics =
            anno_eval::eval::metrics::evaluate_char_masking(&fixture.text, &gold, &predicted);
        documents.push(DocumentResult {
            id: fixture.id.clone(),
            source: fixture.source.clone(),
            text_chars,
            gold_spans: gold.clone(),
            predicted_spans: predicted.clone(),
            metrics,
        });
        evaluation_documents.push(CharMaskingDocument {
            text: fixture.text,
            gold,
            predicted,
        });
    }

    if documents.is_empty() {
        return Err("fixture contains no documents".to_string());
    }

    let results = evaluate_char_masking_documents(&evaluation_documents, args.bootstrap);
    let mut command = String::new();
    if let Ok(target_dir) = std::env::var("CARGO_TARGET_DIR") {
        if !target_dir.is_empty() {
            let _ = write!(command, "CARGO_TARGET_DIR={target_dir} ");
        }
    }
    let _ = write!(
        command,
        "cargo run -p anno-eval --bin pii_char_eval -- {}",
        std::env::args().skip(1).collect::<Vec<_>>().join(" ")
    );
    let commit_sha = args.commit_sha.unwrap_or_else(git_head_sha);

    let artifact = Artifact {
        artifact_kind: ARTIFACT_KIND,
        metric: args.metric,
        title: "PII character-masking evaluation",
        description: "Deterministic pattern + heuristic evaluation over a local fixture. \
                      Not a benchmark: no neural/ML model backends were exercised and no \
                      public PII dataset split was used.",
        command,
        commit_sha,
        commit_sha_source: "git rev-parse HEAD at generation time (the artifact is committed \
                            in a later local commit on the same branch)",
        working_tree_dirty_at_generation: working_tree_dirty(),
        generated_at_utc: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        dataset: DatasetInfo {
            kind: ARTIFACT_KIND,
            name: "repo testdata + synthetic PII documents",
            fixture_path: args.fixture.display().to_string(),
            fixture_sha256,
            documents: documents.len(),
            public_dataset: "ai4privacy/pii-masking-200k",
            public_split: "train (the registry declares only train; no dev/test split is declared)",
            note: "Small hand-assembled fixture. A credible public benchmark requires the \
                   ai4privacy/pii-masking-200k train split with a documented held-out \
                   carve-out (or an equivalent PII corpus with a declared held-out split), \
                   plus one or more neural/ML backends.",
        },
        model_backends_exercised: false,
        backend_scope: "deterministic pattern + heuristic layers only; no neural/ML backends",
        deterministic_paths: vec!["pattern (regex PII scan)", "heuristic NER (StackedNER)"],
        metric_definitions: MetricDefinitions {
            precision: "correctly masked characters / all masked characters",
            recall: "correctly masked characters / all gold PII characters",
            f1: "harmonic mean of character precision and recall",
            f2: "5PR / (4P + R); recall-weighted, beta = 2",
            false_positive_rate: "over-masked non-PII characters / all non-PII characters",
            strict_span_f1: "exact-boundary span F1 (SemEval exact schema; type ignored)",
            overlap_resolution: "overlapping predicted spans are merged into the union of \
                                 masked characters; the same union is applied to gold spans",
            per_entity_type: "each predicted span is attributed to the gold type with the \
                              largest character overlap; unattributed over-masks appear only \
                              in the overall FPR",
        },
        results,
        documents,
    };

    write_outputs(&args.out_json, &args.out_md, &artifact)
}

fn write_outputs(out_json: &Path, out_md: &Path, artifact: &Artifact) -> Result<(), String> {
    if let Some(parent) = out_json.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("failed to create {}: {e}", parent.display()))?;
    }
    if let Some(parent) = out_md.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("failed to create {}: {e}", parent.display()))?;
    }

    let json = serde_json::to_string_pretty(artifact)
        .map_err(|e| format!("failed to serialize artifact: {e}"))?;
    fs::write(out_json, format!("{json}\n"))
        .map_err(|e| format!("failed to write {}: {e}", out_json.display()))?;
    fs::write(out_md, markdown(artifact))
        .map_err(|e| format!("failed to write {}: {e}", out_md.display()))?;

    println!(
        "wrote {} and {} ({} documents, {} result kind)",
        out_json.display(),
        out_md.display(),
        artifact.documents.len(),
        artifact.artifact_kind
    );
    Ok(())
}

fn markdown(artifact: &Artifact) -> String {
    let results = &artifact.results;
    let mut out = String::new();

    let _ = writeln!(out, "# PII character-masking evaluation");
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "> **Artifact kind: `{}`.** This is **not a benchmark**. No neural/ML model \
         backends were exercised; only deterministic pattern scanning and the heuristic \
         NER layer were used. The fixture is small and hand-assembled, so the numbers \
         describe this fixture, not production accuracy.",
        artifact.artifact_kind
    );
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "| Field | Value |\n|---|---|\n| Metric | `{}` |\n| Command | `{}` |\n| \
         Commit | `{}` |\n| Generated (UTC) | {} |\n| Fixture | `{}` |\n| \
         Fixture SHA-256 | `{}` |\n| Documents | {} |\n| Neural/ML model backends | not exercised |\n| \
         Backend scope | {} |\n| Deterministic paths | {} |\n| Public dataset | `{}` |\n| \
         Public split | {} |",
        artifact.metric,
        artifact.command,
        artifact.commit_sha,
        artifact.generated_at_utc,
        artifact.dataset.fixture_path,
        artifact.dataset.fixture_sha256,
        artifact.documents.len(),
        artifact.backend_scope,
        artifact
            .deterministic_paths
            .iter()
            .map(|path| format!("`{path}`"))
            .collect::<Vec<_>>()
            .join(", "),
        artifact.dataset.public_dataset,
        artifact.dataset.public_split,
    );
    let _ = writeln!(out);

    let _ = writeln!(out, "## Headline metrics");
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "| Metric | Value |\n|---|---:|\n| Character precision | {:.2}% |\n| \
         Character recall | {:.2}% |\n| Character F1 | {:.2}% |\n| Character F2 (beta=2) | {:.2}% |\n| \
         False-positive rate (non-PII) | {:.2}% |\n| Strict span F1 (exact boundary) | {:.2}% |",
        results.precision * 100.0,
        results.recall * 100.0,
        results.f1 * 100.0,
        results.f2 * 100.0,
        results.false_positive_rate * 100.0,
        results.strict_span_f1 * 100.0,
    );
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "Counts: `TP={}` masked PII characters, `masked={}` predicted characters, \
         `gold={}` PII characters, `non_pii={}` non-PII characters, `total={}` characters.",
        results.counts.true_positive_chars,
        results.counts.predicted_chars,
        results.counts.gold_chars,
        results.counts.non_pii_chars,
        results.counts.total_chars,
    );
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "F2 is below F1 here because pooled precision exceeds pooled recall; F2 weights \
         the recall shortfall more heavily. That is the intended bias for redaction: \
         missing PII characters is worse than a small over-mask."
    );
    let _ = writeln!(out);

    let _ = writeln!(out, "## Per-entity-type breakdown");
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "| Type | Gold chars | Predicted chars | TP | Precision | Recall | F1 | F2 | FPR |\n\
         |---|---:|---:|---:|---:|---:|---:|---:|---:|"
    );
    let mut types: Vec<_> = results.by_entity_type.values().collect();
    types.sort_by(|a, b| a.entity_type.cmp(&b.entity_type));
    for type_metrics in types {
        let _ = writeln!(
            out,
            "| {} | {} | {} | {} | {:.2}% | {:.2}% | {:.2}% | {:.2}% | {:.2}% |",
            type_metrics.entity_type,
            type_metrics.gold_chars,
            type_metrics.predicted_chars,
            type_metrics.true_positive_chars,
            type_metrics.precision * 100.0,
            type_metrics.recall * 100.0,
            type_metrics.f1 * 100.0,
            type_metrics.f2 * 100.0,
            type_metrics.false_positive_rate * 100.0,
        );
    }
    let _ = writeln!(out);

    let _ = writeln!(out, "## Documents");
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "| Document | Source | Chars | Gold spans | Predicted spans | Recall | FPR |\n\
         |---|---|---:|---:|---:|---:|---:|"
    );
    for document in &artifact.documents {
        let _ = writeln!(
            out,
            "| {} | {} | {} | {} | {} | {:.2}% | {:.2}% |",
            document.id,
            document.source,
            document.text_chars,
            document.gold_spans.len(),
            document.predicted_spans.len(),
            document.metrics.recall * 100.0,
            document.metrics.false_positive_rate * 100.0,
        );
    }
    let _ = writeln!(out);

    let _ = writeln!(out, "## Metric definitions");
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "- **Precision**: {}\n- **Recall**: {}\n- **F1**: {}\n- **F2**: {}\n- **FPR**: {}\n- \
         **Strict span F1**: {}\n- **Overlap resolution**: {}\n- **Per-type breakdown**: {}",
        artifact.metric_definitions.precision,
        artifact.metric_definitions.recall,
        artifact.metric_definitions.f1,
        artifact.metric_definitions.f2,
        artifact.metric_definitions.false_positive_rate,
        artifact.metric_definitions.strict_span_f1,
        artifact.metric_definitions.overlap_resolution,
        artifact.metric_definitions.per_entity_type,
    );
    let _ = writeln!(out);

    let _ = writeln!(out, "## What this artifact does not establish");
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "- It is not a result on `ai4privacy/pii-masking-200k` or any other public PII \
         benchmark split. The dataset registry declares only the `train` split for \
         `ai4privacy/pii-masking-200k`, so an outsider-credible artifact also needs a \
         documented held-out carve-out.\n- It does not exercise neural/ML backends \
         (ONNX, Candle, GLiNER) or trained classical backends (CRF, HMM).\n- The fixture \
         is too small and too synthetic to support an accuracy claim about production \
         redaction.\n- Per-type rows are diagnostic, not additive: a predicted span is \
         attributed to one gold type."
    );

    out
}

fn git_head_sha() -> String {
    git_output(["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".to_string())
}

fn working_tree_dirty() -> Option<bool> {
    git_output(["status", "--porcelain"]).map(|status| !status.trim().is_empty())
}

fn git_output<const N: usize>(args: [&str; N]) -> Option<String> {
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(env!("CARGO_MANIFEST_DIR"))
        .args(args)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout)
        .ok()
        .map(|value| value.trim().to_string())
}

fn hex_sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        let _ = write!(out, "{byte:02x}");
    }
    out
}
