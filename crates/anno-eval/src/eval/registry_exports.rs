//! Registry-derived exports for tooling (test-only).
//!
//! These are **derived artifacts**. They are generated from the dataset registry so that
//! scripts and docs can consume a stable, machine-readable snapshot.
//!
//! We keep the export payload intentionally small: no benchmark numbers, and no long prose.

use crate::eval::dataset_registry::DatasetId;
use serde::Serialize;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize)]
struct DatasetExport {
    id: String,
    name: &'static str,
    description: &'static str,
    url: &'static str,
    language: &'static str,
    domain: &'static str,
    license: &'static str,
    categories: Vec<&'static str>,
    tasks: Vec<&'static str>,
    entity_types: Vec<&'static str>,
    format: &'static str,
    annotation_scheme: &'static str,
    citation: &'static str,
    paper_url: &'static str,
    year: Option<u16>,
    splits: Vec<&'static str>,
    expected_docs: Option<u32>,
}

fn repo_root() -> PathBuf {
    // CARGO_MANIFEST_DIR = .../anno/crates/anno
    let here = Path::new(env!("CARGO_MANIFEST_DIR"));
    here.parent()
        .and_then(|p| p.parent())
        .expect("repo root (.. / .. from crates/anno)")
        .to_path_buf()
}

fn export_one(id: DatasetId) -> DatasetExport {
    DatasetExport {
        id: format!("{:?}", id),
        name: id.name(),
        description: id.description(),
        url: id.download_url(),
        language: id.language(),
        domain: id.domain(),
        license: id.license().unwrap_or(""),
        categories: id.categories().to_vec(),
        tasks: id.tasks_or_inferred(),
        entity_types: id.entity_types().to_vec(),
        format: id.format().unwrap_or(""),
        annotation_scheme: id.annotation_scheme().unwrap_or(""),
        citation: id.citation().unwrap_or(""),
        paper_url: id.paper_url().unwrap_or(""),
        year: id.year(),
        splits: id.splits().to_vec(),
        expected_docs: id.expected_docs(),
    }
}

fn write_json(path: &Path, datasets: &[DatasetExport]) {
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let bytes = serde_json::to_vec_pretty(&serde_json::json!({ "datasets": datasets }))
        .expect("serialize datasets json");
    fs::write(path, bytes).expect("write datasets json");
}

fn write_jsonl(path: &Path, datasets: &[DatasetExport]) {
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let mut f = fs::File::create(path).expect("create datasets jsonl");
    for ds in datasets {
        let line = serde_json::to_string(ds).expect("serialize jsonl line");
        writeln!(f, "{line}").expect("write jsonl line");
    }
}

fn write_markdown(path: &Path, datasets: &[DatasetExport]) {
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }

    let mut out = String::new();
    out.push_str("<!-- Auto-generated from the Rust dataset registry. DO NOT EDIT MANUALLY. -->\n");
    out.push_str("<!-- Run: cargo test -p anno --features eval generate_datasets_markdown -- --ignored -->\n\n");
    out.push_str("# Dataset Registry\n\n");
    out.push_str(&format!("**Total datasets: {}**\n\n", datasets.len()));
    out.push_str("## All Datasets\n\n");
    out.push_str("| ID | Name | Language | Domain | License | Tasks | Categories |\n");
    out.push_str("|----|------|----------|--------|---------|-------|------------|\n");
    for ds in datasets {
        out.push_str(&format!(
            "| `{}` | {} | {} | {} | {} | {} | {} |\n",
            ds.id,
            ds.name,
            ds.language,
            ds.domain,
            if ds.license.is_empty() {
                "—"
            } else {
                ds.license
            },
            if ds.tasks.is_empty() {
                "—".to_string()
            } else {
                ds.tasks.join(", ")
            },
            if ds.categories.is_empty() {
                "—".to_string()
            } else {
                ds.categories.join(", ")
            }
        ));
    }
    fs::write(path, out).expect("write datasets markdown");
}

#[test]
#[ignore]
fn generate_datasets_json() {
    let root = repo_root();
    let out = root.join("generated/datasets_generated.json");
    let datasets: Vec<DatasetExport> = DatasetId::all().iter().copied().map(export_one).collect();
    write_json(&out, &datasets);
}

#[test]
#[ignore]
fn generate_datasets_jsonl() {
    let root = repo_root();
    let out = root.join("generated/datasets_generated.jsonl");
    let datasets: Vec<DatasetExport> = DatasetId::all().iter().copied().map(export_one).collect();
    write_jsonl(&out, &datasets);
}

#[test]
#[ignore]
fn generate_datasets_markdown() {
    let root = repo_root();
    let out = root.join("docs/generated/DATASETS_GENERATED.md");
    let mut datasets: Vec<DatasetExport> =
        DatasetId::all().iter().copied().map(export_one).collect();
    datasets.sort_by_key(|d| d.id.clone());
    write_markdown(&out, &datasets);
}
