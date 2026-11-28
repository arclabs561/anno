//! Query recorded outcomes without starting a loader or an evaluation.

use crate::cli::commands::figure_catalog::{Catalog, Dataset, Record};
use clap::{Args, ValueEnum};
use serde::Serialize;
use std::collections::BTreeMap;
use std::io::{self, Write};
use std::path::PathBuf;

#[derive(Debug, Args)]
#[command(
    after_help = "Examples:\n  anno dataset evidence WikiGold --status scored\n  anno dataset evidence --backend gliner_multitask --status failed\n  anno dataset evidence --domain clinical --status unindexed --json"
)]
pub struct EvidenceArgs {
    /// Select an exact registry ID or search IDs and names by case-insensitive substring
    #[arg(value_name = "DATASET")]
    query: Option<String>,
    /// Restrict registry entries to this domain
    #[arg(long)]
    domain: Option<String>,
    /// Require an observed task on a matching record (such as ner or coref)
    #[arg(long)]
    task: Option<String>,
    /// Require this requested or reported backend on the same record
    #[arg(long)]
    backend: Option<String>,
    /// Restrict the recorded split (case-insensitive substring)
    #[arg(long)]
    split: Option<String>,
    /// Filter by the evidence actually recorded for a dataset
    #[arg(long, value_enum)]
    status: Option<Status>,
    /// List every registry ID (without filters the text view prints a summary)
    #[arg(long)]
    all: bool,
    /// Emit each matching outcome with its sample scope and original source
    #[arg(long)]
    json: bool,
    /// Read a local figure-catalog.json (default discovers a checkout or unpacked figure bundle)
    #[arg(long, env = "ANNO_FIGURE_CATALOG", value_name = "PATH")]
    catalog: Option<PathBuf>,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum Status {
    Scored,
    Failed,
    Empty,
    Unsupported,
    LoadRejected,
    LoadChecked,
    ExporterReplay,
    Unindexed,
}

impl Status {
    fn label(self) -> &'static str {
        match self {
            Self::Scored => "Scored",
            Self::Failed => "Failed evaluation",
            Self::Empty => "Empty evaluation",
            Self::Unsupported => "Unsupported input",
            Self::LoadRejected => "Load rejected",
            Self::LoadChecked => "Passed load check",
            Self::ExporterReplay => "Exporter replay",
            Self::Unindexed => "No indexed evaluation",
        }
    }
}

fn same_name(value: &str, query: &str) -> bool {
    value.eq_ignore_ascii_case(query)
}

fn contains(value: &str, query: &str) -> bool {
    value.to_lowercase().contains(&query.to_lowercase())
}

impl EvidenceArgs {
    fn matches_dataset(&self, dataset: &Dataset, exact_id: bool) -> bool {
        self.query.as_ref().is_none_or(|q| {
            if exact_id {
                &dataset.id == q
            } else {
                contains(&dataset.id, q) || contains(&dataset.name, q)
            }
        }) && self
            .domain
            .as_ref()
            .is_none_or(|d| same_name(&dataset.domain, d))
    }

    fn matches_record(&self, record: &Record) -> bool {
        self.status.is_none_or(|s| s.label() == record.status)
            && self
                .task
                .as_ref()
                .is_none_or(|t| same_name(&record.task, t))
            && self
                .split
                .as_ref()
                .is_none_or(|s| contains(&record.split, s))
            && self.backend.as_ref().is_none_or(|b| {
                same_name(&record.backend, b)
                    || record
                        .reported_backend
                        .split(" | ")
                        .any(|v| same_name(v, b))
            })
    }
}

#[derive(Serialize)]
struct Match<'a> {
    id: &'a str,
    name: &'a str,
    domain: &'a str,
    measurement: &'a str,
    coverage_action: &'a str,
    declared_tasks: &'a [String],
    declared_splits: &'a [String],
    records: Vec<&'a Record>,
    studies: Vec<&'a str>,
}

pub fn run(args: EvidenceArgs) -> Result<(), String> {
    let catalog = Catalog::load(args.catalog.as_deref())?;
    let include_unindexed = args.task.is_none()
        && args.backend.is_none()
        && args.split.is_none()
        && matches!(args.status, None | Some(Status::Unindexed));
    let mut found = Vec::new();
    let exact_id = args
        .query
        .as_ref()
        .is_some_and(|q| catalog.datasets.iter().any(|d| &d.id == q));
    for dataset in catalog
        .datasets
        .iter()
        .filter(|d| args.matches_dataset(d, exact_id))
    {
        let records: Vec<_> = dataset
            .records
            .iter()
            .filter(|r| args.matches_record(r))
            .collect();
        if records.is_empty() && !(include_unindexed && dataset.records.is_empty()) {
            continue;
        }
        let studies = catalog
            .studies
            .iter()
            .filter(|s| s.datasets.contains(&dataset.id))
            .map(|s| s.id.as_str())
            .collect();
        found.push(Match {
            id: &dataset.id,
            name: &dataset.name,
            domain: &dataset.domain,
            measurement: &dataset.measurement,
            coverage_action: &dataset.coverage_action,
            declared_tasks: &dataset.declared_tasks,
            declared_splits: &dataset.declared_splits,
            records,
            studies,
        });
    }
    let mut out = io::BufWriter::new(io::stdout().lock());
    if args.json {
        let output = serde_json::json!({
            "schema": "anno.dataset-evidence/v1", "ci_survey_day": catalog.ci_survey_day,
            "registry_ids": catalog.datasets.len(), "matched_ids": found.len(), "datasets": found,
        });
        serde_json::to_writer_pretty(&mut out, &output).map_err(|e| e.to_string())?;
        writeln!(out).map_err(|e| e.to_string())?;
    } else {
        write_results(&mut out, &found, &catalog, &args).map_err(|e| e.to_string())?;
    }
    out.flush().map_err(|e| e.to_string())
}

fn write_results(
    out: &mut impl Write,
    found: &[Match<'_>],
    catalog: &Catalog,
    args: &EvidenceArgs,
) -> io::Result<()> {
    write!(
        out,
        "{} of {} registry IDs",
        found.len(),
        catalog.datasets.len()
    )?;
    if let Some(day) = &catalog.ci_survey_day {
        write!(out, " · CI survey {day} UTC and recorded corpus studies")?;
    }
    writeln!(out)?;
    let scored = found
        .iter()
        .filter(|item| item.records.iter().any(|r| r.status == "Scored"))
        .count();
    let unindexed = found.iter().filter(|item| item.records.is_empty()).count();
    writeln!(
        out,
        "Unique IDs with scored inputs  {scored}   Without an indexed evaluation  {unindexed}"
    )?;
    if !args.all
        && args.query.is_none()
        && args.status.is_none()
        && args.domain.is_none()
        && args.backend.is_none()
        && args.task.is_none()
        && args.split.is_none()
    {
        return writeln!(out, "Inspect a dataset with  anno dataset evidence WikiGold\nShow scored inputs with  anno dataset evidence --status scored\nList every ID with  anno dataset evidence --all");
    }
    for dataset in found {
        let mut statuses = BTreeMap::<&str, usize>::new();
        for record in &dataset.records {
            *statuses.entry(&record.status).or_default() += 1;
        }
        let outcomes = if statuses.is_empty() {
            "No indexed evaluation".into()
        } else {
            statuses
                .iter()
                .map(|(s, count)| format!("{s} × {count}"))
                .collect::<Vec<_>>()
                .join(", ")
        };
        writeln!(
            out,
            "{}\t{}\t{}\t{outcomes}",
            dataset.id, dataset.domain, dataset.measurement
        )?;
        if found.len() == 1 || args.query.is_some() && found.len() <= 5 {
            if dataset.records.is_empty() {
                writeln!(out, "  {}", dataset.coverage_action)?;
            } else {
                let mut groups: BTreeMap<_, Vec<&Record>> = BTreeMap::new();
                for record in &dataset.records {
                    groups
                        .entry((
                            &record.task,
                            &record.backend,
                            &record.split,
                            &record.status,
                            &record.unit,
                        ))
                        .or_default()
                        .push(record);
                }
                for ((task, backend, split, status, unit), records) in groups {
                    let backend = if backend.is_empty() {
                        "no backend recorded"
                    } else {
                        backend
                    };
                    write!(out, "  {task} · {backend} · {split} · {status}")?;
                    let sizes: Vec<_> = records.iter().filter_map(|r| r.examples).collect();
                    if let (Some(min), Some(max)) = (sizes.iter().min(), sizes.iter().max()) {
                        if min == max {
                            write!(out, " · {min} {unit} per measured record")?;
                        } else {
                            write!(out, " · {min}–{max} {unit} per measured record")?;
                        }
                    }
                    let count = records.len();
                    writeln!(
                        out,
                        " ({count} {})",
                        if count == 1 { "record" } else { "records" }
                    )?;
                }
            }
            for id in &dataset.studies {
                writeln!(out, "  anno viz render {id} -o {id}.html")?;
            }
        }
    }
    if found.is_empty() {
        writeln!(out, "No matching evidence. Use fewer filters or search the dataset name with 'anno dataset evidence NAME'.")?;
    } else {
        writeln!(
            out,
            "Use --json for individual scopes, sample sizes and source receipts."
        )?;
    }
    Ok(())
}
