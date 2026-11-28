use super::{distinct_output, errors, open_report, overlap, ReportPaths, VizFormat};
use crate::cli::commands::figure_catalog::{Catalog, Study};
use anno::viz::errors::ERRORS_SCHEMA;
use anno::viz::overlap::OVERLAP_SCHEMA;
use anno::viz::{Paired, Tradeoff, PAIRED_SCHEMA, TRADEOFF_SCHEMA};
use clap::Args;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::io::{self, Cursor, Read, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, Args)]
struct CatalogArgs {
    /// Use this local figure catalog (defaults to the current bundle or checkout)
    #[arg(long = "catalog", env = "ANNO_FIGURE_CATALOG", value_name = "PATH")]
    path: Option<PathBuf>,
}

#[derive(Debug, Args)]
pub(super) struct ListArgs {
    /// Limit studies to this dataset name (ignores case, spaces and punctuation)
    #[arg(long, value_name = "NAME")]
    dataset: Option<String>,
    /// Emit a JSON array with the ID, title, schema, datasets and guide of each study
    #[arg(long)]
    json: bool,
    #[command(flatten)]
    catalog: CatalogArgs,
}

#[derive(Serialize)]
struct Listing<'a> {
    id: &'a str,
    title: &'a str,
    schema: &'a str,
    datasets: &'a [String],
    guide: &'a str,
}

impl<'a> From<&'a Study> for Listing<'a> {
    fn from(study: &'a Study) -> Self {
        Self {
            id: &study.id,
            title: study.report["title"].as_str().unwrap_or(&study.id),
            schema: study.report["schema"].as_str().unwrap_or(""),
            datasets: &study.datasets,
            guide: &study.guide,
        }
    }
}

fn dataset_key(name: &str) -> String {
    name.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn matching_datasets<'a>(catalog: &'a Catalog, query: &'a str) -> BTreeSet<&'a str> {
    let tags: BTreeSet<_> = catalog
        .studies
        .iter()
        .flat_map(|study| study.datasets.iter().map(String::as_str))
        .collect();
    // Registry IDs can differ only in case. An explicitly named ID selects that dataset alone.
    if tags.contains(query) || catalog.datasets.iter().any(|dataset| dataset.id == query) {
        return BTreeSet::from([query]);
    }
    let key = dataset_key(query);
    tags.into_iter()
        .filter(|id| dataset_key(id) == key)
        .chain(
            catalog
                .datasets
                .iter()
                .filter(|d| dataset_key(&d.name) == key)
                .map(|d| d.id.as_str()),
        )
        .collect()
}

impl ListArgs {
    pub(super) fn run(self) -> Result<(), String> {
        let catalog = Catalog::load(self.catalog.path.as_deref())?;
        let selected = self
            .dataset
            .as_deref()
            .map(|name| matching_datasets(&catalog, name.trim()));
        let studies: Vec<_> = catalog
            .studies
            .iter()
            .filter(|s| {
                selected
                    .as_ref()
                    .is_none_or(|ids| s.datasets.iter().any(|d| ids.contains(d.as_str())))
            })
            .map(Listing::from)
            .collect();
        let stdout = io::stdout().lock();
        let mut output = io::BufWriter::new(stdout);
        if self.json {
            serde_json::to_writer_pretty(&mut output, &studies).map_err(|e| e.to_string())?;
            writeln!(output)
        } else if studies.is_empty() {
            writeln!(
                output,
                "No matching studies. Run `anno viz list` to see the catalog."
            )
        } else {
            for row in studies {
                let label = if row.datasets.is_empty() {
                    row.id.to_owned()
                } else {
                    format!("{}  [{}]", row.id, row.datasets.join(", "))
                };
                writeln!(output, "{label}\n  {}", row.title).map_err(|e| e.to_string())?;
                if !row.guide.is_empty() {
                    writeln!(output, "  {}", row.guide).map_err(|e| e.to_string())?;
                }
            }
            Ok(())
        }
        .and_then(|_| output.flush())
        .map_err(|e| e.to_string())
    }
}

#[derive(Debug, Args)]
pub(super) struct PortableArgs {
    /// Study ID or local report path (existing paths take precedence, '-' or omitted reads stdin)
    #[arg(value_name = "STUDY|INPUT")]
    input: Option<PathBuf>,
    /// Write to a file (default or '-' writes stdout)
    #[arg(short, long, value_name = "PATH")]
    output: Option<PathBuf>,
    /// Output format (default SVG, inferred from .html, .htm or .json filenames)
    #[arg(long, value_enum)]
    format: Option<VizFormat>,
    #[command(flatten)]
    catalog: CatalogArgs,
}

impl PortableArgs {
    pub(super) fn run(self) -> Result<(), String> {
        let catalog = self.catalog.path.as_deref();
        let paths = ReportPaths {
            input: self.input,
            output: self.output,
        };
        let output = paths.output.as_deref().filter(|p| *p != Path::new("-"));
        paths.render_from(
            |path| open_study_or_file(path, catalog, output),
            |mut input, output| {
                let mut bytes = Vec::new();
                input.read_to_end(&mut bytes).map_err(|e| e.to_string())?;
                render(
                    &bytes,
                    self.format.unwrap_or_else(|| VizFormat::for_output(output)),
                )
            },
        )
    }
}

fn open_study_or_file(
    path: &Path,
    override_path: Option<&Path>,
    output: Option<&Path>,
) -> Result<Box<dyn Read>, String> {
    // Only a bare missing name can be a catalog ID. Explicit file paths retain
    // their OS error and never consult the catalog, even outside the checkout.
    let error = match open_report(path) {
        Ok(reader) => return Ok(reader),
        Err(error) => error,
    };
    let Some(name) = path.to_str().filter(|_| {
        error.kind() == io::ErrorKind::NotFound && path.file_name() == Some(path.as_os_str())
    }) else {
        return Err(format!("{}: {error}", path.display()));
    };
    let catalog_path = Catalog::path(override_path)?;
    if let Some(output) = output {
        distinct_output(&catalog_path, output)?;
    }
    let catalog = Catalog::load(Some(&catalog_path))?;
    let study = catalog
        .studies
        .into_iter()
        .find(|s| s.id == name)
        .ok_or_else(|| {
            format!("Unknown study or missing file {name:?}. Run `anno viz list` for study IDs.")
        })?;
    let bytes = serde_json::to_vec_pretty(&study.report).map_err(|e| e.to_string())?;
    Ok(Box::new(Cursor::new(bytes)))
}

const RAW_HELP: &str = "Use `anno viz errors` for raw `anno eval --json` reports or `anno viz overlap` for grounded documents. Compare two eval streams with `anno viz paired --against CANDIDATE`.";

#[derive(Deserialize)]
struct Header {
    schema: Option<String>,
}

fn render(bytes: &[u8], format: VizFormat) -> Result<String, String> {
    // Leave validation, unknown fields and numeric checks to each native view.
    // Parsing the small discriminator without round-tripping the original file
    // preserves its exact integers and the underlying reader's diagnostics.
    let header: Header = serde_json::from_slice(bytes)
        .map_err(|e| format!("Expected one portable Anno JSON report ({e}). {RAW_HELP}"))?;
    let rendered = match header.schema.as_deref() {
        Some(ERRORS_SCHEMA) => errors::render(bytes, format, None),
        Some(OVERLAP_SCHEMA) => overlap::render(bytes, format, None),
        Some(PAIRED_SCHEMA) => {
            let report = Paired::from_json(bytes)?;
            match format {
                VizFormat::Svg => report.render_svg(),
                VizFormat::Html => report.render_html(),
                VizFormat::Json => report.to_json(),
            }
        }
        Some(TRADEOFF_SCHEMA) => {
            let report = Tradeoff::from_json(bytes)?;
            match format {
                VizFormat::Svg => report.render_svg(),
                VizFormat::Html => report.render_html(),
                VizFormat::Json => report.to_json(),
            }
        }
        Some(schema) => Err(format!(
            "Unknown figure schema {schema:?}. Supported schemas are {PAIRED_SCHEMA}, {TRADEOFF_SCHEMA}, {ERRORS_SCHEMA} and {OVERLAP_SCHEMA}. {RAW_HELP}"
        )),
        None => Err(format!("Missing portable figure schema. {RAW_HELP}")),
    }?;
    if matches!(format, VizFormat::Json) {
        // A portable file can include source hashes, study settings and other
        // evidence beyond the renderer schema. Keep those fields on replay.
        let source = std::str::from_utf8(bytes).map_err(|e| e.to_string())?;
        return Ok(format!("{}\n", source.trim_end()));
    }
    Ok(rendered)
}
