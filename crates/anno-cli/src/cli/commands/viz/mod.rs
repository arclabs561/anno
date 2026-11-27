//! Render measured results without loading models or accessing datasets.

mod errors;

use anno::viz::Tradeoff;
use clap::{Args, Subcommand, ValueEnum};
use std::fs::File;
use std::io::{self, BufReader, Read, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, Args)]
pub struct VizArgs {
    #[command(subcommand)]
    view: View,
}

#[derive(Debug, Subcommand)]
enum View {
    /// Plot correct and unmatched predictions per 100 observations from anno.tradeoff/v1 JSON
    Tradeoff(RenderArgs),
    /// Summarize matching errors and character coverage in one or more anno eval --json reports
    Errors(RenderArgs),
}

#[derive(Debug, Args)]
struct RenderArgs {
    #[command(flatten)]
    paths: ReportPaths,

    /// Report format (default SVG, inferred from .html, .htm or .json output files)
    #[arg(long, value_enum)]
    format: Option<VizFormat>,

    /// Set the chart title and retain it in JSON output
    #[arg(long)]
    title: Option<String>,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum VizFormat {
    Svg,
    Html,
    Json,
}

impl VizFormat {
    fn for_output(output: Option<&Path>) -> Self {
        let extension = output.and_then(Path::extension).and_then(|s| s.to_str());
        match extension {
            Some(ext) if ext.eq_ignore_ascii_case("html") || ext.eq_ignore_ascii_case("htm") => {
                Self::Html
            }
            Some(ext) if ext.eq_ignore_ascii_case("json") => Self::Json,
            _ => Self::Svg,
        }
    }
}

impl RenderArgs {
    fn render(
        &self,
        view: impl FnOnce(Box<dyn Read>, VizFormat, Option<&str>) -> Result<String, String>,
    ) -> Result<(), String> {
        self.paths.render(|reader, output| {
            let format = self.format.unwrap_or_else(|| VizFormat::for_output(output));
            view(reader, format, self.title.as_deref())
        })
    }
}

/// File and pipe behavior shared by every view, independently of its formats.
#[derive(Debug, Args)]
struct ReportPaths {
    /// JSON input file (default or '-' reads stdin)
    #[arg(value_name = "INPUT")]
    input: Option<PathBuf>,

    /// Write the report to a file (default or '-' writes stdout)
    #[arg(short, long, value_name = "PATH")]
    output: Option<PathBuf>,
}

impl ReportPaths {
    fn render(
        &self,
        view: impl FnOnce(Box<dyn Read>, Option<&Path>) -> Result<String, String>,
    ) -> Result<(), String> {
        let file = |path: &&Path| *path != Path::new("-");
        let input = self.input.as_deref().filter(file);
        let output = self.output.as_deref().filter(file);
        if let (Some(input), Some(output)) = (input, output) {
            match same_file::is_same_file(input, output) {
                Ok(true) => {
                    return Err(format!(
                        "input and output refer to the same file: {}",
                        input.display()
                    ))
                }
                Err(err) if err.kind() != io::ErrorKind::NotFound => return Err(err.to_string()),
                _ => (),
            }
        }
        let reader: Box<dyn Read> = if let Some(path) = input {
            Box::new(BufReader::new(
                File::open(path).map_err(|e| format!("{}: {e}", path.display()))?,
            ))
        } else {
            Box::new(io::stdin().lock())
        };

        let report = view(reader, output)?;
        if let Some(path) = output {
            super::export::write_output(path, report, true)
                .map_err(|e| format!("{}: {e}", path.display()))
        } else {
            io::stdout()
                .lock()
                .write_all(report.as_bytes())
                .map_err(|e| e.to_string())
        }
    }
}

pub fn run(args: VizArgs) -> Result<(), String> {
    match args.view {
        View::Tradeoff(options) => options.render(|reader, format, title| {
            let mut report = Tradeoff::from_json(reader)?;
            if let Some(title) = title {
                report.title = title.to_owned();
            }
            match format {
                VizFormat::Svg => report.render_svg(),
                VizFormat::Html => report.render_html(),
                VizFormat::Json => report.to_json(),
            }
        }),
        View::Errors(options) => options.render(errors::render),
    }
}
