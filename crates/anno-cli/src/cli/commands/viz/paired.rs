use super::{distinct_output, RenderArgs, VizFormat};
use anno::viz::paired::{Paired, PairedRow, PAIRED_SCHEMA};
use clap::Args;
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::{Path, PathBuf};

#[derive(Debug, Args)]
pub(super) struct PairedArgs {
    #[command(flatten)]
    render: RenderArgs,

    /// Compare the baseline INPUT stream of anno eval --json reports against this candidate stream
    #[arg(long, value_name = "CANDIDATE")]
    against: Option<PathBuf>,

    /// Baseline setting name for two eval streams
    #[arg(long, requires = "against")]
    baseline_name: Option<String>,

    /// Candidate setting name for two eval streams
    #[arg(long, requires = "against")]
    candidate_name: Option<String>,

    /// Cohort name for two eval streams
    #[arg(long, requires = "against")]
    cohort: Option<String>,

    /// Observation unit for two eval streams, such as "sentences". Defaults to "documents"
    #[arg(long, requires = "against", value_name = "UNIT")]
    unit: Option<String>,

    /// Cohort's role, such as "fresh holdout" or "development"
    #[arg(long, requires = "against")]
    role: Option<String>,
}

pub(super) fn run(args: PairedArgs) -> Result<(), String> {
    let input = args.render.paths.input.as_deref();
    if args.against.as_deref() == Some(Path::new("-")) && input.is_none_or(|p| p == Path::new("-"))
    {
        return Err("only one eval stream can use stdin. Supply a file for the other".into());
    }
    if let (Some(against), Some(out)) = (&args.against, &args.render.paths.output) {
        if against != Path::new("-") && out != Path::new("-") {
            distinct_output(against, out)?;
        }
    }
    args.render.render(|input, format, title| {
        let mut report = if let Some(path) = args.against {
            let candidate: Box<dyn Read> = if path == Path::new("-") {
                Box::new(io::stdin().lock())
            } else {
                Box::new(BufReader::new(
                    File::open(&path).map_err(|e| format!("{}: {e}", path.display()))?,
                ))
            };
            let row = PairedRow::from_eval_reports(
                args.cohort.as_deref().unwrap_or("Paired evaluations"),
                args.role.as_deref().unwrap_or("Evaluation"),
                input,
                candidate,
            )?;
            Paired {
                schema: PAIRED_SCHEMA.into(),
                title: "Paired entity evaluation".into(),
                unit: args.unit.unwrap_or_else(|| "documents".into()),
                baseline: args.baseline_name.unwrap_or_else(|| "Baseline".into()),
                candidate: args.candidate_name.unwrap_or_else(|| "Candidate".into()),
                rows: vec![row],
            }
        } else {
            Paired::from_json(input)?
        };
        if let Some(title) = title {
            report.title = title.to_string();
        }
        match format {
            VizFormat::Svg => report.render_svg(),
            VizFormat::Html => report.render_html(),
            VizFormat::Json => report.to_json(),
        }
    })
}
