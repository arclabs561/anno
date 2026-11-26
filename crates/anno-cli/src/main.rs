//! anno - Information Extraction CLI
//!
//! A small CLI for `anno` focused on:
//! - NER (named entity recognition)
//! - within-document coreference
//! - structured pattern extraction (dates, money, emails)
//!
//! Evaluation/benchmarking commands exist behind feature flags.
//!
//! # Usage
//!
//! ```bash
//! # Basic NER extraction
//! anno extract --text "Lynn Conway worked at IBM and Xerox PARC in California."
//!
//! # Debug coreference
//! anno debug --model heuristic --coref -t "Marie Curie moved to Paris. Marie Curie later returned."
//!
//! # Evaluate against gold annotations
//! anno eval -t "Lynn Conway worked at IBM." -g "Lynn Conway:PER:0:11"
//!
//! # Validate annotation files
//! anno validate file.jsonl
//!
//! # Show available models and features
//! anno info
//! ```
//!
//! All logic lives in `crate::cli::*`; this is just the dispatcher.

mod cli;

use std::io;
use std::process::ExitCode;

use clap::{CommandFactory, FromArgMatches, Parser};

fn main() -> ExitCode {
    // Load workspace `.env` (idempotent, does not override existing env vars).
    // This makes `HF_TOKEN`, `ANNO_*` knobs, etc. work without manual exporting.
    anno::env::load_dotenv();

    // Initialize tracing subscriber when instrument feature is enabled
    #[cfg(feature = "instrument")]
    {
        use tracing_subscriber::{fmt, EnvFilter};
        let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"));
        fmt()
            .with_writer(io::stderr)
            .with_env_filter(filter)
            .with_target(true)
            .with_thread_ids(false)
            .init();
    }

    use crate::cli::commands::*;
    use crate::cli::exit_codes;
    use crate::cli::output::color;
    use crate::cli::parser::{Cli, Commands};
    use crate::cli::CliError;
    use clap_complete::generate;

    #[cfg(not(feature = "eval"))]
    if let Some(command) = crate::cli::parser::unavailable_eval_command(
        &std::env::args_os().skip(1).collect::<Vec<_>>(),
    ) {
        eprintln!(
            "error: `anno {command}` requires the `eval` feature. Rebuild anno-cli with `--features eval`.\n\
             To extract these words as text, use `anno -- {command} ...`."
        );
        return ExitCode::from(exit_codes::ERROR_ARGS);
    }
    let matches = Cli::command().get_matches();
    let mut cli = Cli::from_arg_matches(&matches).unwrap_or_else(|err| err.exit());
    if cli.command.is_none() && !cli.text.is_empty() {
        let mut args = ExtractArgs::parse_from(["extract"]);
        args.text = Some(cli.text.join(" "));
        cli.command = Some(Commands::Extract(args));
    }
    crate::cli::output::set_color_mode(cli.color);

    if let Some(name) = &cli.config {
        if let Err(err) = crate::cli::config::apply(name, cli.command.as_mut(), &matches) {
            eprintln!("error: {err}");
            return ExitCode::from(exit_codes::ERROR_ARGS);
        }
    }

    let result: Result<(), CliError> = match cli.command {
        Some(Commands::Extract(args)) => extract::run(args),
        Some(Commands::Debug(args)) => debug::run(args).map_err(CliError::from),
        Some(Commands::Eval(args)) => eval::run(args).map_err(CliError::from),
        Some(Commands::Joint(args)) => joint::run(args),
        Some(Commands::Validate(args)) => validate::run(args).map_err(CliError::from),
        Some(Commands::Analyze(args)) => analyze::run(args).map_err(CliError::from),
        Some(Commands::Dataset(args)) => dataset::run(args).map_err(CliError::from),
        #[cfg(feature = "eval")]
        Some(Commands::Benchmark(args)) => benchmark::run(args).map_err(CliError::from),
        #[cfg(feature = "eval")]
        Some(Commands::Muxer(args)) => muxer::run(args).map_err(CliError::from),
        Some(Commands::Info) => info::run().map_err(CliError::from),
        Some(Commands::Models(args)) => models::run(args).map_err(CliError::from),
        Some(Commands::Parse(args)) => parse::run(args).map_err(CliError::from),
        #[cfg(feature = "eval")]
        Some(Commands::CrossDoc(args)) => crossdoc::run(args).map_err(CliError::from),
        Some(Commands::Enhance(args)) => enhance::run(args).map_err(CliError::from),
        Some(Commands::Pipeline(args)) => pipeline::run(args).map_err(CliError::from),
        Some(Commands::Corpus(args)) => corpus::run(args).map_err(CliError::from),
        Some(Commands::Query(args)) => query::run(args).map_err(CliError::from),
        Some(Commands::Compare(args)) => compare::run(args).map_err(CliError::from),
        Some(Commands::Cache(args)) => cache::run(args).map_err(CliError::from),
        #[cfg(feature = "eval")]
        Some(Commands::History(args)) => history::run(args).map_err(CliError::from),
        Some(Commands::Config(args)) => config::run(args).map_err(CliError::from),
        Some(Commands::Batch(args)) => batch::run(args).map_err(CliError::from),
        Some(Commands::Privacy(args)) => privacy::run(args).map_err(CliError::from),
        Some(Commands::Watch(args)) => watch::run(args).map_err(CliError::from),
        Some(Commands::Domain(args)) => domain::run(args).map_err(CliError::from),
        Some(Commands::Explain(args)) => explain::run(args).map_err(CliError::from),
        Some(Commands::Singleton(args)) => singleton::run(args).map_err(CliError::from),
        Some(Commands::Export(args)) => export::run(args).map_err(CliError::from),
        Some(Commands::Completions { shell }) => {
            generate(shell, &mut Cli::command(), "anno", &mut io::stdout());
            Ok(())
        }
        None => {
            // No subcommand: treat positional args as text to extract
            eprintln!("No input provided. Run `anno --help` for usage.");
            return ExitCode::from(exit_codes::ERROR_ARGS);
        }
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{} {}", color("31", "error:"), e);
            ExitCode::from(e.exit_code())
        }
    }
}
