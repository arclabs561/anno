//! anno - Information Extraction CLI
#![allow(dead_code)]
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
//! anno debug --coref -t "Sophie Wilson designed the ARM processor. She revolutionized computing."
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

use clap::{CommandFactory, Parser};

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
            .with_env_filter(filter)
            .with_target(true)
            .with_thread_ids(false)
            .init();
    }

    use crate::cli::commands::*;
    use crate::cli::exit_codes;
    use crate::cli::output::color;
    use crate::cli::parser::{Cli, Commands, ModelBackend, OutputFormat};
    use crate::cli::CliError;
    use clap_complete::generate;

    let cli = Cli::parse();

    let result: Result<(), CliError> = match cli.command {
        Some(Commands::Extract(args)) => extract::run(args),
        Some(Commands::Debug(args)) => debug::run(args).map_err(CliError::from),
        Some(Commands::Eval(args)) => eval::run(args).map_err(CliError::from),
        Some(Commands::Validate(args)) => validate::run(args).map_err(CliError::from),
        Some(Commands::Analyze(args)) => analyze::run(args).map_err(CliError::from),
        Some(Commands::Dataset(args)) => dataset::run(args).map_err(CliError::from),
        #[cfg(feature = "eval-advanced")]
        Some(Commands::Benchmark(args)) => benchmark::run(args).map_err(CliError::from),
        #[cfg(feature = "eval-advanced")]
        Some(Commands::Muxer(args)) => muxer::run(args).map_err(CliError::from),
        Some(Commands::Info) => info::run().map_err(CliError::from),
        Some(Commands::Models(args)) => models::run(args).map_err(CliError::from),
        #[cfg(feature = "eval-advanced")]
        Some(Commands::CrossDoc(args)) => crossdoc::run(args).map_err(CliError::from),
        Some(Commands::Enhance(args)) => enhance::run(args).map_err(CliError::from),
        Some(Commands::Pipeline(args)) => pipeline::run(args).map_err(CliError::from),
        Some(Commands::Query(args)) => query::run(args).map_err(CliError::from),
        Some(Commands::Compare(args)) => compare::run(args).map_err(CliError::from),
        Some(Commands::Cache(args)) => cache::run(args).map_err(CliError::from),
        #[cfg(feature = "eval")]
        Some(Commands::History(args)) => history::run(args).map_err(CliError::from),
        Some(Commands::Config(args)) => config::run(args).map_err(CliError::from),
        Some(Commands::Batch(args)) => batch::run(args).map_err(CliError::from),
        Some(Commands::Joint(args)) => joint::run(args).map_err(CliError::from),
        Some(Commands::Privacy(args)) => privacy::run(args).map_err(CliError::from),
        Some(Commands::Watch(args)) => watch::run(args).map_err(CliError::from),
        Some(Commands::Domain(args)) => domain::run(args).map_err(CliError::from),
        Some(Commands::Explain(args)) => explain::run(args).map_err(CliError::from),
        Some(Commands::Singleton(args)) => singleton::run(args).map_err(CliError::from),
        Some(Commands::Completions { shell }) => {
            generate(shell, &mut Cli::command(), "anno", &mut io::stdout());
            Ok(())
        }
        None => {
            // No subcommand: treat positional args as text to extract
            if cli.text.is_empty() {
                eprintln!("No input provided. Run `anno --help` for usage.");
                return ExitCode::from(exit_codes::ERROR_ARGS);
            }
            let text = cli.text.join(" ");
            extract::run(crate::cli::commands::ExtractArgs {
                url: None,
                clean: false,
                normalize: false,
                detect_lang: false,
                export_graph: None,
                text: Some(text),
                file: None,
                model: ModelBackend::default(),
                labels: vec![],
                types: None,
                extract_types: None,
                extract_relations: false,
                relation_types: None,
                relation_threshold: None,
                relation_max_span_distance: 120,
                threshold: None,
                expected_types: None,
                format: OutputFormat::default(),
                context_window: None,
                include_sentence: false,
                export: None,
                export_format: "full".to_string(),
                negation: false,
                quantifiers: false,
                verbose: 0,
                quiet: false,
                positional: vec![],
            })
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
