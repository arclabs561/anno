//! Models command - List and compare available models

use super::super::output::color;
use super::super::utils::find_similar_models;
use anno::available_backends;
use anno::backends::catalog::{BackendInfo, BACKEND_CATALOG};
use clap::{Parser, Subcommand};

#[path = "checkpoints/mod.rs"]
mod checkpoints;

/// List and compare available models
#[derive(Parser, Debug)]
pub struct ModelsArgs {
    /// Action to perform
    #[command(subcommand)]
    pub action: ModelsAction,
}

/// Models subcommand actions.
#[derive(Subcommand, Debug)]
pub enum ModelsAction {
    /// List all available models with status
    #[command(visible_alias = "ls")]
    List,

    /// Browse checkpoint sources, entry points and preparation recipes offline.
    Checkpoints {
        /// Show the recipe for a single checkpoint ID.
        #[arg(value_name = "ID")]
        id: Option<String>,

        /// Output a JSON array of matching checkpoint records.
        #[arg(long)]
        json: bool,
    },

    /// Show detailed information about a model
    #[command(visible_alias = "i")]
    Info {
        /// Model name to get info for
        #[arg(value_name = "MODEL")]
        model: String,
    },

    /// Compare available models side-by-side
    #[command(visible_alias = "c")]
    Compare,

    /// Prepare model artifacts for extraction.
    ///
    /// Hub-backed models load from cache or download missing files. Local-only
    /// models validate a prepared export and workers also start their interpreter.
    #[command(visible_alias = "dl")]
    Download {
        /// One or more model backends to download (e.g., gliner, gliner_multitask, bert-onnx).
        #[arg(value_name = "MODEL", required = true)]
        models: Vec<String>,

        /// Also validate the local GLiREL relation export used by GLiNER multi-task.
        #[arg(long, default_value_t = false)]
        include_relation: bool,
    },
}

fn parse_model_backend(s: &str) -> Option<super::super::parser::ModelBackend> {
    use super::super::parser::ModelBackend;
    use clap::ValueEnum;
    let normalized = s.to_ascii_lowercase().replace('_', "-");
    let cli_name = match normalized.as_str() {
        "universalner" => "universal-ner",
        "gliner-onnx" => "gliner",
        other => other,
    };
    ModelBackend::from_str(cli_name, true).ok()
}

fn prepare_backend(backend: super::super::parser::ModelBackend) -> Result<String, String> {
    use super::super::parser::ModelBackend::*;

    match backend {
        Pattern | Heuristic | Minimal | Crf | Hmm | HeuristicCrf => {
            return Ok("no checkpoint download required".into());
        }
        #[cfg(feature = "heuristic-fr")]
        HeuristicFr => return Ok("no checkpoint download required".into()),
        UniversalNer => {
            return Ok("uses a configured LLM service with no local checkpoint to prepare".into())
        }
        _ => {}
    }

    let model = backend.create_model()?;
    let runtime = model.runtime_name();
    match backend {
        Tplinker if runtime == "tplinker-heuristic" => Err(
            "Only the TPLinker heuristic fallback is available. For neural weights, build with ONNX support and prepare the local export with scripts/export_tplinker_onnx.py.".into(),
        ),
        OtterCross | OtterBi | Kniv => Ok(format!("local checkpoint loaded by {runtime} worker")),
        Tplinker => Ok(format!("local checkpoint loaded by {runtime}")),
        #[cfg(feature = "onnx")]
        Adaptner | Biomedical => Ok(format!("local export loaded by {runtime}")),
        #[cfg(feature = "gliner2-fastino")]
        Gliner25 => Ok(format!("local export loaded by {runtime}")),
        Stacked | Auto | Ensemble => Ok(format!("{runtime} initialized (select a component backend to require its weights)")),
        _ => Ok(format!("checkpoint loaded by {runtime}")),
    }
}

/// Resolve a CLI spelling to the catalog entry used for informational output.
///
/// Exact catalog names win so `gliner_onnx` retains its more specific metadata.
/// Otherwise, reuse the CLI parser's aliases and normalize its canonical name to
/// the catalog's underscore-separated convention. This must remain metadata-only:
/// callers of `models list`, `info`, and `compare` must not instantiate a backend.
fn resolve_backend_metadata(name: &str) -> Option<&'static BackendInfo> {
    let normalized = name.to_ascii_lowercase().replace('-', "_");
    BackendInfo::by_name(&normalized).or_else(|| {
        parse_model_backend(name).and_then(|backend| {
            let canonical = backend.name().replace('-', "_");
            BackendInfo::by_name(&canonical)
        })
    })
}

fn compiled_backends() -> std::collections::HashMap<&'static str, bool> {
    available_backends().into_iter().collect()
}

fn build_status(compiled: bool) -> String {
    if compiled {
        color("32", "compiled")
    } else {
        color("90", "not compiled")
    }
}

/// Execute the models command.
pub fn run(args: ModelsArgs) -> Result<(), String> {
    match args.action {
        ModelsAction::Checkpoints { id, json } => return checkpoints::run(id.as_deref(), json),
        ModelsAction::List => {
            println!();
            println!("{}", color("1;36", "Available Models"));
            println!();

            let compiled = compiled_backends();
            for info in BACKEND_CATALOG {
                let is_compiled = compiled.get(info.name).copied().unwrap_or(false);
                println!("  {} {}", build_status(is_compiled), info.name);
            }
            println!();
            println!(
                "Build support reflects Cargo features only; runtime readiness is not checked."
            );
            println!(
                "Use 'anno models info <MODEL>' for detailed information about a specific model."
            );
            println!(
                "Use 'anno models checkpoints' for checkpoint recipes and task-specific commands."
            );
            println!();
        }
        ModelsAction::Info { model } => {
            println!();
            println!("{}: {}", color("1;36", "Model Information"), model);
            println!();

            let backend = resolve_backend_metadata(&model);
            let recipe = checkpoints::find(backend.map_or(model.as_str(), |info| info.name))?;
            if backend.is_none() {
                if let Some(entry) = &recipe {
                    checkpoints::print_details(entry);
                    return Ok(());
                }
            }
            let info = backend.ok_or_else(|| {
                let backends_list: Vec<&str> = BACKEND_CATALOG.iter().map(|info| info.name).collect();
                let suggestions = find_similar_models(&model, &backends_list);
                if !suggestions.is_empty() {
                    format!(
                        "Model '{}' not found. Did you mean: {}? Use 'anno models list' to see all available models.",
                        model,
                        suggestions.join(", ")
                    )
                } else {
                    format!(
                        "Model '{}' not found. Use 'anno models list' to see all available models.",
                        model
                    )
                }
            })?;
            let compiled = compiled_backends().get(info.name).copied().unwrap_or(false);

            println!("  Name: {}", info.name);
            println!("  Build support: {}", build_status(compiled));
            println!("  Implementation: {}", info.status);
            println!("  Runtime readiness: not checked");
            println!("  Description: {}", info.description);
            println!(
                "  Zero-shot NER: {}",
                if info.zero_shot { "yes" } else { "no" }
            );
            println!(
                "  GPU support: {}",
                if info.gpu_support { "yes" } else { "no" }
            );
            if let Some(feature) = info.feature {
                println!("  Required feature: {}", feature);
            }
            if !info.recommended_models.is_empty() {
                println!("  Recommended models:");
                for model_id in info.recommended_models {
                    println!("    - {}", model_id);
                }
            }
            if let Some(entry) = &recipe {
                checkpoints::print_details(entry);
            }
            println!();
        }
        ModelsAction::Compare => {
            println!();
            println!("{}", color("1;36", "Model Comparison"));
            println!();
            println!(
                "{:<22} {:<16} {:<12} Runtime",
                "Model", "Build support", "Status"
            );
            println!("{}", "-".repeat(70));

            let compiled = compiled_backends();
            for info in BACKEND_CATALOG {
                let is_compiled = compiled.get(info.name).copied().unwrap_or(false);
                println!(
                    "{:<22} {:<16} {:<12} not checked",
                    info.name,
                    build_status(is_compiled),
                    info.status,
                );
            }
            println!();
            println!("Comparison uses build metadata only; no model artifacts, configuration, or credentials are loaded.");
        }
        ModelsAction::Download {
            models,
            include_relation: _include_relation,
        } => {
            println!();
            println!("{}", color("1;36", "Preparing model backends"));
            println!();

            let mut any_err = false;
            for m in models {
                let Some(backend) = parse_model_backend(&m) else {
                    any_err = true;
                    if let Some(info) = resolve_backend_metadata(&m) {
                        let compiled = compiled_backends().get(info.name).copied().unwrap_or(false);
                        if let Some(feature) = info.feature.filter(|_| !compiled) {
                            println!(
                                "{} {} requires Cargo feature {}",
                                color("31", "error:"),
                                info.name,
                                feature
                            );
                        } else {
                            println!(
                                "{} {} has no download selector in this CLI build",
                                color("31", "error:"),
                                info.name
                            );
                        }
                    } else {
                        println!("{} Unknown model backend: {}", color("31", "error:"), m);
                    }
                    continue;
                };

                print!("  {} {} ... ", color("36", "→"), backend.name());
                match prepare_backend(backend) {
                    Ok(status) => println!("{}", color("32", &status)),
                    Err(e) => {
                        any_err = true;
                        println!("{}", color("31", "failed"));
                        println!("    {}", e);
                        if let Ok(Some(recipe)) = checkpoints::find(backend.name()) {
                            println!(
                                "    Preparation guide: anno models checkpoints {}",
                                recipe.id
                            );
                        }
                    }
                }

                #[cfg(feature = "onnx")]
                if _include_relation
                    && matches!(backend, super::super::parser::ModelBackend::GlinerMultitask)
                {
                    let dir = dirs::cache_dir()
                        .unwrap_or_else(|| ".cache".into())
                        .join("anno/models/glirel");
                    print!("  {} glirel ... ", color("36", "→"));
                    match anno::backends::glirel::GLiREL::from_local(&dir) {
                        Ok(_) => println!("{}", color("32", "local relation export loaded")),
                        Err(e) => {
                            any_err = true;
                            println!("{}", color("31", "failed"));
                            println!("    {e}");
                        }
                    }
                }
            }

            println!();
            if any_err {
                return Err(
                    "Model preparation failed for one or more requested backends.".to_string(),
                );
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn download_distinguishes_backends_without_checkpoint_artifacts() {
        use super::super::super::parser::ModelBackend;
        assert!(prepare_backend(ModelBackend::Pattern)
            .unwrap()
            .contains("no checkpoint"));
        assert!(prepare_backend(ModelBackend::UniversalNer)
            .unwrap()
            .contains("LLM service"));
    }

    #[test]
    fn metadata_resolver_preserves_cli_aliases() {
        assert_eq!(
            resolve_backend_metadata("regex").map(|info| info.name),
            Some("pattern")
        );
        assert_eq!(
            resolve_backend_metadata("heuristic-crf").map(|info| info.name),
            Some("heuristic_crf")
        );
        #[cfg(feature = "onnx")]
        assert_eq!(
            resolve_backend_metadata("bert").map(|info| info.name),
            Some("bert_onnx")
        );
    }

    #[test]
    fn metadata_resolver_uses_catalog_without_model_construction() {
        let info = resolve_backend_metadata("gliner_onnx").expect("catalog entry");

        assert_eq!(info.name, "gliner_onnx");
        assert_eq!(info.feature, Some("onnx"));
        assert!(!info.description.is_empty());
    }

    #[test]
    fn metadata_resolver_rejects_unknown_backends() {
        assert!(resolve_backend_metadata("definitely-not-a-model").is_none());
    }
}
