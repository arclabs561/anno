//! Models command - List and compare available models

use super::super::output::color;
use super::super::utils::find_similar_models;
use anno::available_backends;
use clap::{Parser, Subcommand};

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

    /// Prefetch/download model artifacts into cache.
    ///
    /// This works by instantiating the model backend(s), which triggers the normal
    /// “download if missing” paths (HF Hub / local cache) used by the rest of the CLI.
    ///
    /// Notes:
    /// - If `ANNO_NO_DOWNLOADS=1` or `HF_HUB_OFFLINE=1` is set, downloads will likely fail.
    /// - For some backends (e.g. `deberta-v3`, `albert`) you must provide local ONNX exports
    ///   via env vars (`DEBERTA_MODEL_PATH`, `ALBERT_MODEL_PATH`).
    #[command(visible_alias = "dl")]
    Download {
        /// One or more model backends to download (e.g., gliner, gliner2, bert-onnx).
        #[arg(value_name = "MODEL", required = true)]
        models: Vec<String>,

        /// Also prefetch the GLiNER2 relation-extraction multitask model (when `gliner2` is included).
        #[arg(long, default_value_t = false)]
        include_relation: bool,
    },
}

fn parse_model_backend(s: &str) -> Option<super::super::parser::ModelBackend> {
    use super::super::parser::ModelBackend;
    match s.to_lowercase().as_str() {
        "pattern" | "regex" => Some(ModelBackend::Pattern),
        "heuristic" | "statistical" => Some(ModelBackend::Heuristic),
        "minimal" => Some(ModelBackend::Minimal),
        "auto" => Some(ModelBackend::Auto),
        "stacked" => Some(ModelBackend::Stacked),
        "crf" => Some(ModelBackend::Crf),
        "hmm" => Some(ModelBackend::Hmm),
        "ensemble" => Some(ModelBackend::Ensemble),
        "heuristic-crf" | "heuristic_crf" | "bilstm-crf" | "bilstm_crf" => {
            Some(ModelBackend::HeuristicCrf)
        }
        "tplinker" | "tplink" => Some(ModelBackend::Tplinker),
        "universal-ner" | "universal_ner" | "universalner" => Some(ModelBackend::UniversalNer),
        #[cfg(feature = "onnx")]
        "gliner" | "gliner_onnx" => Some(ModelBackend::Gliner),
        #[cfg(feature = "onnx")]
        "gliner2" => Some(ModelBackend::Gliner2),
        #[cfg(feature = "onnx")]
        "nuner" => Some(ModelBackend::Nuner),
        #[cfg(feature = "onnx")]
        "w2ner" => Some(ModelBackend::W2ner),
        #[cfg(feature = "onnx")]
        "bert-onnx" | "bert_onnx" | "bert" => Some(ModelBackend::BertOnnx),
        #[cfg(feature = "onnx")]
        "deberta-v3" | "deberta_v3" | "deberta" => Some(ModelBackend::DebertaV3),
        #[cfg(feature = "onnx")]
        "albert" | "albert_ner" => Some(ModelBackend::Albert),
        #[cfg(feature = "onnx")]
        "gliner-poly" | "gliner_poly" => Some(ModelBackend::GlinerPoly),
        #[cfg(feature = "candle")]
        "gliner-candle" | "gliner_candle" => Some(ModelBackend::GlinerCandle),
        #[cfg(feature = "candle")]
        "candle-ner" | "candle_ner" => Some(ModelBackend::CandleNer),
        _ => None,
    }
}

/// Execute the models command.
pub fn run(args: ModelsArgs) -> Result<(), String> {
    match args.action {
        ModelsAction::List => {
            println!();
            println!("{}", color("1;36", "Available Models"));
            println!();

            let backends = available_backends();
            for (name, available) in backends {
                let status = if available {
                    color("32", "✓ Available")
                } else {
                    color("90", "✗ Not available")
                };
                let note = if available {
                    ""
                } else {
                    " (requires feature flag - see anno info)"
                };
                println!("  {} {}{}", status, name, note);
            }
            println!();
            println!(
                "Use 'anno models info <MODEL>' for detailed information about a specific model."
            );
            println!();
        }
        ModelsAction::Info { model } => {
            println!();
            println!("{}: {}", color("1;36", "Model Information"), model);
            println!();

            let backends = available_backends();
            // Try to find model by exact name or common aliases
            let model_lower = model.to_lowercase();
            let found = backends.iter().find(|(n, _)| {
                n.eq_ignore_ascii_case(&model)
                    || (model_lower == "stacked" && n.eq_ignore_ascii_case("StackedNER"))
                    || (model_lower == "pattern" && n.eq_ignore_ascii_case("RegexNER"))
                    || (model_lower == "heuristic" && n.eq_ignore_ascii_case("HeuristicNER"))
                    || (model_lower == "gliner" && n.eq_ignore_ascii_case("GLiNEROnnx"))
                    || (model_lower == "bert" && n.eq_ignore_ascii_case("BertNEROnnx"))
            });

            let (name, available) = if let Some((n, a)) = found {
                (*n, *a)
            } else {
                // Model not found - provide helpful suggestions
                let backends_list: Vec<&str> = backends.iter().map(|(n, _)| *n).collect();
                let suggestions = find_similar_models(&model, &backends_list);
                if !suggestions.is_empty() {
                    println!("{} Model '{}' not found.", color("33", "!"), model);
                    println!("Did you mean:");
                    for sug in &suggestions {
                        println!("  - {}", sug);
                    }
                    println!();
                    println!("Use 'anno models list' to see all available models.");
                } else {
                    println!("{} Model '{}' not found.", color("31", "error:"), model);
                    println!("Use 'anno models list' to see all available models.");
                }
                return Ok(());
            };

            if !available {
                println!(
                    "{} Model '{}' is not available in this build.",
                    color("33", "!"),
                    name
                );
                println!("Enable required feature flags and rebuild.");
                println!();
                println!("Use 'anno info' to see enabled features.");
                return Ok(());
            }

            // Show model details
            println!("  Name: {}", name);
            println!("  Status: {}", color("32", "Available"));
            println!();

            // Try to create model instance to get more details
            use super::super::parser::ModelBackend;
            let backend = match model_lower.as_str() {
                "pattern" | "regex" => ModelBackend::Pattern,
                "heuristic" | "statistical" => ModelBackend::Heuristic,
                "stacked" => ModelBackend::Stacked,
                #[cfg(feature = "onnx")]
                "gliner" => ModelBackend::Gliner,
                #[cfg(feature = "onnx")]
                "gliner2" => ModelBackend::Gliner2,
                _ => {
                    println!("  Note: Detailed information not available for this model.");
                    return Ok(());
                }
            };

            match backend.create_model() {
                Ok(m) => {
                    println!("  Description: {}", m.description());
                    println!();
                    println!("  Supported Entity Types:");
                    for t in m.supported_types() {
                        println!("    - {}", t.as_label());
                    }
                }
                Err(e) => {
                    println!("  {} Failed to load model: {}", color("33", "warning:"), e);
                }
            }
            println!();
        }
        ModelsAction::Compare => {
            println!();
            println!("{}", color("1;36", "Model Comparison"));
            println!();
            println!("{:<20} {:<15} {:<20}", "Model", "Status", "Entity Types");
            println!("{}", "-".repeat(55));

            let backends = available_backends();
            for (name, available) in backends {
                let status = if available {
                    color("32", "Available")
                } else {
                    color("90", "Not available")
                };

                // Try to get entity types if available
                let types_str = if available {
                    use super::super::parser::ModelBackend;
                    let backend_opt = match name.to_lowercase().as_str() {
                        "pattern" | "regexner" => Some(ModelBackend::Pattern),
                        "heuristic" | "heuristicner" => Some(ModelBackend::Heuristic),
                        "stacked" | "stackedner" => Some(ModelBackend::Stacked),
                        _ => None,
                    };

                    if let Some(backend) = backend_opt {
                        if let Ok(m) = backend.create_model() {
                            let types: Vec<String> = m
                                .supported_types()
                                .iter()
                                .map(|t| t.as_label().to_string())
                                .collect();
                            if types.len() <= 5 {
                                types.join(", ")
                            } else {
                                format!("{} (+{} more)", types[..5].join(", "), types.len() - 5)
                            }
                        } else {
                            "N/A".to_string()
                        }
                    } else {
                        "N/A".to_string()
                    }
                } else {
                    "N/A".to_string()
                };

                println!("{:<20} {:<15} {:<20}", name, status, types_str);
            }
            println!();
        }
        ModelsAction::Download {
            models,
            include_relation: _include_relation,
        } => {
            if std::env::var("ANNO_NO_DOWNLOADS")
                .ok()
                .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                || std::env::var("HF_HUB_OFFLINE")
                    .ok()
                    .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            {
                println!(
                    "{} Downloads may fail because ANNO_NO_DOWNLOADS or HF_HUB_OFFLINE is set.",
                    color("33", "warning:")
                );
            }

            println!();
            println!("{}", color("1;36", "Downloading models"));
            println!();

            let mut any_err = false;
            for m in models {
                let Some(backend) = parse_model_backend(&m) else {
                    any_err = true;
                    println!("{} Unknown model backend: {}", color("31", "error:"), m);
                    continue;
                };

                print!("  {} {} ... ", color("36", "→"), backend.name());
                match backend.create_model() {
                    Ok(_model) => {
                        println!("{}", color("32", "ok"));
                    }
                    Err(e) => {
                        any_err = true;
                        println!("{}", color("31", "failed"));
                        println!("    {}", e);
                    }
                }

                // Optional: prefetch relation-capable GLiNER2 weights as well.
                #[cfg(feature = "onnx")]
                {
                    use super::super::parser::ModelBackend;

                    if _include_relation && matches!(backend, ModelBackend::Gliner2) {
                        // Match the dataset CLI’s default relation model id.
                        let rel_id = "onnx-community/gliner-multitask-large-v0.5";
                        print!("  {} gliner2(relation) ... ", color("36", "→"));
                        match anno::backends::gliner2::GLiNER2Onnx::from_pretrained(rel_id) {
                            Ok(_m) => println!("{}", color("32", "ok")),
                            Err(e) => {
                                any_err = true;
                                println!("{}", color("31", "failed"));
                                println!("    {}", e);
                            }
                        }
                    }
                }
            }

            println!();
            if any_err {
                return Err("Some downloads failed. See errors above.".to_string());
            }
        }
    }

    Ok(())
}
