//! Models command - List and compare available models

use super::super::output::color;
use super::super::utils::find_similar_models;
use anno::available_backends;
use anno::backends::catalog::{BackendInfo, BACKEND_CATALOG};
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
        /// One or more model backends to download (e.g., gliner, gliner_multitask, bert-onnx).
        #[arg(value_name = "MODEL", required = true)]
        models: Vec<String>,

        /// Also prefetch the GLiNER multi-task relation-extraction model (when `gliner_multitask` is included).
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
        "heuristic-crf" | "heuristic_crf" => Some(ModelBackend::HeuristicCrf),
        #[cfg(feature = "heuristic-fr")]
        "heuristic-fr" | "heuristic_fr" => Some(ModelBackend::HeuristicFr),
        "tplinker" | "tplink" => Some(ModelBackend::Tplinker),
        "universal-ner" | "universal_ner" | "universalner" => Some(ModelBackend::UniversalNer),
        #[cfg(feature = "onnx")]
        "gliner" | "gliner_onnx" => Some(ModelBackend::Gliner),
        #[cfg(feature = "onnx")]
        "gliner_multitask" => Some(ModelBackend::GlinerMultitask),
        #[cfg(feature = "onnx")]
        "nuner" => Some(ModelBackend::Nuner),
        #[cfg(feature = "onnx")]
        "w2ner" => Some(ModelBackend::W2ner),
        #[cfg(feature = "onnx")]
        "bert-onnx" | "bert_onnx" | "bert" => Some(ModelBackend::BertOnnx),
        #[cfg(feature = "onnx")]
        "deberta-v3" | "deberta_v3" | "deberta" => Some(ModelBackend::DebertaV3),
        #[cfg(feature = "onnx")]
        "biomedical" | "biomedical-ner" | "biomedical_ner" => Some(ModelBackend::Biomedical),
        #[cfg(feature = "onnx")]
        "gliner-pii" | "gliner_pii" | "pii" => Some(ModelBackend::GlinerPii),
        #[cfg(feature = "onnx")]
        "gliner-relex" | "gliner_relex" | "relex" => Some(ModelBackend::GlinerRelex),
        #[cfg(feature = "onnx")]
        "gliner-poly" | "gliner_poly" => Some(ModelBackend::GlinerPoly),
        #[cfg(feature = "candle")]
        "gliner-candle" | "gliner_candle" => Some(ModelBackend::GlinerCandle),
        #[cfg(feature = "candle")]
        "candle-ner" | "candle_ner" => Some(ModelBackend::CandleNer),
        _ => None,
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
            println!();
        }
        ModelsAction::Info { model } => {
            println!();
            println!("{}: {}", color("1;36", "Model Information"), model);
            println!();

            let info = resolve_backend_metadata(&model).ok_or_else(|| {
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

                // Optional: prefetch relation-capable GLiNER multi-task weights as well.
                #[cfg(feature = "onnx")]
                {
                    use super::super::parser::ModelBackend;

                    if _include_relation && matches!(backend, ModelBackend::GlinerMultitask) {
                        // Match the dataset CLI’s default relation model id.
                        let rel_id = "onnx-community/gliner-multitask-large-v0.5";
                        print!("  {} gliner_multitask(relation) ... ", color("36", "→"));
                        match anno::backends::gliner_multitask::GLiNERMultitaskOnnx::from_pretrained(
                            rel_id,
                        ) {
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

#[cfg(test)]
mod tests {
    use super::*;

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
