//! Backends command - List and compare available backends (consolidates info + models)

use super::super::output::{color, type_color};
use super::super::utils::find_similar_models;
use anno::{available_backends, Model, StackedNER};
use clap::{Parser, Subcommand};

/// List and compare available backends
#[derive(Parser, Debug)]
pub struct BackendsArgs {
    /// Action to perform
    #[command(subcommand)]
    pub action: Option<BackendsAction>,
}

/// Available actions for the backends command.
#[derive(Subcommand, Debug)]
pub enum BackendsAction {
    /// List all available backends with status
    #[command(visible_alias = "ls")]
    List,

    /// Show detailed information about a backend
    #[command(visible_alias = "i")]
    Info {
        /// Backend name to get info for
        #[arg(value_name = "BACKEND")]
        backend: String,
    },

    /// Compare available backends side-by-side
    #[command(visible_alias = "c")]
    Compare,
}

/// Run the backends command.
pub fn run(args: BackendsArgs) -> Result<(), String> {
    match args.action {
        None | Some(BackendsAction::List) => {
            // Default: show version, backends, types, features (like old `info`)
            println!();
            println!("{}", color("1;36", "anno"));
            println!("  Information Extraction: NER + Coreference");
            println!();
            println!("{}:", color("1;33", "Version"));
            println!("  {}", env!("CARGO_PKG_VERSION"));
            println!();
            println!("{}:", color("1;33", "Available Backends (this build)"));

            let backends = available_backends();
            for (name, available) in backends {
                let status = if available {
                    color("32", "✓")
                } else {
                    color("90", "✗")
                };
                let note = if available {
                    ""
                } else {
                    " (requires feature flag)"
                };
                println!("  {} {} {}", status, name, note);
            }
            println!();

            let model: Box<dyn Model> = Box::new(StackedNER::default());
            println!("{}:", color("1;33", "Supported Entity Types (stacked)"));
            for t in model.supported_types() {
                let color_code = type_color(t.as_label());
                println!("  {} {}", color(color_code, "*"), t.as_label());
            }
            println!();

            println!("{}:", color("1;33", "Enabled Features"));
            #[allow(clippy::vec_init_then_push)] // Feature-gated pushes can't use vec![]
            let features: Vec<&str> = {
                #[allow(unused_mut)] // Some builds compile with none of these cfg-gated pushes.
                let mut v = Vec::with_capacity(4);
                #[cfg(feature = "onnx")]
                v.push("onnx");
                #[cfg(feature = "candle")]
                v.push("candle");
                #[cfg(feature = "eval")]
                v.push("eval");
                #[cfg(feature = "eval-bias")]
                v.push("eval-bias");
                #[cfg(feature = "eval-advanced")]
                v.push("eval-advanced");
                #[cfg(feature = "discourse")]
                v.push("discourse");
                v
            };
            if features.is_empty() {
                println!("  (default features only)");
            } else {
                println!("  {}", features.join(", "));
            }
            println!();
        }
        Some(BackendsAction::Info { backend }) => {
            println!();
            println!("{}: {}", color("1;36", "Backend Information"), backend);
            println!();

            let backends = available_backends();
            let backend_lower = backend.to_lowercase();
            let found = backends.iter().find(|(n, _)| {
                n.eq_ignore_ascii_case(&backend)
                    || (backend_lower == "stacked" && n.eq_ignore_ascii_case("StackedNER"))
                    || (backend_lower == "pattern" && n.eq_ignore_ascii_case("RegexNER"))
                    || (backend_lower == "heuristic" && n.eq_ignore_ascii_case("HeuristicNER"))
                    || (backend_lower == "gliner" && n.eq_ignore_ascii_case("GLiNEROnnx"))
                    || (backend_lower == "bert" && n.eq_ignore_ascii_case("BertNEROnnx"))
            });

            let (name, available) = if let Some((n, a)) = found {
                (*n, *a)
            } else {
                let backends_list: Vec<&str> = backends.iter().map(|(n, _)| *n).collect();
                let suggestions = find_similar_models(&backend, &backends_list);
                if !suggestions.is_empty() {
                    println!("{} Backend '{}' not found.", color("33", "!"), backend);
                    println!("Did you mean:");
                    for sug in &suggestions {
                        println!("  - {}", sug);
                    }
                    println!();
                    println!("Use 'anno backends list' to see all available backends.");
                } else {
                    println!("{} Backend '{}' not found.", color("31", "error:"), backend);
                    println!("Use 'anno backends list' to see all available backends.");
                }
                return Ok(());
            };

            if !available {
                println!(
                    "{} Backend '{}' is not available in this build.",
                    color("33", "!"),
                    name
                );
                println!("Enable required feature flags and rebuild.");
                println!();
                println!("Use 'anno backends' to see enabled features.");
                return Ok(());
            }

            println!("  Name: {}", name);
            println!("  Status: {}", color("32", "Available"));
            println!();

            use super::super::parser::ModelBackend;
            let backend_enum = match backend_lower.as_str() {
                "pattern" | "regex" => ModelBackend::Pattern,
                "heuristic" | "statistical" => ModelBackend::Heuristic,
                "stacked" => ModelBackend::Stacked,
                #[cfg(feature = "onnx")]
                "gliner" => ModelBackend::Gliner,
                #[cfg(feature = "onnx")]
                "gliner2" => ModelBackend::Gliner2,
                _ => {
                    println!("  Note: Detailed information not available for this backend.");
                    return Ok(());
                }
            };

            match backend_enum.create_model() {
                Ok(m) => {
                    println!("  Description: {}", m.description());
                    println!();
                    println!("  Supported Entity Types:");
                    for t in m.supported_types() {
                        println!("    - {}", t.as_label());
                    }
                }
                Err(e) => {
                    println!(
                        "  {} Failed to load backend: {}",
                        color("33", "warning:"),
                        e
                    );
                }
            }
            println!();
        }
        Some(BackendsAction::Compare) => {
            println!();
            println!("{}", color("1;36", "Backend Comparison"));
            println!();
            println!("{:<20} {:<15} {:<20}", "Backend", "Status", "Entity Types");
            println!("{}", "-".repeat(55));

            let backends = available_backends();
            for (name, available) in backends {
                let status = if available {
                    color("32", "Available")
                } else {
                    color("90", "Not available")
                };

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
    }

    Ok(())
}
