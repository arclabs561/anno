//! Explain a model selection request without constructing the backend.
use crate::cli::commands::model_routing::{
    automatic_contact_rules, extract_candidates, preflight, Preflight,
};
use crate::cli::parser::ModelBackend;
use clap::{Args, Subcommand};
use serde::Serialize;

#[derive(Args, Debug)]
pub struct ExplainArgs {
    #[command(subcommand)]
    operation: Operation,
    /// Emit the routing decision and individual preflight results as JSON.
    #[arg(long, global = true)]
    json: bool,
    #[arg(skip)]
    from_profile: bool,
}

#[derive(Subcommand, Debug)]
enum Operation {
    /// Explain named-entity or joint entity and relation extraction.
    Extract {
        #[arg(long, short)]
        model: Option<ModelBackend>,
        /// Comma-separated label schema sent to a zero-shot model.
        #[arg(long)]
        extract_types: Option<String>,
        #[arg(long)]
        extract_relations: bool,
        /// Output-label filtering does not select a model.
        #[arg(long)]
        types: Option<String>,
    },
    /// Explain the privacy model and local rule selection.
    Privacy {
        #[arg(long, short)]
        model: Option<ModelBackend>,
        #[arg(long, value_delimiter = ',', ignore_case = true,
            value_parser = ["PERSON", "DOB", "ADDRESS", "CONTACT", "ID_NUMBER"])]
        types: Vec<String>,
    },
}

impl ExplainArgs {
    /// Apply the saved model only when this explanation has no CLI override.
    pub(crate) fn inherit_model(&mut self, saved: Option<ModelBackend>) {
        let (Operation::Extract { model, .. } | Operation::Privacy { model, .. }) =
            &mut self.operation;
        if model.is_none() && saved.is_some() {
            *model = saved;
            self.from_profile = true;
        }
    }
}

#[derive(Serialize)]
struct Plan {
    operation: &'static str,
    selection_source: &'static str,
    planned_first_attempt: Option<&'static str>,
    candidates: Vec<Preflight>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    stacked_layers: Vec<Preflight>,
    notes: Vec<&'static str>,
    runtime_checked: bool,
}

fn plan(args: &ExplainArgs) -> Plan {
    let mut notes = Vec::new();
    let (operation, requested, candidates) = match &args.operation {
        Operation::Extract {
            model,
            extract_types,
            extract_relations,
            types,
        } => {
            let custom = extract_types
                .as_deref()
                .is_some_and(|s| s.split(',').any(|label| !label.trim().is_empty()));
            let choices = model.map_or_else(
                || extract_candidates(custom, *extract_relations),
                |model| vec![model],
            );
            #[cfg(not(any(feature = "onnx", feature = "candle")))]
            if custom && model.is_none() {
                notes.push("This build has no native ONNX or Candle GLiNER fallback.");
            }
            if custom
                && model.is_some_and(|b| {
                    super::resolve_backend_metadata(b.name()).is_some_and(|info| !info.zero_shot)
                })
                && !*extract_relations
            {
                notes.push("This explicit backend uses its fixed entity inventory and ignores a custom prompt.");
            }
            if types.is_some() {
                notes.push("--types filters the output after inference.");
            }
            if *extract_relations {
                notes.push("Joint-capable backends produce relations directly. The ordinary extraction fallback uses relation heuristics.");
            }
            (
                "extract",
                *model,
                choices
                    .into_iter()
                    .map(|b| preflight(b, model.is_some(), *extract_relations))
                    .collect::<Vec<_>>(),
            )
        }
        Operation::Privacy { model, types } if automatic_contact_rules(*model, types) => {
            let mut choices = Vec::new();
            #[cfg(feature = "gliner2-fastino")]
            choices.push(preflight(ModelBackend::Gliner2Pii, false, false));
            #[cfg(not(feature = "gliner2-fastino"))]
            choices.push(Preflight {
                checkpoint_override: None,
                model: "gliner2-pii",
                eligible: false,
                local_assets: "not_checked",
                reason: "The gliner2-fastino Cargo feature is disabled.",
            });
            choices.push(Preflight {
                checkpoint_override: None,
                model: "contact-rules",
                eligible: true,
                local_assets: "built_in",
                reason:
                    "Structured and regex contact rules run on every automatic CONTACT request.",
            });
            notes.push("Prepared PII predictions supplement the contact rules. A model load or inference failure leaves the rules available.");
            ("privacy", None, choices)
        }
        Operation::Privacy { model, .. } => {
            notes.push("Structured privacy rules also run. The requested categories filter the resulting PII spans.");
            (
                "privacy",
                *model,
                vec![preflight(model.unwrap_or_default(), model.is_some(), false)],
            )
        }
    };
    let source = if args.from_profile {
        "profile"
    } else if requested.is_some() {
        "command_line"
    } else {
        "automatic"
    };
    let planned = if let Some(model) = requested {
        notes.push("An explicit model is attempted directly, including when its local files are missing. Startup errors propagate to the caller.");
        Some(model.name())
    } else {
        candidates.iter().find(|c| c.eligible).map(|c| c.model)
    };
    if operation == "extract" && requested.is_none() && candidates.len() > 1 {
        notes
            .push("Automatic selection tries the next eligible candidate if initialization fails.");
    }
    if planned.is_none() {
        notes.push("No eligible schema-capable default was found. Prepare a checkpoint with 'anno models checkpoints'.");
    }
    let mut stacked_layers = Vec::new();
    if candidates
        .iter()
        .any(|candidate| matches!(candidate.model, "stacked" | "auto"))
    {
        #[cfg(feature = "onnx")]
        for layer in [
            ModelBackend::BertOnnx,
            ModelBackend::Nuner,
            ModelBackend::Gliner,
        ] {
            stacked_layers.push(preflight(layer, false, false));
        }
        for layer in [ModelBackend::Pattern, ModelBackend::Heuristic] {
            stacked_layers.push(preflight(layer, false, false));
        }
    }
    Plan {
        operation,
        selection_source: source,
        planned_first_attempt: planned,
        candidates,
        stacked_layers,
        notes,
        runtime_checked: false,
    }
}

pub(super) fn run(args: ExplainArgs) -> Result<(), String> {
    let plan = plan(&args);
    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&plan).map_err(|e| e.to_string())?
        );
    } else {
        println!(
            "{} model selection ({})",
            plan.operation, plan.selection_source
        );
        println!(
            "Planned first attempt: {}",
            plan.planned_first_attempt.unwrap_or("none")
        );
        for candidate in &plan.candidates {
            println!(
                "  {}  [{}]  {}",
                candidate.model, candidate.local_assets, candidate.reason
            );
            if let Some(path) = &candidate.checkpoint_override {
                println!("    Configured checkpoint: {path}");
            }
        }
        if !plan.stacked_layers.is_empty() {
            println!("Stacked layers:");
            for layer in &plan.stacked_layers {
                println!(
                    "  {}  [{}]  {}",
                    layer.model, layer.local_assets, layer.reason
                );
            }
        }
        for note in &plan.notes {
            println!("{note}");
        }
        println!(
            "Preflight only. No weights were loaded and no worker or network request was started."
        );
    }
    Ok(())
}
