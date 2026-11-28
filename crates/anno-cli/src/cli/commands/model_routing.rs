//! Filesystem-only preflight for CLI model routing.
use super::super::parser::ModelBackend;
use serde::Serialize;

pub(super) fn extract_candidates(custom_schema: bool, relations: bool) -> Vec<ModelBackend> {
    use ModelBackend::*;
    let mut candidates = Vec::new();
    if relations {
        candidates.push(Glinerel72m);
        #[cfg(feature = "onnx")]
        candidates.push(GlinerRelex);
    }
    if custom_schema {
        candidates.push(OtterCross);
        if !relations {
            candidates.push(Glinerel72m);
        }
        candidates.push(OtterBi);
        #[cfg(feature = "onnx")]
        candidates.push(Gliner);
        #[cfg(feature = "candle")]
        candidates.push(GlinerCandle);
    } else {
        candidates.push(Stacked);
    }
    candidates
}

pub(super) fn automatic_contact_rules(model: Option<ModelBackend>, categories: &[String]) -> bool {
    model.is_none()
        && !categories.is_empty()
        && categories
            .iter()
            .all(|category| category.eq_ignore_ascii_case("CONTACT"))
}

#[derive(Serialize)]
pub(super) struct Preflight {
    pub model: &'static str,
    pub eligible: bool,
    pub local_assets: &'static str,
    pub reason: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checkpoint_override: Option<String>,
}

pub(super) fn preflight(model: ModelBackend, explicit: bool, _joint: bool) -> Preflight {
    use anno_eval::eval::backend_factory::BackendFactory;
    use ModelBackend::*;
    let checkpoint_override = match model {
        #[cfg(feature = "onnx")]
        BertOnnx | Nuner if !explicit => None, // Stacked uses its compiled checkpoint IDs.
        _ => model.checkpoint_override(),
    }
    .map(|value| value.to_string_lossy().into_owned());
    let local = |present| if present { "present" } else { "missing" };
    #[cfg(any(feature = "onnx", feature = "candle"))]
    let hub = |present, eligible| {
        (
            eligible,
            local(present),
            if present {
                "Checkpoint files found in the local cache."
            } else if eligible {
                "Inference may acquire missing checkpoint files from the Hub."
            } else {
                "Checkpoint files are missing and downloads are disabled."
            },
        )
    };
    let (eligible, assets, reason) = match model {
        OtterCross | OtterBi | Kniv | Glinerel72m => {
            let configured = model.checkpoint_override().is_some();
            let cached = BackendFactory::optional_model_files_available(&model.name().replace('-', "_")) == Some(true);
            if matches!(model, Glinerel72m) && !configured && explicit {
                (cached, local(cached), "Local preflight checks the pinned snapshot. Explicit selection can fetch it when downloads are enabled.")
            } else if !configured {
                (false, if cached { "cached_unconfigured" } else { "unconfigured" }, "This route requires the worker checkpoint directory to be configured.")
            } else if cached {
                (true, "present", "Configured checkpoint files passed preflight. The Python interpreter and model have not been started.")
            } else {
                (false, "missing", "The configured directory is missing required checkpoint files.")
            }
        }
        #[cfg(feature = "onnx")]
        BertOnnx | Nuner => {
            let bert = matches!(model, BertOnnx);
            let default = if bert { anno::DEFAULT_BERT_ONNX_MODEL } else { anno::DEFAULT_NUNER_MODEL };
            let configured = if explicit { model.checkpoint_override() } else { None };
            let id = configured.as_deref().and_then(std::ffi::OsStr::to_str).unwrap_or(default);
            if bert {
                hub(anno::BertNEROnnx::cached_assets_available(id), anno::BertNEROnnx::model_files_available(id))
            } else {
                hub(anno::NuNER::cached_assets_available(id), anno::NuNER::model_files_available(id))
            }
        }
        #[cfg(feature = "gliner2-fastino")]
        Gliner25 => {
            let present = model.checkpoint_override().is_some_and(|dir| anno::backends::gliner25::GLiNER25::model_files_available(std::path::Path::new(&dir)));
            (present, local(present), "GLiNER2.5 requires a local FP32 boundary export.")
        }
        #[cfg(feature = "onnx")]
        GlinerRelex => {
            if explicit && !_joint {
                let present = anno::GLiNEROnnx::relex_entity_export_available();
                return Preflight { checkpoint_override, model: model.name(), eligible: present, local_assets: local(present), reason: "Entity-only extraction uses the local GLiNER-RelEx entity graph." };
            }
            let present = anno::GLiNEROnnx::relex_joint_export_available();
            (present, local(present), "Joint routing requires the prepared GLiNER-RelEx graph and joint export marker.")
        }
        #[cfg(feature = "onnx")]
        Gliner => hub(anno::GLiNEROnnx::cached_assets_available(anno::DEFAULT_GLINER_MODEL), anno::GLiNEROnnx::hub_assets_available(anno::DEFAULT_GLINER_MODEL)),
        #[cfg(feature = "onnx")]
        GlinerPii => hub(anno::GLiNEROnnx::cached_assets_available(anno::models::GLINER_PII), anno::GLiNEROnnx::hub_assets_available(anno::models::GLINER_PII)),
        #[cfg(feature = "onnx")]
        GlinerPoly => {
            let present = anno::backends::gliner_poly::GLiNERPoly::local_export_available();
            (present, local(present), "The default GLiNER Poly checkpoint requires a prepared export.")
        }
        #[cfg(feature = "candle")]
        GlinerCandle => {
            use anno::backends::gliner_candle::GLiNERCandle;
            hub(GLiNERCandle::cached_assets_available(anno::DEFAULT_GLINER_CANDLE_MODEL), GLiNERCandle::hub_assets_available(anno::DEFAULT_GLINER_CANDLE_MODEL))
        }
        #[cfg(feature = "gliner2-fastino")]
        Gliner2Pii => {
            let present = anno::backends::gliner2_fastino::GLiNER2Fastino::pii_assets_available();
            (present, local(present), if present { "Prepared local PII export found. Runtime loading has not been tested." }
             else { "Automatic CONTACT inference requires the local PII export. Explicit selection uses the configured directory or pinned Hub revision." })
        }
        #[cfg(feature = "onnx")]
        Adaptner | Biomedical => {
            let present = BackendFactory::optional_model_files_available(model.name()) == Some(true);
            (present, local(present), "This backend requires a local ONNX export.")
        }
        Auto | Stacked => (true, "built_in_fallback", "The stack always includes pattern and heuristic layers. With ONNX it tries BERT and NuNER, then GLiNER if neither loads."),
        #[cfg(feature = "heuristic-fr")]
        HeuristicFr => (true, "built_in", "Uses built-in French rules."),
        UniversalNer => (true, "external_service", "Uses a configured LLM route. No provider or credentials were probed."),
        Ensemble => (true, "built_in_fallback", "Constructs an ensemble with optional model components and built-in layers."),
        Pattern | Minimal | Heuristic | HeuristicCrf | Crf | Hmm => (true, "built_in", "Uses built-in rules or parameters."),
        _ => (true, "not_checked", "No filesystem preflight is defined for this explicit backend."),
    };
    Preflight {
        model: model.name(),
        eligible,
        checkpoint_override,
        local_assets: assets,
        reason,
    }
}
