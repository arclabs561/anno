//! Publish aggregate benchmark JSON and Markdown while retaining the native report format.
use anno_eval::eval::{
    backend_name::BackendName,
    loader::DataSource,
    provenance::{
        ArtifactProvenanceStatus, BackendRunProvenance, DatasetRunProvenance, EvalBuildProvenance,
        EvalRunProvenance, EvalRuntimeProvenance, EvalSkipReason, EvaluatedSampleProvenance,
        EvaluationScheduling, ExecutionProvenanceStatus, InferenceSettingProvenance,
        NerInferenceProvenance,
    },
    task_evaluator::{ComprehensiveEvalResults, TaskEvalResult},
    task_mapping::{get_task_backends, Task},
};
use anno_eval::muxer_harness::classify_failure_kind;
use std::{collections::BTreeSet, env, fs, path::PathBuf};

fn digest(value: Option<String>, lengths: &[usize]) -> Option<String> {
    value.filter(|s| lengths.contains(&s.len()) && s.bytes().all(|b| b.is_ascii_hexdigit()))
}

fn backend(name: &str, known: &BTreeSet<&str>) -> String {
    if let Some(canonical) = BackendName::try_parse(name) {
        canonical.as_str().to_string()
    } else if known.contains(name) {
        name.to_string()
    } else {
        "unreported".to_string()
    }
}

fn metric(name: &str) -> bool {
    matches!(
        name,
        "f1" | "precision"
            | "recall"
            | "accuracy"
            | "macro_f1"
            | "micro_f1"
            | "weighted_f1"
            | "strict_f1"
            | "boundary_f1"
            | "conll_f1"
            | "muc_f1"
            | "b3_f1"
            | "ceaf_f1"
            | "ceafe_f1"
            | "num_examples"
            | "correct"
            | "abstained"
            | "num_gold_relations"
            | "num_predicted_relations"
            | "oracle_docs_used"
            | "predicted_endpoint_strict_f1"
            | "gold_endpoint_strict_f1"
            | "fixed_label_prompt_docs"
            | "gold_label_prompt_docs"
            | "num_docs"
    )
}

fn aggregate(row: TaskEvalResult, known: &BTreeSet<&str>) -> TaskEvalResult {
    let skip = row.skip_reason();
    let p = row.provenance;
    let b = p.build;
    let d = p.dataset;
    let rt = p.runtime;
    let sources = [
        DataSource::S3Cache,
        DataSource::LocalCache,
        DataSource::OriginalUrl,
        DataSource::Skipped,
        DataSource::Embedded,
    ];
    let source = sources
        .iter()
        .find(|s| s.description() == d.data_source)
        .map_or("unreported", DataSource::description);
    let scheduling = if let Some(reason) = skip {
        EvaluationScheduling::NotRun {
            reason: reason.code().into(),
        }
    } else {
        match rt.scheduling {
            EvaluationScheduling::Sequential => EvaluationScheduling::Sequential,
            EvaluationScheduling::Rayon {
                configured_threads,
                effective_threads,
            } => EvaluationScheduling::Rayon {
                configured_threads,
                effective_threads,
            },
            _ => EvaluationScheduling::Unknown {
                reason: "unreported".into(),
            },
        }
    };
    let ner_label_policy = p.ner_label_policy.map(|policy| {
        let labels = row.dataset.entity_types();
        anno_eval::eval::provenance::NerLabelPolicyReport {
            primary_policy: "strict span-and-type micro score over all emitted entities".into(),
            dataset_label_space: policy
                .dataset_label_space
                .into_iter()
                .filter(|s| labels.iter().any(|label| label.eq_ignore_ascii_case(s)))
                .collect(),
            operational_gold_outside_dataset_labels: policy.operational_gold_outside_dataset_labels,
            operational_predictions_outside_dataset_labels: policy
                .operational_predictions_outside_dataset_labels,
            dataset_label_closed_diagnostic: policy.dataset_label_closed_diagnostic,
        }
    });
    let evaluated_sample = p.evaluated_sample.map(|s| EvaluatedSampleProvenance {
        schema_version: s.schema_version,
        unit: s.unit,
        unit_count: s.unit_count,
        gold_item_count: s.gold_item_count,
        sample_sha256: digest(s.sample_sha256, &[64]),
        relation_producer_documents: Default::default(),
    });
    TaskEvalResult {
        task: row.task,
        dataset: row.dataset,
        backend: backend(&row.backend, known),
        backend_display: None,
        seed: row.seed,
        success: row.success,
        num_examples: row.num_examples,
        duration_ms: row.duration_ms,
        error: row.error.as_deref().map(|error| match skip {
            Some(EvalSkipReason::CapabilityUnavailable) => "Feature not available".into(),
            Some(EvalSkipReason::IncompatibleEntityTypes) => "incompatible: entity types".into(),
            _ => classify_failure_kind(error).into(),
        }),
        metrics: row
            .metrics
            .into_iter()
            .filter(|(k, v)| metric(k) && v.is_finite())
            .collect(),
        label_shift: None,
        robustness: None,
        stratified: None,
        confidence_intervals: None,
        kb_version: None,
        provenance: EvalRunProvenance {
            schema_version: p.schema_version,
            build: EvalBuildProvenance {
                package_version: if b
                    .package_version
                    .split('.')
                    .all(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()))
                {
                    b.package_version
                } else {
                    "unreported".into()
                },
                source_revision: digest(b.source_revision, &[40, 64]),
                enabled_features: b
                    .enabled_features
                    .into_iter()
                    .filter(|s| {
                        matches!(
                            s.as_str(),
                            "default"
                                | "eval"
                                | "onnx"
                                | "candle"
                                | "discourse"
                                | "gliner2-fastino"
                                | "gliner2-fastino-candle"
                                | "hf-hub"
                                | "bundled-hmm-params"
                                | "bundled-crf-weights"
                                | "eval-bias"
                                | "eval-parallel"
                                | "eval-profiling"
                                | "eval-extended"
                                | "parallel"
                                | "heuristic-fr"
                                | "llm"
                        )
                    })
                    .collect(),
            },
            dataset: DatasetRunProvenance {
                // A registry URL can point to a local file, an S3 key or a private account.
                source_url: String::new(),
                data_source: source.into(),
                content_sha256: digest(d.content_sha256, &[64]),
                s3_fallback: d.s3_fallback,
                split: d
                    .split
                    .filter(|s| matches!(s.as_str(), "train" | "test" | "dev" | "validation")),
                sentence_count: d.sentence_count,
                entity_count: d.entity_count,
                ..Default::default()
            },
            backend: BackendRunProvenance {
                requested: backend(&p.backend.requested, known),
                effective: backend(&p.backend.effective, known),
                model_artifacts: ArtifactProvenanceStatus::Unknown {
                    reason: "Asset details are omitted from the aggregate CI receipt".into(),
                },
                execution: ExecutionProvenanceStatus::Unknown {
                    reason: "Runtime details are omitted from the aggregate CI receipt".into(),
                },
                ..Default::default()
            },
            runtime: EvalRuntimeProvenance {
                seed: rt.seed,
                cached_only: rt.cached_only,
                max_examples: rt.max_examples,
                confidence_intervals: rt.confidence_intervals,
                robustness: rt.robustness,
                relation_threshold: rt.relation_threshold,
                ner_inference: NerInferenceProvenance {
                    threshold: match rt.ner_inference.threshold {
                        InferenceSettingProvenance::Known { value } if value.is_number() => {
                            InferenceSettingProvenance::Known { value }
                        }
                        _ => InferenceSettingProvenance::Unknown {
                            reason: "unreported".into(),
                        },
                    },
                    chunking: InferenceSettingProvenance::Unknown {
                        reason: "unreported".into(),
                    },
                    nms: InferenceSettingProvenance::Unknown {
                        reason: "unreported".into(),
                    },
                },
                scheduling,
                ..Default::default()
            },
            evaluated_sample,
            observation_id: None,
            muxer_run_id: None,
            policy_cohort_key: None,
            ner_label_policy,
        },
    }
}

fn project(report: ComprehensiveEvalResults) -> ComprehensiveEvalResults {
    let known = Task::all()
        .iter()
        .flat_map(|t| get_task_backends(*t))
        .collect();
    let mut summary = report.summary;
    summary.backends = summary
        .backends
        .iter()
        .map(|name| backend(name, &known))
        .collect();
    ComprehensiveEvalResults {
        results: report
            .results
            .into_iter()
            .map(|row| aggregate(row, &known))
            .collect(),
        summary,
    }
}

fn run() -> Result<(), &'static str> {
    let args: Vec<PathBuf> = env::args_os().skip(1).map(Into::into).collect();
    if !(2..=3).contains(&args.len()) {
        return Err("usage: public_benchmark_report PRIVATE.json PUBLIC.md [PUBLIC.json]");
    }
    let bytes = fs::read(&args[0]).map_err(|_| {
        "No private benchmark receipt was produced. Check the build, inputs and model availability."
    })?;
    let raw = serde_json::from_slice(&bytes)
        .map_err(|_| "Invalid native benchmark receipt; aggregate publication stopped.")?;
    let public = project(raw);
    fs::write(&args[1], public.to_markdown()).map_err(|_| "Unable to write aggregate Markdown")?;
    if let Some(json) = args.get(2) {
        let encoded =
            serde_json::to_vec_pretty(&public).map_err(|_| "Unable to encode aggregate JSON")?;
        fs::write(json, encoded).map_err(|_| "Unable to write aggregate JSON")?;
    }
    Ok(())
}

fn main() -> Result<(), &'static str> {
    run()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn aggregate_keeps_native_schema_and_excludes_paths_labels_and_diagnostic_text() {
        let sentinel = "PRIVATE_PASSAGE";
        let value = json!({
            "task": "NER", "dataset": "WikiGold", "backend": "heuristic", "seed": 42,
            "success": false, "error": format!("dataset parse failed on {sentinel}"),
            "metrics": {"f1": 0.25, (sentinel): 86},
            "num_examples": 10, "duration_ms": 15, "provenance": EvalRunProvenance::default()
        });
        let mut row: TaskEvalResult = serde_json::from_value(value).unwrap();
        let p = &mut row.provenance;
        p.dataset.data_source = "local cache".into();
        p.dataset.sentence_count = 10;
        p.dataset.source_url = format!("s3://{sentinel}/key");
        p.dataset.content_sha256 = Some("ab".repeat(32));
        p.backend.requested = "heuristic".into();
        p.backend.effective = "heuristic".into();
        p.backend.configured_labels = Some(vec![sentinel.into()]);
        p.backend.display_name = Some(sentinel.into());
        p.policy_cohort_key = Some(sentinel.into());
        row.backend_display = Some(sentinel.into());
        let summary = anno_eval::eval::task_evaluator::EvalSummary {
            total_combinations: 1,
            successful: 0,
            failed: 1,
            skipped: 0,
            tasks: vec![row.task],
            datasets: vec![row.dataset],
            backends: vec!["heuristic".into()],
        };
        let public = project(ComprehensiveEvalResults {
            results: vec![row],
            summary,
        });
        let json = serde_json::to_string(&public).unwrap();
        assert!(!json.contains(sentinel) && !public.to_markdown().contains(sentinel));
        let roundtrip: ComprehensiveEvalResults = serde_json::from_str(&json).unwrap();
        let row = &roundtrip.results[0];
        assert_eq!(row.error.as_deref(), Some("dataset"));
        assert_eq!(
            row.provenance.dataset.content_sha256.as_deref(),
            Some("ab".repeat(32).as_str())
        );
        assert_eq!(row.provenance.dataset.sentence_count, 10);
        assert_eq!(row.metrics["f1"], 0.25);
        assert_eq!(row.provenance.backend.effective, "heuristic");
    }
}
