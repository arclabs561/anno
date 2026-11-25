// SPDX-License-Identifier: MIT OR Apache-2.0
// Synthetic Rust syntax fixtures for evaluation advisories.

fn inference() {
    // ruleid: evaluation-loop-backend-recreation
    for doc in docs {
        // ruleid: backend-capability-check
        let per_doc = BackendFactory::create(name)?;
        per_doc.extract_entities(doc)?;
    }
    // ok: evaluation-loop-backend-recreation
    let reused = model;
    for doc in docs { reused.extract_entities(doc)?; }
    // ok: backend-capability-check
    let checked = BackendFactory::create(name)?;
    if !checked.supports(task) { return Err(error); }
    checked.run(doc)?;
}

fn scores() {
    // ruleid: per-example-scores-multiple-iterations
    for score in scores { sum(score); }
    for result in scores { square(result); }
    mark_phase();
    // ok: per-example-scores-multiple-iterations
    for item in first { sum(item); }
    for item in second { sum(item); }
}

fn compute_confidence_intervals() {
    // ruleid: confidence-interval-recomputation
    backend.extract_entities(text);
    // ok: confidence-interval-recomputation
    intervals_from_scores(scores);
}

fn stratify() {
    // ruleid: stratified-metrics-placeholder
    by_entity_type.insert(label, MetricWithCI { mean: overall_f1, ..default_ci });
    // ok: stratified-metrics-placeholder
    by_entity_type.insert(label, MetricWithCI { mean: per_type_f1, ..default_ci });
    // ruleid: temporal-stratification-missing-metadata
    compute_temporal_stratification(docs, None, config);
    // ok: temporal-stratification-missing-metadata
    compute_temporal_stratification(docs, dates, config);
}

fn harmonic(p: f64, r: f64) {
    // ruleid: metric-calculation-without-edge-cases
    let first = 2.0 * (p * r) / (p + r);
    if p + r == 0.0 { return 0.0; }
    // ok: metric-calculation-without-edge-cases
    let checked = 2.0 * (p * r) / (p + r);
}

fn scopes() {
    // ruleid: task-dataset-mismatch
    for task in tasks {
        for dataset in datasets { evaluate(task, dataset, cfg)?; }
    }
    // ok: task-dataset-mismatch
    for task in tasks {
        for dataset in datasets {
            if !is_valid_combination(task, dataset) { continue; }
            evaluate(task, dataset, cfg)?;
        }
    }
    // ruleid: bias-stratification-empty-groups
    for group in groups { compute_metrics(group)?; }
    // ok: bias-stratification-empty-groups
    for group in groups {
        if group.is_empty() { continue; }
        compute_metrics(group)?;
    }
}

fn perturb() {
    // ruleid: robustness-testing-limit-check
    for example in examples { let changed = apply_perturbation(example, seed)?; use_it(changed); }
    // ok: robustness-testing-limit-check
    for example in examples.take(limit) { let changed = apply_perturbation(example, seed)?; use_it(changed); }
}

fn coreference() {
    // ruleid: coref-chain-validation
    save(CorefChain { mentions: Vec::new(), ..chain });
    // ok: coref-chain-validation
    save(CorefChain { mentions: known, ..chain });
    // ruleid: coref-chain-validation
    save(CorefChain::new(vec![]));
    // ok: coref-chain-validation
    save(CorefChain::new(known));
    // ruleid: coref-metric-computation
    compute_muc(gold, predicted)?;
    compute_b3(predicted, gold)?;
}

fn consistent_metrics() {
    // ok: coref-metric-computation
    compute_muc(gold, predicted)?;
    compute_b3(gold, predicted)?;
}
