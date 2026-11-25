// SPDX-License-Identifier: MIT OR Apache-2.0
// Synthetic Rust syntax fixtures for the NLP and ML advisories.

fn entities(text: &str, start: usize, end: usize) {
    // ruleid: entity-creation-without-validation
    check(Entity::new(text, kind, start, start, conf));
    // ok: entity-creation-without-validation
    check(Entity::new(text, kind, start, end, conf));
    // ruleid: entity-created-without-validation
    let unchecked = Entity::new(text, kind, start, end, conf);
    output.push(unchecked);
    // ok: entity-created-without-validation
    let checked = Entity::new(text, kind, start, end, conf);
    if checked.is_valid(text) { output.push(checked); }
}

// ruleid: span-overlap-calculation
fn overlap_score(n: f64, d: f64) -> f64 { n / d }
// ok: span-overlap-calculation
fn overlap_guarded(n: f64, d: f64) -> f64 {
    if d == 0.0 { return 0.0; }
    n / d
}

fn assets() {
    // ruleid: model-download-without-error-context
    let vague = repo.get(path).map_err(|e| Error::Retrieval(format!("Failed: {}", e)))?;
    // ok: model-download-without-error-context
    let clear = repo.get(path).map_err(|e| Error::Retrieval(format!("model {path}: {e}")))?;
    // ruleid: hf-authentication-check
    let selected = api.model(model_id);
    let weights = selected.get(path).map_err(|e| Error::Retrieval(e.to_string()))?;
    // ok: hf-authentication-check
    let selected = api.model(model_id);
    let weights = selected.get(path).map_err(|e| classify_hub_error(model_id, e))?;
    // ruleid: onnx-session-without-error-context
    Session::builder().commit_from_file(path).map_err(|e| Error::Retrieval(e.to_string()))?;
    // ok: onnx-session-without-error-context
    Session::builder().commit_from_file(path).map_err(|e| Error::Retrieval(format!("{path}: {e}")))?;
    // ruleid: tokenizer-load-without-error-context
    Tokenizer::from_file(path).map_err(|e| Error::Retrieval(format!("{}", e)))?;
    // ok: tokenizer-load-without-error-context
    Tokenizer::from_file(path).map_err(|e| Error::Retrieval(format!("tokenizer {path}: {e}")))?;
}

fn estimates(scores: &[f64], tp: usize, fp: usize) {
    // ruleid: confidence-interval-variance-calculation
    let variance = scores.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / scores.len() as f64;
    // ok: confidence-interval-variance-calculation
    let variance = scores.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (scores.len() - 1) as f64;
    // ruleid: f1-precision-recall-validation
    let ratio = tp / (tp + fp);
    if tp + fp == 0 { return 0; }
    // ok: f1-precision-recall-validation
    let checked = tp / (tp + fp);
}

fn cache_and_sessions() {
    // ruleid: per-example-cache-clone-in-loop
    for example in examples {
        let snapshot = cache.lock().unwrap().clone();
        use_it(snapshot, example);
    }
    // ok: per-example-cache-clone-in-loop
    let snapshot = cache.lock().unwrap().clone();
    for example in examples { use_it(&snapshot, example); }
    // ruleid: backend-factory-mismatch
    let wrong = BackendFactory::create("coref_resolver")?;
    wrong.extract_entities(text);
    // ok: backend-factory-mismatch
    let right = BackendFactory::create("heuristic")?;
    right.extract_entities(text);
    // ruleid: session-pool-acquire-without-release
    let lost = pool.acquire()?;
    std::mem::forget(lost);
    // ok: session-pool-acquire-without-release
    let guard = pool.acquire()?;
    drop(guard);
    // ruleid: onnx-session-reuse-check
    for doc in docs { let session = Session::builder().commit_from_file(path)?; use_it(session, doc); }
    // ok: onnx-session-reuse-check
    let session = Session::builder().commit_from_file(path)?;
    for doc in docs { use_it(&session, doc); }
}

fn run_encoded() {
    // ruleid: encoding-overflow-check
    let encoded = tokenizer.encode(text, true)?;
    let ids = encoded.get_ids();
    model.run(ids)?;
}

fn run_bounded_encoding() {
    // ok: encoding-overflow-check
    let encoded = tokenizer.encode(text, true)?;
    let ids = encoded.get_ids();
    if ids.len() > max_len { return Err(too_long); }
    model.run(ids)?;
}

fn first_record(id: u64) {
    // ruleid: dataset-load-without-empty-check
    let rows = load_dataset(id)?;
    rows[0]
}

fn checked_record(id: u64) {
    // ok: dataset-load-without-empty-check
    let rows = load_dataset(id)?;
    rows.first()
}

fn parse(input: &str) {
    // ruleid: json-parsing-without-error-context
    serde_json::from_str(input).map_err(|e| Error::Parse(e.to_string()))?;
    // ok: json-parsing-without-error-context
    serde_json::from_str(input).map_err(|e| Error::Parse(format!("record 3: {e}")))?;
}
