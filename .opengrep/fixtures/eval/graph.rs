// OpenGrep fixture, never compiled or executed.
fn graph_node() {
    // ruleid: graph-node-empty-id
    let bad = GraphNode::new("", "Person", "A");
    // ok: graph-node-empty-id
    let good = GraphNode::new("1", "Person", "A");
}

fn lock_paths() {
    // ruleid: mutex-double-lock-deadlock
    if let Ok(mut guard) = self.cache.lock() {
        *guard = None;
    } else {
        drop(self.cache.lock().unwrap_or_else(|e| e.into_inner()));
    }

    // ok: mutex-double-lock-deadlock
    if let Ok(mut guard) = self.cache.lock() {
        *guard = None;
    } else {
        drop(self.other.lock().unwrap_or_else(|e| e.into_inner()));
    }

    // ruleid: direct-mutex-lock-bypass-helper
    let _guard = cache.lock().unwrap();
    // ok: direct-mutex-lock-bypass-helper
    let _guard = lock(&cache);
}

fn lock_expression() {
    // ruleid: mutex-double-lock-deadlock
    let guard = if let Ok(guard) = cache.lock() {
        guard
    } else {
        cache.lock().unwrap_or_else(|e| e.into_inner())
    };
}

fn variance(scores: &[f64], mean: f64) {
    // ruleid: variance-without-bessel
    let population = scores.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / scores.len() as f64;
    // ok: variance-without-bessel
    let sample = scores.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (scores.len() - 1) as f64;
}

fn slices(text: &str, scores: &[f32], test_texts: &[&str]) {
    // ruleid: byte-indexing-on-text
    let a = &text[..1];
    // ok: byte-indexing-on-text
    let b = &scores[..1];
    // ok: byte-indexing-on-text
    let c = &test_texts[..1];
}
