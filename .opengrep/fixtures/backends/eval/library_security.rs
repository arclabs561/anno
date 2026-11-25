// OpenGrep fixture; these examples are never compiled or run.
fn library_paths(items: &[i32], index: usize, value: i32, divisor: i32) {
    // ruleid: panic-in-library
    panic!("example diagnostic");
    // ruleid: unchecked-array-access
    let item = items[index];
    if index < items.len() {
        // ok: unchecked-array-access
        let item = items[index];
    }
    // ruleid: division-without-zero-check
    let ratio = value / divisor;
    if divisor != 0 {
        // ok: division-without-zero-check
        let ratio = value / divisor;
    }
}

fn guarded(value: i32, divisor: i32) -> Result<i32, ()> {
    if divisor == 0 { return Err(()); }
    // ok: division-without-zero-check
    Ok(value / divisor)
}

#[cfg(test)]
mod checks {
    fn helper() {
        // ok: panic-in-library
        panic!("fixture diagnostic");
    }
}

fn local_blocks() {
    // ruleid: unsafe-block-without-comment
    unsafe { marker(); }
    // ok: unsafe-block-without-comment
    // SAFETY: marker is a no-op in this source fixture.
    unsafe { marker(); }
    // SAFETY: This source-only fixture uses a marker.
    // This second line continues the explanation.
    let recorded =
        unsafe { marker() };
}

// ruleid: unsafe-function-without-docs
unsafe fn no_docs() { marker(); }

// ok: unsafe-function-without-docs
/// Does nothing.
unsafe fn documented() { marker(); }

fn explicit_lock(mutex: Mutex<()>) {
    // ruleid: mutex-lock-without-handling-poison
    let result = mutex.lock();
    // ok: mutex-lock-without-handling-poison
    let guard = mutex.lock().unwrap_or_else(|e| e.into_inner());
}

// ruleid: public-unsafe-function, unsafe-function-without-docs
pub unsafe fn public_marker() { marker(); }

fn deferred_lock(mutex: Mutex<()>) {
    // ok: mutex-lock-without-handling-poison
    let result = mutex.lock();
    match result { Ok(_) => marker(), Err(_) => marker() }
}

fn original_lints(value: Option<usize>) {
    // ruleid: dangerous-unwrap-in-library
    value.unwrap();
    // ruleid: expect-without-context
    value.expect("example");
}
