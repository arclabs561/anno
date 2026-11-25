// OpenGrep fixture; these examples are never compiled or run.
fn report(result: Result<(), InputError>, mutex: Mutex<()>) {
    // ruleid: mutex-lock-without-poison-handling
    let lock = mutex.lock();
    // ok: mutex-lock-without-poison-handling
    let guard = mutex.lock().map_err(LockError::from);
    // ruleid: error-conversion-without-context
    result.map_err(|e| Error::Read(format!("{}", e)));
    // ok: error-conversion-without-context
    result.map_err(|e| Error::Read(format!("reading input: {}", e)));
    // ruleid: result-propagation-without-context
    read()?;
    // ok: result-propagation-without-context
    read().map_err(InputError::from)?;
}

// ruleid: backend-specific-error-types
enum UnmappedError { Example }
// ok: backend-specific-error-types
enum ReadError { Example }
impl From<ReadError> for crate::Error {
    fn from(value: ReadError) -> Self { Self::Read(value) }
}
// ok: backend-specific-error-types
enum BackendMode { Quiet }

fn deferred_lock(mutex: Mutex<()>) {
    // ok: mutex-lock-without-poison-handling
    let result = mutex.lock();
    match result { Ok(_) => marker(), Err(_) => marker() }
}

impl From<OtherError> for crate::Error {
    fn from(value: OtherError) -> Self { Self::Other(value) }
}
// ok: backend-specific-error-types
enum OtherError { First, Second }

// ok: backend-specific-error-types
// enum IllustrativeError { NotAnActualEnum }
