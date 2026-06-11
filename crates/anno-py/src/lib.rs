//! Python bindings for the `anno` entity extraction crate.
//!
//! Thin binding layer: type conversion and error mapping only. All extraction
//! logic lives in `anno`. Built without the `onnx` feature, so the default
//! stack is pattern + heuristic extraction: no model downloads, works offline.
//!
//! Offsets are **character offsets** (Unicode scalar values), which match
//! Python `str` indexing exactly: `text[e.start:e.end] == e.text`.

use std::sync::OnceLock;

use anno::{Model, StackedNER};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

/// Convert an `anno::Error` into the closest Python exception.
///
/// Input problems become `ValueError`; everything else (model init,
/// inference, IO) becomes `RuntimeError`.
fn to_py_err(err: anno::Error) -> PyErr {
    match err {
        anno::Error::InvalidInput(msg) => PyValueError::new_err(msg),
        anno::Error::Parse(msg) => PyValueError::new_err(format!("parse error: {msg}")),
        other => PyRuntimeError::new_err(other.to_string()),
    }
}

/// An extracted entity span.
///
/// `start` and `end` are character offsets (Unicode code points, end
/// exclusive), not byte offsets. They index directly into a Python string:
/// `text[e.start:e.end] == e.text`.
#[pyclass(frozen, module = "anno_py")]
pub struct Entity {
    /// Surface form as it appears in the source text.
    #[pyo3(get)]
    text: String,
    /// Entity type label ("PER", "ORG", "LOC", "EMAIL", "DATE", ...).
    #[pyo3(get)]
    label: String,
    /// Start position, character offset (inclusive).
    #[pyo3(get)]
    start: usize,
    /// End position, character offset (exclusive).
    #[pyo3(get)]
    end: usize,
    /// Confidence score in [0.0, 1.0].
    #[pyo3(get)]
    confidence: f64,
}

#[pymethods]
impl Entity {
    fn __repr__(&self) -> String {
        format!(
            "Entity(text={:?}, label={:?}, start={}, end={}, confidence={:.2})",
            self.text, self.label, self.start, self.end, self.confidence
        )
    }
}

impl From<anno::Entity> for Entity {
    fn from(e: anno::Entity) -> Self {
        Entity {
            label: e.entity_type.to_string(),
            start: e.start(),
            end: e.end(),
            confidence: e.confidence.value(),
            text: e.text,
        }
    }
}

/// Run extraction with the GIL released and convert the results.
fn run_extract(py: Python<'_>, model: &StackedNER, text: &str) -> PyResult<Vec<Entity>> {
    let entities = py
        .detach(|| model.extract_entities(text, None))
        .map_err(to_py_err)?;
    Ok(entities.into_iter().map(Entity::from).collect())
}

/// Entity extractor backed by `anno::StackedNER`.
///
/// Construct once and reuse: construction assembles the backend stack.
/// Thread-safe; `extract` releases the GIL while running.
#[pyclass(frozen, module = "anno_py")]
pub struct Extractor {
    inner: StackedNER,
}

#[pymethods]
impl Extractor {
    /// Create an extractor with the best available backend stack
    /// (pattern + heuristic in this build; no model downloads).
    #[new]
    fn new() -> Self {
        Extractor {
            inner: StackedNER::default(),
        }
    }

    /// Extract entities from `text`.
    ///
    /// Returns a list of `Entity`. Offsets are character offsets into `text`
    /// (end exclusive), so `text[e.start:e.end] == e.text`.
    fn extract(&self, py: Python<'_>, text: &str) -> PyResult<Vec<Entity>> {
        run_extract(py, &self.inner, text)
    }
}

/// Process-wide default model for the module-level `extract()`.
static DEFAULT_MODEL: OnceLock<StackedNER> = OnceLock::new();

/// Extract entities from `text` using a shared default extractor.
///
/// One-liner counterpart to `Extractor().extract(text)`; the underlying
/// model is built once per process and reused. Offsets are character
/// offsets (end exclusive): `text[e.start:e.end] == e.text`.
#[pyfunction]
fn extract(py: Python<'_>, text: &str) -> PyResult<Vec<Entity>> {
    let model = DEFAULT_MODEL.get_or_init(StackedNER::default);
    run_extract(py, model, text)
}

/// Python module definition.
#[pymodule]
fn anno_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Entity>()?;
    m.add_class::<Extractor>()?;
    m.add_function(wrap_pyfunction!(extract, m)?)?;
    Ok(())
}
