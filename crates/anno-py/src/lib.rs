//! Python bindings for the `anno` entity extraction crate.
//!
//! Thin binding layer: type conversion and error mapping only. All extraction
//! logic lives in `anno`. The default Python stack is pattern + heuristic
//! extraction even in an ONNX-enabled wheel: no model downloads, works offline.
//!
//! Offsets are **character offsets** (Unicode scalar values), which match
//! Python `str` indexing exactly: `text[e.start:e.end] == e.text`.

use std::sync::OnceLock;

#[cfg(feature = "onnx")]
use anno::{BertNEROnnx, GLiNEROnnx};
use anno::{HeuristicNER, Model, RegexNER, StackedNER};
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

/// Run Model extraction with the GIL released and convert the results.
fn run_model_extract<M: Model>(py: Python<'_>, model: &M, text: &str) -> PyResult<Vec<Entity>> {
    let entities = py
        .detach(|| model.extract_entities(text, None))
        .map_err(to_py_err)?;
    Ok(entities.into_iter().map(Entity::from).collect())
}

/// The explicit offline stack used by both Python convenience APIs.
///
/// `StackedNER::default()` can attempt ML initialization when `anno/onnx` is
/// enabled. Python's offline backend must retain the no-download behavior in
/// either build, so construct the documented pattern + heuristic fallback.
fn offline_stack() -> StackedNER {
    StackedNER::builder()
        .layer(RegexNER::new())
        .layer(HeuristicNER::new())
        .build()
}

enum ExtractorBackend {
    Offline(StackedNER),
    #[cfg(feature = "onnx")]
    Bert(BertNEROnnx),
    #[cfg(feature = "onnx")]
    Gliner {
        model: GLiNEROnnx,
        labels: Vec<String>,
        threshold: f32,
    },
}

/// Entity extractor with an explicitly selected backend.
///
/// `backend="offline"` is the default pattern + heuristic stack and makes no
/// model downloads. Model-backed backends require a wheel built with the
/// optional `onnx` feature. Construct once and reuse; `extract` releases the
/// GIL while running.
#[pyclass(frozen, module = "anno_py")]
pub struct Extractor {
    backend: ExtractorBackend,
}

#[pymethods]
impl Extractor {
    /// Create an extractor.
    ///
    /// `backend` is one of `"offline"`, `"bert"`, or `"gliner"`.
    /// `model` is a HuggingFace model ID or a local model directory for BERT;
    /// omit it to use the backend's documented default model. `labels` and
    /// `threshold` apply only to GLiNER and replace its default entity types.
    /// Invalid combinations raise `ValueError`; a model backend in a wheel
    /// without ONNX support raises `RuntimeError` rather than falling back.
    #[new]
    #[pyo3(signature = (backend="offline", model=None, labels=None, threshold=None))]
    fn new(
        py: Python<'_>,
        backend: &str,
        model: Option<String>,
        labels: Option<Vec<String>>,
        threshold: Option<f32>,
    ) -> PyResult<Self> {
        #[cfg(not(feature = "onnx"))]
        let _ = py;
        match backend {
            "offline" => {
                if model.is_some() || labels.is_some() || threshold.is_some() {
                    return Err(PyValueError::new_err(
                        "backend='offline' does not accept model, labels, or threshold",
                    ));
                }
                Ok(Self {
                    backend: ExtractorBackend::Offline(offline_stack()),
                })
            }
            "bert" => {
                if labels.is_some() || threshold.is_some() {
                    return Err(PyValueError::new_err(
                        "backend='bert' does not support zero-shot labels or threshold",
                    ));
                }
                let model = model.unwrap_or_else(|| anno::models::BERT_ONNX.to_owned());
                if model.trim().is_empty() {
                    return Err(PyValueError::new_err("model must not be empty"));
                }
                #[cfg(feature = "onnx")]
                {
                    let backend = py.detach(|| BertNEROnnx::new(&model)).map_err(to_py_err)?;
                    return Ok(Self {
                        backend: ExtractorBackend::Bert(backend),
                    });
                }
                #[cfg(not(feature = "onnx"))]
                {
                    let _ = model;
                    Err(PyRuntimeError::new_err(
                        "backend='bert' requires an ONNX-enabled wheel; rebuild with maturin --features extension-module,onnx",
                    ))
                }
            }
            "gliner" => {
                let threshold = threshold.unwrap_or(0.5);
                if !(0.0..=1.0).contains(&threshold) {
                    return Err(PyValueError::new_err(
                        "threshold must be between 0.0 and 1.0",
                    ));
                }
                let model = model.unwrap_or_else(|| anno::models::GLINER.to_owned());
                if model.trim().is_empty() {
                    return Err(PyValueError::new_err("model must not be empty"));
                }
                let labels = labels.unwrap_or_else(|| {
                    vec!["person".into(), "organization".into(), "location".into()]
                });
                if labels.is_empty() || labels.iter().any(|label| label.trim().is_empty()) {
                    return Err(PyValueError::new_err(
                        "labels must contain at least one non-empty label",
                    ));
                }
                #[cfg(feature = "onnx")]
                {
                    let backend = py.detach(|| GLiNEROnnx::new(&model)).map_err(to_py_err)?;
                    return Ok(Self {
                        backend: ExtractorBackend::Gliner {
                            model: backend,
                            labels,
                            threshold,
                        },
                    });
                }
                #[cfg(not(feature = "onnx"))]
                {
                    let _ = (model, labels);
                    Err(PyRuntimeError::new_err(
                        "backend='gliner' requires an ONNX-enabled wheel; rebuild with maturin --features extension-module,onnx",
                    ))
                }
            }
            _ => Err(PyValueError::new_err(
                "backend must be one of: 'offline', 'bert', 'gliner'",
            )),
        }
    }

    /// Extract entities from `text`.
    ///
    /// Returns a list of `Entity`. Offsets are character offsets into `text`
    /// (end exclusive), so `text[e.start:e.end] == e.text`.
    fn extract(&self, py: Python<'_>, text: &str) -> PyResult<Vec<Entity>> {
        match &self.backend {
            ExtractorBackend::Offline(model) => run_model_extract(py, model, text),
            #[cfg(feature = "onnx")]
            ExtractorBackend::Bert(model) => run_model_extract(py, model, text),
            #[cfg(feature = "onnx")]
            ExtractorBackend::Gliner {
                model,
                labels,
                threshold,
            } => {
                let labels: Vec<&str> = labels.iter().map(String::as_str).collect();
                let entities = py
                    .detach(|| model.extract(text, &labels, *threshold))
                    .map_err(to_py_err)?;
                Ok(entities.into_iter().map(Entity::from).collect())
            }
        }
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
    let model = DEFAULT_MODEL.get_or_init(offline_stack);
    run_model_extract(py, model, text)
}

/// Python module definition.
#[pymodule]
fn anno_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__onnx_enabled__", cfg!(feature = "onnx"))?;
    m.add_class::<Entity>()?;
    m.add_class::<Extractor>()?;
    m.add_function(wrap_pyfunction!(extract, m)?)?;
    Ok(())
}
