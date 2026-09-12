//! Python bindings for the `anno` entity extraction crate.
//!
//! Thin binding layer: type conversion and error mapping only. All extraction
//! logic lives in `anno`. The default Python stack is pattern + heuristic
//! extraction even in an ONNX-enabled wheel: no model downloads, works offline.
//!
//! Offsets are **character offsets** (Unicode scalar values), which match
//! Python `str` indexing exactly: `text[e.start:e.end] == e.text`.

use std::collections::BTreeSet;
use std::sync::OnceLock;

#[cfg(feature = "fastino")]
use anno::backends::gliner2_fastino::{GLiNER2Fastino, SUPPORTED_GLINER2_FASTINO_MODEL};
#[cfg(feature = "fastino")]
use anno::backends::inference::ZeroShotNER;
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

/// A Fastino single-label classification result.
///
/// Results are sorted by descending `probability`. The values are softmax
/// probabilities over exactly the labels supplied to
/// [`FastinoExtractor::classify`].
#[derive(Debug)]
#[pyclass(frozen, module = "anno_py")]
pub struct Classification {
    /// One of the labels provided to `FastinoExtractor.classify`.
    #[pyo3(get)]
    label: String,
    /// Softmax probability for `label` within the supplied label set.
    #[pyo3(get)]
    probability: f32,
}

#[pymethods]
impl Classification {
    fn __repr__(&self) -> String {
        format!(
            "Classification(label={:?}, probability={:.4})",
            self.label, self.probability
        )
    }
}

fn validate_labels(labels: Vec<String>) -> PyResult<Vec<String>> {
    if labels.is_empty() || labels.iter().any(|label| label.trim().is_empty()) {
        return Err(PyValueError::new_err(
            "labels must contain at least one non-empty label",
        ));
    }
    Ok(labels)
}

fn validate_threshold(threshold: f32) -> PyResult<f32> {
    if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
        return Err(PyValueError::new_err(
            "threshold must be a finite number between 0.0 and 1.0",
        ));
    }
    Ok(threshold)
}

fn validate_fastino_labels(labels: Vec<String>) -> PyResult<Vec<String>> {
    let labels = validate_labels(labels)?;
    let unique: BTreeSet<&str> = labels.iter().map(String::as_str).collect();
    if unique.len() != labels.len() {
        return Err(PyValueError::new_err(
            "Fastino labels must not contain duplicates",
        ));
    }
    Ok(labels)
}

/// Convert Fastino's classifier output only after proving it is a usable
/// single-label probability distribution.
///
/// The Rust pipeline intentionally returns all zeroes when its count head
/// predicts zero spans. That is not a classification result and must never be
/// exposed as one to Python callers.
#[cfg(any(feature = "fastino", test))]
fn convert_fastino_classifications(
    requested_labels: &[String],
    results: Vec<(String, f32)>,
) -> Result<Vec<Classification>, String> {
    if results.len() != requested_labels.len() {
        return Err(format!(
            "Fastino classifier returned {} scores for {} requested labels",
            results.len(),
            requested_labels.len()
        ));
    }

    let requested: BTreeSet<&str> = requested_labels.iter().map(String::as_str).collect();
    let returned: BTreeSet<&str> = results.iter().map(|(label, _)| label.as_str()).collect();
    if returned.len() != results.len() || returned != requested {
        return Err(
            "Fastino classifier returned labels that do not match the requested labels".into(),
        );
    }

    let mut total = 0.0_f64;
    for (index, (_, probability)) in results.iter().enumerate() {
        if !probability.is_finite() || *probability < 0.0 {
            return Err(format!(
                "Fastino classifier returned an invalid probability at index {index}",
            ));
        }
        if index > 0 && results[index - 1].1 < *probability {
            return Err(
                "Fastino classifier results are not sorted by descending probability".into(),
            );
        }
        total += f64::from(*probability);
    }

    if total == 0.0 {
        return Err(
            "Fastino classifier produced no classification: its internal count head predicted zero spans"
                .into(),
        );
    }
    if (total - 1.0).abs() > 1e-4 {
        return Err(format!(
            "Fastino classifier returned a non-normalized probability distribution (sum {total})",
        ));
    }

    Ok(results
        .into_iter()
        .map(|(label, probability)| Classification { label, probability })
        .collect())
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
        model: Box<GLiNEROnnx>,
        labels: Vec<String>,
        threshold: f32,
    },
}

/// Explicit source-build binding for the experimental fastino GLiNER2 model.
///
/// This is intentionally separate from [`Extractor`]: Fastino has both
/// zero-shot entity extraction and single-label classification, while the
/// other `Extractor` backends do not offer the latter operation. Build with
/// the optional `fastino` Cargo feature; the default Python wheel remains
/// offline-only and never bundles Fastino weights.
#[pyclass(frozen, module = "anno_py")]
pub struct FastinoExtractor {
    #[cfg(feature = "fastino")]
    model: GLiNER2Fastino,
}

#[pymethods]
impl FastinoExtractor {
    /// Load a Fastino GLiNER2 model.
    ///
    /// Omit `model` to use anno's verified, revision-pinned Fastino export.
    /// A supplied Hugging Face model id is accepted for experimentation but
    /// may select a floating revision. Set `ANNO_NO_DOWNLOADS=1` to require a
    /// cached model; cache misses then raise `RuntimeError` without a network
    /// fallback.
    #[new]
    #[pyo3(signature = (model=None))]
    fn new(py: Python<'_>, model: Option<String>) -> PyResult<Self> {
        #[cfg(feature = "fastino")]
        {
            let model = model.unwrap_or_else(|| SUPPORTED_GLINER2_FASTINO_MODEL.to_owned());
            if model.trim().is_empty() {
                return Err(PyValueError::new_err("model must not be empty"));
            }
            let backend = py
                .detach(|| GLiNER2Fastino::from_pretrained(&model))
                .map_err(to_py_err)?;
            Ok(Self { model: backend })
        }
        #[cfg(not(feature = "fastino"))]
        {
            let _ = (py, model);
            Err(PyRuntimeError::new_err(
                "FastinoExtractor requires a Fastino-enabled source build; rebuild with maturin --features extension-module,fastino",
            ))
        }
    }

    /// Extract typed entity spans from `text`.
    ///
    /// `labels` are Fastino's zero-shot entity labels. `threshold` is an
    /// inclusive, finite probability threshold in `[0.0, 1.0]`. Returned
    /// offsets are Python character offsets (end exclusive).
    #[pyo3(signature = (text, labels, threshold=0.5))]
    fn extract(
        &self,
        py: Python<'_>,
        text: &str,
        labels: Vec<String>,
        threshold: f32,
    ) -> PyResult<Vec<Entity>> {
        let labels = validate_fastino_labels(labels)?;
        let threshold = validate_threshold(threshold)?;

        #[cfg(feature = "fastino")]
        {
            let labels: Vec<&str> = labels.iter().map(String::as_str).collect();
            let entities = py
                .detach(|| self.model.extract_with_types(text, &labels, threshold))
                .map_err(to_py_err)?;
            Ok(entities.into_iter().map(Entity::from).collect())
        }
        #[cfg(not(feature = "fastino"))]
        {
            let _ = (py, text, labels, threshold);
            Err(PyRuntimeError::new_err(
                "FastinoExtractor requires a Fastino-enabled source build; rebuild with maturin --features extension-module,fastino",
            ))
        }
    }

    /// Classify `text` into the supplied labels.
    ///
    /// Fastino's current classifier is single-label. Results include every
    /// requested label, ordered by descending probability. Its Rust API has a
    /// future-facing threshold argument which is currently ignored, so this
    /// binding deliberately does not expose one.
    fn classify(
        &self,
        py: Python<'_>,
        text: &str,
        labels: Vec<String>,
    ) -> PyResult<Vec<Classification>> {
        let labels = validate_fastino_labels(labels)?;

        #[cfg(feature = "fastino")]
        {
            let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();
            let results = py
                .detach(|| self.model.classify(text, &label_refs, 0.0))
                .map_err(to_py_err)?;
            convert_fastino_classifications(&labels, results).map_err(PyRuntimeError::new_err)
        }
        #[cfg(not(feature = "fastino"))]
        {
            let _ = (py, text, labels);
            Err(PyRuntimeError::new_err(
                "FastinoExtractor requires a Fastino-enabled source build; rebuild with maturin --features extension-module,fastino",
            ))
        }
    }
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
                    Ok(Self {
                        backend: ExtractorBackend::Bert(backend),
                    })
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
                    Ok(Self {
                        backend: ExtractorBackend::Gliner {
                            model: Box::new(backend),
                            labels,
                            threshold,
                        },
                    })
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
    m.add("__fastino_enabled__", cfg!(feature = "fastino"))?;
    m.add_class::<Entity>()?;
    m.add_class::<Classification>()?;
    m.add_class::<Extractor>()?;
    m.add_class::<FastinoExtractor>()?;
    m.add_function(wrap_pyfunction!(extract, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labels() -> Vec<String> {
        vec!["positive".into(), "negative".into(), "neutral".into()]
    }

    #[test]
    fn fastino_classification_conversion_accepts_a_sorted_distribution() {
        let converted = convert_fastino_classifications(
            &labels(),
            vec![
                ("positive".into(), 0.7),
                ("neutral".into(), 0.2),
                ("negative".into(), 0.1),
            ],
        )
        .expect("valid Fastino classifier distribution");

        assert_eq!(converted[0].label, "positive");
        assert!((converted[0].probability - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn fastino_classification_conversion_rejects_zero_count_scores() {
        let error = convert_fastino_classifications(
            &labels(),
            vec![
                ("positive".into(), 0.0),
                ("negative".into(), 0.0),
                ("neutral".into(), 0.0),
            ],
        )
        .expect_err("all-zero scores are Fastino's no-classification sentinel");

        assert!(error.contains("produced no classification"));
    }

    #[test]
    fn fastino_classification_conversion_rejects_invalid_or_unsorted_results() {
        let cases = [
            vec![
                ("positive".into(), f32::NAN),
                ("negative".into(), 0.0),
                ("neutral".into(), 0.0),
            ],
            vec![
                ("positive".into(), 0.1),
                ("negative".into(), 0.8),
                ("neutral".into(), 0.1),
            ],
            vec![
                ("positive".into(), 0.6),
                ("negative".into(), 0.3),
                ("unexpected".into(), 0.1),
            ],
        ];

        for scores in cases {
            assert!(convert_fastino_classifications(&labels(), scores).is_err());
        }
    }
}
