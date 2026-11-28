//! Otter open-type multilingual NER using pinned local mmBERT checkpoints.
//! A CPU Python session retains the original Transformers model and decoder.
use super::inference::ZeroShotNER;
use super::python_worker::PythonWorker;
use crate::{Entity, EntityType, Error, Language, Model, Provenance, Result};
use serde::Deserialize;
use std::ffi::OsStr;
use std::path::Path;

const DEFAULT_LABELS: &[&str] = &["person", "organization", "location"];

/// Architecture of a published Otter mmBERT checkpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OtterVariant {
    /// A single encoder sees the schema prefix and the input text together.
    Cross,
    /// Separate encoders process entity type names and the input text.
    Bi,
}

impl OtterVariant {
    /// Check snapshot file names and sizes before scheduling an evaluation.
    /// The Python worker verifies the file hashes when it loads the checkpoint.
    pub fn model_files_available(self) -> bool {
        let Some(dir) = std::env::var_os(self.checkpoint_var()) else {
            return false;
        };
        let arch = match self {
            Self::Cross => "cross",
            Self::Bi => "bi",
        };
        let manifest: serde_json::Value =
            serde_json::from_str(include_str!("models.json")).expect("embedded Otter manifest");
        manifest[arch]["files"].as_object().is_some_and(|files| {
            files.iter().all(|(name, file)| {
                Path::new(&dir)
                    .join(name)
                    .metadata()
                    .is_ok_and(|m| m.is_file() && Some(m.len()) == file["size"].as_u64())
            })
        })
    }

    fn name(self) -> &'static str {
        match self {
            Self::Cross => "otter-cross",
            Self::Bi => "otter-bi",
        }
    }

    fn checkpoint_var(self) -> &'static str {
        match self {
            Self::Cross => "OTTER_CROSS_MODEL_DIR",
            Self::Bi => "OTTER_BI_MODEL_DIR",
        }
    }
}

/// A persistent Python session serving a verified, local Otter checkpoint.
pub struct Otter {
    variant: OtterVariant,
    worker: PythonWorker,
}

impl Otter {
    /// Start the CPU runtime and verify the selected model before accepting requests.
    pub fn new(
        python: impl AsRef<OsStr>,
        checkpoint: impl AsRef<Path>,
        variant: OtterVariant,
    ) -> Result<Self> {
        let directory = checkpoint.as_ref();
        if !directory.is_dir() {
            return Err(Error::model_init(format!(
                "Otter needs a local checkpoint directory: {}",
                directory.display()
            )));
        }
        let architecture = match variant {
            OtterVariant::Cross => "cross",
            OtterVariant::Bi => "bi",
        };
        let args = vec![
            directory.as_os_str().to_owned(),
            architecture.into(),
            include_str!("models.json").into(),
        ];
        let worker = PythonWorker::start(
            python.as_ref(),
            include_str!("worker.py"),
            args,
            variant.name(),
        )?;
        Ok(Self { variant, worker })
    }

    /// Read the architecture-specific directory and optional ANNO_OTTER_PYTHON interpreter.
    pub fn from_env(variant: OtterVariant) -> Result<Self> {
        let var = variant.checkpoint_var();
        let checkpoint = std::env::var_os(var).ok_or_else(|| {
            Error::model_init(format!(
                "set {var} to the local pinned {} checkpoint directory",
                variant.name()
            ))
        })?;
        let python = std::env::var_os("ANNO_OTTER_PYTHON").unwrap_or_else(|| "python3".into());
        Self::new(python, checkpoint, variant)
    }
}

#[derive(Deserialize)]
struct Prediction {
    start: usize,
    end: usize,
    label: String,
    text: String,
    score: f64,
}

fn entities(
    text: &str,
    labels: &[&str],
    predictions: Vec<Prediction>,
    name: &'static str,
) -> Result<Vec<Entity>> {
    let offsets = crate::offset::build_char_to_byte_map(text);
    predictions
        .into_iter()
        .map(|p| {
            if p.start >= p.end
                || p.end >= offsets.len()
                || text[offsets[p.start]..offsets[p.end]] != p.text
            {
                return Err(Error::inference(format!(
                    "{name} returned an invalid source span"
                )));
            }
            if !labels.contains(&p.label.as_str()) || !(0.0..=1.0).contains(&p.score) {
                return Err(Error::inference(format!(
                    "{name} returned an unexpected label or score"
                )));
            }
            Ok(Entity::with_provenance(
                p.text,
                EntityType::from_label(&p.label),
                p.start,
                p.end,
                p.score,
                Provenance::ml(name, p.score),
            ))
        })
        .collect()
}

impl crate::sealed::Sealed for Otter {}

impl ZeroShotNER for Otter {
    fn default_threshold(&self) -> f32 {
        match self.variant {
            OtterVariant::Cross => 0.5,
            OtterVariant::Bi => 0.2,
        }
    }

    fn default_types(&self) -> &[&'static str] {
        DEFAULT_LABELS
    }

    fn extract_with_types(
        &self,
        text: &str,
        labels: &[&str],
        threshold: f32,
    ) -> Result<Vec<Entity>> {
        if !(0.0..=1.0).contains(&threshold) {
            return Err(Error::invalid_input(
                "Otter threshold must be a finite probability in [0, 1]",
            ));
        }
        if text.trim().is_empty() || labels.is_empty() {
            return Ok(Vec::new());
        }
        if labels.iter().any(|s| s.trim().is_empty()) {
            return Err(Error::invalid_input("Otter entity labels must be nonempty"));
        }
        let request = serde_json::json!({"text": text, "labels": labels, "threshold": threshold});
        entities(text, labels, self.worker.request(&request)?, self.name())
    }

    fn extract_with_descriptions(
        &self,
        text: &str,
        labels: &[&str],
        threshold: f32,
    ) -> Result<Vec<Entity>> {
        self.extract_with_types(text, labels, threshold)
    }
}

impl Model for Otter {
    fn extract_entities(&self, text: &str, _: Option<Language>) -> Result<Vec<Entity>> {
        self.extract_with_types(text, DEFAULT_LABELS, self.default_threshold())
    }
    fn name(&self) -> &'static str {
        self.variant.name()
    }
    fn description(&self) -> &'static str {
        "Otter multilingual open-type span classifier (CPU, Transformers)"
    }
    fn supported_types(&self) -> Vec<EntityType> {
        DEFAULT_LABELS
            .iter()
            .map(|s| EntityType::from_label(s))
            .collect()
    }
    fn is_available(&self) -> bool {
        self.worker.is_available()
    }
    fn as_zero_shot(&self) -> Option<&dyn ZeroShotNER> {
        Some(self)
    }
    fn version(&self) -> String {
        match self.variant {
            OtterVariant::Cross => "8729188e4f5fc7948d0e9dfd7d7e6d36c2e7270d",
            OtterVariant::Bi => "53e10a09bc71a2e45980a7a257233a28305a5777",
        }
        .into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returned_offsets_must_select_the_exact_unicode_surface() {
        let prediction = |start, end| Prediction {
            start,
            end,
            label: "location".into(),
            text: "東京".into(),
            score: 0.9,
        };
        let text = "🙂 東京";
        assert_eq!(
            entities(text, &["location"], vec![prediction(2, 4)], "otter-bi").unwrap()[0].start(),
            2
        );
        assert!(entities(text, &["location"], vec![prediction(1, 3)], "otter-bi").is_err());
        assert!(entities(text, &["person"], vec![prediction(2, 4)], "otter-bi").is_err());
    }

    #[test]
    #[ignore = "requires both local Otter checkpoints and Python runtime"]
    fn published_checkpoints_find_a_unicode_name_and_reject_truncation() {
        for variant in [OtterVariant::Cross, OtterVariant::Bi] {
            let model = Otter::from_env(variant).unwrap();
            let text = "🙂 Angela Merkel besuchte Berlin.";
            let entities = model
                .extract_with_types(text, &["person", "location"], 0.5)
                .unwrap();
            assert!(entities
                .iter()
                .any(|e| e.text == "Angela Merkel" && e.start() == 2));
            assert!(model
                .extract_entities(&"Berlin ".repeat(1100), None)
                .is_err());
            let document = "Lynn Conway worked at IBM in Boston.";
            let defaults = model.extract_entities(document, None).unwrap();
            if variant == OtterVariant::Bi {
                assert!(defaults.iter().any(|e| e.text == "Boston"));
                let strict = model
                    .extract_with_types(document, DEFAULT_LABELS, 0.5)
                    .unwrap();
                assert!(!strict.iter().any(|e| e.text == "Boston"));
            }
        }
    }
}
