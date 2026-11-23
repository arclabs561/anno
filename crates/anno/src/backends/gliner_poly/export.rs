//! Read the lightweight inventory emitted by the GLiNER exporter without opening ONNX sessions.
use crate::{Error, Result};
use serde::Deserialize;
use std::path::{Component, Path, PathBuf};

pub(super) const FILE: &str = "anno_poly_export.json";

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum LabelInput {
    TokenIds,
    Embeddings,
}

#[derive(Deserialize)]
pub(super) struct Export {
    schema_version: u32,
    label_input: LabelInput,
    external_data: Vec<String>,
}

impl Export {
    pub(super) fn split(&self) -> bool {
        matches!(self.label_input, LabelInput::Embeddings)
    }

    pub(super) fn read(dir: &Path) -> Result<Option<Self>> {
        let path = dir.join(FILE);
        let file = match std::fs::File::open(&path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(Error::Retrieval(format!("Read {FILE}: {error}"))),
        };
        use std::io::Read;
        let mut bytes = Vec::new();
        file.take(65537)
            .read_to_end(&mut bytes)
            .map_err(|e| Error::Retrieval(format!("Read {FILE}: {e}")))?;
        if bytes.len() > 65536 {
            return Err(Error::Retrieval(format!("{FILE} exceeds 64 KiB")));
        }
        let export: Self = serde_json::from_slice(&bytes)
            .map_err(|e| Error::Retrieval(format!("Invalid {FILE}: {e}")))?;
        if export.schema_version != 1 {
            return Err(Error::Retrieval(format!("Unknown {FILE} schema version")));
        }
        let mut required = vec!["model.onnx", "tokenizer.json", "label_tokenizer.json"];
        if export.split() {
            required.push("label_encoder.onnx");
        }
        required.extend(export.external_data.iter().map(String::as_str));
        for name in required {
            let relative = Path::new(name);
            if name.is_empty()
                || name.contains(['\\', ':'])
                || !relative
                    .components()
                    .all(|c| matches!(c, Component::Normal(_)))
            {
                return Err(Error::Retrieval(format!(
                    "Invalid relative asset path in {FILE}"
                )));
            }
            if !dir.join(relative).is_file() {
                return Err(Error::Retrieval(format!(
                    "Missing GLiNER Poly export asset: {name}"
                )));
            }
        }
        Ok(Some(export))
    }

    pub(super) fn ready(dir: &Path) -> bool {
        matches!(Self::read(dir), Ok(Some(_)))
    }

    pub(super) fn hub_dir(cache: &hf_hub::Cache, model: &str) -> Option<PathBuf> {
        let repo = cache.model(model.to_owned());
        [FILE.to_owned(), format!("onnx/{FILE}")]
            .into_iter()
            .find_map(|name| {
                let receipt = repo.get(&name)?;
                // Keep the snapshot directory so data files resolve beside the graph.
                let root = receipt.parent()?;
                Self::ready(root).then(|| root.to_owned())
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::gliner_poly::GLiNERPoly;
    use std::fs;

    #[test]
    fn fused_hub_export_requires_its_declared_sidecar_and_no_label_graph() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = hf_hub::Cache::new(tmp.path().to_owned());
        let id = crate::models::GLINER_POLY;
        let hf = cache.model(id.to_string());
        let dir = hf.pointer_path("fixtures").join("onnx");
        fs::create_dir_all(&dir).unwrap();
        hf.create_ref("fixtures").unwrap();
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/bughunt7_data");
        let assets = fixtures.join("gliner_poly_split");
        fs::copy(assets.join("fused.onnx"), dir.join("model.onnx")).unwrap();
        let tok =
            fixtures.join("hf_home/hub/models--bh7--gliner-span/snapshots/fixture/tokenizer.json");
        for file in ["tokenizer.json", "label_tokenizer.json"] {
            fs::copy(&tok, dir.join(file)).unwrap();
        }
        fs::write(
            dir.join(FILE),
            r#"{"schema_version":1,"label_input":"token_ids","external_data":["fused.weights"]}"#,
        )
        .unwrap();
        assert!(Export::hub_dir(&cache, id).is_none());
        fs::copy(assets.join("fused.weights"), dir.join("fused.weights")).unwrap();
        assert_eq!(Export::hub_dir(&cache, id), Some(dir.clone()));
        let model = GLiNERPoly::from_dir(&dir).unwrap();
        let out = model
            .extract("À John   Smith", &["location", "person"], 0.5)
            .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].text, "John   Smith");
        assert_eq!(out[0].entity_type, crate::EntityType::Person);

        // A pre-inventory bi-encoder config still describes the source tokenizer.
        fs::remove_file(dir.join(FILE)).unwrap();
        fs::write(
            dir.join("gliner_config.json"),
            r#"{"labels_encoder":"test/labels"}"#,
        )
        .unwrap();
        assert!(Export::hub_dir(&cache, id).is_none());
        let explicit = GLiNERPoly::from_dir(&dir).unwrap();
        assert!(explicit.source_only_text);
        assert_eq!(
            explicit
                .extract("À John   Smith", &["person"], 0.5)
                .unwrap()[0]
                .text,
            "John   Smith"
        );
    }

    #[test]
    fn inventory_rejects_escaping_paths_and_malformed_contents() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("export");
        fs::create_dir(&dir).unwrap();
        for file in ["model.onnx", "label_tokenizer.json", "tokenizer.json"] {
            fs::write(dir.join(file), []).unwrap();
        }
        let outside = tmp.path().join("outside");
        fs::write(&outside, []).unwrap();
        for location in [
            outside.to_str().unwrap(),
            "../outside",
            r"C:\assets\weights",
        ] {
            let data = serde_json::json!({"schema_version":1,"label_input":"token_ids","external_data":[location]});
            fs::write(dir.join(FILE), data.to_string()).unwrap();
            assert!(Export::read(&dir).is_err());
        }
        fs::write(dir.join(FILE), "{invalid").unwrap();
        assert!(Export::read(&dir).is_err());
    }
}
