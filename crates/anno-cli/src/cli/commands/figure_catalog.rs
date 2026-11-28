//! Read a local research catalog from a checkout or extracted dataset-figures artifact.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

#[derive(Deserialize)]
pub(crate) struct Catalog {
    pub schema: String,
    pub ci_survey_day: Option<String>,
    #[serde(default)]
    pub datasets: Vec<Dataset>,
    pub studies: Vec<Study>,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct Study {
    pub id: String,
    #[serde(default)]
    pub datasets: Vec<String>,
    #[serde(default)]
    pub guide: String,
    pub report: Value,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct Dataset {
    pub id: String,
    pub name: String,
    pub domain: String,
    pub measurement: String,
    pub coverage_action: String,
    pub declared_tasks: Vec<String>,
    pub declared_splits: Vec<String>,
    pub records: Vec<Record>,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct Record {
    pub status: String,
    pub source: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub task: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub backend: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub reported_backend: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub split: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub scope: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub examples: Option<u64>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub unit: String,
    #[serde(flatten)]
    pub details: BTreeMap<String, Value>,
}

impl Catalog {
    pub(crate) fn path(explicit: Option<&Path>) -> Result<PathBuf, String> {
        if let Some(path) = explicit {
            return Ok(path.to_owned());
        }
        let cwd = std::env::current_dir().map_err(|e| e.to_string())?;
        let exists = |p: &Path| p.try_exists().map_err(|e| format!("{}: {e}", p.display()));
        let bundle = cwd.join("figure-catalog.json");
        if exists(&bundle)? {
            return Ok(bundle);
        }
        for directory in cwd.ancestors() {
            let published = directory.join("docs/eval/figure-catalog.json");
            if exists(&published)? {
                return Ok(published);
            }
            if exists(&directory.join(".git"))? {
                break;
            }
        }
        Err("No local figure-catalog.json found. Run from an Anno checkout or the unpacked dataset-figures artifact, pass --catalog PATH, or set ANNO_FIGURE_CATALOG. Render a portable report directly with 'anno viz render INPUT'.".into())
    }

    pub(crate) fn load(explicit: Option<&Path>) -> Result<Self, String> {
        let path = Self::path(explicit)?;
        let source = path.display();
        let file = std::fs::File::open(&path).map_err(|e| format!("{source}: {e}"))?;
        let catalog: Self = serde_json::from_reader(std::io::BufReader::new(file))
            .map_err(|e| format!("{source}: {e}"))?;
        if catalog.schema != "anno.figure-catalog/v1" {
            return Err(format!(
                "unsupported figure catalog schema {:?} in {source}",
                catalog.schema
            ));
        }
        for (what, ids) in [
            (
                "dataset",
                catalog
                    .datasets
                    .iter()
                    .map(|d| d.id.as_str())
                    .collect::<Vec<_>>(),
            ),
            (
                "study",
                catalog.studies.iter().map(|s| s.id.as_str()).collect(),
            ),
        ] {
            let mut seen = BTreeSet::new();
            for id in ids {
                if id.trim().is_empty() || !seen.insert(id) {
                    return Err(format!("empty or duplicate {what} ID {id:?} in {source}"));
                }
            }
        }
        Ok(catalog)
    }
}
