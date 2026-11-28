//! Stage one JSON evaluation report per dataset occurrence, then publish the completed stream.

use super::super::eval::{comparison_report, MetricKind, PredictionSource};
use crate::cli::parser::ModelBackend;
use anno::core::grounded::EvalComparison;
use std::fs;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;
use tempfile::NamedTempFile;

pub(super) struct Reports {
    target: PathBuf,
    staged: BufWriter<NamedTempFile>,
}

fn parent(path: &Path) -> &Path {
    path.parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
}

// Resolve both existing aliases and output names whose final file has yet to be created.
fn resolved(path: &Path) -> io::Result<PathBuf> {
    match path.canonicalize() {
        Ok(path) => return Ok(path),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error),
    }
    if let Ok(link) = fs::read_link(path) {
        return resolved(&parent(path).join(link));
    }
    let name = path
        .file_name()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "expected a file name"))?;
    Ok(resolved(parent(path))?.join(name))
}

impl Reports {
    pub(super) fn new(target: &Path) -> Result<Self, String> {
        if target == Path::new("-") || target.is_dir() {
            return Err(format!(
                "--reports requires a JSONL file path, got {}",
                target.display()
            ));
        }
        let staged = NamedTempFile::new_in(parent(target))
            .map_err(|e| format!("Cannot stage --reports {}: {e}", target.display()))?;
        Ok(Self {
            target: target.to_owned(),
            staged: BufWriter::new(staged),
        })
    }

    pub(super) fn distinct_from(&self, other: &Path, description: &str) -> Result<(), String> {
        let same = match same_file::is_same_file(&self.target, other) {
            Ok(same) => Ok(same),
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                resolved(&self.target).and_then(|a| resolved(other).map(|b| a == b))
            }
            Err(e) => Err(e),
        }
        .map_err(|e| format!("Cannot compare --reports and {description} paths: {e}"))?;
        if same {
            return Err(format!(
                "--reports {} conflicts with {description} {}. Choose distinct output paths.",
                self.target.display(),
                other.display()
            ));
        }
        Ok(())
    }

    pub(super) fn record(
        &mut self,
        comparison: &EvalComparison,
        id: &str,
        model: ModelBackend,
        elapsed: Duration,
    ) -> Result<(), String> {
        let result = comparison_report(
            comparison,
            MetricKind::Entity,
            Some(id),
            &PredictionSource::Live { model, elapsed },
            None,
        )?;
        serde_json::to_writer(&mut self.staged, &result)
            .map_err(|e| format!("Failed to write --reports case {id}: {e}"))?;
        self.staged
            .write_all(b"\n")
            .map_err(|e| format!("Failed to write --reports case {id}: {e}"))
    }

    pub(super) fn publish(self) -> Result<(), String> {
        let staged = self
            .staged
            .into_inner()
            .map_err(|e| format!("Failed to flush --reports {}: {e}", self.target.display()))?;
        staged
            .persist(&self.target)
            .map_err(|e| format!("Failed to publish --reports {}: {e}", self.target.display()))?;
        Ok(())
    }
}
