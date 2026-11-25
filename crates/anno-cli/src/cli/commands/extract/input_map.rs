use anno::SourceMap;
use serde::Deserialize;
use std::{fs, io::BufReader, path::Path};

#[derive(Deserialize)]
#[serde(untagged)]
enum SavedMap {
    Document {
        text: Option<String>,
        source_map: SourceMap,
    },
    Bare(SourceMap),
}

/// Validate the upstream analyzed edition, then compose ingestion onto that edition.
/// A renderer may have already changed the file before Anno's own cleanup runs.
pub(super) fn compose(
    path: &Path,
    ingest: &SourceMap,
    resolved: &str,
    input_file: Option<&str>,
) -> Result<SourceMap, String> {
    let file = fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let saved: SavedMap = serde_json::from_reader(BufReader::new(file))
        .map_err(|e| format!("invalid input source map at {}: {e}", path.display()))?;
    let (upstream, attached_text) = match saved {
        SavedMap::Document { text, source_map } => (source_map, text),
        SavedMap::Bare(map) => (map, None),
    };
    if let Some(text) = attached_text {
        upstream.validate_analyzed(&text)?;
    } else if ingest.source_sha256.as_ref() == Some(&ingest.analyzed_sha256) {
        upstream.validate_analyzed(resolved)?;
    } else if let Some(file) = input_file.filter(|f| *f != "-") {
        let text = fs::read_to_string(file)
            .map_err(|e| format!("reading the earlier analyzed UTF-8 file {file}: {e}"))?;
        upstream.validate_analyzed(&text)?;
    } else {
        return Err("supply the earlier analyzed text alongside source_map when ingesting a rendered stream".into());
    }
    ingest.compose(&upstream)
}
