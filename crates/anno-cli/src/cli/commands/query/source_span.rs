// SPDX-License-Identifier: MIT OR Apache-2.0
use clap::Args;
use serde_json::json;
use std::{fs, io, path::PathBuf};

#[derive(Args, Debug)]
pub struct SourceSpanArgs {
    /// Saved grounded document. Use - for stdin.
    input: PathBuf,
    /// Inclusive start in Unicode scalar values in the analyzed text.
    start: usize,
    /// Exclusive end in Unicode scalar values in the analyzed text.
    end: usize,
    /// Verify against these exact original bytes before returning any ranges.
    #[arg(long, value_name = "FILE")]
    original: Option<PathBuf>,
}

pub(super) fn run(args: SourceSpanArgs) -> Result<(), String> {
    let doc = super::read_grounded(&args.input)?;
    let chars = args.start..args.end;
    if chars.is_empty() || chars.end > doc.text().chars().count() {
        return Err("expected a nonempty character range within the analyzed text".into());
    }

    let original_sha256 = match doc.source_map() {
        Some(map) => map.source_sha256.as_deref(),
        None => doc.html_source().map(|html| html.source_sha256.as_str()),
    };
    if let Some(path) = &args.original {
        let original = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        if let Some(map) = doc.source_map() {
            map.verify(&original, doc.text())?;
        } else if let Some(legacy) = doc.html_source() {
            let html =
                std::str::from_utf8(&original).map_err(|_| "expected original UTF-8 HTML")?;
            legacy.verify(html, doc.text())?;
        } else {
            return Err("this document has no original source digest to verify".into());
        }
    }
    let fragments = doc.project_source_span(chars.clone())?;
    let result = json!({
        "document_id": doc.id(),
        "analyzed_chars": chars,
        "source_sha256": original_sha256,
        "fragments": fragments,
    });
    serde_json::to_writer(io::stdout().lock(), &result).map_err(|e| e.to_string())
}
