//! Replay four-layer GENIA gold through Anno's entity-list and HTML exporters.
//! `scripts/plot_genia_nesting.py --native-export` supplies one IOB2 split on stdin.

use anno::{export, Entity, EntityCategory, EntityType};
use anno_eval::eval::loader::{AnnotatedSentence, AnnotatedToken, DatasetId};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::error::Error;
use std::io::{self, BufRead, Write};

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn export_sentence(
    layers: &[AnnotatedSentence; 4],
    number: usize,
    out: &mut impl Write,
) -> io::Result<()> {
    let text = layers[0].text();
    // Keep the publisher's original labels (including case) in both exports.
    let entities: Vec<_> = layers
        .iter()
        .flat_map(AnnotatedSentence::entities)
        .map(|gold| {
            Entity::new(
                gold.text,
                EntityType::custom(gold.original_label, EntityCategory::Misc),
                gold.start,
                gold.end,
                1.0,
            )
        })
        .collect();
    writeln!(
        out,
        "{}",
        json!({
            "sentence": number,
            "html": export::to_html(&text, &entities),
            "jsonl": export::to_jsonl(&entities, &number.to_string(), false),
        })
    )
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut out = io::BufWriter::new(io::stdout().lock());
    writeln!(
        out,
        "{}",
        json!({
            "anno_eval_version": env!("CARGO_PKG_VERSION"),
            "export_rs_sha256": sha256(include_bytes!("../../anno/src/export.rs")),
            "loader_rs_sha256": sha256(include_bytes!("../src/eval/loader.rs")),
            "example_rs_sha256": sha256(include_bytes!("genia_export.rs")),
        })
    )?;
    let mut layers: [AnnotatedSentence; 4] = std::array::from_fn(|_| AnnotatedSentence {
        tokens: Vec::new(),
        source_dataset: DatasetId::GENIANested,
    });
    let mut number = 1;
    for (line_index, line) in io::stdin().lock().lines().enumerate() {
        let line = line?;
        let mut fields = line.split_whitespace();
        let Some(token) = fields.next() else {
            if !layers[0].tokens.is_empty() {
                export_sentence(&layers, number, &mut out)?;
                number += 1;
                for layer in &mut layers {
                    layer.tokens.clear();
                }
            }
            continue;
        };
        for layer in &mut layers {
            let tag = fields
                .next()
                .ok_or_else(|| format!("missing GENIA tag on line {}", line_index + 1))?;
            layer.tokens.push(AnnotatedToken {
                text: token.into(),
                ner_tag: tag.into(),
            });
        }
        if fields.next().is_some() {
            return Err(format!("extra GENIA column on line {}", line_index + 1).into());
        }
    }
    if !layers[0].tokens.is_empty() {
        export_sentence(&layers, number, &mut out)?;
    }
    out.flush()?;
    Ok(())
}
