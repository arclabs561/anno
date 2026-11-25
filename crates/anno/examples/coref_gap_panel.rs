//! Stream GAP excerpts into a single explicitly selected resolver. Used by scripts/qa/coref_gap.py.
use anno::backends::coref::{fcoref::FCoref, mention_ranking::MentionRankingCoref, MaverickCoref};
use anno::{CorefBackend, CoreferenceResolver, Entity, EntityCategory, EntityType};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::error::Error;
use std::io::{self, BufRead};

type Span = (usize, usize);

#[derive(Deserialize)]
struct Input {
    id: String,
    text: String,
    mentions: Option<Vec<Span>>,
}

fn supplied_native(model: &MentionRankingCoref, doc: &Input, mentions: &[Span]) -> Vec<Vec<Span>> {
    let mut entities: Vec<_> = mentions
        .iter()
        .map(|&(start, end)| {
            let surface: String = doc.text.chars().skip(start).take(end - start).collect();
            Entity::new(
                surface,
                EntityType::custom("mention", EntityCategory::Misc),
                start,
                end,
                1.0,
            )
        })
        .collect();
    entities.sort_by_key(|e| (e.start(), e.end()));
    let mut groups = BTreeMap::new();
    for e in CoreferenceResolver::resolve(model, &entities) {
        groups
            .entry(e.canonical_id.expect("ranker assigns an ID"))
            .or_insert_with(Vec::new)
            .push((e.start(), e.end()));
    }
    groups.into_values().collect()
}

fn main() -> Result<(), Box<dyn Error>> {
    let requested = std::env::args().nth(1).ok_or("expected backend name")?;
    let native = MentionRankingCoref::new();
    let neural: Option<Box<dyn CorefBackend>> = match requested.as_str() {
        "mention-ranking" => None,
        "fcoref" => Some(Box::new(FCoref::from_path(&std::env::var(
            "FCOREF_MODEL_PATH",
        )?)?)),
        "maverick" => Some(Box::new(MaverickCoref::new(
            std::env::var("MAVERICK_PYTHON")?,
            std::env::var("MAVERICK_MODEL_PATH")?,
            false,
        )?)),
        _ => return Err("expected mention-ranking, fcoref or maverick".into()),
    };
    let model = neural.as_deref().unwrap_or(&native);
    for line in io::stdin().lock().lines() {
        let doc: Input = serde_json::from_str(&line?)?;
        let spans = match (doc.mentions.as_ref(), neural.is_none()) {
            (Some(mentions), true) => supplied_native(&native, &doc, mentions),
            (mentions, _) => {
                let clusters = match mentions {
                    Some(mentions) => model.resolve_mentions(&doc.text, mentions),
                    None => model.resolve(&doc.text),
                }
                .map_err(|e| format!("{} on {}: {e}", model.name(), doc.id))?;
                clusters.into_iter().map(|c| c.spans).collect()
            }
        };
        println!(
            "{}",
            serde_json::json!({"id": doc.id, "producer": model.name(), "clusters": spans})
        );
    }
    Ok(())
}
