//! Stream labeled-corpus excerpts into one resolver without exposing gold links.
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

fn supplied_native(
    model: &MentionRankingCoref,
    doc: &Input,
    mentions: &[Span],
) -> Result<Vec<Vec<Span>>, Box<dyn Error>> {
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
            .entry(
                e.canonical_id
                    .ok_or("native ranker returned an unassigned mention")?,
            )
            .or_insert_with(Vec::new)
            .push((e.start(), e.end()));
    }
    Ok(groups.into_values().collect())
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
        if let Some(mentions) = &doc.mentions {
            let length = doc.text.chars().count();
            if mentions
                .iter()
                .any(|&(start, end)| start >= end || end > length)
            {
                return Err(format!("{} has an invalid supplied character range", doc.id).into());
            }
        }
        let spans = match (doc.mentions.as_ref(), neural.is_none()) {
            (Some(mentions), true) => supplied_native(&native, &doc, mentions)?,
            (mentions, _) => {
                let outcome = match mentions {
                    Some(mentions) => model.resolve_mentions(&doc.text, mentions),
                    None => model.resolve(&doc.text),
                };
                let clusters = match outcome {
                    Ok(clusters) => clusters,
                    Err(e)
                        if mentions.is_some()
                            && requested == "maverick"
                            && e.to_string().contains("must cover complete tokens") =>
                    {
                        println!(
                            "{}",
                            serde_json::json!({
                                "id": doc.id, "producer": model.name(),
                                "unsupported": "supplied_span_splits_token"
                            })
                        );
                        continue;
                    }
                    Err(e) => return Err(format!("{} on {}: {e}", model.name(), doc.id).into()),
                };
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
