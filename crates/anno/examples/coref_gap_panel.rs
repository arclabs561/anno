//! Stream labeled-corpus excerpts into one resolver without exposing gold links.
//! For supplied native mentions, `--trace-ranking` emits JSON decisions on stderr and
//! `--classify-mentions` sets the mention type using only its supplied surface.
#[cfg(feature = "onnx")]
use anno::backends::coref::fcoref::FCoref;
use anno::backends::coref::{
    mention_ranking::{MentionRankingConfig, MentionRankingCoref},
    MaverickCoref,
};
use anno::{CorefBackend, CoreferenceResolver, Entity, EntityCategory, EntityType, MentionType};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::error::Error;
use std::io::{self, BufRead};

type Span = (usize, usize);

struct RankingTrace;

impl log::Log for RankingTrace {
    fn enabled(&self, meta: &log::Metadata<'_>) -> bool {
        meta.target() == "anno::coref_candidates" && meta.level() == log::Level::Trace
    }

    fn log(&self, record: &log::Record<'_>) {
        if self.enabled(record.metadata()) {
            eprintln!("{}", record.args());
        }
    }

    fn flush(&self) {}
}

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
    classify_mentions: bool,
) -> Result<Vec<Vec<Span>>, Box<dyn Error>> {
    let mut entities: Vec<_> = mentions
        .iter()
        .map(|&(start, end)| {
            let surface: String = doc.text.chars().skip(start).take(end - start).collect();
            let mut entity = Entity::new(
                surface,
                EntityType::custom("mention", EntityCategory::Misc),
                start,
                end,
                1.0,
            );
            if classify_mentions {
                entity.mention_type = Some(MentionType::classify(&entity.text));
            }
            entity
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
    let mut args = std::env::args().skip(1);
    let requested = args.next().ok_or("expected backend name")?;
    let be_phrase = requested == "mention-ranking-copula";
    let mut trace_ranking = false;
    let mut classify_mentions = false;
    for flag in args {
        match flag.as_str() {
            "--trace-ranking" => trace_ranking = true,
            "--classify-mentions" => classify_mentions = true,
            _ => return Err(format!("unknown option {flag}").into()),
        }
    }
    if (trace_ranking || classify_mentions) && requested != "mention-ranking" {
        return Err("ranking options require the native mention-ranking backend".into());
    }
    if trace_ranking {
        log::set_logger(&RankingTrace).map_err(|_| "could not install the ranking trace logger")?;
        log::set_max_level(log::LevelFilter::Trace);
    }
    let native = MentionRankingCoref::with_config(MentionRankingConfig {
        enable_be_phrase_detection: be_phrase,
        ..Default::default()
    });
    let neural: Option<Box<dyn CorefBackend>> = match requested.as_str() {
        "mention-ranking" | "mention-ranking-copula" => None,
        #[cfg(not(feature = "onnx"))]
        "fcoref" => return Err("fcoref requires the onnx feature".into()),
        #[cfg(feature = "onnx")]
        "fcoref" => Some(Box::new(FCoref::from_path(&std::env::var(
            "FCOREF_MODEL_PATH",
        )?)?)),
        "maverick" => Some(Box::new(MaverickCoref::new(
            std::env::var("MAVERICK_PYTHON")?,
            std::env::var("MAVERICK_MODEL_PATH")?,
            false,
        )?)),
        _ => {
            return Err(
                "expected mention-ranking, mention-ranking-copula, fcoref or maverick".into(),
            )
        }
    };
    let model = neural.as_deref().unwrap_or(&native);
    let producer = if be_phrase {
        requested.as_str()
    } else {
        model.name()
    };
    for line in io::stdin().lock().lines() {
        let doc: Input = serde_json::from_str(&line?)?;
        if (trace_ranking || classify_mentions) && doc.mentions.is_none() {
            return Err(
                format!("{} requires supplied mentions for ranking options", doc.id).into(),
            );
        }
        if let Some(mentions) = &doc.mentions {
            if be_phrase {
                return Err("mention-ranking-copula requires the raw-text path".into());
            }
            let length = doc.text.chars().count();
            if mentions
                .iter()
                .any(|&(start, end)| start >= end || end > length)
            {
                return Err(format!("{} has an invalid supplied character range", doc.id).into());
            }
        }
        let spans = match (doc.mentions.as_ref(), neural.is_none()) {
            (Some(mentions), true) => {
                if trace_ranking {
                    eprintln!("{}", serde_json::json!({"id": doc.id}));
                }
                supplied_native(&native, &doc, mentions, classify_mentions)?
            }
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
                                "id": doc.id, "producer": producer,
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
            serde_json::json!({"id": doc.id, "producer": producer, "clusters": spans})
        );
    }
    Ok(())
}
