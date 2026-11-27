//! Corpus replay for the entity-only resolver. The input contains no gold links.
use super::*;
use serde::Deserialize;
use std::io::{BufRead, Write};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    id: String,
    text: String,
    mentions: Vec<(usize, usize)>,
}

#[test]
#[ignore = "set ANNO_GAP_TRACE_INPUT and ANNO_GAP_TRACE_OUTPUT to local JSONL paths"]
fn trace_supplied_ranking() {
    trace(MentionRankingCoref::new());
}

#[test]
#[ignore = "local corpus replay changing only the default character-distance cap"]
fn trace_stock_book_distance() {
    let config = MentionRankingConfig {
        max_distance: MentionRankingConfig::book_scale().max_distance,
        ..MentionRankingConfig::default()
    };
    trace(MentionRankingCoref::with_config(config));
}

fn trace(model: MentionRankingCoref) {
    let input = std::fs::File::open(std::env::var("ANNO_GAP_TRACE_INPUT").unwrap()).unwrap();
    let output = std::fs::File::create(std::env::var("ANNO_GAP_TRACE_OUTPUT").unwrap()).unwrap();
    let mut writer = std::io::BufWriter::new(output);
    assert!(matches!(
        model.config.clustering_strategy,
        ClusteringStrategy::LeftToRight
    ));
    assert!(!model.config.enable_global_proper_coref);
    for line in std::io::BufReader::new(input).lines() {
        let doc: Input = serde_json::from_str(&line.unwrap()).unwrap();
        let mut entities = Vec::new();
        let chars: Vec<_> = doc.text.chars().collect();
        for (start, end) in doc.mentions {
            assert!(start < end && end <= chars.len());
            let surface: String = chars[start..end].iter().collect();
            entities.push(Entity::new(
                surface,
                crate::EntityType::custom("mention", crate::EntityCategory::Misc),
                start,
                end,
                1.0,
            ));
        }
        entities.sort_by_key(|e| (e.start(), e.end()));
        let mentions = model.prepare_supplied_mentions(&entities);
        let decisions: Vec<_> = (0..mentions.len())
            .map(|i| {
                let mut candidates = Vec::new();
                // The entity-only resolver calls link_mentions with an empty text argument.
                let selected =
                    model.best_left_antecedent(i, &mentions, "", |j, score, threshold| {
                        candidates.push((j, score, threshold));
                    });
                serde_json::json!({"candidates": candidates, "selected": selected})
            })
            .collect();
        let members: Vec<_> = CoreferenceResolver::resolve(&model, &entities)
            .into_iter()
            .map(|e| (e.start(), e.end(), e.canonical_id.unwrap()))
            .collect();
        writeln!(
            writer,
            "{}",
            serde_json::json!({"id": doc.id, "members": members,
                 "decisions": decisions, "max_distance": model.config.max_distance})
        )
        .unwrap();
    }
    writer.flush().unwrap();
}
