//! Evaluate heuristic cross-document cluster merging on supplied ECB+ XML mention spans.
//! Run with `just cdcr-eval`, or `just cdcr-eval --sentence-index PATH` for the
//! published selected-sentence annotations. ANNO_CDCR_MAX_DOCS limits sorted documents.

use anno_eval::eval::cdcr::{CrossDocCluster, Document};
use anno_eval::eval::cluster_encoder::{CosineMergeScorer, HeuristicClusterEncoder};
use anno_eval::eval::coref_loader::EcbPlusDocument;
use anno_eval::eval::cross_context_eval::{
    evaluate_cross_document_with_links, CrossContextEvalConfig, Topic,
};
use anno_eval::eval::loader::DatasetLoader;
use anno_eval::{Error, Result};
use std::collections::BTreeMap;
use std::path::PathBuf;

fn main() -> Result<()> {
    if let Ok(filter) = std::env::var("ANNO_CDCR_DATASETS") {
        validate_datasets(&filter)?;
    }
    let max_docs = std::env::var("ANNO_CDCR_MAX_DOCS")
        .ok()
        .map(|value| {
            value.parse::<usize>().map_err(|_| {
                Error::InvalidInput("ANNO_CDCR_MAX_DOCS must be a document count".into())
            })
        })
        .transpose()?;

    let mut args = std::env::args_os().skip(1);
    let sentence_index = match (args.next(), args.next(), args.next()) {
        (None, None, None) => None,
        (Some(flag), Some(path), None) if flag == "--sentence-index" => Some(PathBuf::from(path)),
        _ => {
            return Err(Error::InvalidInput(
                "Usage: cdcr_eval [--sentence-index PATH]".into(),
            ))
        }
    };
    let loader = DatasetLoader::new()?;
    let (mut docs, coverage) = if let Some(path) = sentence_index {
        let index = std::fs::read_to_string(&path).map_err(|error| {
            Error::InvalidInput(format!(
                "Failed to read ECB+ selected-sentence index {}: {error}",
                path.display()
            ))
        })?;
        (
            loader.load_or_download_ecb_plus_selected_gold(&index)?,
            "fully anchored annotations in the supplied sentence index",
        )
    } else {
        (
            loader.load_or_download_ecb_plus_gold()?,
            "all anchored XML annotations",
        )
    };
    docs.sort_by(|a, b| (&a.topic, &a.doc_id).cmp(&(&b.topic, &b.doc_id)));
    if let Some(max) = max_docs {
        docs.truncate(max);
    }
    let topics = ecb_topics(docs)?;
    if !topics.iter().any(|topic| topic.documents.len() >= 2) {
        return Err(Error::InvalidInput(
            "Cross-document evaluation requires at least two documents in one ECB+ topic".into(),
        ));
    }

    let documents: usize = topics.iter().map(|t| t.documents.len()).sum();
    let mentions: usize = topics
        .iter()
        .flat_map(|t| &t.documents)
        .map(|d| d.entities.len())
        .sum();
    println!("## ECB+ cross-document merging on annotated XML mentions\n");
    println!("UnifiedCrossContextResolver with heuristic cluster embeddings and cosine scoring. The input covers {coverage}, including singletons.\n");
    println!("Documents {documents} · Topics {} · Supplied mentions {mentions}. Link counts cover mention pairs from different documents.\n", topics.len());
    println!("| Local seeds | Threshold | CoNLL F1 | MUC F1 | B³ F1 | CEAF-e F1 | Gold clusters | Pred clusters | Correct links | Gold links | Pred links |");
    println!("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");

    for (seed_label, gold_clusters) in [("Singletons", false), ("Gold within document", true)] {
        for threshold in [0.3, 0.5, 0.7] {
            let (scores, links) = evaluate_cross_document_with_links(
                &topics,
                HeuristicClusterEncoder::new(64),
                CosineMergeScorer::new(),
                &CrossContextEvalConfig {
                    use_gold_mentions: true,
                    use_gold_clusters: gold_clusters,
                    merge_threshold: threshold,
                    ..Default::default()
                },
            )?;
            println!(
                "| {seed_label} | {threshold:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {} | {} | {} | {} | {} |",
                scores.conll_f1 * 100.0, scores.muc.f1 * 100.0,
                scores.b_cubed.f1 * 100.0, scores.ceaf_e.f1 * 100.0,
                scores.num_gold_clusters, scores.num_pred_clusters, links.correct, links.gold, links.predicted
            );
        }
    }
    Ok(())
}

fn validate_datasets(filter: &str) -> Result<()> {
    for name in filter.split(',').map(str::trim) {
        if !name.eq_ignore_ascii_case("ECBPlus") && !name.eq_ignore_ascii_case("ECB+") {
            return Err(Error::InvalidInput(format!(
                "No scoped cross-document gold reader for dataset {name:?}; this benchmark supports ECB+"
            )));
        }
    }
    Ok(())
}

fn ecb_topics(docs: Vec<EcbPlusDocument>) -> Result<Vec<Topic>> {
    let mut grouped = BTreeMap::new();
    for doc in docs {
        let topic = doc.topic.as_ref().ok_or_else(|| {
            Error::InvalidInput(format!("Missing ECB+ topic for document {}", doc.doc_id))
        })?;
        grouped
            .entry(topic.clone())
            .or_insert_with(Vec::new)
            .push(doc);
    }

    let mut topics = Vec::new();
    for (name, docs) in grouped {
        let mut topic = Topic::new(&name);
        let mut cross_document_ids = BTreeMap::new();
        for doc in docs {
            let mut entities = Vec::new();
            for scoped in doc.chains {
                // An unlabelled/local chain always gets its own cluster, including a singleton.
                let next = topic.gold_clusters.len();
                let group = match scoped.cross_document_id {
                    Some(id) => *cross_document_ids.entry(id).or_insert(next),
                    None => next,
                };
                if group == next {
                    topic.add_gold_cluster(CrossDocCluster::new(group as u64, ""));
                }
                for mention in scoped.chain.mentions {
                    topic.gold_clusters[group].add_mention(&doc.doc_id, entities.len());
                    let ty = anno::EntityType::from_label(
                        mention.entity_type.as_deref().unwrap_or("mention"),
                    );
                    entities.push(anno::Entity::new(
                        mention.text,
                        ty,
                        mention.start,
                        mention.end,
                        1.0,
                    ));
                }
            }
            topic.add_document(Document::new(&doc.doc_id, &doc.text).with_entities(entities));
        }
        topics.push(topic);
    }
    Ok(topics)
}

#[cfg(test)]
mod tests {
    use super::*;
    use anno_eval::eval::coref::{CorefChain, Mention};
    use anno_eval::eval::coref_loader::EcbPlusChain;

    fn example_doc(topic: &str, id: &str, labels: &[Option<&str>]) -> EcbPlusDocument {
        EcbPlusDocument {
            topic: Some(topic.into()),
            doc_id: id.into(),
            text: "Nia ".repeat(labels.len()),
            chains: labels
                .iter()
                .enumerate()
                .map(|(i, label)| EcbPlusChain {
                    cross_document_id: label.map(str::to_owned),
                    chain: CorefChain::new(vec![Mention::new("Nia", 4 * i, 4 * i + 3)]),
                })
                .collect(),
        }
    }

    #[test]
    fn gold_uses_source_scope_and_keeps_local_singletons() {
        let topics = ecb_topics(vec![
            example_doc("alpha", "a", &[None, Some("HUM_A"), Some("HUM_B")]),
            example_doc("alpha", "b", &[None, Some("HUM_B"), Some("HUM_C")]),
            example_doc("beta", "a", &[None, Some("HUM_B")]),
        ])
        .unwrap();
        assert_eq!(topics.len(), 2);
        assert_eq!(topics[0].gold_clusters.len(), 5);
        let pair = topics[0]
            .gold_clusters
            .iter()
            .find(|c| c.mentions.len() == 2)
            .unwrap();
        assert_eq!(pair.mentions, [("a".to_string(), 2), ("b".to_string(), 1)]);
        assert_eq!(topics[1].gold_clusters.len(), 2);
        assert!(validate_datasets("ECBPlus,GAP").is_err());
    }

    #[test]
    fn homonyms_with_distinct_gold_ids_do_not_earn_a_perfect_score() {
        let topics = ecb_topics(vec![
            example_doc("alpha", "a", &[Some("HUM_A")]),
            example_doc("alpha", "b", &[Some("HUM_B")]),
        ])
        .unwrap();
        let (scores, links) = evaluate_cross_document_with_links(
            &topics,
            HeuristicClusterEncoder::new(64),
            CosineMergeScorer::new(),
            &CrossContextEvalConfig::oracle(),
        )
        .unwrap();
        assert_eq!(scores.num_gold_clusters, 2);
        assert_eq!(scores.num_pred_clusters, 1);
        assert_eq!(scores.b_cubed.precision, 0.5);
        assert_eq!((links.correct, links.predicted, links.gold), (0, 1, 0));
    }
}
