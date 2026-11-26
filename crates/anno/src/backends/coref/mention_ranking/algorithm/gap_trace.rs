//! Optional corpus replay driven by scripts/qa/native_gap_loss.py.
use super::*;
use serde::{Deserialize, Serialize};
use std::io::{BufRead, Write};

#[derive(Deserialize)]
struct Document {
    id: String,
    text: String,
}

#[derive(Serialize)]
struct Decision {
    max_antecedents: usize,
    selected: Option<usize>,
    candidates: Vec<(usize, f64, f64)>,
}

#[test]
#[ignore = "requires ANNO_GAP_TRACE_INPUT and ANNO_GAP_TRACE_OUTPUT"]
fn trace_native_gap() {
    let reader = std::io::BufReader::new(
        std::fs::File::open(std::env::var("ANNO_GAP_TRACE_INPUT").unwrap()).unwrap(),
    );
    let mut writer = std::io::BufWriter::new(
        std::fs::File::create(std::env::var("ANNO_GAP_TRACE_OUTPUT").unwrap()).unwrap(),
    );
    let model = MentionRankingCoref::new();
    assert!(matches!(
        model.config.clustering_strategy,
        ClusteringStrategy::LeftToRight
    ));
    assert!(!model.config.enable_global_proper_coref);
    let span = |m: &RankedMention| (m.start, m.end);
    for line in reader.lines() {
        let doc: Document = serde_json::from_str(&line.unwrap()).unwrap();
        let proposals = model.propose_mentions(&doc.text).unwrap();
        let proposed: Vec<_> = proposals.iter().map(span).collect();
        let mut mentions = MentionRankingCoref::deduplicate_mentions(proposals);
        mentions.sort_by_key(span);
        for mention in &mut mentions {
            model.extract_features(mention);
        }
        let decisions: Vec<_> = (0..mentions.len())
            .map(|i| {
                let mut candidates = Vec::new();
                let selected = model.best_left_antecedent(i, &mentions, &doc.text, |j, s, t| {
                    candidates.push((j, s, t))
                });
                Decision {
                    max_antecedents: model
                        .config
                        .max_antecedents_for_type(mentions[i].mention_type),
                    selected,
                    candidates,
                }
            })
            .collect();
        let retained: Vec<_> = mentions.iter().map(span).collect();
        let clusters: Vec<Vec<_>> = model
            .resolve(&doc.text)
            .unwrap()
            .iter()
            .map(|c| c.mentions.iter().map(span).collect())
            .collect();
        let record = serde_json::json!({"id":doc.id,"max_distance":model.config.max_distance,"proposed":proposed,"retained":retained,"decisions":decisions,"clusters":clusters});
        writeln!(writer, "{record}").unwrap();
    }
    writer.flush().unwrap();
}
