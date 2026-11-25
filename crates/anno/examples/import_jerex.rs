//! Ground one source-backed JEREX prediction produced by predict_jerex.py --mention-scores.
//! The source JSON array supplies the exact source_text and the sentence tokens used by the model.
use anno::core::grounded::ConceptRelation;
use anno::{ExtractionMethod, GroundedDocument, Location, Provenance, Signal, SignalId, SourceMap};
use serde::Deserialize;
use std::fs::File;
use std::io::{self, BufReader};

#[derive(Deserialize)]
struct Input {
    title: String,
    source_text: String,
    sents: Vec<Vec<String>>,
}

#[derive(Deserialize)]
struct Prediction {
    title: String,
    checkpoint_sha256: String,
    tokens: Vec<String>,
    mentions: Vec<Mention>,
    entities: Vec<Vertex>,
    relations: Vec<Edge>,
}

#[derive(Deserialize)]
struct Mention {
    start: usize,
    end: usize,
    confidence: f32,
}
#[derive(Deserialize)]
struct Vertex {
    #[serde(rename = "type")]
    label: String,
    mentions: Vec<usize>,
}
#[derive(Deserialize)]
struct Edge {
    head: usize,
    tail: usize,
    #[serde(rename = "type")]
    label: String,
}

fn token_offsets(text: &str, tokens: &[String]) -> Result<Vec<(usize, usize)>, String> {
    let mut rest = text;
    let mut position = 0;
    let mut spans = Vec::with_capacity(tokens.len());
    for token in tokens {
        let next = rest.trim_start();
        position += rest[..rest.len() - next.len()].chars().count();
        if token.is_empty() {
            return Err("empty JEREX token".into());
        }
        rest = next
            .strip_prefix(token)
            .ok_or_else(|| format!("token {token:?} differs from original text at {position}"))?;
        let end = position + token.chars().count();
        spans.push((position, end));
        position = end;
    }
    if !rest.trim().is_empty() {
        return Err("JEREX token list omits trailing source text".into());
    }
    Ok(spans)
}

fn import(
    input: Input,
    prediction: Prediction,
    mut doc: GroundedDocument,
) -> Result<GroundedDocument, String> {
    if doc.text() != input.source_text
        || input.title != prediction.title
        || !input.sents.iter().flatten().eq(prediction.tokens.iter())
    {
        return Err("prediction and source text edition differ".into());
    }
    let hash = prediction.checkpoint_sha256;
    if hash.len() != 64 || !hash.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("expected a SHA-256 digest of the producing checkpoint".into());
    }
    let mut producer =
        Provenance::from_producer("JEREX joint_multi_instance", ExtractionMethod::Neural);
    producer.model_version = Some(format!("sha256:{hash}").into());
    let offsets = token_offsets(doc.text(), &prediction.tokens)?;
    let mut labels = vec![None; prediction.mentions.len()];
    for vertex in &prediction.entities {
        for &index in &vertex.mentions {
            let label = labels
                .get_mut(index)
                .ok_or("entity references an absent mention")?;
            if label.replace(vertex.label.as_str()).is_some() {
                return Err("mention occurs more than once in the predicted partition".into());
            }
        }
    }
    // The supplied text is the original Unicode edition; no spacing is invented by joining tokens.
    doc.record_source_map(SourceMap::identity(&input.source_text))?;
    let byte_offsets: Vec<_> = input
        .source_text
        .char_indices()
        .map(|(i, _)| i)
        .chain(std::iter::once(input.source_text.len()))
        .collect();
    let mut signals = Vec::new();
    for (m, label) in prediction.mentions.iter().zip(labels) {
        if m.start >= m.end || m.end > offsets.len() || !(0.0..=1.0).contains(&m.confidence) {
            return Err("invalid JEREX mention span or probability".into());
        }
        let (start, end) = (offsets[m.start].0, offsets[m.end - 1].1);
        let surface = &input.source_text[byte_offsets[start]..byte_offsets[end]];
        let signal = Signal::new(
            SignalId::ZERO,
            Location::text(start, end),
            surface,
            label.unwrap_or("mention"),
            m.confidence,
        )
        .with_provenance(producer.clone());
        signals.push(
            doc.add_signal_validated(signal)
                .map_err(|e| e.to_string())?,
        );
    }
    let mut endpoints = Vec::new();
    for (i, vertex) in prediction.entities.iter().enumerate() {
        let members: Vec<_> = vertex
            .mentions
            .iter()
            .map(|&index| signals[index])
            .collect();
        let first = members.first().ok_or("JEREX entity has no members")?;
        let surface = doc
            .get_signal(*first)
            .ok_or("missing inserted signal")?
            .surface
            .clone();
        doc.create_track_from_signals(surface, &members)
            .ok_or("invalid JEREX cluster")?;
        endpoints.push(doc.concept_endpoint(i.to_string(), Some(&members))?);
    }
    for edge in prediction.relations {
        let endpoint = |index: usize| {
            endpoints
                .get(index)
                .cloned()
                .ok_or("relation references absent entity")
        };
        doc.record_concept_relation(ConceptRelation {
            head: endpoint(edge.head)?,
            tail: endpoint(edge.tail)?,
            relation_type: edge.label,
            confidence: None,
            provenance: Some(producer.clone()),
        })?;
    }
    Ok(doc)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let read = |path| File::open(path).map(BufReader::new);
    let inputs: Vec<Input> =
        serde_json::from_reader(read(args.next().ok_or("expected source array")?)?)?;
    let predictions: Vec<Prediction> = serde_json::from_reader(read(
        args.next().ok_or("expected scored prediction array")?,
    )?)?;
    if args.next().is_some() || inputs.len() != predictions.len() {
        return Err("expected a source and a prediction array with equal lengths".into());
    }
    let mut output = io::BufWriter::new(io::stdout().lock());
    use std::io::Write;
    for (source, prediction) in inputs.into_iter().zip(predictions) {
        let doc = GroundedDocument::new(&source.title, &source.source_text);
        serde_json::to_writer(&mut output, &import(source, prediction, doc)?)?;
        writeln!(output)?;
    }
    output.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    const SOURCE: &str = include_str!("../../../scripts/qa/joint_baseline/jerex_input.json");
    const PREDICTION: &str =
        include_str!("../../../scripts/qa/joint_baseline/jerex_prediction.json");
    const LEGACY: &str = include_str!("../../../scripts/qa/joint_baseline/legacy_grounded.json");

    #[test]
    fn trained_joint_baseline_preserves_source_evidence_after_track_reassignment() {
        let mut inputs: Vec<Input> = serde_json::from_str(SOURCE).unwrap();
        let input = inputs.remove(0);
        let mut rows: Vec<Prediction> = serde_json::from_str(PREDICTION).unwrap();
        let prediction = rows.remove(0);
        let expected_mentions = prediction.mentions.len();
        let expected_clusters = prediction.entities.len();
        let expected_edges = prediction.relations.len();
        let legacy: GroundedDocument = serde_json::from_str(LEGACY).unwrap();
        let old_signals = serde_json::to_value(legacy.signals()).unwrap();
        let old_count = legacy.signals().len();
        assert!(old_count > 0);
        assert!(legacy.project_source_span(0..5).unwrap().is_none());
        let doc = import(input, prediction, legacy).unwrap();
        let mut doc: GroundedDocument =
            serde_json::from_slice(&serde_json::to_vec(&doc).unwrap()).unwrap();
        assert_eq!(doc.signals().len(), old_count + expected_mentions);
        assert_eq!(doc.tracks().count(), expected_clusters);
        assert_eq!(doc.recorded_concept_relations().len(), expected_edges);
        assert!(doc.recorded_relations().is_empty());
        assert_eq!(
            serde_json::to_value(&doc.signals()[..old_count]).unwrap(),
            old_signals
        );
        let evidence = serde_json::to_value(doc.recorded_concept_relations()).unwrap();
        let mut selected = None;
        for (index, edge) in doc.project_recorded_concept_relations().enumerate() {
            let edge = edge.unwrap();
            let head = edge.head_track.unwrap().id;
            let tail = edge.tail_track.unwrap().id;
            for endpoint in [&edge.relation.head, &edge.relation.tail] {
                for member in endpoint.members.as_ref().unwrap() {
                    let signal = doc.get_signal(member.signal_id).unwrap();
                    let (start, end) = signal.text_offsets().unwrap();
                    let run = doc.project_source_span(start..end).unwrap().unwrap();
                    assert_eq!(run.len(), 1);
                    assert_eq!(
                        &doc.text()[run[0].source_bytes.clone().unwrap()],
                        signal.surface()
                    );
                    assert!(signal.provenance.as_ref().unwrap().model_version.is_some());
                }
            }
            if head != tail && selected.is_none() {
                if let Some(other) = doc
                    .tracks()
                    .map(|t| t.id)
                    .find(|id| *id != head && *id != tail)
                {
                    selected = Some((index, head, tail, other));
                }
            }
        }
        let (index, head, tail, other) =
            selected.expect("fixture has a relation and a third independent cluster");
        let merged = doc.merge_tracks(&[head, other]).unwrap();
        assert_ne!(head, merged);
        let reloaded: GroundedDocument =
            serde_json::from_slice(&serde_json::to_vec(&doc).unwrap()).unwrap();
        let edge = reloaded
            .project_recorded_concept_relations()
            .nth(index)
            .unwrap()
            .unwrap();
        assert!(edge.head_track.is_none());
        assert_eq!(edge.tail_track.unwrap().id, tail);
        assert_eq!(
            serde_json::to_value(reloaded.recorded_concept_relations()).unwrap(),
            evidence
        );
        assert_eq!(
            serde_json::to_value(&reloaded.signals()[..old_count]).unwrap(),
            old_signals
        );
    }
}
