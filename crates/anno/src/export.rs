//! Export entity extraction results to annotation and interchange formats.
//!
//! All functions are pure: they take entities (and optionally relations) and
//! return formatted strings. No file I/O or CLI dependencies.
//!
//! # Formats
//!
//! | Format | Function | Description |
//! |--------|----------|-------------|
//! | brat | [`to_brat`] | Standoff annotation (`.ann`) for brat |
//! | CoNLL | [`to_conll`] | BIO-tagged tokens (one per line) |
//! | JSONL | [`to_jsonl`] | One JSON object per entity |
//! | N-Triples | [`to_ntriples`] | RDF triples |
//! | JSON-LD | [`to_jsonld`] | Linked data with `@context` |
//! | Graph CSV | [`to_graph_csv`] | Node + edge tables for graph DB import |
//!
//! # Example
//!
//! ```
//! use anno::{Model, StackedNER};
//! use anno::export;
//!
//! let m = StackedNER::default();
//! let ents = m.extract_entities("Marie Curie won the Nobel Prize.", None)?;
//! let brat = export::to_brat("Marie Curie won the Nobel Prize.", &ents, false);
//! let conll = export::to_conll("Marie Curie won the Nobel Prize.", &ents);
//! # Ok::<(), anno::Error>(())
//! ```

use anno_core::{Entity, Relation};

/// Export entities in brat standoff format (`.ann`).
///
/// Each entity becomes a `T`-annotation line. When `include_confidence` is true,
/// `A`-annotation lines are added with confidence scores.
///
/// Brat requires UTF-8 byte offsets. Entity start/end are character offsets, so
/// `text` is needed to convert them.
pub fn to_brat(text: &str, entities: &[Entity], include_confidence: bool) -> String {
    let mut lines = Vec::new();
    for (idx, entity) in entities.iter().enumerate() {
        let byte_start: usize = text
            .chars()
            .take(entity.start())
            .map(|c| c.len_utf8())
            .sum();
        let byte_end: usize = text.chars().take(entity.end()).map(|c| c.len_utf8()).sum();
        let tid = format!("T{}", idx + 1);
        let line = format!(
            "{}\t{} {} {}\t{}",
            tid,
            entity.entity_type.as_label(),
            byte_start,
            byte_end,
            entity.text
        );
        if include_confidence {
            let aid = format!("A{}", idx + 1);
            lines.push(line);
            lines.push(format!(
                "{}\tConfidence {} {:.2}",
                aid, tid, entity.confidence
            ));
        } else {
            lines.push(line);
        }
    }
    lines.join("\n")
}

/// Export entities in CoNLL BIO-tagged format.
///
/// Produces one token per line with TAB-separated BIO tags. Trailing
/// punctuation is split into a separate `O` token unless the punctuation
/// falls inside an entity span.
pub fn to_conll(text: &str, entities: &[Entity]) -> String {
    let mut lines = Vec::new();
    let mut byte_idx = 0;
    for word in text.split_whitespace() {
        // Find the byte position of this word in the remaining text
        let byte_start = text[byte_idx..]
            .find(word)
            .map(|i| byte_idx + i)
            .unwrap_or(byte_idx);
        let byte_end = byte_start + word.len();
        byte_idx = byte_end;

        // Convert byte offsets to char offsets for entity comparison
        let word_start = text[..byte_start].chars().count();
        let word_end = text[..byte_end].chars().count();

        let entity_full = entities
            .iter()
            .find(|e| word_start < e.end() && word_end > e.start());

        let trimmed = word.trim_end_matches(['.', ',', ';', ':', '!', '?', ')', ']']);
        let punct = &word[trimmed.len()..];
        let inside_entity = entity_full.map(|e| word_end <= e.end()).unwrap_or(false);

        if inside_entity {
            let tag = match entity_full {
                Some(e) => {
                    if word_start <= e.start() {
                        format!("B-{}", e.entity_type.as_label())
                    } else {
                        format!("I-{}", e.entity_type.as_label())
                    }
                }
                None => "O".to_string(),
            };
            lines.push(format!("{}\t{}", word, tag));
        } else {
            let trimmed_end = word_start + trimmed.chars().count();
            if !trimmed.is_empty() {
                let entity = entities
                    .iter()
                    .find(|e| word_start < e.end() && trimmed_end > e.start());
                let tag = match entity {
                    Some(e) => {
                        if word_start <= e.start() {
                            format!("B-{}", e.entity_type.as_label())
                        } else {
                            format!("I-{}", e.entity_type.as_label())
                        }
                    }
                    None => "O".to_string(),
                };
                lines.push(format!("{}\t{}", trimmed, tag));
            }

            if !punct.is_empty() {
                lines.push(format!("{}\tO", punct));
            }
        }
    }
    lines.join("\n")
}

/// Export entities as JSONL (one JSON object per entity).
///
/// `source_label` is an arbitrary string identifying the source document
/// (e.g., a filename or URI).
pub fn to_jsonl(entities: &[Entity], source_label: &str, include_confidence: bool) -> String {
    entities
        .iter()
        .map(|e| {
            let mut obj = serde_json::json!({
                "text": e.text,
                "type": e.entity_type.as_label(),
                "start": e.start(),
                "end": e.end(),
                "source": source_label,
            });
            if include_confidence {
                obj["confidence"] = serde_json::json!(e.confidence);
            }
            obj.to_string()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

// ---------------------------------------------------------------------------
// RDF helpers
// ---------------------------------------------------------------------------

fn escape_ntriples(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

fn uri_safe(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn entity_uri(base_uri: &str, entity_type: &str, idx: usize, text: &str, start: usize) -> String {
    let base = base_uri.trim_end_matches('/');
    format!(
        "<{}/entity/{}/{}_{}_{}>",
        base,
        entity_type.to_lowercase(),
        idx,
        uri_safe(text),
        start,
    )
}

fn rel_predicate_uri(base_uri: &str, rel_type: &str) -> String {
    let base = base_uri.trim_end_matches('/');
    format!("<{}/rel/{}>", base, uri_safe(rel_type))
}

/// Minimal CSV field escaping.
fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

/// Export entities and relations as RDF N-Triples.
///
/// `doc_label` identifies the source document (used in provenance triples).
pub fn to_ntriples(
    entities: &[Entity],
    relations: &[Relation],
    doc_label: &str,
    base_uri: &str,
) -> String {
    let mut lines = Vec::new();
    let base = base_uri.trim_end_matches('/');
    let anno_ns = format!("{}/vocab#", base);
    let doc = format!("<{}/doc/{}>", base, uri_safe(doc_label));

    let rdf_type = "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>";
    let rdfs_label = "<http://www.w3.org/2000/01/rdf-schema#label>";

    let ent_uri: Vec<String> = entities
        .iter()
        .enumerate()
        .map(|(i, e)| entity_uri(base_uri, e.entity_type.as_label(), i, &e.text, e.start()))
        .collect();

    for (idx, entity) in entities.iter().enumerate() {
        let ent = &ent_uri[idx];
        let type_uri = format!("<{}{}Type>", anno_ns, entity.entity_type.as_label());
        lines.push(format!("{} {} {} .", ent, rdf_type, type_uri));
        lines.push(format!(
            "{} {} \"{}\" .",
            ent,
            rdfs_label,
            escape_ntriples(&entity.text)
        ));
        lines.push(format!(
            "{} <{}startOffset> \"{}\"^^<http://www.w3.org/2001/XMLSchema#integer> .",
            ent,
            anno_ns,
            entity.start()
        ));
        lines.push(format!(
            "{} <{}endOffset> \"{}\"^^<http://www.w3.org/2001/XMLSchema#integer> .",
            ent,
            anno_ns,
            entity.end()
        ));
        lines.push(format!(
            "{} <{}confidence> \"{}\"^^<http://www.w3.org/2001/XMLSchema#float> .",
            ent, anno_ns, entity.confidence
        ));
        lines.push(format!(
            "{} <http://www.w3.org/ns/prov#hadPrimarySource> {} .",
            ent, doc
        ));
    }

    for rel in relations {
        let head_uri = ent_uri.iter().zip(entities.iter()).find_map(|(u, e)| {
            (e.text == rel.head.text && e.start() == rel.head.start()).then_some(u.as_str())
        });
        let tail_uri = ent_uri.iter().zip(entities.iter()).find_map(|(u, e)| {
            (e.text == rel.tail.text && e.start() == rel.tail.start()).then_some(u.as_str())
        });
        if let (Some(h), Some(t)) = (head_uri, tail_uri) {
            let pred = rel_predicate_uri(base_uri, &rel.relation_type);
            lines.push(format!("{} {} {} .", h, pred, t));
        }
    }

    lines.join("\n")
}

/// Export entities and relations as JSON-LD with `@context`.
///
/// `doc_label` identifies the source document.
pub fn to_jsonld(
    entities: &[Entity],
    relations: &[Relation],
    doc_label: &str,
    include_confidence: bool,
    base_uri: &str,
) -> String {
    let base = base_uri.trim_end_matches('/');
    let entity_ns = format!("{}/entity", base);
    let anno_ns = format!("{}/vocab#", base);

    let entity_ids: Vec<String> = entities
        .iter()
        .enumerate()
        .map(|(i, e)| {
            format!(
                "{}/{}/{}_{}_{}",
                entity_ns,
                e.entity_type.as_label().to_lowercase(),
                i,
                uri_safe(&e.text),
                e.start(),
            )
        })
        .collect();

    let mut rel_by_head: std::collections::HashMap<&str, Vec<serde_json::Value>> =
        std::collections::HashMap::new();
    for rel in relations {
        let head_id = entity_ids.iter().zip(entities.iter()).find_map(|(id, e)| {
            (e.text == rel.head.text && e.start() == rel.head.start()).then_some(id.as_str())
        });
        let tail_id = entity_ids.iter().zip(entities.iter()).find_map(|(id, e)| {
            (e.text == rel.tail.text && e.start() == rel.tail.start()).then_some(id.as_str())
        });
        if let (Some(h), Some(t)) = (head_id, tail_id) {
            rel_by_head.entry(h).or_default().push(serde_json::json!({
                "@type": format!("{}/rel/{}", base, uri_safe(&rel.relation_type)),
                "target": { "@id": t }
            }));
        }
    }

    let graph: Vec<serde_json::Value> = entities
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let id = &entity_ids[i];
            let mut node = serde_json::json!({
                "@id": id,
                "@type": format!("{}{}Type", anno_ns, e.entity_type.as_label()),
                "rdfs:label": e.text,
                "anno:startOffset": e.start(),
                "anno:endOffset": e.end(),
                "prov:hadPrimarySource": {
                    "@id": format!("{}/doc/{}", base, uri_safe(doc_label))
                }
            });
            if include_confidence {
                node["anno:confidence"] = serde_json::json!(e.confidence);
            }
            if let Some(rels) = rel_by_head.get(id.as_str()) {
                node["anno:relations"] = serde_json::json!(rels);
            }
            node
        })
        .collect();

    let doc = serde_json::json!({
        "@context": {
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "prov": "http://www.w3.org/ns/prov#",
            "anno": anno_ns,
            "xsd": "http://www.w3.org/2001/XMLSchema#"
        },
        "@graph": graph
    });

    serde_json::to_string_pretty(&doc).unwrap_or_else(|_| "{}".to_string())
}

/// Export entities and relations as graph-compatible CSV tables.
///
/// Returns `(nodes_csv, edges_csv)`. When relations are available, edges are
/// semantic triples. Otherwise, co-occurrence edges (within 200 chars) are used.
///
/// `source_label` identifies the source document.
pub fn to_graph_csv(
    entities: &[Entity],
    relations: &[Relation],
    source_label: &str,
    include_confidence: bool,
) -> (String, String) {
    let entity_ids: Vec<String> = entities
        .iter()
        .enumerate()
        .map(|(i, e)| {
            format!(
                "{}:{}_{}",
                e.entity_type.as_label().to_lowercase(),
                i,
                uri_safe(&e.text)
            )
        })
        .collect();

    // Nodes CSV
    let mut nodes = String::from("id,entity_type,text,start,end,source");
    if include_confidence {
        nodes.push_str(",confidence");
    }
    nodes.push('\n');
    for (i, e) in entities.iter().enumerate() {
        nodes.push_str(&format!(
            "{},{},{},{},{},{}",
            csv_escape(&entity_ids[i]),
            csv_escape(e.entity_type.as_label()),
            csv_escape(&e.text),
            e.start(),
            e.end(),
            csv_escape(source_label),
        ));
        if include_confidence {
            nodes.push_str(&format!(",{:.4}", e.confidence));
        }
        nodes.push('\n');
    }

    // Edges CSV
    let mut edges = String::from("from,to,rel_type,confidence\n");

    if !relations.is_empty() {
        for rel in relations {
            let head_id = entity_ids.iter().zip(entities.iter()).find_map(|(id, e)| {
                (e.text == rel.head.text && e.start() == rel.head.start()).then_some(id.as_str())
            });
            let tail_id = entity_ids.iter().zip(entities.iter()).find_map(|(id, e)| {
                (e.text == rel.tail.text && e.start() == rel.tail.start()).then_some(id.as_str())
            });
            if let (Some(h), Some(t)) = (head_id, tail_id) {
                edges.push_str(&format!(
                    "{},{},{},{:.4}\n",
                    csv_escape(h),
                    csv_escape(t),
                    csv_escape(&rel.relation_type),
                    rel.confidence,
                ));
            }
        }
    } else {
        const COOCCUR_WINDOW: usize = 200;
        for (i, a) in entities.iter().enumerate() {
            for (j, b) in entities.iter().enumerate().skip(i + 1) {
                let distance = if a.end() <= b.start() {
                    b.start().saturating_sub(a.end())
                } else if b.end() <= a.start() {
                    a.start().saturating_sub(b.end())
                } else {
                    0
                };
                if distance <= COOCCUR_WINDOW {
                    edges.push_str(&format!(
                        "{},{},CO_OCCURS,1.0000\n",
                        csv_escape(&entity_ids[i]),
                        csv_escape(&entity_ids[j]),
                    ));
                }
            }
        }
    }

    (nodes, edges)
}

#[cfg(test)]
mod tests {
    use super::*;
    use anno_core::EntityType;

    #[test]
    fn brat_basic() {
        let text = "Apple CEO Tim Cook is here.";
        let entities = vec![
            Entity::new("Apple", EntityType::Organization, 0, 5, 0.9),
            Entity::new("Tim Cook", EntityType::Person, 10, 18, 0.95),
        ];
        let output = to_brat(text, &entities, false);
        assert!(output.contains("T1\tORG 0 5\tApple"));
        assert!(output.contains("T2\tPER 10 18\tTim Cook"));
    }

    #[test]
    fn brat_with_confidence() {
        let text = "Apple is great.";
        let entities = vec![Entity::new("Apple", EntityType::Organization, 0, 5, 0.9)];
        let output = to_brat(text, &entities, true);
        assert!(output.contains("A1\tConfidence T1 0.90"));
    }

    #[test]
    fn conll_splits_trailing_period_from_entity() {
        let text = "Apple CEO Tim Cook.";
        let entities = vec![
            Entity::new("Apple", EntityType::Organization, 0, 5, 0.9),
            Entity::new("Tim Cook", EntityType::Person, 10, 18, 0.9),
        ];
        let output = to_conll(text, &entities);
        let lines: Vec<&str> = output.lines().collect();
        assert_eq!(lines[0], "Apple\tB-ORG");
        assert_eq!(lines[1], "CEO\tO");
        assert_eq!(lines[2], "Tim\tB-PER");
        assert_eq!(lines[3], "Cook\tI-PER");
        assert_eq!(lines[4], ".\tO");
        assert_eq!(lines.len(), 5);
    }

    #[test]
    fn conll_no_trailing_punct() {
        let text = "Tim Cook spoke";
        let entities = vec![Entity::new("Tim Cook", EntityType::Person, 0, 8, 0.9)];
        let output = to_conll(text, &entities);
        let lines: Vec<&str> = output.lines().collect();
        assert_eq!(lines[0], "Tim\tB-PER");
        assert_eq!(lines[1], "Cook\tI-PER");
        assert_eq!(lines[2], "spoke\tO");
    }

    #[test]
    fn jsonl_basic() {
        let entities = vec![Entity::new("Apple", EntityType::Organization, 0, 5, 0.9)];
        let output = to_jsonl(&entities, "test.txt", true);
        let parsed: serde_json::Value = serde_json::from_str(&output).unwrap();
        assert_eq!(parsed["text"], "Apple");
        assert_eq!(parsed["type"], "ORG");
        assert_eq!(parsed["source"], "test.txt");
    }

    #[test]
    fn graph_csv_cooccurrence_fallback() {
        let entities = vec![
            Entity::new("Apple", EntityType::Organization, 0, 5, 0.9),
            Entity::new("Tim Cook", EntityType::Person, 10, 18, 0.95),
        ];
        let (nodes, edges) = to_graph_csv(&entities, &[], "test.txt", false);
        assert!(nodes.contains("Apple"));
        assert!(nodes.contains("Tim Cook"));
        assert!(edges.contains("CO_OCCURS"));
    }

    #[test]
    fn conll_non_ascii() {
        // "cafe" with accent: "caf\u{e9}" is 4 chars but 5 bytes
        let text = "Visit caf\u{e9} in Paris today.";
        let entities = vec![Entity::new("Paris", EntityType::Location, 14, 19, 0.9)];
        let output = to_conll(text, &entities);
        assert!(output.contains("Paris\tB-LOC"), "got: {}", output);
    }

    #[test]
    fn brat_non_ascii() {
        let text = "Visit caf\u{e9} in Paris today.";
        let entities = vec![Entity::new("Paris", EntityType::Location, 14, 19, 0.9)];
        let output = to_brat(text, &entities, false);
        // brat uses byte offsets: "caf\u{e9}" is 5 bytes, so "Paris" starts at byte 15
        let byte_start: usize = text.chars().take(14).map(|c| c.len_utf8()).sum();
        let byte_end: usize = text.chars().take(19).map(|c| c.len_utf8()).sum();
        assert!(
            output.contains(&format!("{} {}", byte_start, byte_end)),
            "got: {}",
            output
        );
    }
}
