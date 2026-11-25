//! Import concept-level relation annotations from one DWIE JSON with reconstructed content.
//! Run with `cargo run -p anno --no-default-features --example import_dwie_relations -- <article.json>`.

use anno::core::grounded::ConceptRelation;
use anno::{ExtractionMethod, GroundedDocument, Location, Provenance, Signal, SignalId};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, BufReader};

#[derive(Deserialize)]
struct Article {
    id: String,
    content: String,
    concepts: Vec<Concept>,
    mentions: Vec<Mention>,
    relations: Vec<Edge>,
}

#[derive(Deserialize)]
struct Concept {
    concept: usize,
    text: Option<String>,
}

#[derive(Deserialize)]
struct Mention {
    begin: usize,
    end: usize,
    text: String,
    concept: usize,
}

#[derive(Deserialize)]
struct Edge {
    s: usize,
    p: String,
    o: usize,
}

fn import(article: Article) -> Result<GroundedDocument, String> {
    let mut doc = GroundedDocument::new(article.id, article.content);
    let provenance = Provenance::from_producer("DWIE annotation", ExtractionMethod::Unknown);
    let mut membership = BTreeMap::new();
    for concept in article.concepts {
        if membership
            .insert(concept.concept, (concept.text, Vec::new()))
            .is_some()
        {
            return Err(format!("duplicate DWIE concept {}", concept.concept));
        }
    }
    for mention in article.mentions {
        let (_, members) = membership
            .get_mut(&mention.concept)
            .ok_or_else(|| format!("unknown DWIE concept {}", mention.concept))?;
        // The signal score denotes presence in the supplied annotation. Entity
        // typing is outside this import; the original concepts can be multi-label.
        let signal = Signal::new(
            SignalId::ZERO,
            Location::text(mention.begin, mention.end),
            mention.text,
            "mention",
            1.0,
        )
        .with_provenance(provenance.clone());
        members.push(
            doc.add_signal_validated(signal)
                .map_err(|error| error.to_string())?,
        );
    }
    let mut endpoints = BTreeMap::new();
    for (id, (name, members)) in membership {
        if let Some(&first) = members.first() {
            let surface = name.unwrap_or_else(|| {
                doc.get_signal(first)
                    .expect("inserted member")
                    .surface
                    .clone()
            });
            doc.create_track_from_signals(surface, &members)
                .ok_or_else(|| format!("could not create track for DWIE concept {id}"))?;
        }
        endpoints.insert(id, doc.concept_endpoint(id.to_string(), Some(&members))?);
    }
    for edge in article.relations {
        let endpoint = |id| {
            endpoints
                .get(&id)
                .cloned()
                .ok_or_else(|| format!("unknown DWIE relation endpoint {id}"))
        };
        doc.record_concept_relation(ConceptRelation {
            head: endpoint(edge.s)?,
            tail: endpoint(edge.o)?,
            relation_type: edge.p,
            confidence: None,
            provenance: Some(provenance.clone()),
        })?;
    }
    Ok(doc)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let path = args
        .next()
        .ok_or("expected one source-backed DWIE JSON file")?;
    if args.next().is_some() {
        return Err("expected one source-backed DWIE JSON file".into());
    }
    let article: Article = serde_json::from_reader(BufReader::new(File::open(path)?))?;
    let document = import(article)?;
    serde_json::to_writer(io::stdout().lock(), &document)?;
    Ok(())
}
