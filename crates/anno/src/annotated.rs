//! Document with extraction results attached.

use crate::{CorefChain, Entity, GroundedDocument, Language, Model, Relation, Result, StackedNER};

/// Text paired with its extraction outputs (entities, relations, coreference chains).
///
/// Holds the source text alongside extracted annotations. Relations and coreference
/// chains are populated only when the backend supports them; otherwise they are empty.
///
/// # Example
///
/// ```rust
/// use anno::AnnotatedDoc;
///
/// let doc = anno::annotate("Lynn Conway worked at IBM and Xerox PARC.")?;
/// assert!(!doc.entities.is_empty());
/// for text in doc.entity_texts() {
///     println!("{text}");
/// }
/// # Ok::<(), anno::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct AnnotatedDoc {
    /// Source text.
    pub text: String,
    /// Extracted entities.
    pub entities: Vec<Entity>,
    /// Extracted relations (empty when the backend does not produce relations).
    pub relations: Vec<Relation>,
    /// Coreference chains (empty when the backend does not produce coreference).
    pub coref_chains: Vec<CorefChain>,
}

impl AnnotatedDoc {
    /// Build an `AnnotatedDoc` from pre-computed parts.
    #[must_use]
    pub fn new(
        text: impl Into<String>,
        entities: Vec<Entity>,
        relations: Vec<Relation>,
        coref_chains: Vec<CorefChain>,
    ) -> Self {
        Self {
            text: text.into(),
            entities,
            relations,
            coref_chains,
        }
    }

    /// Surface-form texts of every extracted entity, in extraction order.
    ///
    /// ```rust
    /// # use anno::AnnotatedDoc;
    /// let doc = anno::annotate("Marie Curie won the Nobel Prize.")?;
    /// let texts = doc.entity_texts();
    /// assert!(texts.contains(&"Marie Curie"));
    /// # Ok::<(), anno::Error>(())
    /// ```
    #[must_use]
    pub fn entity_texts(&self) -> Vec<&str> {
        self.entities.iter().map(|e| e.text.as_str()).collect()
    }
}

/// Extract entities from text using the default backend and return an [`AnnotatedDoc`].
///
/// Creates a [`StackedNER`] and populates entities. Relations and coreference chains
/// are left empty (the default backend does not produce them).
///
/// For control over backend selection or language hints, construct an [`AnnotatedDoc`]
/// directly via [`AnnotatedDoc::new`].
///
/// ```rust
/// let doc = anno::annotate("Grace Hopper invented COBOL at the US Navy.")?;
/// assert!(!doc.entities.is_empty());
/// assert!(doc.relations.is_empty()); // default backend has no RE
/// # Ok::<(), anno::Error>(())
/// ```
pub fn annotate(text: &str) -> Result<AnnotatedDoc> {
    let model = StackedNER::default();
    let entities = model.extract_entities(text, None)?;
    Ok(AnnotatedDoc::new(text, entities, Vec::new(), Vec::new()))
}

/// Extract a document with the default backend into the canonical grounded representation.
///
/// The returned document contains one signal and singleton track for each extracted entity.
/// Use [`annotate_grounded_with`] when reusing a model or supplying a language hint.
///
/// ```rust
/// let doc = anno::annotate_grounded("bio-1", "Ada Lovelace wrote a program.")?;
/// assert_eq!(doc.id(), "bio-1");
/// assert_eq!(doc.text(), "Ada Lovelace wrote a program.");
/// # Ok::<(), anno::Error>(())
/// ```
pub fn annotate_grounded(id: impl Into<String>, text: &str) -> Result<GroundedDocument> {
    let model = StackedNER::default();
    annotate_grounded_with(&model, id, text, None)
}

/// Extract one document with `model` into a [`GroundedDocument`].
///
/// Entity offsets, normalized values, hierarchical confidence, provenance, discontinuous
/// spans, canonical-ID grouping, and knowledge-base IDs are preserved by
/// [`GroundedDocument::from_entities`]. Grounded track IDs are assigned in first-seen entity
/// order. Entities without a canonical ID become separate singleton tracks; this function does
/// not run coreference or knowledge-base linking.
///
/// ```rust
/// use anno::{annotate_grounded_with, StackedNER};
///
/// let model = StackedNER::builder()
///     .layer(anno::RegexNER::new())
///     .layer(anno::HeuristicNER::new())
///     .build();
/// let doc = annotate_grounded_with(
///     &model,
///     "memo-7",
///     "Grace Hopper developed COBOL.",
///     None,
/// )?;
/// assert_eq!(doc.id(), "memo-7");
/// # Ok::<(), anno::Error>(())
/// ```
pub fn annotate_grounded_with(
    model: &dyn Model,
    id: impl Into<String>,
    text: &str,
    language: Option<Language>,
) -> Result<GroundedDocument> {
    let id = id.into();
    let entities = model.extract_entities(text, language)?;
    Ok(GroundedDocument::from_entities(id, text, &entities))
}

/// Extract multiple identified documents with `model` into grounded documents.
///
/// `documents` contains `(id, text)` pairs. Results retain input order and one result is
/// returned for every input, so a failed document does not discard successful neighbours.
/// The model's batch method may use internal batching.
///
/// A backend that returns a result count different from the input count violates
/// [`Model::extract_batch`]'s contract. In that case every input receives an error instead of
/// silently associating extraction results with the wrong document.
///
/// ```rust
/// use anno::{annotate_grounded_batch_with, StackedNER};
///
/// let model = StackedNER::builder()
///     .layer(anno::RegexNER::new())
///     .layer(anno::HeuristicNER::new())
///     .build();
/// let docs = [("one", "Ada Lovelace wrote a program."), ("two", "COBOL was developed.")];
/// let results = annotate_grounded_batch_with(&model, &docs, None);
/// assert_eq!(results.len(), docs.len());
/// # Ok::<(), anno::Error>(())
/// ```
pub fn annotate_grounded_batch_with(
    model: &dyn Model,
    documents: &[(&str, &str)],
    language: Option<Language>,
) -> Vec<Result<GroundedDocument>> {
    let texts: Vec<_> = documents.iter().map(|(_, text)| *text).collect();
    let results = model.extract_batch(&texts, language);
    grounded_documents_from_batch(documents, results)
}

fn grounded_documents_from_batch(
    documents: &[(&str, &str)],
    results: Vec<Result<Vec<Entity>>>,
) -> Vec<Result<GroundedDocument>> {
    if results.len() != documents.len() {
        let message = format!(
            "model returned {} batch results for {} documents",
            results.len(),
            documents.len()
        );
        return documents
            .iter()
            .map(|_| Err(crate::Error::Backend(message.clone())))
            .collect();
    }

    documents
        .iter()
        .zip(results)
        .map(|((id, text), result)| {
            result.map(|entities| GroundedDocument::from_entities(*id, *text, &entities))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AnyModel, EntityType};

    #[test]
    fn annotate_returns_entities() {
        let doc = annotate("Marie Curie won the Nobel Prize.").unwrap();
        assert!(!doc.entities.is_empty(), "should find at least one entity");
        assert!(doc.relations.is_empty());
        assert!(doc.coref_chains.is_empty());
    }

    #[test]
    fn annotate_empty_text() {
        let doc = annotate("").unwrap();
        assert!(doc.entities.is_empty());
        assert!(doc.relations.is_empty());
        assert!(doc.coref_chains.is_empty());
    }

    #[test]
    fn entity_texts_matches_entities() {
        let doc = annotate("Lynn Conway worked at IBM.").unwrap();
        let texts = doc.entity_texts();
        assert_eq!(texts.len(), doc.entities.len());
        for (text, entity) in texts.iter().zip(&doc.entities) {
            assert_eq!(*text, entity.text.as_str());
        }
    }

    #[test]
    fn new_preserves_all_fields() {
        let entities = vec![Entity::new("Alice", crate::EntityType::Person, 0, 5, 0.9)];
        let relations = vec![];
        let chains = vec![];
        let doc = AnnotatedDoc::new("Alice went home.", entities.clone(), relations, chains);
        assert_eq!(doc.text, "Alice went home.");
        assert_eq!(doc.entities.len(), 1);
        assert_eq!(doc.entities[0].text, "Alice");
    }

    #[test]
    fn entity_texts_empty_doc() {
        let doc = AnnotatedDoc::new("nothing here", vec![], vec![], vec![]);
        assert!(doc.entity_texts().is_empty());
    }

    #[test]
    fn annotate_grounded_with_preserves_identified_entity_data() {
        let mut entity = Entity::new("Zo\u{eb}", EntityType::Person, 0, 3, 0.9);
        entity.normalized = Some("Zoe".into());
        let model = AnyModel::new(
            "fixture",
            "fixture model",
            vec![EntityType::Person],
            move |_text, _language| Ok(vec![entity.clone()]),
        );

        let doc = annotate_grounded_with(&model, "person-1", "Zo\u{eb} spoke.", None).unwrap();

        assert_eq!(doc.id(), "person-1");
        assert_eq!(doc.text(), "Zo\u{eb} spoke.");
        assert_eq!(doc.signals().len(), 1);
        assert_eq!(doc.signals()[0].surface, "Zo\u{eb}");
        assert_eq!(doc.signals()[0].text_offsets(), Some((0, 3)));
        assert_eq!(doc.signals()[0].normalized.as_deref(), Some("Zoe"));
        assert_eq!(doc.tracks_map().len(), 1);
        assert!(doc.validate_invariants().is_empty());
    }

    #[test]
    fn annotate_grounded_batch_with_preserves_order_and_document_ids() {
        let model = AnyModel::new(
            "fixture",
            "fixture model",
            vec![EntityType::Person],
            |text, _language| {
                let end = text.chars().count();
                Ok(vec![Entity::new(text, EntityType::Person, 0, end, 0.9)])
            },
        );
        let inputs = [("first", "Ada"), ("second", "Grace")];

        let documents = annotate_grounded_batch_with(&model, &inputs, None);

        assert_eq!(documents.len(), inputs.len());
        let documents: Vec<_> = documents.into_iter().collect::<Result<_>>().unwrap();
        assert_eq!(documents[0].id(), "first");
        assert_eq!(documents[0].signals()[0].surface, "Ada");
        assert_eq!(documents[1].id(), "second");
        assert_eq!(documents[1].signals()[0].surface, "Grace");
    }

    #[test]
    fn malformed_batch_result_count_does_not_misassociate_documents() {
        let inputs = [("first", "Ada"), ("second", "Grace")];
        let results = vec![Ok(vec![Entity::new("Ada", EntityType::Person, 0, 3, 0.9)])];

        let documents = grounded_documents_from_batch(&inputs, results);

        assert_eq!(documents.len(), inputs.len());
        for result in documents {
            assert!(
                matches!(result, Err(crate::Error::Backend(message)) if message.contains("1 batch results for 2 documents"))
            );
        }
    }
}
