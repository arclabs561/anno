//! Document with extraction results attached.

use crate::{CorefChain, Entity, Model, Relation, Result, StackedNER};

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
