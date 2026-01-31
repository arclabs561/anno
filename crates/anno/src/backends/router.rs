use crate::backends::stacked::StackedNER;
use crate::{Entity, EntityType, Model, Result};

/// Automatic model selection - routes to the default model.
///
/// AutoNER is simply an alias for the default model (StackedNER).
/// It does NOT combine multiple models - it just picks one default method.
/// This avoids mixing and matching scores from different models.
pub struct AutoNER {
    default_model: StackedNER,
}

impl AutoNER {
    /// Create a new AutoNER (routes to default StackedNER).
    pub fn new() -> Self {
        Self {
            default_model: StackedNER::default(),
        }
    }
}

impl Default for AutoNER {
    fn default() -> Self {
        Self::new()
    }
}

impl Model for AutoNER {
    fn extract_entities(&self, text: &str, language: Option<&str>) -> Result<Vec<Entity>> {
        // AutoNER just routes to the default model (StackedNER).
        // It does NOT combine multiple models - it picks one method.
        self.default_model.extract_entities(text, language)
    }

    fn supported_types(&self) -> Vec<EntityType> {
        self.default_model.supported_types()
    }

    fn is_available(&self) -> bool {
        self.default_model.is_available()
    }

    fn name(&self) -> &'static str {
        "auto"
    }

    fn description(&self) -> &'static str {
        "Automatic model selection (default: StackedNER)"
    }
}

impl crate::sealed::Sealed for AutoNER {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_ner_creation() {
        let auto = AutoNER::new();
        assert!(auto.is_available());
    }

    #[test]
    fn test_auto_ner_default() {
        let auto = AutoNER::default();
        assert_eq!(auto.name(), "auto");
    }

    #[test]
    fn test_auto_ner_extracts_entities() {
        let auto = AutoNER::new();
        let entities = auto
            .extract_entities("Google CEO Sundar Pichai announced.", None)
            .unwrap();
        // AutoNER should extract at least some entities
        assert!(!entities.is_empty(), "AutoNER should extract entities");
    }

    #[test]
    fn test_auto_ner_supported_types() {
        let auto = AutoNER::new();
        let types = auto.supported_types();
        // Should have at least PER, ORG, LOC from StackedNER
        assert!(!types.is_empty());
    }

    #[test]
    fn test_auto_ner_description() {
        let auto = AutoNER::new();
        assert!(auto.description().contains("Automatic"));
    }

    #[test]
    fn test_auto_ner_empty_text() {
        let auto = AutoNER::new();
        let entities = auto.extract_entities("", None).unwrap();
        assert!(entities.is_empty());
    }
}
