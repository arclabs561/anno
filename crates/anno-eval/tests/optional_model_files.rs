#![cfg(feature = "eval")]

use anno_eval::eval::backend_factory::BackendFactory;

// Process isolation keeps the environment overrides out of unrelated tests.
#[test]
fn empty_optional_model_directories_are_not_available() {
    let dir = tempfile::tempdir().unwrap();
    for key in [
        "OTTER_CROSS_MODEL_DIR",
        "OTTER_BI_MODEL_DIR",
        "KNIV_MODEL_DIR",
        "ANNO_GLINEREL_MODEL_DIR",
        "ADAPTNER_MODEL_DIR",
        "GLINER2_PII_MODEL_DIR",
    ] {
        std::env::set_var(key, dir.path());
    }
    for backend in [
        "otter-cross",
        "otter-bi",
        "kniv",
        "glinerel-72m",
        "adaptner",
        "gliner2-pii",
    ] {
        assert!(!BackendFactory::is_available(backend), "{backend}");
        assert!(
            !BackendFactory::is_available(&backend.replace('-', "_")),
            "{backend}"
        );
    }
    assert!(BackendFactory::create("unknown-checkpoint-qa-fixture").is_err());
}
