#![cfg(all(feature = "eval", feature = "onnx"))]

use anno_eval::eval::backend_factory::BackendFactory;
use std::{fs, path::Path};

// This integration test runs alone in its process so the cache and model overrides
// never affect another test or the user's real Hub cache.
#[test]
fn prepared_4k_export_runs_offline_through_the_evaluation_factory() {
    let tmp = tempfile::tempdir().unwrap();
    std::env::set_var("HF_HOME", tmp.path());
    std::env::set_var("ANNO_NO_DOWNLOADS", "1");
    std::env::remove_var("NUNER_4K_MODEL_PATH");
    let listed = || BackendFactory::available_backends().contains(&"nuner_4k");
    assert!(!listed());
    assert!(matches!(
        BackendFactory::create("nuner_4k"),
        Err(anno::Error::FeatureNotAvailable(_))
    ));

    let out = tmp.path().join("hub/models--numind--NuNER_Zero-4k/onnx");
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../anno/tests/bughunt7_data/hf_home/hub/models--bh7--nuner-4k/snapshots/fixture");
    fs::create_dir_all(&out).unwrap();
    fs::copy(fixture.join("onnx/model.onnx"), out.join("model.onnx")).unwrap();
    assert!(!listed());
    fs::copy(fixture.join("tokenizer.json"), out.join("tokenizer.json")).unwrap();
    assert!(listed());
    let model = BackendFactory::create("nuner_4k").unwrap();
    let entities = model.extract_entities("Call John Smith now", None).unwrap();
    assert_eq!(entities.len(), 1);
    assert_eq!(entities[0].text, "John");
    assert_eq!(entities[0].entity_type, anno::EntityType::Person);
    assert_eq!((entities[0].start(), entities[0].end()), (5, 9));

    // An explicit directory takes precedence over prepared cache assets.
    std::env::set_var("NUNER_4K_MODEL_PATH", tmp.path().join("missing"));
    assert!(!listed());
    std::env::set_var("NUNER_4K_MODEL_PATH", &fixture);
    assert!(listed());
    assert!(BackendFactory::create("nunerzero4k").is_ok());
}
