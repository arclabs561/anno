//! This test has its own process because it changes the working directory.
use anno::backends::universal_ner::UniversalNER;
use std::{env, fs};

#[test]
fn library_constructors_leave_dotenv_to_the_caller() {
    const KEY: &str = "ANNO_TEST_IMPLICIT_DOTENV";
    let original_dir = env::current_dir().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let child = dir.path().join("child");
    fs::create_dir(&child).unwrap();
    fs::write(dir.path().join(".env"), format!("{KEY}=from-parent\n")).unwrap();
    env::remove_var(KEY);
    env::set_current_dir(&child).unwrap();

    UniversalNER::new().unwrap();
    assert!(env::var_os(KEY).is_none());
    UniversalNER::with_config(Default::default()).unwrap();
    assert!(env::var_os(KEY).is_none());

    #[cfg(all(feature = "onnx", not(target_arch = "wasm32")))]
    {
        env::set_var("ANNO_NO_DOWNLOADS", "1");
        env::set_var("HF_HOME", dir.path().join("hf"));
        let Err(error) = anno::BertNEROnnx::new("anno-tests/uncached") else {
            panic!("model loaded from an empty cache");
        };
        assert!(error.to_string().contains("ANNO_NO_DOWNLOADS"), "{error}");
        assert!(env::var_os(KEY).is_none());
    }

    anno::env::load_dotenv();
    assert_eq!(env::var(KEY).as_deref(), Ok("from-parent"));
    env::set_current_dir(original_dir).unwrap();
}
