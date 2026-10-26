//! Library constructors must not import `.env` files from the working
//! directory or its parents. Only the CLI loads `.env`.
//!
//! This is its own test binary because it changes the process working
//! directory.

#[test]
fn universal_ner_does_not_load_dotenv_from_the_working_directory() {
    let key = format!("ANNO_DOTENV_SENTINEL_{}", std::process::id());
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join(".env"), format!("{key}=from-dotenv\n")).unwrap();
    std::env::remove_var(&key);
    std::env::set_current_dir(dir.path()).unwrap();

    let _ = anno::backends::universal_ner::UniversalNER::new();

    assert!(std::env::var(&key).is_err());
}
