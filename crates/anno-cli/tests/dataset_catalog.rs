#![cfg(feature = "eval")]

use anno_eval::eval::{dataset_registry::DatasetId, loader::LoadableDatasetId};
use std::process::Command;

fn catalog(args: &[&str]) -> String {
    let run = Command::new(env!("CARGO_BIN_EXE_anno"))
        .arg("dataset")
        .args(args)
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    String::from_utf8(run.stdout).unwrap()
}

#[test]
fn catalog_selection_counts_the_displayed_standard_loaders() {
    let domain = "wikipedia";
    let expected: Vec<_> = DatasetId::all()
        .iter()
        .filter(|id| {
            id.domain().to_lowercase().contains(domain) && LoadableDatasetId::try_from(**id).is_ok()
        })
        .map(|id| id.name())
        .collect();
    let output = catalog(&["list", "--loadable", "--domain", domain]);
    let actual: Vec<_> = output
        .lines()
        .filter_map(|line| line.strip_prefix("    "))
        .collect();
    assert_eq!(actual, expected);
    assert!(output.contains(&format!(
        "{} entries supported by the standard loader",
        expected.len()
    )));
}

#[test]
fn docred_is_reported_as_catalog_only() {
    let output = catalog(&["check", "--dataset", "DocRED"]);
    assert!(output.contains("DocRED: catalog only"));
    assert!(output.contains("Supported by the standard loader: 0"));
    assert!(!output.contains("Warnings"));
}
