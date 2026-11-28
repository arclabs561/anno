//! Offline inventory of checkpoint entry points and preparation recipes.
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Checkpoint {
    pub(super) id: String,
    command: String,
    source: String,
    preparation: String,
    guide: String,
    notes: String,
}

fn inventory() -> Result<Vec<Checkpoint>, String> {
    let mut entries = Vec::new();
    for data in [include_str!("extraction.json"), include_str!("tasks.json")] {
        let group: Vec<Checkpoint> = serde_json::from_str(data)
            .map_err(|e| format!("Invalid embedded checkpoint inventory: {e}"))?;
        entries.extend(group);
    }
    entries.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(entries)
}

pub(super) fn find(id: &str) -> Result<Option<Checkpoint>, String> {
    let normalized = id.to_ascii_lowercase().replace('_', "-");
    // The CLI accepts both spellings for the same ONNX checkpoint.
    let id = if normalized == "gliner-onnx" {
        "gliner"
    } else {
        &normalized
    };
    Ok(inventory()?.into_iter().find(|entry| entry.id == id))
}

pub(super) fn print_details(entry: &Checkpoint) {
    println!("  Entry point: {}", entry.command);
    if !entry.source.is_empty() {
        println!("  Source: {}", entry.source);
    }
    if !entry.preparation.is_empty() {
        println!("  Preparation: {}", entry.preparation);
    }
    println!("  Guide: {}", entry.guide);
    println!("  {}", entry.notes);
}

pub(super) fn run(id: Option<&str>, json: bool) -> Result<(), String> {
    let entries = match id {
        Some(id) => vec![find(id)?.ok_or_else(|| {
            format!("Unknown checkpoint '{id}'. Run 'anno models checkpoints' to list IDs.")
        })?],
        None => inventory()?,
    };
    if json {
        let output = serde_json::to_string_pretty(&entries).map_err(|e| e.to_string())?;
        println!("{output}");
    } else if id.is_some() {
        let entry = &entries[0];
        println!("{}", entry.id);
        print_details(entry);
    } else {
        for entry in entries {
            println!("{:<26} {}", entry.id, entry.command);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{collections::HashSet, path::Path};

    #[test]
    fn checkpoint_entries_have_distinct_ids_and_resolvable_repo_references() {
        let entries = inventory().unwrap();
        let mut ids = HashSet::new();
        let checkout = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        for entry in &entries {
            assert!(ids.insert(entry.id.as_str()), "duplicate {}", entry.id);
            assert!(!entry.id.is_empty());
            assert!(entry
                .id
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-'));
            assert!(
                !entry.command.is_empty() && !entry.guide.is_empty() && !entry.notes.is_empty()
            );
            // External guides and recipes are shipped in the source checkout.
            if checkout.join("crates/anno-cli/Cargo.toml").is_file() {
                for path in [&entry.preparation, &entry.guide] {
                    if !path.is_empty() {
                        assert!(
                            checkout.join(path).is_file(),
                            "{} references {path}",
                            entry.id
                        );
                    }
                }
            }
        }
        assert_eq!(find("OTTER_BI").unwrap().unwrap().id, "otter-bi");
        assert_eq!(find("relik").unwrap().unwrap().id, "relik");
        assert_eq!(find("gliner_onnx").unwrap().unwrap().id, "gliner");
        assert!(find("unknown-checkpoint").unwrap().is_none());
    }
}
