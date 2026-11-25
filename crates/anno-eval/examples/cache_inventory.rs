//! Count loader-addressable objects in a paginated S3 key listing. Only aggregate
//! counts are printed. An optional locally read sample can establish a digest match.
use anno::core::grounded::html_source::source_sha256;
use anno_eval::eval::dataset_registry::DatasetId;
use serde::Serialize;
use std::{collections::HashSet, env, error::Error, fs};

const PREFIX: &str = "datasets/";
const WIKIGOLD_SHA256: &str = "c797a64d0cf73ed058363f77671486bcfd413e70fda1726ad4a6ba624455225f";

enum Object {
    Legacy,
    Snapshot,
    Pointer,
    Manifest,
}

fn classify(key: &str) -> Option<(DatasetId, Object)> {
    let name = key.strip_prefix(PREFIX)?;
    let (file, kind) = if let Some(path) = name.strip_prefix("by-sha256/") {
        let (hash, file) = path.split_once('/')?;
        if hash.len() != 64 || !hash.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')) {
            return None;
        }
        (file, Object::Snapshot)
    } else if let Some(file) = name.strip_suffix(".latest.json") {
        (file, Object::Pointer)
    } else if let Some(file) = name.strip_suffix(".manifest.json") {
        (file, Object::Manifest)
    } else {
        (name, Object::Legacy)
    };
    Some((DatasetId::from_cache_filename(file)?, kind))
}

#[derive(Default, Serialize)]
struct Inventory {
    objects: usize,
    registered_legacy_payloads: usize,
    registered_snapshots: usize,
    registered_pointers: usize,
    registered_manifests: usize,
    unrecognized_keys: usize,
    unrecognized_keys_matching_bulk_read_exclusions: usize,
    registry_ids_with_payload: usize,
    registry_ids_with_only_metadata: usize,
    registry_pointers_without_any_listed_snapshot: usize,
    payloads_excluded_from_shared_reads: usize,
    payloads_requiring_local_source: usize,
    payloads_approved_for_shared_publication: usize,
    payloads_with_digest_matched_rights_evidence: usize,
}

fn inventory(keys: &[String], sample: Option<(&str, &[u8])>) -> Result<Inventory, &'static str> {
    let mut report = Inventory {
        objects: keys.len(),
        ..Default::default()
    };
    // The bulk sync command filters *name* against these case-sensitive rules.
    // A filename hint alone never supplies a registry identity or digest.
    let mut excluded_names = Vec::new();
    for line in include_str!("../src/eval/shared_cache_exclusions.tsv").lines() {
        let fields: Vec<_> = line.split('\t').collect();
        excluded_names.push(fields[0].trim_end_matches(".cache"));
        excluded_names.extend(fields[2].split(','));
    }
    let mut payload_ids = HashSet::new();
    let mut snapshot_ids = HashSet::new();
    let mut pointer_ids = HashSet::new();
    let mut metadata_ids = HashSet::new();
    for key in keys {
        if !key.starts_with(PREFIX) {
            return Err("unexpected object outside datasets/");
        }
        let Some((id, kind)) = classify(key) else {
            report.unrecognized_keys += 1;
            report.unrecognized_keys_matching_bulk_read_exclusions +=
                usize::from(excluded_names.iter().any(|name| key.contains(name)));
            continue;
        };
        match kind {
            Object::Pointer => {
                report.registered_pointers += 1;
                pointer_ids.insert(id);
                metadata_ids.insert(id);
                continue;
            }
            Object::Manifest => {
                report.registered_manifests += 1;
                metadata_ids.insert(id);
                continue;
            }
            Object::Legacy => report.registered_legacy_payloads += 1,
            Object::Snapshot => {
                report.registered_snapshots += 1;
                snapshot_ids.insert(id);
            }
        }
        payload_ids.insert(id);
        report.payloads_excluded_from_shared_reads += usize::from(id.excluded_from_shared_cache());
        report.payloads_requiring_local_source += usize::from(id.requires_local_source());
        report.payloads_approved_for_shared_publication +=
            usize::from(id.approved_for_shared_publication());
    }
    report.registry_ids_with_payload = payload_ids.len();
    report.registry_ids_with_only_metadata = metadata_ids.difference(&payload_ids).count();
    report.registry_pointers_without_any_listed_snapshot =
        pointer_ids.difference(&snapshot_ids).count();

    // docs/QA.md identifies the exact author release and documents separate
    // terms for its annotations and the underlying Wikipedia text.
    if let Some((key, bytes)) = sample {
        let snapshot = format!("{PREFIX}by-sha256/{WIKIGOLD_SHA256}/WikiGold.cache");
        if !keys.iter().any(|listed| listed == key)
            || (key != "datasets/WikiGold.cache" && key != snapshot)
        {
            return Err("expected a listed WikiGold sample key");
        }
        if source_sha256(std::str::from_utf8(bytes).map_err(|_| "sample is not UTF-8")?)
            != WIKIGOLD_SHA256
        {
            return Err("sample digest differs from reviewed WikiGold release");
        }
        report.payloads_with_digest_matched_rights_evidence = 1;
    }
    Ok(report)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args_os().skip(1).collect();
    let (listing, sample) = match args.as_slice() {
        [listing] => (listing, None),
        [listing, key, sample] => (
            listing,
            Some((key.to_str().ok_or("invalid key")?, fs::read(sample)?)),
        ),
        _ => return Err("usage: cache_inventory KEYS.json [WIKIGOLD_KEY SAMPLE_FILE]".into()),
    };
    // AWS CLI prints null for an absent Contents array in an empty prefix.
    let keys =
        serde_json::from_slice::<Option<Vec<String>>>(&fs::read(listing)?)?.unwrap_or_default();
    let report = inventory(
        &keys,
        sample.as_ref().map(|(key, bytes)| (*key, bytes.as_slice())),
    )?;
    println!("{}", serde_json::to_string(&report)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn separates_payloads_metadata_and_unrecognized_keys() {
        let digest = "a".repeat(64);
        let keys = vec![
            "datasets/WikiGold.cache".into(),
            format!("datasets/by-sha256/{digest}/WikiGold.cache"),
            "datasets/WikiGold.cache.latest.json".into(),
            "datasets/PreCo.cache".into(),
            "datasets/CUAD.cache.latest.json".into(),
            "datasets/CUAD.cache.manifest.json".into(),
            "datasets/Unknown.cache".into(),
            "datasets/WikiGold.json".into(),
            "datasets/by-sha256/invalid/CUAD.cache".into(),
        ];
        let r = inventory(&keys, None).unwrap();
        assert_eq!(
            (
                r.registered_legacy_payloads,
                r.registered_snapshots,
                r.unrecognized_keys
            ),
            (2, 1, 3)
        );
        assert_eq!(
            (
                r.registry_ids_with_payload,
                r.registry_ids_with_only_metadata
            ),
            (2, 1)
        );
        assert_eq!(
            (
                r.registered_pointers,
                r.registered_manifests,
                r.registry_pointers_without_any_listed_snapshot
            ),
            (2, 1, 1)
        );
        assert_eq!(
            (
                r.payloads_excluded_from_shared_reads,
                r.payloads_requiring_local_source
            ),
            (1, 1)
        );
        assert_eq!(r.unrecognized_keys_matching_bulk_read_exclusions, 1);
        assert_eq!(r.payloads_with_digest_matched_rights_evidence, 0);
    }

    #[test]
    fn a_name_or_hash_in_the_key_does_not_verify_bytes() {
        let key = format!("{PREFIX}by-sha256/{WIKIGOLD_SHA256}/WikiGold.cache");
        let keys = vec![key.clone()];
        assert_eq!(
            inventory(&keys, None)
                .unwrap()
                .payloads_with_digest_matched_rights_evidence,
            0
        );
        assert!(inventory(&keys, Some((&key, b"new"))).is_err());
        assert!(inventory(&keys, Some(("datasets/WikiGold.cache", b"new"))).is_err());
        assert!(inventory(&["somewhere/else".into()], None).is_err());
    }
}
