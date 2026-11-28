//! Pair native evaluation streams by exact source and reference identity.

use super::{Arm, PairedRow};
use crate::core::grounded::html_source::source_sha256;
use crate::core::grounded::{EvalComparison, EvalMatch, SignalId};
use crate::viz::check_label;
use crate::viz::errors::{EvalErrors, Report};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::Read;

const EVAL_PAIRING_SCHEMA: &str = "anno.eval-pairing/v1";

/// Source and reference fingerprints for pairing `anno eval --json` reports.
/// Each gold signal ID is local to its own report. A value fingerprints the
/// typed canonical covered ranges. Repeated identical gold occurrences remain
/// separate through their multiplicity. No source text is needed for replay.
#[derive(Debug, Deserialize, Serialize)]
pub struct EvalPairing {
    schema: String,
    document_id: String,
    source_sha256: String,
    #[serde(deserialize_with = "read_gold_keys")]
    gold_keys: BTreeMap<SignalId, String>,
}

// The default map decoder replaces an earlier value if a JSON object repeats a
// key. A signal ID can only identify a single reference in a pairing receipt.
fn read_gold_keys<'de, D: serde::Deserializer<'de>>(
    de: D,
) -> Result<BTreeMap<SignalId, String>, D::Error> {
    struct Keys;
    impl<'de> serde::de::Visitor<'de> for Keys {
        type Value = BTreeMap<SignalId, String>;

        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("an object mapping distinct gold signal IDs to fingerprints")
        }

        fn visit_map<A: serde::de::MapAccess<'de>>(
            self,
            mut map: A,
        ) -> Result<Self::Value, A::Error> {
            let mut keys = BTreeMap::new();
            while let Some((id, digest)) = map.next_entry()? {
                if keys.insert(id, digest).is_some() {
                    return Err(serde::de::Error::custom(format!(
                        "duplicate fingerprint for gold signal ID {id}"
                    )));
                }
            }
            Ok(keys)
        }
    }
    de.deserialize_map(Keys)
}

impl EvalPairing {
    /// SHA-256 of the original evaluation text, shared with saved prediction receipts.
    pub fn source_sha256(&self) -> &str {
        &self.source_sha256
    }

    /// Build the identity block for an evaluation. The source digest is also
    /// the default document ID when the producer has no explicit occurrence ID.
    pub fn from_comparison(
        cmp: &EvalComparison,
        document_id: Option<&str>,
    ) -> Result<Self, String> {
        let source = source_sha256(&cmp.text);
        let document_id = document_id.unwrap_or(&source);
        check_label(document_id, "document ID")?;
        let mut gold_keys = BTreeMap::new();
        for signal in &cmp.gold {
            let coverage = signal.location.coverage();
            let canonical = serde_json::to_string(&(signal.label(), coverage.view().ranges()))
                .map_err(|e| e.to_string())?;
            if gold_keys
                .insert(signal.id, source_sha256(&canonical))
                .is_some()
            {
                return Err(format!("duplicate gold signal ID {}", signal.id));
            }
        }
        Ok(Self {
            schema: EVAL_PAIRING_SCHEMA.into(),
            document_id: document_id.to_owned(),
            source_sha256: source,
            gold_keys,
        })
    }
}

#[derive(Deserialize)]
struct EvalDocument {
    pairing: Option<EvalPairing>,
    elapsed_ms: Option<f64>,
    #[serde(flatten)]
    report: Report,
}

struct Case {
    source: String,
    // Sorting preserves every occurrence, including equivalent duplicate annotations.
    gold: Vec<String>,
    exact: BTreeMap<String, u64>,
    correct: u64,
    unmatched: u64,
    elapsed_ms: Option<f64>,
}

impl EvalDocument {
    fn checked(self) -> Result<(String, Case), String> {
        let pairing = self.pairing.ok_or(
            "missing pairing identities. Please regenerate both inputs with anno eval --json",
        )?;
        if pairing.schema != EVAL_PAIRING_SCHEMA {
            return Err(format!("expected pairing schema {EVAL_PAIRING_SCHEMA}"));
        }
        check_label(&pairing.document_id, "document ID")?;
        let hash = |value: &str| {
            value.len() == 64
                && value
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
        };
        if !hash(&pairing.source_sha256) || !pairing.gold_keys.values().all(|value| hash(value)) {
            return Err("invalid source or gold SHA-256 fingerprint".into());
        }
        if self
            .elapsed_ms
            .is_some_and(|ms| !ms.is_finite() || ms < 0.0)
        {
            return Err("elapsed_ms must be finite and nonnegative".into());
        }
        if self.report.gold_count != pairing.gold_keys.len() {
            return Err("gold count differs from the number of fingerprinted gold IDs".into());
        }
        let mut exact = BTreeMap::new();
        for event in &self.report.matches {
            let id = match event {
                EvalMatch::Correct { gold_id, .. }
                | EvalMatch::Missed { gold_id }
                | EvalMatch::BoundaryError { gold_id, .. }
                | EvalMatch::TypeMismatch { gold_id, .. } => gold_id,
                EvalMatch::Spurious { .. } => continue,
            };
            let key = pairing
                .gold_keys
                .get(id)
                .ok_or("match refers to an unknown gold ID")?;
            if matches!(event, EvalMatch::Correct { .. }) {
                *exact.entry(key.clone()).or_insert(0) += 1;
            }
        }
        let correct = self.report.correct as u64;
        let predicted = self.report.predicted_count as u64;
        // Use the same one-to-one match, source, and count checks as the native error view.
        EvalErrors::from_report(self.report)?;
        let mut gold: Vec<_> = pairing.gold_keys.into_values().collect();
        gold.sort_unstable();
        Ok((
            pairing.document_id,
            Case {
                source: pairing.source_sha256,
                gold,
                exact,
                correct,
                unmatched: predicted - correct,
                elapsed_ms: self.elapsed_ms,
            },
        ))
    }
}

fn cases(
    input: impl Read,
    side: &'static str,
) -> impl Iterator<Item = Result<(String, Case), String>> {
    serde_json::Deserializer::from_reader(input)
        .into_iter::<EvalDocument>()
        .enumerate()
        .map(move |(i, result)| {
            result
                .map_err(|e| e.to_string())
                .and_then(EvalDocument::checked)
                .map_err(|e| format!("{side} report {}: {e}", i + 1))
        })
}

fn accumulate(total: &mut u64, count: u64) -> Result<(), String> {
    *total = total.checked_add(count).ok_or("pooled count exceeds u64")?;
    Ok(())
}

#[derive(Default)]
struct Totals {
    arm: Arm,
    times: Vec<f64>,
}

impl Totals {
    fn add(&mut self, doc: &Case) -> Result<(), String> {
        accumulate(&mut self.arm.correct, doc.correct)?;
        accumulate(&mut self.arm.unmatched, doc.unmatched)?;
        self.times.extend(doc.elapsed_ms);
        Ok(())
    }

    fn finish(mut self, observations: u64) -> Arm {
        // Only report a cohort median when every observation has a measured runtime.
        if !self.times.is_empty() && self.times.len() as u64 == observations {
            self.times.sort_by(f64::total_cmp);
            let lower = self.times[(self.times.len() - 1) / 2];
            let upper = self.times[self.times.len() / 2];
            self.arm.median_ms = Some(lower + (upper - lower) / 2.0);
        }
        self.arm
    }
}

impl PairedRow {
    /// Reduce two whitespace-separated streams of native evaluation reports to
    /// an entity comparison. Document order and local gold IDs may differ.
    /// Every ID, source edition, and reference multiset must pair exactly.
    /// Matching one of two interchangeable duplicate gold occurrences counts
    /// once. A document can contribute both gains and losses.
    pub fn from_eval_reports(
        label: impl Into<String>,
        role: impl Into<String>,
        baseline: impl Read,
        candidate: impl Read,
    ) -> Result<Self, String> {
        let mut by_id = BTreeMap::new();
        for item in cases(baseline, "baseline") {
            let (id, doc) = item?;
            use std::collections::btree_map::Entry;
            match by_id.entry(id) {
                Entry::Vacant(entry) => {
                    entry.insert(doc);
                }
                Entry::Occupied(entry) => {
                    return Err(format!("duplicate baseline document ID {}", entry.key()))
                }
            }
        }
        if by_id.is_empty() {
            return Err("expected at least one baseline eval report".into());
        }
        let observations = by_id.len() as u64;
        let (mut base, mut trial) = (Totals::default(), Totals::default());
        let (mut gained, mut lost) = (0, 0);
        for item in cases(candidate, "candidate") {
            let (id, new) = item?;
            let old = by_id
                .remove(&id)
                .ok_or_else(|| format!("duplicate or unknown candidate document ID {id}"))?;
            if old.source != new.source || old.gold != new.gold {
                return Err(format!(
                    "source or gold reference differs for document {id}"
                ));
            }
            let retained: u64 = old
                .exact
                .iter()
                .map(|(key, &count)| count.min(new.exact.get(key).copied().unwrap_or(0)))
                .sum();
            accumulate(&mut gained, new.correct - retained)?;
            accumulate(&mut lost, old.correct - retained)?;
            base.add(&old)?;
            trial.add(&new)?;
        }
        if let Some((missing, _)) = by_id.first_key_value() {
            return Err(format!("candidate is missing baseline document {missing}"));
        }
        let row = Self {
            label: label.into(),
            role: role.into(),
            observations,
            baseline: base.finish(observations),
            candidate: trial.finish(observations),
            gained,
            lost,
            interval: None,
        };
        check_label(&row.label, "cohort label")?;
        check_label(&row.role, "cohort role")?;
        row.validate()?;
        Ok(row)
    }
}
