// SPDX-License-Identifier: MIT OR Apache-2.0
//! Per-query catalog removal for the withheld-target experiment.
use super::Index;
use std::collections::{HashMap, HashSet};

pub(super) struct OwnerSurfaces {
    owners: HashMap<String, usize>,
    surfaces: Vec<Vec<usize>>,
}

impl OwnerSurfaces {
    pub(super) fn new(index: &Index) -> Self {
        let owners = index
            .ids
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, id)| (id, i))
            .collect();
        let mut surfaces = vec![Vec::new(); index.ids.len()];
        for (sid, surface) in index.surfaces.iter().enumerate() {
            for &(owner, _) in &surface.owners {
                surfaces[owner].push(sid);
            }
        }
        Self { owners, surfaces }
    }

    pub(super) fn without<'a>(&self, index: &Index, ids: impl Iterator<Item = &'a str>) -> Mask {
        let excluded: HashSet<_> = ids.filter_map(|id| self.owners.get(id).copied()).collect();
        let affected: HashSet<_> = excluded
            .iter()
            .flat_map(|&i| self.surfaces[i].iter().copied())
            .collect();
        let mut removed_frequencies = HashMap::new();
        for sid in affected {
            let surface = &index.surfaces[sid];
            if surface.owners.iter().all(|(i, _)| excluded.contains(i)) {
                for &gram in &surface.grams {
                    *removed_frequencies.entry(gram).or_insert(0) += 1;
                }
            }
        }
        Mask {
            excluded,
            removed_frequencies,
        }
    }
}

pub(super) struct Mask {
    excluded: HashSet<usize>,
    removed_frequencies: HashMap<u64, usize>,
}

impl Mask {
    pub(super) fn excludes(&self, owner: usize) -> bool {
        self.excluded.contains(&owner)
    }

    pub(super) fn frequency(&self, gram: u64, full: usize) -> usize {
        full - self.removed_frequencies.get(&gram).copied().unwrap_or(0)
    }
}
