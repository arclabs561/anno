// SPDX-License-Identifier: MIT OR Apache-2.0
//! Standalone candidate retrieval for the ROR study. Input names are normalized by the driver.
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::io::{self, BufRead, BufWriter, Write};

#[path = "context/masked.rs"]
mod masked;

fn trigrams(name: &str) -> Vec<u64> {
    let chars: Vec<_> = [' ', ' ']
        .into_iter()
        .chain(name.chars())
        .chain([' ', ' '])
        .collect();
    let mut result: Vec<_> = chars
        .windows(3)
        .map(|g| (u64::from(g[0]) << 42) | (u64::from(g[1]) << 21) | u64::from(g[2]))
        .collect();
    result.sort_unstable();
    result.dedup();
    result
}

#[derive(Default)]
struct Index {
    ids: Vec<String>,
    aliases: HashMap<String, usize>,
    surfaces: Vec<Surface>,
    postings: HashMap<u64, Vec<usize>>,
}

struct Surface {
    grams: Vec<u64>,
    owners: Vec<(usize, bool)>,
}

#[derive(Clone, Copy)]
struct Score {
    numerator: usize,
    denominator: usize,
}

impl Score {
    fn compare(self, other: Self) -> Ordering {
        ((self.numerator as u128) * other.denominator as u128)
            .cmp(&((other.numerator as u128) * self.denominator as u128))
    }
}

impl Index {
    fn add(&mut self, id: &str, display: &str, names: impl Iterator<Item = impl AsRef<str>>) {
        let owner = self.ids.len();
        self.ids.push(id.to_owned());
        // The driver removes repeated normalized surfaces for a single owner.
        for name in names {
            let name = name.as_ref();
            if name.is_empty() {
                continue;
            }
            let surface_id = *self.aliases.entry(name.to_owned()).or_insert_with(|| {
                let surface_id = self.surfaces.len();
                let grams = trigrams(name);
                for g in &grams {
                    self.postings.entry(*g).or_default().push(surface_id);
                }
                self.surfaces.push(Surface {
                    grams,
                    owners: Vec::new(),
                });
                surface_id
            });
            self.surfaces[surface_id]
                .owners
                .push((owner, name == display));
        }
    }

    fn query(
        &self,
        name: &str,
        excluded: &str,
        mask: Option<&masked::Mask>,
        out: &mut impl Write,
    ) -> io::Result<()> {
        let eligible = |owner: usize| {
            self.ids[owner] != excluded && mask.is_none_or(|m| !m.excludes(owner))
        };
        let mut candidates = HashSet::new();
        let mut display = Vec::new();
        let mut exact = Vec::new();
        if let Some(&a) = self.aliases.get(name) {
            candidates.insert(a);
            for &(owner, is_display) in &self.surfaces[a].owners {
                if eligible(owner) {
                    exact.push(self.ids[owner].as_str());
                    if is_display {
                        display.push(self.ids[owner].as_str());
                    }
                }
            }
        }
        display.sort_unstable();
        exact.sort_unstable();

        let query = trigrams(name);
        let mut rare: Vec<_> = query
            .iter()
            .filter_map(|g| {
                let posting = self.postings.get(g)?;
                let length = mask.map_or(posting.len(), |m| m.frequency(*g, posting.len()));
                (1..=2048).contains(&length).then_some((length, *g, posting))
            })
            .collect();
        rare.sort_unstable_by_key(|(length, gram, _)| (*length, *gram));
        for (_, _, posting) in rare.into_iter().take(3) {
            candidates.extend(posting);
        }
        let mut scores: HashMap<usize, Score> = HashMap::new();
        for surface_id in candidates {
            let surface = &self.surfaces[surface_id];
            let (mut a, mut b, mut overlap) = (0, 0, 0);
            while a < query.len() && b < surface.grams.len() {
                match query[a].cmp(&surface.grams[b]) {
                    Ordering::Less => a += 1,
                    Ordering::Greater => b += 1,
                    Ordering::Equal => {
                        overlap += 1;
                        a += 1;
                        b += 1;
                    }
                }
            }
            let score = Score {
                numerator: 2 * overlap,
                denominator: query.len() + surface.grams.len(),
            };
            for &(owner, _) in &surface.owners {
                if eligible(owner) {
                    let best = scores.entry(owner).or_insert(score);
                    if best.compare(score).is_lt()
                        || (best.compare(score).is_eq() && score.denominator < best.denominator)
                    {
                        *best = score;
                    }
                }
            }
        }
        let count = scores.len();
        let mut blocked: Vec<_> = scores
            .keys()
            .map(|owner| self.ids[*owner].as_str())
            .collect();
        blocked.sort_unstable();
        let rank = |a: &(usize, Score), b: &(usize, Score)| {
            b.1.compare(a.1)
                .then_with(|| self.ids[a.0].cmp(&self.ids[b.0]))
        };
        let mut ranked: Vec<_> = scores.into_iter().collect();
        if ranked.len() > 5 {
            ranked.select_nth_unstable_by(5, rank);
            ranked.truncate(5);
        }
        ranked.sort_unstable_by(rank);
        write!(
            out,
            "{}\t{}\t{}\t",
            display.join(","),
            exact.join(","),
            count
        )?;
        for (i, (owner, score)) in ranked.iter().enumerate() {
            if i > 0 {
                write!(out, ",")?;
            }
            write!(
                out,
                "{}:{}:{}",
                self.ids[*owner], score.numerator, score.denominator
            )?;
        }
        writeln!(out, "\t{}", blocked.join(","))
    }
}

fn main() -> io::Result<()> {
    let mut index = Index::default();
    let mut owner_surfaces = None;
    let mut out = BufWriter::new(io::stdout().lock());
    for line in io::stdin().lock().lines() {
        let line = line?;
        let mut fields = line.split('\t');
        match fields.next() {
            Some("I") => {
                let id = fields.next().expect("catalog accession");
                let display = fields.next().expect("normalized display name");
                index.add(id, display, fields);
                owner_surfaces = None;
            }
            Some("Q") => {
                let excluded = fields.next().expect("excluded accession or empty");
                index.query(fields.next().expect("query"), excluded, None, &mut out)?;
            }
            Some("M") => {
                let ids = fields.next().expect("masked accessions");
                let owners = owner_surfaces
                    .get_or_insert_with(|| masked::OwnerSurfaces::new(&index));
                let mask = owners.without(&index, ids.split(','));
                index.query(fields.next().expect("query"), "", Some(&mask), &mut out)?;
            }
            _ => panic!("unrecognized study row"),
        }
    }
    out.flush()
}
