//! Union-find clustering for coreference resolution.
//!
//! Converts antecedent predictions (each mention's best preceding coreferent)
//! into clusters of co-referring mentions using a disjoint-set data structure.

use super::super::resolve::CorefCluster;

/// Disjoint-set (union-find) with path compression and union by rank.
pub(crate) struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    pub fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    pub fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    pub fn union(&mut self, x: usize, y: usize) {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return;
        }
        match self.rank[rx].cmp(&self.rank[ry]) {
            std::cmp::Ordering::Less => self.parent[rx] = ry,
            std::cmp::Ordering::Greater => self.parent[ry] = rx,
            std::cmp::Ordering::Equal => {
                self.parent[ry] = rx;
                self.rank[rx] += 1;
            }
        }
    }
}

/// A mention with its character span in the original text.
#[derive(Debug, Clone)]
pub(crate) struct MentionSpan {
    /// Token-level start index.
    pub token_start: usize,
    /// Token-level end index (inclusive).
    pub token_end: usize,
    /// Character-level start offset.
    pub char_start: usize,
    /// Character-level end offset.
    pub char_end: usize,
    /// The mention text.
    pub text: String,
}

/// Build coreference clusters from antecedent predictions.
///
/// # Arguments
///
/// * `mentions` - The top-k mentions with their spans
/// * `antecedents` - For each mention i, the index of its best antecedent (or i if no antecedent)
///
/// # Returns
///
/// Clusters with 2+ members, sorted by cluster ID. Singletons are filtered out.
pub(crate) fn build_clusters(mentions: &[MentionSpan], antecedents: &[usize]) -> Vec<CorefCluster> {
    let n = mentions.len();
    if n == 0 {
        return vec![];
    }

    let mut uf = UnionFind::new(n);

    // Link each mention to its antecedent
    for (i, &ante) in antecedents.iter().enumerate() {
        if ante != i {
            uf.union(i, ante);
        }
    }

    // Group mentions by cluster root
    let mut cluster_map: std::collections::HashMap<usize, Vec<usize>> =
        std::collections::HashMap::new();
    for i in 0..n {
        let root = uf.find(i);
        cluster_map.entry(root).or_default().push(i);
    }

    // Build CorefCluster objects, filtering singletons
    let mut clusters: Vec<CorefCluster> = cluster_map
        .into_values()
        .filter(|members| members.len() > 1)
        .enumerate()
        .map(|(id, member_indices)| {
            let mention_texts: Vec<String> = member_indices
                .iter()
                .map(|&i| mentions[i].text.clone())
                .collect();
            let spans: Vec<(usize, usize)> = member_indices
                .iter()
                .map(|&i| (mentions[i].char_start, mentions[i].char_end))
                .collect();

            // Canonical = longest mention
            let canonical = mention_texts
                .iter()
                .max_by_key(|m| m.len())
                .cloned()
                .unwrap_or_default();

            CorefCluster {
                id: id as u32,
                mentions: mention_texts,
                spans,
                canonical,
            }
        })
        .collect();

    clusters.sort_by_key(|c| c.spans.first().map(|s| s.0).unwrap_or(0));
    // Re-assign IDs after sorting
    for (i, c) in clusters.iter_mut().enumerate() {
        c.id = i as u32;
    }
    clusters
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn union_find_basic() {
        let mut uf = UnionFind::new(5);
        uf.union(0, 1);
        uf.union(2, 3);
        assert_eq!(uf.find(0), uf.find(1));
        assert_ne!(uf.find(0), uf.find(2));
        uf.union(1, 3);
        assert_eq!(uf.find(0), uf.find(3));
    }

    #[test]
    fn union_find_singleton() {
        let mut uf = UnionFind::new(3);
        assert_eq!(uf.find(0), 0);
        assert_eq!(uf.find(1), 1);
        assert_eq!(uf.find(2), 2);
    }

    #[test]
    fn build_clusters_filters_singletons() {
        let mentions = vec![
            MentionSpan {
                token_start: 0,
                token_end: 1,
                char_start: 0,
                char_end: 4,
                text: "John".into(),
            },
            MentionSpan {
                token_start: 5,
                token_end: 5,
                char_start: 20,
                char_end: 22,
                text: "He".into(),
            },
            MentionSpan {
                token_start: 10,
                token_end: 11,
                char_start: 40,
                char_end: 44,
                text: "Mary".into(),
            },
        ];
        // John and He are coreferent; Mary is singleton
        let antecedents = vec![0, 0, 2];
        let clusters = build_clusters(&mentions, &antecedents);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].mentions.len(), 2);
        assert!(clusters[0].mentions.contains(&"John".to_string()));
        assert!(clusters[0].mentions.contains(&"He".to_string()));
        assert_eq!(clusters[0].canonical, "John");
    }

    #[test]
    fn build_clusters_empty() {
        let clusters = build_clusters(&[], &[]);
        assert!(clusters.is_empty());
    }

    #[test]
    fn build_clusters_all_singletons() {
        let mentions = vec![
            MentionSpan {
                token_start: 0,
                token_end: 0,
                char_start: 0,
                char_end: 3,
                text: "one".into(),
            },
            MentionSpan {
                token_start: 1,
                token_end: 1,
                char_start: 4,
                char_end: 7,
                text: "two".into(),
            },
        ];
        let antecedents = vec![0, 1]; // each points to self
        let clusters = build_clusters(&mentions, &antecedents);
        assert!(clusters.is_empty());
    }

    #[test]
    fn build_clusters_long_chain() {
        let mentions: Vec<MentionSpan> = (0..4)
            .map(|i| MentionSpan {
                token_start: i * 3,
                token_end: i * 3 + 1,
                char_start: i * 10,
                char_end: i * 10 + 5,
                text: format!("m{}", i),
            })
            .collect();
        // Chain: m0 <- m1 <- m2 <- m3
        let antecedents = vec![0, 0, 1, 2];
        let clusters = build_clusters(&mentions, &antecedents);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].mentions.len(), 4);
    }

    #[test]
    fn build_clusters_two_clusters() {
        let mentions = vec![
            MentionSpan {
                token_start: 0,
                token_end: 0,
                char_start: 0,
                char_end: 5,
                text: "Alice".into(),
            },
            MentionSpan {
                token_start: 2,
                token_end: 2,
                char_start: 10,
                char_end: 13,
                text: "Bob".into(),
            },
            MentionSpan {
                token_start: 5,
                token_end: 5,
                char_start: 20,
                char_end: 23,
                text: "She".into(),
            },
            MentionSpan {
                token_start: 8,
                token_end: 8,
                char_start: 30,
                char_end: 32,
                text: "He".into(),
            },
        ];
        // Alice-She, Bob-He
        let antecedents = vec![0, 1, 0, 1];
        let clusters = build_clusters(&mentions, &antecedents);
        assert_eq!(clusters.len(), 2);
    }
}
