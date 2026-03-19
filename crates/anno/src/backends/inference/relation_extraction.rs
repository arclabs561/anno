//! Relation extraction: configuration, trigger detection, and the main heuristic pipeline.
//!
//! Used by TPLinker and other relation-capable backends.

use super::registry::{LabelDefinition, SemanticRegistry};
use super::RelationTriple;
use crate::{Confidence, Entity, EntityType};
use anno_core::Relation;

/// Configuration for relation extraction.
#[derive(Debug, Clone)]
pub struct RelationExtractionConfig {
    /// Maximum token distance between head and tail
    pub max_span_distance: usize,
    /// Minimum confidence for relation
    pub threshold: Confidence,
    /// Whether to extract relation triggers
    pub extract_triggers: bool,
}

impl Default for RelationExtractionConfig {
    fn default() -> Self {
        Self {
            max_span_distance: 50,
            threshold: Confidence::new(0.5),
            extract_triggers: true,
        }
    }
}

/// Extract relations between entities.
///
/// # Algorithm (Two-Pass)
///
/// 1. Run entity NER to find all entity mentions
/// 2. For each entity pair within distance threshold:
///    - Encode the span between them
///    - Match against relation type embeddings
///    - Optionally identify trigger span
///
/// # Returns
///
/// Relations with head/tail entities and optional trigger spans.
pub fn extract_relations(
    entities: &[Entity],
    text: &str,
    registry: &SemanticRegistry,
    config: &RelationExtractionConfig,
) -> Vec<Relation> {
    let mut relations = Vec::new();
    // `Entity` spans in anno are character offsets, but slicing a Rust `&str` requires byte
    // offsets. Build a converter once so we can safely slice and map trigger spans back.
    let span_converter = crate::offset::SpanConverter::new(text);

    // Get relation labels
    let relation_labels: Vec<_> = registry.relation_labels().collect();
    if relation_labels.is_empty() {
        return relations;
    }

    // Check all entity pairs
    for (i, head) in entities.iter().enumerate() {
        for (j, tail) in entities.iter().enumerate() {
            if i == j {
                continue;
            }

            // Check distance
            let distance = if head.end() <= tail.start() {
                tail.start() - head.end()
            } else {
                head.start().saturating_sub(tail.end())
            };

            if distance > config.max_span_distance {
                continue;
            }

            // Look for relation triggers in the text between entities
            let (span_start, span_end) = if head.end() <= tail.start() {
                (head.end(), tail.start())
            } else {
                (tail.end(), head.start())
            };

            let between_span = span_converter.from_chars(span_start, span_end);
            let between_text = text
                .get(between_span.byte_start..between_span.byte_end)
                .unwrap_or("");

            // Simple heuristic: check for common relation indicators
            let relation_type = detect_relation_type(head, tail, between_text, &relation_labels);

            if let Some((rel_type, mut confidence, trigger)) = relation_type {
                // Apply distance penalty: closer entities are more likely to be related
                // Confidence decays linearly from 1.0 at distance 0 to 0.5 at max_span_distance
                let distance_penalty = if distance < config.max_span_distance {
                    let penalty_factor =
                        1.0 - (distance as f64 / config.max_span_distance as f64) * 0.5;
                    penalty_factor.max(0.5) // Minimum 0.5 confidence even at max distance
                } else {
                    0.5 // At or beyond max distance, apply minimum confidence
                };
                confidence *= distance_penalty;

                if confidence < config.threshold.value() {
                    continue;
                }

                // `detect_relation_type` returns byte offsets into `between_text`.
                let trigger_span = if config.extract_triggers {
                    trigger.map(|(s, e)| {
                        let trigger_start_byte = between_span.byte_start.saturating_add(s);
                        let trigger_end_byte = between_span.byte_start.saturating_add(e);
                        (
                            span_converter.byte_to_char(trigger_start_byte),
                            span_converter.byte_to_char(trigger_end_byte),
                        )
                    })
                } else {
                    None
                };

                relations.push(Relation {
                    head: head.clone(),
                    tail: tail.clone(),
                    relation_type: rel_type.to_string(),
                    trigger_span,
                    confidence: Confidence::new(confidence.clamp(0.0, 1.0)),
                });
            }
        }
    }

    relations
}

/// Extract relations as index-based triples (for joint extraction backends).
///
/// This is the same heuristic logic as [`extract_relations`], but returns
/// [`RelationTriple`] with indices into the provided `entities` slice.
///
/// Notes:
/// - Entity spans are **character offsets**.
/// - Trigger spans are not currently exposed in `RelationTriple`.
#[must_use]
pub fn extract_relation_triples(
    entities: &[Entity],
    text: &str,
    registry: &SemanticRegistry,
    config: &RelationExtractionConfig,
) -> Vec<RelationTriple> {
    let mut triples = Vec::new();
    if entities.len() < 2 {
        return triples;
    }

    // `Entity` spans are character offsets; slicing needs byte offsets.
    let span_converter = crate::offset::SpanConverter::new(text);

    let relation_labels: Vec<_> = registry.relation_labels().collect();
    if relation_labels.is_empty() {
        return triples;
    }

    for (i, head) in entities.iter().enumerate() {
        for (j, tail) in entities.iter().enumerate() {
            if i == j {
                continue;
            }

            // Skip overlapping spans (avoids self-nesting artifacts like "New York" vs "York").
            if head.start() < tail.end() && tail.start() < head.end() {
                continue;
            }

            // Check distance (character offsets)
            let distance = if head.end() <= tail.start() {
                tail.start() - head.end()
            } else {
                head.start().saturating_sub(tail.end())
            };
            if distance > config.max_span_distance {
                continue;
            }

            let (span_start, span_end) = if head.end() <= tail.start() {
                (head.end(), tail.start())
            } else {
                (tail.end(), head.start())
            };

            let between_span = span_converter.from_chars(span_start, span_end);
            let between_text = text
                .get(between_span.byte_start..between_span.byte_end)
                .unwrap_or("");

            if let Some((rel_type, mut confidence, _trigger)) =
                detect_relation_type(head, tail, between_text, &relation_labels)
            {
                // Apply distance penalty (same logic as extract_relations)
                let distance_penalty = if distance < config.max_span_distance {
                    let penalty_factor =
                        1.0 - (distance as f64 / config.max_span_distance as f64) * 0.5;
                    penalty_factor.max(0.5)
                } else {
                    0.5
                };
                confidence *= distance_penalty;

                if confidence < config.threshold.value() {
                    continue;
                }

                triples.push(RelationTriple {
                    head_idx: i,
                    tail_idx: j,
                    relation_type: rel_type.to_string(),
                    confidence: Confidence::new(confidence),
                });
            }
        }
    }

    triples
}

/// Result of relation detection: (label, confidence, optional span).
type RelationMatch<'a> = (&'a str, f64, Option<(usize, usize)>);

/// Detect relation type from context (heuristic fallback).
fn detect_relation_type<'a>(
    head: &Entity,
    tail: &Entity,
    between_text: &str,
    relation_labels: &[&'a LabelDefinition],
) -> Option<RelationMatch<'a>> {
    // Use Unicode-aware lowercasing for multilingual support
    // Note: For CJK languages, case doesn't apply, but this is safe
    let between_lower = between_text.to_lowercase();

    // Normalize relation slugs so datasets that use kebab-case / colon-separated schemas
    // (e.g. DocRED: "part-of", "general-affiliation") can match our canonical patterns
    // (e.g. "PART_OF", "GENERAL_AFFILIATION").
    fn norm_rel_slug(s: &str) -> String {
        // Uppercase + map non-alphanumerics to '_' so we can compare across naming schemes.
        let mut out = String::with_capacity(s.len());
        let mut prev_underscore = false;
        for ch in s.chars() {
            if ch.is_alphanumeric() {
                // Keep Unicode letters/digits; uppercase ASCII for stable matching.
                if ch.is_ascii_alphabetic() {
                    out.push(ch.to_ascii_uppercase());
                } else {
                    out.push(ch);
                }
                prev_underscore = false;
            } else if !prev_underscore {
                out.push('_');
                prev_underscore = true;
            }
        }
        while out.starts_with('_') {
            out.remove(0);
        }
        while out.ends_with('_') {
            out.pop();
        }
        out
    }

    // Common patterns: (relation_slug, triggers, confidence)
    struct RelPattern {
        slug: &'static str,
        triggers: &'static [&'static str],
        confidence: f64,
    }

    let patterns: &[RelPattern] = &[
        // Employment relations
        RelPattern {
            slug: "CEO_OF",
            triggers: &[
                "ceo of",
                "chief executive",
                "chief executive officer",
                "leads",
                "founded",
                "founder of",
            ],
            confidence: 0.8,
        },
        RelPattern {
            slug: "WORKS_FOR",
            triggers: &[
                "works for",
                "works at",
                "employed by",
                "employee of",
                "works with",
                "staff at",
                "member of",
            ],
            confidence: 0.7,
        },
        RelPattern {
            slug: "FOUNDED",
            triggers: &[
                "founded",
                "co-founded",
                "cofounder",
                "started",
                "established",
                "created",
                "launched",
            ],
            confidence: 0.8,
        },
        RelPattern {
            slug: "MANAGES",
            triggers: &[
                "manages",
                "managing",
                "oversees",
                "directs",
                "supervises",
                "runs",
            ],
            confidence: 0.75,
        },
        RelPattern {
            slug: "REPORTS_TO",
            triggers: &["reports to", "reported to", "under", "reports directly to"],
            confidence: 0.7,
        },
        // Location relations
        // NOTE: bare "in" / "at" removed -- they match nearly any between-text and
        // produce nonsensical relations like LOCATED_IN(Doudna, Chemistry).
        RelPattern {
            slug: "LOCATED_IN",
            triggers: &[
                "based in",
                "located in",
                "headquartered in",
                "situated in",
                "found in",
                "offices in",
            ],
            confidence: 0.6,
        },
        RelPattern {
            slug: "BORN_IN",
            triggers: &[
                "born in",
                "native of",
                "from",
                "hails from",
                "originated in",
            ],
            confidence: 0.7,
        },
        RelPattern {
            slug: "LIVES_IN",
            triggers: &["lives in", "resides in", "living in", "based in"],
            confidence: 0.65,
        },
        RelPattern {
            slug: "DIED_IN",
            triggers: &["died in", "passed away in", "deceased in"],
            confidence: 0.8,
        },
        // Temporal relations
        RelPattern {
            slug: "OCCURRED_ON",
            triggers: &["occurred on", "happened on", "took place on", "dated"],
            confidence: 0.6,
        },
        RelPattern {
            slug: "STARTED_ON",
            triggers: &["started on", "began on", "commenced on", "initiated on"],
            confidence: 0.7,
        },
        RelPattern {
            slug: "ENDED_ON",
            triggers: &["ended on", "concluded on", "finished on", "completed on"],
            confidence: 0.7,
        },
        // Organizational relations
        RelPattern {
            slug: "PART_OF",
            triggers: &[
                "part of",
                "member of",
                "belongs to",
                "subsidiary of",
                "division of",
                "branch of",
            ],
            confidence: 0.7,
        },
        RelPattern {
            slug: "ACQUIRED",
            triggers: &[
                "acquired",
                "bought",
                "purchased",
                "took over",
                "merged with",
            ],
            confidence: 0.75,
        },
        RelPattern {
            slug: "MERGED_WITH",
            triggers: &["merged with", "merged into", "combined with", "joined with"],
            confidence: 0.8,
        },
        RelPattern {
            slug: "PARENT_OF",
            triggers: &["parent of", "parent company of", "owns", "owner of"],
            confidence: 0.75,
        },
        // Social relations
        RelPattern {
            slug: "MARRIED_TO",
            triggers: &["married to", "wed to", "spouse of", "husband of", "wife of"],
            confidence: 0.85,
        },
        RelPattern {
            slug: "CHILD_OF",
            triggers: &["son of", "daughter of", "child of", "offspring of"],
            confidence: 0.8,
        },
        RelPattern {
            slug: "SIBLING_OF",
            triggers: &["brother of", "sister of", "sibling of"],
            confidence: 0.8,
        },
        // Academic/Professional
        RelPattern {
            slug: "STUDIED_AT",
            triggers: &[
                "studied at",
                "attended",
                "graduated from",
                "alumni of",
                "educated at",
            ],
            confidence: 0.75,
        },
        RelPattern {
            slug: "TEACHES_AT",
            triggers: &["teaches at", "professor at", "instructor at", "faculty at"],
            confidence: 0.8,
        },
        // Product/Service relations
        RelPattern {
            slug: "DEVELOPS",
            triggers: &[
                "develops",
                "created",
                "built",
                "designed",
                "produces",
                "manufactures",
            ],
            confidence: 0.7,
        },
        RelPattern {
            slug: "USES",
            triggers: &["uses", "utilizes", "employs", "adopts", "implements"],
            confidence: 0.6,
        },
        // Dataset-style relation labels (DocRED/CHisIEC-like)
        //
        // These are the *coarse* label names we actually see in the CrossRE/DocRED-style
        // exports used by this repo (e.g. `docred_dev.json`), which differ from the
        // “canonical” IE labels above.
        RelPattern {
            slug: "NAMED",
            triggers: &[
                "called",
                "known as",
                "also known as",
                "named",
                "referred to as",
                "nickname",
            ],
            confidence: 0.6,
        },
        RelPattern {
            slug: "TYPE_OF",
            triggers: &[
                "type of",
                "kind of",
                "form of",
                "a type of",
                "is a",
                "are a",
            ],
            confidence: 0.6,
        },
        RelPattern {
            slug: "RELATED_TO",
            triggers: &["related to", "associated with", "connected to", "linked to"],
            confidence: 0.55,
        },
        RelPattern {
            slug: "ORIGIN",
            triggers: &[
                "from",
                "born",
                "originated",
                "created by",
                "invented by",
                "derived from",
                "spinoff",
                "spin-off",
            ],
            confidence: 0.55,
        },
        RelPattern {
            slug: "ROLE",
            triggers: &[
                "president",
                "ceo",
                "chair",
                "director",
                "editor",
                "producer",
                "actor",
                "professor",
                "fellow",
                "member",
            ],
            confidence: 0.55,
        },
        RelPattern {
            slug: "TEMPORAL",
            triggers: &[
                "in 19", "in 20", "during", "before", "after", "between", "until", "since",
            ],
            confidence: 0.5,
        },
        RelPattern {
            slug: "PHYSICAL",
            triggers: &["located in", "based in", "headquartered in", "situated at"],
            confidence: 0.55,
        },
        RelPattern {
            slug: "TOPIC",
            triggers: &[
                "topic",
                "about",
                "regarding",
                "focused on",
                "on the topic of",
            ],
            confidence: 0.5,
        },
        RelPattern {
            slug: "OPPOSITE",
            triggers: &["opposite", "contrasts with", "as opposed to"],
            confidence: 0.6,
        },
        RelPattern {
            slug: "WIN_DEFEAT",
            triggers: &["defeated", "beat", "won", "win", "lose", "lost to"],
            confidence: 0.6,
        },
        RelPattern {
            slug: "CAUSE_EFFECT",
            triggers: &["caused", "causes", "leads to", "results in", "because"],
            confidence: 0.55,
        },
        RelPattern {
            slug: "USAGE",
            triggers: &["use", "uses", "used", "using", "utilize", "employ", "adopt"],
            confidence: 0.55,
        },
        RelPattern {
            slug: "ARTIFACT",
            triggers: &[
                "tool",
                "library",
                "framework",
                "system",
                "artifact",
                "implementation",
            ],
            confidence: 0.55,
        },
        RelPattern {
            slug: "COMPARE",
            triggers: &[
                "compare",
                "compared to",
                "versus",
                "vs",
                "better than",
                "worse than",
            ],
            confidence: 0.55,
        },
        RelPattern {
            slug: "GENERAL_AFFILIATION",
            triggers: &[
                "affiliation",
                "affiliated with",
                "member of",
                "part of",
                "associated with",
            ],
            confidence: 0.55,
        },
        // CHisIEC (classical Chinese) relations (match either simplified or traditional labels)
        RelPattern {
            slug: "父母",
            triggers: &["父", "母", "父母"],
            confidence: 0.7,
        },
        RelPattern {
            slug: "兄弟",
            triggers: &["兄", "弟", "兄弟"],
            confidence: 0.7,
        },
        RelPattern {
            slug: "別名",
            triggers: &["別名", "别名"],
            confidence: 0.75,
        },
        RelPattern {
            slug: "到達",
            triggers: &["到", "至", "達", "到達", "到达"],
            confidence: 0.6,
        },
        RelPattern {
            slug: "出生於某地",
            triggers: &["生於", "生于", "出生於", "出生于"],
            confidence: 0.65,
        },
        RelPattern {
            slug: "任職",
            triggers: &["任", "拜", "任職", "任职"],
            confidence: 0.6,
        },
        RelPattern {
            slug: "管理",
            triggers: &["管", "治", "守", "管理"],
            confidence: 0.55,
        },
        RelPattern {
            slug: "駐守",
            triggers: &["駐", "驻", "守", "駐守", "驻守"],
            confidence: 0.55,
        },
        RelPattern {
            slug: "敵對攻伐",
            triggers: &["敵", "敌", "攻", "伐", "戰", "战"],
            confidence: 0.55,
        },
        RelPattern {
            slug: "同僚",
            triggers: &["同僚"],
            confidence: 0.55,
        },
        RelPattern {
            slug: "政治奧援",
            triggers: &["奧援", "奥援"],
            confidence: 0.55,
        },
        // Communication/Interaction
        RelPattern {
            slug: "MET_WITH",
            triggers: &["met with", "met", "met up with", "encountered", "saw"],
            confidence: 0.65,
        },
        RelPattern {
            slug: "SPOKE_WITH",
            triggers: &[
                "spoke with",
                "talked with",
                "discussed with",
                "conversed with",
            ],
            confidence: 0.7,
        },
        // Ownership
        RelPattern {
            slug: "OWNS",
            triggers: &["owns", "owner of", "possesses", "holds"],
            confidence: 0.75,
        },
        // =========================================================================
        // Multilingual relation triggers
        // =========================================================================
        // Spanish (es)
        RelPattern {
            slug: "WORKS_FOR",
            triggers: &["trabaja en", "trabaja para", "empleado de", "trabaja con"],
            confidence: 0.7,
        },
        RelPattern {
            slug: "FOUNDED",
            triggers: &["fundó", "fundada", "creó", "creada", "estableció", "inició"],
            confidence: 0.8,
        },
        RelPattern {
            slug: "LOCATED_IN",
            triggers: &[
                "ubicado en",
                "situado en",
                "basado en",
                "localizado en",
                "sede en",
            ],
            confidence: 0.6,
        },
        RelPattern {
            slug: "BORN_IN",
            triggers: &["nació en", "nacido en", "originario de", "natural de"],
            confidence: 0.7,
        },
        RelPattern {
            slug: "LIVES_IN",
            triggers: &["cerno en", "reside en", "viviendo en"],
            confidence: 0.65,
        },
        RelPattern {
            slug: "MARRIED_TO",
            triggers: &["casado con", "casada con", "esposo de", "esposa de"],
            confidence: 0.85,
        },
        // French (fr)
        RelPattern {
            slug: "WORKS_FOR",
            triggers: &[
                "travaille pour",
                "travaille à",
                "employé de",
                "travaille avec",
            ],
            confidence: 0.7,
        },
        RelPattern {
            slug: "FOUNDED",
            triggers: &["fondé", "fondée", "créé", "créée", "établi", "établie"],
            confidence: 0.8,
        },
        RelPattern {
            slug: "LOCATED_IN",
            triggers: &[
                "situé en",
                "situé à",
                "basé en",
                "basé à",
                "localisé en",
                "siège à",
            ],
            confidence: 0.6,
        },
        RelPattern {
            slug: "BORN_IN",
            triggers: &["né en", "née en", "originaire de", "natif de"],
            confidence: 0.7,
        },
        RelPattern {
            slug: "LIVES_IN",
            triggers: &["vit en", "réside en", "vivant en"],
            confidence: 0.65,
        },
        RelPattern {
            slug: "MARRIED_TO",
            triggers: &["marié avec", "mariée avec", "époux de", "épouse de"],
            confidence: 0.85,
        },
        // German (de)
        RelPattern {
            slug: "WORKS_FOR",
            triggers: &[
                "arbeitet für",
                "arbeitet bei",
                "angestellt bei",
                "arbeitet mit",
            ],
            confidence: 0.7,
        },
        RelPattern {
            slug: "FOUNDED",
            triggers: &[
                "gegründet",
                "gründete",
                "erstellt",
                "errichtet",
                "etabliert",
            ],
            confidence: 0.8,
        },
        RelPattern {
            slug: "LOCATED_IN",
            triggers: &[
                "situiert in",
                "basiert in",
                "befindet sich in",
                "ansässig in",
                "sitz in",
            ],
            confidence: 0.6,
        },
        RelPattern {
            slug: "BORN_IN",
            triggers: &["geboren in", "geboren am", "stammt aus", "gebürtig aus"],
            confidence: 0.7,
        },
        RelPattern {
            slug: "LIVES_IN",
            triggers: &["lebt in", "wohnt in", "lebend in"],
            confidence: 0.65,
        },
        RelPattern {
            slug: "MARRIED_TO",
            triggers: &["verheiratet mit", "ehemann von", "ehefrau von"],
            confidence: 0.85,
        },
        // Chinese (zh) - Simplified
        RelPattern {
            slug: "WORKS_FOR",
            triggers: &["为", "在", "工作于", "就职于", "任职于"],
            confidence: 0.7,
        },
        RelPattern {
            slug: "FOUNDED",
            triggers: &["创立", "创建", "建立", "成立", "创办"],
            confidence: 0.8,
        },
        RelPattern {
            slug: "LOCATED_IN",
            triggers: &["在", "位于", "坐落于", "地处"],
            confidence: 0.6,
        },
        RelPattern {
            slug: "BORN_IN",
            triggers: &["出生于", "生于", "来自", "出生于"],
            confidence: 0.7,
        },
        RelPattern {
            slug: "LIVES_IN",
            triggers: &["居住于", "住在", "生活在"],
            confidence: 0.65,
        },
        RelPattern {
            slug: "MARRIED_TO",
            triggers: &["与...结婚", "嫁给", "娶了"],
            confidence: 0.85,
        },
        // Japanese (ja)
        RelPattern {
            slug: "WORKS_FOR",
            triggers: &["で働く", "に勤務", "に所属", "で就職"],
            confidence: 0.7,
        },
        RelPattern {
            slug: "FOUNDED",
            triggers: &["設立", "創立", "設立した", "創設"],
            confidence: 0.8,
        },
        RelPattern {
            slug: "LOCATED_IN",
            triggers: &["に", "で", "に位置", "に所在"],
            confidence: 0.6,
        },
        RelPattern {
            slug: "BORN_IN",
            triggers: &["に生まれた", "の出身", "で生まれた"],
            confidence: 0.7,
        },
        RelPattern {
            slug: "LIVES_IN",
            triggers: &["に住む", "に居住", "に在住"],
            confidence: 0.65,
        },
        RelPattern {
            slug: "MARRIED_TO",
            triggers: &["と結婚", "と結婚した", "の配偶者"],
            confidence: 0.85,
        },
        // Arabic (ar) - RTL
        RelPattern {
            slug: "WORKS_FOR",
            triggers: &["يعمل في", "يعمل لصالح", "موظف في", "يعمل مع"],
            confidence: 0.7,
        },
        RelPattern {
            slug: "FOUNDED",
            triggers: &["أسس", "أنشأ", "تأسست", "أنشأت"],
            confidence: 0.8,
        },
        RelPattern {
            slug: "LOCATED_IN",
            triggers: &["في", "ب", "يقع في", "موجود في"],
            confidence: 0.6,
        },
        RelPattern {
            slug: "BORN_IN",
            triggers: &["ولد في", "من مواليد", "من"],
            confidence: 0.7,
        },
        RelPattern {
            slug: "LIVES_IN",
            triggers: &["يعيش في", "يسكن في", "مقيم في"],
            confidence: 0.65,
        },
        RelPattern {
            slug: "MARRIED_TO",
            triggers: &["متزوج من", "زوج", "زوجة"],
            confidence: 0.85,
        },
        // Russian (ru)
        RelPattern {
            slug: "WORKS_FOR",
            triggers: &["работает в", "работает на", "работает для", "сотрудник"],
            confidence: 0.7,
        },
        RelPattern {
            slug: "FOUNDED",
            triggers: &["основал", "основала", "создал", "создала", "учредил"],
            confidence: 0.8,
        },
        RelPattern {
            slug: "LOCATED_IN",
            triggers: &["в", "на", "расположен в", "находится в"],
            confidence: 0.6,
        },
        RelPattern {
            slug: "BORN_IN",
            triggers: &["родился в", "родилась в", "родом из", "из"],
            confidence: 0.7,
        },
        RelPattern {
            slug: "LIVES_IN",
            triggers: &["живет в", "проживает в", "живущий в"],
            confidence: 0.65,
        },
        RelPattern {
            slug: "MARRIED_TO",
            triggers: &["женат на", "замужем за", "супруг", "супруга"],
            confidence: 0.85,
        },
    ];

    for pattern in patterns {
        // Find the canonical label in the registry (case-insensitive).
        // We return the label's *original* slug so callers preserve user-provided casing.
        let label = match relation_labels.iter().find(|l| {
            // Match both:
            // - exact canonical names (e.g. "PART_OF")
            // - normalized dataset slugs (e.g. "part-of" -> "PART_OF")
            norm_rel_slug(&l.slug) == pattern.slug || l.slug.eq_ignore_ascii_case(pattern.slug)
        }) {
            Some(l) => *l,
            None => continue,
        };

        for trigger in pattern.triggers {
            if let Some(pos) = between_lower.find(trigger) {
                // Validate entity types make sense for the relation
                let valid = match pattern.slug {
                    // Person-Organization relations
                    "CEO_OF" | "WORKS_FOR" | "FOUNDED" | "MANAGES" | "REPORTS_TO" => {
                        // If either side is unknown/misc, don't reject on type alone (relation datasets
                        // often use a richer schema than `EntityType`).
                        matches!(head.entity_type, EntityType::Custom { .. })
                            || matches!(tail.entity_type, EntityType::Custom { .. })
                            || (matches!(head.entity_type, EntityType::Person)
                                && matches!(tail.entity_type, EntityType::Organization))
                    }
                    // Location relations (any entity can be located in/born in a location)
                    "LOCATED_IN" | "BORN_IN" | "LIVES_IN" | "DIED_IN" => {
                        matches!(tail.entity_type, EntityType::Custom { .. })
                            || matches!(tail.entity_type, EntityType::Location)
                    }
                    // Temporal relations (any entity can have temporal attributes)
                    "OCCURRED_ON" | "STARTED_ON" | "ENDED_ON" => {
                        matches!(tail.entity_type, EntityType::Custom { .. })
                            || matches!(tail.entity_type, EntityType::Date | EntityType::Time)
                    }
                    // Organizational relations
                    "PART_OF" | "ACQUIRED" | "MERGED_WITH" | "PARENT_OF" => {
                        matches!(head.entity_type, EntityType::Custom { .. })
                            || matches!(tail.entity_type, EntityType::Custom { .. })
                            || (matches!(head.entity_type, EntityType::Organization)
                                && matches!(tail.entity_type, EntityType::Organization))
                    }
                    // Social relations
                    "MARRIED_TO" | "CHILD_OF" | "SIBLING_OF" => {
                        matches!(head.entity_type, EntityType::Custom { .. })
                            || matches!(tail.entity_type, EntityType::Custom { .. })
                            || (matches!(head.entity_type, EntityType::Person)
                                && matches!(tail.entity_type, EntityType::Person))
                    }
                    // Academic relations
                    "STUDIED_AT" | "TEACHES_AT" => {
                        matches!(head.entity_type, EntityType::Custom { .. })
                            || matches!(tail.entity_type, EntityType::Custom { .. })
                            || (matches!(head.entity_type, EntityType::Person)
                                && matches!(
                                    tail.entity_type,
                                    EntityType::Organization | EntityType::Location
                                ))
                    }
                    // Product relations
                    "DEVELOPS" | "USES" => {
                        matches!(head.entity_type, EntityType::Custom { .. })
                            || matches!(
                                head.entity_type,
                                EntityType::Organization | EntityType::Person
                            )
                    }
                    // Interaction relations
                    "MET_WITH" | "SPOKE_WITH" => {
                        matches!(head.entity_type, EntityType::Custom { .. })
                            || matches!(tail.entity_type, EntityType::Custom { .. })
                            || (matches!(head.entity_type, EntityType::Person)
                                && matches!(
                                    tail.entity_type,
                                    EntityType::Person | EntityType::Organization
                                ))
                    }
                    // Ownership
                    "OWNS" => {
                        matches!(head.entity_type, EntityType::Custom { .. })
                            || matches!(
                                head.entity_type,
                                EntityType::Person | EntityType::Organization
                            )
                    }
                    _ => true, // Default: allow any combination
                };

                if valid {
                    return Some((
                        label.slug.as_str(),
                        pattern.confidence,
                        Some((pos, pos + trigger.len())),
                    ));
                }
            }
        }
    }

    None
}

// =============================================================================
// Entity-type-based relation fallback
// =============================================================================

/// Maps (head_type, tail_type) -> likely relation types with base confidence.
///
/// Used as a fallback when no trigger pattern fires. Covers both CHisIEC-style
/// and standard NER entity type pairs.
fn get_likely_relations(head_type: &str, tail_type: &str) -> Vec<(&'static str, f32)> {
    let head = head_type.to_uppercase();
    let tail = tail_type.to_uppercase();

    match (head.as_str(), tail.as_str()) {
        // CHisIEC-style entity type codes
        ("PER", "OFI") | ("PERSON", "OFI") => vec![("任職", 0.7)],
        ("OFI", "PER") => vec![("上下級", 0.6)],
        ("PER", "LOC") => vec![("到達", 0.55), ("出生於某地", 0.4)],
        ("LOC", "PER") => vec![("到達", 0.5)],
        ("PER", "PER") => vec![
            ("上下級", 0.45),
            ("同僚", 0.4),
            ("父母", 0.3),
            ("兄弟", 0.3),
        ],
        ("OFI", "LOC") | ("LOC", "OFI") => vec![("管理", 0.5)],
        ("BOOK", "BOOK") | ("BOOK", "PER") | ("PER", "BOOK") => vec![("別名", 0.35)],
        // Person-Organization relations
        ("PERSON", "ORGANIZATION") | ("PER", "ORG") => vec![
            ("WORKS_FOR", 0.7),
            ("FOUNDED", 0.5),
            ("CEO_OF", 0.4),
            ("MEMBER_OF", 0.6),
        ],
        ("ORGANIZATION", "PERSON") | ("ORG", "PER") => {
            vec![("EMPLOYS", 0.7), ("FOUNDED_BY", 0.5), ("LED_BY", 0.4)]
        }
        // Person-Location relations
        ("PERSON", "LOCATION") | ("PERSON", "GPE") | ("PER", "GPE") => {
            vec![("LIVES_IN", 0.6), ("BORN_IN", 0.5), ("VISITED", 0.4)]
        }
        // Organization-Location relations
        ("ORGANIZATION", "LOCATION")
        | ("ORG", "LOC")
        | ("ORGANIZATION", "GPE")
        | ("ORG", "GPE") => vec![
            ("HEADQUARTERED_IN", 0.7),
            ("LOCATED_IN", 0.8),
            ("OPERATES_IN", 0.5),
        ],
        // Product-Organization relations
        ("PRODUCT", "ORGANIZATION") | ("PRODUCT", "ORG") => {
            vec![("MADE_BY", 0.8), ("PRODUCED_BY", 0.7)]
        }
        ("ORGANIZATION", "PRODUCT") | ("ORG", "PRODUCT") => {
            vec![("MAKES", 0.8), ("PRODUCES", 0.7), ("ANNOUNCED", 0.5)]
        }
        // Date relations
        (_, "DATE") | (_, "TIME") => vec![("OCCURRED_ON", 0.5), ("FOUNDED_ON", 0.4)],
        // Default: no strong relation signal
        _ => vec![],
    }
}

// =============================================================================
// Registry-free heuristic relation extraction
// =============================================================================

/// Extract relation triples using heuristics only -- no `SemanticRegistry` needed.
///
/// This is the backend-agnostic entry point for heuristic relation extraction.
/// It combines:
/// - Multilingual trigger-pattern matching (from the full `detect_relation_type` table)
/// - Entity-type-based fallback (when no trigger fires)
/// - Undirected pair deduplication (keeps highest-confidence per pair)
/// - Entity confidence weighting
///
/// Use this instead of [`extract_relation_triples`] when you don't have (or need)
/// a `SemanticRegistry`.
#[must_use]
pub fn extract_relation_triples_simple(
    entities: &[Entity],
    text: &str,
    relation_types: &[&str],
    config: &RelationExtractionConfig,
) -> Vec<RelationTriple> {
    if entities.len() < 2 {
        return Vec::new();
    }

    // Build temporary LabelDefinitions from the string slices so we can reuse
    // the existing `detect_relation_type` machinery.
    let owned_labels: Vec<super::registry::LabelDefinition> = relation_types
        .iter()
        .map(|slug| super::registry::LabelDefinition {
            slug: slug.to_string(),
            description: String::new(),
            category: super::registry::LabelCategory::Relation,
            modality: super::registry::ModalityHint::Any,
            threshold: config.threshold,
        })
        .collect();
    let label_refs: Vec<&super::registry::LabelDefinition> = owned_labels.iter().collect();

    // Normalize relation slugs for fuzzy matching (same logic as detect_relation_type).
    fn norm_rel_slug(s: &str) -> String {
        let mut out = String::with_capacity(s.len());
        let mut prev_underscore = false;
        for ch in s.chars() {
            if ch.is_alphanumeric() {
                if ch.is_ascii_alphabetic() {
                    out.push(ch.to_ascii_uppercase());
                } else {
                    out.push(ch);
                }
                prev_underscore = false;
            } else if !prev_underscore {
                out.push('_');
                prev_underscore = true;
            }
        }
        while out.starts_with('_') {
            out.remove(0);
        }
        while out.ends_with('_') {
            out.pop();
        }
        out
    }

    fn pick_relation_label(canonical: &str, relation_types: &[&str]) -> Option<String> {
        if relation_types.is_empty() {
            return None;
        }
        let want = norm_rel_slug(canonical);
        relation_types
            .iter()
            .find(|r| norm_rel_slug(r) == want)
            .map(|s| (*s).to_string())
    }

    let span_converter = crate::offset::SpanConverter::new(text);
    let text_char_count = text.chars().count();
    let text_char_len = text_char_count.max(1) as f32;

    let mut triples = Vec::new();

    for (i, head) in entities.iter().enumerate() {
        for (j, tail) in entities.iter().enumerate() {
            if i == j {
                continue;
            }

            // Skip overlapping spans.
            if head.start() < tail.end() && tail.start() < head.end() {
                continue;
            }

            // Check distance (character offsets).
            let distance = if head.end() <= tail.start() {
                tail.start() - head.end()
            } else {
                head.start().saturating_sub(tail.end())
            };
            if distance > config.max_span_distance {
                continue;
            }

            let (span_start, span_end) = if head.end() <= tail.start() {
                (head.end(), tail.start())
            } else {
                (tail.end(), head.start())
            };

            let between_span = span_converter.from_chars(span_start, span_end);
            let between_text = text
                .get(between_span.byte_start..between_span.byte_end)
                .unwrap_or("");

            // Try trigger-based detection first (multilingual, entity-type validated).
            if let Some((rel_type, mut confidence, _trigger)) =
                detect_relation_type(head, tail, between_text, &label_refs)
            {
                // Distance penalty (linear decay).
                let distance_penalty = if distance < config.max_span_distance {
                    (1.0 - (distance as f64 / config.max_span_distance as f64) * 0.5).max(0.5)
                } else {
                    0.5
                };
                confidence *= distance_penalty;

                // Incorporate entity confidence.
                confidence *= (head.confidence + tail.confidence) / 2.0;

                if confidence < config.threshold.value() {
                    continue;
                }

                triples.push(RelationTriple {
                    head_idx: i,
                    tail_idx: j,
                    relation_type: rel_type.to_string(),
                    confidence: Confidence::new(confidence),
                });
                continue;
            }

            // Type-based fallback: infer relation from entity type pair.
            let head_center = (head.start() + head.end()) as f32 / 2.0;
            let tail_center = (tail.start() + tail.end()) as f32 / 2.0;
            let proximity = 1.0 - ((head_center - tail_center).abs() / text_char_len).min(1.0);

            if proximity > 0.3 {
                let head_type = head.entity_type.as_label();
                let tail_type = tail.entity_type.as_label();
                for (rel_type, base_score) in get_likely_relations(head_type, tail_type) {
                    if !relation_types.is_empty()
                        && pick_relation_label(rel_type, relation_types).is_none()
                    {
                        continue;
                    }
                    let out_label = pick_relation_label(rel_type, relation_types)
                        .unwrap_or_else(|| rel_type.to_string());

                    let conf_f32 =
                        proximity * base_score * (head.confidence + tail.confidence) as f32 / 2.0;
                    if conf_f32 >= config.threshold.value() as f32 {
                        triples.push(RelationTriple {
                            head_idx: i,
                            tail_idx: j,
                            relation_type: out_label,
                            confidence: Confidence::new(conf_f32 as f64),
                        });
                        break; // One type-based relation per pair
                    }
                }
            }
        }
    }

    // Filter self-relations where head and tail have identical surface text
    // (different mentions of the same entity name at different positions).
    triples.retain(|r| entities[r.head_idx].text != entities[r.tail_idx].text);

    // Sort by confidence descending, then deduplicate per undirected pair.
    triples.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut seen_pairs = std::collections::HashSet::new();
    triples.retain(|r| {
        let canonical = if r.head_idx <= r.tail_idx {
            (r.head_idx, r.tail_idx)
        } else {
            (r.tail_idx, r.head_idx)
        };
        seen_pairs.insert(canonical)
    });

    triples
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Entity, EntityCategory, EntityType};

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Build a registry that contains the given relation slugs.
    fn registry_with_relations(slugs: &[&str]) -> SemanticRegistry {
        let mut builder = SemanticRegistry::builder();
        for slug in slugs {
            builder = builder.add_relation(slug, "test relation");
        }
        builder.build_zero(4)
    }

    /// Convenience: default config with extract_triggers enabled.
    fn default_config() -> RelationExtractionConfig {
        RelationExtractionConfig::default()
    }

    /// Build a person entity at the given character offsets.
    fn person(text: &str, start: usize, end: usize) -> Entity {
        Entity::new(text, EntityType::Person, start, end, 0.9)
    }

    /// Build an organization entity at the given character offsets.
    fn org(text: &str, start: usize, end: usize) -> Entity {
        Entity::new(text, EntityType::Organization, start, end, 0.9)
    }

    /// Build a location entity at the given character offsets.
    fn loc(text: &str, start: usize, end: usize) -> Entity {
        Entity::new(text, EntityType::Location, start, end, 0.9)
    }

    // =======================================================================
    // English pattern tests
    // =======================================================================

    #[test]
    fn test_works_for_pattern_english() {
        let text = "Alice works for Acme Corp in the city.";
        let entities = vec![person("Alice", 0, 5), org("Acme Corp", 16, 25)];
        let reg = registry_with_relations(&["WORKS_FOR"]);
        let rels = extract_relations(&entities, text, &reg, &default_config());

        assert_eq!(rels.len(), 1);
        assert_eq!(rels[0].relation_type, "WORKS_FOR");
        assert_eq!(rels[0].head.text, "Alice");
        assert_eq!(rels[0].tail.text, "Acme Corp");
    }

    #[test]
    fn test_founded_pattern_english() {
        let text = "Bob founded WidgetCo last year.";
        let entities = vec![person("Bob", 0, 3), org("WidgetCo", 12, 20)];
        let reg = registry_with_relations(&["FOUNDED"]);
        let rels = extract_relations(&entities, text, &reg, &default_config());

        assert_eq!(rels.len(), 1);
        assert_eq!(rels[0].relation_type, "FOUNDED");
        assert_eq!(rels[0].head.text, "Bob");
        assert_eq!(rels[0].tail.text, "WidgetCo");
    }

    #[test]
    fn test_located_in_pattern_english() {
        let text = "Acme Corp based in Berlin serves customers.";
        let entities = vec![org("Acme Corp", 0, 9), loc("Berlin", 19, 25)];
        let reg = registry_with_relations(&["LOCATED_IN"]);
        let rels = extract_relations(&entities, text, &reg, &default_config());

        assert_eq!(rels.len(), 1);
        assert_eq!(rels[0].relation_type, "LOCATED_IN");
        assert_eq!(rels[0].head.text, "Acme Corp");
        assert_eq!(rels[0].tail.text, "Berlin");
    }

    #[test]
    fn test_married_to_pattern_english() {
        let text = "Alice married to Bob at the ceremony.";
        let entities = vec![person("Alice", 0, 5), person("Bob", 17, 20)];
        let reg = registry_with_relations(&["MARRIED_TO"]);
        let rels = extract_relations(&entities, text, &reg, &default_config());

        // Both (Alice, Bob) and (Bob, Alice) pairs are checked; both find the
        // trigger in the between-text, so we expect 2 directed relations.
        assert_eq!(rels.len(), 2);
        assert!(rels.iter().all(|r| r.relation_type == "MARRIED_TO"));
    }

    #[test]
    fn test_born_in_pattern_english() {
        let text = "Alice born in Berlin many years ago.";
        let entities = vec![person("Alice", 0, 5), loc("Berlin", 14, 20)];
        let reg = registry_with_relations(&["BORN_IN"]);
        let rels = extract_relations(&entities, text, &reg, &default_config());

        assert_eq!(rels.len(), 1);
        assert_eq!(rels[0].relation_type, "BORN_IN");
    }

    #[test]
    fn test_ceo_of_pattern_english() {
        let text = "Alice ceo of Acme Corp recently.";
        let entities = vec![person("Alice", 0, 5), org("Acme Corp", 13, 22)];
        let reg = registry_with_relations(&["CEO_OF"]);
        let rels = extract_relations(&entities, text, &reg, &default_config());

        assert_eq!(rels.len(), 1);
        assert_eq!(rels[0].relation_type, "CEO_OF");
    }

    // =======================================================================
    // Type constraint tests
    // =======================================================================

    #[test]
    fn test_married_to_requires_person_person() {
        // Two persons: should match in both directions.
        let text = "Alice married to Bob yesterday.";
        let entities = vec![person("Alice", 0, 5), person("Bob", 17, 20)];
        let reg = registry_with_relations(&["MARRIED_TO"]);
        let rels = extract_relations(&entities, text, &reg, &default_config());
        assert_eq!(rels.len(), 2);
    }

    #[test]
    fn test_married_to_rejects_person_org() {
        // Person + Organization: should NOT match MARRIED_TO.
        let text = "Alice married to Acme Corp yesterday.";
        let entities = vec![person("Alice", 0, 5), org("Acme Corp", 17, 26)];
        let reg = registry_with_relations(&["MARRIED_TO"]);
        let rels = extract_relations(&entities, text, &reg, &default_config());
        assert!(
            rels.is_empty(),
            "MARRIED_TO should not match Person-Organization pair"
        );
    }

    #[test]
    fn test_works_for_requires_person_org() {
        // Person + Org: should match.
        let text = "Alice works for Acme Corp here.";
        let entities = vec![person("Alice", 0, 5), org("Acme Corp", 16, 25)];
        let reg = registry_with_relations(&["WORKS_FOR"]);
        assert_eq!(
            extract_relations(&entities, text, &reg, &default_config()).len(),
            1
        );
    }

    #[test]
    fn test_works_for_rejects_loc_loc() {
        // Location + Location: should NOT match WORKS_FOR.
        let text = "Berlin works for Munich today.";
        let entities = vec![loc("Berlin", 0, 6), loc("Munich", 17, 23)];
        let reg = registry_with_relations(&["WORKS_FOR"]);
        let rels = extract_relations(&entities, text, &reg, &default_config());
        assert!(
            rels.is_empty(),
            "WORKS_FOR should not match Location-Location pair"
        );
    }

    // =======================================================================
    // Multilingual trigger tests
    // =======================================================================

    #[test]
    fn test_chinese_founded_pattern() {
        // "X 创立 Y" -- Chinese trigger for FOUNDED.
        let text = "张三 创立 华为公司 在深圳";
        let entities = vec![person("张三", 0, 2), org("华为公司", 5, 9)];
        let reg = registry_with_relations(&["FOUNDED"]);
        let rels = extract_relations(&entities, text, &reg, &default_config());

        assert_eq!(rels.len(), 1);
        assert_eq!(rels[0].relation_type, "FOUNDED");
    }

    #[test]
    fn test_spanish_founded_pattern() {
        // "X fundó Y" -- Spanish trigger for FOUNDED.
        let text = "Carlos fundó Empresa aqui.";
        let entities = vec![person("Carlos", 0, 6), org("Empresa", 13, 20)];
        let reg = registry_with_relations(&["FOUNDED"]);
        let rels = extract_relations(&entities, text, &reg, &default_config());

        assert_eq!(rels.len(), 1);
        assert_eq!(rels[0].relation_type, "FOUNDED");
    }

    #[test]
    fn test_french_married_to_pattern() {
        // "X marié avec Y" -- French trigger for MARRIED_TO.
        // Both directions match for Person-Person symmetric relations.
        let text = "Pierre marié avec Marie hier.";
        let entities = vec![person("Pierre", 0, 6), person("Marie", 18, 23)];
        let reg = registry_with_relations(&["MARRIED_TO"]);
        let rels = extract_relations(&entities, text, &reg, &default_config());

        assert_eq!(rels.len(), 2);
        assert!(rels.iter().all(|r| r.relation_type == "MARRIED_TO"));
    }

    #[test]
    fn test_german_born_in_pattern() {
        // "X geboren in Y" -- German trigger for BORN_IN.
        let text = "Hans geboren in Berlin damals.";
        let entities = vec![person("Hans", 0, 4), loc("Berlin", 16, 22)];
        let reg = registry_with_relations(&["BORN_IN"]);
        let rels = extract_relations(&entities, text, &reg, &default_config());

        assert_eq!(rels.len(), 1);
        assert_eq!(rels[0].relation_type, "BORN_IN");
    }

    // =======================================================================
    // Distance penalty and threshold filtering
    // =======================================================================

    #[test]
    fn test_distance_penalty_close_entities() {
        // Entities adjacent (distance = 1 char of space): minimal penalty.
        let text = "A works for B end.";
        let entities = vec![person("A", 0, 1), org("B", 12, 13)];
        let reg = registry_with_relations(&["WORKS_FOR"]);
        let rels = extract_relations(&entities, text, &reg, &default_config());
        assert_eq!(rels.len(), 1);
        // With distance 11, penalty factor = 1.0 - (11/50)*0.5 = 0.89.
        // Base confidence 0.7 * 0.89 ~ 0.623, above default threshold 0.5.
        assert!(rels[0].confidence > 0.5);
    }

    #[test]
    fn test_distance_penalty_filters_low_confidence() {
        // High threshold should filter out distant, lower-confidence matches.
        let text = "A works for B end.";
        let entities = vec![person("A", 0, 1), org("B", 12, 13)];
        let reg = registry_with_relations(&["WORKS_FOR"]);
        let config = RelationExtractionConfig {
            threshold: Confidence::new(0.95),
            ..default_config()
        };
        let rels = extract_relations(&entities, text, &reg, &config);
        assert!(
            rels.is_empty(),
            "High threshold should filter distance-penalized relation"
        );
    }

    #[test]
    fn test_entities_beyond_max_distance_skipped() {
        // Entities separated by more than max_span_distance should produce no relations.
        let text = "Alice works for Acme Corp";
        let entities = vec![person("Alice", 0, 5), org("Acme Corp", 16, 25)];
        let reg = registry_with_relations(&["WORKS_FOR"]);
        let config = RelationExtractionConfig {
            max_span_distance: 2, // very small
            ..default_config()
        };
        let rels = extract_relations(&entities, text, &reg, &config);
        assert!(
            rels.is_empty(),
            "Entities beyond max_span_distance should be skipped"
        );
    }

    // =======================================================================
    // Edge cases
    // =======================================================================

    #[test]
    fn test_empty_entities_list() {
        let text = "No entities here.";
        let entities: Vec<Entity> = vec![];
        let reg = registry_with_relations(&["WORKS_FOR"]);
        let rels = extract_relations(&entities, text, &reg, &default_config());
        assert!(rels.is_empty());
    }

    #[test]
    fn test_single_entity_no_pairs() {
        let text = "Only Alice here.";
        let entities = vec![person("Alice", 5, 10)];
        let reg = registry_with_relations(&["WORKS_FOR"]);
        let rels = extract_relations(&entities, text, &reg, &default_config());
        assert!(rels.is_empty());
    }

    #[test]
    fn test_no_relation_labels_in_registry() {
        // Registry with only entity labels, no relation labels -- should return empty.
        let reg = SemanticRegistry::standard_ner(4);
        let text = "Alice works for Acme Corp here.";
        let entities = vec![person("Alice", 0, 5), org("Acme Corp", 16, 25)];
        let rels = extract_relations(&entities, text, &reg, &default_config());
        assert!(rels.is_empty());
    }

    #[test]
    fn test_overlapping_spans_skipped_in_triples() {
        // "New York" overlapping with "York": extract_relation_triples skips overlapping spans.
        let text = "New York is in New York State area.";
        let entities = vec![
            loc("New York", 0, 8),
            loc("York", 4, 8), // overlaps with "New York"
        ];
        let reg = registry_with_relations(&["LOCATED_IN"]);
        let triples = extract_relation_triples(&entities, text, &reg, &default_config());
        assert!(
            triples.is_empty(),
            "Overlapping spans should be skipped in extract_relation_triples"
        );
    }

    // =======================================================================
    // Trigger span extraction
    // =======================================================================

    #[test]
    fn test_trigger_span_present_when_enabled() {
        let text = "Alice works for Acme Corp today.";
        let entities = vec![person("Alice", 0, 5), org("Acme Corp", 16, 25)];
        let reg = registry_with_relations(&["WORKS_FOR"]);
        let config = RelationExtractionConfig {
            extract_triggers: true,
            ..default_config()
        };
        let rels = extract_relations(&entities, text, &reg, &config);
        assert_eq!(rels.len(), 1);
        let trigger = rels[0].trigger_span.expect("trigger_span should be Some");
        // Trigger should point to "works for" in the between-text " works for ".
        let trigger_text: String = text
            .chars()
            .skip(trigger.0)
            .take(trigger.1 - trigger.0)
            .collect();
        assert!(
            trigger_text.contains("works for"),
            "trigger text '{}' should contain 'works for'",
            trigger_text
        );
    }

    #[test]
    fn test_trigger_span_absent_when_disabled() {
        let text = "Alice works for Acme Corp today.";
        let entities = vec![person("Alice", 0, 5), org("Acme Corp", 16, 25)];
        let reg = registry_with_relations(&["WORKS_FOR"]);
        let config = RelationExtractionConfig {
            extract_triggers: false,
            ..default_config()
        };
        let rels = extract_relations(&entities, text, &reg, &config);
        assert_eq!(rels.len(), 1);
        assert!(
            rels[0].trigger_span.is_none(),
            "trigger_span should be None when extract_triggers is disabled"
        );
    }

    // =======================================================================
    // Relation direction (subject vs object ordering)
    // =======================================================================

    #[test]
    fn test_relation_direction_head_before_tail() {
        // Head entity appears before tail in text.
        let text = "Alice works for Acme Corp here.";
        let entities = vec![person("Alice", 0, 5), org("Acme Corp", 16, 25)];
        let reg = registry_with_relations(&["WORKS_FOR"]);
        let rels = extract_relations(&entities, text, &reg, &default_config());
        assert_eq!(rels.len(), 1);
        assert_eq!(rels[0].head.text, "Alice");
        assert_eq!(rels[0].tail.text, "Acme Corp");
    }

    #[test]
    fn test_relation_direction_both_orderings_checked() {
        // Both (i,j) and (j,i) are checked. If "Acme Corp ... employed by ... Alice"
        // appears, the (Acme Corp -> Alice) pair should also look at the between text.
        // Here we have "Alice employed by Acme Corp" so head=Alice, tail=Acme Corp.
        let text = "Alice employed by Acme Corp now.";
        let entities = vec![person("Alice", 0, 5), org("Acme Corp", 18, 27)];
        let reg = registry_with_relations(&["WORKS_FOR"]);
        let rels = extract_relations(&entities, text, &reg, &default_config());

        // The trigger "employed by" is in the between-text for (Alice, Acme Corp).
        assert!(!rels.is_empty());
        let forward = rels
            .iter()
            .find(|r| r.head.text == "Alice" && r.tail.text == "Acme Corp");
        assert!(
            forward.is_some(),
            "Should find relation with Alice as head and Acme Corp as tail"
        );
    }

    // =======================================================================
    // extract_relation_triples API
    // =======================================================================

    #[test]
    fn test_extract_relation_triples_basic() {
        let text = "Alice works for Acme Corp here.";
        let entities = vec![person("Alice", 0, 5), org("Acme Corp", 16, 25)];
        let reg = registry_with_relations(&["WORKS_FOR"]);
        let triples = extract_relation_triples(&entities, text, &reg, &default_config());

        assert_eq!(triples.len(), 1);
        assert_eq!(triples[0].head_idx, 0);
        assert_eq!(triples[0].tail_idx, 1);
        assert_eq!(triples[0].relation_type, "WORKS_FOR");
    }

    #[test]
    fn test_extract_relation_triples_single_entity_returns_empty() {
        let text = "Only Alice here.";
        let entities = vec![person("Alice", 5, 10)];
        let reg = registry_with_relations(&["WORKS_FOR"]);
        let triples = extract_relation_triples(&entities, text, &reg, &default_config());
        assert!(triples.is_empty());
    }

    // =======================================================================
    // Dataset-style / normalized slug matching
    // =======================================================================

    #[test]
    fn test_kebab_case_slug_matches_pattern() {
        // DocRED-style "part-of" should match the PART_OF pattern.
        // Both directions match for Org-Org pair.
        let text = "Division part of Corporation here.";
        let entities = vec![org("Division", 0, 8), org("Corporation", 17, 28)];
        let reg = registry_with_relations(&["part-of"]);
        let rels = extract_relations(&entities, text, &reg, &default_config());
        assert_eq!(rels.len(), 2);
        // The returned type should be the registry's slug, not the canonical one.
        assert!(rels.iter().all(|r| r.relation_type == "part-of"));
    }

    #[test]
    fn test_confidence_clamped_to_unit_interval() {
        // Confidence must always be in [0.0, 1.0].
        let text = "Alice works for Acme Corp end.";
        let entities = vec![person("Alice", 0, 5), org("Acme Corp", 16, 25)];
        let reg = registry_with_relations(&["WORKS_FOR"]);
        let rels = extract_relations(&entities, text, &reg, &default_config());
        for r in &rels {
            assert!(
                (0.0..=1.0).contains(&r.confidence.value()),
                "confidence {} not in [0, 1]",
                r.confidence
            );
        }
    }

    #[test]
    fn test_other_entity_type_allows_any_relation() {
        // Custom entity types should bypass type constraints.
        // Both directions match since both entities are Custom.
        let text = "FooEntity married to BarEntity now.";
        let entities = vec![
            Entity::new(
                "FooEntity",
                EntityType::custom("MISC", EntityCategory::Misc),
                0,
                9,
                0.9,
            ),
            Entity::new(
                "BarEntity",
                EntityType::custom("MISC", EntityCategory::Misc),
                21,
                30,
                0.9,
            ),
        ];
        let reg = registry_with_relations(&["MARRIED_TO"]);
        let rels = extract_relations(&entities, text, &reg, &default_config());
        assert_eq!(
            rels.len(),
            2,
            "Other entity type should bypass type constraints (both directions)"
        );
    }

    // =====================================================================
    // Additional English pattern tests (manages, studied_at, child_of)
    // =====================================================================

    #[test]
    fn test_manages_pattern_english() {
        let text = "Alice manages Engineering at the office.";
        let entities = vec![person("Alice", 0, 5), org("Engineering", 14, 25)];
        let reg = registry_with_relations(&["MANAGES"]);
        let rels = extract_relations(&entities, text, &reg, &default_config());
        assert_eq!(rels.len(), 1);
        assert_eq!(rels[0].relation_type, "MANAGES");
        assert_eq!(rels[0].head.text, "Alice");
        assert_eq!(rels[0].tail.text, "Engineering");
    }

    #[test]
    fn test_studied_at_pattern_english() {
        let text = "Alice studied at MIT before her career.";
        let entities = vec![person("Alice", 0, 5), org("MIT", 17, 20)];
        let reg = registry_with_relations(&["STUDIED_AT"]);
        let rels = extract_relations(&entities, text, &reg, &default_config());
        assert_eq!(rels.len(), 1);
        assert_eq!(rels[0].relation_type, "STUDIED_AT");
    }

    #[test]
    fn test_child_of_pattern_english() {
        // Both directions produce a match because the between-text contains
        // "daughter of" regardless of which entity is head vs tail.
        let text = "Alice daughter of Bob in the family.";
        let entities = vec![person("Alice", 0, 5), person("Bob", 18, 21)];
        let reg = registry_with_relations(&["CHILD_OF"]);
        let rels = extract_relations(&entities, text, &reg, &default_config());
        assert_eq!(rels.len(), 2);
        assert!(rels.iter().all(|r| r.relation_type == "CHILD_OF"));
    }

    // =====================================================================
    // Additional type constraint tests
    // =====================================================================

    #[test]
    fn test_studied_at_rejects_org_org() {
        let text = "Acme studied at BigCorp recently.";
        let entities = vec![org("Acme", 0, 4), org("BigCorp", 16, 23)];
        let reg = registry_with_relations(&["STUDIED_AT"]);
        let rels = extract_relations(&entities, text, &reg, &default_config());
        assert!(
            rels.is_empty(),
            "STUDIED_AT should not match Organization-Organization pair"
        );
    }

    #[test]
    fn test_located_in_rejects_person_tail() {
        let text = "Acme based in Alice recently.";
        let entities = vec![org("Acme", 0, 4), person("Alice", 14, 19)];
        let reg = registry_with_relations(&["LOCATED_IN"]);
        let rels = extract_relations(&entities, text, &reg, &default_config());
        assert!(
            rels.is_empty(),
            "LOCATED_IN should not match when tail is Person"
        );
    }

    #[test]
    fn test_part_of_requires_org_org() {
        let text = "Skunkworks part of Lockheed here.";
        let entities = vec![org("Skunkworks", 0, 10), org("Lockheed", 19, 27)];
        let reg = registry_with_relations(&["PART_OF"]);
        let rels = extract_relations(&entities, text, &reg, &default_config());
        assert!(
            !rels.is_empty(),
            "PART_OF should match Organization-Organization pair"
        );
        assert!(rels.iter().all(|r| r.relation_type == "PART_OF"));
    }

    // =====================================================================
    // Distance penalty value verification
    // =====================================================================

    #[test]
    fn test_distance_penalty_monotonically_decreases_confidence() {
        // Two texts, same trigger, different entity distances.
        // Use low threshold so distance penalty does not filter the far pair.
        let text_close = "Alice works for BigCo end.";
        let text_far = "Alice ..... works for ..... BigCo end.";
        let entities_close = vec![person("Alice", 0, 5), org("BigCo", 16, 21)];
        let entities_far = vec![person("Alice", 0, 5), org("BigCo", 28, 33)];
        let reg = registry_with_relations(&["WORKS_FOR"]);
        let low_threshold = RelationExtractionConfig {
            threshold: Confidence::new(0.3),
            ..default_config()
        };
        let rels_close = extract_relations(&entities_close, text_close, &reg, &low_threshold);
        let rels_far = extract_relations(&entities_far, text_far, &reg, &low_threshold);
        assert_eq!(rels_close.len(), 1);
        assert_eq!(rels_far.len(), 1);
        assert!(
            rels_close[0].confidence > rels_far[0].confidence,
            "Closer entities ({:.3}) should have higher confidence than farther ({:.3})",
            rels_close[0].confidence,
            rels_far[0].confidence
        );
    }

    #[test]
    fn test_threshold_filters_marginal_confidence() {
        // Set a threshold just above the penalized confidence to verify filtering.
        let text = "Alice works for Acme Corp end.";
        let entities = vec![person("Alice", 0, 5), org("Acme Corp", 16, 25)];
        let reg = registry_with_relations(&["WORKS_FOR"]);
        // Get actual confidence.
        let rels = extract_relations(&entities, text, &reg, &default_config());
        assert_eq!(rels.len(), 1);
        let actual_conf = rels[0].confidence;
        // Threshold just above actual confidence: should filter.
        let config = RelationExtractionConfig {
            threshold: Confidence::new(actual_conf.value() + 0.01),
            ..default_config()
        };
        let rels2 = extract_relations(&entities, text, &reg, &config);
        assert!(
            rels2.is_empty(),
            "Threshold above confidence should filter the relation"
        );
    }

    // =====================================================================
    // Multiple relations from distinct entity pairs
    // =====================================================================

    #[test]
    fn test_multiple_entity_pairs_yield_multiple_relations() {
        let text = "Alice works for Acme Corp based in Berlin today.";
        let entities = vec![
            person("Alice", 0, 5),
            org("Acme Corp", 16, 25),
            loc("Berlin", 35, 41),
        ];
        let reg = registry_with_relations(&["WORKS_FOR", "LOCATED_IN"]);
        let rels = extract_relations(&entities, text, &reg, &default_config());
        let works_for: Vec<_> = rels
            .iter()
            .filter(|r| r.relation_type == "WORKS_FOR")
            .collect();
        let located_in: Vec<_> = rels
            .iter()
            .filter(|r| r.relation_type == "LOCATED_IN")
            .collect();
        assert!(
            !works_for.is_empty(),
            "Should find WORKS_FOR between Alice and Acme Corp"
        );
        assert!(
            !located_in.is_empty(),
            "Should find LOCATED_IN between Acme Corp and Berlin"
        );
    }

    // =====================================================================
    // Additional multilingual trigger tests
    // =====================================================================

    #[test]
    fn test_chinese_works_for_pattern() {
        // Chinese trigger for WORKS_FOR.
        let text = "李明 工作于 百度公司 在北京";
        let entities = vec![person("李明", 0, 2), org("百度公司", 7, 11)];
        let reg = registry_with_relations(&["WORKS_FOR"]);
        let rels = extract_relations(&entities, text, &reg, &default_config());
        assert_eq!(rels.len(), 1);
        assert_eq!(rels[0].relation_type, "WORKS_FOR");
    }

    #[test]
    fn test_spanish_works_for_pattern() {
        // Spanish trigger for WORKS_FOR.
        let text = "Maria trabaja en Google aqui.";
        let entities = vec![person("Maria", 0, 5), org("Google", 17, 23)];
        let reg = registry_with_relations(&["WORKS_FOR"]);
        let rels = extract_relations(&entities, text, &reg, &default_config());
        assert_eq!(rels.len(), 1);
        assert_eq!(rels[0].relation_type, "WORKS_FOR");
    }

    #[test]
    fn test_french_works_for_pattern() {
        // French trigger for WORKS_FOR.
        let text = "Pierre travaille pour Renault ici.";
        let entities = vec![person("Pierre", 0, 6), org("Renault", 22, 29)];
        let reg = registry_with_relations(&["WORKS_FOR"]);
        let rels = extract_relations(&entities, text, &reg, &default_config());
        assert_eq!(rels.len(), 1);
        assert_eq!(rels[0].relation_type, "WORKS_FOR");
    }

    #[test]
    fn test_german_works_for_pattern() {
        // German trigger for WORKS_FOR.
        let text = "Hans arbeitet bei Siemens dort.";
        let entities = vec![person("Hans", 0, 4), org("Siemens", 18, 25)];
        let reg = registry_with_relations(&["WORKS_FOR"]);
        let rels = extract_relations(&entities, text, &reg, &default_config());
        assert_eq!(rels.len(), 1);
        assert_eq!(rels[0].relation_type, "WORKS_FOR");
    }

    // =====================================================================
    // Unicode multi-byte text with character offsets
    // =====================================================================

    #[test]
    fn test_unicode_multibyte_offsets_correct() {
        // Ensure character offsets work with multi-byte chars in between-text.
        let text = "Ren\u{00e9} travaille pour CNRS ici.";
        // R(0) e(1) n(2) e-acute(3) = 4 chars; CNRS at char 20..24
        let entities = vec![person("Ren\u{00e9}", 0, 4), org("CNRS", 20, 24)];
        let reg = registry_with_relations(&["WORKS_FOR"]);
        let rels = extract_relations(&entities, text, &reg, &default_config());
        assert_eq!(rels.len(), 1);
        assert_eq!(rels[0].relation_type, "WORKS_FOR");
        assert_eq!(rels[0].head.text, "Ren\u{00e9}");
    }

    // =======================================================================
    // QA regression: bare trigger removal
    // =======================================================================

    fn misc(text: &str, start: usize, end: usize) -> Entity {
        Entity::new(
            text,
            EntityType::custom("MISC", EntityCategory::Misc),
            start,
            end,
            0.9,
        )
    }

    #[test]
    fn no_nonsensical_located_in() {
        // "Doudna won the Nobel Prize in Chemistry" -- bare "in" between
        // "Nobel Prize" and "Chemistry" should NOT trigger LOCATED_IN.
        let text = "Doudna won the Nobel Prize in Chemistry for her work.";
        let entities = vec![person("Doudna", 0, 6), misc("Chemistry", 30, 39)];
        let reg = registry_with_relations(&["LOCATED_IN"]);
        let rels = extract_relations(&entities, text, &reg, &default_config());
        assert!(
            rels.is_empty(),
            "bare 'in' should not produce LOCATED_IN(Doudna, Chemistry): {:?}",
            rels
        );
    }

    #[test]
    fn valid_located_in_with_full_trigger() {
        let text = "Apple headquartered in Cupertino today.";
        let entities = vec![org("Apple", 0, 5), loc("Cupertino", 23, 32)];
        let reg = registry_with_relations(&["LOCATED_IN"]);
        // Use a low threshold so the distance penalty doesn't mask the trigger match.
        let config = RelationExtractionConfig {
            threshold: Confidence::new(0.3),
            ..default_config()
        };
        let rels = extract_relations(&entities, text, &reg, &config);
        assert_eq!(rels.len(), 1, "headquartered in should still match");
        assert_eq!(rels[0].relation_type, "LOCATED_IN");
    }

    #[test]
    fn type_guard_blocks_located_in_to_person() {
        // LOCATED_IN requires tail to be Location (or Other/Custom).
        // A Person tail should be blocked.
        let text = "Doudna based in Charpentier for the experiment.";
        let entities = vec![person("Doudna", 0, 6), person("Charpentier", 16, 27)];
        let reg = registry_with_relations(&["LOCATED_IN"]);
        let rels = extract_relations(&entities, text, &reg, &default_config());
        assert!(
            rels.is_empty(),
            "LOCATED_IN with PER tail should be blocked: {:?}",
            rels
        );
    }
}
