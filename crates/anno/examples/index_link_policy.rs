// SPDX-License-Identifier: MIT OR Apache-2.0
//! Compare fixed and adaptive decision horizons on original synthetic names. From a repository checkout:
//! `cargo run --locked -p anno --no-default-features --example index_link_policy`
//! Each case is one track whose three observations already belong together. This measures
//! assignment to a catalog, conditional on those tracks and a fixed candidate block.
//! Costs are illustrative integer units. Scores are similarities, not probabilities.
//! Each policy makes one irreversible decision. Online sees the first observation, buffered
//! sees two, and bulk sees the full track without jointly optimizing across tracks. Repeated
//! observations can copy the same wrong name. The adaptive policy learns when to wait on
//! development families. Test families use the same observation templates with new entities.
//! NIL declares the person absent from the catalog;
//! review defers that judgment at a fixed cost. No independent evidence assumption is needed.

use anno::coalesce::resolver::string_similarity;
use std::collections::{BTreeMap, BTreeSet};

const TRACKS_PER_FAMILY: usize = 24;
const DEV_FAMILIES: std::ops::Range<u32> = 0..48;
const TEST_FAMILIES: std::ops::Range<u32> = 48..144;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Gold {
    Present(usize), // position in the candidate block
    Blocked,        // cataloged, but absent from the block
    Unseen,         // absent from the catalog
}

#[derive(Clone, Copy)]
struct View {
    candidate: usize,
    score: f32,
    margin: f32,
}

struct Case {
    gold: Gold,
    views: [View; 3],
}

fn rank(names: &[String; 2], observations: &[&str; 3]) -> [View; 3] {
    let mut sums = [0.0; 2];
    std::array::from_fn(|step| {
        for (i, name) in names.iter().enumerate() {
            sums[i] += string_similarity(name, observations[step]);
        }
        let candidate = usize::from(sums[1] > sums[0]);
        let n = (step + 1) as f32;
        View {
            candidate,
            score: sums[candidate] / n,
            margin: (sums[candidate] - sums[1 - candidate]) / n,
        }
    })
}

// A fixed integer generator makes the population independent of platform and rand version.
fn draw(state: &mut u32, upper: u32) -> u32 {
    *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
    ((*state as u64 * upper as u64) >> 32) as u32
}

fn population(families: std::ops::Range<u32>) -> Vec<Case> {
    let mut cases = Vec::new();
    for family in families {
        // A query searches its supplied district. Only these two catalog records are in it.
        let surname = format!("Vem{family}");
        let names = [
            format!("Dar{family} {surname}"),
            format!("Ser{family} {surname}"),
        ];
        let neighbor = format!("Tor{family} {surname}"); // catalog record in another district
        let relocated = format!("Kel{family} Mur{family}"); // another out-of-district record
        let stranger = format!("Lom{family} Tas{family}"); // no catalog record
        let mut state = (family + 1).wrapping_mul(2654435761);
        for _ in 0..TRACKS_PER_FAMILY {
            let i = draw(&mut state, 2) as usize;
            let full = names[i].as_str();
            let rival = names[1 - i].as_str();
            let shared = surname.as_str();
            // Most linked tracks are easy; others have delayed hints, a wrong attribution,
            // a shared surname, or a genuinely unseen person with an exact namesake.
            let (gold, observations): (_, [&str; 3]) = match draw(&mut state, 100) {
                0..=27 => (Gold::Present(i), [full, full, full]),
                28..=47 => (Gold::Present(i), [shared, full, full]),
                48..=59 => (Gold::Present(i), [shared, shared, full]),
                60..=64 => (Gold::Present(i), [rival, full, full]),
                65..=68 => (Gold::Present(i), [full, rival, rival]),
                69..=73 => (Gold::Present(i), [shared, shared, shared]),
                74..=79 => (Gold::Blocked, [shared, &neighbor, &neighbor]),
                80..=84 => (Gold::Blocked, [&relocated, &relocated, &relocated]),
                85..=92 => (Gold::Unseen, [stranger.as_str(), &stranger, &stranger]),
                93..=96 => (Gold::Unseen, [shared, shared, shared]),
                _ => (Gold::Unseen, [full, full, full]),
            };
            cases.push(Case {
                gold,
                views: rank(&names, &observations),
            });
        }
    }
    cases
}

#[derive(Clone, Copy, Debug)]
struct Rule {
    min_score: f32,
    min_margin: f32,
    nil_below: f32,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Decision {
    Link(usize),
    Nil,
    Review,
}

impl Rule {
    fn decide(self, view: View) -> Decision {
        if view.margin > 0.0 && view.score >= self.min_score && view.margin >= self.min_margin {
            Decision::Link(view.candidate)
        } else if view.score < self.nil_below {
            Decision::Nil
        } else {
            Decision::Review
        }
    }
}

#[derive(Clone, Copy)]
struct Costs {
    wrong_link: u32,
    false_nil: u32,
    review: u32,
    wait: u32, // charged per track per additional observation interval
}

#[derive(Default, Debug, PartialEq, Eq)]
struct Counts {
    correct_link: u32,
    wrong_link: u32,
    correct_nil: u32,
    false_nil: u32,
    review: u32,
    wait: u32,
    stopped: [u32; 3],
}

impl Counts {
    fn observe(&mut self, gold: Gold, decision: Decision) {
        match (gold, decision) {
            (Gold::Present(i), Decision::Link(j)) if i == j => self.correct_link += 1,
            (_, Decision::Link(_)) => self.wrong_link += 1,
            (Gold::Unseen, Decision::Nil) => self.correct_nil += 1,
            (_, Decision::Nil) => self.false_nil += 1,
            (_, Decision::Review) => self.review += 1,
        }
    }

    fn loss(&self, c: Costs) -> u32 {
        self.wrong_link * c.wrong_link
            + self.false_nil * c.false_nil
            + self.review * c.review
            + self.wait * c.wait
    }
}

fn evaluate<'a>(
    cases: impl IntoIterator<Item = &'a Case>,
    decide: impl Fn(&[View; 3]) -> (usize, Decision),
) -> Counts {
    let mut counts = Counts::default();
    for case in cases {
        let (step, action) = decide(&case.views);
        counts.observe(case.gold, action);
        counts.wait += step as u32;
        counts.stopped[step] += 1;
    }
    counts
}

fn fixed(step: usize, rule: Rule) -> impl Fn(&[View; 3]) -> (usize, Decision) {
    move |views| (step, rule.decide(views[step]))
}

fn fit(dev: &[Case], step: usize, costs: Costs) -> Rule {
    let mut best = None;
    // Exact loss ties retain the first setting (higher score/margin, smaller NIL region).
    // 1.1 allows an all-review comparator; -1.0 turns the NIL decision off.
    for min_score in [1.1, 0.99, 0.82, 0.74, 0.60, 0.49] {
        for min_margin in [0.60, 0.34, 0.20, 0.0] {
            for nil_below in [-1.0, 0.01, 0.34, 0.42, 0.51] {
                let rule = Rule {
                    min_score,
                    min_margin,
                    nil_below,
                };
                let loss = evaluate(dev, fixed(step, rule)).loss(costs);
                if best.is_none_or(|(lowest, _)| loss < lowest) {
                    best = Some((loss, rule));
                }
            }
        }
    }
    best.unwrap().1
}

// The candidate ID carries no information across families. Quarter-point bins for
// score and margin are fixed before fitting. Tied costs and unobserved bins stop now.
fn state(view: View) -> (u8, u8) {
    ((view.score * 4.0) as u8, (view.margin * 4.0) as u8)
}

struct StoppingPolicy {
    rules: [Rule; 3],
    wait_states: [BTreeSet<(u8, u8)>; 2],
}

impl StoppingPolicy {
    fn decide_from(&self, views: &[View; 3], mut step: usize) -> (usize, Decision) {
        while step < self.wait_states.len() && self.wait_states[step].contains(&state(views[step]))
        {
            step += 1;
        }
        (step, self.rules[step].decide(views[step]))
    }

    fn fit(dev: &[Case], rules: [Rule; 3], costs: Costs) -> Self {
        let mut policy = Self {
            rules,
            wait_states: Default::default(),
        };
        // Fit backward. For a development track, compare stopping now with the
        // observed cost of following the already-fitted policy one interval later.
        for step in (0..2).rev() {
            let mut sums: BTreeMap<_, (u32, u32)> = BTreeMap::new();
            for case in dev {
                let view = case.views[step];
                let (now, later) = sums.entry(state(view)).or_default();
                let mut stop_cost = Counts::default();
                stop_cost.observe(case.gold, rules[step].decide(view));
                *now += stop_cost.loss(costs);
                let (stop_at, action) = policy.decide_from(&case.views, step + 1);
                let mut continue_cost = Counts::default();
                continue_cost.observe(case.gold, action);
                continue_cost.wait = (stop_at - step) as u32;
                *later += continue_cost.loss(costs);
            }
            policy.wait_states[step] = sums
                .into_iter()
                .filter_map(|(state, (now, later))| (later < now).then_some(state))
                .collect();
        }
        policy
    }
}

// Moments of paired loss differences across test families, in cost units per track.
// Keep the within-family tracks together. The normal interval describes Monte Carlo
// uncertainty under this generator and the fitted rules, not deployment uncertainty.
fn paired_difference(
    test: &[Case],
    decide: impl Fn(&[View; 3]) -> (usize, Decision),
    reference: impl Fn(&[View; 3]) -> (usize, Decision),
    costs: Costs,
) -> (f64, f64, f64) {
    let (families, remainder) = test.as_chunks::<TRACKS_PER_FAMILY>();
    assert!(remainder.is_empty(), "expected complete test families");
    let differences: Vec<_> = families
        .iter()
        .map(|family| {
            let loss = f64::from(evaluate(family, &decide).loss(costs));
            let baseline = f64::from(evaluate(family, &reference).loss(costs));
            (loss - baseline) / TRACKS_PER_FAMILY as f64
        })
        .collect();
    let n = differences.len() as f64;
    let mean = differences.iter().sum::<f64>() / n;
    let squared_deviations = differences.iter().map(|d| (d - mean).powi(2)).sum::<f64>();
    let half_width = 1.96 * (squared_deviations / ((n - 1.0) * n)).sqrt();
    (mean, mean - half_width, mean + half_width)
}

fn main() {
    let dev = population(DEV_FAMILIES);
    let test = population(TEST_FAMILIES);
    let blocked = test.iter().filter(|c| c.gold == Gold::Blocked).count();
    let unseen = test.iter().filter(|c| c.gold == Gold::Unseen).count();
    let present = test.len() - blocked - unseen;
    println!(
        "dev={} test={} retrieved={} blocked={} unseen={} candidate_recall={}/{}",
        dev.len(),
        test.len(),
        present,
        blocked,
        unseen,
        present,
        present + blocked
    );
    println!("scenario,mode,min_score,min_margin,nil_below,correct_link,wrong_link,correct_nil,false_nil,review,wait,loss,loss_per_track,blocked_link,blocked_nil,blocked_review,unseen_link,unseen_nil,unseen_review,stopped_1,stopped_2,stopped_3,delta_vs_online,delta_lo95,delta_hi95,dev_best_fixed,delta_vs_best,best_lo95,best_hi95");
    for (scenario, costs) in [
        (
            "fast",
            Costs {
                wrong_link: 50,
                false_nil: 35,
                review: 25,
                wait: 7,
            },
        ),
        (
            "careful",
            Costs {
                wrong_link: 240,
                false_nil: 60,
                review: 30,
                wait: 1,
            },
        ),
    ] {
        let rules = [0, 1, 2].map(|step| fit(&dev, step, costs));
        let modes = ["online", "buffered", "bulk", "adaptive"];
        let adaptive = StoppingPolicy::fit(&dev, rules, costs);
        let best_step = (0..3)
            .min_by_key(|&step| evaluate(&dev, fixed(step, rules[step])).loss(costs))
            .unwrap();
        for (step, mode) in modes.into_iter().enumerate() {
            let decide = |views: &[View; 3]| match rules.get(step) {
                Some(rule) => (step, rule.decide(views[step])),
                None => adaptive.decide_from(views, 0),
            };
            let thresholds = match rules.get(step) {
                Some(r) => format!("{:.2},{:.2},{:.2}", r.min_score, r.min_margin, r.nil_below),
                None => ",,".to_owned(), // adaptive uses all three fixed rules
            };
            let n = evaluate(&test, decide);
            let loss = n.loss(costs);
            let blocked = evaluate(test.iter().filter(|c| c.gold == Gold::Blocked), decide);
            let unseen = evaluate(test.iter().filter(|c| c.gold == Gold::Unseen), decide);
            let (delta, lo, hi) = paired_difference(&test, decide, fixed(0, rules[0]), costs);
            let (delta_best, lo_best, hi_best) =
                paired_difference(&test, decide, fixed(best_step, rules[best_step]), costs);
            println!(
                "{scenario},{mode},{thresholds},{},{},{},{},{},{},{},{:.2},{},{},{},{},{},{},{},{},{},{delta:.2},{lo:.2},{hi:.2},{},{delta_best:.2},{lo_best:.2},{hi_best:.2}",
                n.correct_link, n.wrong_link, n.correct_nil, n.false_nil, n.review, n.wait, loss,
                loss as f64 / test.len() as f64,
                blocked.wrong_link, blocked.false_nil, blocked.review,
                unseen.wrong_link, unseen.correct_nil, unseen.review,
                n.stopped[0], n.stopped[1], n.stopped[2], modes[best_step]
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn later_evidence_breaks_a_namesake_tie() {
        let names = ["Dar Vem".to_owned(), "Ser Vem".to_owned()];
        let views = rank(&names, &["Vem", "Ser Vem", "Ser Vem"]);
        let rule = Rule {
            min_score: 0.7,
            min_margin: 0.2,
            nil_below: 0.01,
        };
        assert_eq!(views[0].score, 0.5);
        assert_eq!(rule.decide(views[0]), Decision::Review);
        assert_eq!(rule.decide(views[1]), Decision::Link(1));
    }

    #[test]
    fn adaptive_waits_for_an_informative_hint_only_if_it_pays() {
        let names = ["Dar Vem".to_owned(), "Ser Vem".to_owned()];
        let cases = [
            Case {
                gold: Gold::Present(1),
                views: rank(&names, &["Vem", "Ser Vem", "Ser Vem"]),
            },
            Case {
                gold: Gold::Present(0),
                views: rank(&names, &["Dar Vem"; 3]),
            },
        ];
        let rules = [Rule {
            min_score: 0.7,
            min_margin: 0.2,
            nil_below: 0.01,
        }; 3];
        let costs = Costs {
            wrong_link: 50,
            false_nil: 25,
            review: 10,
            wait: 1,
        };
        let cheap_wait = StoppingPolicy::fit(&cases, rules, costs);
        assert_eq!(
            cheap_wait.decide_from(&cases[0].views, 0),
            (1, Decision::Link(1))
        );
        // Changing unseen future evidence cannot alter a stop on the first observation.
        let changed_future = rank(&names, &["Dar Vem", "Ser Vem", "Ser Vem"]);
        assert_eq!(
            cheap_wait.decide_from(&changed_future, 0),
            (0, Decision::Link(0))
        );
        let expensive_wait = StoppingPolicy::fit(&cases, rules, Costs { wait: 20, ..costs });
        assert_eq!(
            expensive_wait.decide_from(&cases[0].views, 0),
            (0, Decision::Review)
        );
    }

    #[test]
    fn missing_candidate_and_missing_entity_have_different_oracles() {
        let mut counts = Counts::default();
        for gold in [Gold::Blocked, Gold::Unseen] {
            counts.observe(gold, Decision::Nil);
            counts.observe(gold, Decision::Link(0));
        }
        assert_eq!(counts.false_nil, 1);
        assert_eq!(counts.correct_nil, 1);
        assert_eq!(counts.wrong_link, 2);
    }
}
