"""CODI-CRAC22 recency and lexical routing controls. Python 3 standard library."""
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
import subprocess

REVISION = "2f7b2506b87e2fd58da44b4f1d197483deba03ef"
LABELS = ("ordinary", "group", "segment")


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    mid: str
    eid: str
    groups: tuple


def parse(lines):
    words, sentences, mentions = {}, defaultdict(list), [[], []]
    stacks = [[], []]
    sentence = None
    for line in lines:
        if line.startswith("# sent_id"):
            sentence = line.strip()
        cols = line.split()
        if not cols or not cols[0].isdigit():
            continue
        pos = int(cols[0])
        assert pos not in words and sentence is not None
        words[pos] = cols[1].casefold()
        sentences[sentence].append(pos)
        for bucket, col in enumerate((10, 12)):
            for match in re.finditer(r'\(([^()]*)|\)', cols[col]):
                if match.group(1) is not None:
                    attrs = dict(p.split("=", 1) for p in match[1].split("|"))
                    stacks[bucket].append((pos, attrs))
                else:
                    start, attrs = stacks[bucket].pop()
                    groups = tuple(filter(None, attrs.get("ElementOf", "").split(",")))
                    mentions[bucket].append(Span(start, pos, attrs["MarkableID"], attrs["EntityID"], groups))
    assert not any(stacks)
    sentences = [(min(s), max(s)) for s in sentences.values()]
    return words, sentences, mentions


def read_documents(path):
    content = path.read_text().split("# newdoc id = ")[1:]
    for raw in content:
        name, *lines = raw.splitlines()
        yield name, parse(lines)


def examples(doc):
    words, sentences, (identity, deixis) = doc
    # A composite is introduced at its first ordinary markable. Later pronouns
    # can refer to the established composite without reconstructing its members.
    elements = defaultdict(set)
    first = {}
    for span in sorted(identity, key=lambda s: (s.start, s.end, s.mid)):
        first.setdefault(span.eid, span.mid)
        for group in span.groups:
            elements[group].add(span.eid)
    queries = {s.mid: ("group" if first[s.eid] == s.mid and len(elements[s.eid]) >= 2
                       else "ordinary", s) for s in identity}
    segments = [s for s in deixis if s.mid.startswith("dd_markable")]
    for s in deixis:
        if not s.mid.startswith("dd_markable"):
            # Use discourse-deixis annotation for marks shared across columns.
            queries[s.mid] = ("segment", s)
    counts = Counter()
    rows = []
    for kind, s in queries.values():
        surface = " ".join(words[i] for i in range(s.start, s.end + 1))
        solved = False
        positions = "unreachable"
        if kind == "group":
            gold = elements[s.eid]
            live = sorted((m for m in identity if m.end < s.start and "-Pseudo" not in m.eid),
                          key=lambda m: (-m.end, -m.start, m.mid))
            candidates = list(dict.fromkeys(m.eid for m in live))
            counts["groups"] += 1
            counts["groups_with_all_members_observed"] += gold.issubset(candidates)
            counts["groups_all_members_in_top8"] += gold.issubset(candidates[:8])
            if gold.issubset(candidates[:8]):
                positions = ",".join(str(i + 1) for i, e in enumerate(candidates[:8]) if e in gold)
            solved = set(candidates[:2]) == gold
            counts["groups_exact_two_nearest"] += solved
        elif kind == "segment":
            # Only same-chain source spans support the simple single-segment task.
            gold = {(m.start, m.end) for m in segments if m.eid == s.eid and m.end < s.start}
            previous = [sent for sent in sentences if sent[1] < s.start]
            counts["deictics"] += 1
            counts["with_prior_same_chain_segment"] += bool(gold)
            matches = [str(i) for i, span in enumerate(reversed(previous[-3:]), 1) if span in gold]
            if matches:
                positions = matches[0]
            solved = bool(previous and previous[-1] in gold)
            counts["previous_sentence_exact"] += solved
            counts["last_three_sentences_any_exact"] += any(t in gold for t in previous[-3:])
        rows.append((surface, kind, solved, positions))
    return rows, counts


def run(root):
    freq = defaultdict(Counter)
    prevalence = Counter()
    tuples = defaultdict(Counter)
    for file in sorted(root.glob("*train*.CONLLUA")):
        for _, doc in read_documents(file):
            for surface, kind, _, positions in examples(doc)[0]:
                tuples[surface][kind, positions] += 1
                freq[surface][kind] += 1
                prevalence[kind] += 1
    assert all(prevalence[k] for k in LABELS)
    domains = {}
    for file in sorted(root.glob("*test*.CONLLUA")):
        retrieval, matrix, right = Counter(), defaultdict(Counter), Counter()
        docs = 0
        coupled = {name: Counter() for name in ("factorized", "joint")}
        for _, doc in read_documents(file):
            rows, counts = examples(doc)
            docs += 1
            retrieval.update(counts)
            for surface, label, solved, positions in rows:
                seen = freq.get(surface)
                predicted = "ordinary"
                if seen:
                    # Equal class priors, closed vocabulary, ties follow LABELS.
                    predicted = max(LABELS, key=lambda k: seen[k] / prevalence[k])
                options = tuples.get(surface, {})
                fact = (predicted, "unreachable")
                direct = ("ordinary", "unreachable")
                if options:
                    fact = max(sorted(t for t in options if t[0] == predicted), key=options.get)
                    direct = max(sorted(options), key=lambda t: options[t] / prevalence[t[0]])
                for name, answer in (("factorized", fact), ("joint", direct)):
                    k, indexes = answer
                    if k != "ordinary" and indexes != "unreachable":
                        coupled[name][k + "_offered"] += 1
                        coupled[name][k + "_correct"] += answer == (label, positions)
                matrix[label][predicted] += 1
                if predicted == label and solved:
                    right[label] += 1
        domains[file.name] = dict(documents=docs, retrieval=retrieval, position_decoders=coupled,
                                  route_gold_by_predicted=dict(matrix),
                                  correct_route_and_antecedent=right)
    return dict(revision=REVISION, train_mentions=prevalence, test=domains)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    root = p.parse_args().root
    rev = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if rev != REVISION:
        raise ValueError("Expected corpus revision " + REVISION)
    print(json.dumps(run(root), indent=2, sort_keys=True))
