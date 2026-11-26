"""Nested identities and competing antecedents in LitBank's English fiction.

Upstream: https://github.com/dbamman/litbank/tree/3e50db0ffc033d7ccbb94f4d88f6b99210328ed8
Bamman, Popat and Shen (2019), literary entities; Bamman, Lewke and Mansoor
(2020), literary coreference. LitBank annotations are CC BY 4.0. This adapter
selects source-bound cases and derives pair labels from work-local chains.
"""

from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import re
from typing import NamedTuple

# Work-level holdout from earlier Anno dialogue and fold-0 evaluations.
WORKS = {
    11, 27, 33, 36, 41, 62, 73, 77, 78, 84, 105, 233, 367, 502,
    521, 541, 829, 876, 932, 940, 974, 1064, 1327, 2095, 2489, 3268, 8867,
}
# SHA-256 of sorted "<file sha256>  <upstream relative path>\n" entries.
SOURCE_SHA256 = "3ea0274466d12f3731f2007134575282ca2a47aafc4d6ce855a1f239946e3288"
MENTION_TYPE = re.compile(r"(?:NOM|PROP|PRON)_(?:PER|FAC|LOC|GPE|ORG|VEH)")
CHAIN = re.compile(r".+-[0-9]+")
PRONOUNS = {"he", "she", "him", "her", "his", "hers", "himself", "herself"}


class Mention(NamedTuple):
    span: tuple[int, int]
    tag: str
    chain: str


def _load(source: Path):
    if not (source / "coref/brat").is_dir() and (source / "litbank").is_dir():
        source = source / "litbank"
    works = {}
    hashes = {}
    for ann in sorted((source / "coref/brat").glob("*_brat.ann")):
        work = ann.name.partition("_")[0]
        if not work.isdecimal() or int(work) not in WORKS:
            continue
        if work in works:
            raise ValueError(f"duplicate LitBank work {work}")
        entity = source / "entities/brat" / ann.name
        raw = []
        for path in (ann.with_suffix(".txt"), entity.with_suffix(".txt"), ann, entity):
            data = path.read_bytes()
            hashes[path.relative_to(source).as_posix()] = hashlib.sha256(data).hexdigest()
            raw.append(data)
        text, entity_text, coref, entities = raw
        if text != entity_text:
            raise ValueError(f"unaligned LitBank text layers for work {work}")
        text_path = ann.with_suffix(".txt").relative_to(source).as_posix()
        provenance = {
            "work_id": work,
            "gutenberg_url": f"https://www.gutenberg.org/ebooks/{int(work)}",
            "revision": "3e50db0ffc033d7ccbb94f4d88f6b99210328ed8",
            "source_file": text_path,
            "source_sha256": hashes[text_path],
            "coref_annotations": ann.relative_to(source).as_posix(),
            "entity_annotations": entity.relative_to(source).as_posix(),
        }
        works[work] = (text.decode(), coref.decode(), entities.decode(), provenance)
    missing = WORKS - {int(work) for work in works}
    if missing:
        raise ValueError(f"missing LitBank work(s) {sorted(missing)} under {source}")
    manifest = "".join(f"{hashes[name]}  {name}\n" for name in sorted(hashes))
    if hashlib.sha256(manifest.encode()).hexdigest() != SOURCE_SHA256:
        raise ValueError("LitBank BRAT files differ from the pinned release")
    return works


def _annotations(annotation, text):
    tags = defaultdict(list)
    for line in annotation.splitlines():
        if not line.startswith("T"):
            continue
        key, descriptor, surface = line.split("\t", 2)
        tag, start, end = descriptor.split()
        a, b = int(start), int(end)
        if not 0 <= a < b <= len(text) or text[a:b] != surface:
            raise ValueError(f"LitBank annotation {key} does not match its source span")
        tags[a, b].append(tag)
    return tags


def _mentions(text, coref, entities):
    source = _annotations(coref, text)
    entity_types = _annotations(entities, text)
    chains = {tag for tags in source.values() for tag in tags if CHAIN.fullmatch(tag)}
    # Numeric IDs sometimes occur under multiple labels in the same work.
    suffix_count = Counter(tag.rsplit("-", 1)[1] for tag in chains)
    mentions, by_chain = [], defaultdict(set)
    for span, tags in source.items():
        ids = [tag for tag in tags if tag in chains]
        if len(ids) != 1 or suffix_count[ids[0].rsplit("-", 1)[1]] != 1:
            continue
        chain = ids[0]
        by_chain[chain].add(span)
        types = [tag for tag in tags if MENTION_TYPE.fullmatch(tag)]
        if len(types) != 1:
            continue
        tag = types[0]
        if not tag.startswith("PRON_") and entity_types[span].count(tag.split("_")[1]) != 1:
            continue
        mentions.append(Mention(span, tag, chain))
    return mentions, by_chain


def _case_id(work, task, members):
    coords = [str(i) for member in members for i in member.span]
    key = "/".join(["anno-litbank-v1", work, task, *coords])
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _candidates(work, text, mentions):
    names = [m for m in mentions if m.tag == "PROP_PER"]
    for inner in names:
        a, b = inner.span
        for outer in mentions:
            c, d = outer.span
            if outer.tag.startswith("PRON_") or inner.span == outer.span or not c <= a < b <= d:
                continue
            gold = "same" if inner.chain == outer.chain else "different"
            stratum = gold + ("_person" if outer.tag.endswith("_PER") else "_other")
            yield _case_id(work, "nested", [inner, outer]), stratum, gold, {"inner": inner, "outer": outer}

    for target in mentions:
        a, b = target.span
        if target.tag != "PRON_PER" or text[a:b].casefold() not in PRONOUNS:
            continue
        # The latest ending span wins; ties go to the later start.
        nearest = sorted((m for m in names if 0 <= a - m.span[1] <= 600),
                         key=lambda m: (m.span[1], m.span[0]), reverse=True)
        correct = next((m for m in nearest if m.chain == target.chain), None)
        foil = next((m for m in nearest if m.chain != target.chain), None)
        if correct is None or foil is None:
            continue
        if max(correct.span[0], foil.span[0]) < min(correct.span[1], foil.span[1]):
            continue
        if text[slice(*correct.span)].casefold() == text[slice(*foil.span)].casefold():
            continue
        earlier, later = sorted([correct, foil], key=lambda m: m.span)
        stratum = "correct_nearer" if later == correct else "foil_nearer"
        yield (_case_id(work, "competing", [target, earlier, later]), stratum, None,
               {"target": target, "correct": correct, "foil": foil})


def _window(text, members):
    lo = max(0, min(m.span[0] for m in members) - 220)
    hi = min(len(text), max(m.span[1] for m in members) + 220)
    while lo and not text[lo - 1].isspace() and not text[lo].isspace():
        lo -= 1
    while hi < len(text) and not text[hi - 1].isspace() and not text[hi].isspace():
        hi += 1
    return lo, hi


def cases(source: Path):
    """Yield score-time cases; aliases are other exact same-chain spans in the window."""
    works = _load(source)
    selected = []
    for work, (text, coref, entities, _) in works.items():
        mentions, chains = _mentions(text, coref, entities)
        best = {}
        for candidate in _candidates(work, text, mentions):
            case_id, stratum, _, _ = candidate
            if stratum not in best or case_id < best[stratum][0]:
                best[stratum] = candidate
        selected.extend((candidate, work, chains) for candidate in best.values())

    placements = Counter()
    for (case_id, stratum, gold, members), work, chains in sorted(selected, key=lambda item: item[0][0]):
        text = works[work][0]
        lo, hi = _window(text, members.values())
        choice = "target" in members
        result = {
            "id": case_id,
            "group": work,
            "text": text[lo:hi],
            "provenance": {**works[work][3], "window_scalar_span": [lo, hi]},
            "kind": "choice" if choice else "nested",
            "tags": [stratum],
        }
        if choice:
            gold = "AB"[placements[stratum] % 2]
            placements[stratum] += 1
            foil = "B" if gold == "A" else "A"
            members = {"target": members["target"], gold: members["correct"], foil: members["foil"]}
            marked = {m.span for m in members.values()}
            result["aliases"] = [
                [a - lo, b - lo]
                for a, b in sorted(chains[members["target"].chain] - marked)
                if lo <= a < b <= hi
            ]
        result["spans"] = {role: [m.span[0] - lo, m.span[1] - lo] for role, m in members.items()}
        result["gold"] = gold
        yield result
