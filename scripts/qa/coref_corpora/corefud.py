"""Nested spans and competing person mentions from two CorefUD 1.4 dev splits.

Accept the official release ZIP, a dev-only ZIP retaining the official member
paths, or the unmodified component .conllu file. Verify the dev member itself
for all three inputs. Offsets count Unicode characters in the unchanged sentence
strings joined with LF. The cluster IDs establish entity identity; the source
does not identify a unique direct antecedent. Head POS and morphology are the
release's automatic analyses.

CorefUD 1.4: https://hdl.handle.net/11234/1-6108
Dutch OpenBoek (CC BY 4.0): https://clinjournal.org/clinj/article/view/157
French LitBankFr (CC BY-SA 4.0): https://doi.org/10.48694/jcls.3924
"""

import hashlib
import itertools
import re
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile, is_zipfile


@dataclass(frozen=True)
class Component:
    name: str
    language: str
    filename: str
    sha256: str


DUTCH = Component(
    "Dutch-OpenBoek", "nl", "nl_openboek-corefud-dev.conllu",
    "0d3a1f1209d8b3c33b8649db14506978a1cb7c89efec1bdeeebabc544a663313",
)
FRENCH = Component(
    "French-LitBankFr", "fr", "fr_litbankfr-corefud-dev.conllu",
    "3d2867ffb5838d0be1b62e57943f4f0d4ea1eae03a14ab90c736d5da1272b674",
)

# Entity schema in these two pinned files: eid-etype-head, plus closing brackets.
_ENTITY = re.compile(r"\((e\d+)-([a-z_]+)-(\d+)(\)?)|(e\d+)\)")


@dataclass(frozen=True)
class Mention:
    sentence: int
    start: int
    end: int
    entity: str
    entity_type: str
    pos: str
    personal: bool


def _source(source: Path, component: Component) -> str:
    if is_zipfile(source):
        member = f"CorefUD-1.4-public/data/CorefUD_{component.name}/{component.filename}"
        with ZipFile(source) as archive:
            if archive.namelist().count(member) != 1:
                raise ValueError(f"Expected exactly one development member {member}")
            raw = archive.read(member)
    else:
        raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != component.sha256:
        raise ValueError(f"SHA-256 mismatch for CorefUD 1.4 {component.filename}")
    return raw.decode("utf-8")


def _sentence(rows: list[str], text: str, sentence: int, offset: int) -> list[Mention]:
    words = []
    parts = []
    mentions = []
    opened = defaultdict(list)
    cursor = offset
    for i, row in enumerate(rows):
        fields = row.split("\t")
        # These components contain only integer token IDs, contiguous mentions,
        # and sentence-local brackets. Fail if a future source changes that schema.
        if len(fields) != 10 or fields[0] != str(i + 1):
            raise ValueError("Unexpected CorefUD token layout")
        form, misc = fields[1], fields[9].split("|")
        gap = "" if i == len(rows) - 1 or "SpaceAfter=No" in misc else " "
        words.append((cursor, cursor + len(form), fields[3], fields[5].split("|")))
        parts.extend((form, gap))
        cursor += len(form) + len(gap)
        for attr in misc:
            if not attr.startswith("Entity="):
                continue
            brackets = attr.removeprefix("Entity=")
            atoms = list(_ENTITY.finditer(brackets))
            if "".join(a.group() for a in atoms) != brackets:
                raise ValueError("Unexpected CorefUD entity bracket")
            for atom in atoms:
                entity, typ, head, single, closing = atom.groups()
                if entity:
                    opened[entity].append((i, typ, int(head)))
                    if not single:
                        continue
                entity = closing or entity
                if not opened[entity]:
                    raise ValueError("Unmatched CorefUD closing bracket")
                first, typ, head = opened[entity].pop()
                head_index = first + head - 1
                if not first <= head_index <= i:
                    raise ValueError("Mention head falls outside its span")
                _, _, pos, feats = words[head_index]
                mentions.append(Mention(sentence, words[first][0], words[i][1],
                                        entity, typ, pos, "PronType=Prs" in feats))
    if any(opened.values()) or "".join(parts) != text:
        raise ValueError("Mention brackets or token reconstruction disagree with the sentence")
    return sorted(mentions, key=lambda m: (m.start, m.end, m.entity))


def _documents(raw: str):
    group = None
    sentences, sentence_ids, offsets, annotated = [], [], [], []
    offset = 0
    for block in raw.strip("\n").split("\n\n"):
        lines = block.splitlines()
        for line in lines:
            if line.startswith("# newdoc id = "):
                if group is not None:
                    yield group, sentences, sentence_ids, offsets, annotated
                group = line.removeprefix("# newdoc id = ")
                sentences, sentence_ids, offsets, annotated = [], [], [], []
                offset = 0
        sentence_id, = (line.removeprefix("# sent_id = ") for line in lines if line.startswith("# sent_id = "))
        text, = (line.removeprefix("# text = ") for line in lines if line.startswith("# text = "))
        if group is None:
            raise ValueError("CorefUD sentence has no document ID")
        tokens = [line for line in lines if not line.startswith("#")]
        annotated.append(_sentence(tokens, text, len(sentences), offset))
        sentences.append(text)
        sentence_ids.append(sentence_id)
        offsets.append(offset)
        offset += len(text) + 1
    if group is not None:
        yield group, sentences, sentence_ids, offsets, annotated


def _disjoint(left: Mention, right: Mention) -> bool:
    return left.end <= right.start or right.end <= left.start


def _choice(target: Mention, nominals: list[Mention]):
    # Search the same sentence and two preceding sentences. Order candidates
    # by recency; if mentions end together, prefer the shorter mention.
    earlier = sorted((m for m in nominals if m.end <= target.start),
                     key=lambda m: (-m.end, -m.start, m.entity))
    same = [m for m in earlier if m.entity == target.entity]
    different = [m for m in earlier if m.entity != target.entity]
    return next(((a, b) for a in same for b in different if _disjoint(a, b)), None)


def _cases(component: Component, source: Path) -> Iterator[dict]:
    folder = f"CorefUD-1.4-public/data/CorefUD_{component.name}/"
    for group, sentences, sentence_ids, offsets, annotated in _documents(_source(source, component)):
        def emit(kind, marks, tags):
            low = max(0, min(m.sentence for m in marks.values()) - 1)
            high = min(len(sentences) - 1, max(m.sentence for m in marks.values()) + 1)
            text = "\n".join(sentences[low:high + 1])
            # Use location alone for the ID, including the target for a choice.
            # Neither the annotated entity IDs nor the gold answer enter the hash.
            if kind == "nested":
                gold = "same" if marks["inner"].entity == marks["outer"].entity else "different"
                ordered = [marks["inner"], marks["outer"]]
            else:
                ordered = [marks["target"], *sorted((marks["A"], marks["B"]), key=lambda m: m.start)]
            coordinates = ",".join(f"{m.start}-{m.end}" for m in ordered)
            name = f"corefud:1.4:{component.name}:{group}:{kind}:{coordinates}"
            case_id = hashlib.sha256(name.encode()).hexdigest()[:24]
            if kind == "choice":
                # Order from the text first, then neutralize left-to-right bias.
                first, second = ordered[1:]
                if hashlib.sha256(("order/" + name).encode()).digest()[0] & 1:
                    first, second = second, first
                marks = {"target": marks["target"], "A": first, "B": second}
                gold = "A" if first.entity == marks["target"].entity else "B"
            spans = {k: [m.start - offsets[low], m.end - offsets[low]] for k, m in marks.items()}
            # Offsets are absolute in the work reconstructed from LF-joined # text lines.
            provenance = {
                "release_url": "https://hdl.handle.net/11234/1-6108",
                "source_member": folder + component.filename,
                "source_sha256": component.sha256,
                "work_id": group,
                "sentence_ids": sentence_ids[low:high + 1],
                "window_scalar_span": [offsets[low], offsets[high] + len(sentences[high])],
                "component_readme": folder + "README.md",
                "component_license": folder + "LICENSE.txt",
            }
            return {"id": case_id, "group": group, "text": text, "kind": kind,
                    "spans": spans, "gold": gold, "tags": [component.language, *tags],
                    "provenance": provenance}

        for j, mentions in enumerate(annotated):
            for outer, inner in itertools.permutations(mentions, 2):
                if not (outer.start <= inner.start and inner.end <= outer.end):
                    continue
                if (inner.start, inner.end) == (outer.start, outer.end):
                    continue
                tags = []
                if outer.entity_type == inner.entity_type == "person":
                    tags.append("two-person-mentions")
                    if {outer.pos, inner.pos} == {"NOUN", "PROPN"}:
                        tags.append("common-and-proper-heads")
                yield emit("nested", {"inner": inner, "outer": outer}, tags)

            nominals = [m for sentence in annotated[max(0, j - 2):j + 1] for m in sentence
                        if m.entity_type == "person" and m.pos in {"NOUN", "PROPN"}]
            for target in mentions:
                if target.entity_type != "person" or target.pos != "PRON" or not target.personal:
                    continue
                pair = _choice(target, nominals)
                if pair:
                    yield emit("choice", {"target": target, "A": pair[0], "B": pair[1]},
                               ["personal-pronoun", "competing-people"])


def dutch(source: Path) -> Iterator[dict]:
    """Yield Dutch OpenBoek development cases from a local ZIP or standalone split."""
    return _cases(DUTCH, source)


def french(source: Path) -> Iterator[dict]:
    """Yield French LitBankFr development cases from a local ZIP or standalone split."""
    return _cases(FRENCH, source)
