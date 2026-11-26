"""Build nested and competing-antecedent cases from the pinned UD English GUM test split."""

from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re

SOURCE_FILE = "en_gum-ud-test.conllu"
SOURCE_SHA256 = "40854b004a1475412c8cdd0ba9536d77e4876467238d8e035718dd33bd29c1e3"
ENTITY = re.compile(r"\(([^()]+)(\)?)|(\d+)\)")
PERSONAL = {"she", "he", "her", "him", "his", "hers", "they", "them", "their", "theirs"}


def digest(text):
    return sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Mention:
    start: int
    end: int
    chain: int
    named_person: bool
    personal_pronoun: bool


@dataclass
class Sentence:
    sid: str
    text: str
    mentions: list[Mention]


@dataclass(frozen=True)
class Document:
    name: str
    author: str
    source_url: str

    def provenance(self, sentences, start):
        """Start counts Unicode scalars from the first sentence of this LF-joined document."""
        return {"document": self.name, "author": self.author, "sourceURL": self.source_url,
                "sentence_ids": [sentence.sid for sentence in sentences], "window_start": start}


def align(text, rows, words):
    """Align original surface tokens. Only known component boundaries may index text."""
    coords = {}
    cursor = 0
    covered = 0
    for row in rows:
        token = row[0]
        if "." in token:
            continue  # CoNLL-U empty node
        first, last = map(int, token.split("-")) if "-" in token else (int(token), int(token))
        if first <= covered:
            continue  # already aligned via a multiword token
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        form = row[1]
        if not text.startswith(form, cursor):
            raise ValueError("GUM token does not align with its source sentence")
        end = cursor + len(form)
        parts = [words[i][1] for i in range(first, last + 1)]
        if "".join(parts) == form:
            for i, part in enumerate(parts, first):
                coords[i] = cursor, cursor + len(part)
                cursor += len(part)
        else:
            for i in range(first, last + 1):
                coords[i] = (cursor if i == first else None, end if i == last else None)
        covered = last
        cursor = end
    if text[cursor:].strip():
        raise ValueError("Unaligned suffix in GUM source sentence")
    return coords


def parse_sentence(sid, text, rows):
    words = {int(row[0]): row for row in rows if row[0].isdigit()}
    if list(words) != list(range(1, len(words) + 1)):
        raise ValueError(f"Nonconsecutive token IDs in {sid}")
    coords = align(text, rows, words)
    pending = defaultdict(list)
    mentions = []
    for index, row in words.items():
        entity = next((v[7:] for v in row[-1].split("|") if v.startswith("Entity=")), "")
        pos = 0
        while pos < len(entity):
            mark = ENTITY.match(entity, pos)
            if not mark:
                raise ValueError(f"Malformed Entity annotation in {sid}")
            pos = mark.end()
            if mark[1]:
                fields = mark[1].split("-")
                first = index
                chain = int(fields[0])
                if not mark[2]:
                    pending[chain].append((first, fields))
                    continue
            else:
                chain = int(mark[3])
                if not pending[chain]:
                    raise ValueError(f"Unmatched Entity closer in {sid}")
                first, fields = pending[chain].pop()
            start, end = coords[first][0], coords[index][1]
            if start is None or end is None:
                continue
            heads = [first + int(head) - 1 for head in fields[5].split(",")]
            if not all(first <= head <= index for head in heads):
                raise ValueError(f"Entity head outside mention in {sid}")
            person = fields[1] == "person"
            named = person and all(words[h][3] == "PROPN" for h in heads)
            pronoun = person and first == index and row[3] == "PRON" and row[2].lower() in PERSONAL
            mentions.append(Mention(start, end, chain, named, pronoun))
    if any(pending.values()):
        raise ValueError(f"Unclosed Entity annotation in {sid}")
    return Sentence(sid, text, mentions)


def documents(conllu):
    doc = None
    sentences = []
    for block in conllu.strip().split("\n\n"):
        lines = block.splitlines()
        metadata = dict(line[2:].split(" = ", 1) for line in lines
                        if line.startswith("# ") and " = " in line)
        if "newdoc id" in metadata:
            if doc is not None:
                yield doc, sentences
            doc = Document(metadata["newdoc id"], metadata["meta::author"], metadata["meta::sourceURL"])
            sentences = []
        if doc is None:
            raise ValueError("GUM sentence before document ID")
        rows = [line.split("\t") for line in lines if not line.startswith("#")]
        sentences.append(parse_sentence(metadata["sent_id"], metadata["text"], rows))
    if doc is not None:
        yield doc, sentences


def antecedent_choice(doc, window, pronoun, window_start):
    text = "\n".join(s.text for s in window)
    names = []
    offset = 0
    for index, sent in enumerate(window):
        for name in sent.mentions:
            if name.named_person and (index < len(window) - 1 or name.end <= pronoun.start):
                names.append((offset + name.start, offset + name.end, name.chain))
        if index < len(window) - 1:
            offset += len(sent.text) + 1

    def latest(candidates):
        return max(candidates, key=lambda n: (n[1], n[0]), default=None)

    positive = latest(n for n in names if n[2] == pronoun.chain)
    if positive is None:
        return None
    left, right, _ = positive
    other = latest(n for n in names if n[2] != pronoun.chain and (n[1] <= left or right <= n[0])
                   and text[n[0]:n[1]].casefold() != text[left:right].casefold())
    if other is None:
        return None
    locator = f"{doc.name}/{window[-1].sid}/{pronoun.start}"
    reverse = int(digest("gum-display-v1/" + locator), 16) % 2
    a, b = (other, positive) if reverse else (positive, other)
    case = dict(id=digest("gum-choice/" + locator), group=doc.name, text=text, kind="choice",
                provenance=doc.provenance(window, window_start),
                spans={"target": [offset + pronoun.start, offset + pronoun.end],
                       "A": list(a[:2]), "B": list(b[:2])},
                gold="B" if reverse else "A", tags=["personal-pronoun", "competing-names"])
    return digest("gum-panel-v1/" + locator), case


def document_cases(doc, sentences):
    options = []
    sentence_starts = []
    next_start = 0
    for index, sent in enumerate(sentences):
        sentence_starts.append(next_start)
        next_start += len(sent.text) + 1
        for i, mention in enumerate(sent.mentions):
            if mention.named_person:
                for j, outer in enumerate(sent.mentions):
                    if (outer.start <= mention.start and mention.end <= outer.end
                            and (outer.start, outer.end) != (mention.start, mention.end)):
                        yield dict(id=digest(f"gum-nested/{doc.name}/{sent.sid}/{i}/{j}"),
                                   group=doc.name, text=sent.text, kind="nested",
                                   provenance=doc.provenance([sent], sentence_starts[index]),
                                   spans={"inner": [mention.start, mention.end],
                                          "outer": [outer.start, outer.end]},
                                   gold="same" if mention.chain == outer.chain else "different",
                                   tags=["proper-headed-person"])
            if mention.personal_pronoun:
                first = max(0, index - 2)
                option = antecedent_choice(doc, sentences[first:index + 1], mention, sentence_starts[first])
                if option:
                    options.append(option)
    seen = set()
    for _, case in sorted(options, key=lambda o: o[0]):
        if case["text"] in seen:
            continue
        seen.add(case["text"])
        yield case
        if len(seen) == 2:
            break


def cases(source: Path):
    """Yield half-open Unicode scalar offsets. Consecutive sentences are joined by one LF."""
    path = source / SOURCE_FILE if source.is_dir() else source
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("GUM source SHA-256 mismatch (expected the pinned UD test split)")
    for doc, sentences in documents(raw.decode("utf-8")):
        yield from document_cases(doc, sentences)
