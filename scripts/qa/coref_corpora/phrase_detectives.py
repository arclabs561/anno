"""Phrase Detectives 3.0 released expert subset, with source-group holdout and three-sentence windows.

Pass the original local ZIP from https://github.com/dali-ambiguity/Phrase-Detectives-Corpus-3.0.
The public release contains the original expert documents and separate silver annotations.
"""
from bisect import bisect_right
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import re
from xml.etree import ElementTree
from zipfile import ZipFile

ARCHIVE_SHA256 = "36d23678198b572ccce2699238716acd50ed80c871b9069d7c40e9b98e0cea23"
TYPES = {"dn": "DN", "do": "DO", "non_referring": "NR", "predicate": "PR"}
GOLD_ROOT = "PD3.0_Train_dev/gold/"


def _digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _source_group(xml, genre, document):
    if genre == "gutenberg":
        # Multiple excerpts can originate in one book.
        publication = xml.find(".//publicationStmt/web")
        publication_text = " ".join(publication.itertext()) if publication is not None else ""
        url = re.search(r"https?://\S+", publication_text)
        if not url:
            raise ValueError("Missing Gutenberg source URL")
        return f"{genre}/{url.group()}"
    if genre == "gnome":
        filename = xml.findtext(".//sourceDesc/filename")
        if not filename:
            raise ValueError("Missing museum source identifier")
        return f"{genre}/{filename}"
    return f"{genre}/{document}"


def _release_text(data):
    sentences, sentence_starts, token_spans, words, labels = [], [], [], [], {}
    cursor = 0
    for block in data.decode("utf-8").split("# sent_id = ")[1:]:
        header, *lines = block.splitlines()
        sentence = next((line[9:] for line in lines if line.startswith("# text = ")), None)
        rows = [line.split() for line in lines if line.strip() and not line.startswith("#")]
        if any(len(row) != 15 for row in rows) or " ".join(row[1] for row in rows) != sentence:
            raise ValueError(f"CoNLL-UA token sequence differs from sentence {header}")
        sentence_starts.append(cursor)
        sentences.append(sentence)
        for row in rows:
            word = row[1]
            for spec in re.findall(r"\(([^()]+)", row[10]):
                fields = dict(part.split("=", 1) for part in spec.split("|"))
                ident = fields["MarkableID"].removeprefix("markable_")
                if ident in labels:
                    raise ValueError("Duplicate CoNLL-UA markable ID")
                labels[ident] = (len(words), fields)
            words.append(word)
            token_spans.append((cursor, cursor + len(word)))
            cursor += len(word) + 1
    return "\n".join(sentences), sentence_starts, words, token_spans, labels


def _mentions(xml, words, token_spans, labels):
    body = xml.find("body")
    if body is None:
        raise ValueError("Missing MAS-XML body")
    xml_words = list(body.iter("W"))
    if [node.text or "" for node in xml_words] != words:
        raise ValueError("MAS-XML and CoNLL-UA token sequences differ")
    token_index = {node: i for i, node in enumerate(xml_words)}
    spans, pronouns = {}, []
    for mark in body.iter("ne"):
        marked = list(mark.iter("W"))
        if not marked or not mark.get("id"):
            continue
        ident = mark.attrib["id"].removeprefix("ne")
        first, last = token_index[marked[0]], token_index[marked[-1]]
        if ident in spans or last - first + 1 != len(marked):
            raise ValueError("Duplicate or discontinuous MAS-XML markable")
        if ident in labels and labels[ident][0] != first:
            raise ValueError("Markable starts differ between source formats")
        spans[ident] = (token_spans[first][0], token_spans[last][1])
        if first == last and marked[0].get("Lpos") in {"PRP", "PRP$", "EX"}:
            pronouns.append(ident)
    return spans, pronouns


def _judgments(record):
    favoured, voters = set(), defaultdict(set)
    for anchor in record.iter("anchor"):
        kind = anchor.get("type")
        if kind not in TYPES.values():
            continue
        if anchor.get("mode") == "e" and anchor.get("favoured") == "y":
            favoured.add(kind)
        if anchor.get("mode") == "a" and anchor.get("agree") == "y" and anchor.get("user_id"):
            voters[anchor.attrib["user_id"]].add(kind)
    # Require two distinct positive annotation-mode voters per competing type.
    counts = Counter(next(iter(types)) for types in voters.values() if len(types) == 1)
    return favoured, sum(n >= 2 for n in counts.values()) >= 2


def _candidates(spans, start):
    # Work from mention coordinates alone. Preserve the original A/B order when clipping.
    prior = sorted((s for s in set(spans) if s[1] <= start), key=lambda s: (-s[1], s[0]))
    picked = []
    for span in prior:
        if not picked or span[1] <= picked[-1][0]:
            picked.append(span)
            if len(picked) == 2:
                break
    return picked


def _case(ident, group, document, text, starts, spans, fields, chain, judgments):
    target = spans[ident]
    center = bisect_right(starts, target[0]) - 1
    first = max(0, min(center - 1, len(starts) - 3))
    last = min(first + 3, len(starts))
    lo, hi = starts[first], starts[last] - 1 if last < len(starts) else len(text)
    marked = {"target": target}
    for key, span in zip(("A", "B"), _candidates(spans.values(), target[0])):
        if lo <= span[0] and span[1] <= hi:
            marked[key] = span

    typ = TYPES[fields["SemType"]]
    favoured, disagreement = judgments
    if typ not in favoured:
        raise ValueError("Release type has no corresponding favoured expert record")
    aliases = sorted(span for span in chain if span[1] <= target[0]) if typ == "DO" else []
    matching = [label for label in ("A", "B") if marked.get(label) in aliases]
    gold = "unscored"
    if typ == "NR":
        if not fields["EntityID"].endswith("-Pseudo"):
            raise ValueError("Non-referring markable lacks its pseudo-entity ID")
        tag, gold = "explicit_nonreferential", "neither"
    elif typ == "DN":
        tag = "discourse_new"
    elif typ == "PR":
        tag = "predicate"
    elif "ElementOf" in fields:
        tag = "element_of"
    elif matching:
        tag, gold = "visible_identity", "both" if len(matching) == 2 else matching[0]
    elif aliases:
        tag, gold = "known_prior_outside_candidates", "neither"
    else:
        tag = "no_earlier_chain_mention"

    tags = [tag]
    if disagreement:
        tags.append("crowd_type_disagreement")
    if len(favoured) > 1:
        tags.append("multiple_favoured_expert_types")
    visible_aliases = [[a - lo, b - lo] for a, b in aliases if lo <= a and b <= hi]
    return {
        "id": "pd3-" + _digest(document + "#" + ident)[:24], "group": group,
        "text": text[lo:hi], "kind": "choice",
        "spans": {key: [a - lo, b - lo] for key, (a, b) in marked.items()},
        "gold": gold, "tags": tags, "aliases": visible_aliases,
    }


def _document_cases(xml, conllu, group, document, xml_member, text_member):
    text, starts, words, token_spans, labels = _release_text(conllu)
    spans, pronouns = _mentions(xml, words, token_spans, labels)
    publication_path = "./header/fileDesc/publicationStmt"
    notice = xml.find(publication_path + "/availability")
    if notice is None:
        raise ValueError("Missing source publication rights notice")
    rights = "".join(notice.itertext()).strip()
    records = {node.attrib["id"].removeprefix("ne"): node for node in xml.iter("PDante")}
    entities = defaultdict(set)
    for ident, (_, fields) in labels.items():
        if ident in spans:
            entities[fields["EntityID"]].add(spans[ident])
    for ident in pronouns:
        if ident in labels and ident in records:
            fields = labels[ident][1]
            case = _case(ident, group, document, text, starts, spans, fields,
                         entities[fields["EntityID"]], _judgments(records[ident]))
            case["provenance"] = {
                "xml_member": xml_member, "text_member": text_member,
                "markable_xpath": f"./body//ne[@id='ne{ident}']",
                "publication_xpath": publication_path, "rights_notice": rights,
            }
            yield case


def cases(source: Path):
    """Yield eligible held-out pronouns. Coordinates index Python Unicode strings."""
    with source.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != ARCHIVE_SHA256:
            raise ValueError("Unexpected Phrase Detectives 3.0 archive digest")
        stream.seek(0)
        with ZipFile(stream) as archive:
            names = sorted(n for n in archive.namelist() if n.startswith(GOLD_ROOT)
                           and "/masxml/" in n and n.endswith("-game.xml"))
            if len(names) != 45:
                raise ValueError("Expected 45 released expert documents")
            for name in names:
                genre = name.split("/")[2]
                document = name.rsplit("/", 1)[1].removesuffix("-game.xml")
                xml = ElementTree.fromstring(archive.read(name))
                group = _source_group(xml, genre, document)
                if int(_digest("pd3-source-v1/" + group)[:8], 16) % 3 != 0:
                    continue
                conll_name = name.replace("/masxml/", "/CONLLUA/").replace("-game.xml", ".CONLLUA")
                yield from _document_cases(xml, archive.read(conll_name), group,
                                           f"{genre}/{document}", name, conll_name)
