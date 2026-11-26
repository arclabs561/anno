"""Score whole-document CRAC CorefUD predictions at exact source-bound panel mentions.

These two components use integer token IDs and sentence-local contiguous brackets.
The declared schema selects eid-etype-head or eid-etype-head-other. The latter accepts
an omitted fourth field, a trailing dash for an empty field, or a simple fourth value
such as wd:Q42. The etype field may be empty. An opening begins with "(", a closing
ends in ")", and a single-token occurrence has both brackets.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import hashlib
import itertools
from pathlib import Path
import re
from zipfile import BadZipFile

if __package__:
    from . import corefud
else:
    import corefud


COMPONENTS = {"dutch-openboek": corefud.DUTCH, "french-litbankfr": corefud.FRENCH}
# In the official outputs the entity type is usually empty and trailing "other" is omitted.
# System IDs include e1, c1, 0, and d1.e1. Dash is the field separator.
_ID = r"[a-zA-Z0-9_.]+"
_OPEN = rf"\(({_ID})-[a-z_]*-([1-9]\d*)"
_END = rf"(\)?)|({_ID})\)"
_ATOMS = {
    "eid-etype-head": re.compile(_OPEN + _END),
    "eid-etype-head-other": re.compile(_OPEN + r"(?:-[a-zA-Z0-9_.:]*)?" + _END),
}


@dataclass
class Document:
    sentences: list = field(default_factory=list)
    sentence_index: dict = field(default_factory=dict)
    size: int = 0
    tokens: int = 0
    omitted_no_space_flags: int = 0
    owners: dict = field(default_factory=lambda: defaultdict(set))
    entity_sizes: Counter = field(default_factory=Counter)


def _sentences(data):
    work = None
    schema = None
    seen_docs, seen_sents = set(), set()
    for block in data.replace("\r\n", "\n").strip("\n").split("\n\n"):
        headers, rows = {}, []
        for line in block.split("\n"):
            if line.startswith("#"):
                key, _, value = line.partition(" = ")
                if key not in {"# newdoc id", "# sent_id", "# text", "# global.Entity"}:
                    continue
                if rows or key in headers:
                    raise ValueError("Duplicate or misplaced sentence metadata")
                headers[key] = value
            else:
                row = line.split("\t")
                if len(row) != 10 or row[0] != str(len(rows) + 1) or not row[1]:
                    raise ValueError("Expected ten columns and ordered integer surface-token IDs")
                rows.append(row)
        if "# newdoc id" in headers:
            work = headers["# newdoc id"]
            if not work or work in seen_docs:
                raise ValueError("Duplicate or empty source document ID")
            seen_docs.add(work)
        if "# global.Entity" in headers:
            schema = headers.pop("# global.Entity")
        if schema not in _ATOMS:
            raise ValueError("Unsupported or missing CorefUD entity schema")
        sid, text = headers.get("# sent_id"), headers.get("# text")
        if not work or not sid or sid in seen_sents or text is None or not rows:
            raise ValueError("Missing or duplicate source sentence")
        seen_sents.add(sid)
        yield work, sid, text, rows, schema


def _annotation(misc, token, start, end, opened, doc, atoms):
    if "Entity" in misc:
        raise ValueError("Entity attribute is missing its value")
    entity = [attr[7:] for attr in misc if attr.startswith("Entity=")]
    if not entity:
        return
    if len(entity) != 1 or not entity[0]:
        raise ValueError("Empty or repeated Entity attribute")
    value, pos = entity[0], 0
    while pos < len(value):
        match = atoms.match(value, pos)
        if match is None:
            raise ValueError("Unsupported CorefUD entity bracket")
        pos = match.end()
        opening, head, singleton, closing = match.groups()
        if opening is not None:
            opened[opening].append((token, start, int(head)))
            if not singleton:
                continue
        group = opening or closing
        if not opened[group]:
            raise ValueError("Unmatched closing entity bracket")
        first, begin, relative_head = opened[group].pop()
        if relative_head > token - first + 1:
            raise ValueError("Entity head lies outside its mention")
        members = doc.owners[begin, end]
        if group in members:
            raise ValueError("Duplicate predicted span in one entity")
        members.add(group)
        doc.entity_sizes[group] += 1


def _documents(reference, prediction):
    docs = {}
    for gold, pred in zip(_sentences(reference), _sentences(prediction), strict=True):
        work, sid, text, rows, _ = gold
        pwork, psid, ptext, prows, schema = pred
        if (work, sid, text) != (pwork, psid, ptext) or len(rows) != len(prows):
            raise ValueError("Prediction document, sentence, source text, or token count differs from source")
        if work not in docs:
            docs[work] = Document()
        doc = docs[work]
        begin = doc.size + bool(doc.sentences)
        cursor = 0
        opened = defaultdict(list)
        for i, (row, prow) in enumerate(zip(rows, prows)):
            if row[:2] != prow[:2]:
                raise ValueError("Prediction token ID or surface differs from source")
            form = row[1]
            end = cursor + len(form)
            if text[cursor:end] != form:
                raise ValueError("Reference tokens do not reconstruct sentence text")
            misc = prow[9].split("|")
            tight = "SpaceAfter=No" in row[9].split("|")
            ptight = "SpaceAfter=No" in misc
            if ptight and not tight:
                raise ValueError("Prediction spacing contradicts the source")
            # Several official systems omit SpaceAfter from MISC. Use pinned source geometry.
            doc.omitted_no_space_flags += tight and not ptight
            _annotation(misc, i, begin + cursor, begin + end, opened, doc, _ATOMS[schema])
            cursor = end
            if i + 1 < len(rows) and not tight:
                if text[cursor:cursor + 1] != " ":
                    raise ValueError("Reference whitespace does not reconstruct sentence text")
                cursor += 1
        if cursor != len(text) or any(opened.values()):
            raise ValueError("Invalid sentence length or unterminated entity bracket")
        doc.sentence_index[sid] = len(doc.sentences)
        doc.sentences.append((begin, text))
        doc.size = begin + len(text)
        doc.tokens += len(rows)
    return docs


def _score_case(case, doc, component):
    lineage = case["provenance"]
    sentence_ids = lineage["sentence_ids"]
    indices = [doc.sentence_index.get(sid) for sid in sentence_ids]
    if not indices or indices[0] is None or indices != list(range(indices[0], indices[0] + len(indices))):
        raise ValueError("Panel window refers to an unknown or unordered source sentence")
    sentences = doc.sentences[indices[0]:indices[-1] + 1]
    base = sentences[0][0]
    joined = "\n".join(text for _, text in sentences)
    member = f"CorefUD-1.4-public/data/CorefUD_{component.name}/{component.filename}"
    if (lineage["work_id"] != case["group"] or lineage["source_sha256"] != component.sha256
            or lineage["source_member"] != member
            or lineage["window_scalar_span"] != [base, base + len(joined)] or case["text"] != joined):
        raise ValueError("Panel provenance or window text differs from the pinned source")

    roles = ("target", "A", "B") if case["kind"] == "choice" else ("inner", "outer")
    if case["kind"] not in {"choice", "nested"} or set(case["spans"]) != set(roles):
        raise ValueError("Unsupported panel kind or mark set")
    answers = {"A", "B"} if case["kind"] == "choice" else {"same", "different"}
    if case["gold"] not in answers:
        raise ValueError("Unsupported marked decision for these components")
    choices, missing, ambiguous = [], [], []
    for role in roles:
        start, end = case["spans"][role]
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(joined):
            raise ValueError("Panel mark is outside its source window")
        memberships = doc.owners.get((base + start, base + end), set())
        if not memberships:
            missing.append(role)
        if len(memberships) > 1:
            ambiguous.append(role)
        choices.append(memberships or {None})

    # A coextensive mention can belong to several entities. Keep answers invariant across
    # all admissible assignments and expose ambiguity whenever the identity decision changes.
    decisions = set()
    for assignment in itertools.product(*choices):
        if case["kind"] == "nested":
            decision = "unresolved" if missing else "same" if assignment[0] == assignment[1] else "different"
        else:
            target, a, b = assignment
            links = [label for label, entity in (("A", a), ("B", b)) if target is not None and entity == target]
            decision = "both" if len(links) == 2 else next(iter(links), "neither")
        decisions.add(decision)
        if len(decisions) == 2:
            break
    observed = next(iter(decisions)) if len(decisions) == 1 else "ambiguous"
    required = not missing if case["kind"] == "nested" else "target" not in missing and case["gold"] not in missing
    return {"id": case["id"], "kind": case["kind"], "gold": case["gold"], "observed": observed,
            "exact": observed == case["gold"], "missing": missing, "ambiguous_memberships": ambiguous,
            "both_required_marks_seen": required}


def score_document_predictions(cases, gold_source: Path, corpus_name: str, prediction_file: Path,
                               *, include_cases=False):
    """Score all selected panel decisions after validating the complete .conllu prediction.

    Predicted singleton clusters publish exact mentions. They can support a nested-different
    judgment but cannot create an identity link between two distinct spans. No proposal-stage
    information is available in a CoNLL-U submission. Case and scope counts are independent.
    Source tokenization and spacing define coordinates. Missing SpaceAfter=No flags are
    counted, and an explicit no-space flag contradicting the reference is an error.
    """
    if corpus_name not in COMPONENTS:
        raise ValueError("Unsupported document-prediction corpus")
    component = COMPONENTS[corpus_name]
    try:
        reference = corefud._source(gold_source, component)
        raw = prediction_file.read_bytes()
        predicted = raw.decode("utf-8")
    except (OSError, UnicodeError, BadZipFile):
        raise ValueError("Cannot read pinned source and UTF-8 prediction inputs") from None
    docs = _documents(reference, predicted)
    counts, rows, seen = defaultdict(Counter), [], set()
    for case in cases:
        if case["id"] in seen or case["group"] not in docs:
            raise ValueError("Duplicate panel ID or unknown source document")
        seen.add(case["id"])
        result = _score_case(case, docs[case["group"]], component)
        kinds = [case["kind"]]
        if case["kind"] == "nested":
            kinds.append("nested_" + case["gold"])
        for kind in kinds:
            c = counts[kind]
            c["cases"] += 1
            c["marked_decisions_correct"] += result["exact"]
            c["all_marks_exact"] += not result["missing"]
            c["both_required_marks_seen"] += result["both_required_marks_seen"]
            c["observed_" + result["observed"]] += 1
            if kind == "choice":
                c["wrong_single"] += result["observed"] in {"A", "B"} and not result["exact"]
        if include_cases:
            rows.append(result)
    if not seen:
        raise ValueError("No selected document-prediction cases")
    coverage = {"documents": len(docs), "sentences": sum(len(d.sentences) for d in docs.values()),
                "tokens": sum(d.tokens for d in docs.values()),
                "output_clusters": sum(len(d.entity_sizes) for d in docs.values()),
                "output_singleton_clusters": sum(n == 1 for d in docs.values() for n in d.entity_sizes.values()),
                "distinct_output_spans": sum(len(d.owners) for d in docs.values()),
                "multiply_owned_spans": sum(len(v) > 1 for d in docs.values() for v in d.owners.values()),
                "omitted_no_space_flags": sum(d.omitted_no_space_flags for d in docs.values())}
    report = {"scope": "full_document", "source_sha256": component.sha256,
              "prediction_sha256": hashlib.sha256(raw).hexdigest(), "coverage": coverage, "counts": dict(counts)}
    if include_cases:
        report["cases"] = rows
    return report
