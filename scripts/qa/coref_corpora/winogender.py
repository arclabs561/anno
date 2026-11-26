"""Winogender's 720 released sentences. Keep all six variants of a template together.

Source: https://github.com/rudinger/winogender-schemas/tree/1c7f8b481ad8a234b41e9f76a424d6e856e13f7f
The template supplies a semantic role, not an exact mention boundary. Both the
determiner-inclusive and bare-slot spans are aligned to the released sentence.
"""

import csv
import hashlib
import io
from pathlib import Path

PUBLISHER = "https://github.com/rudinger/winogender-schemas/blob/1c7f8b481ad8a234b41e9f76a424d6e856e13f7f"
PINS = {
    "templates.tsv": "496f2e2dc77296bcebcdd8865a8dc5715ce22b772de9abb5f531d5b3114c48a0",
    "all_sentences.tsv": "bd69da16bf228cb8df63fd0843aec2d8b5c36a7f21bf8c7b168bcc83932556c8",
}
FORMS = {
    "male": ("he", "his", "him"),
    "female": ("she", "her", "her"),
    "neutral": ("they", "their", "them"),
}
SLOTS = ("$NOM_PRONOUN", "$POSS_PRONOUN", "$ACC_PRONOUN")


def _read(source, name, columns):
    raw = (source / name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != PINS[name]:
        raise ValueError(f"Expected publisher {name} at the pinned SHA-256")
    reader = csv.reader(io.StringIO(raw.decode("utf-8")), delimiter="\t")
    if next(reader, None) != columns:
        raise ValueError(f"Unexpected Winogender {name} header")
    return list(reader)


def _render(template, occupation, participant, variant):
    tokens = template.split()
    if participant == "someone":
        pos = tokens.index("$PARTICIPANT")
        if pos == 0 or tokens[pos-1].lower() not in {"a", "the"}:
            raise ValueError("Missing participant determiner")
        del tokens[pos-1]

    # A and B consistently mean occupation and participant here. Those source
    # role names and the answer-bearing source ID stay out of the published IDs.
    slots = {"$OCCUPATION": ("A", occupation), "$PARTICIPANT": ("B", participant)}
    slots.update({slot: ("target", form) for slot, form in zip(SLOTS, FORMS[variant], strict=True)})
    words, heads, spans = [], {}, {}
    start = 0
    for token in tokens:
        label, word = slots.get(token, (None, token))
        if not words and word == "someone":
            word = "Someone"
        if words and words[-1] == "they" and word == "was":
            word = "were"
        end = start + len(word)
        if label:
            if label in spans:
                raise ValueError(f"Repeated Winogender slot {label}")
            extent = start
            if label != "target":
                heads[label] = [start, end]
                if words and words[-1].lower() in {"the", "a"}:
                    extent -= len(words[-1]) + 1
            spans[label] = [extent, end]
        words.append(word)
        start = end + 1
    if set(spans) != {"target", "A", "B"}:
        raise ValueError("Missing Winogender role or target")
    return " ".join(words), spans, heads


def cases(source: Path):
    sentences = _read(source, "all_sentences.tsv", ["sentid", "sentence"])
    templates = _read(source, "templates.tsv", ["occupation(0)", "other-participant(1)", "answer", "sentence"])
    released = dict(sentences)
    if len(released) != len(sentences) or len(released) != 720 or len(templates) != 120:
        raise ValueError("Expected 720 distinct release IDs and 120 templates")

    for line, (occupation, participant, answer, template) in enumerate(templates, start=2):
        if answer not in {"0", "1"}:
            raise ValueError("Invalid Winogender answer role")
        group = hashlib.sha256(f"{occupation}/{participant}/{template}".encode()).hexdigest()[:20]
        for role in (participant, "someone"):
            for variant in FORMS:
                source_id = f"{occupation}.{role}.{answer}.{variant}.txt"
                text, spans, heads = _render(template, occupation, role, variant)
                if released.pop(source_id, None) != text:
                    raise ValueError("Reconstructed Winogender row differs from release")
                yield {
                    "id": hashlib.sha256(("anno-winogender-v1/" + source_id).encode()).hexdigest()[:20],
                    "group": group, "text": text, "kind": "choice", "spans": spans,
                    "gold": "A" if answer == "0" else "B",
                    "tags": [variant, "generic" if role == "someone" else "participant"],
                    "boundaries": {"head": heads},
                    # The source ID encodes the answer; these locators belong in
                    # the facilitator's provenance sidecar.
                    "provenance": {
                        "source_row_id": source_id,
                        "template_url": f"{PUBLISHER}/data/templates.tsv#L{line}",
                        "notice_url": f"{PUBLISHER}/LICENSE",
                    },
                }
    if released:
        raise ValueError("Unconsumed Winogender release rows")
